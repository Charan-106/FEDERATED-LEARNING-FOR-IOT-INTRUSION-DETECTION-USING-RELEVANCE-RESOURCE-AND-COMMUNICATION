"""
ARC-FL Controlled Experiment 001 Runner
Compares FedAvg, Equal, and ARC-Weighted aggregation under identical,
deterministic training conditions on three clients starting from initial-v1.
"""

import hashlib
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

# Setup workspace
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
ML_SERVICE_DIR = WORKSPACE_ROOT / "ml-service"
sys.path.insert(0, str(WORKSPACE_ROOT))
sys.path.insert(0, str(ML_SERVICE_DIR))

from app.models.registry import registry
from app.models.resmlp import build_model, load_checkpoint
from app.training.trainer import train_local
from app.arc.scoring import ARCScoringEngine
from app.aggregation.fedavg import compute_weights, aggregate
from scripts.evaluate_benchmark import evaluate_on_dataset, sha256_file

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("experiment_001")


def main():
    print("=" * 80)
    print(" STARTING CONTROLLED THREE-CLIENT FEDERATED LEARNING EXPERIMENT 001")
    print("=" * 80)

    exp_dir = WORKSPACE_ROOT / "storage" / "experiments" / "exp_001"
    exp_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------
    # 1. Base Model & Benchmark Setup
    # -------------------------------------------------------------
    base_model_path = WORKSPACE_ROOT / "storage" / "global-models" / "initial-v1" / "best.pt"
    if not base_model_path.exists():
        logger.error(f"Base model not found: {base_model_path}")
        sys.exit(1)

    base_sha256 = sha256_file(base_model_path)
    logger.info(f"Base Model: initial-v1 (SHA-256: {base_sha256})")

    benchmarks_dir = WORKSPACE_ROOT / "storage" / "benchmarks"
    bench_a_path = benchmarks_dir / "benchmark_primary_capped.csv"
    bench_b_path = benchmarks_dir / "benchmark_secondary_natural.csv"

    if not bench_a_path.exists() or not bench_b_path.exists():
        logger.error("Benchmark datasets missing. Run generate_benchmarks.py first.")
        sys.exit(1)

    bench_a_hash = sha256_file(bench_a_path)
    bench_b_hash = sha256_file(bench_b_path)
    logger.info(f"Benchmark A SHA-256: {bench_a_hash}")
    logger.info(f"Benchmark B SHA-256: {bench_b_hash}")

    # Load artifacts for evaluation
    artifacts_dir = ML_SERVICE_DIR / "artifacts"
    with open(artifacts_dir / "model_config.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)
    import joblib
    scaler = joblib.load(artifacts_dir / "scaler.pkl")
    label_encoder = joblib.load(artifacts_dir / "label_encoder.pkl")

    df_bench_a = pd.read_csv(bench_a_path)
    df_bench_b = pd.read_csv(bench_b_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # -------------------------------------------------------------
    # 2. Local Client Training (Identical Parameters)
    # -------------------------------------------------------------
    client_configs = [
        {
            "client_id": "client-1",
            "name": "Client 1 (Edge Gateway)",
            "dataset": str(WORKSPACE_ROOT / "storage" / "datasets" / "client1_traffic.csv"),
            "comm_metrics": {"bandwidth_mbps": 85.0, "latency_ms": 12.0, "reliability_pct": 99.0},
        },
        {
            "client_id": "client-2",
            "name": "Client 2 (IoT Hub)",
            "dataset": str(WORKSPACE_ROOT / "storage" / "datasets" / "client2_traffic.csv"),
            "comm_metrics": {"bandwidth_mbps": 35.0, "latency_ms": 38.0, "reliability_pct": 96.0},
        },
        {
            "client_id": "client-3",
            "name": "Client 3 (Cellular Sensor)",
            "dataset": str(WORKSPACE_ROOT / "storage" / "datasets" / "client3_traffic.csv"),
            "comm_metrics": {"bandwidth_mbps": 12.0, "latency_ms": 75.0, "reliability_pct": 92.0},
        },
    ]

    # Fixed training hyperparameters
    EPOCHS = 2
    BATCH_SIZE = 32
    LR = 0.001

    trained_clients = []
    logger.info(f"\n--- Phase 1: Training 3 Clients from initial-v1 (Epochs={EPOCHS}, BatchSize={BATCH_SIZE}, LR={LR}) ---")

    for c in client_configs:
        ds_path = Path(c["dataset"])
        ds_hash = sha256_file(ds_path)
        logger.info(f"Training {c['name']} on {ds_path.name} (SHA-256: {ds_hash[:16]}...)...")

        # Set deterministic seed per client
        torch.manual_seed(42)
        np.random.seed(42)

        t_res = train_local(
            dataset_path=str(ds_path),
            global_model_path=str(base_model_path),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            learning_rate=LR,
        )

        update_path = Path(t_res["update_path"])
        update_hash = sha256_file(update_path)
        logger.info(f"  -> Update generated: {update_path.name} (SHA-256: {update_hash[:16]}...), Acc={t_res['accuracy']:.2f}%, Loss={t_res['loss']:.4f}")

        trained_clients.append({
            "client_id": c["client_id"],
            "name": c["name"],
            "dataset_path": str(ds_path),
            "dataset_sha256": ds_hash,
            "samples_trained": t_res["samples_trained"],
            "update_id": t_res["update_id"],
            "update_path": str(update_path),
            "update_sha256": update_hash,
            "accuracy": t_res["accuracy"],
            "loss": t_res["loss"],
            "duration": t_res["training_duration_seconds"],
            "quality_metrics": t_res["quality_metrics"],
            "diversity_metrics": t_res["diversity_metrics"],
            "resource_metrics": t_res["resource_metrics"],
            "comm_metrics": c["comm_metrics"],
        })

    # -------------------------------------------------------------
    # 3. ARC Scoring
    # -------------------------------------------------------------
    logger.info("\n--- Phase 2: ARC Contribution Scoring ---")
    arc_config = {
        "lambda_relevance": 0.40,
        "lambda_resource": 0.30,
        "lambda_communication": 0.30,
        "weight_quality": 0.60,
        "weight_diversity": 0.40,
        "weight_memory": 0.40,
        "weight_processing": 0.40,
        "weight_capacity": 0.20,
        "weight_bandwidth": 0.50,
        "weight_latency": 0.30,
        "weight_reliability": 0.20,
    }
    scoring_engine = ARCScoringEngine(arc_config)

    for c in trained_clients:
        score_res = scoring_engine.score(
            quality_metrics=c["quality_metrics"],
            diversity_metrics=c["diversity_metrics"],
            resource_metrics=c["resource_metrics"],
            comm_metrics=c["comm_metrics"],
        )
        c["arc_scoring"] = score_res
        c["arc_score"] = score_res["overall_arc_score"]
        logger.info(f"  {c['name']}: R={score_res['relevance_score']:.2f}, S={score_res['resource_score']:.2f}, C={score_res['communication_score']:.2f} -> ARC={score_res['overall_arc_score']:.2f}")

    # -------------------------------------------------------------
    # 4. Aggregation Weight Computation
    # -------------------------------------------------------------
    logger.info("\n--- Phase 3: Aggregation Weight Calculations ---")
    sub_for_weights = [
        {
            "submission_id": c["update_id"],
            "client_id": c["client_id"],
            "model_path": c["update_path"],
            "samples_trained": c["samples_trained"],
            "arc_score": c["arc_score"],
        }
        for c in trained_clients
    ]

    weights_fedavg = compute_weights(sub_for_weights, "FEDAVG")
    weights_equal = compute_weights(sub_for_weights, "EQUAL")
    weights_arc = compute_weights(sub_for_weights, "ARC_WEIGHTED")

    for i, c in enumerate(trained_clients):
        c["weights"] = {
            "fedavg": weights_fedavg[i]["normalized_weight"],
            "equal": weights_equal[i]["normalized_weight"],
            "arc": weights_arc[i]["normalized_weight"],
        }
        logger.info(f"  {c['name']}: FedAvg={c['weights']['fedavg']:.4f}, Equal={c['weights']['equal']:.4f}, ARC={c['weights']['arc']:.4f}")

    # -------------------------------------------------------------
    # 5. Tensor Aggregation for Each Strategy
    # -------------------------------------------------------------
    logger.info("\n--- Phase 4: Performing Tensor Aggregation ---")
    aggregation_runs = {
        "fedavg": {
            "name": "FedAvg (Sample Weighted)",
            "submissions": weights_fedavg,
            "candidate_path": exp_dir / "candidate_fedavg.pt",
        },
        "equal": {
            "name": "Equal Aggregation",
            "submissions": weights_equal,
            "candidate_path": exp_dir / "candidate_equal.pt",
        },
        "arc": {
            "name": "ARC-Weighted Aggregation",
            "submissions": weights_arc,
            "candidate_path": exp_dir / "candidate_arc.pt",
        },
    }

    # Reference architecture check
    ref_model = build_model(in_dim=45, n_classes=34)
    ref_keys = set(ref_model.state_dict().keys())

    for method, info in aggregation_runs.items():
        logger.info(f"Aggregating via {info['name']}...")
        cand_res = aggregate(
            submissions_with_weights=info["submissions"],
            base_model_path=str(base_model_path),
            server_learning_rate=0.5,
        )

        # Copy/save into exp_dir
        src_path = Path(cand_res["candidate_path"])
        dst_path = info["candidate_path"]
        import shutil
        shutil.copyfile(src_path, dst_path)

        cand_hash = sha256_file(dst_path)
        info["candidate_sha256"] = cand_hash
        info["meta"] = cand_res
        logger.info(f"  -> Saved {dst_path.name} (SHA-256: {cand_hash[:16]}...)")

    # -------------------------------------------------------------
    # 6. Evaluation Across Benchmarks A and B
    # -------------------------------------------------------------
    logger.info("\n--- Phase 5: Benchmark Evaluations ---")
    eval_candidates = [
        ("base", "Initial Global Model (initial-v1)", base_model_path, base_sha256),
        ("fedavg", "Candidate FedAvg", aggregation_runs["fedavg"]["candidate_path"], aggregation_runs["fedavg"]["candidate_sha256"]),
        ("equal", "Candidate Equal", aggregation_runs["equal"]["candidate_path"], aggregation_runs["equal"]["candidate_sha256"]),
        ("arc", "Candidate ARC-FL", aggregation_runs["arc"]["candidate_path"], aggregation_runs["arc"]["candidate_sha256"]),
    ]

    evaluation_records = {}

    for tag, display_name, m_path, m_hash in eval_candidates:
        logger.info(f"Evaluating {display_name}...")
        model_obj, _ = load_checkpoint(str(m_path))
        model_obj.to(device)
        model_obj.eval()

        # Primary Benchmark (Capped)
        t0 = time.time()
        res_a = evaluate_on_dataset(model_obj, df_bench_a, cfg, scaler, label_encoder, device)
        dur_a = round(time.time() - t0, 2)
        res_a["duration"] = dur_a

        # Secondary Benchmark (Natural)
        t0 = time.time()
        res_b = evaluate_on_dataset(model_obj, df_bench_b, cfg, scaler, label_encoder, device)
        dur_b = round(time.time() - t0, 2)
        res_b["duration"] = dur_b

        evaluation_records[tag] = {
            "name": display_name,
            "checkpoint_path": str(m_path),
            "checkpoint_sha256": m_hash,
            "benchmark_a_primary": res_a,
            "benchmark_b_secondary": res_b,
        }

        logger.info(f"  [{tag.upper()}] Bench A -> Acc={res_a['accuracy']:.2f}%, Macro F1={res_a['macro_f1']:.2f}%, Loss={res_a['loss']:.4f}")
        logger.info(f"  [{tag.upper()}] Bench B -> Acc={res_b['accuracy']:.2f}%, Macro F1={res_b['macro_f1']:.2f}%, Loss={res_b['loss']:.4f}")

    # -------------------------------------------------------------
    # 7. Generate Comparison Plot
    # -------------------------------------------------------------
    plot_path = exp_dir / "experiment_001_comparison.png"
    methods = ["Base (initial-v1)", "FedAvg", "Equal", "ARC-Weighted"]
    tags = ["base", "fedavg", "equal", "arc"]

    bench_a_acc = [evaluation_records[t]["benchmark_a_primary"]["accuracy"] for t in tags]
    bench_a_f1 = [evaluation_records[t]["benchmark_a_primary"]["macro_f1"] for t in tags]
    bench_b_acc = [evaluation_records[t]["benchmark_b_secondary"]["accuracy"] for t in tags]
    bench_b_f1 = [evaluation_records[t]["benchmark_b_secondary"]["macro_f1"] for t in tags]

    x = np.arange(len(methods))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Benchmark A (Primary Capped)
    rects1 = ax1.bar(x - width/2, bench_a_acc, width, label="Accuracy (%)", color="#1976d2")
    rects2 = ax1.bar(x + width/2, bench_a_f1, width, label="Macro F1 (%)", color="#388e3c")
    ax1.set_ylabel("Percentage (%)")
    ax1.set_title("Benchmark A (Colab Class-Capped Distribution)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, rotation=15, ha="right")
    ax1.set_ylim(70, 102)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    ax1.legend(loc="lower right")

    for r in rects1:
        h = r.get_height()
        ax1.annotate(f"{h:.2f}%", xy=(r.get_x() + r.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)
    for r in rects2:
        h = r.get_height()
        ax1.annotate(f"{h:.2f}%", xy=(r.get_x() + r.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)

    # Plot 2: Benchmark B (Secondary Natural)
    rects3 = ax2.bar(x - width/2, bench_b_acc, width, label="Accuracy (%)", color="#0288d1")
    rects4 = ax2.bar(x + width/2, bench_b_f1, width, label="Macro F1 (%)", color="#2e7d32")
    ax2.set_ylabel("Percentage (%)")
    ax2.set_title("Benchmark B (Natural Traffic Distribution)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods, rotation=15, ha="right")
    ax2.set_ylim(70, 102)
    ax2.grid(axis="y", linestyle="--", alpha=0.5)
    ax2.legend(loc="lower right")

    for r in rects3:
        h = r.get_height()
        ax2.annotate(f"{h:.2f}%", xy=(r.get_x() + r.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)
    for r in rects4:
        h = r.get_height()
        ax2.annotate(f"{h:.2f}%", xy=(r.get_x() + r.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)

    plt.suptitle("ARC-FL Experiment 001: Comparative Evaluation of Aggregation Methods", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    logger.info(f"Saved comparison plot to {plot_path}")

    # -------------------------------------------------------------
    # 8. Save Machine-Readable JSON
    # -------------------------------------------------------------
    results_json = {
        "experiment_id": "EXP_001_FEDAVG_VS_EQUAL_VS_ARC",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "COMPLETED",
        "protocol_version": "1.0",
        "hyperparameters": {
            "base_model": "initial-v1",
            "base_model_sha256": base_sha256,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LR,
            "server_delta_lr": 0.5,
        },
        "benchmarks": {
            "benchmark_a_primary": {"file": "benchmark_primary_capped.csv", "sha256": bench_a_hash, "samples": len(df_bench_a)},
            "benchmark_b_secondary": {"file": "benchmark_secondary_natural.csv", "sha256": bench_b_hash, "samples": len(df_bench_b)},
            "benchmark_c_quarantined_test": {"file": "P:\\project\\test.csv", "status": "STRICTLY_QUARANTINED_UNTOUCHED"},
        },
        "clients": trained_clients,
        "arc_configuration": arc_config,
        "aggregation_runs": {
            k: {
                "name": v["name"],
                "candidate_path": str(v["candidate_path"]),
                "candidate_sha256": v["candidate_sha256"],
            }
            for k, v in aggregation_runs.items()
        },
        "evaluations": evaluation_records,
    }

    json_path = exp_dir / "experiment_001_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_json, f, indent=2)
    logger.info(f"Saved machine-readable JSON to {json_path}")

    print("\n" + "=" * 80)
    print(" EXPERIMENT 001 COMPLETED SUCCESSFULLY")
    print(f" JSON results: {json_path}")
    print(f" Plot image:   {plot_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
