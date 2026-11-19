"""
ARC-FL Experiment 002: Stable Federated Learning
Controlled comparison of:
A. FedAvg with corrected BatchNorm handling
B. Equal aggregation with corrected BatchNorm handling
C. ARC aggregation with corrected BatchNorm handling
D. ARC aggregation with corrected BatchNorm handling and FedProx (mu=0.01)
"""

import hashlib
import json
import logging
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

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
logger = logging.getLogger("experiment_002")


def main():
    print("=" * 80)
    print(" STARTING CONTROLLED FEDERATED LEARNING EXPERIMENT 002: STABLE FL")
    print("=" * 80)

    exp_dir = WORKSPACE_ROOT / "storage" / "experiments" / "exp_002"
    exp_dir.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------
    # 1. Base Model & Benchmark Setup
    # -------------------------------------------------------------
    base_model_path = WORKSPACE_ROOT / "storage" / "global-models" / "initial-v1" / "best.pt"
    if not base_model_path.exists():
        logger.error(f"Base model not found: {base_model_path}")
        sys.exit(1)

    base_sha256 = sha256_file(base_model_path)
    logger.info(f"Starting Global Model: initial-v1 (SHA-256: {base_sha256})")

    benchmarks_dir = WORKSPACE_ROOT / "storage" / "benchmarks"
    bench_a_path = benchmarks_dir / "benchmark_primary_capped.csv"
    bench_b_path = benchmarks_dir / "benchmark_secondary_natural.csv"

    bench_a_hash = sha256_file(bench_a_path)
    bench_b_hash = sha256_file(bench_b_path)
    logger.info(f"Benchmark A SHA-256: {bench_a_hash}")
    logger.info(f"Benchmark B SHA-256: {bench_b_hash}")

    artifacts_dir = ML_SERVICE_DIR / "artifacts"
    with open(artifacts_dir / "model_config.json", "r", encoding="utf-8") as f:
        cfg = json.load(f)
    import joblib
    scaler = joblib.load(artifacts_dir / "scaler.pkl")
    label_encoder = joblib.load(artifacts_dir / "label_encoder.pkl")

    df_bench_a = pd.read_csv(bench_a_path)
    df_bench_b = pd.read_csv(bench_b_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Client definitions
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

    EPOCHS = 2
    BATCH_SIZE = 32
    LR = 0.0001
    SERVER_DELTA_LR = 0.5
    MU_FEDPROX = 0.01

    # ARC Scoring Engine
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

    # -------------------------------------------------------------
    # 2. Local Training Batch 1: Corrected BatchNorm (mu=0.0)
    # -------------------------------------------------------------
    logger.info(f"\n--- Local Training Batch 1: Corrected BatchNorm, mu=0.0, LR={LR} ---")
    clients_std = []

    for c in client_configs:
        ds_path = Path(c["dataset"])
        ds_hash = sha256_file(ds_path)
        logger.info(f"Training {c['name']} (Standard, freeze_bn=True, mu=0.0)...")

        torch.manual_seed(42)
        np.random.seed(42)

        t_res = train_local(
            dataset_path=str(ds_path),
            global_model_path=str(base_model_path),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            learning_rate=LR,
            freeze_bn_stats=True,
            mu=0.0,
        )

        up_path = Path(t_res["update_path"])
        up_hash = sha256_file(up_path)

        score_res = scoring_engine.score(
            quality_metrics=t_res["quality_metrics"],
            diversity_metrics=t_res["diversity_metrics"],
            resource_metrics=t_res["resource_metrics"],
            comm_metrics=c["comm_metrics"],
        )

        clients_std.append({
            "client_id": c["client_id"],
            "name": c["name"],
            "dataset_path": str(ds_path),
            "dataset_sha256": ds_hash,
            "samples_trained": t_res["samples_trained"],
            "update_id": t_res["update_id"],
            "update_path": str(up_path),
            "update_sha256": up_hash,
            "accuracy": t_res["accuracy"],
            "loss": t_res["loss"],
            "duration": t_res["training_duration_seconds"],
            "quality_metrics": t_res["quality_metrics"],
            "diversity_metrics": t_res["diversity_metrics"],
            "resource_metrics": t_res["resource_metrics"],
            "comm_metrics": c["comm_metrics"],
            "arc_scoring": score_res,
            "arc_score": score_res["overall_arc_score"],
        })
        logger.info(f"  -> Update: {up_path.name} (SHA-256: {up_hash[:16]}...), Acc={t_res['accuracy']:.2f}%, ARC={score_res['overall_arc_score']:.2f}")

    # -------------------------------------------------------------
    # 3. Local Training Batch 2: Corrected BatchNorm + FedProx (mu=0.01)
    # -------------------------------------------------------------
    logger.info(f"\n--- Local Training Batch 2: Corrected BatchNorm + FedProx (mu={MU_FEDPROX}, LR={LR}) ---")
    clients_prox = []

    for c in client_configs:
        ds_path = Path(c["dataset"])
        ds_hash = sha256_file(ds_path)
        logger.info(f"Training {c['name']} (FedProx, freeze_bn=True, mu={MU_FEDPROX})...")

        torch.manual_seed(42)
        np.random.seed(42)

        t_res = train_local(
            dataset_path=str(ds_path),
            global_model_path=str(base_model_path),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            learning_rate=LR,
            freeze_bn_stats=True,
            mu=MU_FEDPROX,
        )

        up_path = Path(t_res["update_path"])
        up_hash = sha256_file(up_path)

        score_res = scoring_engine.score(
            quality_metrics=t_res["quality_metrics"],
            diversity_metrics=t_res["diversity_metrics"],
            resource_metrics=t_res["resource_metrics"],
            comm_metrics=c["comm_metrics"],
        )

        clients_prox.append({
            "client_id": c["client_id"],
            "name": c["name"],
            "dataset_path": str(ds_path),
            "dataset_sha256": ds_hash,
            "samples_trained": t_res["samples_trained"],
            "update_id": t_res["update_id"],
            "update_path": str(up_path),
            "update_sha256": up_hash,
            "accuracy": t_res["accuracy"],
            "loss": t_res["loss"],
            "duration": t_res["training_duration_seconds"],
            "quality_metrics": t_res["quality_metrics"],
            "diversity_metrics": t_res["diversity_metrics"],
            "resource_metrics": t_res["resource_metrics"],
            "comm_metrics": c["comm_metrics"],
            "arc_scoring": score_res,
            "arc_score": score_res["overall_arc_score"],
        })
        logger.info(f"  -> Update: {up_path.name} (SHA-256: {up_hash[:16]}...), Acc={t_res['accuracy']:.2f}%, ARC={score_res['overall_arc_score']:.2f}")

    # -------------------------------------------------------------
    # 4. Aggregation Weight Calculations
    # -------------------------------------------------------------
    logger.info("\n--- Aggregation Weight Calculations ---")
    sub_std = [
        {"submission_id": c["update_id"], "client_id": c["client_id"], "model_path": c["update_path"], "samples_trained": c["samples_trained"], "arc_score": c["arc_score"]}
        for c in clients_std
    ]
    sub_prox = [
        {"submission_id": c["update_id"], "client_id": c["client_id"], "model_path": c["update_path"], "samples_trained": c["samples_trained"], "arc_score": c["arc_score"]}
        for c in clients_prox
    ]

    weights_fedavg = compute_weights(sub_std, "FEDAVG")
    weights_equal = compute_weights(sub_std, "EQUAL")
    weights_arc = compute_weights(sub_std, "ARC_WEIGHTED")
    weights_arc_prox = compute_weights(sub_prox, "ARC_WEIGHTED")

    for i, c in enumerate(clients_std):
        c["weights"] = {"fedavg": weights_fedavg[i]["normalized_weight"], "equal": weights_equal[i]["normalized_weight"], "arc": weights_arc[i]["normalized_weight"]}
        logger.info(f"  {c['name']} (Standard): FedAvg={c['weights']['fedavg']:.4f}, Equal={c['weights']['equal']:.4f}, ARC={c['weights']['arc']:.4f}")

    for i, c in enumerate(clients_prox):
        c["weights"] = {"arc_fedprox": weights_arc_prox[i]["normalized_weight"]}
        logger.info(f"  {c['name']} (FedProx): ARC-FedProx={c['weights']['arc_fedprox']:.4f}")

    # -------------------------------------------------------------
    # 5. Tensor Aggregation of Four Candidates
    # -------------------------------------------------------------
    logger.info("\n--- Tensor Aggregation of Four Independent Candidates ---")
    exp_candidates_def = {
        "candidate_a_fedavg": {
            "name": "Candidate A: FedAvg (Corrected BN)",
            "submissions": weights_fedavg,
            "path": exp_dir / "candidate_a_fedavg.pt",
        },
        "candidate_b_equal": {
            "name": "Candidate B: Equal (Corrected BN)",
            "submissions": weights_equal,
            "path": exp_dir / "candidate_b_equal.pt",
        },
        "candidate_c_arc": {
            "name": "Candidate C: ARC-FL (Corrected BN)",
            "submissions": weights_arc,
            "path": exp_dir / "candidate_c_arc.pt",
        },
        "candidate_d_arc_fedprox": {
            "name": "Candidate D: ARC-FL + FedProx (Corrected BN, mu=0.01)",
            "submissions": weights_arc_prox,
            "path": exp_dir / "candidate_d_arc_fedprox.pt",
        },
    }

    for tag, cdef in exp_candidates_def.items():
        logger.info(f"Aggregating {cdef['name']}...")
        agg_res = aggregate(
            submissions_with_weights=cdef["submissions"],
            base_model_path=str(base_model_path),
            server_learning_rate=SERVER_DELTA_LR,
        )
        shutil.copyfile(agg_res["candidate_path"], cdef["path"])
        cand_hash = sha256_file(cdef["path"])
        cdef["sha256"] = cand_hash
        logger.info(f"  -> Saved {cdef['path'].name} (SHA-256: {cand_hash[:16]}...)")

    # -------------------------------------------------------------
    # 6. Comprehensive Evaluation Across Benchmarks A and B
    # -------------------------------------------------------------
    logger.info("\n--- Benchmark Evaluations Across All 5 Models ---")
    eval_list = [
        ("base", "Initial Global Model (initial-v1)", base_model_path, base_sha256),
        ("cand_a", exp_candidates_def["candidate_a_fedavg"]["name"], exp_candidates_def["candidate_a_fedavg"]["path"], exp_candidates_def["candidate_a_fedavg"]["sha256"]),
        ("cand_b", exp_candidates_def["candidate_b_equal"]["name"], exp_candidates_def["candidate_b_equal"]["path"], exp_candidates_def["candidate_b_equal"]["sha256"]),
        ("cand_c", exp_candidates_def["candidate_c_arc"]["name"], exp_candidates_def["candidate_c_arc"]["path"], exp_candidates_def["candidate_c_arc"]["sha256"]),
        ("cand_d", exp_candidates_def["candidate_d_arc_fedprox"]["name"], exp_candidates_def["candidate_d_arc_fedprox"]["path"], exp_candidates_def["candidate_d_arc_fedprox"]["sha256"]),
    ]

    eval_results = {}
    for tag, dname, mpath, mhash in eval_list:
        logger.info(f"Evaluating {dname}...")
        m_obj, _ = load_checkpoint(str(mpath))
        m_obj.to(device)
        m_obj.eval()

        t0 = time.time()
        res_a = evaluate_on_dataset(m_obj, df_bench_a, cfg, scaler, label_encoder, device)
        dur_a = round(time.time() - t0, 2)
        res_a["duration"] = dur_a

        t0 = time.time()
        res_b = evaluate_on_dataset(m_obj, df_bench_b, cfg, scaler, label_encoder, device)
        dur_b = round(time.time() - t0, 2)
        res_b["duration"] = dur_b

        eval_results[tag] = {
            "name": dname,
            "checkpoint_path": str(mpath),
            "checkpoint_sha256": mhash,
            "benchmark_a_primary": res_a,
            "benchmark_b_secondary": res_b,
        }
        logger.info(f"  [{tag}] Bench A -> Acc={res_a['accuracy']:.2f}%, Macro F1={res_a['macro_f1']:.2f}%, Loss={res_a['loss']:.4f}")
        logger.info(f"  [{tag}] Bench B -> Acc={res_b['accuracy']:.2f}%, Macro F1={res_b['macro_f1']:.2f}%, Loss={res_b['loss']:.4f}")

    # -------------------------------------------------------------
    # 7. Update Norm and Cosine Similarity Analysis
    # -------------------------------------------------------------
    base_ckpt = torch.load(base_model_path, map_location="cpu", weights_only=False)
    base_state = base_ckpt["model"]

    def flatten_weights(sd):
        tensors = []
        for k, v in sorted(sd.items()):
            if "weight" in k or "bias" in k:
                tensors.append(v.float().view(-1))
        return torch.cat(tensors)

    base_w_vec = flatten_weights(base_state)
    logger.info(f"Base Model Weight Norm: {torch.norm(base_w_vec).item():.4f}")

    client_deltas_std = {}
    for c in clients_std:
        sd = torch.load(c["update_path"], map_location="cpu", weights_only=False)["model"]
        w_vec = flatten_weights(sd)
        delta = w_vec - base_w_vec
        client_deltas_std[c["name"]] = delta

    cnames = list(client_deltas_std.keys())
    cosine_sims = {}
    for i in range(len(cnames)):
        for j in range(i + 1, len(cnames)):
            d1 = client_deltas_std[cnames[i]]
            d2 = client_deltas_std[cnames[j]]
            sim = float(torch.dot(d1, d2) / (torch.norm(d1) * torch.norm(d2)))
            cosine_sims[f"{cnames[i]} vs {cnames[j]}"] = round(sim, 4)

    # -------------------------------------------------------------
    # 8. Generate Visual Comparison Graph
    # -------------------------------------------------------------
    plot_path = exp_dir / "experiment_002_comparison.png"
    labels = ["Initial-v1", "FedAvg (Corr BN)", "Equal (Corr BN)", "ARC-FL (Corr BN)", "ARC + FedProx"]
    plot_tags = ["base", "cand_a", "cand_b", "cand_c", "cand_d"]

    acc_a = [eval_results[t]["benchmark_a_primary"]["accuracy"] for t in plot_tags]
    f1_a = [eval_results[t]["benchmark_a_primary"]["macro_f1"] for t in plot_tags]
    acc_b = [eval_results[t]["benchmark_b_secondary"]["accuracy"] for t in plot_tags]
    f1_b = [eval_results[t]["benchmark_b_secondary"]["macro_f1"] for t in plot_tags]

    x = np.arange(len(labels))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Benchmark A plot
    r1 = ax1.bar(x - width/2, acc_a, width, label="Accuracy (%)", color="#1565c0")
    r2 = ax1.bar(x + width/2, f1_a, width, label="Macro F1 (%)", color="#2e7d32")
    ax1.set_ylabel("Percentage (%)")
    ax1.set_title("Benchmark A (Primary Capped Distribution)")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=15, ha="right")
    ax1.set_ylim(40, 105)
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    ax1.legend(loc="lower right")

    for r in r1:
        h = r.get_height()
        ax1.annotate(f"{h:.2f}%", xy=(r.get_x() + r.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)
    for r in r2:
        h = r.get_height()
        ax1.annotate(f"{h:.2f}%", xy=(r.get_x() + r.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)

    # Benchmark B plot
    r3 = ax2.bar(x - width/2, acc_b, width, label="Accuracy (%)", color="#0288d1")
    r4 = ax2.bar(x + width/2, f1_b, width, label="Macro F1 (%)", color="#388e3c")
    ax2.set_ylabel("Percentage (%)")
    ax2.set_title("Benchmark B (Secondary Natural Traffic)")
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=15, ha="right")
    ax2.set_ylim(40, 105)
    ax2.grid(axis="y", linestyle="--", alpha=0.5)
    ax2.legend(loc="lower right")

    for r in r3:
        h = r.get_height()
        ax2.annotate(f"{h:.2f}%", xy=(r.get_x() + r.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)
    for r in r4:
        h = r.get_height()
        ax2.annotate(f"{h:.2f}%", xy=(r.get_x() + r.get_width() / 2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=8)

    plt.suptitle("ARC-FL Experiment 002: Stable Federated Learning Comparisons", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()
    logger.info(f"Saved comparison plot to {plot_path}")

    # -------------------------------------------------------------
    # 9. Save Machine-Readable JSON
    # -------------------------------------------------------------
    out_json = {
        "experiment_id": "EXP_002_STABLE_FL",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "COMPLETED",
        "protocol_version": "1.0",
        "hyperparameters": {
            "base_model": "initial-v1",
            "base_model_sha256": base_sha256,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "learning_rate": LR,
            "server_delta_lr": SERVER_DELTA_LR,
            "mu_fedprox": MU_FEDPROX,
            "freeze_bn_stats": True,
            "bn_affine_trainable": True,
        },
        "benchmarks": {
            "benchmark_a_primary": {"file": "benchmark_primary_capped.csv", "sha256": bench_a_hash, "samples": len(df_bench_a)},
            "benchmark_b_secondary": {"file": "benchmark_secondary_natural.csv", "sha256": bench_b_hash, "samples": len(df_bench_b)},
            "benchmark_c_quarantined_test": {"file": "P:\\project\\test.csv", "status": "STRICTLY_QUARANTINED_UNTOUCHED"},
        },
        "clients_standard": clients_std,
        "clients_fedprox": clients_prox,
        "cosine_similarities": cosine_sims,
        "candidates": {k: {"name": v["name"], "path": str(v["path"]), "sha256": v["sha256"]} for k, v in exp_candidates_def.items()},
        "evaluations": eval_results,
    }

    json_path = exp_dir / "experiment_002_results.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(out_json, f, indent=2)
    logger.info(f"Saved machine-readable JSON to {json_path}")

    print("\n" + "=" * 80)
    print(" EXPERIMENT 002 EXECUTION COMPLETED")
    print(f" JSON results: {json_path}")
    print(f" Plot image:   {plot_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
