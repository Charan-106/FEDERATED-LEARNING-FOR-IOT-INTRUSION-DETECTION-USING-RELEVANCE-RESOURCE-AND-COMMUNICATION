"""
ARC-FL Experiment 003: Heterogeneous Client Evaluation
Evaluates ARC-FL vs FedAvg vs Equal Aggregation across four distinct heterogeneity dimensions:
- Scenario A: Unequal Data Volume (6000 vs 3000 vs 1000 samples)
- Scenario B: Non-IID Class Distribution (Attack Family Specialization: 6 vs 9 vs 19 classes)
- Scenario C: Resource & Communication Heterogeneity (High-Tier Edge vs Mid-Tier Gateway vs Constrained IoT)
- Scenario D: Data Quality Heterogeneity (Clean C1, C2 vs 30% Corrupted C3)

Evaluates all candidates on:
- Benchmark A: Primary Evaluation Split (N = 50,001, class-capped)
- Benchmark B: Secondary Natural Validation Split (N = 50,000, natural)
- Base Model: initial-v1 (SHA-256 verified)
- Quarantined Test Set (P:/project/test.csv): STRICTLY UNTOUCHED
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, TensorDataset

# Setup paths
WORKSPACE_ROOT = Path(r"C:\Users\chara\.gemini\antigravity\scratch\arc-federated-learning")
sys.path.insert(0, str(WORKSPACE_ROOT / "ml-service"))

from app.aggregation.fedavg import aggregate, compute_weights
from app.arc.scoring import ARCScoringEngine
from app.models.resmlp import load_checkpoint
from app.training.trainer import train_local

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("exp_003")

# Hyperparameters (identical to stable Experiment 002 configuration)
EPOCHS = 2
BATCH_SIZE = 32
LR = 0.0001
SERVER_DELTA_LR = 0.5


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def evaluate_on_dataset(model: nn.Module, df: pd.DataFrame, cfg: dict, scaler, label_encoder, device: torch.device) -> dict:
    feature_cols = cfg["features"]
    X = df[feature_cols].values.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)
    if scaler is not None:
        X = scaler.transform(X)

    y_raw = df["label"].values
    if label_encoder is not None:
        y = label_encoder.transform(y_raw).astype(np.int64)
    else:
        y = y_raw.astype(np.int64)

    X_t = torch.tensor(X, dtype=torch.float32)
    y_t = torch.tensor(y, dtype=torch.long)

    dataset = TensorDataset(X_t, y_t)
    loader = DataLoader(dataset, batch_size=256, shuffle=False)

    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    all_preds = []
    all_targets = []

    model.eval()
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            out = model(batch_x)
            loss = criterion(out, batch_y)
            total_loss += loss.item() * len(batch_y)
            preds = torch.argmax(out, dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_targets.extend(batch_y.cpu().numpy())

    avg_loss = total_loss / len(df)
    acc = float(accuracy_score(all_targets, all_preds) * 100)
    macro_f1 = float(f1_score(all_targets, all_preds, average="macro", zero_division=0) * 100)
    weighted_f1 = float(f1_score(all_targets, all_preds, average="weighted", zero_division=0) * 100)
    macro_prec = float(precision_score(all_targets, all_preds, average="macro", zero_division=0) * 100)
    macro_rec = float(recall_score(all_targets, all_preds, average="macro", zero_division=0) * 100)

    # Per-class metrics
    per_class = {}
    classes = label_encoder.classes_ if label_encoder is not None else [str(i) for i in range(34)]
    for idx, cname in enumerate(classes):
        mask_t = (np.array(all_targets) == idx)
        support = int(np.sum(mask_t))
        if support == 0:
            continue
        p_c = float(precision_score(all_targets, all_preds, labels=[idx], average="macro", zero_division=0) * 100)
        r_c = float(recall_score(all_targets, all_preds, labels=[idx], average="macro", zero_division=0) * 100)
        f_c = float(f1_score(all_targets, all_preds, labels=[idx], average="macro", zero_division=0) * 100)
        per_class[cname] = {
            "precision": round(p_c, 2),
            "recall": round(r_c, 2),
            "f1": round(f_c, 2),
            "support": support,
        }

    return {
        "accuracy": round(acc, 4),
        "macro_precision": round(macro_prec, 4),
        "macro_recall": round(macro_rec, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "loss": round(avg_loss, 6),
        "sample_count": len(df),
        "per_class": per_class,
    }


def run_scenario(
    scenario_id: str,
    scenario_name: str,
    client_specs: list[dict],
    base_model_path: Path,
    base_sha256: str,
    df_bench_a: pd.DataFrame,
    df_bench_b: pd.DataFrame,
    cfg: dict,
    scaler,
    label_encoder,
    device: torch.device,
    exp_dir: Path,
    scoring_engine: ARCScoringEngine,
) -> dict:
    logger.info(f"\n=======================================================")
    logger.info(f" EXECUTING SCENARIO: {scenario_name} ({scenario_id})")
    logger.info(f"=======================================================")

    scen_dir = exp_dir / scenario_id
    scen_dir.mkdir(parents=True, exist_ok=True)

    client_results = []
    submissions = []

    for c in client_specs:
        ds_path = Path(c["dataset_path"])
        ds_hash = sha256_file(ds_path)
        logger.info(f"Training {c['name']} on {ds_path.name} ({c['samples']} samples)...")

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

        # ARC Scoring
        score_res = scoring_engine.score(
            quality_metrics=t_res["quality_metrics"],
            diversity_metrics=t_res["diversity_metrics"],
            resource_metrics=c.get("resource_telemetry", t_res["resource_metrics"]),
            comm_metrics=c["comm_metrics"],
        )

        c_info = {
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
            "resource_metrics": c.get("resource_telemetry", t_res["resource_metrics"]),
            "comm_metrics": c["comm_metrics"],
            "arc_scoring": score_res,
            "arc_score": score_res["overall_arc_score"],
        }
        client_results.append(c_info)
        submissions.append({
            "submission_id": t_res["update_id"],
            "client_id": c["client_id"],
            "model_path": str(up_path),
            "samples_trained": t_res["samples_trained"],
            "arc_score": score_res["overall_arc_score"],
        })
        logger.info(f"  -> {c['name']}: Acc={t_res['accuracy']:.2f}%, ARC Score={score_res['overall_arc_score']:.2f}")

    # Compute weights for all 3 strategies
    w_fedavg = compute_weights(submissions, "FEDAVG")
    w_equal = compute_weights(submissions, "EQUAL")
    w_arc = compute_weights(submissions, "ARC_WEIGHTED")

    for i, c in enumerate(client_results):
        c["weights"] = {
            "fedavg": w_fedavg[i]["normalized_weight"],
            "equal": w_equal[i]["normalized_weight"],
            "arc": w_arc[i]["normalized_weight"],
        }
        logger.info(f"  Weights for {c['name']}: FedAvg={c['weights']['fedavg']:.4f}, Equal={c['weights']['equal']:.4f}, ARC={c['weights']['arc']:.4f}")

    # Perform Tensor Aggregation for 3 candidates
    candidates_def = {
        "candidate_fedavg": {
            "name": f"{scenario_name} - FedAvg",
            "strategy": "FEDAVG",
            "submissions": w_fedavg,
            "path": scen_dir / f"{scenario_id}_fedavg.pt",
        },
        "candidate_equal": {
            "name": f"{scenario_name} - Equal",
            "strategy": "EQUAL",
            "submissions": w_equal,
            "path": scen_dir / f"{scenario_id}_equal.pt",
        },
        "candidate_arc": {
            "name": f"{scenario_name} - ARC-FL",
            "strategy": "ARC_WEIGHTED",
            "submissions": w_arc,
            "path": scen_dir / f"{scenario_id}_arc.pt",
        },
    }

    cand_evaluations = {}
    for ckey, cdef in candidates_def.items():
        logger.info(f"Aggregating {cdef['name']}...")
        agg_res = aggregate(
            submissions_with_weights=cdef["submissions"],
            base_model_path=str(base_model_path),
            server_learning_rate=SERVER_DELTA_LR,
        )
        shutil.copyfile(agg_res["candidate_path"], cdef["path"])
        cand_hash = sha256_file(cdef["path"])
        cdef["sha256"] = cand_hash

        # Evaluate candidate
        m_obj, _ = load_checkpoint(str(cdef["path"]))
        m_obj.to(device)
        m_obj.eval()

        t0 = time.time()
        res_a = evaluate_on_dataset(m_obj, df_bench_a, cfg, scaler, label_encoder, device)
        res_a["duration"] = round(time.time() - t0, 2)

        t0 = time.time()
        res_b = evaluate_on_dataset(m_obj, df_bench_b, cfg, scaler, label_encoder, device)
        res_b["duration"] = round(time.time() - t0, 2)

        cand_evaluations[ckey] = {
            "name": cdef["name"],
            "strategy": cdef["strategy"],
            "checkpoint_path": str(cdef["path"]),
            "checkpoint_sha256": cand_hash,
            "benchmark_a_primary": res_a,
            "benchmark_b_secondary": res_b,
        }
        logger.info(f"  [{ckey}] Bench A -> Acc={res_a['accuracy']:.2f}%, Macro F1={res_a['macro_f1']:.2f}%, Loss={res_a['loss']:.4f}")
        logger.info(f"  [{ckey}] Bench B -> Acc={res_b['accuracy']:.2f}%, Macro F1={res_b['macro_f1']:.2f}%, Loss={res_b['loss']:.4f}")

    return {
        "scenario_id": scenario_id,
        "scenario_name": scenario_name,
        "clients": client_results,
        "candidates": cand_evaluations,
    }


def main():
    logger.info("=" * 80)
    logger.info(" STARTING CONTROLLED FEDERATED LEARNING EXPERIMENT 003: HETEROGENEOUS EVALUATION")
    logger.info("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Paths
    base_model_path = WORKSPACE_ROOT / "storage" / "global-models" / "initial-v1" / "best.pt"
    benchmarks_dir = WORKSPACE_ROOT / "storage" / "benchmarks"
    bench_a_path = benchmarks_dir / "benchmark_primary_capped.csv"
    bench_b_path = benchmarks_dir / "benchmark_secondary_natural.csv"
    artifacts_dir = WORKSPACE_ROOT / "ml-service" / "artifacts"
    exp_dir = WORKSPACE_ROOT / "storage" / "experiments" / "exp_003"
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Cryptographic integrity checks
    base_sha256 = sha256_file(base_model_path)
    bench_a_sha256 = sha256_file(bench_a_path)
    bench_b_sha256 = sha256_file(bench_b_path)

    logger.info(f"Starting Global Model: initial-v1 (SHA-256: {base_sha256})")
    logger.info(f"Benchmark A SHA-256: {bench_a_sha256}")
    logger.info(f"Benchmark B SHA-256: {bench_b_sha256}")

    # Load artifacts & benchmarks
    import joblib
    with open(artifacts_dir / "model_config.json") as f:
        cfg = json.load(f)
    scaler = joblib.load(artifacts_dir / "scaler.pkl")
    label_encoder = joblib.load(artifacts_dir / "label_encoder.pkl")

    logger.info("Loading Benchmark A...")
    df_bench_a = pd.read_csv(bench_a_path)
    logger.info(f"Benchmark A loaded: {len(df_bench_a)} rows")

    logger.info("Loading Benchmark B...")
    df_bench_b = pd.read_csv(bench_b_path)
    logger.info(f"Benchmark B loaded: {len(df_bench_b)} rows")

    # Evaluate Initial Base Model once as invariant baseline
    logger.info("\nEvaluating Invariant Base Model (initial-v1)...")
    base_model, _ = load_checkpoint(str(base_model_path))
    base_model.to(device)
    base_eval_a = evaluate_on_dataset(base_model, df_bench_a, cfg, scaler, label_encoder, device)
    base_eval_b = evaluate_on_dataset(base_model, df_bench_b, cfg, scaler, label_encoder, device)
    logger.info(f"  [Base] Bench A -> Acc={base_eval_a['accuracy']:.2f}%, Macro F1={base_eval_a['macro_f1']:.2f}%, Loss={base_eval_a['loss']:.4f}")
    logger.info(f"  [Base] Bench B -> Acc={base_eval_b['accuracy']:.2f}%, Macro F1={base_eval_b['macro_f1']:.2f}%, Loss={base_eval_b['loss']:.4f}")

    base_eval_summary = {
        "name": "Initial Global Model (initial-v1)",
        "checkpoint_path": str(base_model_path),
        "checkpoint_sha256": base_sha256,
        "benchmark_a_primary": base_eval_a,
        "benchmark_b_secondary": base_eval_b,
    }

    scoring_engine = ARCScoringEngine({})

    # =========================================================================
    # SCENARIO A: Unequal Data Volume (6000 vs 3000 vs 1000)
    # =========================================================================
    scen_a_specs = [
        {
            "client_id": "client-1",
            "name": "Client 1 (Large Volume - 6000)",
            "dataset_path": str(exp_dir / "scenario_a" / "client1_vol6000.csv"),
            "samples": 6000,
            "comm_metrics": {"bandwidth_mbps": 85.0, "latency_ms": 12.0, "reliability_pct": 99.0},
        },
        {
            "client_id": "client-2",
            "name": "Client 2 (Medium Volume - 3000)",
            "dataset_path": str(exp_dir / "scenario_a" / "client2_vol3000.csv"),
            "samples": 3000,
            "comm_metrics": {"bandwidth_mbps": 50.0, "latency_ms": 25.0, "reliability_pct": 98.0},
        },
        {
            "client_id": "client-3",
            "name": "Client 3 (Small Volume - 1000)",
            "dataset_path": str(exp_dir / "scenario_a" / "client3_vol1000.csv"),
            "samples": 1000,
            "comm_metrics": {"bandwidth_mbps": 30.0, "latency_ms": 40.0, "reliability_pct": 96.0},
        },
    ]
    res_scen_a = run_scenario("scenario_a", "Scenario A: Unequal Data Volume", scen_a_specs, base_model_path, base_sha256, df_bench_a, df_bench_b, cfg, scaler, label_encoder, device, exp_dir, scoring_engine)

    # =========================================================================
    # SCENARIO B: Non-IID Class Distribution (Attack Family Specialization)
    # =========================================================================
    scen_b_specs = [
        {
            "client_id": "client-1",
            "name": "Client 1 (Volumetric Floods Specialist)",
            "dataset_path": str(exp_dir / "scenario_b" / "client1_non_iid.csv"),
            "samples": 1000,
            "comm_metrics": {"bandwidth_mbps": 60.0, "latency_ms": 20.0, "reliability_pct": 98.0},
        },
        {
            "client_id": "client-2",
            "name": "Client 2 (Botnet & Protocol Specialist)",
            "dataset_path": str(exp_dir / "scenario_b" / "client2_non_iid.csv"),
            "samples": 1000,
            "comm_metrics": {"bandwidth_mbps": 60.0, "latency_ms": 20.0, "reliability_pct": 98.0},
        },
        {
            "client_id": "client-3",
            "name": "Client 3 (Recon & Web Attacks Specialist)",
            "dataset_path": str(exp_dir / "scenario_b" / "client3_non_iid.csv"),
            "samples": 1000,
            "comm_metrics": {"bandwidth_mbps": 60.0, "latency_ms": 20.0, "reliability_pct": 98.0},
        },
    ]
    res_scen_b = run_scenario("scenario_b", "Scenario B: Non-IID Class Distribution", scen_b_specs, base_model_path, base_sha256, df_bench_a, df_bench_b, cfg, scaler, label_encoder, device, exp_dir, scoring_engine)

    # =========================================================================
    # SCENARIO C: Resource and Communication Heterogeneity
    # =========================================================================
    scen_c_specs = [
        {
            "client_id": "client-1",
            "name": "Client 1 (High-Tier Edge Server)",
            "dataset_path": str(exp_dir / "scenario_b" / "client1_non_iid.csv"),
            "samples": 1000,
            "resource_telemetry": {
                "available_ram_gb": 24.0,
                "total_ram_gb": 32.0,
                "cpu_idle_pct": 85.0,
                "throughput_samples_per_sec": 1200.0,
                "label": "SIMULATED_HIGH_TIER_EDGE",
            },
            "comm_metrics": {"bandwidth_mbps": 100.0, "latency_ms": 8.0, "reliability_pct": 99.5, "label": "SIMULATED_100MBPS_FIBER"},
        },
        {
            "client_id": "client-2",
            "name": "Client 2 (Mid-Tier Gateway)",
            "dataset_path": str(exp_dir / "scenario_b" / "client2_non_iid.csv"),
            "samples": 1000,
            "resource_telemetry": {
                "available_ram_gb": 4.0,
                "total_ram_gb": 8.0,
                "cpu_idle_pct": 40.0,
                "throughput_samples_per_sec": 450.0,
                "label": "SIMULATED_MID_TIER_GATEWAY",
            },
            "comm_metrics": {"bandwidth_mbps": 30.0, "latency_ms": 45.0, "reliability_pct": 95.0, "label": "SIMULATED_30MBPS_BROADBAND"},
        },
        {
            "client_id": "client-3",
            "name": "Client 3 (Low-Tier Constrained Sensor)",
            "dataset_path": str(exp_dir / "scenario_b" / "client3_non_iid.csv"),
            "samples": 1000,
            "resource_telemetry": {
                "available_ram_gb": 0.15,
                "total_ram_gb": 1.0,
                "cpu_idle_pct": 8.0,
                "throughput_samples_per_sec": 60.0,
                "label": "SIMULATED_LOW_TIER_SENSOR",
            },
            "comm_metrics": {"bandwidth_mbps": 2.0, "latency_ms": 250.0, "reliability_pct": 80.0, "label": "SIMULATED_2MBPS_CELLULAR"},
        },
    ]
    res_scen_c = run_scenario("scenario_c", "Scenario C: Resource & Communication Heterogeneity", scen_c_specs, base_model_path, base_sha256, df_bench_a, df_bench_b, cfg, scaler, label_encoder, device, exp_dir, scoring_engine)

    # =========================================================================
    # SCENARIO D: Data Quality Heterogeneity
    # =========================================================================
    scen_d_specs = [
        {
            "client_id": "client-1",
            "name": "Client 1 (Clean Data)",
            "dataset_path": str(exp_dir / "scenario_d" / "client1_clean.csv"),
            "samples": 1000,
            "comm_metrics": {"bandwidth_mbps": 70.0, "latency_ms": 15.0, "reliability_pct": 99.0},
        },
        {
            "client_id": "client-2",
            "name": "Client 2 (Clean Data)",
            "dataset_path": str(exp_dir / "scenario_d" / "client2_clean.csv"),
            "samples": 1000,
            "comm_metrics": {"bandwidth_mbps": 70.0, "latency_ms": 15.0, "reliability_pct": 99.0},
        },
        {
            "client_id": "client-3",
            "name": "Client 3 (30% Corrupted Data)",
            "dataset_path": str(exp_dir / "scenario_d" / "client3_corrupted.csv"),
            "samples": 1000,
            "comm_metrics": {"bandwidth_mbps": 70.0, "latency_ms": 15.0, "reliability_pct": 99.0},
        },
    ]
    res_scen_d = run_scenario("scenario_d", "Scenario D: Data Quality Heterogeneity", scen_d_specs, base_model_path, base_sha256, df_bench_a, df_bench_b, cfg, scaler, label_encoder, device, exp_dir, scoring_engine)

    # Compile Final Structured JSON Results
    experiment_data = {
        "experiment_id": "EXP_003_HETEROGENEOUS_CLIENT_EVAL",
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
            "freeze_bn_stats": True,
            "bn_affine_trainable": True,
        },
        "benchmarks": {
            "benchmark_a_primary": {"file": "benchmark_primary_capped.csv", "sha256": bench_a_sha256, "samples": len(df_bench_a)},
            "benchmark_b_secondary": {"file": "benchmark_secondary_natural.csv", "sha256": bench_b_sha256, "samples": len(df_bench_b)},
            "benchmark_c_quarantined": {"file": "P:\\project\\test.csv", "status": "STRICTLY_QUARANTINED_UNTOUCHED"},
        },
        "base_model_evaluation": base_eval_summary,
        "scenarios": {
            "scenario_a": res_scen_a,
            "scenario_b": res_scen_b,
            "scenario_c": res_scen_c,
            "scenario_d": res_scen_d,
        },
    }

    # Generate 4-Panel Visualization Plot
    logger.info("\nGenerating 4-panel comparison visualization...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle("Experiment 003: Heterogeneous Client Evaluation (ARC vs FedAvg vs Equal)", fontsize=16, fontweight="bold")

    scenarios = [
        ("scenario_a", "Scenario A: Unequal Data Volume", axes[0, 0]),
        ("scenario_b", "Scenario B: Non-IID Class Distribution", axes[0, 1]),
        ("scenario_c", "Scenario C: Resource & Comm Heterogeneity", axes[1, 0]),
        ("scenario_d", "Scenario D: Data Quality Heterogeneity", axes[1, 1]),
    ]

    labels = ["Base", "FedAvg", "Equal", "ARC-FL"]

    for sc_key, sc_title, ax in scenarios:
        sc_data = experiment_data["scenarios"][sc_key]
        cands = sc_data["candidates"]

        acc_a = [base_eval_a["accuracy"], cands["candidate_fedavg"]["benchmark_a_primary"]["accuracy"], cands["candidate_equal"]["benchmark_a_primary"]["accuracy"], cands["candidate_arc"]["benchmark_a_primary"]["accuracy"]]
        acc_b = [base_eval_b["accuracy"], cands["candidate_fedavg"]["benchmark_b_secondary"]["accuracy"], cands["candidate_equal"]["benchmark_b_secondary"]["accuracy"], cands["candidate_arc"]["benchmark_b_secondary"]["accuracy"]]
        f1_a = [base_eval_a["macro_f1"], cands["candidate_fedavg"]["benchmark_a_primary"]["macro_f1"], cands["candidate_equal"]["benchmark_a_primary"]["macro_f1"], cands["candidate_arc"]["benchmark_a_primary"]["macro_f1"]]

        x = np.arange(len(labels))
        width = 0.25

        ax.bar(x - width, acc_a, width, label="Bench A Acc (%)", color="#1f77b4", alpha=0.9)
        ax.bar(x, acc_b, width, label="Bench B Acc (%)", color="#2ca02c", alpha=0.9)
        ax.bar(x + width, f1_a, width, label="Bench A Macro F1 (%)", color="#ff7f0e", alpha=0.9)

        ax.set_title(sc_title, fontweight="bold", fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontweight="bold")
        ax.set_ylim(70, 102)
        ax.grid(axis="y", linestyle="--", alpha=0.5)
        ax.legend(loc="lower right", fontsize=9)

        # Annotate values
        for i in range(len(labels)):
            ax.text(x[i] - width, acc_a[i] + 0.5, f"{acc_a[i]:.1f}", ha="center", fontsize=8, rotation=90)
            ax.text(x[i], acc_b[i] + 0.5, f"{acc_b[i]:.1f}", ha="center", fontsize=8, rotation=90)
            ax.text(x[i] + width, f1_a[i] + 0.5, f"{f1_a[i]:.1f}", ha="center", fontsize=8, rotation=90)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot_path = exp_dir / "experiment_003_comparison.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()
    logger.info(f"Saved comparison plot to {plot_path}")

    # Copy plot to brain artifact directory for UI display
    brain_plot = Path(r"C:\Users\chara\.gemini\antigravity\brain\e12abf92-5a88-4d58-a6af-27f5821ed251\experiment_003_comparison.png")
    shutil.copyfile(plot_path, brain_plot)

    # Save JSON results
    json_path = exp_dir / "experiment_003_results.json"
    with open(json_path, "w") as f:
        json.dump(experiment_data, f, indent=2)
    logger.info(f"Saved machine-readable JSON to {json_path}")

    logger.info("=" * 80)
    logger.info(" EXPERIMENT 003 EXECUTION COMPLETED SUCCESSFULLY")
    logger.info(f" JSON results: {json_path}")
    logger.info(f" Plot image:   {plot_path}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
