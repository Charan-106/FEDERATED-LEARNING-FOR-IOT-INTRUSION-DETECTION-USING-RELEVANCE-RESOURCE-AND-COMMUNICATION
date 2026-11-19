"""
ARC-FL Experiment 004: Multi-Seed, Multi-Round Federated Learning Benchmark
Compares four aggregation methods over 5 communication rounds across 5 independent seeds:
1. FedAvg: Sample-proportional weighting
2. Equal: Uniform weighting across clients
3. ARC-FL (Original): lambda_R=0.40, lambda_S=0.30, lambda_C=0.30
4. ARC-FL (Relevance-Focused): lambda_R=0.60, lambda_S=0.20, lambda_C=0.20

Random Seeds: [42, 101, 202, 303, 404]
Communication Rounds: 5

Strict research integrity constraints:
- Base model initial-v1 is preserved invariant.
- Benchmarks A and B are used exclusively for post-round evaluation.
- P:/project/test.csv is STRICTLY QUARANTINED.
- No model is published to production.
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
from scipy import stats
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
logger = logging.getLogger("exp_004")

# Protocol constants
SEEDS = [42, 101, 202, 303, 404]
NUM_ROUNDS = 5
EPOCHS_PER_ROUND = 2
BATCH_SIZE = 32
LR = 0.0001
SERVER_DELTA_LR = 0.5
DROPOUT_ENABLED = False  # Controlled baseline: all 3 clients participate across all 5 rounds (0% dropout)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1024 * 1024):
            h.update(chunk)
    return h.hexdigest()


def build_benchmark_loader(df: pd.DataFrame, cfg: dict, scaler, label_encoder, batch_size: int = 1024):
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
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    classes = list(label_encoder.classes_) if label_encoder is not None else [str(i) for i in range(34)]
    return loader, len(df), classes


def evaluate_on_loader(model: nn.Module, loader: DataLoader, sample_count: int, classes: list, device: torch.device) -> dict:
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

    avg_loss = total_loss / sample_count
    acc = float(accuracy_score(all_targets, all_preds) * 100)
    macro_f1 = float(f1_score(all_targets, all_preds, average="macro", zero_division=0) * 100)
    weighted_f1 = float(f1_score(all_targets, all_preds, average="weighted", zero_division=0) * 100)
    macro_prec = float(precision_score(all_targets, all_preds, average="macro", zero_division=0) * 100)
    macro_rec = float(recall_score(all_targets, all_preds, average="macro", zero_division=0) * 100)

    # Per-class metrics
    per_class = {}
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
        "sample_count": sample_count,
        "per_class": per_class,
    }


def evaluate_on_dataset(model: nn.Module, df: pd.DataFrame, cfg: dict, scaler, label_encoder, device: torch.device) -> dict:
    loader, count, classes = build_benchmark_loader(df, cfg, scaler, label_encoder)
    return evaluate_on_loader(model, loader, count, classes, device)


def run_experiment_004():
    """
    Main execution loop for Experiment 004.
    Runs 5 independent seeds x 5 rounds x 4 aggregation methods.
    """
    logger.info("=" * 80)
    logger.info(" STARTING CONTROLLED FEDERATED LEARNING EXPERIMENT 004")
    logger.info(f" Seeds: {SEEDS} | Rounds: {NUM_ROUNDS} | Methods: 4")
    logger.info("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using compute device: {device}")

    # Paths
    base_model_path = WORKSPACE_ROOT / "storage" / "global-models" / "initial-v1" / "best.pt"
    benchmarks_dir = WORKSPACE_ROOT / "storage" / "benchmarks"
    bench_a_path = benchmarks_dir / "benchmark_primary_capped.csv"
    bench_b_path = benchmarks_dir / "benchmark_secondary_natural.csv"
    artifacts_dir = WORKSPACE_ROOT / "ml-service" / "artifacts"
    exp_dir = WORKSPACE_ROOT / "storage" / "experiments" / "exp_004"
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Verification of baseline
    base_sha256 = sha256_file(base_model_path)
    bench_a_sha256 = sha256_file(bench_a_path)
    bench_b_sha256 = sha256_file(bench_b_path)

    logger.info(f"Base Model Verified: {base_sha256}")
    logger.info(f"Benchmark A Verified: {bench_a_sha256}")
    logger.info(f"Benchmark B Verified: {bench_b_sha256}")

    import joblib
    with open(artifacts_dir / "model_config.json") as f:
        cfg = json.load(f)
    scaler = joblib.load(artifacts_dir / "scaler.pkl")
    label_encoder = joblib.load(artifacts_dir / "label_encoder.pkl")

    logger.info("Loading Benchmark A (Primary)...")
    df_bench_a = pd.read_csv(bench_a_path)
    logger.info("Loading Benchmark B (Secondary)...")
    df_bench_b = pd.read_csv(bench_b_path)

    logger.info("Building Benchmark A & B DataLoaders...")
    loader_bench_a, count_a, classes_a = build_benchmark_loader(df_bench_a, cfg, scaler, label_encoder, batch_size=1024)
    loader_bench_b, count_b, classes_b = build_benchmark_loader(df_bench_b, cfg, scaler, label_encoder, batch_size=1024)

    # Invariant base model evaluation
    base_model, _ = load_checkpoint(str(base_model_path))
    base_model.to(device)
    base_eval_a = evaluate_on_loader(base_model, loader_bench_a, count_a, classes_a, device)
    base_eval_b = evaluate_on_loader(base_model, loader_bench_b, count_b, classes_b, device)
    logger.info(f"Base Benchmark A: Acc={base_eval_a['accuracy']:.2f}%, F1={base_eval_a['macro_f1']:.2f}%, Loss={base_eval_a['loss']:.4f}")
    logger.info(f"Base Benchmark B: Acc={base_eval_b['accuracy']:.2f}%, F1={base_eval_b['macro_f1']:.2f}%, Loss={base_eval_b['loss']:.4f}")

    # Client partition specifications (Heterogeneous Volume Skew: 6,000 / 3,000 / 1,000 + Data Corruption Defense)
    # Datasets are isolated copies stored in storage/experiments/exp_004/datasets/
    client_specs = [
        {
            "client_id": "client-1",
            "name": "Client 1 (High-Volume Clean Edge Server)",
            "dataset_path": str(WORKSPACE_ROOT / "storage" / "experiments" / "exp_004" / "datasets" / "client1_clean_6000.csv"),
            "expected_samples": 6000,
            "resource_telemetry": {"available_ram_gb": 24.0, "total_ram_gb": 32.0, "cpu_idle_pct": 85.0, "throughput_samples_per_sec": 1200.0, "provenance": "SIMULATED_HIGH_TIER_EDGE"},
            "comm_metrics": {"bandwidth_mbps": 100.0, "latency_ms": 8.0, "reliability_pct": 99.5, "provenance": "SIMULATED_100MBPS_FIBER"},
        },
        {
            "client_id": "client-2",
            "name": "Client 2 (Mid-Volume Clean Gateway)",
            "dataset_path": str(WORKSPACE_ROOT / "storage" / "experiments" / "exp_004" / "datasets" / "client2_clean_3000.csv"),
            "expected_samples": 3000,
            "resource_telemetry": {"available_ram_gb": 4.0, "total_ram_gb": 8.0, "cpu_idle_pct": 40.0, "throughput_samples_per_sec": 450.0, "provenance": "SIMULATED_MID_TIER_GATEWAY"},
            "comm_metrics": {"bandwidth_mbps": 30.0, "latency_ms": 45.0, "reliability_pct": 95.0, "provenance": "SIMULATED_30MBPS_BROADBAND"},
        },
        {
            "client_id": "client-3",
            "name": "Client 3 (Low-Volume Corrupted IoT Sensor)",
            "dataset_path": str(WORKSPACE_ROOT / "storage" / "experiments" / "exp_004" / "datasets" / "client3_corrupted_1000.csv"),
            "expected_samples": 1000,
            "resource_telemetry": {"available_ram_gb": 0.15, "total_ram_gb": 1.0, "cpu_idle_pct": 8.0, "throughput_samples_per_sec": 60.0, "provenance": "SIMULATED_LOW_TIER_SENSOR"},
            "comm_metrics": {"bandwidth_mbps": 2.0, "latency_ms": 250.0, "reliability_pct": 80.0, "provenance": "SIMULATED_2MBPS_CELLULAR"},
        },
    ]

    # Aggregation method configurations
    methods = {
        "fedavg": {
            "name": "FedAvg",
            "strategy": "FEDAVG",
            "arc_engine": None,
        },
        "equal": {
            "name": "Equal Aggregation",
            "strategy": "EQUAL",
            "arc_engine": None,
        },
        "arc_orig": {
            "name": "ARC-FL (Original: 0.40, 0.30, 0.30)",
            "strategy": "ARC_WEIGHTED",
            "arc_engine": ARCScoringEngine({"lambda_relevance": 0.40, "lambda_resource": 0.30, "lambda_communication": 0.30}),
        },
        "arc_rel": {
            "name": "ARC-FL (Relevance-Focused: 0.60, 0.20, 0.20)",
            "strategy": "ARC_WEIGHTED",
            "arc_engine": ARCScoringEngine({"lambda_relevance": 0.60, "lambda_resource": 0.20, "lambda_communication": 0.20}),
        },
    }

    # Structure to hold all round evaluations
    all_runs = {}

    for seed in SEEDS:
        logger.info(f"\n=======================================================")
        logger.info(f" STARTING SEED {seed} ({SEEDS.index(seed)+1}/{len(SEEDS)})")
        logger.info(f"=======================================================")

        all_runs[f"seed_{seed}"] = {}

        # Trajectory models for each method
        trajectory_models = {}
        for mkey in methods:
            init_model_copy = exp_dir / f"init_{mkey}_seed_{seed}.pt"
            shutil.copyfile(base_model_path, init_model_copy)
            trajectory_models[mkey] = init_model_copy

        for round_idx in range(1, NUM_ROUNDS + 1):
            logger.info(f"\n--- [Seed {seed}] Communication Round {round_idx}/{NUM_ROUNDS} ---")
            round_data = {}

            for mkey, mdef in methods.items():
                logger.info(f"Method: {mdef['name']}")
                current_global = trajectory_models[mkey]

                client_updates = []
                submissions = []

                for c in client_specs:
                    t_res = train_local(
                        dataset_path=c["dataset_path"],
                        global_model_path=str(current_global),
                        epochs=EPOCHS_PER_ROUND,
                        batch_size=BATCH_SIZE,
                        learning_rate=LR,
                        freeze_bn_stats=True,
                        seed=seed + round_idx * 10,
                    )

                    up_path = Path(t_res["update_path"])
                    up_hash = sha256_file(up_path)

                    # Compute score if ARC method
                    if mdef["arc_engine"] is not None:
                        score_res = mdef["arc_engine"].score(
                            quality_metrics=t_res["quality_metrics"],
                            diversity_metrics=t_res["diversity_metrics"],
                            resource_metrics=c["resource_telemetry"],
                            comm_metrics=c["comm_metrics"],
                        )
                        arc_val = score_res["overall_arc_score"]
                    else:
                        score_res = {}
                        arc_val = 50.0

                    client_updates.append({
                        "client_id": c["client_id"],
                        "name": c["name"],
                        "samples_trained": t_res["samples_trained"],
                        "update_path": str(up_path),
                        "update_sha256": up_hash,
                        "update_norm": t_res.get("update_norm", 0.0),
                        "accuracy": t_res["accuracy"],
                        "loss": t_res["loss"],
                        "arc_score": arc_val,
                        "scoring_breakdown": score_res,
                    })

                    submissions.append({
                        "submission_id": t_res["update_id"],
                        "client_id": c["client_id"],
                        "model_path": str(up_path),
                        "samples_trained": t_res["samples_trained"],
                        "arc_score": arc_val,
                    })

                # Compute strategy weights
                w_list = compute_weights(submissions, mdef["strategy"])
                for i, cu in enumerate(client_updates):
                    cu["weight"] = w_list[i]["normalized_weight"]

                # Strict Weight Verification Check
                norm_w = [round(w["normalized_weight"], 4) for w in w_list]
                if mkey == "fedavg":
                    assert norm_w == [0.60, 0.30, 0.10], f"FedAvg weights mismatch: expected [0.60, 0.30, 0.10], got {norm_w}"
                elif mkey == "equal":
                    assert norm_w == [0.3333, 0.3333, 0.3333], f"Equal weights mismatch: expected [0.3333, 0.3333, 0.3333], got {norm_w}"

                # Perform Tensor Aggregation
                cand_round_path = exp_dir / f"model_{mkey}_seed_{seed}_r{round_idx}.pt"
                agg_res = aggregate(
                    submissions_with_weights=w_list,
                    base_model_path=str(current_global),
                    server_learning_rate=SERVER_DELTA_LR,
                )
                shutil.copyfile(agg_res["candidate_path"], cand_round_path)
                cand_hash = sha256_file(cand_round_path)

                # Update trajectory for next round
                trajectory_models[mkey] = cand_round_path

                # Evaluate on Benchmark A & B
                m_obj, _ = load_checkpoint(str(cand_round_path))
                m_obj.to(device)
                m_obj.eval()

                t0 = time.time()
                eval_a = evaluate_on_loader(m_obj, loader_bench_a, count_a, classes_a, device)
                eval_a["duration"] = round(time.time() - t0, 2)

                t0 = time.time()
                eval_b = evaluate_on_loader(m_obj, loader_bench_b, count_b, classes_b, device)
                eval_b["duration"] = round(time.time() - t0, 2)

                round_data[mkey] = {
                    "method": mdef["name"],
                    "strategy": mdef["strategy"],
                    "checkpoint_path": str(cand_round_path),
                    "checkpoint_sha256": cand_hash,
                    "weights": [cu["weight"] for cu in client_updates],
                    "client_metrics": client_updates,
                    "benchmark_a": eval_a,
                    "benchmark_b": eval_b,
                }
                logger.info(f"  -> Round {round_idx} {mkey}: Bench A Acc={eval_a['accuracy']:.2f}%, F1={eval_a['macro_f1']:.2f}% | Bench B Acc={eval_b['accuracy']:.2f}%, F1={eval_b['macro_f1']:.2f}%")

            all_runs[f"seed_{seed}"][f"round_{round_idx}"] = round_data

    # Statistical Aggregation Across 5 Seeds
    logger.info("\nComputing Statistical Aggregates Across 5 Independent Seeds...")
    stat_summary = compute_statistics(all_runs, methods)

    # Save Output Artifacts
    results_json = {
        "experiment_id": "EXP_004_MULTI_SEED_MULTI_ROUND_FL",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "COMPLETED",
        "protocol_version": "1.0",
        "seeds": SEEDS,
        "rounds": NUM_ROUNDS,
        "dropout_enabled": DROPOUT_ENABLED,
        "dropout_notes": "Controlled baseline: all 3 clients participate across all 5 rounds (0% dropout)",
        "client_datasets": [
            {
                "client_id": c["client_id"],
                "name": c["name"],
                "dataset_path": c["dataset_path"],
                "expected_samples": c["expected_samples"],
                "sha256": sha256_file(Path(c["dataset_path"])),
            }
            for c in client_specs
        ],
        "base_model": {"path": str(base_model_path), "sha256": base_sha256, "bench_a": base_eval_a, "bench_b": base_eval_b},
        "statistical_summary": stat_summary,
        "runs": all_runs,
    }

    json_path = exp_dir / "experiment_004_results.json"
    with open(json_path, "w") as f:
        json.dump(results_json, f, indent=2)
    logger.info(f"Saved complete JSON results: {json_path}")

    # Generate Convergence and Comparison Plots
    generate_plots(all_runs, stat_summary, exp_dir)
    logger.info("Experiment 004 execution completed successfully.")


def compute_statistics(all_runs: dict, methods: dict) -> dict:
    """Compute mean, std, 95% CI, and paired t-tests across seeds."""
    summary = {}
    for mkey, mdef in methods.items():
        summary[mkey] = {
            "name": mdef["name"],
            "round_5_final": {
                "bench_a_acc": [],
                "bench_a_f1": [],
                "bench_b_acc": [],
                "bench_b_f1": [],
            }
        }
        for s in SEEDS:
            r5 = all_runs[f"seed_{s}"]["round_5"][mkey]
            summary[mkey]["round_5_final"]["bench_a_acc"].append(r5["benchmark_a"]["accuracy"])
            summary[mkey]["round_5_final"]["bench_a_f1"].append(r5["benchmark_a"]["macro_f1"])
            summary[mkey]["round_5_final"]["bench_b_acc"].append(r5["benchmark_b"]["accuracy"])
            summary[mkey]["round_5_final"]["bench_b_f1"].append(r5["benchmark_b"]["macro_f1"])

        # Compute descriptive stats
        for metric in ["bench_a_acc", "bench_a_f1", "bench_b_acc", "bench_b_f1"]:
            vals = summary[mkey]["round_5_final"][metric]
            mean_v = float(np.mean(vals))
            std_v = float(np.std(vals, ddof=1))
            # 95% Confidence Interval with t-distribution (df=4, t=2.776)
            ci_half = 2.776 * (std_v / np.sqrt(len(vals)))
            summary[mkey]["round_5_final"][f"{metric}_mean"] = round(mean_v, 4)
            summary[mkey]["round_5_final"][f"{metric}_std"] = round(std_v, 4)
            summary[mkey]["round_5_final"][f"{metric}_ci95"] = [round(mean_v - ci_half, 4), round(mean_v + ci_half, 4)]

    # Paired Student's t-test comparing ARC methods against FedAvg
    fedavg_acc_a = summary["fedavg"]["round_5_final"]["bench_a_acc"]
    fedavg_f1_a = summary["fedavg"]["round_5_final"]["bench_a_f1"]

    for mkey in ["arc_orig", "arc_rel", "equal"]:
        m_acc_a = summary[mkey]["round_5_final"]["bench_a_acc"]
        m_f1_a = summary[mkey]["round_5_final"]["bench_a_f1"]

        t_stat_acc, p_val_acc = stats.ttest_rel(m_acc_a, fedavg_acc_a)
        t_stat_f1, p_val_f1 = stats.ttest_rel(m_f1_a, fedavg_f1_a)

        # Cohen's d effect size
        diff_acc = np.array(m_acc_a) - np.array(fedavg_acc_a)
        d_acc = float(np.mean(diff_acc) / np.std(diff_acc, ddof=1)) if np.std(diff_acc, ddof=1) > 0 else 0.0

        summary[mkey]["paired_vs_fedavg"] = {
            "p_value_acc_bench_a": round(float(p_val_acc), 4),
            "p_value_f1_bench_a": round(float(p_val_f1), 4),
            "cohens_d_acc": round(d_acc, 4),
        }

    return summary


def generate_plots(all_runs: dict, stat_summary: dict, exp_dir: Path):
    """Generate round-over-round convergence curves and final comparison plots."""
    methods = ["fedavg", "equal", "arc_orig", "arc_rel"]
    method_labels = {
        "fedavg": "FedAvg",
        "equal": "Equal",
        "arc_orig": "ARC (Orig: 0.4/0.3/0.3)",
        "arc_rel": "ARC (Relevance: 0.6/0.2/0.2)",
    }
    colors = {"fedavg": "#1f77b4", "equal": "#7f7f7f", "arc_orig": "#2ca02c", "arc_rel": "#d62728"}

    # 1. Round-by-round Convergence Plot (Benchmark A)
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle("Experiment 004: 5-Round Convergence Trajectories (5-Seed Mean)", fontsize=14, fontweight="bold")

    rounds = np.arange(1, NUM_ROUNDS + 1)

    for m in methods:
        mean_acc_a = []
        mean_f1_a = []
        for r in range(1, NUM_ROUNDS + 1):
            accs = [all_runs[f"seed_{s}"][f"round_{r}"][m]["benchmark_a"]["accuracy"] for s in SEEDS]
            f1s = [all_runs[f"seed_{s}"][f"round_{r}"][m]["benchmark_a"]["macro_f1"] for s in SEEDS]
            mean_acc_a.append(np.mean(accs))
            mean_f1_a.append(np.mean(f1s))

        axes[0].plot(rounds, mean_acc_a, marker="o", label=method_labels[m], color=colors[m], linewidth=2)
        axes[1].plot(rounds, mean_f1_a, marker="s", label=method_labels[m], color=colors[m], linewidth=2)

    axes[0].set_title("Benchmark A Accuracy Convergence", fontweight="bold")
    axes[0].set_xlabel("Communication Round")
    axes[0].set_ylabel("Accuracy (%)")
    axes[0].grid(True, linestyle="--", alpha=0.5)
    axes[0].legend()

    axes[1].set_title("Benchmark A Macro F1 Convergence", fontweight="bold")
    axes[1].set_xlabel("Communication Round")
    axes[1].set_ylabel("Macro F1 (%)")
    axes[1].grid(True, linestyle="--", alpha=0.5)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(exp_dir / "convergence_trajectories.png", dpi=300)
    plt.close()

    # 2. Final Round 5 Comparison with 95% Confidence Intervals
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(methods))
    width = 0.35

    means_acc = [stat_summary[m]["round_5_final"]["bench_a_acc_mean"] for m in methods]
    stds_acc = [stat_summary[m]["round_5_final"]["bench_a_acc_std"] for m in methods]
    means_f1 = [stat_summary[m]["round_5_final"]["bench_a_f1_mean"] for m in methods]
    stds_f1 = [stat_summary[m]["round_5_final"]["bench_a_f1_std"] for m in methods]

    ax.bar(x - width/2, means_acc, width, yerr=stds_acc, capsize=5, label="Benchmark A Accuracy (%)", color="#1f77b4", alpha=0.9)
    ax.bar(x + width/2, means_f1, width, yerr=stds_f1, capsize=5, label="Benchmark A Macro F1 (%)", color="#ff7f0e", alpha=0.9)

    ax.set_title("Round 5 Final Evaluation Across 5 Random Seeds (Mean ± Std)", fontweight="bold", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels([method_labels[m] for m in methods], fontweight="bold")
    ax.set_ylim(70, 100)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="lower right")

    plt.tight_layout()
    plt.savefig(exp_dir / "final_statistical_comparison.png", dpi=300)
    plt.close()


if __name__ == "__main__":
    run_experiment_004()
