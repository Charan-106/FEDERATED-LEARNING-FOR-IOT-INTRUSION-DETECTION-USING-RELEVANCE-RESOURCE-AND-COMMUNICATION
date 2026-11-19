"""
ARC-FL Unified Research Evaluation Tool
Evaluates any PyTorch model checkpoint (.pt) against the standardized benchmarks
specified in RESEARCH_EVALUATION_PROTOCOL.md.
"""

import argparse
import hashlib
import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

# Setup workspace paths
WORKSPACE_ROOT = Path(__file__).resolve().parent.parent
ML_SERVICE_DIR = WORKSPACE_ROOT / "ml-service"
sys.path.insert(0, str(ML_SERVICE_DIR))

from app.models.resmlp import load_checkpoint

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("evaluate_benchmark")


def sha256_file(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def evaluate_on_dataset(
    model: nn.Module,
    df: pd.DataFrame,
    cfg: dict,
    scaler,
    label_encoder,
    device: torch.device,
    batch_size: int = 1024,
) -> dict:
    """Run batch evaluation and return full protocol metrics."""
    feature_cols = cfg.get("features", [])
    if not feature_cols or not all(c in df.columns for c in feature_cols):
        # Fallback to non-label, non-Telnet columns
        feature_cols = [c for c in df.columns if c not in ("label", "Telnet")][:45]

    X_raw = df[feature_cols].values.astype(np.float32)
    y_raw = df["label"].values if "label" in df.columns else df.iloc[:, -1].values

    # Preprocessing
    if scaler is not None:
        X_scaled = scaler.transform(X_raw)
    else:
        X_scaled = X_raw

    if label_encoder is not None:
        y_encoded = label_encoder.transform(y_raw).astype(np.int64)
    else:
        y_encoded = y_raw.astype(np.int64)

    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y_encoded, dtype=torch.long)

    criterion = nn.CrossEntropyLoss()
    all_preds = []
    all_labels = []
    total_loss = 0.0

    model.eval()
    with torch.no_grad():
        for i in range(0, len(X_tensor), batch_size):
            xb = X_tensor[i : i + batch_size].to(device)
            yb = y_tensor[i : i + batch_size].to(device)

            logits = model(xb)
            loss = criterion(logits, yb)
            preds = logits.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy().tolist())
            all_labels.extend(yb.cpu().numpy().tolist())
            total_loss += loss.item() * len(yb)

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    n = len(all_labels)

    acc = float(accuracy_score(all_labels, all_preds) * 100)
    macro_p = float(precision_score(all_labels, all_preds, average="macro", zero_division=0) * 100)
    macro_r = float(recall_score(all_labels, all_preds, average="macro", zero_division=0) * 100)
    macro_f1 = float(f1_score(all_labels, all_preds, average="macro", zero_division=0) * 100)
    weighted_f1 = float(f1_score(all_labels, all_preds, average="weighted", zero_division=0) * 100)
    avg_loss = float(total_loss / n)

    # Per-class metrics
    class_names = (
        label_encoder.classes_.tolist()
        if label_encoder is not None
        else [str(i) for i in range(cfg.get("n_classes", 34))]
    )
    p_per = precision_score(all_labels, all_preds, average=None, zero_division=0)
    r_per = recall_score(all_labels, all_preds, average=None, zero_division=0)
    f_per = f1_score(all_labels, all_preds, average=None, zero_division=0)
    s_per = np.bincount(all_labels, minlength=len(class_names))

    per_class = {}
    for idx, cname in enumerate(class_names):
        per_class[cname] = {
            "precision": round(p_per[idx] * 100, 2),
            "recall": round(r_per[idx] * 100, 2),
            "f1": round(f_per[idx] * 100, 2),
            "support": int(s_per[idx]),
        }

    return {
        "accuracy": round(acc, 4),
        "macro_precision": round(macro_p, 4),
        "macro_recall": round(macro_r, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "loss": round(avg_loss, 6),
        "sample_count": n,
        "per_class": per_class,
    }


def evaluate_benchmark(
    model_path: str,
    benchmark_type: str = "both",
    save_result: bool = True,
) -> dict:
    start_time = time.time()
    mp = Path(model_path)
    if not mp.is_absolute():
        mp = WORKSPACE_ROOT / model_path

    if not mp.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {mp}")

    model_hash = sha256_file(mp)
    logger.info(f"Loading checkpoint: {mp} (SHA-256: {model_hash[:16]}...)")

    artifacts_dir = ML_SERVICE_DIR / "artifacts"
    cfg_path = artifacts_dir / "model_config.json"
    scaler_path = artifacts_dir / "scaler.pkl"
    encoder_path = artifacts_dir / "label_encoder.pkl"

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    scaler = joblib.load(scaler_path)
    label_encoder = joblib.load(encoder_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_checkpoint(str(mp))
    model.to(device)
    model.eval()

    benchmarks_dir = WORKSPACE_ROOT / "storage" / "benchmarks"
    primary_file = benchmarks_dir / "benchmark_primary_capped.csv"
    secondary_file = benchmarks_dir / "benchmark_secondary_natural.csv"

    eval_targets = []
    if benchmark_type in ("primary", "both"):
        eval_targets.append(("primary_capped", primary_file))
    if benchmark_type in ("secondary", "both"):
        eval_targets.append(("secondary_natural", secondary_file))

    results = {}
    for b_name, b_path in eval_targets:
        if not b_path.exists():
            logger.error(f"Benchmark file not found: {b_path}. Run generate_benchmarks.py first.")
            continue

        b_hash = sha256_file(b_path)
        logger.info(f"Evaluating on {b_name} ({b_path.name})...")
        df_bench = pd.read_csv(b_path)

        t0 = time.time()
        metrics = evaluate_on_dataset(model, df_bench, cfg, scaler, label_encoder, device)
        dur = round(time.time() - t0, 2)
        metrics["duration_seconds"] = dur

        results[b_name] = {
            "benchmark_name": b_name,
            "benchmark_path": str(b_path),
            "benchmark_sha256": b_hash,
            "sample_count": len(df_bench),
            "metrics": metrics,
        }

    total_duration = round(time.time() - start_time, 2)

    payload = {
        "protocol_version": "1.0",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": {
            "path": str(mp),
            "name": mp.parent.name if mp.parent.name != "artifacts" else mp.name,
            "sha256": model_hash,
        },
        "evaluation_duration_total_seconds": total_duration,
        "benchmarks": results,
    }

    if save_result:
        results_dir = benchmarks_dir / "results"
        results_dir.mkdir(parents=True, exist_ok=True)
        model_tag = mp.parent.name if mp.parent.name != "artifacts" else "artifacts"
        ts_tag = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        res_file = results_dir / f"{model_tag}_{benchmark_type}_{ts_tag}.json"
        with open(res_file, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        logger.info(f"Saved audit result to {res_file}")

    # Print summary
    print("\n" + "=" * 80)
    print(f" ARC-FL RESEARCH BENCHMARK EVALUATION RESULTS")
    print(f" Model: {mp.name} in {mp.parent.name}")
    print(f" SHA-256: {model_hash}")
    print("=" * 80)
    for b_name, res in results.items():
        m = res["metrics"]
        print(f"\n--- {b_name.upper()} (N={res['sample_count']:,}) ---")
        print(f"  Accuracy:         {m['accuracy']:.4f}%")
        print(f"  Macro F1-Score:   {m['macro_f1']:.4f}%")
        print(f"  Macro Precision:  {m['macro_precision']:.4f}%")
        print(f"  Macro Recall:     {m['macro_recall']:.4f}%")
        print(f"  Weighted F1:      {m['weighted_f1']:.4f}%")
        print(f"  Loss:             {m['loss']:.6f}")
        print(f"  Evaluation Time:  {m['duration_seconds']}s")
    print("\n" + "=" * 80)

    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ARC-FL Benchmark Evaluator")
    parser.add_argument(
        "--model",
        type=str,
        default="storage/global-models/initial-v1/best.pt",
        help="Path to .pt model checkpoint",
    )
    parser.add_argument(
        "--benchmark",
        type=str,
        default="both",
        choices=["primary", "secondary", "both"],
        help="Which benchmark to evaluate against",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save result JSON",
    )
    args = parser.parse_args()

    evaluate_benchmark(
        model_path=args.model,
        benchmark_type=args.benchmark,
        save_result=not args.no_save,
    )
