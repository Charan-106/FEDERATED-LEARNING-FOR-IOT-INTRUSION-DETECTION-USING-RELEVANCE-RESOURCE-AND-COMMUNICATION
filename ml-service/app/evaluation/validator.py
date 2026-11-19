"""
Server-Side Model Validation

Evaluates a model checkpoint against a held-out validation dataset.
Used to assess candidate models before publishing them as the new global model.

Metrics computed:
- Accuracy (overall)
- Macro-averaged Precision, Recall, F1-score
- Cross-entropy loss
- Confusion matrix (for detailed per-class analysis)
"""

import hashlib
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from app.config import config
from app.models.registry import registry
from app.models.resmlp import load_checkpoint

logger = logging.getLogger(__name__)


def sha256_file(filepath: Path) -> str:
    h = hashlib.sha256()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()


def evaluate_model(
    model_path: str,
    max_samples: int = 5000,
    benchmark_type: str = "primary_capped",
) -> dict:
    """
    Evaluate a model checkpoint against a standardized research benchmark.
    Conforms to RESEARCH_EVALUATION_PROTOCOL.md.

    Args:
        model_path: Absolute or relative path to the .pt checkpoint
        max_samples: Maximum number of validation samples to use
        benchmark_type: 'primary_capped', 'secondary_natural', or 'default'

    Returns:
        Dict with accuracy, macro precision/recall/F1, weighted F1, loss, and hashes.
    """
    start_time = time.time()

    # Resolve model path
    p = Path(model_path)
    if not p.is_absolute():
        p = Path(config.STORAGE_ROOT) / model_path

    if not p.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {p}")

    model_hash = sha256_file(p)
    logger.info(f"Evaluating model: {p} (SHA-256: {model_hash[:16]}...)")

    # Resolve benchmark dataset path based on protocol
    benchmarks_dir = Path(config.STORAGE_ROOT) / "benchmarks"
    primary_bench = benchmarks_dir / "benchmark_primary_capped.csv"
    secondary_bench = benchmarks_dir / "benchmark_secondary_natural.csv"

    if benchmark_type == "primary_capped" and primary_bench.exists():
        val_path = primary_bench
        active_benchmark_name = "primary_capped"
    elif benchmark_type == "secondary_natural" and secondary_bench.exists():
        val_path = secondary_bench
        active_benchmark_name = "secondary_natural"
    elif primary_bench.exists():
        val_path = primary_bench
        active_benchmark_name = "primary_capped"
    else:
        val_path = Path(config.VALIDATION_DATASET_PATH)
        active_benchmark_name = "raw_validation"

    if not val_path.exists():
        logger.warning(f"Validation dataset not found: {val_path}")
        return {
            "error": "Validation dataset not available",
            "path": str(val_path),
            "note": "Set VALIDATION_DATASET_PATH or generate research benchmarks",
        }

    dataset_hash = sha256_file(val_path)

    # ================================================================
    # Load dataset
    # ================================================================
    df = pd.read_csv(str(val_path))
    if len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42)

    # Ensure registry or artifacts are available
    if not registry.is_loaded:
        try:
            registry.load_active_model()
        except Exception as e:
            logger.warning(f"Could not load active model from registry: {e}")

    scaler = registry.get_scaler() if registry.is_loaded else None
    label_encoder = registry.get_label_encoder() if registry.is_loaded else None
    cfg = registry.get_config() if registry.is_loaded else None

    # Fallback to ml-service/artifacts if registry not loaded
    artifacts_dir = Path(__file__).resolve().parents[2] / "artifacts"
    if scaler is None and (artifacts_dir / "scaler.pkl").exists():
        import joblib
        scaler = joblib.load(artifacts_dir / "scaler.pkl")
    if label_encoder is None and (artifacts_dir / "label_encoder.pkl").exists():
        import joblib
        label_encoder = joblib.load(artifacts_dir / "label_encoder.pkl")
    if cfg is None and (artifacts_dir / "model_config.json").exists():
        import json
        with open(artifacts_dir / "model_config.json", "r", encoding="utf-8") as f:
            cfg = json.load(f)

    # Extract 45 features matching model_config.json feature order
    if cfg and 'features' in cfg and all(col in df.columns for col in cfg['features']):
        X = df[cfg['features']].values.astype(np.float32)
    else:
        feature_cols = [c for c in df.columns if c not in ('label', 'Telnet')]
        if len(feature_cols) > 45:
            feature_cols = feature_cols[:45]
        X = df[feature_cols].values.astype(np.float32)

    if 'label' in df.columns:
        y_raw = df['label'].values
    else:
        y_raw = df.iloc[:, -1].values

    if scaler is not None:
        X = scaler.transform(X)

    if label_encoder is not None:
        y = label_encoder.transform(y_raw).astype(np.int64)
    else:
        y = y_raw.astype(np.int64)

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    # ================================================================
    # Load and evaluate model
    # ================================================================
    model, _ = load_checkpoint(str(p))
    model.eval()

    criterion = nn.CrossEntropyLoss()

    all_preds = []
    all_labels = []
    total_loss = 0.0

    # Process in batches to avoid OOM
    batch_size = 512
    with torch.no_grad():
        for i in range(0, len(X_tensor), batch_size):
            X_batch = X_tensor[i:i + batch_size]
            y_batch = y_tensor[i:i + batch_size]

            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            preds = logits.argmax(dim=1)

            all_preds.extend(preds.numpy().tolist())
            all_labels.extend(y_batch.numpy().tolist())
            total_loss += loss.item() * len(y_batch)

    # ================================================================
    # Compute metrics
    # ================================================================
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    n = len(all_labels)

    accuracy = accuracy_score(all_labels, all_preds) * 100
    macro_precision = precision_score(all_labels, all_preds, average="macro", zero_division=0) * 100
    macro_recall = recall_score(all_labels, all_preds, average="macro", zero_division=0) * 100
    macro_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0) * 100
    weighted_f1 = f1_score(all_labels, all_preds, average="weighted", zero_division=0) * 100
    avg_loss = total_loss / n

    # Compute confusion matrix (convert to list for JSON serialization)
    cm = confusion_matrix(all_labels, all_preds)
    cm_list = cm.tolist()

    duration = time.time() - start_time
    logger.info(f"Evaluation complete on {active_benchmark_name}: acc={accuracy:.2f}%, F1={macro_f1:.2f}%, duration={duration:.1f}s")

    return {
        "accuracy": round(accuracy, 4),
        "macro_precision": round(macro_precision, 4),
        "macro_recall": round(macro_recall, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "loss": round(avg_loss, 6),
        "sample_count": n,
        "confusion_matrix": cm_list,
        "evaluation_duration_seconds": round(duration, 2),
        "dataset": str(val_path),
        "benchmark_name": active_benchmark_name,
        "dataset_sha256": dataset_hash,
        "model_sha256": model_hash,
    }
