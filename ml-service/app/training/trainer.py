"""
Local Edge Training Simulator

Simulates local PyTorch training on a client's dataset partition.
In production, clients train on their own device; this module supports
server-side simulation for testing and benchmarking the FL pipeline.

Training procedure:
1. Load dataset CSV partition
2. Apply StandardScaler transform
3. Load global model checkpoint
4. Train for specified epochs with Adam optimizer + CrossEntropy loss
5. Return metrics and optionally save the updated state dict
"""

import logging
import time
import uuid
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import psutil
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from app.config import config
from app.models.registry import registry
from app.models.resmlp import load_checkpoint

logger = logging.getLogger(__name__)


def train_local(
    dataset_path: str,
    global_model_path: str,
    epochs: int = 2,
    batch_size: int = 256,
    learning_rate: float = 1e-4,
    max_samples: Optional[int] = None,
    mu: float = 0.0,
    freeze_bn_stats: bool = True,
    seed: int = 42,
    progress_callback=None,
) -> dict:
    """
    Perform local training on a dataset partition.

    Args:
        dataset_path: Path to the client's CSV dataset
        global_model_path: Path to the global model checkpoint to start from
        epochs: Number of local training epochs
        batch_size: Mini-batch size
        learning_rate: Adam learning rate
        max_samples: Cap on samples to use (for memory safety)
        mu: FedProx proximal regularization coefficient (0.0 = standard local training)
        freeze_bn_stats: Whether to freeze BatchNorm running stats in eval mode (prevents corruption)
        seed: Deterministic random seed for DataLoader shuffling and batch order

    Returns:
        Dict with training metrics and path to the saved update file.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    start_time = time.time()
    logger.info(f"Starting local training: {dataset_path} (freeze_bn_stats={freeze_bn_stats}, mu={mu}, seed={seed})")

    # ================================================================
    # Load and preprocess dataset
    # ================================================================
    df = pd.read_csv(dataset_path)

    # ----------------------------------------------------------------
    # Compute genuine Quality Metrics (Q_i) directly on local dataset
    # ----------------------------------------------------------------
    total_cells = int(df.shape[0] * df.shape[1])
    missing_cells = int(df.isna().sum().sum())
    missing_ratio = float(missing_cells / total_cells) if total_cells > 0 else 0.0

    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0 and len(df) > 0:
        inf_cells = int(np.isinf(df[numeric_cols].values).sum())
        inf_ratio = float(inf_cells / (len(df) * len(numeric_cols)))
    else:
        inf_cells = 0
        inf_ratio = 0.0

    # ----------------------------------------------------------------
    # Compute genuine Diversity Metrics (D_i) directly on local dataset
    # ----------------------------------------------------------------
    label_col = 'label' if 'label' in df.columns else df.columns[-1]
    class_counts = df[label_col].value_counts()
    class_count = int(len(class_counts))
    probs = (class_counts / class_counts.sum()).values
    entropy = float(-np.sum(probs * np.log2(probs + 1e-12))) if len(probs) > 0 else 0.0
    minority_coverage = float(min(1.0, class_count / 34.0))

    quality_metrics = {
        "missing_ratio": round(missing_ratio, 6),
        "inf_ratio": round(inf_ratio, 6),
        "preprocessing_success": 1.0,
        "missing_cells": missing_cells,
        "total_cells": total_cells,
    }

    diversity_metrics = {
        "class_count": class_count,
        "entropy": round(entropy, 4),
        "minority_class_coverage": round(minority_coverage, 4),
    }

    if max_samples is not None and max_samples > 0 and len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=seed)
        logger.info(f"Sampled {max_samples} rows from {len(df)} total for training")
    else:
        logger.info(f"Utilizing full private dataset partition: {len(df)} rows")

    # Extract 45 features matching model_config.json feature order
    cfg = registry.get_config() if registry.is_loaded else None
    if cfg and 'features' in cfg and all(col in df.columns for col in cfg['features']):
        X = df[cfg['features']].values.astype(np.float32)
    else:
        # Fallback: exclude 'label' and 'Telnet' if present
        feature_cols = [c for c in df.columns if c not in ('label', 'Telnet')]
        if len(feature_cols) > 45:
            feature_cols = feature_cols[:45]
        X = df[feature_cols].values.astype(np.float32)

    # Impute NaNs and Infs for numerical stability during gradient steps
    X = np.nan_to_num(X, nan=0.0, posinf=1e6, neginf=-1e6)

    if 'label' in df.columns:
        y_raw = df['label'].values
    else:
        y_raw = df.iloc[:, -1].values

    # Apply scaler if available
    scaler = registry.get_scaler() if registry.is_loaded else None
    label_encoder = registry.get_label_encoder() if registry.is_loaded else None

    # Fallback to artifacts if registry not loaded in standalone mode
    artifacts_dir = Path(__file__).resolve().parents[2] / "artifacts"
    if scaler is None and (artifacts_dir / "scaler.pkl").exists():
        import joblib
        scaler = joblib.load(artifacts_dir / "scaler.pkl")
    if label_encoder is None and (artifacts_dir / "label_encoder.pkl").exists():
        import joblib
        label_encoder = joblib.load(artifacts_dir / "label_encoder.pkl")

    if scaler is not None:
        X = scaler.transform(X)

    # ----------------------------------------------------------------
    # Compute Feature Noise Score (z-score domain violation in normalized space)
    # ----------------------------------------------------------------
    extreme_outliers = np.abs(X) > 6.0
    outlier_ratio = float(np.sum(extreme_outliers) / X.size) if X.size > 0 else 0.0
    feature_noise_score = round(max(0.0, 1.0 - outlier_ratio * 5.0) * 100.0, 2)
    quality_metrics["outlier_ratio"] = round(outlier_ratio, 6)
    quality_metrics["feature_noise_score"] = feature_noise_score

    # Parse labels: supports both pre-encoded numeric integers (0..33) and categorical strings
    is_numeric = False
    try:
        y_as_num = pd.to_numeric(y_raw, errors='coerce')
        if not np.isnan(y_as_num).any() and np.all(y_as_num >= 0) and np.all(y_as_num < 34):
            y = y_as_num.astype(np.int64)
            is_numeric = True
    except Exception:
        is_numeric = False

    if not is_numeric:
        if label_encoder is not None:
            y_str = [str(val) for val in y_raw]
            known_classes = set(label_encoder.classes_)
            y_clean = [s if s in known_classes else label_encoder.classes_[0] for s in y_str]
            y = label_encoder.transform(y_clean).astype(np.int64)
        else:
            y = pd.to_numeric(y_raw, errors='coerce').fillna(0).astype(np.int64).values

    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.long)

    # ================================================================
    # Load global model as starting point (single load)
    # ================================================================
    model, _ = load_checkpoint(global_model_path)

    # ----------------------------------------------------------------
    # Compute Reference Model Label Agreement Score (out-of-fold reference inference heuristic)
    # Evaluated on a representative partition slice (up to 256 samples) for sub-second startup
    # ----------------------------------------------------------------
    try:
        with torch.no_grad():
            model.eval()
            eval_slice_size = min(len(y), 256)
            ref_logits = model(X_tensor[:eval_slice_size])
            ref_probs = torch.softmax(ref_logits, dim=1).cpu().numpy()
            true_label_probs = ref_probs[np.arange(eval_slice_size), y[:eval_slice_size]]
            disagreement_count = int(np.sum(true_label_probs < 0.10))
            disagreement_ratio = float(disagreement_count / eval_slice_size) if eval_slice_size > 0 else 0.0
            label_agreement_score = round(max(0.0, 1.0 - disagreement_ratio) * 100.0, 2)
    except Exception as e:
        logger.warning(f"Reference model label agreement check fallback: {e}")
        disagreement_ratio = 0.0
        label_agreement_score = None

    quality_metrics["reference_disagreement_ratio"] = round(disagreement_ratio, 6)
    quality_metrics["label_agreement_score"] = label_agreement_score
    # Backward compatibility alias
    quality_metrics["suspicious_label_ratio"] = round(disagreement_ratio, 6)
    quality_metrics["label_consistency_score"] = label_agreement_score

    if progress_callback:
        progress_callback({
            "event": "DATASET_VALIDATED",
            "stage": "VALIDATING_DATASET",
            "samples": len(df),
            "features": X.shape[1] if hasattr(X, "shape") else 45,
            "progress": 25,
            "message": f"Loaded and validated {len(df)} samples across {X.shape[1] if hasattr(X, 'shape') else 45} input features."
        })

    dataset = TensorDataset(X_tensor, y_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    model.train()

    # Phase 1 Requirement: Freeze BatchNorm running statistics in eval mode
    # Do NOT set track_running_stats=False. Putting BN modules in eval mode
    # preserves pre-trained running_mean and running_var while keeping
    # affine weights and biases trainable for edge adaptation.
    if freeze_bn_stats:
        for m in model.modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                m.eval()

    # Clone base parameters if FedProx proximal regularization is enabled
    base_params = [p.detach().clone() for p in model.parameters()] if mu > 0.0 else []

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    if progress_callback:
        progress_callback({
            "event": "TRAINING_STARTED",
            "stage": "TRAINING_EPOCH_1",
            "epochs": epochs,
            "total_batches": len(loader),
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "progress": 30,
            "message": f"Global checkpoint loaded. Starting PyTorch training ({epochs} epochs, {len(loader)} batches/epoch)."
        })

    # ================================================================
    # Training loop
    # ================================================================
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    epoch_history = []

    for epoch in range(epochs):
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_samples = 0
        running_loss = 0.0
        epoch_start_time = time.time()

        for batch_idx, (X_batch, y_batch) in enumerate(loader):
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)

            # Phase 2 Requirement: FedProx proximal penalty
            if mu > 0.0:
                prox_term = sum(((p - p_base) ** 2).sum() for p, p_base in zip(model.parameters(), base_params))
                loss = loss + (mu / 2.0) * prox_term

            loss.backward()
            optimizer.step()

            preds = logits.argmax(dim=1)
            epoch_correct += (preds == y_batch).sum().item()
            epoch_samples += len(y_batch)
            loss_val = float(loss.item())
            epoch_loss += loss_val * len(y_batch)
            running_loss += loss_val
            running_avg_loss = running_loss / (batch_idx + 1)

            if progress_callback:
                overall_batch_idx = epoch * len(loader) + (batch_idx + 1)
                total_all_batches = max(1, epochs * len(loader))
                progress_pct = int(30 + (overall_batch_idx / total_all_batches) * 50)
                elapsed_so_far = round(time.time() - start_time, 2)
                progress_callback({
                    "event": "BATCH_PROGRESS",
                    "stage": f"TRAINING_EPOCH_{epoch + 1}",
                    "epoch": epoch + 1,
                    "total_epochs": epochs,
                    "batch": batch_idx + 1,
                    "total_batches": len(loader),
                    "batch_loss": round(loss_val, 6),
                    "running_loss": round(running_avg_loss, 6),
                    "progress": progress_pct,
                    "elapsed_seconds": elapsed_so_far,
                    "is_batch_event": True,
                    "message": f"Epoch {epoch+1}/{epochs} [Batch {batch_idx+1}/{len(loader)}] - Batch Loss: {loss_val:.4f} (Avg: {running_avg_loss:.4f})"
                })

        avg_epoch_loss = epoch_loss / epoch_samples
        epoch_accuracy = epoch_correct / epoch_samples * 100
        epoch_duration = time.time() - epoch_start_time
        logger.info(f"Epoch {epoch+1}/{epochs}: loss={avg_epoch_loss:.4f}, acc={epoch_accuracy:.2f}% (duration: {epoch_duration:.2f}s)")

        if progress_callback:
            progress_callback({
                "event": "EPOCH_COMPLETED",
                "stage": f"TRAINING_EPOCH_{epoch + 1}",
                "epoch": epoch + 1,
                "total_epochs": epochs,
                "epoch_loss": round(avg_epoch_loss, 6),
                "epoch_accuracy": round(epoch_accuracy, 4),
                "duration_seconds": round(epoch_duration, 3),
                "progress": 30 + int(((epoch + 1) / epochs) * 50),
                "message": f"Epoch {epoch+1}/{epochs} completed: Loss {avg_epoch_loss:.4f}, Accuracy {epoch_accuracy:.2f}%"
            })

        epoch_history.append({
            "epoch": epoch + 1,
            "loss": round(avg_epoch_loss, 6),
            "accuracy": round(epoch_accuracy, 4),
            "samples": epoch_samples,
            "batches": len(loader),
            "duration_seconds": round(epoch_duration, 3),
        })

        total_loss += epoch_loss
        total_correct += epoch_correct
        total_samples += epoch_samples

    final_accuracy = total_correct / total_samples * 100
    final_loss = total_loss / total_samples
    training_duration = time.time() - start_time

    # Measure genuine resource usage (safe against missing sensors)
    try:
        mem = psutil.virtual_memory()
        available_ram = round(mem.available / 1e9, 2)
        total_ram = round(mem.total / 1e9, 2)
    except Exception:
        available_ram = None
        total_ram = None

    try:
        cpu_pct = psutil.cpu_percent(interval=0.1)
        cpu_idle = round(100.0 - cpu_pct, 2) if cpu_pct is not None else None
    except Exception:
        cpu_idle = None

    throughput = round(total_samples / training_duration, 2) if training_duration > 0 else 0.0

    # Compute parameter update norm ||Delta theta||_2
    try:
        ref_ckpt = torch.load(global_model_path, map_location="cpu", weights_only=False)
        ref_state = ref_ckpt["model"]
        diff_sq = sum(
            torch.sum((p.cpu() - ref_state[k].cpu()).float() ** 2).item()
            for k, p in model.state_dict().items()
            if "weight" in k or "bias" in k
        )
        update_norm = round(float(np.sqrt(diff_sq)), 4)
    except Exception as e:
        logger.warning(f"Update norm calculation fallback: {e}")
        update_norm = 0.0

    # ================================================================
    # Save updated model state dict
    # ================================================================
    if progress_callback:
        progress_callback({
            "event": "GENERATING_UPDATE",
            "stage": "GENERATING_MODEL_UPDATE",
            "progress": 85,
            "message": "Computing parameter update norm and saving checkpoint..."
        })

    update_id = str(uuid.uuid4())
    update_path = Path(config.client_updates_dir) / f"{update_id}.pt"
    update_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save({"model": model.state_dict(), "update_id": update_id}, str(update_path))
    logger.info(f"Model update saved: {update_path} (update_norm={update_norm})")

    if progress_callback:
        progress_callback({
            "event": "TRAINING_COMPLETE",
            "stage": "UPDATE_SAVED",
            "progress": 90,
            "update_id": update_id,
            "accuracy": round(final_accuracy, 4),
            "loss": round(final_loss, 6),
            "training_duration_seconds": round(training_duration, 2),
            "message": f"Edge training successfully finished. Update {update_id} saved."
        })

    return {
        "update_id": update_id,
        "update_path": str(update_path),
        "accuracy": round(final_accuracy, 4),
        "loss": round(final_loss, 6),
        "samples_trained": len(df),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "mu": mu,
        "freeze_bn_stats": freeze_bn_stats,
        "batchnorm_handling": "Frozen eval mode (running statistics preserved, affine weights trainable)",
        "framework": f"PyTorch {torch.__version__}",
        "device": "CPU" if not torch.cuda.is_available() else "CUDA",
        "update_norm": update_norm,
        "training_duration_seconds": round(training_duration, 2),
        "epoch_history": epoch_history,
        "resource_metrics": {
            "available_ram_gb": available_ram,
            "total_ram_gb": total_ram,
            "cpu_idle_pct": cpu_idle,
            "throughput_samples_per_sec": throughput,
        },
        "quality_metrics": quality_metrics,
        "diversity_metrics": diversity_metrics,
    }
