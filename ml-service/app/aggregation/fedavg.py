"""
Federated Averaging (FedAvg) Aggregation Engine

Implements three aggregation strategies:
1. FEDAVG: Weight by number of training samples (original McMahan et al. FedAvg)
2. EQUAL: Uniform weights across all clients
3. ARC_WEIGHTED: Weight by ARC scores (rewards higher-quality contributors)

Aggregation procedure:
- Load each client's .pt checkpoint
- Validate key/shape compatibility against reference architecture
- Reject updates with NaN/Inf parameters (poisoning defense)
- Perform weighted parameter averaging
- Special handling for BatchNorm statistics:
  - running_mean, running_var: weighted average
  - num_batches_tracked: sum across clients
- Save aggregated model to storage/candidates/
"""

import hashlib
import logging
import uuid
from pathlib import Path
from typing import Optional

import torch

from app.config import config
from app.models.resmlp import build_model, load_checkpoint

logger = logging.getLogger(__name__)


def compute_weights(submissions: list[dict], strategy: str) -> list[dict]:
    """
    Compute normalized aggregation weights for each submission.

    Args:
        submissions: List of dicts with keys:
            - submission_id, client_id, samples_trained, arc_score
        strategy: 'FEDAVG', 'EQUAL', or 'ARC_WEIGHTED'

    Returns:
        List of dicts with raw_weight and normalized_weight added.
    """
    if not submissions or len(submissions) < 1:
        raise ValueError("Cannot compute aggregation weights: submissions list is empty.")

    if strategy == "FEDAVG":
        # Weight proportional to number of local training samples
        raw = []
        for s in submissions:
            samples = s.get("samples_trained")
            if samples is None:
                raise ValueError(f"Missing samples_trained for client {s.get('client_id')} in FEDAVG strategy.")
            val = float(samples)
            if val <= 0:
                raise ValueError(f"Invalid samples_trained ({val}) for client {s.get('client_id')}. Must be > 0.")
            raw.append(val)
    elif strategy == "EQUAL":
        raw = [1.0 for _ in submissions]
    elif strategy == "ARC_WEIGHTED":
        # Weight proportional to ARC score (higher contributor quality = more influence)
        raw = []
        for s in submissions:
            arc = s.get("arc_score")
            if arc is None:
                raise ValueError(f"Missing arc_score for client {s.get('client_id')} in ARC_WEIGHTED strategy.")
            val = float(arc)
            if val < 0 or val > 100:
                raise ValueError(f"Invalid arc_score ({val}) for client {s.get('client_id')}. Must be between 0 and 100.")
            raw.append(val)
    else:
        raise ValueError(f"Unknown aggregation strategy: {strategy}")

    total = sum(raw)
    if total <= 0:
        raise ValueError(f"Total aggregation weight sum is {total} (must be > 0). Cannot normalize weights for strategy {strategy}. Silent fallback disabled.")

    result = []
    for sub, r in zip(submissions, raw):
        result.append({
            **sub,
            "raw_weight": r,
            "normalized_weight": r / total,
        })

    return result


def _load_state_dict(model_path: str) -> Optional[dict]:
    """Load state dict from a client .pt file. Returns None on error."""
    path = Path(config.STORAGE_ROOT) / model_path if not Path(model_path).is_absolute() else Path(model_path)
    
    if not path.exists():
        logger.error(f"Client model not found: {path}")
        return None

    try:
        ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict) and "model" in ckpt:
            return ckpt["model"]
        elif isinstance(ckpt, dict):
            return ckpt
        else:
            logger.error(f"Unrecognized checkpoint format: {path}")
            return None
    except Exception as e:
        logger.error(f"Failed to load checkpoint {path}: {e}")
        return None


def _validate_state_dict(state_dict: dict, reference_keys: set, reference_shapes: dict) -> tuple[bool, str]:
    """
    Validate a client state dict against the reference model architecture.
    
    Checks:
    1. All expected keys are present (no missing layers)
    2. All tensor shapes match the reference
    3. No NaN or Inf parameters (simple poisoning check)
    """
    client_keys = set(state_dict.keys())

    # Check for missing or unexpected keys
    missing = reference_keys - client_keys
    if missing:
        return False, f"Missing keys: {list(missing)[:5]}"

    # Check shapes and check for NaN/Inf
    for key in reference_keys:
        if key not in state_dict:
            continue
        client_tensor = state_dict[key]
        ref_shape = reference_shapes[key]

        if client_tensor.shape != ref_shape:
            return False, f"Shape mismatch for {key}: expected {ref_shape}, got {client_tensor.shape}"

        if torch.isnan(client_tensor).any() or torch.isinf(client_tensor).any():
            return False, f"NaN/Inf detected in parameter {key} - possible corrupted update"

    return True, "OK"


def aggregate(
    submissions_with_weights: list[dict],
    base_model_path: Optional[str] = None,
    server_learning_rate: float = 0.5,
) -> dict:
    """
    Perform federated parameter averaging over accepted client submissions.
    Supports conservative delta-scaling to prevent catastrophic forgetting on non-IID data.

    Args:
        submissions_with_weights: List of submission dicts with:
            - submission_id, client_id, model_path (relative to storage root)
            - normalized_weight, samples_trained
        base_model_path: Optional path to the baseline model from which clients trained.
        server_learning_rate: Global delta scaling rate eta_g in [0.05, 1.0]. Default: 0.5.

    Returns:
        Dict with candidate_id, candidate_path, checksum, and per-client metadata.
    """
    if len(submissions_with_weights) < 1:
        raise ValueError("Need at least 1 client update to aggregate")

    # Build reference model to get expected keys and shapes
    reference_model = build_model(in_dim=45, n_classes=34)
    ref_state = reference_model.state_dict()
    reference_keys = set(ref_state.keys())
    reference_shapes = {k: v.shape for k, v in ref_state.items()}

    # Resolve base model state for delta calculation
    base_state = None
    resolved_base_path = None
    if base_model_path:
        p = Path(base_model_path) if Path(base_model_path).is_absolute() else Path(config.STORAGE_ROOT) / base_model_path
        if p.exists():
            resolved_base_path = p
    
    if resolved_base_path is None:
        # Fallback to active model in storage or artifacts
        default_p = Path(config.STORAGE_ROOT) / "global-models" / "initial-v1" / "best.pt"
        if default_p.exists():
            resolved_base_path = default_p
        else:
            artifact_p = Path(__file__).parent.parent.parent / "artifacts" / "best.pt"
            if artifact_p.exists():
                resolved_base_path = artifact_p

    if resolved_base_path and resolved_base_path.exists():
        try:
            base_ckpt = torch.load(str(resolved_base_path), map_location="cpu", weights_only=False)
            base_state = base_ckpt["model"] if isinstance(base_ckpt, dict) and "model" in base_ckpt else base_ckpt
            logger.info(f"Loaded base model for delta aggregation: {resolved_base_path}")
        except Exception as e:
            logger.warning(f"Could not load base model for delta scaling: {e}")
            base_state = None

    # Collect valid client state dicts
    valid_clients = []
    for sub in submissions_with_weights:
        state_dict = _load_state_dict(sub["model_path"])
        if state_dict is None:
            logger.warning(f"Skipping {sub['client_id']}: could not load checkpoint")
            continue

        is_valid, msg = _validate_state_dict(state_dict, reference_keys, reference_shapes)
        if not is_valid:
            logger.warning(f"Skipping {sub['client_id']}: validation failed - {msg}")
            continue

        valid_clients.append({**sub, "state_dict": state_dict})
        logger.info(f"Client {sub['client_id']} validated OK (weight={sub['normalized_weight']:.4f})")

    if len(valid_clients) < 1:
        raise ValueError(f"Only {len(valid_clients)} valid client(s) after validation; need ≥ 1")

    # Renormalize weights over valid clients only
    total_weight = sum(c["normalized_weight"] for c in valid_clients)
    for c in valid_clients:
        c["effective_weight"] = c["normalized_weight"] / total_weight

    # Ensure eta_g is bounded
    eta_g = max(0.05, min(float(server_learning_rate), 1.0))

    # ================================================================
    # Weighted parameter averaging & Delta scaling
    # ================================================================
    aggregated_state = {}

    for key in reference_keys:
        is_batchnorm_tracked = key.endswith("num_batches_tracked")
        is_batchnorm_running = key.endswith("running_mean") or key.endswith("running_var")

        tensors = [c["state_dict"][key].float() for c in valid_clients]
        weights = [c["effective_weight"] for c in valid_clients]

        if is_batchnorm_tracked:
            # Phase 1 Requirement: Correct num_batches_tracked handling
            # Do NOT sum lifetime counters across clients.
            # Preserve base counter + maximum delta trained in this round.
            if base_state is not None and key in base_state:
                base_tracked = base_state[key].item()
                deltas = [max(0, int(t.item() - base_tracked)) for t in tensors]
                max_delta = max(deltas, default=0)
                aggregated_state[key] = torch.tensor(base_tracked + max_delta, dtype=torch.int64)
            else:
                aggregated_state[key] = torch.tensor(max(int(t.item()) for t in tensors), dtype=torch.int64)
        elif is_batchnorm_running:
            # Weighted average for BN running stats with conservative base interpolation
            client_avg = sum(w * t for w, t in zip(weights, tensors))
            if base_state is not None and key in base_state:
                aggregated_state[key] = (1.0 - eta_g) * base_state[key].float() + eta_g * client_avg
            else:
                aggregated_state[key] = client_avg
        else:
            # Trainable parameters: compute weighted client average
            client_avg = sum(w * t for w, t in zip(weights, tensors))
            if base_state is not None and key in base_state:
                # Delta aggregation: W_new = W_base + eta_g * (W_client_avg - W_base)
                delta = client_avg - base_state[key].float()
                aggregated_state[key] = base_state[key].float() + eta_g * delta
            else:
                aggregated_state[key] = client_avg

    # ================================================================
    # Load aggregated weights into model and save
    # ================================================================
    result_model = build_model(in_dim=45, n_classes=34)
    result_model.load_state_dict(aggregated_state, strict=True)
    result_model.eval()

    # Save to storage/candidates/ with a unique ID
    candidate_id = str(uuid.uuid4())
    candidates_dir = config.candidates_dir
    candidate_path = candidates_dir / f"{candidate_id}.pt"

    # Save in the same format as training checkpoints
    torch.save({"model": result_model.state_dict(), "candidate_id": candidate_id}, str(candidate_path))

    # Compute SHA256 checksum for integrity verification
    with open(candidate_path, "rb") as f:
        checksum = hashlib.sha256(f.read()).hexdigest()

    logger.info(f"Aggregated model saved: {candidate_path} (checksum: {checksum[:16]}...)")

    return {
        "candidate_id": candidate_id,
        "candidate_path": str(candidate_path),
        "checksum": checksum,
        "valid_client_count": len(valid_clients),
        "skipped_client_count": len(submissions_with_weights) - len(valid_clients),
        "client_weights": [
            {
                "client_id": c["client_id"],
                "submission_id": c["submission_id"],
                "effective_weight": c["effective_weight"],
            }
            for c in valid_clients
        ],
    }
