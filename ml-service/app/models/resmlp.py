"""
ResMLP Architecture - Exact copy from storage/global-models/initial-v1/model.py
Extends with checkpoint loading utilities for the ML service.

Architecture: preact_resmlp_v3
- Input: 45 network traffic features (CICIoT2023 dataset)
- Output: 34 IoT attack classes
- Parameters: ~4.98M
"""

import torch
import torch.nn as nn
from pathlib import Path


class PreActResBlock(nn.Module):
    """Pre-activation residual block with SiLU activation and dropout."""

    def __init__(self, width, dropout=0.2):
        super().__init__()
        self.bn1 = nn.BatchNorm1d(width)
        self.fc1 = nn.Linear(width, width)
        self.bn2 = nn.BatchNorm1d(width)
        self.fc2 = nn.Linear(width, width)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(p=dropout)

        # Zero-initialize last BN weight for stable training (ReZero-style)
        nn.init.zeros_(self.bn2.weight)

    def forward(self, x):
        out = self.fc1(self.act(self.bn1(x)))
        out = self.fc2(self.dropout(self.act(self.bn2(out))))
        return x + out


class ResMLP(nn.Module):
    """
    Residual MLP classifier for tabular IoT network traffic data.
    Uses pre-activation blocks with skip connections for stable deep learning.
    """

    def __init__(self, in_dim, n_classes, hidden=None, dropout=0.2):
        super().__init__()
        hidden = hidden or [768, 768, 768, 768]
        width = hidden[0]

        self.in_bn = nn.BatchNorm1d(in_dim)

        self.stem = nn.Sequential(
            nn.Linear(in_dim, width),
            nn.BatchNorm1d(width),
            nn.SiLU(),
            nn.Dropout(p=dropout)
        )

        self.blocks = nn.Sequential(
            *[PreActResBlock(width, dropout) for _ in hidden]
        )

        self.out_bn = nn.BatchNorm1d(width)

        self.head = nn.Sequential(
            nn.SiLU(),
            nn.Linear(width, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            nn.Linear(256, n_classes)
        )

    def forward(self, x):
        x = self.stem(self.in_bn(x))
        x = self.blocks(x)
        return self.head(self.out_bn(x))


def build_model(in_dim: int = 45, n_classes: int = 34, hidden=None, dropout: float = 0.2) -> ResMLP:
    """Factory function to construct the ResMLP model with standard config."""
    return ResMLP(in_dim=in_dim, n_classes=n_classes, hidden=hidden, dropout=dropout)


def load_checkpoint(path: str) -> tuple[ResMLP, dict]:
    """
    Load a ResMLP checkpoint from disk.

    Handles two checkpoint formats:
    1. Raw state_dict (direct torch.save of model.state_dict())
    2. Dict with 'model' key: {'model': state_dict, ...metadata...}

    Args:
        path: Absolute path to the .pt checkpoint file

    Returns:
        Tuple of (model instance with weights loaded, metadata dict)
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    # Load checkpoint - weights_only=False needed for full checkpoint dicts
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)

    metadata = {}

    # Detect format: dict with 'model' key vs raw state_dict
    if isinstance(ckpt, dict) and "model" in ckpt:
        state_dict = ckpt["model"]
        # Extract any available metadata
        metadata = {k: v for k, v in ckpt.items() if k != "model"}
    elif isinstance(ckpt, dict):
        # Could be a raw state_dict (keys are layer names)
        state_dict = ckpt
    else:
        raise ValueError(f"Unrecognized checkpoint format in {path}")

    # Instantiate model with standard CICIoT2023 config
    model = build_model(in_dim=45, n_classes=34)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    return model, metadata
