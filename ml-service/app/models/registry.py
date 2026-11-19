"""
Model Registry - Singleton that holds the loaded global model, scaler, and label encoder.
Loaded once at startup and shared across all request handlers.

The active model is determined by scanning storage/global-models/ for a model_config.json
with isActive=true, or by falling back to the alphabetically-last version.
"""

import json
import logging
from pathlib import Path
from typing import Optional

import joblib
import torch

from app.config import config
from app.models.resmlp import ResMLP, load_checkpoint

logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    Singleton registry for the active federated learning model.
    
    Responsibilities:
    - Find and load the active model checkpoint
    - Load the associated scaler (StandardScaler) and label encoder
    - Provide thread-safe accessor methods for inference code
    """

    _instance: Optional["ModelRegistry"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._model: Optional[ResMLP] = None
        self._scaler = None
        self._label_encoder = None
        self._model_config: dict = {}
        self._model_path: Optional[Path] = None
        self._initialized = True

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def get_model(self) -> ResMLP:
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load_active_model() first.")
        return self._model

    def get_scaler(self):
        if self._scaler is None:
            raise RuntimeError("Scaler not loaded. Call load_active_model() first.")
        return self._scaler

    def get_label_encoder(self):
        if self._label_encoder is None:
            raise RuntimeError("Label encoder not loaded.")
        return self._label_encoder

    def get_config(self) -> dict:
        return self._model_config

    def load_active_model(self) -> dict:
        """
        Scan storage/global-models/ to find and load the active model.
        Returns metadata dict describing what was loaded.
        """
        global_models_dir = config.global_models_dir

        if not global_models_dir.exists():
            raise FileNotFoundError(f"Global models directory not found: {global_models_dir}")

        # Find all version subdirectories
        version_dirs = [d for d in global_models_dir.iterdir() if d.is_dir()]
        if not version_dirs:
            raise FileNotFoundError(f"No model versions found in {global_models_dir}")

        # Look for a directory with model_config.json marking isActive
        active_dir: Optional[Path] = None
        for vdir in version_dirs:
            config_file = vdir / "model_config.json"
            if config_file.exists():
                with open(config_file) as f:
                    cfg = json.load(f)
                if cfg.get("isActive", False):
                    active_dir = vdir
                    self._model_config = cfg
                    break

        # Fallback: use the most recently modified directory
        if active_dir is None:
            active_dir = sorted(version_dirs, key=lambda d: d.stat().st_mtime)[-1]
            logger.warning(f"No isActive model found; loading latest: {active_dir.name}")
            config_file = active_dir / "model_config.json"
            if config_file.exists():
                with open(config_file) as f:
                    self._model_config = json.load(f)

        checkpoint_path = active_dir / "best.pt"
        scaler_path = active_dir / "scaler.pkl"
        label_encoder_path = active_dir / "label_encoder.pkl"

        logger.info(f"Loading model from: {checkpoint_path}")
        self._model, checkpoint_meta = load_checkpoint(str(checkpoint_path))
        self._model_path = checkpoint_path

        # Load scaler - trained StandardScaler for the 45 CICIoT2023 features
        if scaler_path.exists():
            self._scaler = joblib.load(str(scaler_path))
            logger.info("Scaler loaded via joblib")
        else:
            logger.warning(f"Scaler not found at {scaler_path}")

        # Load label encoder - maps class indices to attack class names
        if label_encoder_path.exists():
            self._label_encoder = joblib.load(str(label_encoder_path))
            logger.info("Label encoder loaded via joblib")
        else:
            logger.warning(f"Label encoder not found at {label_encoder_path}")

        param_count = sum(p.numel() for p in self._model.parameters())
        logger.info(f"Model loaded: {param_count:,} parameters")

        return {
            "version": active_dir.name,
            "checkpoint": str(checkpoint_path),
            "parameters": param_count,
            "config": self._model_config,
        }


# Global singleton instance
registry = ModelRegistry()
