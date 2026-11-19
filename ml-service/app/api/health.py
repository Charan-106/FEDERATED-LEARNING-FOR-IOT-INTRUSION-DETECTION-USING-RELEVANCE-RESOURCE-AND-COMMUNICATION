"""
Health Check Router - GET /health

Returns service status, model load state, and PyTorch version.
This endpoint is exempt from API key authentication.
"""

import torch
from fastapi import APIRouter

from app.models.registry import registry

router = APIRouter()
_cached_model_params = None


@router.get("/health")
async def health():
    """Check if the ML service is running and the model is loaded."""
    global _cached_model_params
    if _cached_model_params is None and registry.is_loaded:
        try:
            _cached_model_params = sum(p.numel() for p in registry.get_model().parameters())
        except Exception:
            _cached_model_params = None

    return {
        "status": "ok",
        "service": "arc-fl-ml-service",
        "version": "1.0.0",
        "model_loaded": registry.is_loaded,
        "pytorch_version": torch.__version__,
        "model_params": _cached_model_params,
        "model_config": registry.get_config() if registry.is_loaded else None,
    }
