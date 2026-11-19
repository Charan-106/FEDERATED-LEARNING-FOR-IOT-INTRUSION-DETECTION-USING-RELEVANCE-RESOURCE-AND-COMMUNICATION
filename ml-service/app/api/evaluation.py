"""
Evaluation Router - /evaluation/

Endpoints:
  POST /evaluation/validate   → Evaluate a candidate model
  GET  /evaluation/baseline   → Evaluate the current active model
"""

import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.evaluation.validator import evaluate_model
from app.models.registry import registry

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/evaluation", tags=["Evaluation"])


class ValidateRequest(BaseModel):
    model_path: str
    max_samples: int = 5000
    benchmark_type: str = "primary_capped"


@router.post("/validate")
async def validate_model(request: ValidateRequest):
    """
    Evaluate a model checkpoint against the standardized research benchmark.
    Returns accuracy, macro precision/recall/F1, weighted F1, loss, and hashes.
    """
    logger.info(f"Validation requested for: {request.model_path} (benchmark: {request.benchmark_type})")
    
    try:
        result = evaluate_model(
            request.model_path,
            max_samples=request.max_samples,
            benchmark_type=request.benchmark_type,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Evaluation error: {str(e)}")

    return result


@router.get("/baseline")
async def evaluate_baseline():
    """
    Evaluate the currently active global model against the validation dataset.
    Used to establish the baseline metrics before a new round.
    """
    if not registry.is_loaded:
        raise HTTPException(status_code=503, detail="No model loaded in registry")

    # Get path from registry
    model_path = str(registry._model_path)
    
    try:
        result = evaluate_model(model_path, max_samples=5000, benchmark_type="primary_capped")
    except Exception as e:
        logger.error(f"Baseline evaluation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Evaluation error: {str(e)}")

    return {"baseline": True, "model_path": model_path, **result}
