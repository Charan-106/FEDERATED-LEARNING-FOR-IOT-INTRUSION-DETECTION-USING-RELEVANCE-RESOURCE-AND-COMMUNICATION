"""
Aggregation Router - /aggregation/

Endpoints:
  POST /aggregation/preview  → Preview weights without executing
  POST /aggregation/merge    → Execute federated aggregation
"""

import logging
from typing import List

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.aggregation.fedavg import aggregate, compute_weights

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/aggregation", tags=["Aggregation"])


class SubmissionInfo(BaseModel):
    submission_id: str
    client_id: str
    model_path: str
    samples_trained: int = 1
    arc_score: float = 50.0


class PreviewRequest(BaseModel):
    round_id: str
    strategy: str = "FEDAVG"
    submissions: List[SubmissionInfo]


class MergeRequest(BaseModel):
    job_id: str
    round_id: str
    strategy: str = "FEDAVG"
    base_model_path: str = None
    server_learning_rate: float = 0.5
    submissions: List[SubmissionInfo]


@router.post("/preview")
async def preview_weights(request: PreviewRequest):
    """
    Compute aggregation weights for a set of submissions without executing the merge.
    Returns the normalized weight each client would receive under the chosen strategy.
    """
    if len(request.submissions) < 1:
        raise HTTPException(status_code=400, detail="Need at least 1 submission to preview")

    subs = [s.model_dump() for s in request.submissions]
    weighted = compute_weights(subs, request.strategy)

    return {
        "round_id": request.round_id,
        "strategy": request.strategy,
        "submission_count": len(weighted),
        "weights": [
            {
                "submission_id": w["submission_id"],
                "client_id": w["client_id"],
                "raw_weight": w["raw_weight"],
                "normalized_weight": w["normalized_weight"],
                "samples_trained": w.get("samples_trained"),
                "arc_score": w.get("arc_score"),
            }
            for w in weighted
        ],
        "preview": True,
    }


@router.post("/merge")
async def merge_models(request: MergeRequest):
    """
    Execute federated model aggregation.
    
    Loads each client's .pt file, validates architecture compatibility,
    performs weighted parameter averaging, and saves the candidate model.
    """
    if len(request.submissions) < 1:
        raise HTTPException(status_code=400, detail="Need at least 1 submission to aggregate")

    logger.info(f"Starting aggregation: job={request.job_id}, strategy={request.strategy}, clients={len(request.submissions)}")

    # Compute normalized weights
    subs = [s.model_dump() for s in request.submissions]
    weighted_subs = compute_weights(subs, request.strategy)

    try:
        result = aggregate(
            weighted_subs,
            base_model_path=request.base_model_path,
            server_learning_rate=request.server_learning_rate,
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        logger.error(f"Aggregation failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Aggregation error: {str(e)}")

    return {
        "job_id": request.job_id,
        "round_id": request.round_id,
        "strategy": request.strategy,
        "server_learning_rate": request.server_learning_rate,
        **result,
    }
