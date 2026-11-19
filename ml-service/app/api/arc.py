"""
ARC Scoring Router - /arc/

Endpoints:
  POST /arc/calculate  → Compute ARC score from client telemetry
  GET  /arc/config     → Return current ARC formula configuration
"""

import logging
from typing import Any, Optional

from fastapi import APIRouter
from pydantic import BaseModel

from app.arc.scoring import ARCScoringEngine

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/arc", tags=["ARC Scoring"])


class QualityMetrics(BaseModel):
    model_config = {"extra": "allow"}
    quality_score: Optional[float] = None
    feature_noise_score: Optional[float] = None
    label_agreement_score: Optional[float] = None
    label_consistency_score: Optional[float] = None
    missing_ratio: Optional[float] = 0.0
    inf_ratio: Optional[float] = 0.0
    preprocessing_success: Optional[float] = 1.0
    missing_cells: Optional[int] = 0
    total_cells: Optional[int] = 0


class DiversityMetrics(BaseModel):
    diversity_score: Optional[float] = None
    class_count: Optional[int] = 10
    entropy: Optional[float] = 2.5
    minority_class_coverage: Optional[float] = 0.5


class ResourceMetrics(BaseModel):
    available_ram_gb: Optional[float] = None
    total_ram_gb: Optional[float] = None
    cpu_idle_pct: Optional[float] = None
    throughput_samples_per_sec: Optional[float] = None


class CommMetrics(BaseModel):
    bandwidth_mbps: Optional[float] = None
    latency_ms: Optional[float] = None
    reliability_pct: Optional[float] = None


class ARCCalculateRequest(BaseModel):
    submission_id: str
    client_id: str
    round_id: str
    local_accuracy: float = 0.0
    local_loss: float = 0.0
    samples_trained: int = 0
    # ARC formula weights override (optional - uses DB config if not provided)
    arc_config: Optional[dict] = None
    # Telemetry metrics
    quality_metrics: QualityMetrics = QualityMetrics()
    diversity_metrics: DiversityMetrics = DiversityMetrics()
    resource_metrics: ResourceMetrics = ResourceMetrics()
    comm_metrics: CommMetrics = CommMetrics()


@router.post("/calculate")
async def calculate_arc(request: ARCCalculateRequest):
    """
    Compute the ARC score for a client submission.
    
    Uses telemetry data reported by the client to compute the three
    ARC components (Relevance, Resource, Communication) and their
    weighted composite score.
    """
    logger.info(f"ARC calculation for submission={request.submission_id}, client={request.client_id}")

    # Use provided arc_config override or fall back to default weights
    arc_config = request.arc_config or {}
    engine = ARCScoringEngine(arc_config)

    result = engine.score(
        quality_metrics=request.quality_metrics.model_dump(),
        diversity_metrics=request.diversity_metrics.model_dump(),
        resource_metrics=request.resource_metrics.model_dump(),
        comm_metrics=request.comm_metrics.model_dump(),
    )

    return {
        "submission_id": request.submission_id,
        "client_id": request.client_id,
        **result,
    }


@router.get("/config")
async def get_arc_config():
    """Return the current ARC formula version and default weight configuration."""
    return {
        "formula_version": "1.0",
        "formula": "ARC_i = 0.40*R_i + 0.30*S_i + 0.30*C_i",
        "components": {
            "R_i": "Relevance = 0.60*Quality + 0.40*Diversity",
            "S_i": "Resource  = 0.40*Memory + 0.40*Processing + 0.20*Capacity",
            "C_i": "Communication = 0.50*Bandwidth + 0.30*Latency + 0.20*Reliability",
        },
        "default_weights": {
            "lambda_relevance": 0.40,
            "lambda_resource": 0.30,
            "lambda_communication": 0.30,
            "weight_quality": 0.60,
            "weight_diversity": 0.40,
            "weight_memory": 0.40,
            "weight_processing": 0.40,
            "weight_capacity": 0.20,
            "weight_bandwidth": 0.50,
            "weight_latency": 0.30,
            "weight_reliability": 0.20,
        },
        "score_range": "[0, 100]",
    }
