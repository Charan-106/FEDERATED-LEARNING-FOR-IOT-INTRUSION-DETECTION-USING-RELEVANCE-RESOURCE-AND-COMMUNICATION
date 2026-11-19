"""
ARC-FL ML Service - FastAPI Application

Startup events:
  - Load active global model from storage
  - Validate storage directories exist

Authentication:
  - All endpoints except GET /health require X-API-Key header
  - API key is validated against ML_SERVICE_API_KEY env var

CORS:
  - Configured to only allow requests from the Node.js backend
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from app.api.aggregation import router as aggregation_router
from app.api.arc import router as arc_router
from app.api.evaluation import router as evaluation_router
from app.api.health import router as health_router
from app.api.training import router as training_router
from app.config import config
from app.models.registry import registry

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle events."""
    # ---- Startup ----
    logger.info("ARC-FL ML Service starting up...")
    try:
        meta = registry.load_active_model()
        logger.info(f"Active model loaded: version={meta['version']}, params={meta['parameters']:,}")
    except Exception as e:
        logger.warning(f"Could not load model at startup: {e}")
        logger.warning("Service will start without a model. Load one via the registry.")

    yield

    # ---- Shutdown ----
    logger.info("ARC-FL ML Service shutting down.")


app = FastAPI(
    title="ARC-FL ML Service",
    description="Federated Learning ML backend: ARC scoring, aggregation, evaluation, training",
    version="1.0.0",
    lifespan=lifespan,
)

# ============================================================
# CORS - only allow the Node.js backend
# ============================================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5000", "http://localhost:5000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# API Key Authentication Middleware
# Applies to all routes EXCEPT /health
# ============================================================
@app.middleware("http")
async def api_key_middleware(request: Request, call_next):
    """
    Validates X-API-Key header on all non-health endpoints.
    Returns 403 if the key is missing or incorrect.
    """
    # Health endpoint is public (for monitoring/orchestration)
    if request.url.path == "/health":
        return await call_next(request)

    api_key = request.headers.get("X-API-Key", "")
    if not config.ML_SERVICE_API_KEY:
        # If no key is configured, skip auth (dev mode warning)
        logger.warning("ML_SERVICE_API_KEY not set - authentication disabled!")
        return await call_next(request)

    if api_key != config.ML_SERVICE_API_KEY:
        from fastapi.responses import JSONResponse
        return JSONResponse(
            status_code=403,
            content={"error": "Forbidden", "message": "Invalid or missing X-API-Key header"},
        )

    return await call_next(request)


# ============================================================
# Mount Routers
# ============================================================
app.include_router(health_router)
app.include_router(arc_router)
app.include_router(aggregation_router)
app.include_router(evaluation_router)
app.include_router(training_router)
