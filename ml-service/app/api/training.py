"""
Training Router - /training/

Endpoints:
  POST /training/start  → Trigger local training simulation
  POST /training/stream → Stream real-time training events via SSE
"""

import asyncio
import json
import logging
import threading
from typing import Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from app.training.trainer import train_local

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/training", tags=["Training"])


class TrainingRequest(BaseModel):
    dataset_path: str
    global_model_path: str
    epochs: int = 2
    batch_size: int = 256
    learning_rate: float = 1e-4
    max_samples: Optional[int] = None
    freeze_bn_stats: bool = True


@router.post("/start")
async def start_training(request: TrainingRequest):
    """
    Trigger local training on a dataset partition using the global model as starting point.
    Runs non-blocking in a threadpool so health checks and other endpoints are never blocked.
    """
    logger.info(
        f"Local training requested: dataset={request.dataset_path}, epochs={request.epochs}, "
        f"batch_size={request.batch_size}, lr={request.learning_rate}, max_samples={request.max_samples}"
    )

    try:
        result = await run_in_threadpool(
            train_local,
            dataset_path=request.dataset_path,
            global_model_path=request.global_model_path,
            epochs=request.epochs,
            batch_size=request.batch_size,
            learning_rate=request.learning_rate,
            max_samples=request.max_samples,
            freeze_bn_stats=request.freeze_bn_stats,
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Training error: {str(e)}")

    return result


@router.post("/stream")
async def stream_training(request: TrainingRequest):
    """
    Stream real-time training progress events directly from the PyTorch training loop
    using Server-Sent Events (SSE).
    """
    logger.info(
        f"Local training stream requested: dataset={request.dataset_path}, epochs={request.epochs}, "
        f"batch_size={request.batch_size}, lr={request.learning_rate}, max_samples={request.max_samples}"
    )

    loop = asyncio.get_running_loop()
    event_queue: asyncio.Queue = asyncio.Queue()

    def callback(event_data):
        loop.call_soon_threadsafe(event_queue.put_nowait, event_data)

    def run_training_worker():
        try:
            result = train_local(
                dataset_path=request.dataset_path,
                global_model_path=request.global_model_path,
                epochs=request.epochs,
                batch_size=request.batch_size,
                learning_rate=request.learning_rate,
                max_samples=request.max_samples,
                freeze_bn_stats=request.freeze_bn_stats,
                progress_callback=callback,
            )
            loop.call_soon_threadsafe(event_queue.put_nowait, {"event": "RESULT", "stage": "RESULT", "data": result})
        except Exception as e:
            logger.error(f"Stream training worker failed: {e}", exc_info=True)
            loop.call_soon_threadsafe(event_queue.put_nowait, {"event": "ERROR", "stage": "ERROR", "error": str(e)})
        finally:
            loop.call_soon_threadsafe(event_queue.put_nowait, None)

    worker_thread = threading.Thread(target=run_training_worker, daemon=True)
    worker_thread.start()

    async def event_generator():
        while True:
            event = await event_queue.get()
            if event is None:
                break
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
