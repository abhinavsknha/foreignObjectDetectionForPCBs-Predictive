from fastapi import FastAPI, UploadFile, File, HTTPException
from starlette.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import logging
import os
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv

from .database import engine, Base
from . import models
from .celery_worker import celery_app, process_image_inference, run_inference
from . import pipeline

load_dotenv()
log = logging.getLogger("pcb.api")

# Inspection records are optional: the API starts and serves inference even if the database is down.
try:
    Base.metadata.create_all(bind=engine)
except Exception as e:  # noqa: BLE001
    log.warning("Database unavailable, results will not be stored: %s", e)

app = FastAPI(title="PCB Defect Detection API")

# Enable CORS for the Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Local storage directory to mock S3 for local development
STORAGE = Path("local_storage")
STORAGE.mkdir(exist_ok=True)
app.mount("/local_storage", StaticFiles(directory="local_storage"), name="local_storage")

MAX_UPLOAD_BYTES = 25 * 1024 * 1024
ALLOWED_EXT = {"png", "jpg", "jpeg", "bmp", "webp", "tif", "tiff"}

# "celery" (default): tasks go through Redis to a Celery worker.
# "inline": run in this process, for local development without Redis.
INLINE = os.getenv("INFERENCE_MODE", "celery").lower() == "inline"
_pool = ThreadPoolExecutor(max_workers=1) if INLINE else None
_inline_tasks = {}
if INLINE:
    _pool.submit(pipeline.get_models)  # load both detectors in the background at start-up


def _resolve_in_storage(file_key: str) -> Path:
    path = Path(file_key).resolve()
    if STORAGE.resolve() not in path.parents or not path.is_file():
        raise HTTPException(status_code=400, detail="Unknown file_key")
    return path


@app.get("/")
def read_root():
    return {"status": "ok", "message": "API is running"}


_worker_seen = 0.0


@app.get("/api/health")
def health():
    """Used by the UI status pill: reports whether an inference worker can take a job."""
    global _worker_seen
    if INLINE:
        return {"api": "ok", "mode": "inline", "worker": True, "models_loaded": pipeline.models_loaded()}
    try:
        if celery_app.control.ping(timeout=2.0):
            _worker_seen = time.time()
    except Exception:  # noqa: BLE001
        pass
    # A worker that is busy with a job cannot answer a ping, so a reply in the last minute still counts.
    return {"api": "ok", "mode": "celery", "worker": time.time() - _worker_seen < 60}


@app.post("/api/upload/local")
async def local_upload(file: UploadFile = File(...)):
    """
    This endpoint mocks the direct-to-S3 upload for local development.
    In production, Next.js would PUT directly to a Cloudflare R2 presigned URL.
    """
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Image is larger than 25 MB")
    if cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR) is None:
        raise HTTPException(status_code=400, detail="The file is not a readable image")

    ext = (file.filename or "").rsplit(".", 1)[-1].lower()
    ext = ext if ext in ALLOWED_EXT else "png"
    file_path = STORAGE / f"{uuid.uuid4()}.{ext}"
    file_path.write_bytes(data)
    return {"file_key": file_path.as_posix(), "message": "Uploaded to local storage mock"}


@app.post("/api/inspect")
async def inspect(file: UploadFile = File(...)):
    """Upload and queue in one request, so a slow tunnel costs one round trip instead of two."""
    saved = await local_upload(file)
    return await run_in_threadpool(trigger_inference, saved["file_key"])


def _run_inline(task_id: str, file_key: str):
    def progress(stage, route=None):
        _inline_tasks[task_id].update(status="processing", stage=stage, route=route)

    try:
        result = run_inference(file_key, progress)
        _inline_tasks[task_id] = {"status": "completed", "result": result}
    except Exception as e:  # noqa: BLE001
        log.exception("Inference failed")
        _inline_tasks[task_id] = {"status": "failed", "error": str(e)}


@app.post("/api/inference/trigger")
def trigger_inference(file_key: str):
    """Queues the inspection (Redis and Celery) or runs it in this process (inline mode)."""
    _resolve_in_storage(file_key)
    if INLINE:
        task_id = str(uuid.uuid4())
        _inline_tasks[task_id] = {"status": "processing", "stage": "queued", "route": None}
        _pool.submit(_run_inline, task_id, file_key)
        return {"status": "queued", "task_id": task_id, "file_key": file_key}
    task = process_image_inference.delay(file_key=file_key)
    return {"status": "queued", "task_id": task.id, "file_key": file_key}


@app.get("/api/inference/status/{task_id}")
def get_inference_status(task_id: str):
    if INLINE:
        return _inline_tasks.get(task_id) or {"status": "failed", "error": "Unknown task"}

    from celery.result import AsyncResult

    result = AsyncResult(task_id, app=celery_app)
    if result.state == "SUCCESS":
        return {"status": "completed", "result": result.result}
    if result.state == "FAILURE":
        return {"status": "failed", "error": str(result.result)}
    info = result.info if result.state == "PROGRESS" and isinstance(result.info, dict) else {}
    return {"status": "processing", "stage": info.get("stage", "queued"), "route": info.get("route")}
