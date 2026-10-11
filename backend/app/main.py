from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import shutil
import uuid
from dotenv import load_dotenv
from .database import engine, Base
from . import models
from .celery_worker import process_image_inference

load_dotenv()

# Create database tables
Base.metadata.create_all(bind=engine)

app = FastAPI(title="PCB Defect Detection API")

# Enable CORS for the Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi.staticfiles import StaticFiles

# Local storage directory to mock S3 for local development
os.makedirs("local_storage", exist_ok=True)
app.mount("/local_storage", StaticFiles(directory="local_storage"), name="local_storage")

@app.get("/")
def read_root():
    return {"status": "ok", "message": "API is running"}

@app.post("/api/upload/local")
async def local_upload(file: UploadFile = File(...)):
    """
    This endpoint mocks the direct-to-S3 upload for local development.
    In production, Next.js would PUT directly to a Cloudflare R2 presigned URL.
    """
    file_extension = file.filename.split(".")[-1]
    unique_filename = f"{uuid.uuid4()}.{file_extension}"
    file_path = f"local_storage/{unique_filename}"
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    return {"file_key": file_path, "message": "Uploaded to local storage mock"}

@app.post("/api/inference/trigger")
def trigger_inference(file_key: str):
    """
    Triggers the Celery worker via Redis.
    """
    # Delay sends the task to Redis for Celery to pick up
    task = process_image_inference.delay(file_key=file_key)
    return {"status": "queued", "task_id": task.id, "file_key": file_key}

@app.get("/api/inference/status/{task_id}")
def get_inference_status(task_id: str):
    from celery.result import AsyncResult
    from .celery_worker import celery_app
    
    result = AsyncResult(task_id, app=celery_app)
    
    if result.ready():
        return {"status": "completed", "result": result.result}
    
    return {"status": "processing"}
