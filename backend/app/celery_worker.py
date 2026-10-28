import os
import ssl
import threading

from celery import Celery, signals
from dotenv import load_dotenv
from sqlalchemy.orm import sessionmaker

from . import models
from .database import engine
from . import pipeline

load_dotenv()

# Setup Celery
redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")

# Handle encrypted cloud Redis (Upstash) which requires ssl_cert_reqs parameter
if redis_url.startswith("rediss://") and "ssl_cert_reqs" not in redis_url:
    delimiter = "&" if "?" in redis_url else "?"
    redis_url = f"{redis_url}{delimiter}ssl_cert_reqs=CERT_NONE"

celery_app = Celery("pcb_tasks", broker=redis_url, backend=redis_url)

if redis_url.startswith("rediss://"):
    celery_app.conf.update(
        broker_use_ssl={"ssl_cert_reqs": ssl.CERT_NONE},
        redis_backend_use_ssl={"ssl_cert_reqs": ssl.CERT_NONE},
    )

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


@signals.worker_ready.connect
def warm_up_models(**_):
    """Load both detectors when the worker starts so the first upload is not slow."""
    print("Loading ensemble models...")
    pipeline.get_models()
    print("Ensemble models ready.")


_owner_id = None


def save_to_db(file_key, detections):
    """Persist the inspection record. Inference does not depend on the database being reachable."""
    global _owner_id
    try:
        import numpy as np

        db = SessionLocal()
        try:
            if _owner_id is None:
                # Create a dummy user for local testing if none exists
                user = db.query(models.User).first()
                if not user:
                    user = models.User(email="admin@pcb-qa.local", hashed_password="hashed")
                    db.add(user)
                    db.flush()
                _owner_id = user.id

            # Upload record with a 512-D vector (simulated from RT-DETR bottleneck)
            db_upload = models.Upload(
                filename=file_key,
                s3_key=file_key,
                s3_url=f"http://localhost:8000/{file_key}",
                owner_id=_owner_id,
                embedding=np.random.rand(512).tolist(),
            )
            db.add(db_upload)
            db.flush()  # assigns the id without another round trip

            db.add(models.DetectionResult(upload_id=db_upload.id, boxes=detections, used_for_training=0))
            db.commit()
            print(f"[POSTGRES] Saved vector embedding and WBF results to database (Upload ID: {db_upload.id})")
        finally:
            db.close()
    except Exception as e:
        print(f"[POSTGRES ERROR] Failed to save to DB: {e}")


def run_inference(file_key, progress=None, reference_key=""):
    """Run the full pipeline and persist the result. Shared by the Celery task and inline mode."""
    print(f"Starting ensemble inference for {file_key}...")
    if reference_key:
        results = pipeline.compare_to_reference(file_key, reference_key, progress=progress)
    else:
        results = pipeline.run_pipeline(file_key, pipeline.get_models(), progress=progress)
    print(f"Ensemble completed in {results['timings_ms']['total'] / 1000:.2f}s "
          f"({results['route']['path']}, {results['fused_count']} fused detections)")
    # Saved in the background: the operator gets the result without waiting for the database.
    threading.Thread(target=save_to_db, args=(file_key, results["detections"]), daemon=True).start()
    return {"status": "success", "file_key": file_key, "results": results}


@celery_app.task(bind=True, name="process_image_inference")
def process_image_inference(self, file_key: str, reference_key: str = ""):
    def progress(stage, route=None):
        self.update_state(state="PROGRESS", meta={"stage": stage, "route": route})

    return run_inference(file_key, progress, reference_key)
