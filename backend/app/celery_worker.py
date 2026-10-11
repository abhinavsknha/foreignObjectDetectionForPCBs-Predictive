import os
import time
from celery import Celery
from dotenv import load_dotenv
from sqlalchemy.orm import sessionmaker
from .database import engine
from . import models

# ML Imports
from ultralytics import YOLO, RTDETR

import ssl

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

# Load Models globally so they stay in memory
# Suppress heavy logging
print("Loading Ensemble Models...")
yolo_model = YOLO("../MODEL RESULTS/MORE TRAINING/YOLO/best.pt")
rtdetr_model = RTDETR("../MODEL RESULTS/MORE TRAINING/DETR/best.pt")

# Standard PCB Defect Mapping
CLASS_MAP = {
    "SH": "Short",
    "SP": "Spur",
    "SC": "Spurious Copper",
    "OP": "Open",
    "MB": "Mouse Bite",
    "HB": "Hole Breakout",
    "CS": "Conductor Scratch",
    "CFO": "Conductor Foreign Object",
    "BMFO": "Base Material Foreign Object"
}

def calculate_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

def weighted_box_fusion(yolo_dets, rtdetr_dets, iou_thr=0.5):
    """
    Simplified Weighted Box Fusion (WBF)
    Combines predictions from both models based on IoU and confidence weights.
    """
    all_dets = yolo_dets + rtdetr_dets
    if not all_dets:
        return []

    # Group overlapping boxes by class
    fused_boxes = []
    
    # Sort by confidence
    all_dets.sort(key=lambda x: x["confidence"], reverse=True)
    
    used = [False] * len(all_dets)
    
    for i, det1 in enumerate(all_dets):
        if used[i]: continue
        
        group = [det1]
        used[i] = True
        
        for j, det2 in enumerate(all_dets):
            if used[j] or det1["class"] != det2["class"]:
                continue
            
            if calculate_iou(det1["bbox"], det2["bbox"]) > iou_thr:
                group.append(det2)
                used[j] = True
                
        # Calculate WBF for the group
        total_conf = sum(d["confidence"] for d in group)
        avg_conf = total_conf / len(group)
        
        # Weighted coordinates
        w_x1 = sum(d["bbox"][0] * d["confidence"] for d in group) / total_conf
        w_y1 = sum(d["bbox"][1] * d["confidence"] for d in group) / total_conf
        w_x2 = sum(d["bbox"][2] * d["confidence"] for d in group) / total_conf
        w_y2 = sum(d["bbox"][3] * d["confidence"] for d in group) / total_conf
        
        source = "ENSEMBLE (WBF)" if len(group) > 1 else group[0]["source"]
        
        fused_boxes.append({
            "class": det1["class"],
            "full_name": CLASS_MAP.get(det1["class"], det1["class"]),
            "confidence": avg_conf,
            "bbox": [w_x1, w_y1, w_x2, w_y2],
            "source": source
        })
        
    return fused_boxes

def parse_results(results, source_name):
    detections = []
    names = results[0].names
    for box in results[0].boxes:
        conf = float(box.conf[0])
        cls = int(box.cls[0])
        class_name = names[cls]
        coords = box.xyxy[0].tolist()
        detections.append({
            "class": class_name,
            "confidence": conf,
            "bbox": coords,
            "source": source_name
        })
    return detections

@celery_app.task(name="process_image_inference")
def process_image_inference(file_key: str):
    print(f"Starting TRUE ENSEMBLE inference for {file_key}...")
    start_time = time.time()
    
    # Lowering base threshold to 0.20 so the frontend UI can filter dynamically via a slider.
    CONF_THRESH = 0.20
    
    yolo_res = yolo_model(file_key, conf=CONF_THRESH, verbose=False)
    yolo_dets = parse_results(yolo_res, "YOLO26n")
    
    rtdetr_res = rtdetr_model(file_key, conf=CONF_THRESH, verbose=False)
    rtdetr_dets = parse_results(rtdetr_res, "RT-DETR")
    
    # Perform Weighted Box Fusion
    fused_detections = weighted_box_fusion(yolo_dets, rtdetr_dets, iou_thr=0.4)
    
    # Generate tips based on defects
    tips = []
    classes_found = set(d["class"] for d in fused_detections)
    
    if "SH" in classes_found:
        tips.append("Short: Unintended connection. Inspect wave soldering temp and residual copper.")
    if "SP" in classes_found:
        tips.append("Spur: Irregular protrusion found. Check etching chemical balance.")
    if "SC" in classes_found:
        tips.append("Spurious Copper: Unwanted residue on base material. Review etching exposure timing.")
    if "OP" in classes_found:
        tips.append("Open: Interrupted conductor path. Halt line and check photoresist application.")
    if "MB" in classes_found:
        tips.append("Mouse Bite: Gap at conductor edge. Inspect routing/milling tools for wear.")
    if "HB" in classes_found:
        tips.append("Hole Breakout: Hole center deviation. Recalibrate CNC drilling coordinates.")
    if "CS" in classes_found:
        tips.append("Conductor Scratch: Inspect conveyor belts and automated handling arms for sharp edges.")
    if "CFO" in classes_found:
        tips.append("Conductor Foreign Object: Dirt/bubbles on conductor. Clean lamination rollers.")
    if "BMFO" in classes_found:
        tips.append("Base Material Foreign Object: Contamination on base. Check cleanroom air filters.")
    
    # Save to PostgreSQL
    try:
        db = SessionLocal()
        # Create a dummy user for local testing if none exists
        user = db.query(models.User).first()
        if not user:
            user = models.User(email="admin@pcb-qa.local", hashed_password="hashed")
            db.add(user)
            db.commit()
            db.refresh(user)
            
        # Create upload record with a 512-D vector (simulated from RT-DETR bottleneck)
        import numpy as np
        simulated_embedding = np.random.rand(512).tolist()
        
        db_upload = models.Upload(
            filename=file_key,
            s3_key=file_key,
            s3_url=f"http://localhost:8000/{file_key}",
            owner_id=user.id,
            embedding=simulated_embedding
        )
        db.add(db_upload)
        db.commit()
        db.refresh(db_upload)
        
        # Save WBF boxes
        db_detection = models.DetectionResult(
            upload_id=db_upload.id,
            boxes=fused_detections,
            used_for_training=0
        )
        db.add(db_detection)
        db.commit()
        print(f"[POSTGRES] Saved vector embedding and WBF results to database (Upload ID: {db_upload.id})")
        db.close()
    except Exception as e:
        print(f"[POSTGRES ERROR] Failed to save to DB: {e}")
    
    end_time = time.time()
    print(f"Ensemble completed in {end_time - start_time:.2f}s")
    
    return {
        "status": "success",
        "file_key": file_key,
        "results": {
            "yolo_count": len(yolo_dets),
            "rtdetr_count": len(rtdetr_dets),
            "fused_count": len(fused_detections),
            "detections": fused_detections,
            "tips": tips
        }
    }
