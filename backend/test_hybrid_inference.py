import os
import time
import cv2
from ultralytics import YOLO, RTDETR

# Setup Paths
YOLO_WEIGHTS = "../yolo26n.pt"
RTDETR_WEIGHTS = "../MODEL RESULTS/RETR/results/runs/detect/PCB_RTDETR_L_Run/weights/best.pt"

# Load Models
print("Loading YOLO Model...")
yolo_model = YOLO(YOLO_WEIGHTS)

print("Loading RT-DETR Model...")
detr_model = RTDETR(RTDETR_WEIGHTS)

def run_hybrid_inference(image_path: str, conf_threshold: float = 0.5):
    """
    Hybrid logic:
    1. Run YOLO (fast).
    2. Check confidence of detections.
    3. If any detection is below the threshold, or if NO defects are found (but we suspect one),
       fallback to RT-DETR (more accurate, slower).
    """
    print(f"\n--- Processing Image: {os.path.basename(image_path)} ---")
    
    # 1. Run YOLO
    t0 = time.time()
    yolo_results = yolo_model(image_path, verbose=False)[0]
    yolo_time = time.time() - t0
    
    yolo_boxes = yolo_results.boxes
    print(f"[YOLO] Found {len(yolo_boxes)} detections in {yolo_time:.3f}s")
    
    needs_fallback = False
    
    if len(yolo_boxes) == 0:
        print("[Decision] YOLO found 0 defects. Triggering RT-DETR to double check.")
        needs_fallback = True
    else:
        for box in yolo_boxes:
            conf = float(box.conf[0])
            print(f"  - YOLO Det: {yolo_results.names[int(box.cls[0])]} | Conf: {conf:.2f}")
            if conf < conf_threshold:
                print(f"[Decision] Low confidence detection ({conf:.2f} < {conf_threshold}). Triggering RT-DETR.")
                needs_fallback = True
                break
                
    if not needs_fallback:
        print("[Result] YOLO confidence is high enough. Using YOLO results.")
        return format_results(yolo_results, source="YOLO")
        
    # 2. Fallback to RT-DETR
    t1 = time.time()
    detr_results = detr_model(image_path, verbose=False)[0]
    detr_time = time.time() - t1
    
    detr_boxes = detr_results.boxes
    print(f"[RT-DETR] Found {len(detr_boxes)} detections in {detr_time:.3f}s")
    
    for box in detr_boxes:
        conf = float(box.conf[0])
        print(f"  - DETR Det: {detr_results.names[int(box.cls[0])]} | Conf: {conf:.2f}")
        
    print("[Result] Using RT-DETR results.")
    return format_results(detr_results, source="RT-DETR")


def format_results(results, source):
    boxes = []
    for box in results.boxes:
        boxes.append({
            "class": results.names[int(box.cls[0])],
            "confidence": float(box.conf[0]),
            "bbox": box.xyxy[0].cpu().numpy().tolist()
        })
    return {"source": source, "detections": boxes}


if __name__ == "__main__":
    # Test on a few sample images
    import glob
    
    # Find some sample images
    sample_dir = "../Dataset/DsPCBSD+/Data_YOLO/images/train/"
    sample_images = glob.glob(os.path.join(sample_dir, "*.jpg"))[:3]
    
    if not sample_images:
        print("No sample images found! Check path.")
    else:
        for img_path in sample_images:
            res = run_hybrid_inference(img_path, conf_threshold=0.65)
            print("Final Output:", res)
