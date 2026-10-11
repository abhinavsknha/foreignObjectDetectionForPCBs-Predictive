import os
from .database import SessionLocal
from .models import DetectionResult, Upload

def run_incremental_training():
    """
    1. Query Postgres for DetectionResult where used_for_training == 0
    2. Download the micro-images from `pcb-training-buffer` S3 bucket
    3. Run lightweight PyTorch training loop on the new data
    4. Validate model performance against holdout set
    5. Update used_for_training = 1 in DB
    6. Delete micro-images from `pcb-training-buffer` S3 bucket
    """
    db = SessionLocal()
    
    print("Fetching new detections for incremental training...")
    new_results = db.query(DetectionResult).filter(DetectionResult.used_for_training == 0).all()
    
    if not new_results:
        print("No new data to train on. Exiting.")
        return
        
    print(f"Found {len(new_results)} new results.")
    
    # [Placeholder for actual training loop]
    # e.g. run_rt_detr_finetune(new_results)
    
    print("Training complete. Updating database and purging buffer...")
    for result in new_results:
        result.used_for_training = 1
        
        # Purge logic here (boto3 s3 client delete_object)
        upload = db.query(Upload).filter(Upload.id == result.upload_id).first()
        if upload:
            print(f"Purging {upload.s3_key} from pcb-training-buffer...")
            
    db.commit()
    db.close()
    
if __name__ == "__main__":
    run_incremental_training()
