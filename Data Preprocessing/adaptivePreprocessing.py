import os
import cv2
import numpy as np
import glob
import shutil
from tqdm import tqdm

# Configuration
INPUT_ROOT = r"C:\Users\Abhinav\Documents\VIT\4th Year\Projects\FOD_Predictive_Detection\Data Preprocessing\Classified_Dataset\images\train"
OUTPUT_ROOT = r"C:\Users\Abhinav\Documents\VIT\4th Year\Projects\FOD_Predictive_Detection\Data Preprocessing\Adaptive_Dataset\images\train"

# Threshold for complexity (Laplacian Variance)
# Low values mean smooth/blurry, high values mean many sharp edges/details
COMPLEXITY_THRESHOLD = 100.0 

# Resolution Settings
LOW_RES = (224, 224)
HIGH_RES = (640, 640)

def analyze_complexity(image):
    """Calculates the Laplacian variance to measure edge density."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return variance

def run_adaptive_preprocessing(limit=10):
    """Processes images with dynamic resolution scaling."""
    if not os.path.exists(OUTPUT_ROOT):
        os.makedirs(OUTPUT_ROOT)

    # Get list of images
    images = glob.glob(os.path.join(INPUT_ROOT, "*.jpg"))
    
    # For testing, we only take the first 'limit' images
    images = images[:limit]
    
    print(f"Starting Adaptive Preprocessing on {len(images)} images...")
    print(f"Threshold: {COMPLEXITY_THRESHOLD}\n")
    print(f"{'File Name':<25} | {'Variance':<10} | {'Resized To':<15} | {'Status'}")
    print("-" * 65)

    for img_path in images:
        img_name = os.path.basename(img_path)
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"{img_name:<25} | Error reading image.")
            continue

        # 1. Analyze Complexity
        variance = analyze_complexity(img)
        
        # 2. Adaptive Decision
        if variance < COMPLEXITY_THRESHOLD:
            target_size = LOW_RES
            status = "LOW_RES (Efficient)"
        else:
            target_size = HIGH_RES
            status = "HIGH_RES (Precise)"
            
        # 3. Process (Resize)
        resized_img = cv2.resize(img, target_size)
        
        # 4. Save
        save_path = os.path.join(OUTPUT_ROOT, img_name)
        cv2.imwrite(save_path, resized_img)
        
        print(f"{img_name[:25]:<25} | {variance:<10.2f} | {str(target_size):<15} | {status}")

    print(f"\nProcessing Complete. Outputs saved to: {OUTPUT_ROOT}")

if __name__ == "__main__":
    run_adaptive_preprocessing(limit=10)