import os
import glob
import matplotlib.pyplot as plt
import cv2
from collections import Counter

# Configuration
DATASET_PATH = r"Dataset/DsPCBSD+/Data_YOLO"
OUTPUT_DIR = r"EDA_Results"

# Class names based on data.yaml
CLASS_NAMES = ["SH", "SP", "SC", "OP", "MB", "HB", "CS", "CFO", "BMFO"]

def ensure_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def analyze_class_balance():
    print("Analyzing class balance...")
    label_files = glob.glob(os.path.join(DATASET_PATH, 'labels', '**', '*.txt'), recursive=True)
    
    class_counts = Counter()
    invalid_boxes = 0
    total_boxes = 0
    
    for label_file in label_files:
        with open(label_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    xc, yc, w, h = map(float, parts[1:5])
                    
                    class_counts[class_id] += 1
                    total_boxes += 1
                    
                    # Verify bounding box bounds
                    if w <= 0 or h <= 0 or xc < 0 or xc > 1 or yc < 0 or yc > 1:
                        invalid_boxes += 1

    # Plotting
    classes = [CLASS_NAMES[i] if i < len(CLASS_NAMES) else str(i) for i in class_counts.keys()]
    counts = list(class_counts.values())
    
    plt.figure(figsize=(10, 6))
    plt.bar(classes, counts, color='skyblue')
    plt.xlabel('Defect Classes')
    plt.ylabel('Number of Instances')
    plt.title('Class Balance in DsPCBSD+ Dataset')
    
    for i, v in enumerate(counts):
        plt.text(i, v + max(counts)*0.01, str(v), ha='center', va='bottom')
        
    ensure_dir(OUTPUT_DIR)
    plt.savefig(os.path.join(OUTPUT_DIR, 'class_balance.png'))
    plt.close()
    
    print(f"Total bounding boxes: {total_boxes}")
    print(f"Invalid bounding boxes found: {invalid_boxes}")
    print(f"Class balance chart saved to {OUTPUT_DIR}/class_balance.png\n")

def visualize_sample_annotations():
    print("Verifying bounding box annotations visually...")
    image_files = glob.glob(os.path.join(DATASET_PATH, 'images', 'train', '*.jpg'))
    if not image_files:
        print("No training images found (.jpg). Trying .png...")
        image_files = glob.glob(os.path.join(DATASET_PATH, 'images', 'train', '*.png'))
        
    if not image_files:
        print("Could not find any images to visualize.")
        return

    # Pick the first image as a sample
    sample_img_path = image_files[0]
    sample_label_path = sample_img_path.replace('images', 'labels').rsplit('.', 1)[0] + '.txt'
    
    img = cv2.imread(sample_img_path)
    if img is None:
        print(f"Failed to read image {sample_img_path}")
        return
        
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h_img, w_img, _ = img.shape
    
    if os.path.exists(sample_label_path):
        with open(sample_label_path, 'r') as f:
            for line in f.readlines():
                parts = line.strip().split()
                if len(parts) >= 5:
                    class_id = int(parts[0])
                    xc, yc, w, h = map(float, parts[1:5])
                    
                    # YOLO format to pixel coordinates
                    x_center, y_center = int(xc * w_img), int(yc * h_img)
                    width, height = int(w * w_img), int(h * h_img)
                    
                    x1 = int(x_center - width / 2)
                    y1 = int(y_center - height / 2)
                    x2 = int(x_center + width / 2)
                    y2 = int(y_center + height / 2)
                    
                    class_name = CLASS_NAMES[class_id] if class_id < len(CLASS_NAMES) else str(class_id)
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(img, class_name, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    plt.figure(figsize=(8, 8))
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Sample Annotations: {os.path.basename(sample_img_path)}")
    plt.savefig(os.path.join(OUTPUT_DIR, 'sample_annotation_check.png'))
    plt.close()
    
    print(f"Sample annotation check saved to {OUTPUT_DIR}/sample_annotation_check.png")

if __name__ == "__main__":
    ensure_dir(OUTPUT_DIR)
    analyze_class_balance()
    visualize_sample_annotations()
    print("EDA initialization complete!")
