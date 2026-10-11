import os
import cv2
import glob
# Constants
INPUT_ROOT = r"C:\Users\Abhinav\Documents\VIT\4th Year\Projects\FOD_Predictive_Detection\Dataset\DsPCBSD+\Data_YOLO"
OUTPUT_ROOT = r"C:\Users\Abhinav\Documents\VIT\4th Year\Projects\FOD_Predictive_Detection\Dataset\DsPCBSD+\Processed_640"
TARGET_SIZE = (640, 640)

def ensure_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

def process_dataset():
    for split in ['train', 'val']:
        img_dir = os.path.join(INPUT_ROOT, 'images', split)
        out_img_dir = os.path.join(OUTPUT_ROOT, 'images', split)
        out_label_dir = os.path.join(OUTPUT_ROOT, 'labels', split)
        
        ensure_dir(out_img_dir)
        ensure_dir(out_label_dir)
        
        image_files = glob.glob(os.path.join(img_dir, '*.jpg'))
        
        print(f"Processing {split} set ({len(image_files)} images)...")
        
        for img_path in image_files:
            # Read and Resize
            img = cv2.imread(img_path)
            if img is None: continue
            
            resized_img = cv2.resize(img, TARGET_SIZE)
            
            # Save Image
            out_img_path = os.path.join(out_img_dir, os.path.basename(img_path))
            cv2.imwrite(out_img_path, resized_img)
            
            # Copy Labels
            label_path = img_path.replace('images', 'labels').rsplit('.', 1)[0] + '.txt'
            if os.path.exists(label_path):
                out_label_path = os.path.join(out_label_dir, os.path.basename(label_path))
                with open(label_path, 'r') as f:
                    content = f.read()
                with open(out_label_path, 'w') as f:
                    f.write(content)

if __name__ == "__main__":
    process_dataset()
    print(f"Data preprocessing complete. Processed dataset saved to: {OUTPUT_ROOT}")
