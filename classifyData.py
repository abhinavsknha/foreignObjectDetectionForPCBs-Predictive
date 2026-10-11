import os
import shutil
import glob

# Configuration
INPUT_ROOT = r"C:\Users\Abhinav\Documents\VIT\4th Year\Projects\FOD_Predictive_Detection\Data Preprocessing\Processed_640"
OUTPUT_ROOT = r"C:\Users\Abhinav\Documents\VIT\4th Year\Projects\FOD_Predictive_Detection\Data Preprocessing\Classified_Dataset"

def organize_dataset():
    splits = ["train", "val"]
    
    for split in splits:
        print(f"Processing {split} split...")
        
        # Input directories
        img_src_base = os.path.join(INPUT_ROOT, "images", split)
        lbl_src_base = os.path.join(INPUT_ROOT, "labels", split)
        
        # Output directories
        img_dst_base = os.path.join(OUTPUT_ROOT, "images", split)
        lbl_dst_base = os.path.join(OUTPUT_ROOT, "labels", split)
        
        # Get all images
        images = glob.glob(os.path.join(img_src_base, "*.jpg"))
        
        for img_path in images:
            # Corresponding label file
            lbl_path = img_path.replace("images", "labels").rsplit(".", 1)[0] + ".txt"
            
            if not os.path.exists(lbl_path):
                print(f"Warning: No label found for {img_path}. Skipping.")
                continue
                
            # Read the first label in the file to determine the class
            with open(lbl_path, 'r') as f:
                first_line = f.readline().strip()
                if not first_line:
                    continue
                class_id = first_line.split()[0] # The first number is the class index

            # Target directories for this specific class
            target_img_dir = os.path.join(img_dst_base, class_id)
            target_lbl_dir = os.path.join(lbl_dst_base, class_id)
            
            os.makedirs(target_img_dir, exist_ok=True)
            os.makedirs(target_lbl_dir, exist_ok=True)
            
            # Copy/Move files
            shutil.copy2(img_path, os.path.join(target_img_dir, os.path.basename(img_path)))
            shutil.copy2(lbl_path, os.path.join(target_lbl_dir, os.path.basename(lbl_path)))

    print(f"Done! Organized dataset is at: {OUTPUT_ROOT}")

if __name__ == "__main__":
    organize_dataset()