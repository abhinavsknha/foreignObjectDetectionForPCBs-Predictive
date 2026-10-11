import sys
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import shutil
import warnings
from typing import Tuple, Optional, List, Dict
from tqdm import tqdm
import matplotlib.pyplot as plt
import yaml

# Allow importing from the same directory
sys.path.insert(0, str(Path(__file__).parent))

from adaptive_preprocessor import DefectScaleRouter
from super_resolution import SuperResolver

class AdaptivePreprocessingPipeline:
    def __init__(self, 
                 sr_method: str = 'enhanced_bicubic',
                 sr_scale: int = 2,
                 target_size: Tuple[int, int] = (640, 640),
                 area_threshold: float = 0.005,
                 device: str = 'cpu'):
        """Initialize router and super-resolver."""
        self.router = DefectScaleRouter(area_threshold=area_threshold)
        self.sr = SuperResolver(method=sr_method, scale=sr_scale, device=device)
        self.target_size = target_size
    
    def process_single(self, image: np.ndarray, label_path: Optional[Path] = None) -> Tuple[np.ndarray, dict]:
        """
        Full pipeline: route → optional SR → resize to target_size → return.
        Returns (processed_image, metadata_dict)
        """
        metadata = {
            'original_shape': image.shape,
            'needs_sr': False,
            'defects_analyzed': 0,
            'micro_defects': 0,
            'defect_classes': []
        }
        
        # Use the router's route_single method
        needs_sr, route_meta = self.router.route_single(image, label_path)
        metadata['needs_sr'] = needs_sr
        metadata.update({k: v for k, v in route_meta.items() if k != 'needs_sr'})
        
        # Extract defect info from route metadata
        if route_meta.get('mode') == 'training' and 'classes' in route_meta:
            metadata['defect_classes'] = route_meta['classes']
            metadata['defects_analyzed'] = len(route_meta['classes'])
            # Count micro-defects (those with small area)
            metadata['micro_defects'] = 1 if needs_sr else 0
        
        if needs_sr:
            processed_image = self.sr.enhance(image)
        else:
            processed_image = image.copy()
            
        if processed_image.shape[:2] != self.target_size:
            processed_image = cv2.resize(processed_image, self.target_size, interpolation=cv2.INTER_LINEAR)
            
        return processed_image, metadata
        
    def _create_yaml(self, output_dir: Path, prefix: str):
        yaml_content = {
            'path': str(output_dir.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'names': {
                0: 'SH', 1: 'SP', 2: 'SC', 3: 'OP', 4: 'MB', 
                5: 'HB', 6: 'CS', 7: 'CFO', 8: 'BMFO'
            }
        }
        yaml_path = output_dir.parent / f"pcb_data_{prefix}.yaml"
        with open(yaml_path, 'w') as f:
            yaml.dump(yaml_content, f, default_flow_style=False)

    def process_dataset(self, 
                       input_img_dir: Path, 
                       input_label_dir: Path,
                       output_dir: Path,
                       create_all_sr_variant: bool = True) -> pd.DataFrame:
        """
        Process entire dataset split.
        """
        split_name = input_img_dir.name
        
        adaptive_img_dir = output_dir / 'adaptive' / 'images' / split_name
        adaptive_lbl_dir = output_dir / 'adaptive' / 'labels' / split_name
        adaptive_img_dir.mkdir(parents=True, exist_ok=True)
        adaptive_lbl_dir.mkdir(parents=True, exist_ok=True)
        self._create_yaml(output_dir / 'adaptive', 'adaptive')
        
        if create_all_sr_variant:
            all_sr_img_dir = output_dir / 'all_sr' / 'images' / split_name
            all_sr_lbl_dir = output_dir / 'all_sr' / 'labels' / split_name
            all_sr_img_dir.mkdir(parents=True, exist_ok=True)
            all_sr_lbl_dir.mkdir(parents=True, exist_ok=True)
            self._create_yaml(output_dir / 'all_sr', 'all_sr')

        image_files = list(input_img_dir.glob('*.jpg')) + list(input_img_dir.glob('*.png'))
        
        results = []
        for img_path in tqdm(image_files, desc=f"Processing {split_name}"):
            try:
                img = cv2.imread(str(img_path))
                if img is None:
                    warnings.warn(f"Failed to read image {img_path}. Skipping.")
                    continue
                
                label_path = input_label_dir / f"{img_path.stem}.txt"
                
                # Adaptive Processing
                adaptive_img, metadata = self.process_single(img, label_path)
                cv2.imwrite(str(adaptive_img_dir / img_path.name), adaptive_img)
                if label_path.exists():
                    shutil.copy2(label_path, adaptive_lbl_dir / label_path.name)
                
                # All SR Processing
                if create_all_sr_variant:
                    all_sr_img = self.sr.enhance(img)
                    all_sr_img = cv2.resize(all_sr_img, self.target_size, interpolation=cv2.INTER_LINEAR)
                    cv2.imwrite(str(all_sr_img_dir / img_path.name), all_sr_img)
                    if label_path.exists():
                        shutil.copy2(label_path, all_sr_lbl_dir / label_path.name)
                        
                results.append({
                    'image_name': img_path.name,
                    'needs_sr': metadata['needs_sr'],
                    'total_defects': metadata['defects_analyzed'],
                    'micro_defects': metadata['micro_defects'],
                    'defect_classes': metadata['defect_classes']
                })
                
            except Exception as e:
                warnings.warn(f"Error processing {img_path}: {str(e)}")
                
        df = pd.DataFrame(results)
        df.to_csv(output_dir / 'routing_statistics.csv', index=False)
        return df

    def generate_report(self, stats_df: pd.DataFrame, output_dir: Path):
        """Generate summary report"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print("\n=== Preprocessing Pipeline Report ===")
        print(f"Total images processed: {len(stats_df)}")
        if len(stats_df) > 0:
            sr_needed = stats_df['needs_sr'].sum()
            print(f"Images requiring SR: {sr_needed} ({sr_needed/len(stats_df)*100:.1f}%)")
            
            # 1. Routing breakdown table
            all_classes = []
            needs_sr_classes = []
            for _, row in stats_df.iterrows():
                classes = row.get('defect_classes', [])
                if isinstance(classes, str):
                    try:
                        classes = eval(classes)
                    except:
                        classes = []
                all_classes.extend(classes)
                if row['needs_sr']:
                    needs_sr_classes.extend(classes)
            
            class_counts = pd.Series(all_classes).value_counts().sort_index()
            sr_class_counts = pd.Series(needs_sr_classes).value_counts().sort_index()
            
            breakdown_df = pd.DataFrame({
                'Total': class_counts,
                'Needs_SR': sr_class_counts
            }).fillna(0).astype(int)
            breakdown_df.index.name = 'Class_ID'
            breakdown_df.to_csv(output_dir / 'class_routing_breakdown.csv')
            print("\nRouting breakdown saved to 'class_routing_breakdown.csv'")
            
            # 2. Routing distribution bar chart
            plt.figure(figsize=(8, 6))
            plt.bar(['Needs SR', 'Pass-through'], [sr_needed, len(stats_df) - sr_needed], color=['#ff9999','#66b3ff'])
            plt.title('Adaptive Routing Distribution')
            plt.ylabel('Number of Images')
            plt.savefig(output_dir / 'routing_distribution.png', dpi=300, bbox_inches='tight')
            plt.close()
            print("Routing distribution chart saved to 'routing_distribution.png'")

if __name__ == '__main__':
    # Setup paths
    base_dir = Path(r"C:\Users\Abhinav\Documents\VIT\4th Year\Projects\FOD_Predictive_Detection")
    dataset_dir = base_dir / "Dataset" / "DsPCBSD+" / "Data_YOLO"
    input_img_dir = dataset_dir / "images" / "train"
    input_label_dir = dataset_dir / "labels" / "train"
    
    output_dir = base_dir / "ai_workplace" / "results" / "pipeline_test"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    pipeline = AdaptivePreprocessingPipeline(
        sr_method='enhanced_bicubic',
        sr_scale=2,
        target_size=(640, 640),
        area_threshold=0.005
    )
    
    if not input_img_dir.exists():
        print(f"Warning: Dataset path {input_img_dir} does not exist.")
        print("Creating mock dataset for testing...")
        input_img_dir.mkdir(parents=True, exist_ok=True)
        input_label_dir.mkdir(parents=True, exist_ok=True)
        
        for i in range(5):
            img = np.random.randint(0, 255, (400, 400, 3), dtype=np.uint8)
            cv2.imwrite(str(input_img_dir / f"test_{i}.jpg"), img)
            with open(input_label_dir / f"test_{i}.txt", 'w') as f:
                f.write("0 0.5 0.5 0.05 0.05\n") # small defect -> should trigger SR
                if i % 2 == 0:
                    f.write("1 0.2 0.2 0.2 0.2\n") # large defect
    
    # Process only first 30 images for demo
    image_files = list(input_img_dir.glob('*.jpg')) + list(input_img_dir.glob('*.png'))
    demo_files = image_files[:30]
    
    print(f"Running pipeline demo on {len(demo_files)} images...")
    
    # Create temporary directories for demo to use the standard process_dataset workflow
    demo_img_dir = output_dir / "demo_input" / "images" / "train"
    demo_lbl_dir = output_dir / "demo_input" / "labels" / "train"
    demo_img_dir.mkdir(parents=True, exist_ok=True)
    demo_lbl_dir.mkdir(parents=True, exist_ok=True)
    
    for f in demo_files:
        shutil.copy2(f, demo_img_dir / f.name)
        lbl_file = input_label_dir / f"{f.stem}.txt"
        if lbl_file.exists():
            shutil.copy2(lbl_file, demo_lbl_dir / lbl_file.name)
            
    stats_df = pipeline.process_dataset(
        input_img_dir=demo_img_dir,
        input_label_dir=demo_lbl_dir,
        output_dir=output_dir,
        create_all_sr_variant=True
    )
    
    pipeline.generate_report(stats_df, output_dir)
    
    # 3. Before/after SR comparison grid (PNG) for 4 sample micro-defect images
    micro_df = stats_df[stats_df['needs_sr'] == True]
    sample_names = micro_df['image_name'].head(4).tolist()
    
    if len(sample_names) > 0:
        fig, axes = plt.subplots(len(sample_names), 2, figsize=(10, 5 * len(sample_names)))
        if len(sample_names) == 1:
            axes = [axes]
        
        for i, name in enumerate(sample_names):
            orig_img = cv2.imread(str(demo_img_dir / name))
            orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
            
            sr_img = cv2.imread(str(output_dir / 'adaptive' / 'images' / 'train' / name))
            sr_img = cv2.cvtColor(sr_img, cv2.COLOR_BGR2RGB)
            
            axes[i][0].imshow(orig_img)
            axes[i][0].set_title(f'Original: {name}')
            axes[i][0].axis('off')
            
            axes[i][1].imshow(sr_img)
            axes[i][1].set_title(f'Adaptive SR + Resize: {name}')
            axes[i][1].axis('off')
            
        plt.tight_layout()
        plt.savefig(output_dir / 'sr_comparison_grid.png', dpi=300)
        plt.close()
        print("SR comparison grid saved to 'sr_comparison_grid.png'")
    else:
        print("No images with micro-defects found to generate comparison grid.")
        
    print(f"Results saved to: {output_dir}")
