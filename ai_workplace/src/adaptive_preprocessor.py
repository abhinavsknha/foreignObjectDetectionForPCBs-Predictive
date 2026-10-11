import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, Tuple, List, Dict
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

CLASS_NAMES = {
    0: 'SH', 1: 'SP', 2: 'SC', 3: 'OP', 
    4: 'MB', 5: 'HB', 6: 'CS', 7: 'CFO', 8: 'BMFO'
}

def parse_yolo_labels(label_path: Path) -> List[Dict]:
    """Parse YOLO format label file."""
    labels = []
    if not label_path.exists():
        return labels
    
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                labels.append({
                    'class_id': int(parts[0]),
                    'xc': float(parts[1]),
                    'yc': float(parts[2]),
                    'w': float(parts[3]),
                    'h': float(parts[4])
                })
    return labels

class DefectScaleRouter:
    def __init__(self, area_threshold=0.005, edge_threshold=0.03, contrast_threshold=25.0, hf_threshold=0.7):
        self.area_threshold = area_threshold
        self.edge_threshold = edge_threshold
        self.contrast_threshold = contrast_threshold
        self.hf_threshold = hf_threshold

    def route_single(self, image: np.ndarray, label_path: Optional[Path] = None) -> Tuple[bool, dict]:
        """
        Determine if SR is needed.
        Returns (needs_sr, metadata)
        """
        metadata = {}
        needs_sr = False
        
        if label_path is not None and label_path.exists():
            # Training Mode
            labels = parse_yolo_labels(label_path)
            metadata['mode'] = 'training'
            metadata['classes'] = [lbl['class_id'] for lbl in labels]
            
            if labels:
                areas = [lbl['w'] * lbl['h'] for lbl in labels]
                min_area = min(areas)
                avg_area = sum(areas) / len(areas)
                metadata['min_area_ratio'] = min_area
                metadata['avg_area_ratio'] = avg_area
                
                if min_area < self.area_threshold:
                    needs_sr = True
            else:
                metadata['min_area_ratio'] = 1.0
                metadata['avg_area_ratio'] = 1.0
                
        else:
            # Inference Mode
            metadata['mode'] = 'inference'
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
                
            h, w = gray.shape
            
            # Edge density
            edges = cv2.Canny(gray, 50, 150)
            edge_density = np.count_nonzero(edges) / (h * w)
            metadata['edge_density'] = edge_density
            
            # Local contrast variance
            blurred = cv2.GaussianBlur(gray, (31, 31), 0)
            diff = cv2.absdiff(gray, blurred)
            contrast_var = np.std(diff)
            metadata['contrast_var'] = contrast_var
            
            # FFT high frequency energy ratio
            f = np.fft.fft2(gray)
            fshift = np.fft.fftshift(f)
            magnitude_spectrum = np.abs(fshift) ** 2
            
            cy, cx = h // 2, w // 2
            radius = min(h, w) // 8
            
            y, x = np.ogrid[:h, :w]
            mask = (x - cx)**2 + (y - cy)**2 <= radius**2
            
            low_freq_energy = np.sum(magnitude_spectrum[mask])
            total_energy = np.sum(magnitude_spectrum)
            
            if total_energy == 0:
                hf_ratio = 0
            else:
                hf_ratio = (total_energy - low_freq_energy) / total_energy
            metadata['hf_ratio'] = hf_ratio
            
            if (edge_density < self.edge_threshold and contrast_var < self.contrast_threshold) or (hf_ratio < self.hf_threshold):
                needs_sr = True
                
        metadata['needs_sr'] = needs_sr
        return needs_sr, metadata

    def route_batch(self, image_dir: Path, label_dir: Optional[Path] = None) -> pd.DataFrame:
        """Process directory, return DataFrame with routing decisions"""
        results = []
        image_paths = list(image_dir.glob('*.jpg')) + list(image_dir.glob('*.png'))
        
        for img_path in image_paths:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
                
            lbl_path = None
            if label_dir is not None:
                lbl_path = label_dir / f"{img_path.stem}.txt"
                
            needs_sr, metadata = self.route_single(img, lbl_path)
            
            row = {'image_name': img_path.name, 'needs_sr': needs_sr}
            row.update(metadata)
            results.append(row)
            
        return pd.DataFrame(results)

    def compute_routing_statistics(self, df: pd.DataFrame) -> dict:
        """Compute summary stats: % routed to SR, per-class routing breakdown"""
        stats = {}
        if df.empty:
            return stats
            
        total = len(df)
        sr_count = df['needs_sr'].sum()
        stats['total_images'] = total
        stats['sr_routed'] = int(sr_count)
        stats['sr_percentage'] = (sr_count / total) * 100 if total > 0 else 0
        
        if 'classes' in df.columns:
            class_counts = {c: {'total': 0, 'sr': 0} for c in CLASS_NAMES.keys()}
            for _, row in df.iterrows():
                classes = row.get('classes', [])
                if isinstance(classes, list):
                    for c in classes:
                        class_counts[c]['total'] += 1
                        if row['needs_sr']:
                            class_counts[c]['sr'] += 1
            stats['class_breakdown'] = class_counts
            
        return stats

    def visualize_routing(self, image: np.ndarray, metadata: dict, save_path: Optional[Path] = None):
        """Save side-by-side visualization showing metrics and routing decision"""
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        # Convert BGR to RGB
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title('Image')
        plt.axis('off')
        
        plt.subplot(1, 2, 2)
        plt.axis('off')
        
        text = f"Needs SR: {metadata.get('needs_sr', False)}\n"
        text += f"Mode: {metadata.get('mode', 'N/A')}\n"
        
        if metadata.get('mode') == 'training':
            text += f"Min Area Ratio: {metadata.get('min_area_ratio', 1.0):.4f}\n"
            text += f"Avg Area Ratio: {metadata.get('avg_area_ratio', 1.0):.4f}\n"
        else:
            text += f"Edge Density: {metadata.get('edge_density', 0.0):.4f}\n"
            text += f"Contrast Var: {metadata.get('contrast_var', 0.0):.2f}\n"
            text += f"HF Ratio: {metadata.get('hf_ratio', 0.0):.4f}\n"
            
        plt.text(0.1, 0.5, text, fontsize=12, va='center')
        
        if save_path:
            save_path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()

if __name__ == '__main__':
    base_dir = Path(r"C:\Users\Abhinav\Documents\VIT\4th Year\Projects\FOD_Predictive_Detection")
    img_dir = base_dir / "Dataset" / "DsPCBSD+" / "Data_YOLO" / "images" / "train"
    lbl_dir = base_dir / "Dataset" / "DsPCBSD+" / "Data_YOLO" / "labels" / "train"
    
    out_dir = base_dir / "ai_workplace" / "results"
    (out_dir / "tables").mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    
    router = DefectScaleRouter()
    
    # Process only the first 20 images for demonstration
    results = []
    if img_dir.exists():
        img_paths = list(img_dir.glob('*.jpg'))[:20]
        logging.info(f"Processing {len(img_paths)} images...")
        
        for img_path in img_paths:
            img = cv2.imread(str(img_path))
            if img is not None:
                lbl_path = lbl_dir / f"{img_path.stem}.txt"
                needs_sr, metadata = router.route_single(img, lbl_path)
                
                row = {'image_name': img_path.name, 'needs_sr': needs_sr}
                row.update(metadata)
                results.append(row)
                
        if results:
            df = pd.DataFrame(results)
            csv_path = out_dir / "tables" / "routing_statistics.csv"
            
            # Since classes is a list, we might want to drop or stringify it for CSV
            df_csv = df.copy()
            if 'classes' in df_csv.columns:
                df_csv['classes'] = df_csv['classes'].apply(lambda x: str(x) if isinstance(x, list) else x)
                
            df_csv.to_csv(csv_path, index=False)
            logging.info(f"Saved statistics to {csv_path}")
            
            stats = router.compute_routing_statistics(df)
            logging.info(f"Stats: {stats}")
            
            if 'class_breakdown' in stats:
                classes = []
                sr_counts = []
                direct_counts = []
                
                for cid, cdata in stats['class_breakdown'].items():
                    if cdata['total'] > 0:
                        classes.append(CLASS_NAMES[cid])
                        sr_counts.append(cdata['sr'])
                        direct_counts.append(cdata['total'] - cdata['sr'])
                        
                if classes:
                    x = np.arange(len(classes))
                    width = 0.35
                    
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.bar(x - width/2, sr_counts, width, label='SR Routed')
                    ax.bar(x + width/2, direct_counts, width, label='Direct')
                    
                    ax.set_ylabel('Count')
                    ax.set_title('Routing Distribution by Class')
                    ax.set_xticks(x)
                    ax.set_xticklabels(classes)
                    ax.legend()
                    
                    fig_path = out_dir / "figures" / "routing_distribution.png"
                    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    logging.info(f"Saved distribution plot to {fig_path}")
    else:
        logging.error(f"Image directory not found: {img_dir}")
