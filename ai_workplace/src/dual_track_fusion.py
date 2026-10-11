import time
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union

import cv2
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

try:
    from ensemble_boxes import weighted_boxes_fusion
except ImportError:
    # Native NumPy implementation of Weighted Box Fusion (Solovyev et al., 2021)
    def weighted_boxes_fusion(boxes_list, scores_list, labels_list, weights=None, iou_thr=0.5, skip_box_thr=0.0):
        if len(boxes_list) == 0:
            return np.array([]), np.array([]), np.array([])
        if weights is None:
            weights = [1.0] * len(boxes_list)
        weights = np.array(weights, dtype=float)

        all_labels = set()
        for l in labels_list:
            all_labels.update(l)

        overall_boxes, overall_scores, overall_labels = [], [], []

        for label in all_labels:
            boxes, scores, model_indices = [], [], []
            for m_idx, (b_m, s_m, l_m) in enumerate(zip(boxes_list, scores_list, labels_list)):
                for b, s, l in zip(b_m, s_m, l_m):
                    if l == label and s >= skip_box_thr:
                        boxes.append(b)
                        scores.append(s)
                        model_indices.append(m_idx)

            if not boxes:
                continue

            boxes = np.array(boxes)
            scores = np.array(scores)
            model_indices = np.array(model_indices)

            order = np.argsort(-scores)
            boxes = boxes[order]
            scores = scores[order]
            model_indices = model_indices[order]

            clusters = []
            for i in range(len(boxes)):
                box = boxes[i]
                matched = False
                for cluster in clusters:
                    cluster_boxes = boxes[cluster]
                    ious = [bbox_iou(box, cb) for cb in cluster_boxes]
                    if max(ious) > iou_thr:
                        cluster.append(i)
                        matched = True
                        break
                if not matched:
                    clusters.append([i])

            for cluster in clusters:
                c_boxes = boxes[cluster]
                c_scores = scores[cluster]
                c_weights = weights[model_indices[cluster]]

                w_scores = c_scores * c_weights
                sum_w = np.sum(w_scores)

                fused_box = np.sum(c_boxes * w_scores[:, None], axis=0) / (sum_w + 1e-8)
                fused_score = min(1.0, sum_w / np.sum(weights))

                overall_boxes.append(fused_box)
                overall_scores.append(fused_score)
                overall_labels.append(label)

        return np.array(overall_boxes), np.array(overall_scores), np.array(overall_labels)

# Note: In actual usage, ensure ultralytics is installed.
try:
    from ultralytics import YOLO, RTDETR
except ImportError:
    print("Warning: ultralytics is not installed. Please install it using 'pip install ultralytics'")

CLASS_NAMES = {0: 'SH', 1: 'SP', 2: 'SC', 3: 'OP', 4: 'MB', 5: 'HB', 6: 'CS', 7: 'CFO', 8: 'BMFO'}


def bbox_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    """Compute IoU between two bounding boxes [x1, y1, x2, y2]."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    if inter_area == 0:
        return 0.0
    
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


class DualTrackFusion:
    def __init__(self, 
                 yolo_weights: str,
                 detr_weights: str,
                 img_size: int = 640,
                 device: str = 'cpu',
                 conf_threshold: float = 0.25,
                 yolo_weight: float = 1.0,
                 detr_weight: float = 1.5,
                 iou_threshold: float = 0.5):
        """Load both models."""
        from ultralytics import YOLO, RTDETR
        self.yolo_model = YOLO(yolo_weights)
        self.detr_model = RTDETR(detr_weights)
        
        self.img_size = img_size
        self.device = device
        self.conf_threshold = conf_threshold
        self.yolo_weight = yolo_weight
        self.detr_weight = detr_weight
        self.iou_threshold = iou_threshold

    def _process_predictions(self, results, width: int, height: int) -> Tuple[List[List[float]], List[float], List[int]]:
        """Extract boxes, scores, labels and normalize boxes to [0,1]."""
        if not results or len(results) == 0:
            return [], [], []
        
        boxes_list = []
        scores_list = []
        labels_list = []
        
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return [], [], []
            
        for box in boxes:
            # xyxy format
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0].cpu().numpy())
            cls = int(box.cls[0].cpu().numpy())
            
            if conf >= self.conf_threshold:
                # Normalize to 0-1
                nx1, nx2 = np.clip([x1/width, x2/width], 0, 1)
                ny1, ny2 = np.clip([y1/height, y2/height], 0, 1)
                
                # Check for valid box dimensions
                if nx1 < nx2 and ny1 < ny2:
                    boxes_list.append([nx1, ny1, nx2, ny2])
                    scores_list.append(conf)
                    labels_list.append(cls)
                
        return boxes_list, scores_list, labels_list

    def predict_single(self, image_path: str) -> dict:
        """Run both models and fuse. Returns dict with results."""
        img = cv2.imread(str(image_path))
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
        h, w = img.shape[:2]
        
        # Run inference
        yolo_results = self.yolo_model.predict(source=image_path, imgsz=self.img_size, device=self.device, verbose=False)
        detr_results = self.detr_model.predict(source=image_path, imgsz=self.img_size, device=self.device, verbose=False)
        
        y_boxes, y_scores, y_labels = self._process_predictions(yolo_results, w, h)
        d_boxes, d_scores, d_labels = self._process_predictions(detr_results, w, h)
        
        # Prepare for WBF
        boxes_list = []
        scores_list = []
        labels_list = []
        weights = []
        
        if len(y_boxes) > 0:
            boxes_list.append(y_boxes)
            scores_list.append(y_scores)
            labels_list.append(y_labels)
            weights.append(self.yolo_weight)
            
        if len(d_boxes) > 0:
            boxes_list.append(d_boxes)
            scores_list.append(d_scores)
            labels_list.append(d_labels)
            weights.append(self.detr_weight)
            
        # Perform Fusion
        fused_boxes = []
        fused_scores = []
        fused_labels = []
        
        if len(boxes_list) > 0:
            fused_b, fused_s, fused_l = weighted_boxes_fusion(
                boxes_list, 
                scores_list, 
                labels_list, 
                weights=weights, 
                iou_thr=self.iou_threshold, 
                skip_box_thr=0.0
            )
            
            # De-normalize
            for i, box in enumerate(fused_b):
                x1, y1, x2, y2 = box
                fused_boxes.append([x1*w, y1*h, x2*w, y2*h])
                fused_scores.append(float(fused_s[i]))
                fused_labels.append(int(fused_l[i]))
                
        # Also de-normalize yolo and detr for output dict
        denorm_y_boxes = [[b[0]*w, b[1]*h, b[2]*w, b[3]*h] for b in y_boxes]
        denorm_d_boxes = [[b[0]*w, b[1]*h, b[2]*w, b[3]*h] for b in d_boxes]

        return {
            'yolo_boxes': denorm_y_boxes, 'yolo_scores': y_scores, 'yolo_labels': y_labels,
            'detr_boxes': denorm_d_boxes, 'detr_scores': d_scores, 'detr_labels': d_labels,
            'fused_boxes': fused_boxes, 'fused_scores': fused_scores, 'fused_labels': fused_labels,
            'yolo_count': len(y_boxes), 'detr_count': len(d_boxes), 'fused_count': len(fused_boxes)
        }

    def predict_batch(self, image_dir: Path, limit: int = None) -> pd.DataFrame:
        """Run fusion on batch of images. Returns per-image results."""
        image_paths = list(Path(image_dir).glob("*.jpg")) + list(Path(image_dir).glob("*.png"))
        if limit:
            image_paths = image_paths[:limit]
            
        results = []
        for p in image_paths:
            print(f"Processing {p.name}...")
            res = self.predict_single(str(p))
            results.append({
                'image_name': p.name,
                'yolo_count': res['yolo_count'],
                'detr_count': res['detr_count'],
                'fused_count': res['fused_count']
            })
            
        return pd.DataFrame(results)

    def _load_labels(self, label_path: Path, width: int, height: int) -> List[dict]:
        """Load YOLO format labels and convert to xyxy."""
        gt_boxes = []
        if not label_path.exists():
            return gt_boxes
            
        with open(label_path, 'r') as f:
            lines = f.readlines()
            
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                cls_id = int(parts[0])
                cx, cy, bw, bh = map(float, parts[1:5])
                
                # YOLO format is normalized [cx, cy, w, h]
                x1 = (cx - bw/2) * width
                y1 = (cy - bh/2) * height
                x2 = (cx + bw/2) * width
                y2 = (cy + bh/2) * height
                
                gt_boxes.append({'box': [x1, y1, x2, y2], 'label': cls_id})
                
        return gt_boxes

    def grid_search_weights(self, val_image_dir: Path, val_label_dir: Path,
                            weight_combos: list = None, iou_thresholds: list = None,
                            limit: int = 100) -> pd.DataFrame:
        """Grid search optimal WBF parameters."""
        if weight_combos is None:
            weight_combos = [[1.0, 1.0], [1.0, 1.5], [1.5, 1.0], [1.0, 2.0], [2.0, 1.0]]
        if iou_thresholds is None:
            iou_thresholds = [0.4, 0.5, 0.6]
            
        image_paths = list(Path(val_image_dir).glob("*.jpg")) + list(Path(val_image_dir).glob("*.png"))
        if limit:
            image_paths = image_paths[:limit]
            
        print(f"Grid search over {len(image_paths)} images...")
        
        # Pre-compute single model predictions to save time
        img_data = []
        for p in image_paths:
            img = cv2.imread(str(p))
            if img is None: continue
            h, w = img.shape[:2]
            
            label_p = val_label_dir / f"{p.stem}.txt"
            gt = self._load_labels(label_p, w, h)
            
            yr = self.yolo_model.predict(source=str(p), imgsz=self.img_size, device=self.device, verbose=False)
            dr = self.detr_model.predict(source=str(p), imgsz=self.img_size, device=self.device, verbose=False)
            
            yb, ys, yl = self._process_predictions(yr, w, h)
            db, ds, dl = self._process_predictions(dr, w, h)
            
            img_data.append({
                'path': p, 'w': w, 'h': h, 'gt': gt,
                'yb': yb, 'ys': ys, 'yl': yl,
                'db': db, 'ds': ds, 'dl': dl
            })
            
        results = []
        
        for w_yolo, w_detr in weight_combos:
            for iou_thr in iou_thresholds:
                print(f"Testing weights YOLO:{w_yolo}, DETR:{w_detr} | WBF IoU:{iou_thr}")
                
                tp = 0
                fp = 0
                fn = 0
                
                for data in img_data:
                    b_list, s_list, l_list, w_list = [], [], [], []
                    if len(data['yb']) > 0:
                        b_list.append(data['yb'])
                        s_list.append(data['ys'])
                        l_list.append(data['yl'])
                        w_list.append(w_yolo)
                    if len(data['db']) > 0:
                        b_list.append(data['db'])
                        s_list.append(data['ds'])
                        l_list.append(data['dl'])
                        w_list.append(w_detr)
                        
                    fused_b, fused_s, fused_l = [], [], []
                    if len(b_list) > 0:
                        fused_b, fused_s, fused_l = weighted_boxes_fusion(
                            b_list, s_list, l_list, weights=w_list, iou_thr=iou_thr, skip_box_thr=0.0
                        )
                        
                    # De-normalize predictions
                    preds = []
                    for i, box in enumerate(fused_b):
                        x1, y1, x2, y2 = box
                        preds.append({
                            'box': [x1*data['w'], y1*data['h'], x2*data['w'], y2*data['h']],
                            'label': int(fused_l[i])
                        })
                        
                    # Evaluate tp, fp, fn per image
                    gt_unmatched = list(range(len(data['gt'])))
                    
                    for p_box in preds:
                        best_iou = 0
                        best_gt_idx = -1
                        
                        for idx in gt_unmatched:
                            gt_box = data['gt'][idx]
                            if p_box['label'] == gt_box['label']:
                                iou = bbox_iou(p_box['box'], gt_box['box'])
                                if iou > best_iou:
                                    best_iou = iou
                                    best_gt_idx = idx
                                    
                        if best_iou > 0.5:
                            tp += 1
                            gt_unmatched.remove(best_gt_idx)
                        else:
                            fp += 1
                            
                    fn += len(gt_unmatched)
                    
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                results.append({
                    'yolo_weight': w_yolo,
                    'detr_weight': w_detr,
                    'wbf_iou_thr': iou_thr,
                    'precision': precision,
                    'recall': recall,
                    'f1_score': f1
                })
                
        return pd.DataFrame(results)

    def visualize_comparison(self, image_path: str, save_path: Path):
        """Create 1x3 figure: YOLO-only | DETR-only | Fused detections."""
        res = self.predict_single(image_path)
        img = cv2.imread(str(image_path))
        if img is None:
            return
            
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        def draw_boxes(image, boxes, labels, scores, title):
            viz = image.copy()
            for box, label, score in zip(boxes, labels, scores):
                x1, y1, x2, y2 = map(int, box)
                cls_name = CLASS_NAMES.get(label, str(label))
                cv2.rectangle(viz, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(viz, f"{cls_name} {score:.2f}", (x1, max(y1-5, 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            return viz
            
        img_yolo = draw_boxes(img, res['yolo_boxes'], res['yolo_labels'], res['yolo_scores'], "YOLO")
        img_detr = draw_boxes(img, res['detr_boxes'], res['detr_labels'], res['detr_scores'], "DETR")
        img_fused = draw_boxes(img, res['fused_boxes'], res['fused_labels'], res['fused_scores'], "Fused")
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=300)
        axes[0].imshow(img_yolo)
        axes[0].set_title(f"YOLO (N={res['yolo_count']})")
        axes[0].axis('off')
        
        axes[1].imshow(img_detr)
        axes[1].set_title(f"RT-DETR (N={res['detr_count']})")
        axes[1].axis('off')
        
        axes[2].imshow(img_fused)
        axes[2].set_title(f"Fused WBF (N={res['fused_count']})")
        axes[2].axis('off')
        
        plt.tight_layout()
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
        plt.close()

    def benchmark_latency(self, image_path: str, num_runs: int = 20) -> dict:
        """Measure latency for models and fusion overhead."""
        print(f"Benchmarking on {image_path} with {num_runs} runs...")
        
        # Warmup
        self.predict_single(image_path)
        
        times = {'yolo': [], 'detr': [], 'sequential': [], 'fusion_overhead': [], 'total': []}
        
        for i in range(num_runs):
            t0 = time.time()
            self.yolo_model.predict(source=image_path, imgsz=self.img_size, device=self.device, verbose=False)
            t1 = time.time()
            times['yolo'].append((t1-t0)*1000)
            
            t2 = time.time()
            self.detr_model.predict(source=image_path, imgsz=self.img_size, device=self.device, verbose=False)
            t3 = time.time()
            times['detr'].append((t3-t2)*1000)
            
            t4 = time.time()
            self.predict_single(image_path)
            t5 = time.time()
            times['total'].append((t5-t4)*1000)
            times['sequential'].append(times['yolo'][-1] + times['detr'][-1])
            times['fusion_overhead'].append(max(0, times['total'][-1] - times['sequential'][-1]))
            
        results = {k: np.mean(v) for k, v in times.items()}
        results['runs'] = num_runs
        return results

if __name__ == '__main__':
    # Demo code
    proj_dir = Path(r"C:\Users\Abhinav\Documents\VIT\4th Year\Projects\FOD_Predictive_Detection")
    
    # Auto-discover YOLO weights
    yolo_candidates = [
        proj_dir / "MODEL RESULTS/YOLO/best.pt",
        proj_dir / "ai_workplace/weights/yolo26n_best.pt",
        proj_dir / "ai_workplace/weights/yolo26n.pt",
        proj_dir / "runs/YOLO26n_baseline/weights/best.pt",
        proj_dir / "yolo26n.pt"
    ]
    yolo_w = next((p for p in yolo_candidates if p.exists()), None)
    
    # Auto-discover RT-DETR weights
    detr_candidates = [
        proj_dir / "MODEL RESULTS/RETR/results/runs/detect/PCB_RTDETR_L_Run/weights/best.pt",
        proj_dir / "MODEL RESULTS/RETR/results/rtdetr-l.pt",
        proj_dir / "ai_workplace/weights/rtdetr_l_best.pt",
        proj_dir / "ai_workplace/weights/rtdetr-l.pt"
    ]
    detr_w = next((p for p in detr_candidates if p.exists()), None)
    
    if not yolo_w or not detr_w:
        print("Weights check:")
        print(f"  YOLO weight:    {yolo_w} (found: {yolo_w is not None})")
        print(f"  RT-DETR weight: {detr_w} (found: {detr_w is not None})")
        print("Please ensure both trained models are available.")
    else:
        print("Models found!")
        print(f"  YOLO:    {yolo_w}")
        print(f"  RT-DETR: {detr_w}")
        print("Initializing Dual Track Fusion...")
        fusion = DualTrackFusion(
            yolo_weights=str(yolo_w),
            detr_weights=str(detr_w),
            device='cpu' # fallback to CPU for demo if CUDA not forced
        )
        
        val_img_dir = proj_dir / "Dataset/DsPCBSD+/Data_YOLO/images/val"
        if val_img_dir.exists():
            val_images = list(val_img_dir.glob("*.jpg"))[:3]
            
            results_dir = proj_dir / "ai_workplace/results"
            results_dir.joinpath("tables").mkdir(parents=True, exist_ok=True)
            results_dir.joinpath("figures").mkdir(parents=True, exist_ok=True)
            
            # Predict single
            for img_p in val_images:
                res = fusion.predict_single(str(img_p))
                print(f"Image {img_p.name}: YOLO={res['yolo_count']}, DETR={res['detr_count']}, Fused={res['fused_count']}")
                
                # Visualize
                viz_path = results_dir / f"figures/fusion_comparison_{img_p.stem}.png"
                fusion.visualize_comparison(str(img_p), viz_path)
                print(f"Saved visualization to {viz_path}")
            
            if val_images:
                # Benchmark
                bench = fusion.benchmark_latency(str(val_images[0]), num_runs=5)
                print(f"Benchmark results: {bench}")
                
                # Save bench results to table
                df_bench = pd.DataFrame([bench])
                bench_csv = results_dir / "tables/fusion_results.csv"
                df_bench.to_csv(bench_csv, index=False)
                print(f"Saved benchmark results to {bench_csv}")
        else:
            print(f"Validation image directory not found: {val_img_dir}")
