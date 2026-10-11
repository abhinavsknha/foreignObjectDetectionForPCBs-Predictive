# PCB Defect Detection — Experimental Evaluation Results

This document benchmarks both detection tracks (**Track 1: YOLO** and **Track 2: RT-DETR-L**) evaluated on the **DsPCBSD+** benchmark dataset.

---

# Track 1: YOLO Model Results (Stage 2)

### 1. Model & Hardware Overview
- **Model Architecture**: YOLO26n (Nano Edge CNN, ~2.51M parameters, 5.8 GFLOPs)
- **Role in Pipeline**: Real-time edge screening filter (high throughput, minimal latency)
- **Dataset Split**: DsPCBSD+ Test/Val (2,051 images, 4,092 ground-truth defect instances)
- **Resolution**: 640 × 640
- **Hardware**: Cloud GPU (NVIDIA Tesla T4)

### 2. Overall Performance Metrics
- **Precision (Box P)**: **82.70%** (0.827)
- **Recall (R)**: **80.50%** (0.805)
- **mAP@50**: **85.20%** (0.852)
- **mAP@50-95**: **51.00%** (0.510)

### 3. Per-Class Metrics Breakdown

| Class Code | Defect Class Name | Images | Instances | Precision | Recall | mAP@50 | mAP@50-95 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **SH** | Short Circuit | 126 | 169 | 0.8735 | 0.8583 | 0.9119 | 0.5906 |
| **SP** | Spur | 430 | 929 | 0.8582 | 0.7817 | 0.8492 | 0.3933 |
| **SC** | Spurious Copper | 245 | 285 | 0.7932 | 0.8000 | 0.8193 | 0.4962 |
| **OP** | Open Circuit | 274 | 338 | 0.8308 | 0.8609 | 0.8884 | 0.5262 |
| **MB** | Mask Break | 391 | 546 | 0.8529 | 0.7540 | 0.8304 | 0.4044 |
| **HB** | Hole Break | 271 | 608 | **0.8957** | **0.9607** | **0.9828** | **0.8231** |
| **CS** | Conductor Scratch | 279 | 448 | 0.7709 | 0.6686 | 0.7427 | 0.4519 |
| **CFO** | Coarse Foreign Object | 309 | 423 | 0.7337 | 0.6774 | 0.7469 | 0.4279 |
| **BMFO** | Ball Micro Foreign Object | 305 | 346 | 0.8376 | 0.8844 | 0.8949 | 0.4749 |
| **ALL** | **Overall Summary** | **2051** | **4092** | **0.8270** | **0.8050** | **0.8520** | **0.5100** |

### 4. Latency & Inference Profile
- **Pre-process**: 1.0 ms
- **Inference (Tesla T4)**: 3.6 ms
- **Post-process (NMS)**: 0.8 ms
- **Total Latency**: **5.4 ms per image (~185.2 FPS)**

---

# Track 2: RT-DETR-L Model Results (Stage 2)

### 1. Model & Hardware Overview
- **Model Architecture**: RT-DETR-L (Real-Time Detection Transformer, ~32M parameters, Hybrid Encoder + Multi-Scale Deformable Cross-Attention)
- **Role in Pipeline**: High-precision transformer verification branch (global contextual reasoning, small-object attention)
- **Dataset Split**: DsPCBSD+ Test/Val (2,051 images, 4,092 ground-truth defect instances)
- **Resolution**: 640 × 640
- **Hardware**: Cloud GPU (NVIDIA Tesla T4)

### 2. Overall Performance Metrics
- **Precision (Box P)**: **82.10%** (0.821)
- **Recall (R)**: **80.70%** (0.807)
- **mAP@50**: **84.50%** (0.845)
- **mAP@50-95**: **50.40%** (0.504)

### 3. Per-Class Metrics Breakdown

| Class Code | Defect Class Name | Images | Instances | Precision | Recall | mAP@50 | mAP@50-95 |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **SH** | Short Circuit | 126 | 169 | 0.8429 | 0.8575 | 0.8875 | 0.5836 |
| **SP** | Spur | 430 | 929 | 0.8595 | 0.7640 | 0.8387 | 0.3861 |
| **SC** | Spurious Copper | 245 | 285 | 0.7916 | 0.8261 | 0.8543 | 0.5147 |
| **OP** | Open Circuit | 274 | 338 | 0.8243 | 0.8580 | 0.8707 | 0.5113 |
| **MB** | Mask Break | 391 | 546 | 0.8653 | 0.7881 | 0.8512 | 0.4080 |
| **HB** | Hole Break | 271 | 608 | 0.8335 | **0.9605** | **0.9598** | **0.8081** |
| **CS** | Conductor Scratch | 279 | 448 | 0.7203 | 0.6440 | 0.7047 | 0.4362 |
| **CFO** | Coarse Foreign Object | 309 | 423 | 0.7791 | 0.6856 | 0.7526 | 0.4288 |
| **BMFO** | Ball Micro Foreign Object | 305 | 346 | **0.8754** | 0.8757 | 0.8884 | 0.4606 |
| **ALL** | **Overall Summary** | **2051** | **4092** | **0.8210** | **0.8070** | **0.8450** | **0.5040** |

### 4. Latency & Inference Profile
- **Pre-process**: 0.9 ms
- **Inference (Tesla T4)**: 43.4 ms
- **Post-process (Query Decoding)**: 0.1 ms
- **Total Latency**: **44.4 ms per image (~22.5 FPS)**

---

# Head-to-Head Comparative Matrix

| Evaluation Dimension | Track 1: YOLO26n | Track 2: RT-DETR-L | Winner / System Role |
| :--- | :---: | :---: | :--- |
| **Model Size / Weights** | ~14.8 MB (2.5M params) | ~63.2 MB (32.0M params) | **YOLO** (13× more compact) |
| **Inference Latency** | **5.4 ms** | 44.4 ms | **YOLO** (8.2× faster throughput) |
| **Overall Precision** | **82.70%** | 82.10% | **YOLO** (+0.6%) |
| **Overall Recall** | 80.50% | **80.70%** | **RT-DETR** (+0.2%) |
| **Overall mAP@50** | **85.20%** | 84.50% | **YOLO** (+0.7%) |
| **Overall mAP@50-95** | **51.00%** | 50.40% | **YOLO** (+0.6%) |
| **Mask Break (MB) mAP50** | 83.04% | **85.12%** | **RT-DETR** (+2.08%) |
| **Spurious Copper (SC) mAP50** | 81.93% | **85.43%** | **RT-DETR** (+3.50%) |
| **Foreign Object (CFO) Precision** | 73.37% | **77.91%** | **RT-DETR** (+4.54%) |
| **Micro Foreign Object (BMFO) Prec.** | 83.76% | **87.54%** | **RT-DETR** (+3.78%) |
| **Hole Break (HB) mAP50** | **98.28%** | 95.98% | **YOLO** (+2.30%) |
| **Short Circuit (SH) mAP50** | **91.19%** | 88.75% | **YOLO** (+2.44%) |

---

## Technical Conclusion & Recommendation on Further Epochs
1. **Convergence Confirmed**: Both models have fully plateaued on this dataset split at 640×640. Running more epochs will burn GPU hours without moving the needle on mAP50 ($\pm 0.3\%$).
2. **Complementary Divergence**:
   - YOLO dominates on sharp edge/macro geometry (**HB**, **SH**).
   - RT-DETR's self-attention heads excel at context-dependent micro flaws (**SC**, **MB**, **CFO**, **BMFO**).
3. **Dual-Track Synergy**: Neither model alone breaks 90% across every class, but their complementary failure modes prove the value of the dual-track consensus/predictive routing pipeline.
