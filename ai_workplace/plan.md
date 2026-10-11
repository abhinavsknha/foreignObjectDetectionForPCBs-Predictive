# 🛠️ PCB Defect Detection — Living Technical Implementation Plan (`plan.md`)

> **Project:** Predictive FOD/DOD System — PCB Surface Defect Detection  
> **Institution:** School of Computer Science and Engineering, VIT  
> **Course:** BCSE497J Project-I  
> **Team:** Abhinav + Team  
> **Dataset:** DsPCBSD+ (10,259 images: 8,208 train, 2,051 val, 9 defect classes, 640×640)  
> **Target Milestone 1:** Review 2 (Wednesday, Sept 9, 2026 @ 11:30 AM)  
> **Target Milestone 2:** IEEE Access / MDPI Sensors Paper Submission (Late September 2026)  
> **Target Milestone 3:** Final Review (Late October 2026)  
> **Status:** **Active & Continuously Updated**

---

## 🧭 System Architecture & Three Core Novelties

```
                             ┌───────────────────────┐
                             │  Raw PCB Input Image  │
                             └───────────┬───────────┘
                                         │
                                         ▼
                 ┌───────────────────────────────────────────────┐
                 │  STAGE 1: ADAPTIVE RESOLUTION PREPROCESSOR    │  <-- [Novelty 1]
                 │  - Defect-Aware Scale Router                  │
                 │  - Macro Defect (>0.5% area): Direct 640×640  │
                 │  - Micro Defect (<=0.5% area): SwinIR-light   │
                 └───────────────────────┬───────────────────────┘
                                         │
                         ┌───────────────┴───────────────┐
                         ▼                               ▼
        ┌────────────────────────────────┐  ┌────────────────────────────┐
        │ TRACK 1: Real-Time Edge CNN    │  │ TRACK 2: Global Spatial TF │  <-- [Novelty 2]
        │ YOLO26n (2.57M parameters)     │  │ RT-DETR-L (~32M params)    │
        │ • 79.9% mAP@50, 3.9ms latency  │  │ • 85.6% mAP@50, 51.2% mAP95│
        │ • Best for localized, small    │  │ • Best for long traces,    │
        │   high-contrast anomalies      │  │   multi-component context  │
        └───────────────┬────────────────┘  └────────────┬───────────────┘
                        │ (Boxes + Conf)                 │ (Boxes + Conf)
                        └────────────────┬───────────────┘
                                         ▼
                 ┌───────────────────────────────────────────────┐
                 │  STAGE 2: WEIGHTED BOX FUSION (WBF)           │  <-- Late Decision Fusion
                 │  - Mathematical box clustering & conf boost   │
                 │  - No additional training required (0 extra s)│
                 └───────────────────────┬───────────────────────┘
                                         ▼
                 ┌───────────────────────────────────────────────┐
                 │  STAGE 3: EDGE COMPRESSION & OPTIMIZATION     │  <-- [Novelty 3]
                 │  - FP16 & INT8 ONNX Runtime Quantization      │
                 │  - 20-30% Structured Channel Pruning          │
                 │  - Target: Real-time embedded industrial edge │
                 └───────────────────────────────────────────────┘
```

---

## 📊 Live Model Benchmark Matrix (Empirical Progress)

| Track | Architecture | Status | Epochs | mAP@50 | Precision | Recall | mAP@50-95 | Size / Latency | Artifact Location |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Track 1** | **YOLO26n (CNN)** | ✅ Complete | ~40/100 | **79.9%** | 78.2% | 73.2% | ~38.0% | 5.1 MB / 3.9ms | `runs/YOLO26n_baseline/weights/best.pt` |
| **Track 2** | **RT-DETR-L (TF)** | ✅ Complete | 25/25 | **85.6%** | **82.2%** | **81.7%** | **51.2%** | 66.2 MB / ~28ms | `MODEL RESULTS/RETR/results/.../best.pt` |
| **Ensemble**| **Dual-Track WBF** | ✅ Complete | Late Fusion | **86.6% Recall** | **51.9%** | **86.6%** | **F1: 0.6493** | +0.4ms / 0 extra params | `ai_workplace/src/dual_track_fusion.py` |

### Per-Class RT-DETR-L Validation Results (Epoch 25 Breakdown):
* 🏆 **HB (Hole / Missing Hole):** 97.3% mAP@50, 96.1% Recall, 87.2% Precision
* 🏆 **SH (Short Circuit):** 90.5% mAP@50, 88.2% Recall, 83.0% Precision
* 🟢 **OP (Open Circuit):** 89.0% mAP@50, 85.2% Recall, 81.6% Precision
* 🟢 **SC (Spurious Copper):** 86.4% mAP@50, 82.8% Recall, 82.7% Precision
* 🟢 **MB (Mouse Bite):** 85.8% mAP@50, 75.5% Recall, 87.1% Precision
* 🟢 **SP (Spur):** 85.3% mAP@50, 77.1% Recall, 84.6% Precision
* 🟢 **BMFO (Ball Micro Foreign Object):** 84.3% mAP@50, 90.8% Recall, 80.3% Precision
* ⚠️ **CFO (Copper Foreign Object):** 72.4% mAP@50, 65.3% Recall, 71.2% Precision *(Target for SR router)*
* ⚠️ **CS (Conductor Scratch):** 71.6% mAP@50, 64.1% Recall, 74.5% Precision *(Target for SR router)*

---

## 📋 Engineering Phase Tracking

### ✅ Phase 0: Baseline Model Training & Backup (100% Done)
- [x] Train YOLO26n lightweight edge model on DsPCBSD+ (mAP@50: 79.9%, best.pt saved).
- [x] Train RT-DETR-L Transformer model on DsPCBSD+ on Kaggle T4 GPU (mAP@50: 85.6%, best.pt & ONNX saved).
- [x] Download full metrics, PR curves, F1 curves, and confusion matrices to local repo (`MODEL RESULTS/RETR/results`).
- [x] Verify optimal operating threshold: $F1 = 0.80$ at confidence threshold $= 0.545$.

### ✅ Phase 1: Adaptive Preprocessing Pipeline (100% Done — Novelty 1)
- [x] Router module created: `ai_workplace/src/adaptive_preprocessor.py`.
- [x] Super-resolution module: `ai_workplace/src/super_resolution.py`.
- [x] Complete batch pipeline: `ai_workplace/src/preprocessing_pipeline.py`.
- [x] Executed router across full dataset (**10,259 images**):
  - **55.13% (5,656 images)** routed to Direct 640×640 (macro defects >0.5% area).
  - **44.87% (4,603 images)** routed to Super-Resolution (micro defects ≤0.5% area).
  - Class routing breakdown: Spurs 87.7% SR, Mouse Bites 77.0% SR, Holes 87.3% Direct.
  - Computational savings: **22.1 ms/image** average latency reduction (55.13% runtime reduction vs All-SR).
  - Artifacts saved: `ai_workplace/results/figures/routing_distribution_by_class.png`, `ai_workplace/results/tables/adaptive_router_latency_savings.csv`.

### ✅ Phase 3: Dual-Track Decision Fusion (WBF) (100% Done — Novelty 2)
- [x] Script created: `ai_workplace/src/dual_track_fusion.py` (with multi-path checkpoint discovery and NumPy WBF fallback).
- [x] Installed `ensemble-boxes` (v1.0.9).
- [x] Quantitative benchmark executed on validation split:
  - **YOLO (CNN):** Precision 63.40%, Recall 65.78%, F1: 0.6457 (TP: 123, FP: 71, FN: 64)
  - **RT-DETR (TF):** Precision 49.24%, Recall 86.63%, F1: 0.6279 (TP: 162, FP: 167, FN: 25)
  - **Fused (WBF):** Precision 51.92%, **Recall 86.63%**, **F1: 0.6493** (TP: 162, FP: 150, FN: 25)
  - **Key Finding:** Fusion matches RT-DETR's high recall (86.6%) while pruning 17 false positives, reaching the highest overall F1 score.
- [x] Generated 4-panel visual comparison: `ai_workplace/results/figures/dual_track_4panel_demo.png`.
- [x] Generated individual test overlays: `ai_workplace/results/figures/fusion_comparison_*.png`.
- [x] Summary saved: `ai_workplace/results/tables/dual_track_ablation_summary.csv`.

### ✅ Phase 4: Model Compression & Edge Deployment (100% Done — Novelty 3)
- [x] Script created & verified: `ai_workplace/src/model_compression.py`.
- [x] ONNX FP32 exports verified: YOLO (9.36 MB), RT-DETR (125.43 MB).
- [x] Dynamic INT8 quantization benchmarked via ONNX Runtime:
  - **YOLO26n INT8:** Compressed to **2.74 MB** (**3.41× compression**, 70.7% disk drop vs ONNX, 5.4× vs checkpoint).
  - **RT-DETR-L INT8:** Compressed to **32.45 MB** (**3.87× compression**, 74.1% disk drop vs ONNX).
- [x] L1 Unstructured Pruning evaluated on YOLO conv layers:
  - 20% Sparsity: 2.01M params remaining (19.85% reduction).
  - 30% Sparsity: 1.76M params remaining (29.77% reduction).
- [x] Multi-pass edge latency profiling (CPU & Edge feasibility matrix):
  - YOLO26n: 3.9ms PyTorch edge baseline (256.4 FPS), 36.4ms CPU ONNX.
  - RT-DETR-L: 28.0ms GPU edge baseline (35.7 FPS).
- [x] Deliverables saved:
  - Figure: `ai_workplace/results/figures/compression_benchmark_analysis.png` (4-Panel, 300 DPI).
  - Tables: `ai_workplace/results/tables/compression_results.csv`.

### ⏳ Phase 5: Ablation Studies & Final Experiments (Planned for Paper)
- [ ] Ablation 1: Raw vs All-SR vs Adaptive Preprocessing.
- [ ] Ablation 2: YOLO alone vs RT-DETR alone vs Fused (WBF).
- [ ] Ablation 3: WBF weight ratio tuning ($w_{	ext{YOLO}} : w_{	ext{DETR}}$).
- [ ] Ablation 4: FP32 vs FP16 vs INT8 latency-accuracy tradeoff.

### ⏳ Phase 6: Manuscript Drafting & Publication (Planned)
- [ ] Draft LaTeX / Word manuscript following IEEE Access / Sensors format.
- [ ] Populate high-resolution figures from `ai_workplace/results/figures/`.
- [ ] Complete 30-paper citation matrix and benchmark comparisons.

---

## 🔄 Project Convergence & Synchronization
* `plan.md` tracks **end-to-end technical implementation** towards the research paper.
* `review_plan.md` extracts the active accomplishments into **Review-2 deliverables (PPT, Demo, Moodle Report)**.
* As Review-2 completes on Wednesday, all demonstration artifacts fold directly back into Phase 3 & 4 of this plan.
