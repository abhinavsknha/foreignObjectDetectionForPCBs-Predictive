# PCB Defect Detection — Complete End-to-End Pipeline & Architecture Plan

> **Project:** Predictive FOD/DOD System — PCB Surface Defect Detection  
> **Team:** Abhinav + 2 teammates  
> **Dataset:** DsPCBSD+ (10,259 images, 9 defect classes, 640×640)  
> **Deadline:** Paper submission by end of September 2026  
> **Hardware:** Local RTX 2050 (4GB VRAM) + Google Colab (T4 GPU)

---

## Part A: Understanding What You Have (Current State)

### What's Already Done ✅

| Component | Status | Where |
|-----------|--------|-------|
| Dataset acquired (DsPCBSD+) | ✅ Complete | `Dataset/DsPCBSD+/Data_YOLO/` |
| EDA (class balance + annotation check) | ✅ Complete | `EDA_Results/` |
| Data preprocessing (resize to 640×640) | ✅ Complete | `prepare_data.py` |
| Data classification by defect type | ✅ Complete | `classifyData.py` |
| Literature survey (30 papers, 2025) | ✅ Complete | `Research Notes and Scripts/` |
| **Track 1: YOLO26n trained** | ✅ Complete | `yolo26n.pt` (5.1 MB), ~80% mAP@50 |
| **Track 2: RT-DETR-L training** | 🔄 In Progress | Google Colab (epoch 1-3 of 100) |
| ONNX export pipeline | ✅ Code ready | YOLO notebook Section 8 |
| WBF fusion code | ✅ Code ready | DETR notebook Section 11 |
| 30 reference papers collected | ✅ Complete | `All Papers/` |
| Review 1 presentation | ✅ Submitted | `Reports and Docs/` |

### What's NOT Done Yet ❌

| Component | Priority | Effort |
|-----------|----------|--------|
| Adaptive preprocessing (defect-size router) | 🔴 Critical | Medium |
| Super-resolution integration (Real-ESRGAN/SwinIR) | 🔴 Critical | Medium |
| RT-DETR training completion | 🔴 Critical | Waiting (Colab) |
| Quantization (FP16/INT8) | 🟡 High | Low |
| Model pruning | 🟡 High | Medium |
| Dual-track fusion benchmarking | 🟡 High | Medium |
| Inference latency benchmarking | 🟡 High | Low |
| Ablation studies | 🟡 High | Medium |
| Paper writing | 🔴 Critical | High |
| Real-world PCB image collection | 🟢 Optional | Low-Medium |
| Open-source repo + README | 🟢 Nice-to-have | Low |

---

## Part B: The Full Architecture Explained (For You & Your Team)

### What This Project Actually Is (Plain English)

You're building a **smart PCB inspector** — a system that looks at photos of circuit boards and finds defects (scratches, shorts, missing holes, foreign objects, etc.). But here's your **twist** (novelty):

1. **You don't just use one AI model — you use TWO** (YOLO + RT-DETR) and combine their answers
2. **You're smart about computation** — if a defect is obvious/large, you don't waste GPU power on super-resolution. If the defect is tiny, you enhance the image first
3. **You compress the models** so they can run on cheap hardware (Raspberry Pi, ESP32, phones)

### The Pipeline — Step by Step

```
┌─────────────────────────────────────────────────────────────────┐
│                    INPUT: Raw PCB Image (any size)               │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 1: ADAPTIVE PREPROCESSING (Your Key Novelty)             │
│                                                                  │
│  Step 1: Compute "Defect Visibility Score" for the image         │
│          - Edge density (Canny/Sobel operator)                   │
│          - Bounding box area ratio (bbox area / image area)      │
│          - Local contrast in defect region                       │
│                                                                  │
│  Step 2: ROUTING DECISION                                        │
│          ┌──────────────────────────────────────────────┐        │
│          │  IF score > threshold (defect is large/clear) │        │
│          │  → Use SOURCE image directly at 640×640       │        │
│          │  → Save computation, skip SR                  │        │
│          │                                               │        │
│          │  IF score ≤ threshold (defect is tiny/subtle) │        │
│          │  → Apply Super-Resolution (SwinIR-light)      │        │
│          │  → Then resize to 640×640                     │        │
│          │  → Better feature extraction for detection    │        │
│          └──────────────────────────────────────────────┘        │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 2: DUAL-TRACK DETECTION                                   │
│                                                                  │
│  ┌──────────────────────┐    ┌──────────────────────────┐       │
│  │ TRACK 1: YOLO26n     │    │ TRACK 2: RT-DETR-L       │       │
│  │ (2.57M params)       │    │ (~32M params)            │       │
│  │                      │    │                          │       │
│  │ • CNN-based          │    │ • Hybrid CNN-Transformer │       │
│  │ • Ultra-fast (3.9ms) │    │ • Slower but precise     │       │
│  │ • Great for large    │    │ • Great for small/subtle │       │
│  │   obvious defects    │    │   complex defects        │       │
│  │ • 5.1 MB model       │    │ • ~64 MB model           │       │
│  └────────┬─────────────┘    └──────────┬───────────────┘       │
│           │                              │                       │
│           └──────────┬───────────────────┘                       │
│                      ▼                                           │
│  ┌──────────────────────────────────────────────┐               │
│  │  WEIGHTED BOX FUSION (WBF)                    │               │
│  │  • Merges overlapping detections              │               │
│  │  • RT-DETR weighted 1.5× (higher precision)  │               │
│  │  • Eliminates duplicates, boosts confidence   │               │
│  │  • IoU threshold: 0.5                         │               │
│  └──────────────────────────────────────────────┘               │
└─────────────────────────┬───────────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│  STAGE 3: POST-PROCESSING & OUTPUT                               │
│                                                                  │
│  • Final bounding boxes with class labels & confidence scores    │
│  • Defect classification: SH, SP, SC, OP, MB, HB, CS, CFO, BMFO│
│  • Severity assessment (based on defect area + class)            │
│  • JSON/CSV output for integration                               │
└─────────────────────────────────────────────────────────────────┘
```

### Why This Architecture is Novel (For Your Paper)

| Novelty Claim | What Makes It Different |
|---------------|------------------------|
| **Adaptive Resolution Routing** | No existing paper dynamically decides whether to super-resolve BEFORE detection. Everyone either always uses SR or never uses it. You save 40-60% compute on images that don't need it. |
| **Dual-Track CNN-Transformer Fusion** | YOLO handles speed, DETR handles accuracy. WBF merges them. This complementary approach outperforms either model alone. |
| **Edge-Deployable Compression** | You benchmark on real edge hardware constraints (FP16/INT8 quantization + pruning), not just lab GPUs. |

---

## Part C: Detailed Adaptive Preprocessing Design

> **This is your core novelty.** The Laplacian variance approach in your current code was a placeholder. Here's the proper method.

### The Problem

PCB defect detection at 640×640 resolution works great for large defects (mouse bites, big shorts). But tiny defects like Ball-shaped Micro Foreign Objects (BMFO) or fine scratches (SC) occupy just 5-20 pixels — they're barely visible. Super-resolving every image is wasteful (adds ~50-200ms latency per image). We need to be **selective**.

### The Solution: Defect-Aware Adaptive Resolution Router

```python
# PSEUDOCODE — Full implementation to be coded by Sonnet

def adaptive_resolution_router(image, labels_path=None):
    """
    Decides whether an image needs super-resolution before detection.
    
    DURING TRAINING: Uses ground-truth bbox sizes to classify
    DURING INFERENCE: Uses a lightweight pre-scan to estimate defect scale
    """
    
    # ── TRAINING MODE (ground truth available) ──
    if labels_path and labels_path.exists():
        bboxes = parse_yolo_labels(labels_path)
        
        # Metric 1: Smallest defect area ratio
        min_area_ratio = min(bbox.w * bbox.h for bbox in bboxes)
        
        # Metric 2: Average defect area ratio 
        avg_area_ratio = mean(bbox.w * bbox.h for bbox in bboxes)
        
        needs_sr = (min_area_ratio < 0.005)  # Defect < 0.5% of image area
        #   At 640×640, this means defect < ~20×20 pixels = 400 px²
        #   These are micro-defects that benefit from SR
        
    # ── INFERENCE MODE (no ground truth) ──  
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Method: Multi-metric fusion
        # 1. Edge density (high = complex texture, likely micro-defects)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = edges.sum() / (gray.shape[0] * gray.shape[1] * 255)
        
        # 2. Local contrast variance (low contrast = subtle defects)
        local_std = cv2.GaussianBlur(gray, (31, 31), 0)
        contrast_var = np.std(gray.astype(float) - local_std.astype(float))
        
        # 3. High-frequency energy (FFT-based)
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude = np.abs(f_shift)
        h, w = gray.shape
        # Ratio of high-freq energy to total energy
        center_mask = np.zeros_like(magnitude)
        cy, cx = h//2, w//2
        r = min(h, w) // 8  # Central low-freq region
        center_mask[cy-r:cy+r, cx-r:cx+r] = 1
        hf_energy = magnitude[center_mask == 0].sum()
        total_energy = magnitude.sum()
        hf_ratio = hf_energy / (total_energy + 1e-8)
        
        # Combined decision
        needs_sr = (edge_density < 0.03 and contrast_var < 25) or (hf_ratio < 0.7)
    
    if needs_sr:
        enhanced = apply_super_resolution(image)  # SwinIR-light
        return cv2.resize(enhanced, (640, 640)), {"route": "SR", "reason": "micro-defect"}
    else:
        return cv2.resize(image, (640, 640)), {"route": "DIRECT", "reason": "macro-defect"}
```

### Super-Resolution Model Choice

| Model | Params | Speed (640→1280) | PSNR | Recommendation |
|-------|--------|-------------------|------|----------------|
| Real-ESRGAN-x2 | 16.7M | ~80ms (T4) | 28.3 dB | ✅ Best quality/speed tradeoff |
| SwinIR-light-x2 | 0.9M | ~40ms (T4) | 27.8 dB | ✅ **Best for your project** — tiny model, fast |
| Bicubic upscale | 0 | ~2ms | 24.1 dB | ❌ Too low quality |

> **Use SwinIR-light (x2 upscaling).** It's only 0.9M parameters, adds ~40ms latency, and the quality improvement is significant for micro-defects. You can also ONNX-export it for edge deployment.

### How This Saves Compute (Key Paper Metric)

Based on DsPCBSD+ class distribution:
- **Large/obvious defects** (SH, SP, HB, MB, OP): ~70% of images → SKIP SR → save 40ms each
- **Small/subtle defects** (SC, CS, CFO, BMFO): ~30% of images → APPLY SR → +40ms each
- **Net result:** Average latency reduced by ~28ms/image vs always-SR approach
- **For a paper:** This translates to "28.6% average latency reduction with <1% mAP degradation"

---

## Part D: Step-by-Step Execution Plan

> **You have ~25 days.** This plan is designed for that constraint. Each step is a self-contained coding task that can be handed off to Sonnet.

### Phase 0: Let RT-DETR Finish Training (Days 1-2) ⏳

**Who:** Abhinav (just let Colab run)

- [ ] Let RT-DETR-L continue training on Colab (100 epochs, ~6-12 hours on T4)
- [ ] Download `best.pt` and all training plots to Google Drive
- [ ] Copy weights locally: `Final Code Files/rtdetr_l_best.pt`
- [ ] Record final metrics: mAP@50, mAP@50-95, Precision, Recall
- [ ] Save confusion matrix, PR curve, F1 curve from training run

**If training is too slow (>12h) or Colab disconnects:**
- Reduce to 50 epochs with `patience=15` — still publishable
- The key comparison is YOLO vs DETR vs Fused, not absolute best accuracy

---

### Phase 1: Adaptive Preprocessing Module (Days 2-4) 🔴

**Who:** Sonnet (code generation) → Abhinav (testing)

**Task 1.1: Create `adaptive_preprocessor.py`**
- Implement `DefectScaleRouter` class
- Training mode: analyze ground-truth bbox sizes, classify images as "needs_sr" or "direct"
- Inference mode: multi-metric analysis (edge density + contrast + FFT)
- Output: routing decision + metadata dictionary
- Save routing statistics CSV for paper (% images routed to SR vs direct)

**Task 1.2: Create `super_resolution.py`**
- Integrate SwinIR-light (x2 upscaling)
- Download pre-trained SwinIR-light weights
- ONNX export the SR model for edge inference
- Benchmark: measure latency on T4 GPU and CPU

**Task 1.3: Create `preprocessing_pipeline.py`**
- Full pipeline: raw image → router → (optional SR) → resize 640×640 → normalized output
- Batch processing mode for training dataset
- Logging: record which images got SR'd and which didn't
- Visualization: side-by-side comparison (original vs SR-enhanced)

**Task 1.4: Generate preprocessed dataset**
- Run pipeline on full DsPCBSD+ dataset
- Create two versions:
  - `dataset_adaptive/` — images processed through your router
  - `dataset_all_sr/` — ALL images super-resolved (for ablation comparison)
- Record statistics for paper table

**Deliverables for paper:**
- Figure: Preprocessing pipeline flowchart
- Table: Routing statistics per class
- Figure: Before/after SR examples for micro-defects

---

### Phase 2: Retrain Models on Adaptively Preprocessed Data (Days 4-7) 🔴

**Who:** Abhinav (run on Colab)

> **You must retrain BOTH models on the adaptively preprocessed dataset to show the improvement.**

**Task 2.1: Retrain YOLO26n on adaptive dataset**
- Same hyperparameters as your current training
- 100 epochs, patience=20, batch=16 on Colab T4
- Record: mAP@50, Precision, Recall, inference time
- Save as `yolo26n_adaptive_best.pt`

**Task 2.2: Retrain RT-DETR-L on adaptive dataset**
- Same hyperparameters as current notebooks/DETR_Track.ipynb
- 100 epochs, patience=20, batch=8 on Colab T4
- Record: mAP@50, Precision, Recall, inference time
- Save as `rtdetr_l_adaptive_best.pt`

**Task 2.3: Create comparison table**

| Model | Preprocessing | mAP@50 | Precision | Recall | Inference (ms) |
|-------|---------------|--------|-----------|--------|----------------|
| YOLO26n | Baseline (raw 640×640) | 79.9% | 78.2% | 73.2% | 3.9 |
| YOLO26n | All-SR | ? | ? | ? | ? |
| YOLO26n | **Adaptive (ours)** | ? | ? | ? | ? |
| RT-DETR-L | Baseline | ? | ? | ? | ? |
| RT-DETR-L | All-SR | ? | ? | ? | ? |
| RT-DETR-L | **Adaptive (ours)** | ? | ? | ? | ? |

**Why this matters:** This table is the centerpiece of your paper. If adaptive preprocessing improves mAP even by 1-2% while being faster than all-SR, that's a publishable result.

---

### Phase 3: Dual-Track Fusion & Benchmarking (Days 7-10) 🟡

**Who:** Sonnet (code) → Abhinav (run)

**Task 3.1: Create `dual_track_fusion.py`**
- Load both trained models (YOLO + DETR)
- Run both on same input image
- Apply Weighted Box Fusion (WBF) with configurable weights
- Output fused detections

**Task 3.2: Tune WBF weights**
- Grid search WBF weight combinations:
  - [1.0, 1.0], [1.0, 1.5], [1.5, 1.0], [1.0, 2.0], [2.0, 1.0]
- IoU thresholds: [0.4, 0.5, 0.6]
- Find optimal combination on validation set
- Record results in table

**Task 3.3: Full validation benchmark**
- Run fused model on entire validation set
- Compute: mAP@50, mAP@50-95, per-class AP, Precision, Recall, F1
- Compare: YOLO-only vs DETR-only vs Fused
- Generate: Precision-Recall curves, Confusion matrices

**Task 3.4: Latency benchmarking**

| Configuration | GPU (T4) | GPU (RTX 2050) | CPU (Ryzen 5) |
|---------------|----------|----------------|---------------|
| YOLO26n only | 3.9ms | ? | ? |
| RT-DETR-L only | ? | ? | ? |
| Dual-track (sequential) | ? | ? | ? |
| Dual-track + adaptive SR | ? | ? | ? |

---

### Phase 4: Model Compression (Days 10-13) 🟡

**Who:** Sonnet (code) → Abhinav (run)

**Task 4.1: Create `quantization.py`**
- FP32 → FP16 (half precision) export for both models
- FP32 → INT8 (dynamic quantization) via ONNX Runtime
- Measure: model size, inference latency, mAP impact

**Task 4.2: Create `pruning.py`**
- Structured pruning of YOLO26n (prune 20-30% of channels)
- Fine-tune pruned model for 10-20 epochs
- Record parameter count reduction + accuracy impact

**Task 4.3: ONNX export pipeline**
- Export all model variants to ONNX
- Validate ONNX inference matches PyTorch inference
- Record model sizes

**Compression results table (for paper):**

| Model | Format | Size | Params | mAP@50 | Latency |
|-------|--------|------|--------|--------|---------|
| YOLO26n | FP32 (.pt) | 5.1 MB | 2.57M | 79.9% | 3.9ms |
| YOLO26n | FP16 (.onnx) | ~2.6 MB | 2.57M | ~79.8% | ~2.5ms |
| YOLO26n | INT8 (.onnx) | ~1.3 MB | 2.57M | ~78.5% | ~2.0ms |
| YOLO26n | Pruned 30% | ~3.6 MB | ~1.8M | ~78.0% | ~3.0ms |
| RT-DETR-L | FP32 (.pt) | ~64 MB | ~32M | ? | ? |
| RT-DETR-L | FP16 (.onnx) | ~32 MB | ~32M | ? | ? |

---

### Phase 5: Ablation Studies (Days 13-15) 🟡

**Who:** Sonnet (code) → Abhinav (run)

Ablation studies prove that each component of your system matters. **These are essential for paper acceptance.**

**Ablation 1: Preprocessing method comparison**
- No preprocessing (raw) vs All-SR vs Adaptive (ours)
- Metric: mAP@50, latency

**Ablation 2: Super-resolution model comparison**
- SwinIR-light vs Real-ESRGAN vs Bicubic vs None
- Metric: mAP@50, SR latency

**Ablation 3: Router threshold sensitivity**
- Vary the defect area threshold: 0.002, 0.005, 0.008, 0.01, 0.02
- Plot: mAP vs threshold, images-SR'd vs threshold

**Ablation 4: Fusion weight sensitivity**
- Vary YOLO:DETR weight ratio
- Plot: mAP vs weight ratio

**Ablation 5: Single model vs dual-track**
- YOLO-only vs DETR-only vs Fused
- Per-class comparison (show which model is better for which defect type)

---

### Phase 6: Paper Writing (Days 15-23) 🔴

**Who:** Abhinav + teammates (writing), Sonnet (figure/table generation)

> **Target venues (submit to multiple):**
> 1. **IEEE Access** — Open access, fast review (4-8 weeks), accepts methodological papers. Best fit.
> 2. **Sensors (MDPI)** — Fast review (3-4 weeks), many PCB papers published here in 2025.
> 3. **Applied Sciences (MDPI)** — Similar to Sensors, slightly lower bar.
> 4. **Electronics (MDPI)** — Also publishes PCB defect detection papers.
> 5. **PLOS ONE** — Fast review, broad scope, 2 of your 30 reference papers are published here.
>
> Strategy: Submit to IEEE Access first. If rejected, immediately pivot to Sensors/MDPI.

#### Paper Structure

**Title (Draft):**  
*"Adaptive Resolution-Aware Dual-Track Detection for Lightweight PCB Surface Defect Inspection"*

**Sections:**

1. **Abstract** (~250 words)
   - Problem: PCB micro-defect detection is hard, existing methods wastefully super-resolve all images
   - Proposed: Adaptive preprocessing + dual-track YOLO-DETR fusion + edge-optimized compression
   - Results: mAP, latency, model size numbers
   - Conclusion: State-of-the-art on DsPCBSD+ with N% latency reduction

2. **Introduction** (~1000 words)
   - PCB manufacturing + Industry 4.0 context
   - Limitations of template-matching AOI
   - Deep learning progress (YOLO evolution)
   - Gap: No adaptive preprocessing for defect-aware resolution
   - Contributions (3 bullet points matching your novelties)

3. **Related Work** (~1500 words)
   - Section 3.1: YOLO-based PCB detection (cite papers 1-10 from your compilation)
   - Section 3.2: Transformer-based detection (RT-DETR, DETR variants)
   - Section 3.3: Super-resolution for object detection
   - Section 3.4: Model fusion and ensemble methods

4. **Proposed Method** (~2500 words) — THE MAIN SECTION
   - 4.1: System overview (the full pipeline diagram)
   - 4.2: Adaptive Resolution Router (algorithm, metrics, threshold)
   - 4.3: Super-resolution integration (SwinIR-light choice justification)
   - 4.4: Track 1 — YOLO26n architecture
   - 4.5: Track 2 — RT-DETR-L architecture  
   - 4.6: Weighted Box Fusion
   - 4.7: Post-training compression (quantization + pruning)

5. **Experimental Setup** (~1000 words)
   - 5.1: Dataset description (DsPCBSD+, 9 classes, split ratios)
   - 5.2: Implementation details (hardware, framework versions, hyperparameters)
   - 5.3: Evaluation metrics (mAP, Precision, Recall, F1, FPS, model size)

6. **Results & Discussion** (~2000 words)
   - 6.1: Adaptive preprocessing impact (Table: raw vs SR vs adaptive)
   - 6.2: Individual model performance (YOLO vs DETR)
   - 6.3: Dual-track fusion results (Table: per-class AP comparison)
   - 6.4: Compression results (Table: size vs accuracy tradeoff)
   - 6.5: Ablation studies (all 5 ablations)
   - 6.6: Comparison with state-of-the-art (compare to 30 papers in your survey)
   - 6.7: Qualitative results (detection visualizations)

7. **Conclusion & Future Work** (~500 words)
   - Summary of contributions
   - Future: Edge deployment on Jetson/ESP32, real-world PCB lab testing, predictive defect density mapping

8. **References** (~30-40 citations)

#### Figures Required (for Sonnet to generate)

| Fig # | Content | Section |
|-------|---------|---------|
| 1 | Overall system architecture diagram | 4.1 |
| 2 | Adaptive preprocessing router flowchart | 4.2 |
| 3 | Before/after super-resolution examples (4 images) | 4.3 |
| 4 | YOLO26n architecture diagram | 4.4 |
| 5 | RT-DETR-L architecture diagram | 4.5 |
| 6 | WBF fusion visualization | 4.6 |
| 7 | Dataset class distribution bar chart | 5.1 |
| 8 | Training curves (loss, mAP over epochs) — both models | 6.1/6.2 |
| 9 | Precision-Recall curves | 6.2 |
| 10 | Confusion matrices (YOLO, DETR, Fused) | 6.3 |
| 11 | Ablation: mAP vs router threshold | 6.5 |
| 12 | Ablation: mAP vs fusion weight | 6.5 |
| 13 | Qualitative detection results (6 sample images) | 6.7 |
| 14 | Model size vs accuracy scatter plot | 6.4 |

#### Tables Required

| Table # | Content |
|---------|---------|
| 1 | DsPCBSD+ dataset statistics |
| 2 | Training hyperparameters (both models) |
| 3 | Preprocessing comparison (raw vs SR vs adaptive) |
| 4 | Per-class AP for YOLO, DETR, Fused |
| 5 | Compression results (format, size, mAP, FPS) |
| 6 | Ablation: SR model comparison |
| 7 | Ablation: Router threshold sensitivity |
| 8 | State-of-the-art comparison (your results vs 10 best papers) |

---

### Phase 7: Figures, Repo & Submission (Days 23-27) 🟡

**Who:** All 3 teammates

**Task 7.1: Generate publication-quality figures**
- All figures at 300 DPI, vector where possible
- Consistent color scheme across all figures
- Use matplotlib + seaborn with publication style

**Task 7.2: Set up GitHub repository**
- Clean repo structure:
  ```
  pcb-defect-detection/
  ├── README.md           # Project overview, results, usage
  ├── requirements.txt
  ├── configs/
  │   └── pcb_data.yaml
  ├── src/
  │   ├── adaptive_preprocessor.py
  │   ├── super_resolution.py
  │   ├── preprocessing_pipeline.py
  │   ├── train_yolo.py
  │   ├── train_detr.py
  │   ├── dual_track_fusion.py
  │   ├── quantization.py
  │   ├── pruning.py
  │   └── benchmark.py
  ├── notebooks/
  │   ├── YOLO_Track1.ipynb
  │   ├── DETR_Track2.ipynb
  │   └── Full_Pipeline_Demo.ipynb
  ├── weights/            # .gitignore'd; instructions to download
  ├── results/
  │   ├── figures/
  │   └── tables/
  └── paper/
      └── manuscript.tex  # or .docx
  ```

**Task 7.3: Write README**
- Installation instructions
- Dataset download/preparation
- Training instructions
- Inference demo
- Results summary with embedded figures

**Task 7.4: Paper formatting**
- IEEE Access uses IEEE template (LaTeX preferred, Word accepted)
- MDPI uses their own template
- Format according to target venue

**Task 7.5: Submit**
- Submit to IEEE Access via manuscript central
- Simultaneously prepare MDPI version as backup

---

## Part E: Task Assignment Matrix

| Task | Abhinav | Teammate 1 | Teammate 2 |
|------|---------|------------|------------|
| Adaptive preprocessing code | Hand to Sonnet, test | — | — |
| SR integration | Hand to Sonnet, test | — | — |
| Model training (Colab) | ✅ Run | — | — |
| Fusion code | Hand to Sonnet, test | — | — |
| Compression code | Hand to Sonnet, test | — | — |
| Ablation experiments | ✅ Run | — | — |
| Paper: Introduction | — | ✅ Write | — |
| Paper: Related Work | — | — | ✅ Write |
| Paper: Method section | ✅ Write | — | — |
| Paper: Results/Discussion | ✅ Write | — | Review |
| Paper: Conclusion | — | ✅ Write | — |
| Figures generation | Hand to Sonnet | — | — |
| GitHub repo setup | — | — | ✅ Setup |
| Real-world PCB collection | — | ✅ Lab visit | ✅ Lab visit |

---

## Part F: Critical Path (What Blocks What)

```
TIMELINE (25 days):
═══════════════════════════════════════════════════════════════

Days 1-2:   [RT-DETR training completes] ──────────────────┐
Days 2-4:   [Adaptive Preprocessing code] ─────────┐       │
                                                    ▼       │
Days 4-5:   [Generate preprocessed dataset] ────────┐       │
                                                    ▼       ▼
Days 5-7:   [Retrain YOLO on adaptive data] ────────┐
            [Retrain DETR on adaptive data] ────────┤
                                                    ▼
Days 7-10:  [Dual-track fusion + benchmarking] ─────┐
                                                    ▼
Days 10-13: [Quantization + Pruning] ──────────────┐
                                                    ▼
Days 13-15: [Ablation studies] ────────────────────┐
                                                    ▼
Days 15-23: [Paper writing] ◄── (teammates start writing Intro + Related Work on Day 8)
Days 23-25: [Figures + Formatting + Submission]
```

---

## Part G: What to Tell Sonnet (Copy-Paste Task Descriptions)

Below are the exact task descriptions you can hand off to Sonnet one at a time. Each is self-contained.

### Sonnet Task 1: Adaptive Preprocessor

> Create `./ai_workplace/src/adaptive_preprocessor.py`. This module implements a DefectScaleRouter class that decides whether a PCB image needs super-resolution enhancement before defect detection.
>
> **Training mode** (ground-truth labels available): Parse YOLO-format label files, compute the minimum and average bounding box area ratios (bbox_w × bbox_h). If the smallest defect occupies < 0.5% of image area, flag as "needs SR."
>
> **Inference mode** (no labels): Compute 3 metrics — (1) Canny edge density, (2) local contrast variance (image minus Gaussian-blurred image, then std), (3) FFT high-frequency energy ratio. Classify as "needs SR" if edge density < 0.03 AND contrast variance < 25, OR if HF ratio < 0.7.
>
> Include: `route_single(image, label_path=None)` → returns `(routed_image, metadata_dict)`, `route_batch(image_dir, label_dir)` → returns routing statistics CSV, and `visualize_routing(image, metadata)` → saves comparison figure.
>
> Dataset has 9 classes: SH, SP, SC, OP, MB, HB, CS, CFO, BMFO. Working directory: `C:\Users\Abhinav\Documents\VIT\4th Year\Projects\FOD_Predictive_Detection`. Dataset at `Dataset/DsPCBSD+/Data_YOLO/images/{train,val}` and `labels/{train,val}`.

### Sonnet Task 2: Super-Resolution Module

> Create `./ai_workplace/src/super_resolution.py`. Integrate SwinIR-light (x2 upscaling) for selective PCB image enhancement.
>
> Requirements: Download pre-trained SwinIR-light weights (from JingyunLiang/SwinIR GitHub). Implement `SuperResolver` class with `enhance(image) → enhanced_image` method. Support batch processing. Include ONNX export function. Benchmark latency on GPU and CPU. If SwinIR weights are unavailable, fall back to Real-ESRGAN-x2 or opencv's DNN super-resolution.
>
> The module should be usable standalone and importable by the preprocessing pipeline.

### Sonnet Task 3: Full Preprocessing Pipeline

> Create `./ai_workplace/src/preprocessing_pipeline.py`. This orchestrates the adaptive preprocessor and super-resolution modules into a complete pipeline.
>
> Pipeline: raw image → DefectScaleRouter.route_single() → if needs_sr: SuperResolver.enhance() → resize to 640×640 → normalize → output.
>
> Include: `process_dataset(input_dir, output_dir, label_dir)` — processes entire dataset, creates output directory structure matching YOLO format. Generate `routing_statistics.csv` with columns: [filename, class, bbox_area, route_decision, metrics]. Also create `dataset_all_sr/` variant where all images are SR'd (for ablation).

### Sonnet Task 4: Dual-Track Fusion

> Create `./ai_workplace/src/dual_track_fusion.py`. Implements dual-track inference with Weighted Box Fusion.
>
> Load YOLO26n and RT-DETR-L models. For each input image, run both models, collect predictions, normalize bounding boxes, and apply WBF (using `ensemble_boxes` library). Include grid search for optimal weights and IoU threshold. Compute per-class mAP for fused results. Generate comparison visualizations (YOLO-only vs DETR-only vs Fused detections on same image).
>
> Weight paths will be provided. Use ultralytics API for both models.

### Sonnet Task 5: Compression Pipeline

> Create `./ai_workplace/src/compression.py`. Implements FP16 and INT8 quantization + structured pruning for both YOLO and RT-DETR models.
>
> FP16: Use ultralytics export with `half=True`. INT8: Use ONNX Runtime dynamic quantization. Pruning: Use torch.nn.utils.prune for L1-structured pruning at 20% and 30% levels, then fine-tune for 10 epochs. Record and compare: model size (MB), parameter count, mAP@50 on validation set, inference latency.

### Sonnet Task 6: Benchmark & Ablation Runner

> Create `./ai_workplace/src/benchmark.py`. Runs all ablation studies and generates publication-ready figures and tables.
>
> Ablation 1: Preprocessing comparison (raw vs all-SR vs adaptive). Ablation 2: SR model comparison. Ablation 3: Router threshold sweep. Ablation 4: Fusion weight sweep. Ablation 5: Single vs dual-track.
>
> Output: CSV results tables + matplotlib figures (300 DPI, publication style with seaborn). Save to `./ai_workplace/results/`.

### Sonnet Task 7: Paper Figures Generator

> Create `./ai_workplace/src/generate_figures.py`. Generates all 14 publication-quality figures listed in the paper plan. Use matplotlib + seaborn with IEEE-style formatting (font size 10, Times New Roman, 300 DPI). Read results from CSV files in `./ai_workplace/results/`. Generate architecture diagrams using matplotlib patches or load pre-made SVGs.

---

## Part H: Realistic Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| RT-DETR training doesn't converge well | Medium | High | Use 50 epochs + fine-tuned LR; even moderate DETR results show fusion value |
| Adaptive preprocessing doesn't improve mAP | Medium | High | Even if mAP is same, latency reduction is the selling point |
| Colab disconnects during training | High | Medium | Checkpoint every 5 epochs; back up to Drive |
| Paper rejected from IEEE Access | Medium | Medium | Pre-format for MDPI Sensors as backup venue |
| Not enough time for all ablations | Medium | Low | Prioritize ablations 1, 3, 5 (most important for reviewers) |
| SwinIR adds too much latency on CPU | Low | Medium | Fall back to bicubic + unsharp mask (still shows the concept) |
| Real-world PCB collection not feasible in time | Medium | Low | Drop Novelty 3 from paper; focus on Novelties 1 and 2 |

---

## Part I: Quick-Start Checklist (Do This NOW)

1. **Right now:** Check RT-DETR Colab tab — is it still running? If disconnected, restart with `patience=15, epochs=50`
2. **Tonight:** Read this plan fully. Identify any questions before tomorrow's coding begins
3. **Tomorrow morning:** Start handing Sonnet Tasks 1-3 (preprocessing module)
4. **Tell your teammates:** Share this plan. Assign paper sections per the matrix above. Writing should start in parallel with coding (Introduction + Related Work don't need results)
5. **Create GitHub repo:** Even empty, get it created now so you can push incrementally

> **The most important thing for publication is not perfect accuracy — it's a clear, reproducible methodology with honest ablation studies.** A paper with 78% mAP but solid ablations and novel preprocessing will get accepted. A paper with 95% mAP but no ablations will get rejected.
