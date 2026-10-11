# 🎯 Review 2 Tactical Execution Plan (`review_plan.md`)

> **Event:** BCSE497J Project-I Review-2 (Offline Mode)  
> **Date & Time:** **Wednesday, September 9, 2026 @ 11:30 AM**  
> **Institution:** School of Computer Science and Engineering, VIT  
> **Venue:** Assigned Review Hall (Report by 11:30 AM sharp)  
> **Evaluation Criteria:** 4 Components (all include Q&A)  
> **Target Status:** **Distinction (Max Marks across all 4 Rubrics)**

---

## 🏛️ Official Rubric Alignment & Scoring Strategy

The university evaluates Review-2 on 4 defined components:

```
┌────────────────────────────────────────────────────────────────────────┐
│                        REVIEW-2 RUBRIC BREAKDOWN                       │
├────┬──────────────────────────────────────────┬────────────────────────┤
│ No.│ University Component                     │ Our Deliverable / Edge │
├────┼──────────────────────────────────────────┼────────────────────────┤
│ 1  │ Literature Survey & Gap Identification   │ 30 papers (2025/2026)  │
│    │                                          │ 3 distinct gaps framed │
├────┼──────────────────────────────────────────┼────────────────────────┤
│ 2  │ Objectives, Methodology & Architecture   │ 3 measurable goals +   │
│    │                                          │ Dual-Track Architecture│
├────┼──────────────────────────────────────────┼────────────────────────┤
│ 3  │ Implementation (60–75%) & Demonstration  │ >75% completed! Both   │
│    │                                          │ YOLO & DETR trained!   │
├────┼──────────────────────────────────────────┼────────────────────────┤
│ 4  │ Presentation, Draft Report & Outcome     │ 8-Slide PPT + Guide    │
│    │                                          │ Email Proof + Moodle   │
└────┴──────────────────────────────────────────┴────────────────────────┘
```

---

## 📑 Component-by-Component Defense Strategy

### Component 1: Literature Survey and Gap Identification
* **What Examiners Want:** Proof that you did not pick an arbitrary problem; clear understanding of 2025–2026 state-of-the-art.
* **Our Content:**
  * Synthesized review of **30 contemporary research papers** (collected in `All Papers/`).
  * **Identified Research Gaps:**
    1. **Gap 1 (Processing Inefficiency):** Existing AOI systems either upscale every single image (introducing 50–100ms needless latency on macro defects) or upscale none (failing on hairline scratches and micro particles).
    2. **Gap 2 (Single-Paradigm Bottleneck):** Standalone CNNs (YOLOv8/11) struggle with long-range trace connectivity, while pure Vision Transformers are computationally heavy for edge deployment.
    3. **Gap 3 (Lack of Edge Compression Benchmarks):** Most academic papers report raw server GPU benchmarks without verifying FP16/INT8 quantization for factory floor embedded devices.

---

### Component 2: Objectives, Methodology and Architecture
* **What Examiners Want:** Clear engineering flow and sound theoretical formulation.
* **Project Title:** *(Preserved strictly as submitted in Review 1 — modification not permitted)*.
* **Three Measurable Objectives:**
  1. *Objective 1:* Develop a Defect-Aware Adaptive Resolution Router that reduces computational overhead by dynamically categorizing boards into direct vs. super-resolution pipelines.
  2. *Objective 2:* Construct a Dual-Track Complementary Detection Framework pairing a high-speed CNN (YOLO26n) with an attention-based Transformer (RT-DETR-L), fused via Weighted Box Fusion (WBF).
  3. *Objective 3:* Benchmark and optimize the unified system for edge execution via ONNX FP16/INT8 quantization without degrading mAP.
* **System Flowchart:** Input Image ➔ Adaptive Router ➔ Dual-Track (YOLO + RT-DETR) ➔ WBF Late Fusion ➔ Edge Quantization.

---

### Component 3: Implementation (60% to 75%) and Live Demonstration
* **What Examiners Want:** Working code, solid empirical results, and a live demonstration showing that the project is not just theoretical slides.
* **Current Implementation Progress: >75% (Exceeds Review-2 requirement!)**
  * ✅ **Dataset Preparation (100%):** 10,259 images in DsPCBSD+, 9 defect classes, verified class balance.
  * ✅ **Track 1 (CNN) Completed (100%):** YOLO26n trained to convergence: **79.9% mAP@50**, 78.2% Precision, 73.2% Recall, 3.9ms latency.
  * ✅ **Track 2 (Transformer) Completed (100%):** RT-DETR-L trained for 25 epochs on Kaggle T4 GPU: **85.6% mAP@50**, 82.2% Precision, 81.7% Recall, **51.2% mAP@50-95**.
  * ✅ **Model Export (100%):** Both models successfully exported to ONNX FP16 (`best.onnx`).
  * 🔄 **Dual-Track WBF Fusion (Ready for Demo):** `ai_workplace/src/dual_track_fusion.py` configured to run inference combining both models.
* **Live Demonstration Script (2-Minute Sequence):**
  1. Open visualizer / python demo.
  2. Load a sample test board containing a subtle defect (e.g. hair-line Conductor Scratch or Hole anomaly).
  3. Show **YOLO prediction** (fast, tight, but lower confidence on subtle traces).
  4. Show **RT-DETR prediction** (catches global trace context).
  5. Show **WBF Fused Output** (confidence reinforced to >90%, redundant proposals eliminated).
  6. Present the **F1-Confidence curve** proving that **0.55** is the scientifically optimal decision threshold.

---

### Component 4: Presentation, Draft Report, and Guide Approval
* **Crucial Rule:** All students must communicate with their project guide through email and obtain approval for Review 2 components. **Approval proof must be placed on Slide 2 of the PPT.**

#### PPT Slide-by-Slide Layout (8 Slides Maximum):
* **Slide 1:** Title of Project, Course Code (BCSE497J), Team Members, Register Numbers, Project Guide.
* **Slide 2:** **Guide Approval Evidence** *(Screenshot of guide's email approving Review 2 PPT, Draft Report, Demonstration, and Attendance)*.
* **Slide 3:** Literature Review Summary & 3 Research Gaps.
* **Slide 4:** Project Objectives & Novelty Highlights.
* **Slide 5:** System Architecture & Detailed Flowchart.
* **Slide 6:** Implementation Progress (Dataset + Training Curves + Confusion Matrix).
* **Slide 7:** Experimental Results & Comparative Table (YOLO: 79.9% vs RT-DETR: 85.6%).
* **Slide 8:** Live Demonstration / Qualitative Predictions + Timeline to Final Review (Late October 2026).

---

## ⏰ Step-by-Step Action Plan (Countdown to Wednesday)

```
Monday (Today)                Tuesday                       Wednesday (Review Day)
───────────────               ─────────                     ──────────────────────
[1] Run WBF Fusion Demo       [4] Assemble 8-Slide PPT      [7] Arrive at venue by 11:15 AM
[2] Export Side-by-Side Visual[5] Email Guide for Approval  [8] Deliver 8-min presentation
[3] Compile Draft Report Doc  [6] Upload Report on Moodle   [9] Run flawless live demo & Q&A
```

### Day 1: Monday, Sept 7 (Technical Outputs & Visual Assets) — 100% COMPLETE ✅
- [x] **Action 1.1:** Run `dual_track_fusion.py` to produce fused results on validation images. *(Completed: F1=0.6493, 17 false alarms eliminated, max recall 86.63% preserved)*.
- [x] **Action 1.2:** Generate a crisp 4-panel comparison figure: `Raw Image | YOLO | RT-DETR | WBF Fused` and save to `ai_workplace/results/figures/dual_track_4panel_demo.png`.
- [x] **Action 1.3:** Run Adaptive Resolution Scale Router across full dataset (10,259 images: 55.13% Direct, 44.87% SR, 22.1ms/img latency reduction).
- [x] **Action 1.4:** Execute Phase 4 Edge Model Compression (ONNX Dynamic INT8: YOLO compressed to 2.74MB [3.41×], RT-DETR to 32.45MB [3.87×]; 4-panel analysis figure saved).
- [ ] **Action 1.5:** Prepare draft report text and slide deck content ready for Tuesday execution.

### Day 2: Tuesday, Sept 8 (Approvals, Submission & Rehearsal)
- [ ] **Action 2.1:** Populate the PPT template slides using our generated tables and figures.
- [ ] **Action 2.2:** Send official email to Project Guide with Draft Report and PPT attached requesting approval for Review-2.
- [ ] **Action 2.3:** Once guide responds, paste the approval email screenshot directly onto **Slide 2**.
- [ ] **Action 2.4:** Upload the Review 2 Report Document to **Moodle**.
- [ ] **Action 2.5:** Rehearse the 8-minute presentation with teammates (each teammate takes 2-3 slides).

### Day 3: Wednesday, Sept 9 @ 11:30 AM (Execution Day)
- [ ] Report to assigned venue by **11:15 AM** (offline mode).
- [ ] Carry laptop with charged battery, HDMI adapter, and offline copies of PPT + demo script.
- [ ] Confidently defend the 4 components.

---

## 🔄 Convergence with `plan.md` & IEEE Paper
* The comparative metrics table (YOLO 79.9% vs RT-DETR 85.6% vs WBF) generated for Review-2 becomes **Table 1** of our IEEE Access / Sensors paper.
* The 4-panel demonstration visual becomes **Figure 3** of the manuscript.
* After Wednesday, we transition immediately to **Phase 1 (Adaptive Preprocessor Benchmark)** and **Phase 4 (Quantization/Pruning)** in `plan.md`.
