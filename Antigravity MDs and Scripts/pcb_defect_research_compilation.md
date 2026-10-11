# PCB Defect Detection — Research Paper Compilation & SciSpace Prompt

---

## 1. SciSpace Prompt

> [!TIP]
> Copy-paste the prompt below directly into SciSpace to search and organize related literature in tabular form.

```
Search for research papers published in 2025 on the topic of "Lightweight Deep Learning-Based Defect Detection on Printed Circuit Boards (PCBs) Using Improved YOLO Architectures." I want papers that focus on:

1. Improved YOLOv8/v9/v10/v11/v12 architectures for PCB surface defect detection
2. Lightweight object detection models for Automated Optical Inspection (AOI) of PCBs
3. Attention mechanisms (CBAM, coordinate attention, SimAM) integrated into YOLO for small/tiny PCB defect detection
4. Knowledge distillation and model compression for edge deployment of PCB inspection systems
5. GAN-based or diffusion-based data augmentation for rare PCB defect classes
6. Transformer-YOLO hybrid architectures (DETR, RT-DETR, Swin Transformer) for PCB defect detection
7. Multi-scale feature fusion (FPN, BiFPN, PANet) for detecting micro-level defects on PCBs
8. Few-shot learning and Siamese networks for PCB anomaly detection with limited training data
9. U-Net and segmentation-based approaches for PCB surface defect localization
10. Benchmark evaluations on standard PCB datasets (DeepPCB, DsPCBSD+, HRIPCB, PKU-PCB)

For each paper found, provide the following columns in a table:
- S.No.
- Paper Title
- Author(s) (First author et al.)
- Year of Publication
- Journal / Conference Name
- DOI (if available)
- Key Methodology (e.g., YOLOv11 + Attention, GAN + YOLO, Transformer)
- Dataset Used
- Best Reported mAP (%)
- Relevance to "Lightweight PCB Defect Detection using Improved YOLO"

Sort by relevance and recency. Prioritize 2025 publications.
```

---

## 2. Compilation of 30 Research Papers (2025) — Merged from Multiple Sources

> **Sources merged:** Original compilation (20 papers) + SciSpace finalPCBtable (10 papers) + combinedPCB_YOLO2025 search results (2 highly relevant papers). Duplicates removed. Sorted by relevance.

### Top 5 — Highest Relevance (Core Papers)

| S.No. | Paper Title | Author(s) | Year | Journal / Conference | DOI | Key Methodology | Dataset & mAP |
|:-----:|:-----------|:----------|:----:|:---------------------|:----|:----------------|:-------------|
| 1 | **A Lightweight PCB Defect Detection Algorithm Based on an Improved YOLOv11s** | Xuhui Ye et al. | 2025 | IEEE Access | — | Improved YOLOv11s with lightweight backbone, attention integration, model compression | DsPCBSD+ / DeepPCB |
| 2 | **Lightweight PCB Defect Detection Method Based on SCF-YOLO** | Yazhou Li et al. | 2025 | PLOS ONE | 10.1371/journal.pone.0318033 | SCF-YOLO + MobileNet backbone + Knowledge distillation (YOLOv8-l teacher); 25% param reduction | PKU-PCB |
| 3 | **GESC-YOLO: Improved Lightweight PCB Defect Detection** | Xiangqiang Kong et al. | 2025 | Sensors (MDPI) | — | YOLOv8n + C2f-GE + Slim-neck (GSConv) + Coordinate Attention; 99% mAP | PCB dataset; 99% mAP |
| 4 | **A Lightweight PCB Defect Detection Method Based on Group Convolution and Adaptive Pruning** | Yuezhong Chu et al. | 2025 | PeerJ Computer Science | 10.7717/peerj-cs.3243 | YOLOv8n + C2f_Star + GS_Detect (GroupConv) + LAMP pruning | PKU-Market-PCB; 96.7% mAP@50 |
| 5 | **MAS-YOLO: Lightweight Detection Based on Improved YOLOv12** | Xupeng Yin et al. | 2025 | Applied Sciences (MDPI) | — | First YOLOv12 PCB model; MECS attention + AHFIN neck + Slide Alignment Loss | DeepPCB / PKU-PCB |

### Papers 6–10 — High Relevance (YOLO Variants)

| S.No. | Paper Title | Author(s) | Year | Journal / Conference | DOI | Key Methodology | Dataset & mAP |
|:-----:|:-----------|:----------|:----:|:---------------------|:----|:----------------|:-------------|
| 6 | **YOLO-DefXpert: Advanced Defect Detection Using Improved YOLOv11** | Prabu Selvam et al. | 2025 | IEEE Access | — | YOLOv11 + Swin Transformer + CBAM + DCNv2 | HRIPCB; 99.0% mAP@50 |
| 7 | **A High-Accuracy PCB Defect Detection Algorithm Based on Improved YOLOv12** | Zhi Chen et al. | 2025 | Symmetry | 10.3390/sym17070978 | YOLOv12 + GAM + BiFPN + HWD + PPA head + NWD loss | PKU-PCB / DeepPCB; 90.4% mAP@50 |
| 8 | **Small Defect Detection in PCBs Based on Multiscale Edge Strengthening and Improved YOLOv10** | Wei-Xun Chen et al. | 2025 | Scientific Reports | 10.1038/s41598-025-20387-x | YOLOv10 (MS-YOLO) + MSES + SPDConv + LAMP pruning + Dual-distillation | DeepPCB / PKU-PCB; 98.9% mAP@50 |
| 9 | **PCB Defect Detection Based on Improved YOLOv8n** | Ao Wang et al. | 2025 | AICIT Conference | 10.1109/aicit65974.2025.11282815 | YOLOv8n + HFPM + DKAM + DFFN + DSDH | HRIPCB / PKU-PCB |
| 10 | **PCB Board Defect Detection Algorithm Based on Improved YOLOv8** | Meijie Zhao et al. | 2025 | ISAEECE Conference | 10.1109/isaeece66033.2025.11160235 | Improved YOLOv8 + PConv + MLCA | PCB dataset |

### Papers 11–15 — High Relevance (Novel Architectures & Attention)

| S.No. | Paper Title | Author(s) | Year | Journal / Conference | DOI | Key Methodology | Dataset & mAP |
|:-----:|:-----------|:----------|:----:|:---------------------|:----|:----------------|:-------------|
| 11 | **NeoPCB-YOLO: Next-generation Enhanced Optimization YOLO** | Anxu Wei | 2025 | EIECS Conference | 10.1109/eiecs67708.2025.11283608 | NeoPCB-YOLO + FDConv (Frequency Domain Convolution) + multi-attention | DeepPCB; >95.6% mAP@50 |
| 12 | **RF-YOLO: Enhanced YOLOv8 Combining RFCAConv and FDPN** | Xinyu Wu et al. | 2025 | ICICML Conference | 10.1109/icicml67980.2025.11333440 | YOLOv8 + RFCAConv + FDPN | PKU-PCB; 91.6% mAP@0.5 |
| 13 | **ABF-YOLO: Fine-Grained PCB Defect Detection with Axial Attention** | Xueming He et al. | 2025 | IEEE Trans. Consumer Electronics | 10.1109/tce.2025.3597022 | Axial Attention + Bidirectional Feature Fusion | DeepPCB |
| 14 | **Application Research of CycleGAN and YOLO in PCB Infrared Defect Detection** | Chao Yang et al. | 2025/26 | arXiv (Preprint) | — | CycleGAN (cross-modal augmentation) + Lightweight YOLOv8 | Pseudo-IR (CycleGAN) |
| 15 | **YOLOv8x-HICAUps: Attention Based Approach for PCB Defect Detection** | M. Mounika et al. | 2025 | IJEAST | — | YOLOv8x + HorNet backbone + CBAM + Attention-based upsampling | PCB dataset; 98.3% accuracy |

### Papers 16–20 — High Relevance (GAN, Distillation, Edge Deployment)

| S.No. | Paper Title | Author(s) | Year | Journal / Conference | DOI | Key Methodology | Dataset & mAP |
|:-----:|:-----------|:----------|:----:|:---------------------|:----|:----------------|:-------------|
| 16 | **Defect Detection Network in PCB Circuit Devices Based on GAN Enhanced YOLOv11** | Jiayi Huang et al. | 2025 | Applied & Computational Engineering / arXiv | — | GAN augmentation + YOLOv11 | Custom PCB dataset |
| 17 | **Enhanced YOLOv11 Framework for High Precision PCB Defect Detection** | Zeinab F. Elsharkawy | 2025 | Scientific Reports (Nature) | — | YOLOv11 + EMA + CARAFE + EIoU loss | PCB dataset |
| 18 | **PCB-YOLO: Coordinate Attention and Multi-Scale Feature Fusion** | Ze Wei et al. | 2025 | PLOS ONE | — | YOLOv8n + FFCA + WIPIoU loss | PCB dataset |
| 19 | **YOLO-HDEW: An Efficient PCB Defect Detection Model** | Chuanwang Song et al. | 2025 | Electronics (MDPI) | — | P2-level high-res detection + Edge-enhanced Multi-scale Parallel Attention | PCB dataset |
| 20 | **Research on Lightweight PCB Defect Detection (YOLOv11-CGL)** | Jiankang Yu et al. | 2025 | IJCATR, Vol. 14, Issue 09 | — | YOLOv11 + Context-Guided module + Knowledge distillation; 26% compression | PCB dataset |

### Papers 21–30 — Supporting Papers (Alternative Approaches & Benchmarks)

| S.No. | Paper Title | Author(s) | Year | Journal / Conference | DOI | Key Methodology | Dataset & mAP |
|:-----:|:-----------|:----------|:----:|:---------------------|:----|:----------------|:-------------|
| 21 | **VR-YOLO: Enhancing PCB Defect Detection with Viewpoint Robustness** | Hengyi Zhu et al. | 2025 | arXiv | — | YOLOv8 + DSE augmentation + KOF with angular loss | Custom PCB; 98.9% mAP |
| 22 | **Vision Transformer and Mamba-Attention Fusion for PCB Defect Detection** | Asim Niaz et al. | 2025 | PLOS ONE | — | ViT-Mamba hybrid + artificial defect generation | PCB dataset; 99.69% mAP |
| 23 | **A Lightweight Transformer for PCB Defects Detection (LPDETR)** | Yuanchen Niu et al. | 2025 | FASTA 2025 (IEEE) | — | Lightweight RT-DETR; ~30% size reduction | PCB dataset |
| 24 | **Deep Learning-Enhanced Defects Detection for Printed Circuit Boards** | Van-Truong Nguyen et al. | 2025 | Results in Engineering (Elsevier) | — | ORB + RANSAC + U-Net segmentation | Custom PCB; 97% accuracy |
| 25 | **Performance Analysis of YOLO11 Models in PCB Defect Detection** | Mehmet Dayıoğlu et al. | 2025 | J. Northern Aegean Tech. Sciences | — | YOLO11 n/s/m/l/x benchmark (6 defect types) | PCB dataset |
| 26 | **Defect Detection of PCB Based on Improved YOLOv11** | Jianqiu Sun et al. | 2025 | MSIEID 2025 (Atlantis Press) | — | YOLOv11 + MSCA Attention + CARAFE | PCB dataset |
| 27 | **PCB Defect Detection Based on Lightweight DL Fusion Model** | Yuling Wang et al. | 2025 | Sensors (MDPI) | — | MobileNet v3 Small-CA + multi-scale fusion | PCB dataset |
| 28 | **MSA-DETR: Multi-Scale Alignment Detection Transformer for PCB** | Renjie Zhang et al. | 2025 | Measurement Sci. & Tech. (IOP) | — | Cross-scale feature fusion + spatial alignment | PCB dataset |
| 29 | **ResNet-SE-CBAM Siamese Networks for Few-Shot PCB Defect Classification** | Chao-Hsiang Hsiao et al. | 2025 | Sensors (MDPI) | — | Siamese + ResNet + SE + CBAM + triplet loss | PCB dataset |
| 30 | **SM-FSOD: Meta-Learning for Few-Shot PCB Defect Detection** | Xinnan Shao et al. | 2025 | Electronics (MDPI) | — | Meta-learning + defect-aware prototypes | PCB dataset |

> **Note:** Paper #31 ("Deep Learning Algorithms for Defect Detection on Electronic Assemblies: A Systematic Literature Review" by Montoya Magaña et al., 2025, MDPI) is included in the Word document as a comprehensive survey reference but excluded from the numbered table for brevity.

---

## 3. Base Paper Recommendation

> [!IMPORTANT]
> **Recommended Base Paper: Paper #1 — "A Lightweight PCB Defect Detection Algorithm Based on an Improved YOLOv11s"**
> 
> Authors: Xuhui Ye, Lilu Huang, Jun Wang, Xinyu Hu, Jing Xu, Chupeng Zhang  
> Journal: IEEE Access (2025/2026)  
> Already available as PDF in the project directory.

### Why This Should Be the Base Paper

| Criterion | Justification |
|:----------|:-------------|
| **Direct Topic Match** | It proposes a **lightweight improved YOLOv11s** specifically for **PCB defect detection** — exactly our project scope. |
| **Latest Architecture** | Uses YOLOv11 (the newest YOLO variant), making the work state-of-the-art and publication-worthy. |
| **Lightweight Focus** | Explicitly addresses model compression and efficiency — critical for industrial AOI edge deployment. |
| **Reputable Journal** | Published in **IEEE Access** (open-access, high-impact IEEE journal), which is widely accepted in VIT academic reviews. |
| **Reproducibility** | The paper uses publicly available PCB datasets (DeepPCB, DsPCBSD+), and the DsPCBSD+ dataset is already present in our `archive/` folder. |
| **Clear Extension Path** | We can extend this work by: (a) adding attention mechanisms (CBAM, SimAM), (b) integrating GAN-based augmentation for rare defects, (c) testing on our DsPCBSD+ dataset, or (d) deploying on edge hardware (Jetson Nano). |

---

## 4. Topic Summary (300–400 Words)

### Lightweight Deep Learning-Based Defect Detection on Printed Circuit Boards Using Improved YOLO Architectures

Printed Circuit Boards (PCBs) are the backbone of modern electronics, serving as the foundation for virtually every electronic device — from smartphones and laptops to automotive systems and medical equipment. During the manufacturing process, PCBs are susceptible to surface defects and foreign object contamination, including solder bridges, open circuits, short circuits, mouse bites, missing holes, spurious copper deposits, pin holes, and microscopic metal debris (Foreign Object Debris or FOD). These defects, if left undetected, can cause catastrophic failures in end products, making automated defect detection a critical quality assurance step in smart manufacturing.

Traditional Automated Optical Inspection (AOI) systems rely on rule-based template matching, which is brittle under varying lighting conditions, PCB design changes, and novel defect types. Recent advances in deep learning, particularly the YOLO (You Only Look Once) family of object detectors, have demonstrated remarkable accuracy in real-time PCB defect detection. However, deploying full-scale YOLO models in resource-constrained industrial environments (edge devices, embedded systems on the production line) remains challenging due to high computational demands and large model sizes.

This project focuses on developing a **lightweight, improved YOLO-based architecture** for real-time PCB surface defect detection that achieves high detection accuracy while maintaining low computational overhead suitable for edge deployment. Building upon the YOLOv11s architecture as a base, the proposed approach explores several enhancements: integration of efficient attention mechanisms (such as Coordinate Attention and CBAM) for better small-defect feature extraction, lightweight backbone substitutions (MobileNet-style blocks, GhostNet modules) to reduce parameters, advanced feature fusion strategies (multi-scale neck designs) for detecting defects at varying scales, and GAN-based data augmentation to address class imbalance in rare defect categories.

The project utilizes the DsPCBSD+ (Dataset of PCB Surface Defects) benchmark, which contains over 10,000 annotated images with multiple defect categories, formatted in YOLO-compatible structure. The goal is to build a model that achieves a mean Average Precision (mAP) exceeding 95% while reducing model parameters by at least 25% compared to baseline YOLO architectures, thereby enabling deployment on edge platforms such as NVIDIA Jetson Nano for real-time production line inspection.

This research directly addresses the growing industrial demand for AI-driven quality inspection that is both accurate and efficient, contributing to the broader goals of Industry 4.0 and sustainable smart manufacturing.

**(Word Count: ~340)**

---

## 5. Key Datasets Referenced

| Dataset | Description | Availability |
|:--------|:-----------|:------------|
| **DsPCBSD+** | 10,000+ images, **9 defect classes**, YOLO format | ✅ **Already in our project** (`archive/DsPCBSD+/`) |
| **DeepPCB** | 1,500 aligned pairs, 6 defect types | [GitHub](https://github.com/tangsanli5201/DeepPCB) |
| **HRIPCB** | High-resolution PCB defect images | Academic request |
| **PKU-PCB** | Peking University PCB dataset | Academic request |

### DsPCBSD+ — 9 Defect Classes (from `data.yaml`)

| Code | Full Name | Type | FOD-Related? |
|:----:|:----------|:-----|:------------:|
| SH | Short Circuit | Manufacturing Defect | No |
| SP | Spur | Manufacturing Defect | No |
| SC | Spurious Copper | Manufacturing Defect | No |
| OP | Open Circuit | Manufacturing Defect | No |
| MB | Mouse Bite | Manufacturing Defect | No |
| HB | Hole Break | Manufacturing Defect | No |
| CS | Copper Sliver | Manufacturing Defect | No |
| **CFO** | **Coarse Foreign Object** | **Foreign Object Debris** | **✅ Yes** |
| **BMFO** | **Ball-shaped Micro Foreign Object** | **Foreign Object Debris** | **✅ Yes** |

> [!IMPORTANT]
> The **CFO** and **BMFO** classes are directly related to Foreign Object Debris (FOD) detection — making this dataset uniquely suited for our project topic of "PCB FOD Predictive Detection." Most other PCB datasets (DeepPCB, HRIPCB) only cover manufacturing defects, not foreign objects.

---

## 6. Research Gap & Our Contribution Angle

Based on the 30 papers surveyed, here are the **open research gaps** we can target:

1. **YOLOv11 + Attention + GAN augmentation combined pipeline** — No single paper combines all three for PCB defect detection.
2. **Edge deployment benchmarks on DsPCBSD+** — Most papers benchmark on DeepPCB; using DsPCBSD+ with edge deployment metrics (FPS on Jetson Nano) would be novel.
3. **Predictive FOD analysis** — Moving beyond pure detection to predicting defect-prone regions based on spatial defect density patterns on PCBs.
4. **Cross-dataset generalization** — Training on one PCB dataset and testing on another to prove model robustness.
5. **Frequency-domain feature extraction** — Only NeoPCB-YOLO explores frequency-domain convolutions; this is an underexplored direction.
6. **Infrared + visible light multi-modal fusion** — CycleGAN-based cross-modal approaches are nascent; combining IR and visible data for comprehensive defect detection is an open area.

---

## 7. Key Insights from 2025 Literature

- **Architectural Shift:** Most studies have moved from YOLOv5/v7 to YOLOv8 and YOLOv10, with the first implementations of YOLOv12 appearing in 2025. YOLOv11 variants are increasingly popular.
- **Lightweight Strategies:** Group convolutions (GroupConv), Partial convolutions (PConv), and structural pruning (LAMP) are the primary methods for reducing parameter counts. Knowledge distillation from larger teacher models is gaining traction.
- **Micro-Defect Handling:** BiFPN and SPDConv integration is critical for detecting tiny anomalies. Multi-scale edge strengthening structures are emerging as effective solutions.
- **Data Scarcity Solutions:** GANs (CycleGAN) bridge data gaps for IR/X-ray inspection. Few-shot and meta-learning address rare defect class detection.
- **Attention Mechanism Integration:** CBAM, coordinate attention, axial attention, and decomposed kernel attention are systematically integrated into YOLO for improved small-object PCB detection.
- **Edge Deployment Focus:** Multiple papers target embedded systems (Jetson Nano, TX2) with depthwise separable heads, pruning, and compression, indicating strong industry demand.
