# Adaptive Resolution Routing and Dual-Track CNN-Transformer Fusion for High-Throughput Printed Circuit Board Surface Defect and Foreign Object Debris Detection

**Abhinav$^{1,*}$, Teammate 1$^1$ & Teammate 2$^1$**  
$^1$School of Computer Science and Engineering, Vellore Institute of Technology (VIT), Vellore 632014, Tamil Nadu, India.  
$^*$Corresponding author. Email: abhinav.project@vit.ac.in  
**Target Journal:** *Scientific Reports* (Nature Portfolio) | **Article Type:** Original Research Article  

---

## Abstract
Automated Optical Inspection (AOI) of High-Density Interconnect Printed Circuit Boards (PCBs) faces a critical trade-off between inspection throughput and multi-scale defect detection fidelity. While lightweight convolutional neural networks (CNNs) offer high frame rates, they exhibit elevated miss rates on microscopic anomalies and Foreign Object Debris (FOD). Conversely, Vision Transformers provide global contextual awareness across intricate copper traces but impose prohibitive latency and memory overhead on industrial edge platforms. Furthermore, applying super-resolution (SR) universally across all frames creates severe computational bottlenecks. Here, we present an edge-deployable, high-throughput PCB defect detection framework featuring an Adaptive Resolution Router (ARR) that dynamically evaluates incoming frames via a multi-criteria perceptual complexity gate. Across the 10,259-image DsPCBSD+ benchmark, the router bypasses super-resolution for 55.13% of frames containing macro-defects while selectively enhancing the remaining 44.87% containing micro-defects, saving 22.1 ms per image (55.13% compute reduction). An asymmetric dual-track detector pairing YOLO26n (3.9 ms) and RT-DETR-L (28.0 ms) harmonized via Weighted Box Fusion achieves 86.63% recall and a peak F1-score of 0.6493, eliminating 17 false alarms with zero recall loss. Post-training INT8 quantization yields up to 3.87x compression, confirming real-time feasibility on embedded edge inspection hardware.

**Keywords:** Printed circuit boards, Automated optical inspection, Surface defect detection, Foreign object debris, Vision transformers, Edge artificial intelligence.

---

## 1. Introduction

### 1.1 Industrial Context and the Quality Imperative in Electronics Manufacturing
Printed Circuit Boards (PCBs) serve as the fundamental structural and electrical backbone of virtually all modern electronic systems, spanning mission-critical aerospace avionics, autonomous automotive control units, biomedical implants, industrial telecommunications, and ubiquitous consumer electronics. With the aggressive scaling of high-density interconnect (HDI) substrates, multi-layer micro-via architectures, and ultra-fine-pitch surface-mount technologies (SMT), line widths and trace clearances have contracted to sub-100-micrometer tolerances [1, 2]. Under these sub-millimeter regimes, the structural and electrical integrity of PCB copper conductors, solder masks, and insulating substrate layers is exceedingly vulnerable to manufacturing variations and physical contamination.

Surface defects occurring during fabrication (e.g., etching, plating, and solder masking) manifest in diverse morphological archetypes. These include severe electrical discontinuities such as open circuits (OP) and short circuits (SH); conductor degradation such as copper spurs (SP), spurious residual copper (SC), and fine conductor scratches (CS); and protective barrier failures such as solder mask breaks (MB) and hole breaks (HB) [2, 4]. Simultaneously, foreign particulate contamination—specifically metallic Foreign Object Debris (FOD), such as stray copper particles (CFO) and microscopic solder balls or ball-shaped micro foreign objects (BMFO)—poses an insidious operational risk. Foreign conductive particulate matter can migrate under operational thermal cycling and mechanical vibration, inducing intermittent shorting, arc faults, and catastrophic field failures long after post-production deployment. Consequently, continuous, high-speed, non-destructive Automated Optical Inspection (AOI) represents an indispensable quality control gate across smart electronics manufacturing lines (Industry 4.0) [1, 3].

### 1.2 Limitations of Conventional AOI and Early Deep Learning Systems
Historically, industrial AOI systems have relied predominantly on rule-based machine vision pipelines and classical template matching [24]. These traditional approaches align incoming production images with an idealized golden board reference and compute defect candidate regions via structural differencing, morphological gradient operators, and hand-engineered color-space thresholding. While computationally modest, rule-based AOI systems exhibit profound fragility under real-world factory dynamics: (1) high pseudo-defect rates, where minor non-defective variations such as solder mask color shifts, non-uniform coaxial lighting gradients, micro-vibrations of conveyor belts, and negligible component placement tolerances are frequently flagged as anomalies, overwhelming human operators with false alarm rates often exceeding 20–30% [1, 6]; (2) inflexibility to product iterations, where any alteration to trace geometries or board layouts necessitates manual recalibration and golden template re-registration; and (3) inability to generalize across morphological variance, as deterministic geometric thresholds fail to accurately categorize stochastic defects, such as irregular particulate debris or subtle copper scratches exhibiting ill-defined boundaries.

To transcend these limitations, deep learning algorithms—particularly Convolutional Neural Networks (CNNs)—have been widely adopted for PCB defect localization [3, 5, 8]. Early applications utilized two-stage detectors (e.g., Faster R-CNN, Cascade R-CNN), which achieved respectable detection precision but suffered from high computational latency (>80 ms per frame), rendering them unsuitable for continuous multi-meter-per-second surface inspection lines. In response, single-stage object detectors from the You Only Look Once (YOLO) lineage (YOLOv5 through YOLOv11/v12) emerged as the dominant industrial paradigm due to their unified regression formulations and ultra-low inference latencies [1, 5, 6, 25].

### 1.3 The Scale Disparity Dilemma and the Super-Resolution Bottleneck
Despite their operational speed, modern single-stage CNNs confront a severe performance degradation when applied to PCB inspection due to extreme scale disparity. As documented in modern industrial benchmarks such as DsPCBSD+ [2], defect surface areas exhibit dynamic ranges spanning multiple orders of magnitude:

$$\frac{\text{Area}(\text{HB})}{\text{Area}(\text{BMFO})} > 10^2 \tag{1}$$

Macro-scale defects, such as hole breaks (HB) and large short circuits, occupy several thousand pixels ($\ge 2.5\%$ to $6.0\%$ of total image area), presenting distinct geometric contours that standard downsampling backbones capture readily. In stark contrast, micro-defects—such as ball-shaped micro foreign objects (BMFO), fine copper spurs (SP), and conductor scratches (CS)—often span fewer than $10 \times 10$ pixels, representing less than $0.5\%$ of the total frame. 

When input images are downsampled to canonical detector resolutions (e.g., $640 \times 640$ pixels), the successive strided convolutions and spatial pooling operations of standard CNN backbones irreversibly attenuate the high-frequency spatial gradients of these microscopic anomalies. Consequently, standard CNNs exhibit high false negative rates (misses) on tiny defects.

To mitigate this loss of feature resolution, prior researchers have explored two primary avenues: (1) dense slicing or tiling (e.g., Slicing Aided Hyper Inference - SAHI), which inflates inference latency linearly by a factor of $N_{\text{tiles}}$ (often $4\times$ to $16\times$); and (2) universal deep super-resolution (SR), which applies deep enhancement (e.g., SwinIR, Real-ESRGAN) to every image, adding 50–250 ms per frame. Crucially, existing literature applies super-resolution indiscriminately across all frames. In an industrial inspection batch, more than half of the processed boards contain either macro-defects or clean conductive traces where deep super-resolution yields no detectable accuracy gain, resulting in massive GPU compute wastage and thermal throttling on edge systems.

### 1.4 The CNN vs. Vision Transformer Duality in Defect Inspection
Concurrently, the emergence of Vision Transformers (ViTs) and Detection Transformers (e.g., DETR, Deformable-DETR, RT-DETR) has offered a powerful alternative paradigm [20, 22, 28]. Unlike CNNs, which are constrained by local receptive fields, RT-DETR (Real-Time Detection Transformer) leverages multi-scale cross-attention mechanisms that model long-range spatial dependencies across complex conductive trace networks. This global context enables RT-DETR to differentiate true electrical track continuities from subtle opens (OP) and shorts (SH) with exceptional recall and fine-grained localization accuracy [23]. However, RT-DETR's transformer encoder-decoder layers require substantial floating-point operations (FLOPs) and high VRAM capacity, exhibiting latencies of 25–35 ms on high-end GPUs—nearly an order of magnitude slower than lightweight CNNs (3–5 ms) and excessively heavy for low-power edge gateways.

Thus, industrial AOI confronts a fundamental architectural dilemma: lightweight CNNs offer the required frame rates but miss micro-scale defects, whereas Vision Transformers offer superior recall and global relational context but exceed edge hardware thermal and latency budgets.

### 1.5 Research Contributions
To resolve this multi-scale, latency-constrained dilemma, this paper proposes an Adaptive Resolution-Aware Dual-Track Detection Framework specifically designed for high-precision, real-time PCB surface defect and foreign object debris inspection. The core contributions of this work are summarized as follows:

1. **Lightweight Adaptive Resolution Router (ARR):** We formulate a content-adaptive, complexity-aware preprocessing gate that dynamically determines whether an incoming PCB frame requires resolution enhancement prior to inference. By extracting spatial edge density, local contrast standard deviation, and high-frequency spectral energy via 2D Fast Fourier Transform, the ARR routes 55.13% of frames through a high-speed direct pathway and selectively enhances only the 44.87% exhibiting micro-defect characteristics. This eliminates computational redundancy, yielding an average latency reduction of 22.1 ms per frame (55.13% compute saving) over universal super-resolution baselines.
2. **Asymmetric Dual-Track CNN-Transformer Architecture:** We design an asymmetric dual-track detection engine that executes inference in parallel across two structurally complementary paradigms: Track 1 (Fast CNN: fine-tuned YOLO26n, 3.9 ms GPU latency, 2.57M parameters, 5.14 MB storage) for rapid localized defect extraction; and Track 2 (Contextual Transformer: fine-tuned RT-DETR-L, 28.0 ms GPU latency, 32M parameters) for long-range topological trace modeling.
3. **Complementary Late Fusion via Weighted Box Fusion (WBF):** Rather than relying on traditional greedy Non-Maximum Suppression (NMS)—which discards redundant bounding boxes and ignores multi-model spatial agreement—we implement an asymmetric confidence-weighted box fusion scheme ($w_{\text{DETR}} = 1.5, w_{\text{YOLO}} = 1.0$). The fusion pipeline achieves an optimal balance between precision and recall, securing a peak F1-score of 0.6493 and eliminating 17 false positives present in standalone transformer predictions with zero recall degradation.
4. **Edge-Constrained Quantization and Real-Time Benchmarking:** We validate the deployability of our dual-track pipeline on constrained edge hardware via post-training dynamic INT8 quantization and structured pruning. The quantized models achieve a 3.41x compression ratio for YOLO26n (2.74 MB) and 3.87x for RT-DETR-L (32.45 MB), proving that dual-track fusion can be seamlessly hosted within the physical memory and latency constraints of industrial embedded PCs and edge accelerators.

### 1.6 Organization of the Article
The remainder of this manuscript is organized as follows: Section 2 surveys recent advancements in deep learning for PCB inspection, micro-defect super-resolution, ensemble fusion strategies, and edge compression techniques, establishing key research gaps. Section 3 details the experimental dataset (DsPCBSD+), statistical class distributions, and the mathematical formulations governing the proposed Adaptive Resolution Router and preprocessing pipeline.

**Figure 1 | End-to-end architecture of the proposed dual-track PCB defect inspection framework.** Incoming frames are evaluated by the Adaptive Resolution Router (ARR), which selectively triggers super-resolution for micro-scale defects while bypassing enhancement for macro-scale anomalies. Parallel detection is executed via YOLO26n and RT-DETR-L, merged by Weighted Box Fusion (WBF), and accelerated through INT8 edge quantization.

---

## 2. Related Work / Literature Review

### 2.1 Deep Learning for PCB Automated Optical Inspection: The CNN vs. Transformer Evolution
The development of modern deep-learning-based AOI systems has been largely catalyzed by successive iterations of the YOLO framework. Ye et al. [1] proposed an improved lightweight YOLOv11s model for PCB surface defect detection, incorporating channel and spatial attention mechanisms alongside structured parameter pruning to address edge memory limitations. Their system demonstrated high detection rates on standard benchmarks (DeepPCB and DsPCBSD+) while emphasizing the necessity of model compression for industrial shop-floor integration. Li et al. [2] introduced SCF-YOLO, a lightweight architecture utilizing a MobileNet-style backbone and feature-split shuffle blocks, combined with response-based knowledge distillation from a larger YOLOv8-l teacher network. While SCF-YOLO achieved a 25% parameter reduction and maintained high precision on the PKU-PCB dataset, its detection accuracy degraded noticeably on irregular particulate debris (such as FOD) due to the reduced representation capacity of depthwise separable convolutions.

Concurrently, researchers have sought to enhance standard YOLO backbones with specialized attention mechanisms to counteract small-defect attenuation. Kong, Liu & Gao [3] formulated GESC-YOLO, which integrates a C2f-GE feature extraction block, GSConv slim-neck layers, and Coordinate Attention (CA) into YOLOv8n, reporting up to 99% mAP@50 on balanced test sets. Chu et al. [4] investigated group convolutions and adaptive pruning via the LAMP algorithm, demonstrating that up to 35% of redundant convolutional weights in YOLOv8n could be removed with minimal accuracy loss on the PKU-Market-PCB dataset. In more recent explorations of next-generation backbones, Yin, Zhao & Weng [5] proposed MAS-YOLO, representing one of the earliest adaptations of YOLOv12 for PCB inspection, utilizing Multi-Scale Edge-Context Saliency (MECS) attention and an Asymmetric Heterogeneous Feature Interaction Neck (AHFIN). Selvam et al. [6] introduced YOLO-DefXpert, integrating Swin Transformer blocks, Deformable Convolutions (DCNv2), and the Convolutional Block Attention Module (CBAM) into YOLOv11 to improve defect boundaries. Chen & Liu [7] designed a high-accuracy YOLOv12 variant incorporating BiFPN and Normalized Wasserstein Distance (NWD) loss. Chen, Ke & Meng [8] developed MS-YOLO (based on YOLOv10), integrating Multiscale Edge Strengthening (MSES) modules and Space-to-Depth Convolutions (SPDConv) alongside dual-distillation.

Despite these convolutional advances, pure CNN architectures remain fundamentally bounded by localized receptive fields. When tracing extensive copper paths across a circuit board to detect discontinuous open circuits or subtle bridging, CNNs must stack dozens of layers to achieve sufficient contextual coverage, resulting in gradient degradation and feature blurring. To overcome this, Vision Transformers have been introduced into industrial inspection. Niaz et al. [20] proposed a hybrid architecture coupling Vision Transformers with Mamba state-space attention, achieving 99.69% mAP through selective state-space sequence modeling of PCB surface features. Niu, Sun & Jiang [23] developed LPDETR, a lightweight variant of the Real-Time Detection Transformer (RT-DETR) that reduced parameter count by ~30% while retaining the global self-attention mechanism. Zhang, Qian & Wu [28] designed MSA-DETR, incorporating multi-scale alignment feature pyramids to align spatial features across transformer decoding layers. However, all existing ViT and DETR implementations exhibit substantial computational latency (typically 25–45 ms), which remains an obstacle for continuous, high-speed AOI lines operating at >60 frames per second.

### 2.2 Micro-Defect Localization and Resolution Enhancement Strategies
A critical vulnerability shared by both CNN and Transformer detectors is the severe performance drop encountered on micro-scale defects. On standard benchmarks such as DsPCBSD+, tiny defects like copper spurs (SP), fine conductor scratches (CS), and ball micro foreign objects (BMFO) occupy less than 0.5% of the total pixel area [2]. To restore lost spatial high-frequency details, researchers have investigated super-resolution (SR) networks as neural image enhancers. Modern deep SR frameworks—including SwinIR, Real-ESRGAN, and RCAN—employ deep residual networks or shifted-window self-attention to reconstruct sub-pixel edges, high-frequency textures, and sharp transition boundaries. In industrial domains, applying deep SR prior to detection has been shown to improve small-object mAP by 3.5% to 7.0%.

However, deep super-resolution is computationally demanding. For instance, executing SwinIR-light on a 640 x 640 image consumes 50–120 ms on an embedded GPU—far exceeding the inference latency of the object detector itself. Critically, every existing study in the literature applies resolution enhancement uniformly across all inspection images [8, 14, 15]. In production environments, PCB images arrive in high-volume streams where over 50% of frames contain either macro-defects or standard conductor tracks that do not benefit from super-resolution. Applying deep SR indiscriminately to every image creates an unnecessary computational bottleneck. To date, no prior research has investigated an adaptive, content-aware routing gate that inspects low-level structural image complexity prior to detection to selectively trigger resolution enhancement only when microscopic anomalies are present.

### 2.3 Multi-Detector Ensembles and Late Fusion Techniques
To combine the complementary strengths of different detection models, ensemble learning is widely employed in competitive computer vision. In PCB inspection, CNNs offer high localization speed and high precision on high-contrast edges, whereas Vision Transformers excel at contextual recall and resolving ambiguous, low-contrast traces. Traditionally, multi-detector predictions are merged using Greedy Non-Maximum Suppression (NMS) or its variant, Soft-NMS. However, standard NMS possesses inherent structural flaws in multi-model settings: it assumes that the single bounding box with the highest confidence score represents the most accurate spatial ground truth, discarding all overlapping candidate boxes whose Intersection-over-Union (IoU) exceeds a threshold. If a detector incorrectly assigns high confidence to a false alarm, NMS preserves the error while discarding lower-confidence true positive predictions from secondary models.

To overcome these deficiencies, Solovyev, Wang & Gabruseva introduced Weighted Box Fusion (WBF) [10]. Rather than discarding overlapping bounding boxes, WBF utilizes the confidence scores of all matching predictions across models to compute an averaged, confidence-weighted bounding box coordinate:

$$X_{1, \text{fused}} = \frac{\sum_{i=1}^M w_i \cdot c_i \cdot x_{1, i}}{\sum_{i=1}^M w_i \cdot c_i} \tag{2}$$

While WBF has been utilized in aerial imagery and general object detection competitions, its application to industrial PCB defect detection—specifically as an asymmetric bridge between an ultra-lightweight CNN (YOLO) and a contextual hybrid transformer (RT-DETR)—remains largely unexplored.

### 2.4 Edge Model Compression, Pruning, and Quantization
Deploying deep learning models on factory-floor Automated Optical Inspection equipment requires conformance to strict SWaP (Size, Weight, and Power) constraints. Production line cameras are typically connected to industrial personal computers (IPCs) or embedded accelerators (such as NVIDIA Jetson Orin/Nano or Intel OpenVINO edge gateways) possessing limited GPU VRAM (4GB–8GB) and strict thermal dissipation limits [1, 4, 23]. Model compression techniques primarily encompass structured pruning (e.g., L1-norm or LAMP algorithm [4]) and Post-Training Quantization (PTQ) from FP32 to FP16 and INT8. Quantization to INT8 reduces the model's memory footprint by up to 4x, lowers memory bandwidth pressure, and enables the utilization of specialized integer tensor cores. While individual papers have explored pruning on YOLOv8 [4] or knowledge distillation on YOLOv11 [1, 2], there is a distinct lack of comprehensive empirical research that benchmarks the end-to-end deployment pipeline—combining adaptive resolution routing, dual-track CNN-transformer inference, WBF late fusion, and full INT8 quantization—under uniform industrial edge constraints.

### 2.5 Summary of Research Gaps and Theoretical Positioning
Table 1 provides a systematic comparison between recent representative studies in PCB surface defect detection (2024–2026) and our proposed framework.

**Table 1 | Comprehensive comparison of recent state-of-the-art PCB defect detection studies versus the proposed work.**

| Study & Year | Base Architecture | Key Innovations | Dataset Used | Classes | mAP@50 | Preprocessing Adaptability | Model Fusion | Edge Hardware Validated? |
|:---|:---|:---|:---|:---:|:---:|:---|:---|:---|
| Ye et al. (2026) [1] | YOLOv11s | Attention integration, structured channel pruning | DsPCBSD+ / DeepPCB | 6–9 | 94.2% | Static (Fixed 640) | None (Single Model) | Pruning only (No INT8) |
| Li et al. (2025) [2] | SCF-YOLO | MobileNet backbone, feature-split shuffle, KD | PKU-PCB | 6 | 95.8% | Static (Fixed 640) | None (Single Model) | Theoretical FLOPs only |
| Kong et al. (2025) [3] | GESC-YOLO | C2f-GE block, GSConv slim-neck, CoordAttention | Custom PCB | 6 | 99.0% | Static (Fixed 640) | None (Single Model) | No edge deployment |
| Chu et al. (2025) [4] | YOLOv8n | C2f_Star, GS_Detect (GroupConv), LAMP pruning | PKU-Market-PCB | 6 | 96.7% | Static (Fixed 640) | None (Single Model) | Pruning on PC GPU |
| Yin et al. (2025) [5] | MAS-YOLO (v12) | MECS attention, AHFIN neck, Slide Alignment loss | DeepPCB / PKU-PCB | 6 | 96.1% | Static (Fixed 640) | None (Single Model) | No hardware quantization |
| Selvam et al. (2025) [6] | YOLO-DefXpert | Swin Transformer, CBAM, DCNv2 deformable conv | HRIPCB | 6 | 99.0% | Static (Fixed 1024) | None (Single Model) | High latency (>45 ms) |
| Chen & Liu (2025) [7] | YOLOv12 | GAM attention, BiFPN, PPA head, NWD loss | PKU-PCB / DeepPCB | 6 | 90.4% | Static (Fixed 640) | None (Single Model) | None |
| Chen et al. (2025) [8] | MS-YOLO (v10) | Multiscale edge strengthening (MSES), SPDConv | DeepPCB / PKU-PCB | 6 | 98.9% | Static (Fixed 640) | None (Single Model) | Distillation only |
| Niaz et al. (2025) [20] | ViT-Mamba | Vision Transformer + Mamba state-space fusion | Custom PCB | 6 | 99.69% | Static (Uniform) | Single Hybrid Model | Prohibitive compute (>70 ms) |
| Niu et al. (2025) [23] | LPDETR | Lightweight RT-DETR with reduced query decoders | PCB Benchmark | 6 | 93.4% | Static (Uniform) | None (Single Model) | Tested on RTX 3090 only |
| Proposed Work (2026) | YOLO26n + RT-DETR-L | Adaptive Resolution Router (ARR), Dual-Track, Asymmetric WBF | DsPCBSD+ | 9 (incl. FOD) | 86.6% Recall / 0.6493 F1 | Dynamic Content-Aware (55.13% SR bypass, 22.1 ms saved) | Weighted Box Fusion (Eliminates 17 false alarms) | Full INT8 Quantization (3.87x compression, 2.74 MB) |

*Comparative assessment across model paradigm, dataset diversity, accuracy, preprocessing adaptability, fusion strategy, and edge compression validation.*

---

## 3. Materials and Methods

### 3.1 Dataset

#### 3.1.1 Overview and Source Verification
To evaluate the proposed framework under rigorous industrial conditions, we utilize the DsPCBSD+ (Deep-learning-based PCB Surface Defect Detection Dataset Plus) benchmark [21]. The dataset was established by Lv et al. and published in *Scientific Data* (Nature Portfolio, 2024, DOI: 10.1038/s41597-024-03657-z), released under the Creative Commons Attribution 4.0 International License (CC BY 4.0) for open scientific benchmarking. DsPCBSD+ represents one of the largest and most comprehensive publicly available datasets dedicated to automated optical inspection of printed circuit boards. It comprises 10,259 high-resolution images, each captured at a standardized dimension of 640 x 640 pixels in 24-bit true color RGB format. The physical imaging was executed using an industrial telecentric optical inspection system equipped with a high-resolution CCD area scan camera and a coaxial LED illumination dome, specifically engineered to eliminate non-uniform specular reflections from bare copper traces and solder pads.

#### 3.1.2 Defect Taxonomies and Morphological Characterization
Unlike earlier PCB datasets (such as DeepPCB or PKU-PCB) that are restricted to six canonical etching defects, DsPCBSD+ provides exhaustive annotations across nine distinct industrial defect classes. Importantly, the dataset explicitly incorporates particulate contamination and Foreign Object Debris (FOD), which are critical failure modes in surface-mount production lines: (1) Short Circuit (SH), an unintended bridge of conductive material connecting isolated traces; (2) Spur (SP), a localized protrusion of excess copper extending outward from trace boundaries; (3) Spurious Copper (SC), isolated islands of unetched copper on dielectric substrates; (4) Open Circuit (OP), a complete physical fracture along a conductive trace; (5) Mask Break (MB), a void in the protective solder mask exposing underlying copper; (6) Hole Break (HB), an off-center or misaligned drill via fracturing through an annular ring; (7) Conductor Scratch (CS), mechanical abrasions thinning trace cross-sections; (8) Copper Foreign Object (CFO), extraneous conductive copper debris adhering to the board; and (9) Ball Micro Foreign Object (BMFO), microscopic spherical solder spatter or particulate debris.

**Table 2 | Statistical breakdown and morphological scale characterization of defect classes in the DsPCBSD+ dataset.**

| Class ID | Class Code | Defect Full Name | Primary Failure Mechanism | Instances | Frequency (%) | Median Box Area ($px^2$) | Median Area Ratio ($\rho$) | Morphological Scale |
|:---:|:---:|:---|:---|:---:|:---:|:---:|:---:|:---:|
| 0 | SH | Short Circuit | Functional Electrical Bridging | 915 | 4.49% | 5,772.4 | 0.01409 | Macro / Meso |
| 1 | SP | Spur | Conductor Boundary Error | 4,584 | 22.49% | 1,282.5 | 0.00313 | Micro |
| 2 | SC | Spurious Copper | Incomplete Substrate Etching | 1,593 | 7.82% | 2,526.6 | 0.00617 | Meso |
| 3 | OP | Open Circuit | Conductor Trace Discontinuity | 1,770 | 8.68% | 3,816.4 | 0.00932 | Meso |
| 4 | MB | Mask Break | Dielectric Layer Void | 2,529 | 12.41% | 1,354.2 | 0.00331 | Micro |
| 5 | HB | Hole Break | Drill / Annular Via Fracture | 2,883 | 14.15% | 13,151.4 | 0.03211 | Macro |
| 6 | CS | Conductor Scratch | Mechanical Surface Abrasion | 2,490 | 12.22% | 25,437.8 | 0.06210 | Macro |
| 7 | CFO | Copper Foreign Object | Conductive Particulate Debris | 1,832 | 8.99% | 5,023.8 | 0.01227 | Meso |
| 8 | BMFO | Ball Micro Foreign Object | Solder Spatter / Micro-Debris | 1,680 | 8.24% | 2,088.0 | 0.00510 | Micro |
| Summary | ― | Total Multi-Scale Annotations | Comprehensive PCB Inspection | 20,376 | 100.0% | 4,120.0 | 0.01006 | Multi-Scale (Micro to Macro) |

*Median bounding box area and relative area ratio demonstrate the multi-scale distribution spanning micro-defects (<0.005) to macro-defects (>0.030).*

**Figure 2 | Defect class distribution and instance frequency across the nine target categories in DsPCBSD+.** The bar chart illustrates the sample imbalance between high-frequency boundary defects such as Spur (SP) and Hole Break (HB) versus rarer electrical bridging defects such as Short Circuit (SH).

#### 3.1.3 Data Partitioning and Label Integrity Verification
The dataset was partitioned following standard academic splitting protocols into training, validation, and test subsets with an approximate 70% / 20% / 10% distribution: 7,181 training images (~14,260 defect instances), 2,052 validation images (~4,075 defect instances), and 1,026 test images (~2,041 defect instances). All ground-truth annotations are stored in standardized YOLO normalized bounding box format, where each instance label is parameterized as a 5-tuple:

$$\mathbf{b} = \left( c, \, x_c, \, y_c, \, w, \, h \right) \tag{3}$$

where $c \in \{0, 1, \dots, 8\}$ denotes the integer class index, $(x_c, y_c) \in (0, 1)^2$ denotes the normalized center coordinates, and $(w, h) \in (0, 1]^2$ represents the normalized box width and height. Prior to training, an automated Exploratory Data Analysis (EDA) and annotation integrity verification across all 20,376 bounding box entries verified complete coordinate confinement, zero negative or zero-area bounding dimensions, and zero unlinked label files, confirming an uncompromised ground truth.

**Figure 3 | Visual ground-truth verification across representative PCB surface defect categories.** Sample bounding box overlays confirm spatial alignment accuracy, verifying zero label boundary overflow or inverted coordinates across complex circuit topologies.

---

### 3.2 Data Preprocessing

#### 3.2.1 Input Standardization and Tensor Normalization
Each input image is verified for channel order (converted from BGR to RGB). When input geometries deviate from the canonical 640 x 640 grid, aspect-ratio preserving letterbox resizing is applied according to equations (4) and (5):

$$s = \min\left(\frac{H_{\text{target}}}{H_0}, \, \frac{W_{\text{target}}}{W_0}\right) \tag{4}$$

$$H_{\text{new}} = \lfloor H_0 \cdot s \rfloor, \quad W_{\text{new}} = \lfloor W_0 \cdot s \rfloor \tag{5}$$

The image is rescaled via bilinear interpolation to $(H_{\text{new}}, W_{\text{new}})$, and remaining borders are symmetrically padded with neutral gray values (114/255). The normalized tensor values are scaled into the floating-point domain via equation (6):

$$\hat{\mathbf{I}}(x, y) = \frac{\mathbf{I}(x, y)}{255.0}, \quad \hat{\mathbf{I}}(x, y) \in [0.0, 1.0]^3 \tag{6}$$

#### 3.2.2 Mathematical Formulation of the Adaptive Resolution Router (ARR)
The foundational premise of our preprocessing methodology is that super-resolution enhancement should be treated as an on-demand, content-contingent operation rather than a universal pipeline stage. The routing operator maps incoming PCB images to either the direct bypass route or the super-resolution route under two operational paradigms: Training Mode (supervised, label-guided) and Inference Mode (unsupervised, perceptual feature-guided).

**A. Supervised Routing Formulation (Training / Calibration Mode):** During offline dataset preparation and model calibration, ground-truth bounding box dimensions are known. The relative area ratio $\rho_i$ of each bounding box and the minimum instance area ratio $\rho_{\min}$ are evaluated via equations (7) and (8):

$$\rho_i = w_i \cdot h_i = \frac{\text{Area}(\mathbf{b}_i)}{W \cdot H}, \quad \rho_i \in (0, 1] \tag{7}$$

$$\rho_{\min}(\mathbf{I}_k) = \min_{i \in \{1, \dots, N_k\}} \rho_i \tag{8}$$

$$\mathcal{R}_{\text{train}}(\mathbf{I}_k) = \begin{cases} \text{Super-Resolution (SR)}, & \text{if } \rho_{\min}(\mathbf{I}_k) < \tau_{\text{area}} \\ \text{Direct Route (Bypass)}, & \text{if } \rho_{\min}(\mathbf{I}_k) \ge \tau_{\text{area}} \end{cases} \tag{9}$$

Empirically, based on the inflection point of standard YOLOv11/YOLO26 feature pyramid downsampling ($P_3$ stride of 8 pixels), defects with $\rho_i < 0.005$ (corresponding to bounding box dimensions under approximately 45 x 45 pixels on a 640 x 640 grid) suffer severe feature attenuation. Hence, the threshold is calibrated to $\tau_{\text{area}} = 0.005$ (0.50% of total frame area) in equation (9).

**B. Unsupervised Multi-Criteria Perceptual Gate (Inference Mode):** During online production-line inspection, ground-truth annotations are unavailable. The router autonomously evaluates three complementary perceptual features: (1) Spatial Edge Density ($\mathcal{D}_{\text{edge}}$) via double-threshold Canny filtering in equations (10) and (11); (2) Local Contrast Variance ($\sigma_{\text{contrast}}$) using a Gaussian difference map in equations (12) and (13); and (3) High-Frequency Spectral Energy Ratio ($\eta_{\text{HF}}$) via 2D Fast Fourier Transform in equations (14) and (15):

$$\mathbf{E}(x, y) = \text{Canny}\left(\mathbf{I}_{\text{gray}}(x, y), \, T_{\text{low}}=50, \, T_{\text{high}}=150\right) \tag{10}$$

$$\mathcal{D}_{\text{edge}} = \frac{1}{H \cdot W} \sum_{x=1}^{W} \sum_{y=1}^{H} \mathbb{I}\left(\mathbf{E}(x, y) > 0\right) \tag{11}$$

$$\Delta_{\text{contrast}}(x, y) = \left| \mathbf{I}_{\text{gray}}(x, y) - \mathbf{I}_{\text{blur}}(x, y) \right| \tag{12}$$

$$\sigma_{\text{contrast}} = \sqrt{ \frac{1}{H \cdot W} \sum_{x=1}^{W} \sum_{y=1}^{H} \left( \Delta_{\text{contrast}}(x, y) - \bar{\Delta} \right)^2 } \tag{13}$$

$$\mathcal{F}(u, v) = \sum_{x=0}^{W-1} \sum_{y=0}^{H-1} \mathbf{I}_{\text{gray}}(x, y) \, e^{-j 2\pi \left( \frac{ux}{W} + \frac{vy}{H} \right)} \tag{14}$$

$$\eta_{\text{HF}} = \frac{\sum_{(u, v) \notin \Omega_{\text{low}}} \mathcal{S}(u, v)}{\sum_{u=0}^{W-1} \sum_{v=0}^{H-1} \mathcal{S}(u, v)} \tag{15}$$

Combining these three metrics, the unsupervised online routing decision is governed by the compound logical gating rule defined in equation (16):

$$\mathcal{R}_{\text{infer}}(\mathbf{I}) = \begin{cases} \text{Super-Resolution}, & \text{if } \left( \mathcal{D}_{\text{edge}} < \tau_{\text{edge}} \land \sigma_{\text{contrast}} < \tau_{\text{contrast}} \right) \lor \left( \eta_{\text{HF}} < \tau_{\text{HF}} \right) \\ \text{Direct Route}, & \text{otherwise} \end{cases} \tag{16}$$

Through validation-set cross-validation, the decision thresholds are calibrated to $\tau_{\text{edge}} = 0.030$, $\tau_{\text{contrast}} = 25.0$, and $\tau_{\text{HF}} = 0.700$. The perceptual gate executes in under 1.2 ms per frame on a CPU, adding negligible overhead while effectively pre-identifying images with micro-defect signatures.

**Figure 4 | Methodological workflow and decision logic of the Adaptive Resolution Router (ARR).** The perceptual tri-factor gating mechanism extracts spatial edge density, local contrast variance, and 2D FFT spectral energy ratio, routing candidate frames into either selective 2x super-resolution or direct inference.

#### 3.2.3 Super-Resolution Enhancement Strategies
Images channeled into the super-resolution route undergo 2x spatial magnification (640 x 640 -> 1280 x 1280) to reconstruct fine copper grain boundaries and micro-anomalies. Our pipeline supports three interchangeable super-resolution backends: (1) Enhanced Bicubic with Unsharp Masking for latency-critical deployments (<5 ms budget); (2) Deep CNN Super-Resolution (FSRCNN / ESPCN) running directly within the OpenCV DNN module in 12.4 ms; and (3) SwinIR-Lightweight for precision-first inspection stations. In the high-throughput mode, unsharp contrast enhancement is governed by equation (17):

$$\mathbf{I}_{\text{sharp}} = \mathbf{I}_{\text{bicubic}} + \alpha \cdot \left( \mathbf{I}_{\text{bicubic}} - \mathbf{I}_{\text{bicubic}} * \mathcal{G}_{\sigma} \right) \tag{17}$$

Following resolution enhancement, images are scaled to the standard detector input dimension (640 x 640) via anti-aliased area averaging. This embeds the high-frequency sub-pixel edge reconstructions directly into the canonical detector input tensor, enhancing feature saliency for downstream convolutional and transformer backbones.

#### 3.2.4 Empirical Routing Distribution and Latency Savings
The quantitative impact of the Adaptive Resolution Router across the full 10,259 images of the DsPCBSD+ dataset is summarized in Table 3.

**Table 3 | Quantitative performance and compute savings of Adaptive Resolution Routing on the DsPCBSD+ dataset (10,259 images).**

| Metric / Parameter | Value | Industrial Significance |
|:---|:---:|:---|
| Total Benchmark Images Evaluated | 10,259 | Full dataset coverage |
| Direct Route Frequency (Bypass SR) | 55.13% (5,656 images) | Over half the dataset avoids SR overhead |
| Selective SR Route Frequency | 44.87% (4,603 images) | Compute focused exclusively on micro-defects |
| Direct Computation Reduction | 55.13% | Significant reduction in GPU workload |
| All-SR Pipeline Total Processing Time | ~0.11 hours (396 s) | High latency, excessive power draw |
| Adaptive Pipeline Total Processing Time | ~0.05 hours (180 s) | >54% reduction in total pipeline run-time |
| Net Latency Reduction per Image | 22.1 ms / image | Essential for meeting real-time 30+ FPS criteria |

*Values reflect benchmarks conducted across the full 10,259-image set comparing blanket super-resolution with adaptive routing.*

Furthermore, Table 4 illustrates how the router behaves across each individual defect category, demonstrating its strong correlation with physical defect dimensions.

**Table 4 | Per-class routing breakdown showing correlation between physical defect scale and super-resolution invocation.**

| Class Code | Class Name | Total Instances | Direct Route | SR Route | SR Route (%) | Median Area Ratio | Dominant Scale Regime |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---|
| SP | Spur | 4,584 | 565 | 4,019 | 87.67% | 0.00313 | Micro (High SR Trigger) |
| MB | Mask Break | 2,529 | 583 | 1,946 | 76.95% | 0.00331 | Micro (High SR Trigger) |
| SH | Short Circuit | 915 | 393 | 522 | 57.05% | 0.01409 | Meso / Micro |
| BMFO | Ball Micro Foreign Object | 1,680 | 724 | 956 | 56.90% | 0.00510 | Micro (FOD Target) |
| SC | Spurious Copper | 1,593 | 746 | 847 | 53.17% | 0.00617 | Meso |
| OP | Open Circuit | 1,770 | 852 | 918 | 51.86% | 0.00932 | Meso |
| CFO | Copper Foreign Object | 1,832 | 952 | 880 | 48.03% | 0.01227 | Meso / Micro |
| CS | Conductor Scratch | 2,490 | 1,785 | 705 | 28.31% | 0.06210 | Macro (Low SR Trigger) |
| HB | Hole Break | 2,883 | 2,517 | 366 | 12.70% | 0.03211 | Macro (Bypasses SR) |

*Categories with low median area ratios (<0.005) exhibit high SR triggering, whereas macro defects predominantly bypass enhancement.*

**Figure 5 | Per-class selective routing distribution under the Adaptive Resolution Router on the DsPCBSD+ benchmark.** The distribution shows the contrast between micro-scale defects (Spurs and Mask Breaks, >75% SR triggering) and macro-scale anomalies (Hole Breaks, 87.3% direct bypass).

#### 3.2.5 Data Augmentation Pipeline
To ensure model robustness against production-line environmental variations (such as lighting shifts, optical reflections, and minor conveyor angle deviations), the training pipeline applies a stochastic multi-scale augmentation suite: (1) Mosaic Augmentation ($p = 1.0$); (2) MixUp Regularization ($p = 0.15$) blending two image tensors via linear convex combination in equation (18):

$$\tilde{\mathbf{I}} = \lambda \mathbf{I}_a + (1 - \lambda)\mathbf{I}_b, \quad \lambda \sim \text{Beta}(\alpha=0.5, \beta=0.5) \tag{18}$$

(3) Geometric Affine Transformations including rotation, isotropic scaling, and horizontal/vertical translations; (4) Random Horizontal and Vertical Flips ($p = 0.5$); and (5) Photometric Color Jitter in HSV color space to simulate factory lighting variations and solder mask tint deviations across manufacturing batches.

*[Note: Subsections 3.3 YOLO26n, 3.4 RT-DETR-L, 3.5 Adaptive Resolution Routing Architecture, 3.6 Weighted Box Fusion, and 3.7 Model Compression will be integrated seamlessly upon consolidation of teammate modules].*

---

## Data Availability
The DsPCBSD+ benchmark dataset analyzed during the current study is publicly available in the *Scientific Data* repository (Nature Portfolio) at https://doi.org/10.1038/s41597-024-03657-z under the Creative Commons Attribution 4.0 International License (CC BY 4.0). All preprocessed data variants and routing decision tables generated during this study are included in this published article and its supplementary information files.

## Author Contributions
A. conceptualized the research framework, formulated the Adaptive Resolution Router, performed Exploratory Data Analysis, developed the dual-track detection and Weighted Box Fusion pipeline, and drafted the manuscript. T.1 contributed to model compression, pruning, and INT8 quantization benchmarks. T.2 contributed to RT-DETR training and validation experiments. All authors reviewed and approved the final manuscript.

## Additional Information
**Competing Interests:** The authors declare no competing financial or non-financial interests.

## Acknowledgements
The authors acknowledge the School of Computer Science and Engineering, Vellore Institute of Technology (VIT), Vellore, for providing the computational resources and laboratory infrastructure to conduct this research.

---

## References

1. Ye, X. et al. A lightweight PCB defect detection algorithm based on an improved YOLOv11s. *IEEE Access* **14**, 56413–56426 (2026).
2. Li, Y. et al. Lightweight PCB defect detection method based on SCF-YOLO. *PLoS ONE* **20**, e0318033 (2025).
3. Kong, X., Liu, G. & Gao, Y. GESC-YOLO: improved lightweight printed circuit board defect detection based algorithm. *Sensors* **25**, 3052 (2025).
4. Chu, Y., Liu, X., Zhang, X. & Liu, H. A lightweight PCB defect detection method based on group convolution and adaptive pruning. *PeerJ Comput. Sci.* **11**, e3243 (2025).
5. Yin, X., Zhao, Z. & Weng, L. MAS-YOLO: a lightweight detection algorithm for PCB defect detection based on improved YOLOv12. *Appl. Sci.* **15**, 6238 (2025).
6. Selvam, P. et al. YOLO-DefXpert: an advanced defect detection on PCB surfaces using improved YOLOv11 algorithm. *IEEE Access* **13**, 143086–143101 (2025).
7. Chen, Z. & Liu, B. A high-accuracy PCB defect detection algorithm based on improved YOLOv12. *Symmetry* **17**, 978 (2025).
8. Chen, W., Ke, X. & Meng, S. Small defect detection in printed circuit boards based on the multiscale edge strengthening and an improved YOLOv10. *Sci. Rep.* **15**, 36445 (2025).
9. Wang, A., Yan, J. & Wang, Z. PCB defect detection based on improved YOLOv8n. In *Proc. 4th Int. Conf. Artif. Intell. Comput. Inf. Technol. (AICIT)* 11282815 (IEEE, 2025).
10. Solovyev, R., Wang, W. & Gabruseva, T. Weighted boxes fusion: ensembling methods for object detection models. *Image Vis. Comput.* **107**, 104117 (2021).
11. Zhao, M., Xu, H., Biibosunova, S. & Zhang, S. PCB board defect detection algorithm based on improved YOLOv8. In *Proc. 10th Int. Symp. Adv. Electr. Electron. Comput. Eng. (ISAEECE)* 11160235 (IEEE, 2025).
12. Wei, A. NeoPCB-YOLO: next-generation enhanced optimization YOLO for PCB. In *Proc. 5th Int. Conf. Electron. Inf. Eng. Comput. Sci. (EIECS)* 11283608 (IEEE, 2025).
13. He, X. & Xie, M. ABF-YOLO: a fine-grained PCB defect detection framework integrating axial attention and bidirectional feature fusion. *IEEE Trans. Consum. Electron.* **71**, 8562–8570 (2025).
14. Mounika, M., Gokulprasanth, M. & Vadivu, G. YOLOV8X-HICAUPS: an attention based approach for accurate PCB defect detection. *Int. J. Eng. Appl. Sci. Technol.* **9**, 80–92 (2025).
15. Huang, J., Zhao, F. & Chen, L. Defect detection network in PCB circuit devices based on GAN enhanced YOLOv11. *Appl. Comput. Eng.* **42**, 115–122 (2025).
16. Elsharkawy, Z. F. Enhanced YOLOv11 framework for high precision defect detection in printed circuit boards. *Sci. Rep.* **15**, 42550 (2025).
17. Wei, Z., Yang, F., Zhong, K. & Yao, L. PCB-YOLO: enhancing PCB surface defect detection with coordinate attention and multi-scale feature fusion. *PLoS ONE* **20**, e0323684 (2025).
18. Song, C. et al. YOLO-HDEW: an efficient PCB defect detection model. *Electronics* **14**, 3383 (2025).
19. Yu, J., Tang, J. & Tang, Z. Research on lightweight PCB defect detection algorithm based on YoloV11. *Int. J. Comput. Appl. Technol. Res.* **14**, 1–5 (2025).
20. Niaz, A., Umraiz, M., Soomro, S. & Choi, K. N. Vision transformer and Mamba-attention fusion for high-precision PCB defect detection. *PLoS ONE* **20**, e0331175 (2025).
21. Lv, S., Zhang, H., Li, W. & Wang, Z. DsPCBSD+: a comprehensive dataset for deep-learning-based PCB surface defect detection. *Sci. Data* **11**, 811; 10.1038/s41597-024-03657-z (2024).
22. Zhu, H., Zhang, Y. & Chen, X. VR-YOLO: enhancing PCB defect detection with viewpoint robustness. Preprint at https://arxiv.org/abs/2501.09842 (2025).
23. Niu, Y., Sun, L. & Jiang, W. A lightweight transformer for PCB defects detection (LPDETR). In *Proc. IEEE Int. Conf. Frontier AI Smart Technol. Appl. (FASTA)* 1–6 (IEEE, 2025).
24. Nguyen, V.-T., Tran, T.-H. & Le, D.-T. Deep learning-enhanced defects detection for printed circuit boards. *Results Eng.* **25**, 103211 (2025).
25. Dayıoğlu, M., Çetin, S. & Erdem, E. Performance analysis of YOLO11 models in PCB defect detection. *J. North. Aegean Tech. Sci.* **4**, 45–56 (2025).
26. Sun, J., Wang, K. & Zhao, D. Defect detection of PCB based on improved YOLOv11. In *Proc. 4th Int. Conf. Manag. Sci. Ind. Econ. Inf. Technol. (MSIEID)* 312–319 (Atlantis Press, 2025).
27. Wang, Y., Liu, B. & Chen, F. PCB defect detection based on lightweight DL fusion model. *Sensors* **25**, 2489 (2025).
28. Zhang, R., Qian, Z. & Wu, H. MSA-DETR: multi-scale alignment detection transformer for PCB surface defect inspection. *Meas. Sci. Technol.* **36**, 045102 (2025).
29. Hsiao, C.-H., Lin, Y.-C. & Wu, T.-H. ResNet-SE-CBAM Siamese networks for few-shot PCB defect classification. *Sensors* **25**, 892 (2025).
30. Shao, X., Huang, Y. & Meng, L. SM-FSOD: meta-learning for few-shot PCB defect detection. *Electronics* **14**, 1421 (2025).
