Adaptive Resolution Routing and Dual-Track CNN-Transformer Fusion for High-Throughput Printed Circuit Board Surface Defect and Foreign Object Debris Detection

Abhinav1,*, Teammate 11 & Teammate 21

1School of Computer Science and Engineering, Vellore Institute of Technology (VIT), Vellore 632014, Tamil Nadu, India.
*Corresponding author: email: abhinav.project@vit.ac.in

Abstract

Automated Optical Inspection (AOI) of High-Density Interconnect Printed Circuit Boards (PCBs) faces a critical trade-off between inspection throughput and multi-scale defect detection fidelity. While lightweight convolutional neural networks (CNNs) offer high frame rates, they exhibit elevated miss rates on microscopic anomalies and Foreign Object Debris (FOD). Conversely, Vision Transformers provide global contextual awareness across intricate copper traces but impose prohibitive latency and memory overhead on industrial edge platforms. Furthermore, applying super-resolution (SR) universally across all frames creates severe computational bottlenecks. Here, we present an edge-deployable, high-throughput PCB defect detection framework featuring an Adaptive Resolution Router (ARR) that dynamically evaluates incoming frames via a multi-criteria perceptual complexity gate. Across the 10,259-image DsPCBSD+ benchmark, the router bypasses super-resolution for 55.13% of frames containing macro-defects while selectively enhancing the remaining 44.87% containing micro-defects, saving 22.1 ms per image (55.13% compute reduction). An asymmetric dual-track detector pairing YOLO26n (3.9 ms) and RT-DETR-L (28.0 ms) harmonized via Weighted Box Fusion achieves 86.63% recall and a peak F1-score of 0.6493, eliminating 17 false alarms with zero recall loss. Post-training INT8 quantization yields up to 3.87x compression, confirming real-time feasibility on embedded edge inspection hardware.

Keywords: Printed circuit boards, Automated optical inspection, Surface defect detection, Foreign object debris, Vision transformers, Edge artificial intelligence.

1. Introduction

1.1 Industrial Context and the Quality Imperative in Electronics Manufacturing

Printed Circuit Boards (PCBs) serve as the fundamental structural and electrical backbone of virtually all modern electronic systems, spanning mission-critical aerospace avionics, autonomous automotive control units, biomedical implants, industrial telecommunications, and ubiquitous consumer electronics. With the aggressive scaling of high-density interconnect (HDI) substrates, multi-layer micro-via architectures, and ultra-fine-pitch surface-mount technologies (SMT), line widths and trace clearances have contracted to sub-100-micrometer tolerances [1, 2]. Under these sub-millimeter regimes, the structural and electrical integrity of PCB copper conductors, solder masks, and insulating substrate layers is exceedingly vulnerable to manufacturing variations and physical contamination.

Surface defects occurring during fabrication (e.g., etching, plating, and solder masking) manifest in diverse morphological archetypes. These include severe electrical discontinuities such as open circuits (OP) and short circuits (SH); conductor degradation such as copper spurs (SP), spurious residual copper (SC), and fine conductor scratches (CS); and protective barrier failures such as solder mask breaks (MB) and hole breaks (HB) [2, 4]. Simultaneously, foreign particulate contamination—specifically metallic Foreign Object Debris (FOD), such as stray copper particles (CFO) and microscopic solder balls or ball-shaped micro foreign objects (BMFO)—poses an insidious operational risk. Foreign conductive particulate matter can migrate under operational thermal cycling and mechanical vibration, inducing intermittent shorting, arc faults, and catastrophic field failures long after post-production deployment. Consequently, continuous, high-speed, non-destructive Automated Optical Inspection (AOI) represents an indispensable quality control gate across smart electronics manufacturing lines (Industry 4.0) [1, 3].

1.2 Limitations of Conventional AOI and Early Deep Learning Systems

Historically, industrial AOI systems have relied predominantly on rule-based machine vision pipelines and classical template matching [24]. These traditional approaches align incoming production images with an idealized golden board reference and compute defect candidate regions via structural differencing, morphological gradient operators, and hand-engineered color-space thresholding. While computationally modest, rule-based AOI systems exhibit profound fragility under real-world factory dynamics: (1) high pseudo-defect rates, where minor non-defective variations such as solder mask color shifts, non-uniform coaxial lighting gradients, micro-vibrations of conveyor belts, and negligible component placement tolerances are frequently flagged as anomalies, overwhelming human operators with false alarm rates often exceeding 20–30% [1, 6]; (2) inflexibility to product iterations, where any alteration to trace geometries or board layouts necessitates manual recalibration and golden template re-registration; and (3) inability to generalize across morphological variance, as deterministic geometric thresholds fail to accurately categorize stochastic defects, such as irregular particulate debris or subtle copper scratches exhibiting ill-defined boundaries.

To transcend these limitations, deep learning algorithms—particularly Convolutional Neural Networks (CNNs)—have been widely adopted for PCB defect localization [3, 5, 8]. Early applications utilized two-stage detectors (e.g., Faster R-CNN, Cascade R-CNN), which achieved respectable detection precision but suffered from high computational latency (>80 ms per frame), rendering them unsuitable for continuous multi-meter-per-second surface inspection lines. In response, single-stage object detectors from the You Only Look Once (YOLO) lineage (YOLOv5 through YOLOv11/v12) emerged as the dominant industrial paradigm due to their unified regression formulations and ultra-low inference latencies [1, 5, 6, 25].

1.3 The Scale Disparity Dilemma and the Super-Resolution Bottleneck

Despite their operational speed, modern single-stage CNNs confront a severe performance degradation when applied to PCB inspection due to extreme scale disparity. As documented in modern industrial benchmarks such as DsPCBSD+ [21], defect surface areas exhibit dynamic ranges spanning multiple orders of magnitude:

Macro-scale defects, such as hole breaks (HB) and large short circuits, occupy several thousand pixels (>= 2.5% to 6.0% of total image area), presenting distinct geometric contours that standard downsampling backbones capture readily. In stark contrast, micro-defects—such as ball-shaped micro foreign objects (BMFO), fine copper spurs (SP), and conductor scratches (CS)—often span fewer than 10 x 10 pixels, representing less than 0.5% of the total frame. When input images are downsampled to canonical detector resolutions (e.g., 640 x 640 pixels), the successive strided convolutions and spatial pooling operations of standard CNN backbones irreversibly attenuate the high-frequency spatial gradients of these microscopic anomalies. Consequently, standard CNNs exhibit high false negative rates (misses) on tiny defects.

To mitigate this loss of feature resolution, prior researchers have explored two primary avenues: (1) dense slicing or tiling (e.g., Slicing Aided Hyper Inference - SAHI), which inflates inference latency linearly by a factor of N_tiles (often 4x to 16x); and (2) universal deep super-resolution (SR), which applies deep enhancement (e.g., SwinIR, Real-ESRGAN) to every image, adding 50–250 ms per frame. Crucially, existing literature applies super-resolution indiscriminately across all frames. In an industrial inspection batch, more than half of the processed boards contain either macro-defects or clean conductive traces where deep super-resolution yields no detectable accuracy gain, resulting in massive GPU compute wastage and thermal throttling on edge systems.

1.4 The CNN vs. Vision Transformer Duality in Defect Inspection

Concurrently, the emergence of Vision Transformers (ViTs) and Detection Transformers (e.g., DETR, Deformable-DETR, RT-DETR) has offered a powerful alternative paradigm [20, 22, 28]. Unlike CNNs, which are constrained by local receptive fields, RT-DETR (Real-Time Detection Transformer) leverages multi-scale cross-attention mechanisms that model long-range spatial dependencies across complex conductive trace networks. This global context enables RT-DETR to differentiate true electrical track continuities from subtle opens (OP) and shorts (SH) with exceptional recall and fine-grained localization accuracy [23]. However, RT-DETR's transformer encoder-decoder layers require substantial floating-point operations (FLOPs) and high VRAM capacity, exhibiting latencies of 25–35 ms on high-end GPUs—nearly an order of magnitude slower than lightweight CNNs (3–5 ms) and excessively heavy for low-power edge gateways.

Thus, industrial AOI confronts a fundamental architectural dilemma: lightweight CNNs offer the required frame rates but miss micro-scale defects, whereas Vision Transformers offer superior recall and global relational context but exceed edge hardware thermal and latency budgets.

1.5 Research Contributions

To resolve this multi-scale, latency-constrained dilemma, this paper proposes an Adaptive Resolution-Aware Dual-Track Detection Framework specifically designed for high-precision, real-time PCB surface defect and foreign object debris inspection. The core contributions of this work are summarized as follows:

1. Lightweight Adaptive Resolution Router (ARR): We formulate a content-adaptive, complexity-aware preprocessing gate that dynamically determines whether an incoming PCB frame requires resolution enhancement prior to inference. By extracting spatial edge density, local contrast standard deviation, and high-frequency spectral energy via 2D Fast Fourier Transform, the ARR routes 55.13% of frames through a high-speed direct pathway and selectively enhances only the 44.87% exhibiting micro-defect characteristics. This eliminates computational redundancy, yielding an average latency reduction of 22.1 ms per frame (55.13% compute saving) over universal super-resolution baselines.

2. Asymmetric Dual-Track CNN-Transformer Architecture: We design an asymmetric dual-track detection engine that executes inference in parallel across two structurally complementary paradigms: Track 1 (Fast CNN: fine-tuned YOLO26n, 3.9 ms GPU latency, 2.507M parameters, 14.85 MB PyTorch / 9.36 MB ONNX) for rapid localized defect extraction; and Track 2 (Contextual Transformer: fine-tuned RT-DETR-L, 28.0 ms GPU latency, 32.825M parameters) for long-range topological trace modeling.

3. Complementary Late Fusion via Weighted Box Fusion (WBF): Rather than relying on traditional greedy Non-Maximum Suppression (NMS)—which discards redundant bounding boxes and ignores multi-model spatial agreement—we implement an asymmetric confidence-weighted box fusion scheme (w_DETR = 1.5, w_YOLO = 1.0). The fusion pipeline achieves an optimal balance between precision and recall, securing a peak F1-score of 0.6493 and eliminating 17 false positives present in standalone transformer predictions with zero recall degradation (maintaining 86.63% recall).

4. Edge-Constrained Quantization and Real-Time Benchmarking: We validate the deployability of our dual-track pipeline on constrained edge hardware via post-training dynamic INT8 quantization and structured pruning. The quantized models achieve a 3.41x compression ratio for YOLO26n (2.74 MB) and 3.87x for RT-DETR-L (32.45 MB), proving that dual-track fusion can be seamlessly hosted within the physical memory and latency constraints of industrial embedded PCs and edge accelerators.

1.6 Organization of the Article

The remainder of this manuscript is organized as follows: Section 2 surveys recent advancements in deep learning for PCB inspection, micro-defect super-resolution, ensemble fusion strategies, and edge compression techniques, establishing key research gaps. Section 3 details the experimental dataset (DsPCBSD+), statistical class distributions, the mathematical formulations governing the proposed Adaptive Resolution Router, individual detector architectures (YOLO26n and RT-DETR-L), Weighted Box Fusion, model compression, experimental setup, and evaluation metrics. Section 4 presents comprehensive empirical findings across detector baselines, fusion ablation, adaptive routing savings, pruning, INT8 quantization, and hardware deployment latencies. Section 5 discusses architectural behaviors, fusion mechanisms, computational cost trade-offs, industrial deployment implications, literature comparisons, and research limitations. Section 6 concludes the study.


> **Figure 1 | End-to-end architecture of the proposed dual-track PCB defect inspection framework. Incoming frames are evaluated by the Adaptive Resolution Router (ARR), which selectively triggers super-resolution for micro-scale defects while bypassing enhancement for macro-scale anomalies. Parallel detection is executed via YOLO26n and RT-DETR-L, merged by Weighted Box Fusion (WBF), and accelerated through INT8 edge quantization.**

2. Related Work / Literature Review

2.1 Deep Learning for PCB Automated Optical Inspection: The CNN vs. Transformer Evolution

The development of modern deep-learning-based AOI systems has been largely catalyzed by successive iterations of the YOLO framework. Ye et al. [1] proposed an improved lightweight YOLOv11s model for PCB surface defect detection, incorporating channel and spatial attention mechanisms alongside structured parameter pruning to address edge memory limitations. Their system demonstrated high detection rates on standard benchmarks (DeepPCB and DsPCBSD+) while emphasizing the necessity of model compression for industrial shop-floor integration. Li et al. [2] introduced SCF-YOLO, a lightweight architecture utilizing a MobileNet-style backbone and feature-split shuffle blocks, combined with response-based knowledge distillation from a larger YOLOv8-l teacher network. While SCF-YOLO achieved a 25% parameter reduction and maintained high precision on the PKU-PCB dataset, its detection accuracy degraded noticeably on irregular particulate debris (such as FOD) due to the reduced representation capacity of depthwise separable convolutions.

Concurrently, researchers have sought to enhance standard YOLO backbones with specialized attention mechanisms to counteract small-defect attenuation. Kong et al. [3] formulated GESC-YOLO, which integrates a C2f-GE feature extraction block, GSConv slim-neck layers, and Coordinate Attention (CA) into YOLOv8n, reporting up to 99% mAP@50 on balanced test sets. Chu et al. [4] investigated group convolutions and adaptive pruning via the LAMP algorithm, demonstrating that up to 35% of redundant convolutional weights in YOLOv8n could be removed with minimal accuracy loss on the PKU-Market-PCB dataset. In more recent explorations of next-generation backbones, Yin et al. [5] proposed MAS-YOLO, representing one of the earliest adaptations of YOLOv12 for PCB inspection, utilizing Multi-Scale Edge-Context Saliency (MECS) attention and an Asymmetric Heterogeneous Feature Interaction Neck (AHFIN). Selvam et al. [6] introduced YOLO-DefXpert, integrating Swin Transformer blocks, Deformable Convolutions (DCNv2), and the Convolutional Block Attention Module (CBAM) into YOLOv11 to improve defect boundaries. Chen and Liu [7] designed a high-accuracy YOLOv12 variant incorporating BiFPN and Normalized Wasserstein Distance (NWD) loss. Chen, Ke, and Meng [8] developed MS-YOLO (based on YOLOv10), integrating Multiscale Edge Strengthening (MSES) modules and Space-to-Depth Convolutions (SPDConv) alongside dual-distillation.

Despite these convolutional advances, pure CNN architectures remain fundamentally bounded by localized receptive fields. When tracing extensive copper paths across a circuit board to detect discontinuous open circuits or subtle bridging, CNNs must stack dozens of layers to achieve sufficient contextual coverage, resulting in gradient degradation and feature blurring. To overcome this, Vision Transformers have been introduced into industrial inspection. Niaz et al. [20] proposed a hybrid architecture coupling Vision Transformers with Mamba state-space attention, achieving 99.69% mAP through selective state-space sequence modeling of PCB surface features. Niu et al. [23] developed LPDETR, a lightweight variant of the Real-Time Detection Transformer (RT-DETR) that reduced parameter count by ~30% while retaining the global self-attention mechanism. Zhang et al. [28] designed MSA-DETR, incorporating multi-scale alignment feature pyramids to align spatial features across transformer decoding layers. However, all existing ViT and DETR implementations exhibit substantial computational latency (typically 25–45 ms), which remains an obstacle for continuous, high-speed AOI lines operating at >60 frames per second.

2.2 Micro-Defect Localization and Resolution Enhancement Strategies

A critical vulnerability shared by both CNN and Transformer detectors is the severe performance drop encountered on micro-scale defects. On standard benchmarks such as DsPCBSD+, tiny defects like copper spurs (SP), fine conductor scratches (CS), and ball micro foreign objects (BMFO) occupy less than 0.5% of the total pixel area [21]. To restore lost spatial high-frequency details, researchers have investigated super-resolution (SR) networks as neural image enhancers. Modern deep SR frameworks—including SwinIR, Real-ESRGAN, and RCAN—employ deep residual networks or shifted-window self-attention to reconstruct sub-pixel edges, high-frequency textures, and sharp transition boundaries. In industrial domains, applying deep SR prior to detection has been shown to improve small-object mAP by 3.5% to 7.0%.

However, deep super-resolution is computationally demanding. For instance, executing SwinIR-light on a 640 x 640 image consumes 50–120 ms on an embedded GPU—far exceeding the inference latency of the object detector itself. Critically, every existing study in the literature applies resolution enhancement uniformly across all inspection images [8, 14, 15]. In production environments, PCB images arrive in high-volume streams where over 50% of frames contain either macro-defects or standard conductor tracks that do not benefit from super-resolution. Applying deep SR indiscriminately to every image creates an unnecessary computational bottleneck. To date, no prior research has investigated an adaptive, content-aware routing gate that inspects low-level structural image complexity prior to detection to selectively trigger resolution enhancement only when microscopic anomalies are present.

2.3 Multi-Detector Ensembles and Late Fusion Techniques

To combine the complementary strengths of different detection models, ensemble learning is widely employed in competitive computer vision. In PCB inspection, CNNs offer high localization speed and high precision on high-contrast edges, whereas Vision Transformers excel at contextual recall and resolving ambiguous, low-contrast traces. Traditionally, multi-detector predictions are merged using Greedy Non-Maximum Suppression (NMS) or its variant, Soft-NMS. However, standard NMS possesses inherent structural flaws in multi-model settings: it assumes that the single bounding box with the highest confidence score represents the most accurate spatial ground truth, discarding all overlapping candidate boxes whose Intersection-over-Union (IoU) exceeds a threshold. If a detector incorrectly assigns high confidence to a false alarm, NMS preserves the error while discarding lower-confidence true positive predictions from secondary models.

To overcome these deficiencies, Solovyev et al. introduced Weighted Box Fusion (WBF) [10]. Rather than discarding overlapping bounding boxes, WBF utilizes the confidence scores of all matching predictions across models to compute an averaged, confidence-weighted bounding box coordinate:

While WBF has been utilized in aerial imagery and general object detection competitions, its application to industrial PCB defect detection—specifically as an asymmetric bridge between an ultra-lightweight CNN (YOLO) and a contextual hybrid transformer (RT-DETR)—remains largely unexplored.

2.4 Edge Model Compression, Pruning, and Quantization

Deploying deep learning models on factory-floor Automated Optical Inspection equipment requires conformance to strict SWaP (Size, Weight, and Power) constraints. Production line cameras are typically connected to industrial personal computers (IPCs) or embedded accelerators (such as NVIDIA Jetson Orin/Nano or Intel OpenVINO edge gateways) possessing limited GPU VRAM (4GB–8GB) and strict thermal dissipation limits [1, 4, 23]. Model compression techniques primarily encompass structured pruning (e.g., L1-norm or LAMP algorithm [4]) and Post-Training Quantization (PTQ) from FP32 to FP16 and INT8. Quantization to INT8 reduces the model's memory footprint by up to 4x, lowers memory bandwidth pressure, and enables the utilization of specialized integer tensor cores. While individual papers have explored pruning on YOLOv8 [4] or knowledge distillation on YOLOv11 [1, 2], there is a distinct lack of comprehensive empirical research that benchmarks the end-to-end deployment pipeline—combining adaptive resolution routing, dual-track CNN-transformer inference, WBF late fusion, and full INT8 quantization—under uniform industrial edge constraints.

2.5 Summary of Research Gaps and Theoretical Positioning

Table 1 provides a systematic comparison between recent representative studies in PCB surface defect detection (2024–2026) and our proposed framework.


**Table 1 | Comprehensive comparison of recent state-of-the-art PCB defect detection studies versus the proposed work.**

Comparative assessment across model paradigm, dataset diversity, accuracy, preprocessing adaptability, fusion strategy, and edge compression validation.

3. Materials and Methods

3.1 Dataset

3.1.1 Overview and Source Verification

To evaluate the proposed framework under rigorous industrial conditions, we utilize the DsPCBSD+ (Deep-learning-based PCB Surface Defect Detection Dataset Plus) benchmark [21]. The dataset was established by Lv et al. and published in Scientific Data (Nature Portfolio, 2024, DOI: 10.1038/s41597-024-03657-z), released under the Creative Commons Attribution 4.0 International License (CC BY 4.0) for open scientific benchmarking. DsPCBSD+ represents one of the largest and most comprehensive publicly available datasets dedicated to automated optical inspection of printed circuit boards. It comprises 10,259 high-resolution images, each captured at a standardized dimension of 640 x 640 pixels in 24-bit true color RGB format. The physical imaging was executed using an industrial telecentric optical inspection system equipped with a high-resolution CCD area scan camera and a coaxial LED illumination dome, specifically engineered to eliminate non-uniform specular reflections from bare copper traces and solder pads.

3.1.2 Defect Taxonomies and Morphological Characterization

Unlike earlier PCB datasets (such as DeepPCB or PKU-PCB) that are restricted to six canonical etching defects, DsPCBSD+ provides exhaustive annotations across nine distinct industrial defect classes. Importantly, the dataset explicitly incorporates particulate contamination and Foreign Object Debris (FOD), which are critical failure modes in surface-mount production lines: (1) Short Circuit (SH), an unintended bridge of conductive material connecting isolated traces; (2) Spur (SP), a localized protrusion of excess copper extending outward from trace boundaries; (3) Spurious Copper (SC), isolated islands of unetched copper on dielectric substrates; (4) Open Circuit (OP), a complete physical fracture along a conductive trace; (5) Mask Break (MB), a void in the protective solder mask exposing underlying copper; (6) Hole Break (HB), an off-center or misaligned drill via fracturing through an annular ring; (7) Conductor Scratch (CS), mechanical abrasions thinning trace cross-sections; (8) Copper Foreign Object (CFO), extraneous conductive copper debris adhering to the board; and (9) Ball Micro Foreign Object (BMFO), microscopic spherical solder spatter or particulate debris.


**Table 2 | Statistical breakdown and morphological scale characterization of defect classes in the DsPCBSD+ dataset.**

Median bounding box area and relative area ratio demonstrate the multi-scale distribution spanning micro-defects (<0.005) to macro-defects (>0.030).


> **Figure 2 | Defect class distribution and instance frequency across the nine target categories in DsPCBSD+. The bar chart illustrates the sample imbalance between high-frequency boundary defects such as Spur (SP) and Hole Break (HB) versus rarer electrical bridging defects such as Short Circuit (SH).**

3.1.3 Data Partitioning and Label Integrity Verification

The dataset was partitioned following standard academic splitting protocols into training, validation, and test subsets with an approximate 70% / 20% / 10% distribution: 7,181 training images (~14,260 defect instances), 2,052 validation images (~4,075 defect instances), and 1,026 test images (~2,041 defect instances). All ground-truth annotations are stored in standardized YOLO normalized bounding box format, where each instance label is parameterized as a 5-tuple:

where c in {0, 1, ..., 8} denotes the integer class index, (x_c, y_c) in (0, 1)^2 denotes the normalized center coordinates, and (w, h) in (0, 1]^2 represents the normalized box width and height. Prior to training, an automated Exploratory Data Analysis (EDA) and annotation integrity verification across all 20,376 bounding box entries verified complete coordinate confinement, zero negative or zero-area bounding dimensions, and zero unlinked label files, confirming an uncompromised ground truth.


> **Figure 3 | Visual ground-truth verification across representative PCB surface defect categories. Sample bounding box overlays confirm spatial alignment accuracy, verifying zero label boundary overflow or inverted coordinates across complex circuit topologies.**

3.2 Data Preprocessing

3.2.1 Input Standardization and Tensor Normalization

Each input image is verified for channel order (converted from BGR to RGB). When input geometries deviate from the canonical 640 x 640 grid, aspect-ratio preserving letterbox resizing is applied according to equations (4) and (5):

The image is rescaled via bilinear interpolation to (H_new, W_new), and remaining borders are symmetrically padded with neutral gray values (114/255). The normalized tensor values are scaled into the floating-point domain via equation (6):

3.2.2 Mathematical Formulation of the Adaptive Resolution Router (ARR)

The foundational premise of our preprocessing methodology is that super-resolution enhancement should be treated as an on-demand, content-contingent operation rather than a universal pipeline stage. The routing operator maps incoming PCB images to either the direct bypass route or the super-resolution route under two operational paradigms: Training Mode (supervised, label-guided) and Inference Mode (unsupervised, perceptual feature-guided).

A. Supervised Routing Formulation (Training / Calibration Mode): During offline dataset preparation and model calibration, ground-truth bounding box dimensions are known. The relative area ratio rho_i of each bounding box and the minimum instance area ratio rho_min are evaluated via equations (7) and (8):

Empirically, based on the inflection point of standard YOLOv11/YOLO26 feature pyramid downsampling (P_3 stride of 8 pixels), defects with rho_i < 0.005 (corresponding to bounding box dimensions under approximately 45 x 45 pixels on a 640 x 640 grid) suffer severe feature attenuation. Hence, the threshold is calibrated to tau_area = 0.005 (0.50% of total frame area) in equation (9).

B. Unsupervised Multi-Criteria Perceptual Gate (Inference Mode): During online production-line inspection, ground-truth annotations are unavailable. The router autonomously evaluates three complementary perceptual features: (1) Spatial Edge Density (D_edge) via double-threshold Canny filtering in equations (10) and (11); (2) Local Contrast Variance (sigma_contrast) using a Gaussian difference map in equations (12) and (13); and (3) High-Frequency Spectral Energy Ratio (eta_HF) via 2D Fast Fourier Transform in equations (14) and (15):

Combining these three metrics, the unsupervised online routing decision is governed by the compound logical gating rule defined in equation (16):

Through validation-set cross-validation, the decision thresholds are calibrated to tau_edge = 0.030, tau_contrast = 25.0, and tau_HF = 0.700. The perceptual gate executes in under 1.2 ms per frame on a CPU, adding negligible overhead while effectively pre-identifying images with micro-defect signatures.


> **Figure 4 | Methodological workflow and decision logic of the Adaptive Resolution Router (ARR). The perceptual tri-factor gating mechanism extracts spatial edge density, local contrast variance, and 2D FFT spectral energy ratio, routing candidate frames into either selective 2x super-resolution or direct inference.**

3.2.3 Super-Resolution Enhancement Strategies

Images channeled into the super-resolution route undergo 2x spatial magnification (640 x 640 -> 1280 x 1280) to reconstruct fine copper grain boundaries and micro-anomalies. Our pipeline supports three interchangeable super-resolution backends: (1) Enhanced Bicubic with Unsharp Masking for latency-critical deployments (<5 ms budget); (2) Deep CNN Super-Resolution (FSRCNN / ESPCN) running directly within the OpenCV DNN module in 12.4 ms; and (3) SwinIR-Lightweight for precision-first inspection stations. In the high-throughput mode, unsharp contrast enhancement is governed by equation (17):

Following resolution enhancement, images are scaled to the standard detector input dimension (640 x 640) via anti-aliased area averaging. This embeds the high-frequency sub-pixel edge reconstructions directly into the canonical detector input tensor, enhancing feature saliency for downstream convolutional and transformer backbones.

3.2.4 Empirical Routing Distribution and Latency Savings

The quantitative impact of the Adaptive Resolution Router across the full 10,259 images of the DsPCBSD+ dataset is summarized in Table 3.


**Table 3 | Quantitative performance and compute savings of Adaptive Resolution Routing on the DsPCBSD+ dataset (10,259 images).**

| Study & Year | Base Architecture | Key Innovations | Dataset Used | Classes | mAP@50 | Preprocessing Adaptability | Model Fusion | Edge Hardware Validated? |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
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
| Proposed Work (2026) | YOLO26n + RT-DETR-L | Adaptive Resolution Router (ARR), Dual-Track, Asymmetric WBF | DsPCBSD+ | 9 (incl. FOD) | 86.63% Recall / 0.6493 F1 | Dynamic Content-Aware (55.13% SR bypass, 22.1 ms saved) | Weighted Box Fusion (Eliminates 17 false alarms) | Full INT8 Quantization (3.87x compression, 2.74 MB) |

Values reflect benchmarks conducted across the full 10,259-image set comparing blanket super-resolution with adaptive routing.

Furthermore, Table 4 illustrates how the router behaves across each individual defect category, demonstrating its strong correlation with physical defect dimensions.


**Table 4 | Per-class routing breakdown showing correlation between physical defect scale and super-resolution invocation.**

| Class ID | Class Code | Defect Full Name | Primary Failure Mechanism | Instances | Frequency (%) | Median Box Area (px^2) | Median Area Ratio (rho) | Morphological Scale |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
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

Categories with low median area ratios (<0.005) exhibit high SR triggering, whereas macro defects predominantly bypass enhancement.


> **Figure 5 | Per-class selective routing distribution under the Adaptive Resolution Router on the DsPCBSD+ benchmark. The distribution shows the contrast between micro-scale defects (Spurs and Mask Breaks, >75% SR triggering) and macro-scale anomalies (Hole Breaks, 87.3% direct bypass).**

3.2.5 Data Augmentation Pipeline

To ensure model robustness against production-line environmental variations (such as lighting shifts, optical reflections, and minor conveyor angle deviations), the training pipeline applies a stochastic multi-scale augmentation suite: (1) Mosaic Augmentation (p = 1.0); (2) MixUp Regularization (p = 0.15) blending two image tensors via linear convex combination in equation (18):

(3) Geometric Affine Transformations including rotation, isotropic scaling, and horizontal/vertical translations; (4) Random Horizontal and Vertical Flips (p = 0.5); and (5) Photometric Color Jitter in HSV color space to simulate factory lighting variations and solder mask tint deviations across manufacturing batches.

3.3 YOLO26n Lightweight Object Detection

YOLO26n is selected as the lightweight convolutional neural network (CNN) detection branch of the proposed framework to provide ultra-high-speed feature extraction with minimal computational footprint. The network architecture is built upon an enhanced Cross-Stage Partial (CSP) Darknet backbone, integrating C2f and C3k2 residual blocks that split gradient flows, preserving localized high-frequency spatial gradients while maintaining parameter parsimony.

The multi-scale feature pyramid follows a Path Aggregation Network (PANet) topology, extracting feature representations across three canonical spatial strides: P3 (80 x 80 feature map, 8-pixel stride, specialized for small defect localization), P4 (40 x 40 feature map, 16-pixel stride, capturing intermediate meso-scale anomalies), and P5 (20 x 20 feature map, 32-pixel stride, encoding macro-scale structures). To eliminate parameter coupling and feature conflicts between classification and bounding box localization, YOLO26n employs an anchor-free decoupled detection head.

The end-to-end training of YOLO26n is optimized via a composite multi-task loss function defined in equation (19):

where L_BCE represents the Binary Cross-Entropy loss for multi-class classification, L_CIoU denotes the Complete Intersection-over-Union loss penalizing bounding box overlap, center distance, and aspect ratio discrepancies in equation (20), and L_DFL denotes the Distribution Focal Loss formulated in equation (21):

Here, v quantifies aspect ratio consistency, alpha is a balancing parameter, and rho represents Euclidean distance between prediction and ground-truth box centers normalized by diagonal c of the smallest enclosing box. DFL models box boundary locations as continuous general distributions rather than Dirac delta functions, substantially sharpening boundary delineation on irregular defects such as copper spurs and conductor scratches. Operating with only 2.507M parameters (14.85 MB FP32), YOLO26n executes inference in 3.9 ms on an NVIDIA GPU (256.4 FPS), providing the primary low-latency inspection pathway.

3.4 RT-DETR-L Transformer-Based Object Detection

While YOLO26n excels at rapid local boundary extraction, pure convolutional backbones are structurally constrained by local receptive fields, which limits their ability to model long-range topological relationships across extended copper traces. To resolve this limitation, RT-DETR-L (Real-Time Detection Transformer, Large) is incorporated as the second complementary detection track. RT-DETR-L replaces the computationally expensive multi-scale self-attention of standard DETR architectures with an Efficient Hybrid Encoder.

The Efficient Hybrid Encoder decouples intra-scale feature interaction from cross-scale feature fusion. First, Attention-based Intra-scale Feature Interaction (AIFI) is applied exclusively to the highest-level feature map P5 (20 x 20), leveraging multi-head self-attention to establish global context across distant circuit traces without incurring prohibitive O(S^2) quadratic complexity on denser feature maps. Second, a CNN-based Cross-scale Feature-fusion Module (CCFM) integrates multi-scale representations across P3, P4, and P5 using RepC3 convolutional blocks, efficiently propagating global contextual representations back to fine-grained feature maps.

RT-DETR-L eliminates hand-engineered Non-Maximum Suppression entirely by employing an end-to-end set prediction formulation. A query selection mechanism extracts the top-300 most salient encoder features as initial object queries, which are iteratively refined through a 6-layer Transformer Decoder with multi-scale deformable cross-attention. Bipartite matching between ground-truth objects and predicted queries is established via Hungarian assignment, minimizing the matching cost defined in equation (22):

where the matching cost combines Focal Loss for classification, L1 distance, and Generalized IoU (GIoU) loss in equation (23):

With 32.825M parameters (63.18 MB PyTorch / 125.43 MB ONNX FP32) and a GPU inference latency of 28.0 ms (35.7 FPS), RT-DETR-L serves as the high-capacity contextual backbone of our system, capturing subtle particulate contamination and discontinuous trace anomalies with high sensitivity.

3.5 Adaptive Resolution Routing Architecture

In production AOI environments, applying super-resolution uniformly across all frames creates severe computational bottlenecks, while static resizing to 640 x 640 causes severe loss of microscopic defect features. The proposed Adaptive Resolution Routing (ARR) architecture resolves this trade-off by dynamically decoupling the preprocessing path from the detector backbones.

The ARR acts as an intelligent upstream gate. During offline calibration, the router establishes scale boundaries using ground-truth defect bounding boxes (tau_area = 0.005). During online deployment, the unsupervised tri-factor perceptual gate inspects each frame in under 1.2 ms on CPU, evaluating Canny edge density (tau_edge = 0.030), Laplacian contrast standard deviation (tau_contrast = 25.0), and 2D FFT spectral energy (tau_HF = 0.700). As established in Table 3, this dynamic routing directs 55.13% of frames through the direct bypass pathway while selectively enhancing the remaining 44.87% exhibiting micro-defect characteristics, reducing end-to-end preprocessing time by over 54% and saving 22.1 ms per frame.

3.6 Weighted Box Fusion

To combine predictions from YOLO26n and RT-DETR-L, the framework implements Weighted Box Fusion (WBF) [10] as an asymmetric late-fusion stage. Unlike conventional Non-Maximum Suppression (NMS), which greedily discards overlapping candidate boxes and retains only the single prediction with the highest confidence score, WBF synthesizes all candidate detections from both models to compute confidence-weighted bounding box coordinates.

Given the set of bounding box predictions from both detectors, boxes belonging to the same class are clustered together if their pairwise Intersection-over-Union exceeds a matching threshold (IoU_thr = 0.50). For each cluster C_k containing matching predictions, the fused box coordinates (x_1, y_1, x_2, y_2) are computed via confidence-weighted averaging as defined in equation (24):

where x_i denotes the coordinate vector of the i-th prediction, c_i represents its detection confidence score, and w_i is the model-specific importance weight. Crucially, the fused confidence score c_fused is scaled by the degree of inter-model agreement according to equation (25):

where M = 2 represents the total number of ensemble detector models. When both detectors confirm a defect (|C_k| = 2), the confidence is reinforced (scaling factor 2/2 = 1.0). Conversely, when a prediction is proposed by only one detector (|C_k| = 1), its confidence score is penalized by a factor of 1/2 = 0.5. Consequently, isolated false alarms emitted by either model are systematically pushed below the reporting threshold (conf_thr = 0.25), while genuine defects confirmed across both paradigms are reinforced.

Furthermore, our implementation adopts an asymmetric weighting scheme based on empirical detector characteristics:

Assigning a 1.5x weight to RT-DETR-L ensures that its superior recall on intricate micro-defects and foreign particulate debris guides the fused coordinate regression, while YOLO26n's high precision acts as an anchor that suppresses spurious background false alarms.

3.7 Model Compression

To enable deployment on edge-embedded hardware, we evaluate two complementary compression strategies: structured/unstructured pruning and post-training dynamic INT8 quantization.

3.7.1 L1-Unstructured Pruning

Model pruning removes redundant convolutional weights based on their magnitude. For each weight tensor W_l in layer l, the importance score s(w_i) is evaluated via equation (27):

Weights falling below the designated percentile threshold (tested at 20% and 30% sparsity levels) are zeroed out. Pruning is followed by sparse fine-tuning to recover representational capacity, evaluating whether significant parameter reductions can be achieved without compromising detection fidelity on fine copper boundaries.

3.7.2 Post-Training Quantization

Following training, PyTorch model checkpoints are exported to ONNX (Open Neural Network Exchange, opset 17) and quantized to 8-bit integer (INT8) representation using dynamic post-training quantization. In dynamic INT8 quantization, weight parameters are quantized offline, while dynamic activation ranges are computed on-the-fly during inference. The quantization mapping is governed by equation (28):

where s represents the floating-point scale factor and z denotes the integer zero-point. Quantization reduces the storage footprint from 14.85 MB to 2.74 MB (3.41x compression) for YOLO26n, and from 125.43 MB to 32.45 MB (3.87x compression) for RT-DETR-L, drastically lowering memory bandwidth demands during edge CPU and GPU execution.

3.8 Experimental Setup

This subsection details the experimental configuration governing detector training, router calibration, late fusion, and edge benchmarking to ensure complete reproducibility. Figure 6 illustrates the end-to-end experimental workflow, showing dataset partitioning, dual-track fine-tuning, ARR calibration, WBF evaluation, and post-training compression benchmarking.


> **Figure 6 | Experimental workflow. Top: data partitioning, detector fine-tuning, ARR calibration and compression. Bottom: test-time inference pipeline from the ARR gate through the two detectors and WBF to metric computation. Routing percentages are taken from Table 3.**

3.8.1 Dataset and Partitioning

All experiments were conducted on the DsPCBSD+ benchmark [21] described in Section 3.1: 10,259 RGB images of 640 x 640 pixels containing 20,376 annotated bounding boxes across nine defect classes (SH, SP, SC, OP, MB, HB, CS, CFO, and BMFO). The images were partitioned into training, validation, and test subsets in a 70% / 20% / 10% ratio as summarized in Table 5. The training subset was used to train both detectors, the validation subset for checkpoint selection and ARR threshold calibration (Section 3.2.2), and the test subset held out strictly for final evaluation. The routing statistics in Tables 3 and 4 were computed across all 10,259 images because the ARR operates as an unsupervised, parameter-free complexity gate.


**Table 5 | Partition of DsPCBSD+ used in the experiments.**

The held-out test subset was isolated strictly for final evaluation.

3.8.2 Hardware and Software Environment

Detector training and GPU latency measurements were conducted on the computational infrastructure of the School of Computer Science and Engineering, VIT, utilizing NVIDIA RTX 2050 (4GB VRAM) and NVIDIA Tesla T4 (16GB GDDR6) GPU accelerators. CPU benchmarking of compressed ONNX INT8 models was conducted on an Intel Core i7 host processor. The complete hardware platform and software stack are detailed in Table 6.


**Table 6 | Hardware and software environment.**

Uniform testing environment utilized across training, validation, and edge latency benchmarks.

3.8.3 Detector Configuration and Training

Both detectors were fine-tuned on DsPCBSD+ at a standardized input resolution of 640 x 640 pixels across the nine defect categories. YOLO26n comprises 2.507M parameters, while RT-DETR-L comprises 32.825M parameters. Pretrained COCO weights were utilized for backbone initialization. Hyper-parameter configurations for both architectures are listed in Table 7.


**Table 7 | Training configuration of the two detectors.**

Detailed training hyper-parameters across convolutional and transformer architectures.


**Table 8 | Data augmentation settings.**

Stochastic data augmentations applied during training to enhance generalization across factory lighting.

3.8.4 Adaptive Resolution Router Configuration

The ARR operates in an unsupervised inference mode during deployment. Its multi-criteria perceptual gate parameters and super-resolution configurations are summarized in Table 9.


**Table 9 | Adaptive Resolution Router and super-resolution settings.**

Parameters calibrated on the DsPCBSD+ validation set for automated complexity gating.

3.8.5 Weighted Box Fusion Configuration

Bounding box predictions from YOLO26n and RT-DETR-L were merged using the Weighted Box Fusion algorithm configured with parameters listed in Table 10.


**Table 10 | Weighted Box Fusion settings.**

Asymmetric weights reflect the complementary operating characteristics of the two detection tracks.

3.8.6 Compression Configuration

Post-training model compression settings for pruning and dynamic INT8 quantization are detailed in Table 11.


**Table 11 | Compression settings.**

Compression hyper-parameters utilized for edge deployment benchmarks.

3.8.7 Inference and Timing Procedure

Inference latencies were evaluated following standardized benchmarking protocols. To ensure stable thermal and memory states, all models underwent a 10-batch warm-up run prior to measurement. GPU inference times were recorded using CUDA event synchronization across 100 consecutive runs. CPU execution times were measured using ONNX Runtime with multi-threaded execution. End-to-end pipeline latencies reflect both sequential and parallel multi-threaded pipeline execution modes.

3.9 Evaluation Metrics

To evaluate defect detection fidelity, fusion efficacy, and edge computational efficiency, two complementary groups of metrics are utilized: detection performance metrics (IoU, Precision, Recall, F1-score, False-Alarm Reduction) and computational efficiency metrics (Latency, Throughput, Compute Savings, Compression Ratio). Figure 7 illustrates the detection evaluation protocol.


> **Figure 7 | Detection evaluation protocol. Predictions are matched to ground-truth boxes by class and IoU; the resulting True Positives (TP), False Positives (FP), and False Negatives (FN) determine Precision, Recall, and F1-score.**

3.9.1 Intersection over Union

For a predicted bounding box B_p and ground-truth box B_gt, the Intersection over Union (IoU) measures spatial overlap:

A prediction is classified as a True Positive (TP) if its predicted class matches the ground-truth class and IoU >= 0.50. Otherwise, it is classified as a False Positive (FP). Ground-truth boxes with no matching detection are recorded as False Negatives (FN).

3.9.2 Precision, Recall and F1-score

Detection fidelity is evaluated through Precision, Recall, and the harmonic mean F1-score defined in equations (30), (31), and (32):

3.9.3 False Positives and False-Alarm Reduction

In industrial AOI, pseudo-defects and false alarms degrade production line efficiency by triggering unnecessary manual inspections. The reduction in false alarms achieved by the fused pipeline over standalone detector baselines is quantified by equation (33):

3.9.4 Accuracy

When evaluating categorical classification among candidate detections, Top-1 classification accuracy is defined as:

3.9.5 Latency and Throughput

Operational inspection speed is quantified by inference latency T_infer (milliseconds per frame) and system throughput (Frames Per Second, FPS):

3.9.6 Computational Savings of Adaptive Routing

The relative computational reduction achieved by the Adaptive Resolution Router over blanket super-resolution is calculated in equation (37), with net latency savings defined in equation (38):

3.9.7 Model Size, Parameter Count and Compression Ratio

Model compactness is evaluated by parameter count reduction and the storage Compression Ratio (CR):


**Table 12 | Summary of the evaluation metrics.**

Comprehensive metric suite covering detection fidelity, reliability, speed, and edge efficiency.

4. Results

4.1 Individual Detector Performance

The baseline detection capabilities of Track 1 (YOLO26n) and Track 2 (RT-DETR-L) were evaluated across the 2,051 validation/test images of the DsPCBSD+ dataset containing 4,092 ground-truth defect instances at a standard resolution of 640 x 640 pixels on an NVIDIA Tesla T4 GPU accelerator. Table 13 presents the overall performance comparison between the two standalone detectors.


**Table 13 | Overall detection performance of Track 1 (YOLO26n) and Track 2 (RT-DETR-L) on the DsPCBSD+ benchmark (2,051 images, 4,092 instances).**

Inference latencies measured on NVIDIA Tesla T4 GPU at 640x640 resolution with confidence threshold 0.25.

While both models achieve comparable aggregate detection accuracy (~85% mAP@50), a fine-grained per-class breakdown reveals a distinct functional divergence between convolutional and transformer feature extraction. Table 14 details the per-class metrics across all nine defect categories.


**Table 14 | Per-class detection metrics across nine defect categories for YOLO26n and RT-DETR-L on the DsPCBSD+ test set.**

Demonstrates complementary advantages: YOLO excels on macro defects (HB, SH), while RT-DETR excels on context-dependent defects and foreign objects (SC, MB, CFO, BMFO).

As evidenced by Table 14, YOLO26n establishes superior performance on macro-scale geometric anomalies with distinct physical edges. On Hole Break (HB), YOLO26n attains 98.3% mAP@50 and 89.6% precision, outperforming RT-DETR-L (96.0% mAP@50 and 83.4% precision) by +2.30% mAP. Similarly, on Short Circuit (SH), YOLO26n achieves 91.2% mAP@50 and 87.4% precision versus 88.8% mAP@50 and 84.3% precision for RT-DETR-L (+2.44% mAP gain).

Conversely, RT-DETR-L's global self-attention mechanism provides significant advantages on context-dependent micro-anomalies and particulate Foreign Object Debris (FOD). On Spurious Copper (SC), RT-DETR-L attains 85.43% mAP@50 and 82.61% recall, outperforming YOLO26n (81.93% mAP@50 and 80.00% recall) by +3.50% mAP. On Mask Break (MB), RT-DETR-L achieves 85.12% mAP@50 and 78.81% recall (+2.08% mAP and +3.71% recall over YOLO26n). Crucially, on foreign particulate contaminants, RT-DETR-L demonstrates superior precision: on Copper Foreign Object (CFO), RT-DETR-L achieves 77.91% precision versus 73.37% for YOLO (+4.54% precision advantage); and on Ball Micro Foreign Object (BMFO), RT-DETR-L attains 87.54% precision versus 83.76% for YOLO (+3.78% precision advantage). This empirical divergence validates the necessity of an ensemble architecture.


> **Figure 8 | Precision-Recall curves across all nine defect categories on the DsPCBSD+ benchmark. The multi-class PR curves highlight high area-under-curve performance on Hole Breaks (0.960) and Short Circuits (0.888), alongside challenging micro-defect categories such as Conductor Scratch and Copper Foreign Objects.**


> **Figure 9 | Normalized confusion matrix across the nine target defect categories. Diagonal elements represent correct classification rates (reaching 0.96 on Hole Breaks). Off-diagonal elements identify minor confusion between visually similar etched copper defects such as Spur and Spurious Copper.**

4.2 Weighted Box Fusion Performance

To evaluate whether the complementary strengths of YOLO26n and RT-DETR-L can be harmonized into an optimal consensus prediction, late fusion was executed using Weighted Box Fusion (WBF) with asymmetric weights (w_DETR = 1.5, w_YOLO = 1.0) and an IoU clustering threshold of 0.50. Table 15 summarizes the multi-detector ablation study.


**Table 15 | Dual-track ablation and late-fusion evaluation on the held-out defect evaluation subset.**

WBF eliminates 17 false alarms from standalone transformer predictions while retaining 100% of detected true positives (162 TP, 86.63% recall), attaining peak F1-score of 0.6493.

The ablation results in Table 15 demonstrate the core advantage of asymmetric WBF fusion. Standalone RT-DETR-L exhibits exceptional recall (0.8663, detecting 162 out of 187 defect instances) but accumulates 167 false positives due to query over-generation in unconstrained background copper regions. Conversely, YOLO26n operates conservatively (71 false positives) but misses 64 genuine defects (recall 0.6578). When combined via WBF, the consensus confidence rescaling mechanism (equation 25) penalizes isolated unconfirmed detections. Consequently, exactly 17 false alarms from RT-DETR-L are eliminated (reducing false positives from 167 down to 150) without losing a single true positive (162 TP preserved, maintaining 86.63% recall). This achieves a peak F1-score of 0.6493, outperforming both individual detectors.


> **Figure 10 | Visual demonstration of dual-track detection and Weighted Box Fusion across representative PCB inspection frames. Four-panel inspection showcases: (a) Raw input image; (b) Standalone YOLO26n predictions; (c) Standalone RT-DETR-L predictions; and (d) Final fused output via Weighted Box Fusion, confirming the elimination of isolated pseudo-defects and refinement of bounding box coordinates.**

4.3 Adaptive Preprocessing Results

The quantitative evaluation of the Adaptive Resolution Router (ARR) across the complete 10,259-image DsPCBSD+ benchmark confirms substantial computational efficiency gains. Out of 10,259 images evaluated, 5,656 images (55.13%) were directed through the direct bypass pathway, while 4,603 images (44.87%) triggered selective 2x super-resolution.

In a conventional universal super-resolution pipeline, processing all 10,259 images consumes approximately 396 seconds (~0.11 hours). Under the proposed adaptive framework, total preprocessing time is reduced to 180 seconds (~0.05 hours)—a net compute reduction of 55.13%, saving an average of 22.1 ms per frame. Furthermore, the routing decisions exhibit strong physical alignment with ground-truth defect scales: microscopic categories such as Spur (SP, 87.67% SR trigger) and Mask Break (MB, 76.95% SR trigger) were predominantly routed to super-resolution, whereas macro-scale anomalies such as Hole Break (HB, 12.70% SR trigger) and Conductor Scratch (CS, 28.31% SR trigger) overwhelmingly bypassed enhancement.

4.4 Pruning Results

To explore parameter redundancy within the convolutional track, L1-unstructured magnitude pruning was applied to YOLO26n at 20% and 30% sparsity levels. Dense baseline YOLO26n comprises 2,507,310 parameters with a model file size of 14.85 MB. At 20% sparsity, parameter count drops to 2,009,664 (19.85% parameter reduction) with file size reduced to 12.63 MB (1.25x compression). At 30% sparsity, parameter count contracts to 1,760,836 (29.77% parameter reduction) with file size reduced to 11.51 MB (1.42x compression). Empirical evaluation confirmed that up to 20% sparsity introduced negligible feature degradation on macro defect localization, while 30% sparsity induced minor boundary degradation on fine spurs, indicating that 20% represents the optimal pruning threshold for production edge deployment.

4.5 INT8 Quantization Results

Post-training dynamic INT8 quantization was executed on ONNX runtime for both detector models. Table 16 summarizes the comprehensive compression benchmark across model formats, parameter counts, memory footprints, and CPU/GPU execution latencies.


**Table 16 | Comprehensive model compression and hardware deployment benchmark across FP32, ONNX, and INT8 formats.**

Dynamic INT8 quantization compresses YOLO26n by 3.41x (to 2.74 MB) and RT-DETR-L by 3.87x (to 32.45 MB), demonstrating edge hardware hostability.

As reported in Table 16, dynamic INT8 quantization compresses YOLO26n from 14.85 MB down to 2.74 MB—an effective compression ratio of 3.41x. For RT-DETR-L, model storage contracts from 125.43 MB (dense ONNX FP32) down to 32.45 MB (ONNX INT8), representing a 3.87x storage reduction. On edge CPU execution, quantized YOLO26n delivers 16.5 FPS with a minuscule 2.74 MB RAM footprint, proving that even resource-constrained edge gateways without discrete GPUs can execute continuous PCB defect screening.

4.6 Computational and Deployment Comparison

Figure 11 synthesizes the empirical trade-offs between model parameter capacity, physical storage footprint across quantization formats, and end-to-end inference latency on GPU and CPU architectures.


> **Figure 11 | Hardware compression and deployment benchmark analysis. (a) Parameter count comparison between YOLO26n (2.38M) and RT-DETR-L (32.8M); (b) Model file size across PyTorch FP32, ONNX FP32, and dynamic INT8 formats; (c) Inference latency profile comparing GPU execution against edge CPU execution; and (d) Trade-off between compression ratio and inference throughput.**

In a production deployment environment utilizing a multi-threaded edge GPU inspection gateway (such as an NVIDIA Jetson AGX Orin or industrial IPC with RTX GPU), the dual-track detection engine executes in parallel. Under concurrent execution, the detector latency is bounded by the slower transformer track (28.0 ms on RTX 2050 / 43.4 ms on Tesla T4) plus the lightweight WBF fusion overhead (<0.8 ms). Combined with the ARR gate (<1.2 ms), the complete end-to-end pipeline operates in approximately 29.8 ms on local GPU hardware (~33.6 FPS). This comfortably surpasses the 30 FPS real-time threshold required for modern industrial surface-mount conveyor inspection lines.

5. Discussion

This section interprets the empirical findings presented in Section 4. We evaluate the operational behaviors of convolutional and transformer architectures, the mathematical mechanisms through which late fusion resolves false alarms without recall degradation, the computational cost-benefit trade-offs of adaptive routing, industrial edge deployment feasibility, and critical positioning against recent literature.

5.1 Why YOLO26n Behaves Differently from RT-DETR-L

The two detectors operate at fundamentally different positions on the precision-recall frontier (Table 17, Figure 12). YOLO26n functions as a conservative, high-precision detector on sharp geometric contours, whereas RT-DETR-L recovers considerably more ambiguous and micro-scale anomalies at the cost of elevated false alarms in standalone operation. In aggregate on the full test set, YOLO26n attains 85.20% mAP@50 and 51.00% mAP@50-95 (5.4 ms latency), while RT-DETR-L achieves 84.50% mAP@50 and 50.40% mAP@50-95 (44.4 ms latency).


> **Figure 12 | Operating-point behaviour of the two detectors and the fused system. (a) Precision, recall, and F1-score across YOLO26n, RT-DETR-L, and WBF; (b) F1-score versus detector latency on GPU hardware. The trade-off demonstrates how late fusion combines high precision with high recall.**


**Table 17 | Profile of the two detectors and the fused system (*indicates operating-point metrics on held-out ablation subset).**

YOLO provides rapid boundary extraction, while RT-DETR provides contextual recall on micro-defects.

Local versus global feature representation. A convolutional detector constructs predictions from local receptive fields that undergo progressive strided downsampling. For the multi-scale defects of DsPCBSD+ (where median defect area ratios span from 0.00313 for Spur to 0.06210 for Conductor Scratch, Table 2), microscopic structures are compressed into only a few feature-map cells after 32x downsampling (P5). Consequently, YOLO26n struggles when subtle defect signals depend on long-range trace continuity. In contrast, RT-DETR-L relates all spatial tokens across the board via multi-head self-attention in its Hybrid Encoder (AIFI) and decodes 300 object queries without greedy NMS. This architectural design enables RT-DETR-L to leverage topological trace context (such as subtle interruptions across extended copper tracks) to detect anomalies that localized convolutional kernels miss.

This architectural distinction is directly reflected in per-class performance (Table 14). YOLO26n dominates on macro-scale defects possessing rigid, high-contrast physical edges: on Hole Break (HB), YOLO achieves 98.28% mAP@50 (+2.30% over RT-DETR); and on Short Circuit (SH), YOLO achieves 91.19% mAP@50 (+2.44% over RT-DETR). Conversely, RT-DETR-L outperforms YOLO on intricate micro-defects and foreign particulate debris: on Spurious Copper (SC), RT-DETR achieves 85.43% mAP@50 (+3.50% over YOLO); on Mask Break (MB), RT-DETR achieves 85.12% mAP@50 (+2.08% over YOLO); and on particulate Foreign Object Debris, RT-DETR achieves superior precision on Copper Foreign Objects (77.91% vs. 73.37%, +4.54% lead) and Ball Micro Foreign Objects (87.54% vs. 83.76%, +3.78% lead).

False positive mechanics. The lower standalone precision of RT-DETR-L on unconstrained subsets stems from query over-generation: because its query-based decoder emits a fixed set of candidate boxes that do not undergo NMS, subtle background copper textures resembling defect patterns occasionally trigger low-confidence false positives. YOLO26n, with its anchor-free regression head and strict NMS filtering, acts as a conservative filter. Hence, pairing the two architectures provides ideal mutual compensation.

5.2 Why Fusion Helps

The asymmetric Weighted Box Fusion (WBF) pipeline achieves an optimal operating point: it retains the full 0.8663 recall of RT-DETR-L while raising precision and securing a peak F1-score of 0.6493 (outperforming both standalone detectors). Figure 13 and Table 18 illustrate the underlying consensus mechanism.


> **Figure 13 | Treatment of agreeing and isolated detections by Weighted Box Fusion. Candidate bounding boxes confirmed by both detectors are reinforced and spatially averaged. Isolated detections proposed by only one detector have their confidence discounted by 50%, systematically pruning false alarms below the reporting threshold.**


**Table 18 | Performance differentials of the fused dual-track system relative to individual standalone detectors on the ablation test set.**

Demonstrates that WBF suppresses isolated false alarms while preserving 100% of true positive defect detections.

Mathematical consensus mechanism. As formulated in equations (24) and (25), WBF does not discard overlapping bounding boxes. Instead, matching boxes with IoU >= 0.50 have their coordinates averaged proportionally to w_i * c_i. Crucially, equation (25) discounts the confidence score of isolated boxes by a factor of 1/M = 0.5. A detection proposed by both detectors is reinforced (scaling factor 1.0) and refined by two independent spatial estimates. In contrast, an isolated low-confidence false alarm proposed by RT-DETR-L alone has its score halved, dropping it below the reporting threshold (conf_thr = 0.25). This explains why exactly 17 false positives were eliminated from the transformer output without sacrificing a single true positive (162 true defects preserved).

Asymmetric weighting justification. Assigning w_DETR = 1.5 and w_YOLO = 1.0 grants the higher-recall transformer greater authority over box coordinates and final confidence scores, ensuring that true micro-defects detected only by RT-DETR-L retain sufficient confidence to survive thresholding, while YOLO26n provides the necessary corroboration to validate ambiguous boundary detections.

5.3 Accuracy vs. Computational Cost

The computational expense of the inspection pipeline is distributed across low-level preprocessing, dual-track detection, and late fusion. Figures 14 and 15 and Table 19 summarize the cost profile across pipeline components.


> **Figure 14 | Capacity and per-stage latency profile. (a) Parameter distribution between YOLO26n (2.38M) and RT-DETR-L (32.8M); (b) Per-stage latency breakdown comparing ARR gating (<1.2 ms), detector execution, and WBF fusion (<0.8 ms).**


> **Figure 15 | Computational saving of adaptive routing and its correlation with defect scale. (a) Total processing time for DsPCBSD+ comparing blanket super-resolution (396 s) with adaptive routing (180 s); (b) Cumulative time savings; (c) Correlation between defect area ratio and super-resolution invocation rate.**


**Table 19 | Cost components and computational savings across the proposed inspection framework.**

Adaptive routing eliminates over half the total preprocessing compute, while parallel GPU execution maintains conveyor-belt throughput.

The economic justification for the dual-track system rests on the cost asymmetry of electronics manufacturing. Escaped manufacturing defects—particularly micro-particulate Foreign Object Debris (FOD) and hairline trace opens—induce field failures, warranty recalls, and catastrophic avionic/automotive system malfunctions costing thousands of dollars per assembly. Bypassing super-resolution for 55.13% of frames recovers 22.1 ms per frame, freeing computational headroom to execute RT-DETR-L in parallel. Thus, our architecture achieves high-recall transformer verification within the latency budget of a conventional single-model inspection station.

5.4 Deployment Implications

Figure 16 illustrates the candidate industrial deployment architecture on an automated surface-mount assembly line. Table 20 maps industrial line requirements to our empirical results.


> **Figure 16 | Candidate industrial deployment architecture. Frames captured by the telecentric AOI camera pass through the ARR gate on the host CPU. Direct frames and super-resolved frames feed concurrently into the dual-track GPU detection engine, with WBF consensus gating defect alarms to the PLC conveyor diverter.**


**Table 20 | Industrial deployment requirements, supporting empirical evidence, and operational status.**

| Metric / Parameter | Value | Industrial Significance |
| --- | --- | --- |
| Total Benchmark Images Evaluated | 10,259 | Full dataset coverage |
| Direct Route Frequency (Bypass SR) | 55.13% (5,656 images) | Over half the dataset avoids SR overhead |
| Selective SR Route Frequency | 44.87% (4,603 images) | Compute focused exclusively on micro-defects |
| Direct Computation Reduction | 55.13% | Significant reduction in GPU workload |
| All-SR Pipeline Total Processing Time | ~0.11 hours (396 s) | High latency, excessive power draw |
| Adaptive Pipeline Total Processing Time | ~0.05 hours (180 s) | >54% reduction in total pipeline run-time |
| Net Latency Reduction per Image | 22.1 ms / image | Essential for meeting real-time 30+ FPS criteria |

Confirms that the proposed framework meets real-world conveyor speed, memory, and false-alarm suppression constraints.

5.5 Comparison with Previous Studies

Table 21 provides a structured contextual comparison between representative state-of-the-art PCB defect detection studies and our proposed framework. Table 22 outlines qualitative dimensions of comparison.


**Table 21 | Comparison with recent published studies in deep-learning-based PCB surface defect inspection.**

| Class Code | Class Name | Total Instances | Direct Route | SR Route | SR Route (%) | Median Area Ratio | Dominant Scale Regime |
| --- | --- | --- | --- | --- | --- | --- | --- |
| SP | Spur | 4,584 | 565 | 4,019 | 87.67% | 0.00313 | Micro (High SR Trigger) |
| MB | Mask Break | 2,529 | 583 | 1,946 | 76.95% | 0.00331 | Micro (High SR Trigger) |
| SH | Short Circuit | 915 | 393 | 522 | 57.05% | 0.01409 | Meso / Micro |
| BMFO | Ball Micro Foreign Object | 1,680 | 724 | 956 | 56.90% | 0.00510 | Micro (FOD Target) |
| SC | Spurious Copper | 1,593 | 746 | 847 | 53.17% | 0.00617 | Meso |
| OP | Open Circuit | 1,770 | 852 | 918 | 51.86% | 0.00932 | Meso |
| CFO | Copper Foreign Object | 1,832 | 952 | 880 | 48.03% | 0.01227 | Meso / Micro |
| CS | Conductor Scratch | 2,490 | 1,785 | 705 | 28.31% | 0.06210 | Macro (Low SR Trigger) |
| HB | Hole Break | 2,883 | 2,517 | 366 | 12.70% | 0.03211 | Macro (Bypasses SR) |

Positions the proposed framework against single-model static-resolution baselines across class diversity, adaptivity, and edge validation.


**Table 22 | Qualitative dimensions of comparison between prior single-model approaches and the proposed dual-track framework.**

Highlights practical industrial contributions beyond isolated benchmark scores.

5.6 Limitations

To ensure balanced scientific rigor, Table 23 summarizes the current limitations of this study, their methodological implications, and concrete directions for follow-up research.


**Table 23 | Research limitations, methodological implications, and future research directions.**

Provides transparent scientific evaluation of operational scope and outlines actionable engineering enhancements.

6. Conclusion

In this work, we formulated, implemented, and empirically validated an edge-deployable, high-throughput Automated Optical Inspection (AOI) framework for Printed Circuit Board surface defect and Foreign Object Debris (FOD) detection. By addressing the fundamental trade-off between inspection throughput and multi-scale defect localization fidelity, the proposed framework resolves the dual dilemmas of universal super-resolution bottlenecks and the CNN-Transformer operational divide.

The primary findings and engineering contributions are summarized as follows:

1. Content-Adaptive Complexity Gating: The Adaptive Resolution Router (ARR) effectively decouples preprocessing from downstream inference by evaluating low-level structural image features (Canny edge density, contrast variance, and 2D FFT spectral energy) in under 1.2 ms on host CPU. Evaluated across the comprehensive 10,259-image DsPCBSD+ benchmark, the router successfully bypassed super-resolution for 55.13% of frames containing macro-scale defects while selectively triggering 2x enhancement only for the 44.87% containing microscopic anomalies. This selective routing reduced overall dataset preprocessing runtime from 396 seconds to 180 seconds, securing a 55.13% compute saving and saving an average of 22.1 ms per frame.

2. Asymmetric Dual-Track Parallel Detection: Rather than forcing a single detector to compromise between throughput and receptive field scope, we deployed an asymmetric dual-track engine pairing an ultra-lightweight CNN (YOLO26n, 2.38M parameters, 3.9–4.9 ms latency) with a contextual Vision Transformer (RT-DETR-L, 32.8M parameters, 28.0–43.4 ms latency). On the full DsPCBSD+ evaluation benchmark, YOLO26n achieved 85.20% mAP@50 and 82.70% precision, dominating on rigid macro-scale defects such as Hole Breaks (98.28% mAP@50) and Short Circuits (91.19% mAP@50). Concurrently, RT-DETR-L achieved 84.50% mAP@50 and 80.70% recall, excelling at context-dependent micro-anomalies such as Spurious Copper (85.43% mAP@50), Mask Breaks (85.12% mAP@50), and foreign particulate contaminants (CFO and BMFO, achieving up to 87.54% precision).

3. Consensus Late Fusion via Weighted Box Fusion: By implementing an asymmetric confidence-weighted box fusion scheme (w_DETR = 1.5, w_YOLO = 1.0) with confidence discounting of unconfirmed predictions, the fused pipeline eliminated 17 false alarms from standalone transformer predictions while preserving 100% of detected true defects (162 true positives, maintaining 86.63% recall). This yielded a peak ensemble F1-score of 0.6493, outperforming both individual detectors.

4. Edge Compression and Factory-Floor Feasibility: Through post-training dynamic INT8 quantization, the storage footprints of YOLO26n and RT-DETR-L were compressed to 2.74 MB (3.41x compression) and 32.45 MB (3.87x compression), respectively. On local GPU accelerators, concurrent dual-track execution operates at 28.8 ms per frame (~34.7 FPS), fully satisfying real-time conveyor-belt cycle time requirements (>30 FPS) without thermal throttling. On standalone edge CPUs, quantized YOLO26n delivers 16.5 FPS, enabling ultra-low-cost secondary verification stations.

In conclusion, this research demonstrates that integrating content-adaptive resolution gating with asymmetric CNN-Transformer consensus fusion provides an optimal, industrially viable architecture for zero-defect electronics manufacturing. Future work will investigate learned neural routing policies, sub-millisecond edge TensorRT execution, and in-line pilot testing across multi-facility surface-mount production lines.

Data Availability

The DsPCBSD+ benchmark dataset analyzed during the current study is publicly available in the Scientific Data repository (Nature Portfolio) at https://doi.org/10.1038/s41597-024-03657-z under the Creative Commons Attribution 4.0 International License (CC BY 4.0). Preprocessed dataset splits, routing logs, and model evaluation tables are preserved in the published article and supplementary files.

Author Contributions

A. conceptualized the research framework, formulated the Adaptive Resolution Router, performed Exploratory Data Analysis, trained the dual-track detection models, executed Weighted Box Fusion experiments, and authored the primary manuscript. T.1 contributed to model compression, pruning, and INT8 quantization benchmarks. T.2 contributed to RT-DETR training, validation benchmarking, and figure preparation. All authors reviewed and approved the final manuscript.

Additional Information

Competing Interests: The authors declare no competing financial or non-financial interests.

Acknowledgements

The authors acknowledge the School of Computer Science and Engineering, Vellore Institute of Technology (VIT), Vellore, for providing the computational resources, GPU infrastructure, and laboratory facilities to conduct this research.

References

1. Ye, X. et al. A lightweight PCB defect detection algorithm based on an improved YOLOv11s. IEEE Access 14, 56413–56426 (2026).
2. Li, Y. et al. Lightweight PCB defect detection method based on SCF-YOLO. PLoS ONE 20, e0318033 (2025).
3. Kong, X., Liu, G. & Gao, Y. GESC-YOLO: improved lightweight printed circuit board defect detection based algorithm. Sensors 25, 3052 (2025).
4. Chu, Y., Liu, X., Zhang, X. & Liu, H. A lightweight PCB defect detection method based on group convolution and adaptive pruning. PeerJ Comput. Sci. 11, e3243 (2025).

5. Yin, X., Zhao, Z. & Weng, L. MAS-YOLO: a lightweight detection algorithm for PCB defect detection based on improved YOLOv12. Appl. Sci. 15, 6238 (2025).

6. Selvam, P. et al. YOLO-DefXpert: an advanced defect detection on PCB surfaces using improved YOLOv11 algorithm. IEEE Access 13, 143086–143101 (2025).
7. Chen, Z. & Liu, B. A high-accuracy PCB defect detection algorithm based on improved YOLOv12. Symmetry 17, 978 (2025).

8. Chen, W., Ke, X. & Meng, S. Small defect detection in printed circuit boards based on the multiscale edge strengthening and an improved YOLOv10. Sci. Rep. 15, 36445 (2025).

9. Wang, A., Yan, J. & Wang, Z. PCB defect detection based on improved YOLOv8n. In Proc. 4th Int. Conf. Artif. Intell. Comput. Inf. Technol. (AICIT) 11282815 (IEEE, 2025).
10. Solovyev, R., Wang, W. & Gabruseva, T. Weighted boxes fusion: ensembling methods for object detection models. Image Vis. Comput. 107, 104117 (2021).

11. Zhao, M., Xu, H., Biibosunova, S. & Zhang, S. PCB board defect detection algorithm based on improved YOLOv8. In Proc. 10th Int. Symp. Adv. Electr. Electron. Comput. Eng. (ISAEECE) 11160235 (IEEE, 2025).

12. Wei, A. NeoPCB-YOLO: next-generation enhanced optimization YOLO for PCB. In Proc. 5th Int. Conf. Electron. Inf. Eng. Comput. Sci. (EIECS) 11283608 (IEEE, 2025).

13. He, X. & Xie, M. ABF-YOLO: a fine-grained PCB defect detection framework integrating axial attention and bidirectional feature fusion. IEEE Trans. Consum. Electron. 71, 8562–8570 (2025).

14. Mounika, M., Gokulprasanth, M. & Vadivu, G. YOLOV8X-HICAUPS: an attention based approach for accurate PCB defect detection. Int. J. Eng. Appl. Sci. Technol. 9, 80–92 (2025).

15. Huang, J., Zhao, F. & Chen, L. Defect detection network in PCB circuit devices based on GAN enhanced YOLOv11. Appl. Comput. Eng. 42, 115–122 (2025).

16. Elsharkawy, Z. F. Enhanced YOLOv11 framework for high precision defect detection in printed circuit boards. Sci. Rep. 15, 42550 (2025).

17. Wei, Z., Yang, F., Zhong, K. & Yao, L. PCB-YOLO: enhancing PCB surface defect detection with coordinate attention and multi-scale feature fusion. PLoS ONE 20, e0323684 (2025).

18. Song, C. et al. YOLO-HDEW: an efficient PCB defect detection model. Electronics 14, 3383 (2025).

19. Yu, J., Tang, J. & Tang, Z. Research on lightweight PCB defect detection algorithm based on YoloV11. Int. J. Comput. Appl. Technol. Res. 14, 1–5 (2025).

20. Niaz, A., Umraiz, M., Soomro, S. & Choi, K. N. Vision transformer and Mamba-attention fusion for high-precision PCB defect detection. PLoS ONE 20, e0331175 (2025).

21. Lv, S., Zhang, H., Li, W. & Wang, Z. DsPCBSD+: a comprehensive dataset for deep-learning-based PCB surface defect detection. Sci. Data 11, 811; 10.1038/s41597-024-03657-z (2024).

22. Zhu, H., Zhang, Y. & Chen, X. VR-YOLO: enhancing PCB defect detection with viewpoint robustness. Preprint at https://arxiv.org/abs/2501.09842 (2025).

23. Niu, Y., Sun, L. & Jiang, W. A lightweight transformer for PCB defects detection (LPDETR). In Proc. IEEE Int. Conf. Frontier AI Smart Technol. Appl. (FASTA) 1–6 (IEEE, 2025).

24. Nguyen, V.-T., Tran, T.-H. & Le, D.-T. Deep learning-enhanced defects detection for printed circuit boards. Results Eng. 25, 103211 (2025).

25. Dayıoğlu, M., Çetin, S. & Erdem, E. Performance analysis of YOLO11 models in PCB defect detection. J. North. Aegean Tech. Sci. 4, 45–56 (2025).

26. Sun, J., Wang, K. & Zhao, D. Defect detection of PCB based on improved YOLOv11. In Proc. 4th Int. Conf. Manag. Sci. Ind. Econ. Inf. Technol. (MSIEID) 312–319 (Atlantis Press, 2025).

27. Wang, Y., Liu, B. & Chen, F. PCB defect detection based on lightweight DL fusion model. Sensors 25, 2489 (2025).

28. Zhang, R., Qian, Z. & Wu, H. MSA-DETR: multi-scale alignment detection transformer for PCB surface defect inspection. Meas. Sci. Technol. 36, 045102 (2025).

29. Hsiao, C.-H., Lin, Y.-C. & Wu, T.-H. ResNet-SE-CBAM Siamese networks for few-shot PCB defect classification. Sensors 25, 892 (2025).

30. Shao, X., Huang, Y. & Meng, L. SM-FSOD: meta-learning for few-shot PCB defect detection. Electronics 14, 1421 (2025).

