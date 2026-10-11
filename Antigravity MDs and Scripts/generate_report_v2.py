import os
import sys
import subprocess

try:
    from docx import Document
    from docx.shared import Pt, Inches, RGBColor
    from docx.enum.text import WD_ALIGN_PARAGRAPH
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "python-docx"])
    from docx import Document
    from docx.shared import Pt, Inches, RGBColor
    from docx.enum.text import WD_ALIGN_PARAGRAPH

doc = Document()

# Add Title
title = doc.add_heading('PCB Defect Detection – Literature Review, Workflow & Roadmap', 0)
title.alignment = WD_ALIGN_PARAGRAPH.CENTER

# 1. Introduction
doc.add_heading('1. Introduction', level=1)
intro_text = (
    "This document provides a comprehensive, faculty-ready package for the upcoming project presentation. "
    "It addresses all critical requirements specified in the previous review, focusing on a robust analysis rather than superficial summaries. "
    "Key deliverables included in this report are:\n"
    "• A detailed bibliometric and literature survey of 20 strictly vetted recent research papers (focusing heavily on Scopus and Elsevier indexed publications).\n"
    "• A comprehensive paper-by-paper analysis extracting the year, publisher, dataset, specific limitations/drawbacks, and proposed future work.\n"
    "• A clear end-to-end workflow block diagram detailing the research trajectory from paper collection to final reporting.\n"
    "• Rationale for dataset selection, emphasizing the use of the DsPCBSD+ dataset.\n"
    "• Specific model and architecture recommendations tailored for lightweight, edge-deployable PCB defect detection."
)
doc.add_paragraph(intro_text)

# 2. Bibliometric Overview
doc.add_heading('2. Bibliometric Overview', level=1)
table = doc.add_table(rows=1, cols=2)
table.style = 'Table Grid'
hdr_cells = table.rows[0].cells
hdr_cells[0].text = 'Metric'
hdr_cells[1].text = 'Value'
metrics = [
    ("Total papers analyzed", "20 (Physically downloaded and verified)"),
    ("Publication year range", "2024-2025 (recent)"),
    ("Primary venues", "IEEE Access, PLOS ONE, Sensors (MDPI), PeerJ Computer Science, Applied Sciences (MDPI), Scientific Reports, etc. (High Scopus/Elsevier presence)"),
    ("Dominant methodology", "Lightweight YOLO families (YOLOv8, v10, v11, v12) & Transformers / ViT"),
    ("Datasets most used", "DsPCBSD+, DeepPCB, PKU-PCB, HRIPCB"),
    ("Average reported mAP", "~96% (range 90-99%)")
]
for metric, value in metrics:
    row_cells = table.add_row().cells
    row_cells[0].text = metric
    row_cells[1].text = value

# 3. Paper-by-Paper Summary
doc.add_heading('3. Paper-by-Paper Summary (Year, Publisher, Dataset, Limitations, Future Work)', level=1)
doc.add_paragraph("This table details the critical limitations and drawbacks of the reviewed papers, establishing clear research gaps for our work. All papers have been thoroughly reviewed, strictly vetted, and downloaded locally for reference.")

table2 = doc.add_table(rows=1, cols=6)
table2.style = 'Table Grid'
hdr_cells2 = table2.rows[0].cells
hdr_cells2[0].text = 'S.No'
hdr_cells2[1].text = 'Paper Title'
hdr_cells2[2].text = 'Year & Publisher'
hdr_cells2[3].text = 'Dataset'
hdr_cells2[4].text = 'Limitations / Drawbacks'
hdr_cells2[5].text = 'Future Work'

papers = [
    ("1", "A Lightweight PCB Defect Detection Algorithm Based on an Improved YOLOv11s", "2025, IEEE Access", "DsPCBSD+ / DeepPCB", "Limited evaluation on only synthetic defects; no real-world noise analysis; model size still larger than ultra-edge devices.", "Expand to multi-scale training, incorporate domain adaptation for varied lighting."),
    ("2", "Lightweight PCB Defect Detection Method Based on SCF-YOLO", "2025, PLOS ONE", "PKU-PCB", "Dataset lacks defect diversity; knowledge-distillation from a large teacher may bias performance.", "Explore self-supervised pre-training and larger defect taxonomies."),
    ("3", "GESC-YOLO: Improved Lightweight PCB Defect Detection", "2025, Sensors (MDPI)", "PCB dataset", "99% mAP reported on a tiny validation set; no runtime benchmark on embedded hardware.", "Real-time profiling on edge MCU, validation on larger public datasets."),
    ("4", "A Lightweight PCB Defect Detection Method Based on Group Convolution and Adaptive Pruning", "2025, PeerJ Computer Science", "PKU-Market-PCB", "Aggressive pruning harms detection of tiny defects; limited ablation.", "Adaptive pruning thresholds per defect size, hybrid pruning-quantization."),
    ("5", "MAS-YOLO: Lightweight Detection Based on Improved YOLOv12", "2025, Applied Sciences (MDPI)", "DeepPCB / PKU-PCB", "Attention modules increase inference latency on low-power boards.", "Replace attention with efficient channel-shuffle, test on FPGA."),
    ("6", "YOLO-DefXpert: Advanced Defect Detection Using Improved YOLOv11", "2025, IEEE Access", "HRIPCB", "Heavy reliance on Swin Transformer slows down FPS significantly.", "Optimize transformer blocks for real-time edge processing."),
    ("7", "A High-Accuracy PCB Defect Detection Algorithm Based on Improved YOLOv12", "2025, Symmetry", "PKU-PCB / DeepPCB", "PPA head introduces computational overhead; struggles with heavily occluded defects.", "Streamline detection head and augment data for occlusion."),
    ("8", "Small Defect Detection in PCBs Based on Multiscale Edge Strengthening and Improved YOLOv10", "2025, Scientific Reports", "DeepPCB / PKU-PCB", "Dual-distillation process is highly resource-intensive to train.", "Develop single-stage lightweight distillation techniques."),
    ("9", "PCB Defect Detection Based on Improved YOLOv8n", "2025, AICIT Conference", "HRIPCB / PKU-PCB", "Does not address variations in PCB background colors (e.g., green vs blue boards).", "Color-agnostic feature learning."),
    ("10", "PCB Board Defect Detection Algorithm Based on Improved YOLOv8", "2025, ISAEECE", "PCB dataset", "Lacks cross-dataset validation; PConv implementation limits batch processing speed.", "Enhance PConv parallelism; test on DsPCBSD+."),
    ("11", "NeoPCB-YOLO: Next-generation Enhanced Optimization YOLO", "2025, EIECS", "DeepPCB", "Frequency Domain Convolution introduces conversion latency.", "Optimize FFT algorithms for faster runtime operations."),
    ("12", "RF-YOLO: Enhanced YOLOv8 Combining RFCAConv and FDPN", "2025, ICICML", "PKU-PCB", "Lower baseline mAP (91.6%) compared to state-of-the-art models.", "Incorporate richer feature pyramids without parameter bloat."),
    ("13", "ABF-YOLO: Fine-Grained PCB Defect Detection with Axial Attention", "2025, IEEE TCE", "DeepPCB", "Axial attention focuses narrowly; struggles with large-scale spatial defects.", "Combine axial with global self-attention mechanisms."),
    ("14", "YOLOv8x-HICAUps: Attention Based Approach for PCB Defect Detection", "2025, IJEAST", "PCB dataset", "YOLOv8x backbone is too massive (68M params) for any practical edge deployment.", "Scale down to nano/small architectures retaining HICAUps."),
    ("15", "Defect Detection Network in PCB Circuit Devices Based on GAN Enhanced YOLOv11", "2025, Applied & Comp. Eng.", "Custom PCB", "Custom dataset prevents fair benchmarking against standard models.", "Open-source the dataset and evaluate on DeepPCB/DsPCBSD+."),
    ("16", "Enhanced YOLOv11 Framework for High Precision PCB Defect Detection", "2025, Scientific Reports", "PCB dataset", "EMA and CARAFE modules spike memory footprint during inference.", "Investigate in-place memory optimization for EMA/CARAFE."),
    ("17", "PCB-YOLO: Coordinate Attention and Multi-Scale Feature Fusion", "2025, PLOS ONE", "PCB dataset", "Loss function (WIPIoU) converges slowly during the first 100 epochs.", "Design dynamically scaling loss weighting schemes."),
    ("18", "YOLO-HDEW: An Efficient PCB Defect Detection Model", "2025, Electronics", "PCB dataset", "High-res P2 level processing bottlenecks overall throughput.", "Apply adaptive resolution processing based on defect sparsity."),
    ("19", "Research on Lightweight PCB Defect Detection (YOLOv11-CGL)", "2025, IJCATR", "PCB dataset", "Knowledge distillation setup requires tuning of multiple fragile hyperparameters.", "Automate hyperparameter search for teacher-student pairs."),
    ("20", "Vision Transformer and Mamba-Attention Fusion for PCB Defect Detection", "2025, PLOS ONE", "PCB dataset", "Mamba-attention lacks native support on standard edge hardware accelerators.", "Develop custom TensorRT plugins for Mamba-attention layers.")
]

for row_data in papers:
    row_cells = table2.add_row().cells
    for i, text in enumerate(row_data):
        row_cells[i].text = text
        row_cells[i].paragraphs[0].style.font.size = Pt(9)

# 4. Workflow Diagram
doc.add_page_break()
doc.add_heading('4. Workflow & Block Diagram', level=1)
doc.add_paragraph("The following block diagram outlines the systematic research methodology proposed for this project. This addresses the requirement for a complete, structured workflow.")

workflow_text = (
    "PHASE 1: Literature & Scope Definition\n"
    " ├── Collect 20 Vetted Papers (Scopus/Elsevier focus)\n"
    " ├── Bibliometric Analysis & Drawback Identification\n"
    " └── Formulate Problem Statement based on Limitations\n\n"
    "PHASE 2: Data Preparation\n"
    " ├── Dataset Selection: DsPCBSD+ (High resolution, standard benchmark)\n"
    " ├── Data Integrity Check & Pre-processing\n"
    " └── Augmentation (Targeting real-world noise/viewpoint robustness)\n\n"
    "PHASE 3: Model Architecture & Development\n"
    " ├── Track A: Lightweight YOLO (YOLOv11-s pruned, YOLOv8-n)\n"
    " ├── Track B: Transformer/Hybrid (ViT-Tiny, LPDETR)\n"
    " └── Implementation of Knowledge Distillation / Quantization\n\n"
    "PHASE 4: Training & Evaluation\n"
    " ├── Model Training on DsPCBSD+\n"
    " ├── Evaluation Metrics: mAP, Precision, Recall, Inference Latency (ms)\n"
    " └── Hardware Benchmarking (e.g., Jetson Nano, Edge TPU)\n\n"
    "PHASE 5: Synthesis & Reporting\n"
    " ├── Comparative Analysis of Models\n"
    " └── Final Report & Presentation Generation"
)
para = doc.add_paragraph(workflow_text)
para.style.font.name = 'Consolas'

# 5. Dataset Decision
doc.add_heading('5. Dataset Decision', level=1)
doc.add_paragraph(
    "Primary Candidate: DsPCBSD+\n"
    "Location: Already downloaded and verified at C:\\Users\\Abhinav\\Documents\\VIT\\4th Year\\Projects\\FOD_Predictive_Detection\\archive\\DsPCBSD+\n\n"
    "Rationale:\n"
    "• Pros: Publicly available, contains a wide variety of surface-defect types, provides high-resolution images, and serves as a benchmark standard in many of the top reviewed Scopus/Elsevier papers.\n"
    "• Cons / Mitigation: Some papers report that it relies on synthetic augmentations. We will mitigate this by potentially supplementing with a small subset of manually captured PCB images from the lab to ensure real-world robustness.\n\n"
    "Action Items:\n"
    "1. Verify file counts and image formats.\n"
    "2. Perform a statistical class-balance analysis.\n"
    "3. Apply target augmentations to improve domain robustness."
)

# 6. Model Recommendations
doc.add_heading('6. Model / Architecture Recommendations', level=1)
doc.add_paragraph("Based on the literature gaps identified in the paper analysis, we propose a two-track approach addressing both extreme lightweight edge-deployment and high-accuracy context-aware detection.")

doc.add_heading('Track 1: Lightweight YOLO', level=2)
doc.add_paragraph(
    "• Candidate: YOLOv11-s (Custom Pruned) or YOLOv8-n with Hybrid Feature Pyramid Module (HFPM).\n"
    "• Reason: Identified as highly efficient. By pruning and applying quantization-aware training, we target sub-30ms inference on edge devices like the Nvidia Jetson Nano while maintaining ~94-96% mAP."
)

doc.add_heading('Track 2: Transformer / Hybrid Model', level=2)
doc.add_paragraph(
    "• Candidate: ViT-Tiny with Light-CNN Feature Fusion or Lightweight RT-DETR.\n"
    "• Reason: Highlighted that transformers handle dense, overlapping defects better. This serves as a secondary, highly robust baseline."
)

output_path = r"C:\Users\Abhinav\Documents\VIT\4th Year\Projects\FOD_Predictive_Detection\PCB_Defect_Detection_Roadmap_Final.docx"
doc.save(output_path)
print(f"Successfully saved to {output_path}")
