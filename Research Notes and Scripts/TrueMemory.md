# Project TrueMemory: Predictive FOD/DOD System

## Core Concept
- **Goal:** Shift from reactive detection to **predictive prevention** of Foreign Object Damage (FOD).
- **Domains Evaluated:** Aviation (FOD on runways), PCB (metal particles), Automotive (road hazards), Marine (plastic waste).

## Key Directives (From Dr. Parthiban K)
1. **Focus:** Decide the project domain strictly based on the **availability of datasets**.
2. **Actionable Metric:** Determine *where* the FOD happens and *which part* is affected.

## Current Status & Next Steps
- **Evaluation Complete:** Extensive research has been done on dataset availability and research density across all 4 domains.
- **Research Saturation (Density Analysis):**
  - **Marine (Ship Water Intake):** *Extremely Low (Niche)*. Almost no computer vision literature exists for predicting filter clogging or monitoring marine debris near ship intakes.
  - **PCB Manufacturing (Foreign Objects):** *Low to Moderate*. Highly active for general defects, but specifically detecting foreign objects/metal debris is a wide research gap.
  - **Aviation (Runway/Engine Intake):** *Moderate*. Standardized but active (e.g., YOLO and radar integration). Secured airport environments limit real-world dataset acquisition.
  - **Automotive (Road Hazards):** *Extremely High (Over-saturated)*. Very crowded domain due to the autonomous driving boom.
- **Aviation Pivot Required?:** The team originally thought Aviation was too complex, but Computer Vision analysis shows it is actually the **easiest** to implement because runways are uniform, clean backgrounds compared to chaotic roads or murky water. Also, Aviation has the most robust datasets available (FOD-A, Roboflow).
- **Alternative:** If Aviation is strictly rejected, **PCB Manufacturing** (Detecting Foreign Metal Particles on Tracks) is the next best option due to excellent dataset availability (DsPCBSD+, PCB-Bank) and high publishability.

## Dataset Repository Links
- **Aviation:**
  - [Roboflow Universe (FOD-brais)](https://universe.roboflow.com/dylog-rxjz2/fod-brais)
  - [Roboflow Universe (FOD-bi62f)](https://universe.roboflow.com/test-momjj/fod-bi62f/dataset/1)
  - [FOD-A Dataset (Kaggle/GitHub)](https://github.com/FOD-UNOmaha/FOD-data)
- **PCB:**
  - **DsPCBSD+ (Dataset of PCB Surface Defects):** Available on [Figshare](https://doi.org/10.6084/m9.figshare.24970329) or community mirrored on [Kaggle](https://www.kaggle.com/datasets/enisteper/dataset-of-pcb-surface-defects-dspcbsd) and [Roboflow Universe](https://universe.roboflow.com/pcb/dspcbsd-plus). (Contains 10,000+ images with CFO / BMFO classifications).
  - **DeepPCB:** Available on [GitHub tangsanli5201/DeepPCB](https://github.com/tangsanli5201/DeepPCB). (Contains 1,500 aligned pairs of PCB traces with shorts/opens).
  - **PCB-Bank:** Available on [GitHub SSRheart/PCB-Bank](https://github.com/SSRheart/PCB-Bank). (Download via BaiduYun code `mx57` or request email for RealIAD subsets).
- **Automotive:**
  - RDD2022, TyreNet
- **Marine:**
  - TrashCan, TACO

## Project Structure
- `src/`: For YOLOv8/Model training and inference scripts.
- `data/`: For dataset storage.
- `Project_Charter.md`: Original team notes.
- `TrueMemory.md`: This file, serving as the active context and memory for the AI.
