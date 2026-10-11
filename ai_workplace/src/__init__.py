# PCB Defect Detection - Adaptive Preprocessing Pipeline
# ai_workplace/src package

from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATASET_ROOT = PROJECT_ROOT / "Dataset" / "DsPCBSD+" / "Data_YOLO"
AI_WORKPLACE = PROJECT_ROOT / "ai_workplace"
RESULTS_DIR = AI_WORKPLACE / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"

CLASS_NAMES = {
    0: "SH",    # Short Circuit
    1: "SP",    # Spur
    2: "SC",    # Spurious Copper / Scratches
    3: "OP",    # Open Circuit
    4: "MB",    # Mouse Bite / Mask Break
    5: "HB",    # Hole Break
    6: "CS",    # Copper Sliver / Cremation
    7: "CFO",   # Coarse Foreign Object
    8: "BMFO",  # Ball-shaped Micro Foreign Object
}

NUM_CLASSES = 9
