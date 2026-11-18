"""Validation-set predictions for the two paper checkpoints (no retraining, no tuning)."""
import json
import shutil
import sys
from pathlib import Path

from ultralytics import RTDETR, YOLO

ROOT = Path(r"C:\Users\Abhinav\Documents\VIT\4th Year\Projects\FOD_Predictive_Detection")
OUT = ROOT / "ai_workplace" / "results" / "predictions"
OUT.mkdir(parents=True, exist_ok=True)
SCRATCH = Path(sys.argv[1])

yaml = SCRATCH / "dspcbsd_official.yaml"
yaml.write_text(
    f"path: {(ROOT / 'Dataset/DsPCBSD+/Data_YOLO').as_posix()}\n"
    "train: images/train\nval: images/val\nnc: 9\n"
    "names:\n  0: SH\n  1: SP\n  2: SC\n  3: OP\n  4: MB\n  5: HB\n  6: CS\n  7: CFO\n  8: BMFO\n")

runs = [("yolo26n", YOLO, ROOT / "MODEL RESULTS/MORE TRAINING/YOLO/best.pt"),
        ("rtdetr_l", RTDETR, ROOT / "MODEL RESULTS/MORE TRAINING/DETR/best.pt")]
summary = {}
for name, cls, ckpt in runs:
    model = cls(str(ckpt))
    r = model.val(data=str(yaml), split="val", imgsz=640, batch=8, device="cpu",
                  save_json=True, plots=False, project=str(SCRATCH / "runs"), name=name, exist_ok=True)
    shutil.copy(Path(r.save_dir) / "predictions.json", OUT / f"{name}_val_predictions.json")
    names = model.names
    rows = ["class,precision,recall,mAP50,mAP50-95"]
    for i in range(len(names)):
        rows.append(f"{names[i]},{r.box.p[i]:.4f},{r.box.r[i]:.4f},{r.box.ap50[i]:.4f},{r.box.ap[i]:.4f}")
    (OUT / f"{name}_val_per_class.csv").write_text("\n".join(rows) + "\n")
    summary[name] = {"precision": round(float(r.box.mp), 4), "recall": round(float(r.box.mr), 4),
                     "mAP50": round(float(r.box.map50), 4), "mAP50-95": round(float(r.box.map), 4)}
    print(name, summary[name], flush=True)

summary["_note"] = ("DsPCBSD+ official validation partition, 2,051 images / 4,092 instances, imgsz 640, "
                    "conf 0.001 / IoU 0.7 (Ultralytics val defaults); checkpoints from MODEL RESULTS/MORE TRAINING.")
(OUT / "val_summary.json").write_text(json.dumps(summary, indent=2))
print("done")
