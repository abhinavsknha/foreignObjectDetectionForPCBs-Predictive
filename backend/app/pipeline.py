"""Inference pipeline used by the web back end.

The Adaptive Resolution Router (ARR), the 2x super-resolution of Eq. 9 and the asymmetric
Weighted Box Fusion follow the evaluation notebook: the router looks at the image at its own
resolution and decides whether super-resolution is applied, both detectors then run on the
same image (letterboxed to 640 x 640 by the detector), and only the fused output is returned.
"""
import os
import threading
import time
from pathlib import Path

import cv2
import numpy as np
from ensemble_boxes import weighted_boxes_fusion

FRAME = 640
DETECTOR_CONF = 0.05          # detector confidence floor before fusion (evaluation setting)
REPORT_CONF = 0.25            # fused boxes below this score are dropped
W_YOLO, W_DETR = 1.0, 1.5     # asymmetric track weights
FUSE_IOU = 0.50
TAU_EDGE, TAU_CONTRAST, TAU_HF = 0.030, 25.0, 0.700
SR_FACTOR, SR_ALPHA, SR_SIGMA = 2, 1.5, 3
# "gate" applies super-resolution when the resolution check asks for it (as documented);
# "off" still runs and reports the check but never enhances.
SR_POLICY = os.getenv("SR_POLICY", "gate").lower()

CLASS_MAP = {
    "SH": "Short",
    "SP": "Spur",
    "SC": "Spurious Copper",
    "OP": "Open",
    "MB": "Mouse Bite",
    "HB": "Hole Breakout",
    "CS": "Conductor Scratch",
    "CFO": "Conductor Foreign Object",
    "BMFO": "Base Material Foreign Object",
}

TIPS = {
    "SH": "Short: Unintended connection. Inspect wave soldering temp and residual copper.",
    "SP": "Spur: Irregular protrusion found. Check etching chemical balance.",
    "SC": "Spurious Copper: Unwanted residue on base material. Review etching exposure timing.",
    "OP": "Open: Interrupted conductor path. Halt line and check photoresist application.",
    "MB": "Mouse Bite: Gap at conductor edge. Inspect routing/milling tools for wear.",
    "HB": "Hole Breakout: Hole center deviation. Recalibrate CNC drilling coordinates.",
    "CS": "Conductor Scratch: Inspect conveyor belts and automated handling arms for sharp edges.",
    "CFO": "Conductor Foreign Object: Dirt/bubbles on conductor. Clean lamination rollers.",
    "BMFO": "Base Material Foreign Object: Contamination on base. Check cleanroom air filters.",
}

_REPO_ROOT = Path(__file__).resolve().parents[2]
_BACKEND_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------- models
class Models:
    def __init__(self, yolo, detr):
        self.yolo, self.detr = yolo, detr
        if yolo.names != detr.names:
            raise RuntimeError("YOLO and RT-DETR checkpoints disagree on class names")
        self.names = dict(yolo.names)


_models = None
_models_lock = threading.Lock()


def _weights_path(env_name, candidates):
    configured = os.getenv(env_name)
    paths = ([Path(configured)] if configured else []) + candidates
    for p in paths:
        if p.exists():
            return p
    raise FileNotFoundError(
        f"No weights found for {env_name}. Set {env_name} or place the file at one of: "
        + ", ".join(str(p) for p in candidates)
    )


def get_models():
    """Load both detectors once per process."""
    global _models
    with _models_lock:
        if _models is None:
            from ultralytics import RTDETR, YOLO

            yolo_path = _weights_path("YOLO_WEIGHTS", [
                _BACKEND_ROOT / "models" / "yolo_best.pt",
                _REPO_ROOT / "MODEL RESULTS" / "MORE TRAINING" / "YOLO" / "best.pt",
            ])
            detr_path = _weights_path("DETR_WEIGHTS", [
                _BACKEND_ROOT / "models" / "detr_best.pt",
                _REPO_ROOT / "MODEL RESULTS" / "MORE TRAINING" / "DETR" / "best.pt",
            ])
            loaded = Models(YOLO(str(yolo_path)), RTDETR(str(detr_path)))
            # The first predict call in a process is slow; run one on a blank frame now
            # so the first real upload is not the one that pays for it.
            blank = np.zeros((FRAME, FRAME, 3), np.uint8)
            for m in (loaded.yolo, loaded.detr):
                m.predict(source=blank, imgsz=FRAME, conf=DETECTOR_CONF, verbose=False)
            _models = loaded
        return _models


def models_loaded():
    return _models is not None


# ---------------------------------------------------------------- size limit
def limit_size(img):
    """Keep the native resolution; only shrink very large scans so the long side is at most 640."""
    h, w = img.shape[:2]
    if max(h, w) <= FRAME:
        return img
    scale = FRAME / max(h, w)
    return cv2.resize(img, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------- ARR gate
def check_resolution(resized):
    """Edge density, local contrast and FFT high-frequency ratio decide the route (Eq. 4.1)."""
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    d_edge = float(np.count_nonzero(cv2.Canny(gray, 50, 150)) / (h * w))

    blurred = cv2.GaussianBlur(gray, (31, 31), 0)
    sigma_contrast = float(np.std(cv2.absdiff(gray, blurred)))

    mag = np.abs(np.fft.fftshift(np.fft.fft2(gray))) ** 2
    cy, cx = h // 2, w // 2
    radius = min(h, w) // 8
    y, x = np.ogrid[:h, :w]
    low = np.sum(mag[(x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2])
    total = np.sum(mag)
    eta_hf = float((total - low) / total) if total > 0 else 0.0

    low_texture = d_edge < TAU_EDGE and sigma_contrast < TAU_CONTRAST
    low_hf = eta_hf < TAU_HF
    needs_sr = bool(low_texture or low_hf)

    reasons = []
    if low_texture:
        reasons.append(f"edge density {d_edge:.3f} < {TAU_EDGE} and contrast {sigma_contrast:.1f} < {TAU_CONTRAST}")
    if low_hf:
        reasons.append(f"high-frequency energy {eta_hf:.3f} < {TAU_HF}")
    return {
        "needs_sr": needs_sr,
        "path": "super_resolution" if needs_sr else "direct",
        "reason": ("Weak texture: " + "; ".join(reasons)) if needs_sr
                  else "Edge density, contrast and high-frequency energy are all above their thresholds",
        "metrics": {"edge_density": d_edge, "contrast": sigma_contrast, "hf_ratio": eta_hf},
        "thresholds": {"edge_density": TAU_EDGE, "contrast": TAU_CONTRAST, "hf_ratio": TAU_HF},
    }


def apply_super_resolution(img, factor=SR_FACTOR, alpha=SR_ALPHA):
    """2x bicubic upscale, unsharp mask (paper Eq. 9), area-averaged back to the frame size."""
    h, w = img.shape[:2]
    up = cv2.resize(img, (int(w * factor), int(h * factor)), interpolation=cv2.INTER_CUBIC)
    blurred = cv2.GaussianBlur(up, (0, 0), sigmaX=SR_SIGMA)
    sharp = cv2.addWeighted(up, 1.0 + alpha, blurred, -alpha, 0)
    return cv2.resize(sharp, (w, h), interpolation=cv2.INTER_AREA)


# ---------------------------------------------------------------- detection and fusion
def _predict(model, frame):
    r = model.predict(source=frame, imgsz=FRAME, conf=DETECTOR_CONF, verbose=False)[0]
    b = r.boxes
    if b is None or len(b) == 0:
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int)
    return (b.xyxyn.cpu().numpy().clip(0, 1), b.conf.cpu().numpy(),
            b.cls.cpu().numpy().astype(int))


def fuse_predictions_wbf(yolo_boxes, yolo_scores, yolo_classes,
                         detr_boxes, detr_scores, detr_classes,
                         w_yolo=W_YOLO, w_detr=W_DETR, iou_thr=FUSE_IOU, conf_thr=REPORT_CONF):
    """Asymmetric Weighted Box Fusion with confidence discounting."""
    if len(yolo_boxes) == 0 and len(detr_boxes) == 0:
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int)
    if len(yolo_boxes) == 0:
        keep = detr_scores * (w_detr / (w_yolo + w_detr)) >= conf_thr
        return detr_boxes[keep], detr_scores[keep] * 0.5, detr_classes[keep]
    if len(detr_boxes) == 0:
        keep = yolo_scores * (w_yolo / (w_yolo + w_detr)) >= conf_thr
        return yolo_boxes[keep], yolo_scores[keep] * 0.5, yolo_classes[keep]

    f_boxes, f_scores, f_labels = weighted_boxes_fusion(
        [yolo_boxes, detr_boxes], [yolo_scores, detr_scores], [yolo_classes, detr_classes],
        weights=[w_yolo, w_detr], iou_thr=iou_thr, skip_box_thr=0.01)
    keep = f_scores >= conf_thr
    return f_boxes[keep], f_scores[keep], f_labels[keep].astype(int)


def _iou(box, boxes):
    x1 = np.maximum(box[0], boxes[:, 0]); y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2]); y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    a = (box[2] - box[0]) * (box[3] - box[1])
    b = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / (a + b - inter + 1e-9)


def _best_match(box, label, boxes, scores, classes, min_iou=0.4):
    """Highest-scoring detector box of the same class that overlaps the fused box."""
    if len(boxes) == 0:
        return None
    ok = (classes == label) & (_iou(box, boxes) >= min_iou)
    return float(scores[ok].max()) if ok.any() else None


# ---------------------------------------------------------------- full run
def run_pipeline(image_path, models, progress=None):
    """Resolution check, optional super-resolution, both detectors, fusion.

    progress(stage, route=None) is called as each stage starts so the UI can show it.
    Returned boxes are in the pixel coordinates of the uploaded image.
    """
    emit = progress or (lambda *a, **k: None)
    t_start = time.perf_counter()

    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("The uploaded file could not be read as an image.")
    H, W = img.shape[:2]

    emit("checking_resolution")
    t = time.perf_counter()
    base = limit_size(img)
    route = check_resolution(base)
    route["native"] = {"width": W, "height": H}
    route["gate_flagged"] = route["needs_sr"]
    if route["needs_sr"] and SR_POLICY == "off":
        route.update(needs_sr=False, path="direct",
                     reason=route["reason"] + ". Enhancement is switched off on this server, so the image is used as uploaded")
    route["method"] = (f"Bicubic {SR_FACTOR}x + unsharp mask (alpha {SR_ALPHA})" if route["needs_sr"] else None)
    timings = {"resolution_check": (time.perf_counter() - t) * 1000}

    if route["needs_sr"]:
        emit("enhancing", route=route)
        t = time.perf_counter()
        base = apply_super_resolution(base)
        timings["super_resolution"] = (time.perf_counter() - t) * 1000

    emit("detecting", route=route)
    t = time.perf_counter()
    yb, ys, yc = _predict(models.yolo, base)
    timings["yolo"] = (time.perf_counter() - t) * 1000
    t = time.perf_counter()
    db, ds, dc = _predict(models.detr, base)
    timings["rtdetr"] = (time.perf_counter() - t) * 1000

    emit("fusing", route=route)
    t = time.perf_counter()
    fb, fs, fc = fuse_predictions_wbf(yb, ys, yc, db, ds, dc)
    timings["fusion"] = (time.perf_counter() - t) * 1000

    detections = []
    for box, score, label in zip(fb, fs, fc):
        y_conf = _best_match(box, label, yb, ys, yc)
        d_conf = _best_match(box, label, db, ds, dc)
        source = ("ENSEMBLE (WBF)" if (y_conf is not None and d_conf is not None)
                  else "YOLO26n" if y_conf is not None else "RT-DETR")
        x1, x2 = (float(np.clip(box[0] * W, 0, W)), float(np.clip(box[2] * W, 0, W)))
        y1, y2 = (float(np.clip(box[1] * H, 0, H)), float(np.clip(box[3] * H, 0, H)))
        name = models.names[int(label)]
        detections.append({
            "class": name,
            "full_name": CLASS_MAP.get(name, name),
            "confidence": float(score),
            "bbox": [x1, y1, x2, y2],
            "source": source,
            "yolo_conf": y_conf,
            "detr_conf": d_conf,
        })
    detections.sort(key=lambda d: d["confidence"], reverse=True)

    tips = [TIPS[c] for c in dict.fromkeys(d["class"] for d in detections) if c in TIPS]
    timings["total"] = (time.perf_counter() - t_start) * 1000
    return {
        "yolo_count": int((ys >= REPORT_CONF).sum()),
        "rtdetr_count": int((ds >= REPORT_CONF).sum()),
        "fused_count": len(detections),
        "detections": detections,
        "tips": tips,
        "route": route,
        "image": {"width": W, "height": H},
        "timings_ms": {k: round(v, 1) for k, v in timings.items()},
        "fusion": {"w_yolo": W_YOLO, "w_detr": W_DETR, "iou": FUSE_IOU, "min_confidence": REPORT_CONF},
    }
