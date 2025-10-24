import os
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from ultralytics.utils.plotting import Colors

# =========================
# 0) Pfade & Konfiguration
# =========================
IMG_DIR    = Path("/Users/marcschneider/Documents/MasterArbeit_Experimente/NEW/PlantDoc-3/test/images")
LABELS_DIR = Path("/Users/marcschneider/Documents/MasterArbeit_Experimente/NEW/PlantDoc-3/test/labels")
OUT_DIR    = Path("./review_exports_reduced_025conf")
IMGSZ      = 640

SINGLE_IMAGE = None  # Einzelbild für Testen, sonst None

# Klassen-Namen (aus deiner YAML)
NAMES = [
    "Apple Scab Leaf","Apple leaf","Apple rust leaf","Bell_pepper leaf","Bell_pepper leaf spot",
    "Blueberry leaf","Cherry leaf","Corn Gray leaf spot","Corn leaf blight","Corn rust leaf",
    "Peach leaf","Potato leaf early blight","Potato leaf late blight","Raspberry leaf","Soyabean leaf",
    "Squash Powdery mildew leaf","Strawberry leaf","Tomato Early blight leaf","Tomato Septoria leaf spot",
    "Tomato leaf","Tomato leaf bacterial spot","Tomato leaf late blight","Tomato leaf mosaic virus",
    "Tomato leaf yellow virus","Tomato mold leaf","grape leaf","grape leaf black rot",
]
NC = 27

# Modelle (nur YOLOv8x & BiFormer)
MODELS = [
    (Path("/Users/marcschneider/Documents/MasterArbeit_Experimente/NEW/yolo_baseline/yolov8x_pretrained_split2/weights/best.pt"), 0.25, "yolov8x", "YOLOv8x"),
    (Path("/Users/marcschneider/Documents/MasterArbeit_Experimente/NEW/backbone_experiments/biformer_yolov8_custom_split1/weights/best.pt"), 0.25, "biformer", "BiFormerS-YOLOv8s"),
]

# Farben
palette = Colors()

# =========================
# Hilfsfunktionen
# =========================
def yolo_to_xyxy(box, img_w, img_h):
    cx, cy, w, h = box
    cx *= img_w; cy *= img_h
    w *= img_w; h *= img_h
    x1 = int(max(0, cx - w / 2)); y1 = int(max(0, cy - h / 2))
    x2 = int(min(img_w - 1, cx + w / 2)); y2 = int(min(img_h - 1, cy + h / 2))
    return x1, y1, x2, y2

def draw_boxes_on_image_adaptive(img_rgb, boxes_xyxy, class_ids=None, labels=None, box_thickness=4):
    out = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    font = cv2.FONT_HERSHEY_SIMPLEX
    white = (0, 0, 0)
    pad = 4

    for i, (x1, y1, x2, y2) in enumerate(boxes_xyxy):
        if class_ids is not None and i < len(class_ids) and class_ids[i] >= 0:
            col = tuple(int(c) for c in palette(int(class_ids[i]), bgr=True))
        else:
            col = (0, 255, 0)
        cv2.rectangle(out, (x1, y1), (x2, y2), col, box_thickness)

        if labels is not None and i < len(labels):
            text = str(labels[i])
            (tw, th), base = cv2.getTextSize(text, font, 0.7, 1)
            cv2.rectangle(out, (x1, y1 - th - 2*pad), (x1 + tw + 2*pad, y1), col, -1)
            cv2.putText(out, text, (x1 + pad, y1 - pad), font, 0.7, white, 1, cv2.LINE_AA)

    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)

def load_gt_for_image(img_path: Path):
    lbl_path = LABELS_DIR / (img_path.stem + ".txt")
    if not lbl_path.exists():
        return None, None
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        return None, None
    h, w = img_bgr.shape[:2]

    boxes, class_names = [], []
    with open(lbl_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(float(parts[0]))
            cx, cy, bw, bh = map(float, parts[1:5])
            x1, y1, x2, y2 = yolo_to_xyxy((cx, cy, bw, bh), w, h)
            boxes.append((x1, y1, x2, y2))
            class_names.append(NAMES[cls_id] if 0 <= cls_id < len(NAMES) else str(cls_id))
    return boxes, class_names

def collect_images(img_dir: Path):
    IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    return [p for p in sorted(img_dir.rglob("*")) if p.suffix.lower() in IMG_EXTS]

def iou_xyxy(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2]-a[0]) * max(0.0, a[3]-a[1])
    area_b = max(0.0, b[2]-b[0]) * max(0.0, b[3]-b[1])
    union = area_a + area_b - inter + 1e-9
    return inter / union

def f1_for_image(pred_boxes, pred_cls, gt_boxes, gt_cls, iou_thr=0.5):
    if len(gt_boxes) == 0 and len(pred_boxes) == 0:
        return 1.0
    if len(gt_boxes) == 0 or len(pred_boxes) == 0:
        return 0.0
    tp = 0; fp = 0
    gt_unused = {c: set(np.where(gt_cls == c)[0]) for c in np.unique(gt_cls)}
    for i, box in enumerate(pred_boxes):
        c = pred_cls[i]
        candidates = list(gt_unused.get(c, []))
        best = -1; best_iou = 0
        for j in candidates:
            iou = iou_xyxy(box, gt_boxes[j])
            if iou >= iou_thr and iou > best_iou:
                best_iou, best = iou, j
        if best >= 0:
            tp += 1; gt_unused[c].remove(best)
        else:
            fp += 1
    fn = sum(len(v) for v in gt_unused.values())
    prec = tp / (tp + fp + 1e-9)
    rec = tp / (tp + fn + 1e-9)
    return 2*prec*rec / (prec + rec + 1e-9)

# =========================
# Main
# =========================
def review_folder_reduced():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    image_paths = [SINGLE_IMAGE] if SINGLE_IMAGE and SINGLE_IMAGE.exists() else collect_images(IMG_DIR)
    if not image_paths:
        print("Keine Bilder gefunden."); return

    # Modelle laden
    model_objs = []
    for (w, c, tag, caption) in MODELS:
        m = YOLO(str(w)); m.model.eval()
        model_objs.append((m, c, tag, caption))

    counters = {"yolov8x_better":0, "biformer_better":0, "even":0}

    print(f"Gefundene Bilder: {len(image_paths)}")

    for img_path in image_paths:
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Ground Truth
        gt_boxes, gt_names = load_gt_for_image(img_path)
        if gt_boxes:
            name_to_id = {n: i for i, n in enumerate(NAMES)}
            gt_cls_ids = np.array([name_to_id.get(n, -1) for n in gt_names], int)
            gt_img = draw_boxes_on_image_adaptive(img_rgb, gt_boxes, class_ids=gt_cls_ids, labels=gt_names)
        else:
            gt_img = img_rgb.copy()
            gt_cls_ids = np.array([]); gt_boxes = []

        panels = [gt_img]; captions = ["(a) Ground Truth"]
        scores = {}

        for (model, conf_thr, tag, caption) in model_objs:
            results = model.predict(
                source=str(img_path),
                imgsz=IMGSZ,
                save=False, show=False, verbose=False,
                agnostic_nms=True, conf=conf_thr
            )
            pred_bgr = results[0].plot()
            pred_rgb = cv2.cvtColor(pred_bgr, cv2.COLOR_BGR2RGB)
            panels.append(pred_rgb)
            captions.append("(b) YOLOv8x" if tag=="yolov8x" else "(c) BiFormerS-YOLOv8s")

            b = results[0].boxes
            if b is not None and len(b) > 0:
                pred_boxes = b.xyxy.cpu().numpy()
                pred_cls = b.cls.cpu().numpy().astype(int)
            else:
                pred_boxes, pred_cls = np.zeros((0,4)), np.array([], int)
            f1 = f1_for_image(pred_boxes, pred_cls, np.array(gt_boxes), gt_cls_ids)
            scores[tag] = f1

        # Gewinner bestimmen
        y8x, bif = scores.get("yolov8x",0), scores.get("biformer",0)
        if abs(y8x - bif) < 1e-3:
            best = "even"
        elif y8x > bif:
            best = "yolov8x_better"
        else:
            best = "biformer_better"

        counters[best] += 1
        out_stem = f"{best}{counters[best]}"

        # Plot mit Beschriftung unter den Bildern
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        for ax, img, cap in zip(axes, panels, captions):
            ax.imshow(img); ax.axis("off")
            ax.text(0.5, -0.05, cap, transform=ax.transAxes,
                    ha="center", va="top", fontsize=14, fontweight="bold")

        # Speichern
        fig.savefig(str(OUT_DIR / f"{out_stem}.png"), dpi=600, bbox_inches="tight")
        fig.savefig(str(OUT_DIR / f"{out_stem}.pdf"), dpi=600, bbox_inches="tight")
        plt.close(fig)

        print(f"{out_stem} gespeichert ({scores})")

if __name__ == "__main__":
    review_folder_reduced()
