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
OUT_DIR    = Path("./review_exports_modular_025conf")
IMGSZ      = 640
IOU_THR    = 0.5
EVEN_TIE_ATOL = 1e-3

# Klassen-Namen (aus YAML)
NAMES = [
    "Apple Scab Leaf","Apple leaf","Apple rust leaf","Bell_pepper leaf","Bell_pepper leaf spot",
    "Blueberry leaf","Cherry leaf","Corn Gray leaf spot","Corn leaf blight","Corn rust leaf",
    "Peach leaf","Potato leaf early blight","Potato leaf late blight","Raspberry leaf","Soyabean leaf",
    "Squash Powdery mildew leaf","Strawberry leaf","Tomato Early blight leaf","Tomato Septoria leaf spot",
    "Tomato leaf","Tomato leaf bacterial spot","Tomato leaf late blight","Tomato leaf mosaic virus",
    "Tomato leaf yellow virus","Tomato mold leaf","grape leaf","grape leaf black rot",
]
NC = 27

MODELS = [
    (Path("/Users/marcschneider/Documents/MasterArbeit_Experimente/NEW/yolo_baseline/yolov8s_official_split0/weights/best.pt"), 0.25),
    (Path("/Users/marcschneider/Documents/MasterArbeit_Experimente/NEW/modular_experiments/yolov8s_P3C2fBF_official_split0/weights/best.pt"), 0.25),
    (Path("/Users/marcschneider/Documents/MasterArbeit_Experimente/NEW/modular_experiments/yolov8s_P3C2fSTRV1_official_split2/weights/best.pt"), 0.25),
    (Path("/Users/marcschneider/Documents/MasterArbeit_Experimente/NEW/modular_experiments/yolov8s_P10C2fPVTV2_official_split0/weights/best.pt"), 0.25),
]
TAGS = ["yolov8s", "c2f-bf", "c2f-str", "c2f-pvt"]
CAPTIONS = [
    "Ground Truth",
    "YOLOv8s",
    "YOLOv8s-P3C2f-BF",
    "YOLOv8s-P3C2f-STR",
    "YOLOv8s-P5C2f-PVT",
]
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

palette = Colors()

# =========================
# Hilfsfunktionen
# =========================
def yolo_to_xyxy(box, img_w, img_h):
    cx, cy, w, h = box
    cx *= img_w; cy *= img_h; w *= img_w; h *= img_h
    x1 = int(max(0, cx - w / 2)); y1 = int(max(0, cy - h / 2))
    x2 = int(min(img_w - 1, cx + w / 2)); y2 = int(min(img_h - 1, cy + h / 2))
    return x1, y1, x2, y2

def draw_boxes_on_image(img_rgb, boxes_xyxy, class_ids=None, labels=None, thickness=3):
    out = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    for i, (x1, y1, x2, y2) in enumerate(boxes_xyxy):
        color = tuple(int(c) for c in palette(int(class_ids[i]) if class_ids is not None else i, bgr=True))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        if labels is not None and i < len(labels):
            cv2.putText(out, labels[i], (x1, max(15, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)

def load_gt(img_path):
    lbl_path = LABELS_DIR / (img_path.stem + ".txt")
    if not lbl_path.exists():
        return [], []
    img = cv2.imread(str(img_path))
    h, w = img.shape[:2]
    boxes, cls_names = [], []
    with open(lbl_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(float(parts[0]))
            x1, y1, x2, y2 = yolo_to_xyxy(list(map(float, parts[1:5])), w, h)
            boxes.append((x1, y1, x2, y2))
            cls_names.append(NAMES[cls_id])
    return boxes, cls_names

def iou_xyxy(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (a[2]-a[0])*(a[3]-a[1]); area_b = (b[2]-b[0])*(b[3]-b[1])
    return inter / (area_a + area_b - inter + 1e-9)

def f1_for_image(pred_boxes, pred_cls, gt_boxes, gt_cls, iou_thr=IOU_THR):
    if len(gt_boxes) == 0 and len(pred_boxes) == 0: return 1.0, 0, 0, 0
    if len(gt_boxes) == 0: return 0.0, 0, len(pred_boxes), 0
    if len(pred_boxes) == 0: return 0.0, 0, 0, len(gt_boxes)

    gt_unused = {c: set(np.where(gt_cls == c)[0]) for c in np.unique(gt_cls)}
    tp = 0; fp = 0
    for i, pbox in enumerate(pred_boxes):
        c = int(pred_cls[i])
        candidates = list(gt_unused.get(c, []))
        best_j, best_iou = -1, 0.0
        for j in candidates:
            iou = iou_xyxy(pbox, gt_boxes[j])
            if iou >= iou_thr and iou > best_iou:
                best_iou, best_j = iou, j
        if best_j >= 0:
            tp += 1; gt_unused[c].remove(best_j)
        else:
            fp += 1
    fn = sum(len(v) for v in gt_unused.values())
    precision = tp / (tp + fp + 1e-9)
    recall    = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    return f1, tp, fp, fn

# =========================
# Hauptfunktion
# =========================
def review_folder():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    image_paths = [p for p in IMG_DIR.glob("*") if p.suffix.lower() in IMG_EXTS]
    print(f"Gefundene Bilder: {len(image_paths)}")

    # Modelle laden
    model_objs = []
    for (weights, conf), tag, cap in zip(MODELS, TAGS, CAPTIONS[1:]):
        model = YOLO(str(weights))
        model.model.eval()
        model_objs.append((model, conf, tag, cap))

    counters = {tag: 0 for tag in TAGS}
    counters.update({"even": 0, "transformer_better": 0})

    for img_path in image_paths:
        img_bgr = cv2.imread(str(img_path))
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        gt_boxes, gt_names = load_gt(img_path)
        name_to_id = {n: i for i, n in enumerate(NAMES)}
        gt_cls_ids = np.array([name_to_id.get(n, -1) for n in gt_names], int)
        gt_boxes_arr = np.array(gt_boxes, float)

        gt_img = draw_boxes_on_image(img_rgb, gt_boxes, gt_cls_ids, gt_names) if gt_boxes else img_rgb.copy()

        panels = [gt_img]
        captions = ["a) Ground Truth"]
        model_scores = {}

        # --- Modelle vorhersagen ---
        for model, conf_thr, tag, caption in model_objs:
            results = model.predict(str(img_path), imgsz=IMGSZ, conf=conf_thr, agnostic_nms=True, verbose=False)
            pred = results[0]
            pred_rgb = cv2.cvtColor(pred.plot(), cv2.COLOR_BGR2RGB)
            panels.append(pred_rgb)
            captions.append(f"{chr(ord('a')+len(captions))}) {caption}")

            b = pred.boxes
            pred_boxes = b.xyxy.cpu().numpy().astype(float) if b is not None and len(b) else np.zeros((0, 4))
            pred_cls   = b.cls.cpu().numpy().astype(int) if b is not None and len(b) else np.array([], int)

            f1, *_ = f1_for_image(pred_boxes, pred_cls, gt_boxes_arr, gt_cls_ids, IOU_THR)
            model_scores[tag] = f1

        # --- Bestes Modell bestimmen ---
        max_score = max(model_scores.values()) if model_scores else 0
        winners = [t for t, s in model_scores.items() if abs(s - max_score) <= EVEN_TIE_ATOL]

        transformer_tags = {"c2f-bf", "c2f-str", "c2f-pvt"}
        yolov8_score = model_scores.get("yolov8s", -np.inf)
        transf_better = [t for t in transformer_tags if model_scores.get(t, -np.inf) > yolov8_score + EVEN_TIE_ATOL]

        if len(winners) > 1:
            if len(transf_better) >= 2:
                best_tag = "transformer_better"
            else:
                best_tag = "even"
        else:
            best_tag = winners[0]

        counters[best_tag] = counters.get(best_tag, 0) + 1

        # --- Figure speichern ---
        out_stem = f"{best_tag}_better{counters[best_tag]}" if best_tag not in ["even", "transformer_better"] else f"{best_tag}{counters[best_tag]}"
        fig, axes = plt.subplots(2, 3, figsize=(16, 10))
        axes = axes.flatten()
        for ax, img, cap in zip(axes, panels, captions):
            ax.imshow(img)
            ax.axis("off")
            ax.text(0.5, -0.02, cap, transform=ax.transAxes, ha="center", va="top", fontsize=14, fontweight="bold")
        for ax in axes[len(panels):]:
            ax.axis("off")
        fig.savefig(OUT_DIR / f"{out_stem}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"{img_path.name} → {out_stem}  {model_scores} | Transformer gleich gut: {transf_better}")

if __name__ == "__main__":
    review_folder()
