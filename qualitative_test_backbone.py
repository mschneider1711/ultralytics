import os
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
from ultralytics import YOLO
from ultralytics.utils.plotting import Colors
# Palette einmal erzeugen (gern ganz oben vor der for-Schleife)
palette = Colors()


# =========================
# 0) Pfade & Konfiguration
# =========================
IMG_DIR    = Path("/Users/marcschneider/Documents/MasterArbeit_Experimente/NEW/PlantDoc-3/test/images")
LABELS_DIR = Path("/Users/marcschneider/Documents/MasterArbeit_Experimente/NEW/PlantDoc-3/test/labels")
OUT_DIR    = Path("./review_exports6")
IMGSZ      = 640
EVEN_TIE_ATOL = 1e-3     # Gleichstands-Toleranz
PRIORITY_ATOL = 1e-3     # Toleranz für "BiFormer besser als YOLOv8x"# EINZELBILD statt Ordner
SINGLE_IMAGE = None
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

MODELS = [
    (Path("/Users/marcschneider/Documents/MasterArbeit_Experimente/NEW/yolo_baseline/yolov8x_pretrained_split2/weights/best.pt"), 0.25),
    (Path("/Users/marcschneider/Documents/MasterArbeit_Experimente/NEW/backbone_experiments/biformer_yolov8_custom_split1/weights/best.pt"), 0.25),
    (Path("/Users/marcschneider/Documents/MasterArbeit_Experimente/NEW/backbone_experiments/pvt2_yolov8_custom_split2/weights/best.pt"), 0.25),
    (Path("/Users/marcschneider/Documents/MasterArbeit_Experimente/NEW/backbone_experiments/swinsv2_yolov8_custom_split2/weights/best.pt"), 0.25),
]
TAGS = ["yolov8x", "biformer", "pvt", "swins"]  # Dateinamens-Präfixe je Modell

# Fest definierte Bildunterschriften (werden mit a)/b)/c) … kombiniert)
CAPTIONS = [
    "Ground Truth",
    "YOLOv8x",
    "BiFormerS-YOLOv8s",
    "PVTv2-B2-Li-YOLOv8s",
    "SwinS-YOLOv8s",
]

# Unterstützte Bildendungen
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

# IoU-Schwelle für Übereinstimmung (für F1)
IOU_THR = 0.5
EVEN_ATOL = 1e-6  # Toleranz für "alle gleich gut/schlecht"

# =========================
# 1) Utilities
# =========================
def yolo_to_xyxy(box, img_w, img_h):
    """YOLO (cx, cy, w, h) normalisiert -> (x1, y1, x2, y2) Pixel"""
    cx, cy, w, h = box
    cx *= img_w
    cy *= img_h
    w  *= img_w
    h  *= img_h
    x1 = int(max(0, cx - w / 2))
    y1 = int(max(0, cy - h / 2))
    x2 = int(min(img_w - 1, cx + w / 2))
    y2 = int(min(img_h - 1, cy + h / 2))
    return x1, y1, x2, y2

from ultralytics.utils.plotting import Colors
palette = Colors()

def draw_boxes_on_image_adaptive(
    img_rgb,
    boxes_xyxy,
    class_ids=None,
    labels=None,
    box_thickness=4,
):
    out = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    font = cv2.FONT_HERSHEY_SIMPLEX
    pad = 4
    white = (0, 0, 0)

    def text_size(t, fs_, thick_):
        (tw, th), base = cv2.getTextSize(t, font, fs_, thick_)
        rect_w = tw + 2 * pad
        rect_h = th + base + 2 * pad
        return tw, th, base, rect_h, rect_w

    for i, (x1, y1, x2, y2) in enumerate(boxes_xyxy):
        # Farbe wie Ultralytics
        if class_ids is not None and i < len(class_ids) and class_ids[i] >= 0:
            col = tuple(int(c) for c in palette(int(class_ids[i]), bgr=True))
        else:
            col = (0, 255, 0)

        cv2.rectangle(out, (x1, y1), (x2, y2), col, box_thickness)

        text = None
        if labels is not None and i < len(labels) and labels[i]:
            text = str(labels[i])
        if not text:
            continue

        bw = max(1, x2 - x1)
        bh = max(1, y2 - y1)

        fs = float(np.clip(bh * 0.035, 0.6, 1.6))
        thickness = max(1, int(round(box_thickness * 0.6)))

        # 1) Einzeilig versuchen und verkleinern bis es passt
        tw, th, base, rect_h, rect_w = text_size(text, fs, thickness)
        while (rect_w > bw - 2 or rect_h > bh * 0.28) and fs > 0.5:
            fs *= 0.9
            tw, th, base, rect_h, rect_w = text_size(text, fs, thickness)

        # 2) Falls immer noch zu groß: zweizeilig umbrechen
        two_lines = False
        if (rect_w > bw - 2 or rect_h > bh * 0.28):
            parts = text.split()
            if len(parts) >= 2:
                mid = len(parts) // 2
                for cut in range(mid, 0, -1):
                    t1 = " ".join(parts[:cut])
                    t2 = " ".join(parts[cut:])
                    tw1, th1, b1, r1h, r1w = text_size(t1, fs, thickness)
                    tw2, th2, b2, r2h, r2w = text_size(t2, fs, thickness)
                    rect_w2 = max(r1w, r2w)
                    rect_h2 = r1h + r2h
                    if rect_w2 <= bw - 2 and rect_h2 <= bh * 0.36:
                        text1, text2 = t1, t2
                        two_lines = True
                        rect_w, rect_h = rect_w2, rect_h2
                        break

        # 3) Wenn selbst zweizeilig innen zu dominant -> oberhalb platzieren
        place_outside = rect_h > bh * 0.36
        if place_outside:
            y_top = y1 - rect_h - 2
            if y_top < 0:
                while rect_h > bh * 0.36 and fs > 0.45:
                    fs *= 0.9
                    if two_lines:
                        _, _, _, r1h, r1w = text_size(text1, fs, thickness)
                        _, _, _, r2h, r2w = text_size(text2, fs, thickness)
                        rect_w = max(r1w, r2w); rect_h = r1h + r2h
                    else:
                        _, _, _, rect_h, rect_w = text_size(text, fs, thickness)
                y_top = y1 - rect_h - 2
                if y_top < 0:
                    place_outside = False

        x_left = x1
        if x_left + rect_w > x2:
            x_left = x2 - rect_w
        x_left = max(0, x_left)

        if place_outside:
            y_bottom = max(0, y1 - 2)
            y_top = max(0, y_bottom - rect_h)
        else:
            y_top = y1
            y_bottom = y1 + rect_h
            if y_bottom > y2:
                y_bottom = y2
                y_top = y_bottom - rect_h

        # Zeichnen
        if two_lines:
            _, _, _, r1h, r1w = text_size(text1, fs, thickness)
            _, _, _, r2h, r2w = text_size(text2, fs, thickness)
            rect_w = max(r1w, r2w)
            cv2.rectangle(out, (int(x_left), int(y_top)),
                          (int(x_left + rect_w), int(y_top + r1h + r2h)), col, -1)
            tx = int(x_left + pad)
            ty1 = int(y_top + r1h - pad)
            ty2 = int(y_top + r1h + r2h - pad)
            cv2.putText(out, text1, (tx, ty1), font, fs, white, thickness, cv2.LINE_AA)
            cv2.putText(out, text2, (tx, ty2), font, fs, white, thickness, cv2.LINE_AA)
        else:
            cv2.rectangle(out, (int(x_left), int(y_top)),
                          (int(x_left + rect_w), int(y_top + rect_h)), col, -1)
            tx = int(x_left + pad)
            ty = int(y_top + rect_h - pad)
            cv2.putText(out, text, (tx, ty), font, fs, white, thickness, cv2.LINE_AA)

    return cv2.cvtColor(out, cv2.COLOR_BGR2RGB)




def load_gt_for_image(img_path: Path):
    """
    Lädt YOLO-Labeldatei (falls vorhanden) und gibt Pixel-Boxen + KLASSENNAMEN zurück.
    Erwartet Labelpfad: LABELS_DIR / <stem>.txt
    """
    if not LABELS_DIR or not LABELS_DIR.exists():
        return None, None

    lbl_path = LABELS_DIR / (img_path.stem + ".txt")
    if not lbl_path.exists():
        return None, None

    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        return None, None
    h, w = img_bgr.shape[:2]

    boxes = []
    class_names = []
    try:
        with open(lbl_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                cls_id = int(float(parts[0]))
                cx, cy, bw, bh = map(float, parts[1:5])
                x1, y1, x2, y2 = yolo_to_xyxy((cx, cy, bw, bh), w, h)
                boxes.append((x1, y1, x2, y2))
                if 0 <= cls_id < len(NAMES):
                    class_names.append(NAMES[cls_id])
                else:
                    class_names.append(str(cls_id))
        return boxes, class_names
    except Exception as e:
        print(f"[Warn] Konnte Labels nicht laden: {lbl_path} -> {e}")
        return None, None

def collect_images(img_dir: Path):
    return [p for p in sorted(img_dir.rglob("*")) if p.suffix.lower() in IMG_EXTS]

# ====== IoU / F1 Hilfsfunktionen ======
def iou_xyxy(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2]-a[0]) * max(0.0, a[3]-a[1])
    area_b = max(0.0, b[2]-b[0]) * max(0.0, b[3]-b[1])
    union = area_a + area_b - inter + 1e-9
    return inter / union

def f1_for_image(pred_boxes, pred_cls, gt_boxes, gt_cls, iou_thr=IOU_THR):
    """
    Greedy One-to-One Matching per Klasse; gibt F1, TP, FP, FN zurück.
    """
    if len(gt_boxes) == 0 and len(pred_boxes) == 0:
        return 1.0, 0, 0, 0
    if len(pred_boxes) == 0 and len(gt_boxes) > 0:
        return 0.0, 0, 0, len(gt_boxes)
    if len(pred_boxes) > 0 and len(gt_boxes) == 0:
        return 0.0, 0, len(pred_boxes), 0

    gt_unused = {c: set(np.where(gt_cls == c)[0]) for c in np.unique(gt_cls)}
    tp = 0; fp = 0
    for i, pbox in enumerate(pred_boxes):
        c = int(pred_cls[i])
        candidates = list(gt_unused.get(c, []))
        best_j = -1; best_iou = 0.0
        for j in candidates:
            iou = iou_xyxy(pbox, gt_boxes[j])
            if iou >= iou_thr and iou > best_iou:
                best_iou = iou; best_j = j
        if best_j >= 0:
            tp += 1
            gt_unused[c].remove(best_j)
        else:
            fp += 1
    fn = sum(len(s) for s in gt_unused.values())
    precision = tp / (tp + fp + 1e-9)
    recall    = tp / (tp + fn + 1e-9)
    f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
    return float(f1), tp, fp, fn

# =========================
# 2) Main Review-Loop
# =========================

def review_folder():
    assert IMG_DIR.exists(), f"IMG_DIR existiert nicht: {IMG_DIR}"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    if SINGLE_IMAGE and SINGLE_IMAGE.exists():
        image_paths = [SINGLE_IMAGE]
    else:
        image_paths = collect_images(IMG_DIR)
    if not image_paths:
        print("Keine Bilder gefunden."); return
    ...


    # Modelle initialisieren (+ zugehöriger TAG & Caption)
    model_objs = []  # (model, conf, tag, caption)
    for (w_c_idx, (w, c)) in enumerate(MODELS):
        if not w.exists():
            raise FileNotFoundError(f"Gewicht nicht gefunden: {w}")
        m = YOLO(str(w))
        m.model.eval()
        tag = TAGS[w_c_idx]
        caption = CAPTIONS[w_c_idx + 1]  # +1 weil CAPTIONS[0] GT ist
        model_objs.append((m, c, tag, caption))

    # Zähler für Dateinamen
    counters = {tag: 0 for tag in TAGS}
    counters["even"] = 0
 
    print(f"Gefundene Bilder: {len(image_paths)}")
    print("Alle Abbildungen werden automatisch bewertet und gespeichert.")

    for idx, img_path in enumerate(image_paths, 1):
        # Originalbild laden
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"[Warn] Konnte Bild nicht laden: {img_path}")
            continue
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # GT laden/zeichnen
        gt_boxes, gt_names = load_gt_for_image(img_path)


        # Klassen-IDs aus den GT-Namen (unverändert wie bei dir)
        name_to_id = {n: i for i, n in enumerate(NAMES)}
        gt_cls_ids = np.array([name_to_id.get(n, -1) for n in (gt_names or [])], dtype=int)
        gt_boxes_arr = np.array(gt_boxes, dtype=float) if gt_boxes else np.zeros((0, 4), dtype=float)

        if gt_boxes:
            gt_img = draw_boxes_on_image_adaptive(
                img_rgb,
                gt_boxes,
                class_ids=gt_cls_ids,
                labels=gt_names,
                box_thickness=4
            )
        else:
            gt_img = img_rgb.copy()



        # Klassen → IDs
        name_to_id = {n: i for i, n in enumerate(NAMES)}
        gt_cls_ids = np.array([name_to_id.get(n, -1) for n in (gt_names or [])], dtype=int)
        gt_boxes_arr = np.array(gt_boxes, dtype=float) if gt_boxes else np.zeros((0, 4), dtype=float)

        # Predictions + Panels + Scores
        panels = [gt_img]
        captions = ["a) " + CAPTIONS[0]]
        model_scores = {}  # tag -> f1

        for (model, conf_thr, tag, caption) in model_objs:
            results = model.predict(
                source=str(img_path),
                imgsz=IMGSZ,
                save=False, show=False, verbose=False,
                agnostic_nms=True, conf=conf_thr
            )

            # Panel-Bild
            pred_bgr = results[0].plot()
            pred_rgb = cv2.cvtColor(pred_bgr, cv2.COLOR_BGR2RGB)
            panels.append(pred_rgb)
            # a)/b)/c) beschriften
            label_prefix = chr(ord('a') + len(captions)) + ") "
            captions.append(label_prefix + caption)

            # Boxes/Klassen für Auswertung
            b = results[0].boxes
            if b is None or b.xyxy is None or len(b) == 0:
                pred_boxes = np.zeros((0, 4), dtype=float)
                pred_cls = np.array([], dtype=int)
            else:
                pred_boxes = b.xyxy.cpu().numpy().astype(float)
                pred_cls = b.cls.cpu().numpy().astype(int)

            f1, tp, fp, fn = f1_for_image(pred_boxes, pred_cls, gt_boxes_arr, gt_cls_ids, IOU_THR)
            model_scores[tag] = f1

            # --- Bestes Modell bestimmen (BiFormer/YOLOv8x explizit behandeln) ---
            if model_scores:
                bif = model_scores.get("biformer", -np.inf)
                y8x = model_scores.get("yolov8x", -np.inf)

                if np.isfinite(bif) and np.isfinite(y8x):
                    if bif > y8x + PRIORITY_ATOL:
                        best_tag = "biformer"
                    elif y8x > bif + PRIORITY_ATOL:
                        best_tag = "yolov8x"   # <-- jetzt wird YOLOv8x klar als besser gezählt
                    else:
                        # fast gleich -> generische Tie-Logik über alle Modelle
                        max_score = max(model_scores.values())
                        winners = [t for t, s in model_scores.items()
                                if abs(s - max_score) <= EVEN_TIE_ATOL]
                        best_tag = "even" if len(winners) >= 2 else winners[0]
                else:
                    # Falls eines der beiden fehlt, normale Max-Logik
                    max_score = max(model_scores.values())
                    winners = [t for t, s in model_scores.items()
                            if abs(s - max_score) <= EVEN_TIE_ATOL]
                    best_tag = "even" if len(winners) >= 2 else winners[0]
            else:
                best_tag = "even"



        # Dateiname festlegen
        counters[best_tag] += 1
        if best_tag == "even":
            out_stem = f"even{counters['even']}"
        else:
            out_stem = f"{best_tag}_better{counters[best_tag]}"

        # ===== Layout wählen: 2x3 für gute Lesbarkeit =====
        n_panels = len(panels)
        ncols = 3
        nrows = int(np.ceil(n_panels / ncols))

        # Layout-Setup (ersetzen)
        fig_w, fig_h = 16, 10
        fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), constrained_layout=False)
        axes = np.atleast_2d(axes).reshape(nrows, ncols)

        # enger zusammenrücken: kleinere Außenränder + minimaler Abstand zwischen Subplots
        fig.subplots_adjust(left=0.015, right=0.985, top=0.985, bottom=0.06, wspace=0.01, hspace=0.08)

        font_size = 16
        for i, (img, cap) in enumerate(zip(panels, captions)):
            r, c = divmod(i, ncols)
            ax = axes[r, c]
            ax.imshow(img); ax.axis("off")
            ax.text(
                0.5, -0.02, cap,            # näher ans Bild (vorher -0.03)
                transform=ax.transAxes,
                ha="center", va="top",
                fontsize=font_size, fontweight="bold"
            )


        # Ungenutzte Achsen ausblenden
        for j in range(n_panels, nrows * ncols):
            r, c = divmod(j, ncols)
            axes[r, c].axis("off")

        # Speichern (PNG + PDF)
        png_path = OUT_DIR / f"{out_stem}.png"
        fig.savefig(str(png_path), dpi=300, bbox_inches="tight")
        plt.close(fig)

        # Konsolen-Log
        score_str = ", ".join([f"{k}:{v:.3f}" for k, v in model_scores.items()])
        print(f"{img_path.name} → {out_stem}  [{score_str}]")

if __name__ == "__main__":
    review_folder()
