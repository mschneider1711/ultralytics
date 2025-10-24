import json
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# 📂 Deine Pfade
gt_json = "/Users/marcschneider/Documents/PlantDoc.v4i.coco/test/_annotations.coco.json"
pred_json_1 = "/Users/marcschneider/Documents/ultralytics_costum/runs/detect/val80/predictions.json"
pred_json_2 = "/Users/marcschneider/Documents/ultralytics_costum/runs/detect/val79/predictions.json"
fixed_json_1 = "/Users/marcschneider/Documents/ultralytics_costum/runs/detect/val80/predictions_fixed.json"
fixed_json_2 = "/Users/marcschneider/Documents/ultralytics_costum/runs/detect/val79/predictions_fixed.json"

# 🚀 Lade GT
cocoGt = COCO(gt_json)

# Mapping: filename (ohne .jpg/.jpeg) → image_id
filename_to_id = {}
for img in cocoGt.loadImgs(cocoGt.getImgIds()):
    base_filename = img["file_name"].split(".")[0]
    filename_to_id[base_filename] = img["id"]

def fix_predictions(pred_json_path, fixed_json_path):
    """Ersetzt image_id-Strings durch numerische IDs und speichert als fixed JSON"""
    with open(pred_json_path, "r") as f:
        preds = json.load(f)
    
    # Ersetze image_id
    for pred in preds:
        base_filename = pred["image_id"].split(".")[0]  # Entferne .jpg.rf...
        if base_filename in filename_to_id:
            pred["image_id"] = filename_to_id[base_filename]
        else:
            print(f"⚠️ WARNUNG: {base_filename} nicht in GT gefunden!")

    # Speichern
    with open(fixed_json_path, "w") as f:
        json.dump(preds, f, indent=2)
    print(f"✅ Repariertes JSON gespeichert: {fixed_json_path}")

# 🔄 Beide JSONs fixen
fix_predictions(pred_json_1, fixed_json_1)
fix_predictions(pred_json_2, fixed_json_2)

# 📝 Funktion für COCOEval
def evaluate_model(pred_json_fixed, model_name):
    cocoDt = cocoGt.loadRes(pred_json_fixed)
    cocoEval = COCOeval(cocoGt, cocoDt, iouType="bbox")
    cocoEval.evaluate()
    cocoEval.accumulate()
    print(f"\n📊 Ergebnisse für {model_name}:")
    cocoEval.summarize()
    return cocoEval

# ✅ Modelle evaluieren
eval1 = evaluate_model(fixed_json_1, "Modell 1")
eval2 = evaluate_model(fixed_json_2, "Modell 2")

# 🏷️ Klassen-Namen
cats = cocoGt.loadCats(cocoGt.getCatIds())
cat_names = [c['name'] for c in cats]
cat_ids = cocoGt.getCatIds()

# 📊 AP pro Klasse extrahieren
def get_ap_per_class(cocoEval):
    precisions = cocoEval.eval['precision']  # [TxRxKxAxM]
    ap_per_class = np.zeros(len(cat_names))
    for idx in range(len(cat_names)):
        # Mittelwert über IoUs, recalls, area=all, maxDets
        precision = precisions[:, :, idx, 0, -1]
        precision = precision[precision > -1]
        if precision.size == 0:
            ap = float('nan')
        else:
            ap = np.mean(precision)
        ap_per_class[idx] = ap
    return ap_per_class

ap1 = get_ap_per_class(eval1)
ap2 = get_ap_per_class(eval2)

# 📏 Dominante Objektgröße pro Klasse
def get_dominant_size_per_class():
    dominant_sizes = []
    for cat_id in cat_ids:
        anns = cocoGt.loadAnns(cocoGt.getAnnIds(catIds=cat_id))
        small, medium, large = 0, 0, 0
        for ann in anns:
            w, h = ann['bbox'][2], ann['bbox'][3]
            area = w * h
            if area < 32**2:
                small += 1
            elif area < 96**2:
                medium += 1
            else:
                large += 1
        if max(small, medium, large) == small:
            dominant_sizes.append("🔵 small")
        elif max(small, medium, large) == medium:
            dominant_sizes.append("🟢 medium")
        else:
            dominant_sizes.append("🟣 large")
    return dominant_sizes

dominant_sizes = get_dominant_size_per_class()

# 📋 Vergleichstabelle
print("\n📊 Vergleich der mAP pro Klasse mit dominanter Objektgröße:")
print(f"{'Klasse':25s} | {'Modell 1':>8s} | {'Modell 2':>8s} | {'Dominant':>10s} | Besser")
print("-" * 75)
for name, score1, score2, size in zip(cat_names, ap1, ap2, dominant_sizes):
    winner = "🟢 M1" if score1 > score2 else "🟣 M2"
    print(f"{name:25s} | {score1:8.3f} | {score2:8.3f} | {size:>10s} | {winner}")

