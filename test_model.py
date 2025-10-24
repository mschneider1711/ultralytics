# ===== Validate & Save Predictions (Ultralytics YOLOv8) =====
import os
from datetime import datetime
from ultralytics import YOLO
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from torch import nn
import torch



# ---- Pfade/Parameter anpassen ----
model_path = "/Users/marcschneider/Documents/MasterArbeit_Experimente/NEW/yolo_baseline/yolov8x_pretrained_split2/weights/best.pt"
model_path = "/Users/marcschneider/Documents/MasterArbeit_Experimente/NEW/modular_experiments/yolov8s_P3C2fSTRV1_official_split2/weights/best.pt"
model_path = "/Users/marcschneider/Documents/MasterArbeit_Experimente/NEW/backbone_experiments/pvt2_yolov8_custom_split2/weights/best.pt"
config_path = "/Users/marcschneider/Documents/ultralytics_costum/ultralytics/cfg/models/v8_costum3/pvt2_yolov8.yaml"
data_yaml  = "/Users/marcschneider/Documents/MasterArbeit_Experimente/NEW/PlantDoc-3/splits/split2/data.yaml"
test_img_dir = "/Users/marcschneider/Documents/MasterArbeit_Experimente/NEW/PlantDoc-3/test/images"
imgsz      = 640

# WICHTIG: conf=None  -> beste F1 auf Validation wird gesucht und verwendet
#          conf=0.293 -> exakt diese Schwelle wird genutzt und P/R/F1 dafür berechnet
conf       = None
iou        = 0.7
split      = "test"   # "val" oder "test" 
project    = "/Users/marcschneider/Documents/MasterArbeit_Experimente/NEW/runs"
run_name   = f"{Path(model_path).stem}_{split}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"

# ---- Modell laden ----
model = YOLO(model_path)

# ---- 1) VALIDIEREN ----
print("==> Validiere Modell …")
metrics = model.val(
    data=data_yaml,
    imgsz=imgsz,
    split=split,
    save_json=True,
    save_txt=True,
    plots=True,
    project=project,
    name=f"{run_name}_test",
    exist_ok=True,
    verbose=False,
    conf=(conf if conf is not None else 0.001),  # niedrige conf, damit PR-Kurven nicht abgeschnitten werden
)

# ---- Kurven extrahieren und P/R/F1 bei gewünschter conf bestimmen ----
def _as_mean_1d(y):
    y = np.asarray(y)
    return np.nanmean(y, axis=0) if y.ndim == 2 else y

def get_curves_from_curve_results(metrics):
    cr = getattr(metrics, "curve_results", None) or getattr(metrics, "curves_results", None)
    if cr is None:
        raise RuntimeError("metrics.curve_results nicht gefunden.")
    # 0: PR (Recall -> Precision), 1: F1-Confidence, 2: Precision-Confidence, 3: Recall-Confidence
    pr_x, pr_y, *_ = cr[0]
    x_f1, y_f1, *_ = cr[1]
    x_p,  y_p,  *_ = cr[2]
    x_r,  y_r,  *_ = cr[3]
    # auf gleiche Länge bringen
    m = min(len(x_f1), len(x_p), len(x_r))
    return {
        "conf": np.asarray(x_f1[:m]),
        "f1_conf": _as_mean_1d(y_f1)[:m],
        "p_conf":  _as_mean_1d(y_p)[:m],
        "r_conf":  _as_mean_1d(y_r)[:m],
        "pr_x":    np.asarray(pr_x),
        "pr_y":    _as_mean_1d(pr_y),
    }

curves = get_curves_from_curve_results(metrics)

# --- Auswahl: gegebene conf verwenden ODER best-F1 auf Validation bestimmen ---
if conf is None:
    idx = int(np.nanargmax(curves["f1_conf"]))
    chosen_conf = float(curves["conf"][idx])
    chosen_f1   = float(curves["f1_conf"][idx])
    chosen_p    = float(curves["p_conf"][idx])
    chosen_r    = float(curves["r_conf"][idx])
    conf_source = "best F1 on validation"
else:
    # Nächstgelegenen Kurvenpunkt zur gewünschten conf nehmen
    diffs = np.abs(curves["conf"] - conf)
    idx = int(np.nanargmin(diffs))
    chosen_conf = float(curves["conf"][idx])
    chosen_f1   = float(curves["f1_conf"][idx])
    chosen_p    = float(curves["p_conf"][idx])
    chosen_r    = float(curves["r_conf"][idx])
    conf_source = f"user-specified (target={conf:.3f}, used={chosen_conf:.3f})"

print(f"[CONF] {conf_source}")
print(f"[VAL-curves @ conf={chosen_conf:.3f}] F1={chosen_f1:.3f}, P={chosen_p:.3f}, R={chosen_r:.3f}")

# ---- Plot: F1 / P / R vs. Confidence + Marker bei gewählter Schwelle ----
plt.figure()
plt.plot(curves["conf"], curves["f1_conf"], label="F1")
plt.plot(curves["conf"], curves["p_conf"],  "--", label="Precision")
plt.plot(curves["conf"], curves["r_conf"],  ":", label="Recall")
plt.scatter([chosen_conf], [chosen_f1], zorder=5)
plt.annotate(f"F1={chosen_f1:.3f}\nconf={chosen_conf:.3f}\nP={chosen_p:.3f}, R={chosen_r:.3f}",
             (chosen_conf, chosen_f1), xytext=(10,14), textcoords="offset points")
plt.xlabel("Confidence threshold"); plt.ylabel("Score")
#plt.title("F1 / Precision / Recall vs. Confidence (validation curves)")
plt.legend(); plt.grid(alpha=0.25); plt.show()

# ---- PR-Plot + mAP@0.5 Hinweis ----
plt.figure()
plt.plot(curves["pr_x"], curves["pr_y"], color="blue")
map50 = float(metrics.box.map50)
plt.text(0.6, 0.05, f"all classes {map50:.3f} mAP@0.5",
         fontsize=10, color="blue",
         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.5", alpha=0.9))
plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision–Recall Curve (validation)")
plt.grid(alpha=0.25); plt.show()

# ---- (Optional) Konfusionsmatrix wie gehabt ----
cm = metrics.confusion_matrix.matrix
names = metrics.confusion_matrix.names + ["background"]
cm_norm = cm.astype(np.float32) / (cm.sum(axis=0, keepdims=True) + 1e-12)

def plot_confusion_matrix(cm, class_names, title="Confusion Matrix", normalize=False, figsize=(14,12)):
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set(xticks=np.arange(len(class_names)), yticks=np.arange(len(class_names)),
           xticklabels=class_names, yticklabels=class_names,
           xlabel="True label", ylabel="Predicted label", title=title)
    plt.setp(ax.get_xticklabels(), rotation=60, ha="right", rotation_mode="anchor")
    thresh = cm.max() / 2. if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            val = cm[i, j]
            txt = f"{val:.2f}" if normalize else (f"{int(val)}" if val > 0 else "")
            ax.text(j, i, txt, ha="center", va="center",
                    color="white" if val > thresh else "black", fontsize=6)
    plt.tight_layout(); plt.show()

plot_confusion_matrix(cm, names, title="Confusion Matrix (counts)", normalize=False)
plot_confusion_matrix(cm_norm, names, title="Confusion Matrix (normalized)", normalize=True)

# # ---- 2) PREDICTIONS ALS BILDER SPEICHERN ----
# print("\n==> Speichere visualisierte Predictions als Bilder …")
# pred_run = model.predict(
#     source=test_img_dir,
#     imgsz=imgsz,
#     save=True,
#     save_conf=True,
#     project=project,
#     name=f"{run_name}_pred_images",
#     exist_ok=True,
#     max_det=300,
#     verbose=False,
#     conf=chosen_conf,   # exakt den gewählten Threshold verwenden
#     iou=iou
# )

# out_dir = os.path.join(project, f"{run_name}_pred_images")
# print(f"\nFertig ✅  Gerenderte Bilder gespeichert unter:\n{out_dir}")
