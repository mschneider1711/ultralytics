from pathlib import Path
import random
import yaml
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

# ====== CONFIG ======
DATASET_PATH = Path("/Users/marcschneider/Documents/MasterArbeit_Experimente/NEW/PlantDoc-3")
SPLIT = "train"   # "train" oder "test"
N_SAMPLES = 4
OUTPUT_NAME = f"sample_preview_{SPLIT}.png"

# Schriftgrößen
TITLE_SIZE = 18
AXIS_LABEL_SIZE = 18
TICK_SIZE = 18
ANNOT_SIZE = 14

# ====== Klassen laden ======
yaml_path = DATASET_PATH / "data.yaml"
if not yaml_path.exists():
    raise FileNotFoundError(f"data.yaml not found at {yaml_path}")

with open(yaml_path, "r") as f:
    data = yaml.safe_load(f)
class_names = data.get("names", [])
if not class_names:
    raise ValueError("No class names found in data.yaml under key 'names'.")

# ====== Bild- und Labelpfade sammeln ======
img_dir = DATASET_PATH / SPLIT / "images"
lbl_dir = DATASET_PATH / SPLIT / "labels"
if not img_dir.is_dir() or not lbl_dir.is_dir():
    raise FileNotFoundError(f"Missing images or labels folder for split '{SPLIT}'.")

def matching_label_path(img_path: Path) -> Path:
    return lbl_dir / (img_path.stem + ".txt")

image_paths = []
for p in sorted(img_dir.iterdir()):
    if p.suffix.lower() in {".jpg", ".jpeg", ".png"} and matching_label_path(p).exists():
        image_paths.append(p)

if not image_paths:
    raise RuntimeError(f"No images with labels found in {img_dir}")

rand_imgs = random.sample(image_paths, k=min(N_SAMPLES, len(image_paths)))

# ====== Hilfsfunktionen ======
def yolo_to_xyxy(box, w, h):
    xc, yc, bw, bh = box
    xc *= w; yc *= h; bw *= w; bh *= h
    x1 = xc - bw / 2
    y1 = yc - bh / 2
    x2 = xc + bw / 2
    y2 = yc + bh / 2
    return x1, y1, x2, y2

def read_labels(label_path: Path):
    anns = []
    with open(label_path, "r") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            parts = s.split()
            try:
                cls = int(parts[0])
                xc, yc, bw, bh = map(float, parts[1:5])
                anns.append((cls, (xc, yc, bw, bh)))
            except Exception:
                continue
    return anns

# ====== Plot vorbereiten ======
n = len(rand_imgs)
letters = ["(a)", "(b)", "(c)", "(d)"]

fig, axes = plt.subplots(1, n, figsize=(4.0*n, 4.5), squeeze=False)
axes = axes[0]

for idx, img_path in enumerate(rand_imgs):
    ax = axes[idx]
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    ax.imshow(img)
    ax.set_axis_off()

    lbl_path = matching_label_path(img_path)
    anns = read_labels(lbl_path)

    for cls, box in anns:
        x1, y1, x2, y2 = yolo_to_xyxy(box, w, h)
        width = max(0.0, x2 - x1)
        height = max(0.0, y2 - y1)

        rect = Rectangle((x1, y1), width, height,
                         linewidth=2, edgecolor='tab:blue', facecolor='none')
        ax.add_patch(rect)

        label_txt = class_names[cls] if 0 <= cls < len(class_names) else f"class {cls}"
        text_y = y1 - 4 if y1 > 10 else y1 + 14

        ax.text(x1, text_y, label_txt,
                fontsize=ANNOT_SIZE, color='white',
                bbox=dict(facecolor='black', alpha=0.6, pad=2, edgecolor='none'))

    # Beschriftung unter dem Bild
    ax.text(0.5, -0.08, letters[idx], transform=ax.transAxes,
            ha='center', va='top', fontsize=AXIS_LABEL_SIZE)

# Weniger Abstand zwischen den Subplots
plt.subplots_adjust(wspace=0.05, hspace=0.05)

out_path = (DATASET_PATH.parent / OUTPUT_NAME)
plt.savefig(out_path, dpi=200, bbox_inches="tight")
plt.show()
print(f"Saved: {out_path}")
