from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import yaml

# === Baseline-Dataset (nur ein Split mit train & test) ===
baseline_path = Path("/Users/marcschneider/Documents/MasterArbeit_Experimente/NEW/PlantDoc-3")

# Splits, die ausgewertet werden
splits = ["train", "test"]

# === Klassen aus data.yaml laden ===
yaml_path = baseline_path / "data.yaml"
if not yaml_path.exists():
    raise FileNotFoundError(f"data.yaml not found at: {yaml_path}")

with open(yaml_path, "r") as f:
    data = yaml.safe_load(f)
class_names = data.get("names", [])
if not class_names:
    raise ValueError("No 'names' found in data.yaml")

num_classes = len(class_names)
x = np.arange(num_classes) * 0.2

# Output dir (Basisordner)
output_dir = baseline_path
output_dir.mkdir(parents=True, exist_ok=True)

# === Zählen: Bilder pro Klasse, über alle Splits summiert ===
total_class_counts = Counter()

for split in splits:
    label_dir = baseline_path / split / "labels"
    if label_dir.is_dir():
        for label_file in label_dir.glob("*.txt"):
            image_classes = set()
            with open(label_file, "r") as f:
                for line in f:
                    if not line.strip():
                        continue
                    parts = line.split()
                    try:
                        class_id = int(parts[0])
                    except Exception:
                        continue
                    if 0 <= class_id < num_classes:
                        image_classes.add(class_id)
            for cid in image_classes:
                total_class_counts[cid] += 1
    else:
        print(f"[Info] No labels folder found: {label_dir}")

# In Listenform bringen
counts_total = [total_class_counts[i] for i in range(num_classes)]
global_max = max(counts_total)

print(f"Total images counted (train+test): {sum(counts_total)}")

# === Plot-Settings ===
TITLE_SIZE = 18
AXIS_LABEL_SIZE = 18
TICK_SIZE = 14
ANNOT_SIZE = 12
bar_color = "tab:blue"
bar_width = 0.1

# === Ein Gesamt-Plot ===
plt.figure(figsize=(16, 9))
plt.bar(x, counts_total, width=bar_width, color=bar_color)

# Headroom für Labels
headroom = max(2, 0.08 * global_max)
plt.ylim(0, global_max + headroom)

# Werte über den Balken
for i, c in enumerate(counts_total):
    if c > 0:
        plt.text(
            x[i], c + 0.02 * global_max + 0.5,
            str(c), ha='center', va='bottom',
            fontsize=ANNOT_SIZE, rotation=90
        )

plt.xticks(x, class_names, rotation=90, fontsize=TICK_SIZE)
plt.xlabel("Classes", fontsize=AXIS_LABEL_SIZE)
plt.ylabel("Number of images", fontsize=AXIS_LABEL_SIZE)
plt.yticks(fontsize=TICK_SIZE)
plt.grid(True, axis='y', linestyle='--', alpha=0.3)

out_path = output_dir / "class_distribution_total.png"
plt.tight_layout()
plt.savefig(out_path, dpi=200, bbox_inches="tight")
plt.show()
print(f"Saved: {out_path}")
