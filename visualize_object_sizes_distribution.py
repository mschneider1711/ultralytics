from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

# === Pfade definieren ===
dataset_paths = {
    "run1": Path("/Users/marcschneider/Documents/PlantDoc.v4i.yolov8"),
    "run2": Path("/Users/marcschneider/Documents/PlantDoc.v4i.yolov8_run2"),
    "run3": Path("/Users/marcschneider/Documents/PlantDoc.v4i.yolov8_run3"),
}
splits = ["valid"]

# === Objektgrößenkategorien definieren (rel. Fläche)
def size_category(area):
    if area < 0.01:
        return "small"
    elif area < 0.1:
        return "medium"
    else:
        return "large"

# === Objektgrößen zählen pro Split und Run
size_counts_per_run = {}

for run_name, dataset_path in dataset_paths.items():
    split_counts = {split: {"small": 0, "medium": 0, "large": 0} for split in splits}
    
    for split in splits:
        label_dir = dataset_path / split / "labels"
        for label_file in label_dir.glob("*.txt"):
            with open(label_file, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        _, x_center, y_center, width, height = map(float, parts[:5])
                        area = width * height
                        category = size_category(area)
                        split_counts[split][category] += 1
    
    size_counts_per_run[run_name] = split_counts

# === Plotten: Balkendiagramm nach Größe pro Split
categories = ["small", "medium", "large"]
x = np.arange(len(splits))
width = 0.25

plt.figure(figsize=(10, 6))

for idx, (run_name, counts_per_split) in enumerate(size_counts_per_run.items()):
    sizes_per_cat = {cat: [counts_per_split[split][cat] for split in splits] for cat in categories}
    
    for i, cat in enumerate(categories):
        offset = (i - 1) * width  # small: -1, medium: 0, large: +1
        plt.bar(x + offset + idx * (width * 3), sizes_per_cat[cat], width, label=f"{run_name} - {cat}")

plt.xticks(x + width, splits)
plt.xlabel("Datensplits")
plt.ylabel("Anzahl Objekte")
plt.title("📏 Objektgrößenverteilung pro Split")
plt.legend()
plt.tight_layout()
plt.show()
