from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np

# === Alle Datasets definieren ===
dataset_paths = {
    "run1": Path("/Users/marcschneider/Documents/PlantDoc.v4i.yolov8"),
    "run2": Path("/Users/marcschneider/Documents/PlantDoc.v4i.yolov8_run2"),
    "run3": Path("/Users/marcschneider/Documents/PlantDoc.v4i.yolov8_run3"),
}
splits = ["train"]  # oder auch ["train", "valid", "test"]

# === Klassenlabels (für X-Achse)
class_names = [
    'Apple Scab Leaf', 'Apple leaf', 'Apple rust leaf', 'Bell_pepper leaf spot', 'Bell_pepper leaf',
    'Blueberry leaf', 'Cherry leaf', 'Corn Gray leaf spot', 'Corn leaf blight', 'Corn rust leaf',
    'Peach leaf', 'Potato leaf early blight', 'Potato leaf late blight', 'Potato leaf',
    'Raspberry leaf', 'Soyabean leaf', 'Soybean leaf', 'Squash Powdery mildew leaf', 'Strawberry leaf',
    'Tomato Early blight leaf', 'Tomato Septoria leaf spot', 'Tomato leaf bacterial spot',
    'Tomato leaf late blight', 'Tomato leaf mosaic virus', 'Tomato leaf yellow virus', 'Tomato leaf',
    'Tomato mold leaf', 'Tomato two spotted spider mites leaf', 'grape leaf black rot', 'grape leaf'
]

num_classes = len(class_names)
x = np.arange(num_classes)
width = 0.25  # Breite der Balken

# === Klassen zählen pro Dataset
counts_per_run = {}

for run_name, dataset_path in dataset_paths.items():
    class_counts = Counter()
    for split in splits:
        label_dir = dataset_path / split / "labels"
        for label_file in label_dir.glob("*.txt"):
            with open(label_file, "r") as f:
                for line in f:
                    if line.strip():
                        class_id = int(line.strip().split()[0])
                        class_counts[class_id] += 1
    counts = [class_counts[i] for i in range(num_classes)]
    total = sum(counts)
    print(f"🔢 Gesamtanzahl Instanzen in {run_name} ({', '.join(splits)}): {total}")
    counts_per_run[run_name] = counts

# === Plot
plt.figure(figsize=(18, 6))
for idx, (run_name, counts) in enumerate(counts_per_run.items()):
    offset = idx * width - width  # zentrieren
    plt.bar(x + offset, counts, width=width, label=run_name)

# Zahlen optional über Balken schreiben
for idx, (run_name, counts) in enumerate(counts_per_run.items()):
    offset = idx * width - width
    for i, count in enumerate(counts):
        if count > 0:
            plt.text(x[i] + offset, count + 1, str(count), ha='center', va='bottom', fontsize=7, rotation=90)

plt.xticks(x, class_names, rotation=90)
plt.xlabel("Klassen")
plt.ylabel("Anzahl Instanzen")
plt.title("📊 Klassenverteilung – Vergleich run1, run2, run3")
plt.legend()
plt.tight_layout()
plt.show()
