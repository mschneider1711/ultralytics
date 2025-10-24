from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import yaml

# =========================
# Konfiguration
# =========================
# Beliebig viele Runs eintragen (sprechende Namen → Pfade)
DATASET_PATHS = {
    "run1": Path("/Users/marcschneider/Documents/MasterArbeit_Experimente/NEW/PlantDoc-3/splits/split0"),
    "run2": Path("/Users/marcschneider/Documents/MasterArbeit_Experimente/NEW/PlantDoc-3/splits/split1"),
    "run3": Path("/Users/marcschneider/Documents/MasterArbeit_Experimente/NEW/PlantDoc-3/splits/split2"),
}

# Gewünschter Split: "train", "val" oder "test"
SELECTED_SPLIT = "val"

# Wo der Plot gespeichert wird
OUTPUT_DIR = Path("./plots")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Schriftgrößen (wie in deinem Original)
TITLE_SIZE = 18
AXIS_LABEL_SIZE = 18
TICK_SIZE = 18
ANNOT_SIZE = 14


# =========================
# Hilfsfunktionen
# =========================
def load_class_names(paths: dict[str, Path]) -> list[str]:
    """Lade class names aus dem ersten gefundenen data.yaml und prüfe Konsistenz."""
    first_names = None
    first_yaml = None
    for run_name, base_path in paths.items():
        yaml_path = base_path / "data.yaml"
        if yaml_path.exists():
            with open(yaml_path, "r") as f:
                data = yaml.safe_load(f)
                names = data.get("names", [])
            if first_names is None:
                first_names = names
                first_yaml = yaml_path
            else:
                if names != first_names:
                    print(f"[Warnung] Klasseninkonsistenz zwischen {first_yaml} und {yaml_path}. "
                          f"Der Plot verwendet die Klassen aus {first_yaml}.")
            print(f"Loaded classes from {yaml_path}")
    if first_names is None:
        raise FileNotFoundError("Keine data.yaml in den angegebenen Runs gefunden!")
    return first_names


def count_instances_for_split(base_path: Path, split: str, num_classes: int) -> list[int]:
    """Zähle Instanzen (Zeilen in *.txt-Labeldateien) je Klasse für einen Split."""
    label_dir = base_path / split / "labels"
    class_counts = Counter()
    if not label_dir.is_dir():
        print(f"[Info] Kein Labels-Ordner gefunden: {label_dir}")
        return [0] * num_classes

    for label_file in label_dir.glob("*.txt"):
        with open(label_file, "r") as f:
            for line in f:
                s = line.strip()
                if not s:
                    continue
                parts = s.split()
                try:
                    class_id = int(parts[0])
                except Exception:
                    continue
                if 0 <= class_id < num_classes:
                    class_counts[class_id] += 1

    return [class_counts[i] for i in range(num_classes)]


# =========================
# Klassen laden
# =========================
class_names = load_class_names(DATASET_PATHS)
num_classes = len(class_names)
x = np.arange(num_classes)

# =========================
# Zählen pro Run (ausgewählter Split)
# =========================
runs = list(DATASET_PATHS.keys())
counts_per_run = {}
totals = {}

for run_name, base_path in DATASET_PATHS.items():
    counts = count_instances_for_split(base_path, SELECTED_SPLIT, num_classes)
    counts_per_run[run_name] = counts
    totals[run_name] = sum(counts)
    print(f"Total instances in {run_name} – {SELECTED_SPLIT}: {totals[run_name]}")

# =========================
# Plot (alle Runs in einem Diagramm)
# =========================
n_runs = len(runs)
width = 0.9 / max(n_runs, 1)  # Gruppierte Balken, 90% der Breite
plt.figure(figsize=(21, 9))

for idx, run_name in enumerate(runs):
    counts = counts_per_run[run_name]
    offset = (idx - (n_runs - 1) / 2) * width
    bars = plt.bar(x + offset, counts, width=width, label=run_name)

    # Werte in die Balken schreiben
    for rect, count in zip(bars, counts):
        if count > 0:
            plt.text(
                rect.get_x() + rect.get_width() / 2,
                rect.get_height() / 2,
                str(count),
                ha="center",
                va="center",
                fontsize=ANNOT_SIZE,
                rotation=90,
                color="white",
            )

plt.xticks(x, class_names, rotation=90, fontsize=TICK_SIZE)
plt.xlabel("Classes", fontsize=AXIS_LABEL_SIZE)
plt.ylabel("Number of instances", fontsize=AXIS_LABEL_SIZE)
#plt.title(f"Class distribution – {SELECTED_SPLIT}", fontsize=TITLE_SIZE)
plt.yticks(fontsize=TICK_SIZE)
plt.grid(True, axis="y", linestyle="--", alpha=0.3)
plt.legend(title="Runs", frameon=False, fontsize=AXIS_LABEL_SIZE, title_fontsize=AXIS_LABEL_SIZE)

out_path = OUTPUT_DIR / f"class_distribution_{SELECTED_SPLIT}_all_runs.png"
plt.tight_layout()
plt.savefig(out_path, dpi=200, bbox_inches="tight")
plt.show()
print(f"Saved: {out_path}")
