import os
import shutil
import random
import time
import glob
import torch
from ultralytics import YOLO
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import StratifiedShuffleSplit
from collections import Counter

# Basispfad zum ursprünglichen Datensatz
BASE_DIR = Path("/Users/marcschneider/Documents/PlantDoc.v4i.yolov8")
DATA_YAML_NAME = "data.yaml"

def shuffle_dataset(seed: int, run_id: int):
    new_dir = BASE_DIR.parent / f"{BASE_DIR.name}_run{run_id}"

    if new_dir.exists():
        print(f"Überspringe Shuffle: Ordner {new_dir} existiert bereits.")
        return new_dir

    # Neue Ordnerstruktur erstellen (nur für train und valid)
    for split in ['train', 'valid']:
        (new_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (new_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    # TEST-Set unverändert übernehmen
    shutil.copytree(BASE_DIR / "test", new_dir / "test", dirs_exist_ok=True)

    # Alle Labels aus original train+valid einlesen
    label_paths = []
    for subset in ['train', 'valid']:
        label_dir = BASE_DIR / subset / "labels"
        label_paths.extend(label_dir.glob("*.txt"))

    image_label_pairs = []
    labels_flat = []
    for lbl_path in label_paths:
        img_path = lbl_path.parent.parent / "images" / (lbl_path.stem + ".jpg")
        if not img_path.exists():
            img_path = lbl_path.parent.parent / "images" / (lbl_path.stem + ".png")
        if not img_path.exists():
            continue
        with open(lbl_path, "r") as f:
            lines = f.readlines()
            if lines:
                class_id = int(lines[0].split()[0])  # erste Klasse im Label
                labels_flat.append(class_id)
                image_label_pairs.append((img_path, lbl_path))

    # Klassen zählen
    class_counts = Counter(labels_flat)
    print(f"Anzahl Klassen: {len(class_counts)}")
    print(f"Klassenverteilung: {class_counts}")

    # --- Seltene Klassen (<2 Instanzen) in guaranteed_train packen
    guaranteed_train = []
    remaining_pairs = []
    remaining_labels = []

    for (pair, label) in zip(image_label_pairs, labels_flat):
        if class_counts[label] < 2:
            guaranteed_train.append((pair, label))
        else:
            remaining_pairs.append(pair)
            remaining_labels.append(label)

    if len(remaining_pairs) == 0:
        raise ValueError("❌ Keine ausreichenden Daten für stratified Split vorhanden.")

    # Stratified split in train/valid
    sss = StratifiedShuffleSplit(n_splits=1, train_size=0.78, test_size=0.22, random_state=seed)
    train_idx, val_idx = next(sss.split(remaining_pairs, remaining_labels))

    train_split = [remaining_pairs[i] for i in train_idx]
    train_labels = [remaining_labels[i] for i in train_idx]
    val_split = [remaining_pairs[i] for i in val_idx]
    val_labels = [remaining_labels[i] for i in val_idx]

    # Seltene Klassen zu train hinzufügen
    train_split += [pair for pair, _ in guaranteed_train]
    train_labels += [label for _, label in guaranteed_train]

    # Daten kopieren
    splits = {
        "train": train_split,
        "valid": val_split
    }

    for split_name, pairs in splits.items():
        for img_path, lbl_path in tqdm(pairs, desc=f"Run {run_id} - Kopiere {split_name}"):
            shutil.copy(img_path, new_dir / split_name / "images" / img_path.name)
            shutil.copy(lbl_path, new_dir / split_name / "labels" / lbl_path.name)

    # data.yaml kopieren
    shutil.copy(BASE_DIR / DATA_YAML_NAME, new_dir / DATA_YAML_NAME)
    print(f"✅ Shuffle abgeschlossen: {new_dir}")
    return new_dir

# Gerät wählen
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# Pfad zu YAML-Konfigurationen
config_path = "/Users/marcschneider/Documents/ultralytics_costum/ultralytics/cfg/models/v8_costum"
configs = glob.glob(f"{config_path}/*.yaml")

# Run 2 & 3 mit Seeds
for run_id, seed in zip([2, 3], [123, 321]):
    run_dir = shuffle_dataset(seed=seed, run_id=run_id)
    data_yaml_path = str(run_dir / DATA_YAML_NAME)
    print(f"Verwende data.yaml aus: {data_yaml_path}")

    for config in configs:
        model_name = os.path.basename(config).replace(".yaml", "")
        project = "yolov8s_modular_experiments"
        name = f"{model_name}_run{run_id}"
        save_dir = os.path.join(project, name)
        os.makedirs(save_dir, exist_ok=True)

        shutil.copy(config, os.path.join(save_dir, os.path.basename(config)))

        try:
            time.sleep(1)
            model = YOLO(config)
            model.train(
                data=data_yaml_path,
                epochs=300,
                batch=16,
                imgsz=640,
                project=project,
                name=name,
                exist_ok=True,
            )
        except Exception as e:
            print(f"Fehler beim Training {model_name}_run{run_id}: {e}")

        print(f"✅ Training abgeschlossen: {model_name}_run{run_id}")
