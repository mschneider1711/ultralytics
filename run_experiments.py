import os
import shutil
import random
import time
import glob
import torch
from ultralytics import YOLO
from pathlib import Path
from tqdm import tqdm

# Basispfad zum ursprünglichen Datensatz
BASE_DIR = Path("/Users/marcschneider/Documents/PlantDoc.v4i.yolov8")
DATA_YAML_NAME = "data.yaml"

# Funktion: Shufflen und kopieren
def shuffle_dataset(seed: int, run_id: int):
    new_dir = BASE_DIR.parent / f"{BASE_DIR.name}_run{run_id}"

    if new_dir.exists():
        print(f"Überspringe Shuffle: Ordner {new_dir} existiert bereits.")
        return new_dir  # statt exit(1)

    for split in ['train', 'valid', 'test']:
        (new_dir / split / "images").mkdir(parents=True, exist_ok=True)
        (new_dir / split / "labels").mkdir(parents=True, exist_ok=True)

    IMAGE_SUFFIXES = ('.jpg', '.jpeg', '.png')
    all_image_label_pairs = []

    for subset in ['train', 'valid', 'test']:
        img_dir = BASE_DIR / subset / "images"
        lbl_dir = BASE_DIR / subset / "labels"
        for img_file in img_dir.iterdir():
            if img_file.suffix.lower() in IMAGE_SUFFIXES:
                label_file = lbl_dir / (img_file.stem + ".txt")
                if label_file.exists():
                    all_image_label_pairs.append((img_file, label_file))

    random.seed(seed)
    random.shuffle(all_image_label_pairs)

    n_total = len(all_image_label_pairs)
    n_train = int(n_total * 0.78)
    n_val = int(n_total * 0.12)
    n_test = n_total - n_train - n_val

    splits = {
        "train": all_image_label_pairs[:n_train],
        "valid": all_image_label_pairs[n_train:n_train + n_val],
        "test": all_image_label_pairs[n_train + n_val:]
    }

    for split, pairs in splits.items():
        for img_path, lbl_path in tqdm(pairs, desc=f"Run {run_id} - Kopiere {split}"):
            shutil.copy(img_path, new_dir / split / "images" / img_path.name)
            shutil.copy(lbl_path, new_dir / split / "labels" / lbl_path.name)

    shutil.copy(BASE_DIR / DATA_YAML_NAME, new_dir / DATA_YAML_NAME)
    print(f"Shuffle abgeschlossen: {new_dir}")

    return new_dir

# Gerät wählen
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

# Pfad zu YAML-Konfigurationen
config_path = "/Users/marcschneider/Documents/ultralytics_costum/ultralytics/cfg/models/v8_costum"
configs = glob.glob(f"{config_path}/*.yaml")

# Zwei verschiedene Seeds ausprobieren
for run_id, seed in zip([2, 3], [123, 321]):
    run_dir = shuffle_dataset(seed=seed, run_id=run_id)

    data_yaml_path = str(run_dir / DATA_YAML_NAME)

    print(f"Use data.yaml from: {data_yaml_path}")

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

        print(f"Training abgeschlossen: {model_name}_run{run_id}")
