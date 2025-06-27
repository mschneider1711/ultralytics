import os
import shutil
import random
from pathlib import Path
from tqdm import tqdm

# Originalpfad
BASE_DIR = Path("/Users/marcschneider/Documents/PlantDoc.v4i.yolov8")
NEW_DIR_NAME = BASE_DIR.name + "_run2"
NEW_SPLIT_DIR = BASE_DIR.parent / NEW_DIR_NAME

# Skript abbrechen, wenn Zielordner bereits existiert
if NEW_SPLIT_DIR.exists():
    print(f"Ordner {NEW_SPLIT_DIR} existiert bereits. Bitte löschen oder umbenennen.")
    exit(1)

# Zielstruktur anlegen: train/valid/test jeweils mit images & labels
for split in ['train', 'valid', 'test']:
    (NEW_SPLIT_DIR / split / "images").mkdir(parents=True, exist_ok=True)
    (NEW_SPLIT_DIR / split / "labels").mkdir(parents=True, exist_ok=True)

# Bildformate definieren
IMAGE_SUFFIXES = ('.jpg', '.jpeg', '.png')

# Alle Bild-Label-Paare aus Original zusammenstellen
all_image_label_pairs = []

for subset in ['train', 'valid', 'test']:
    img_dir = BASE_DIR / subset / "images"
    lbl_dir = BASE_DIR / subset / "labels"
    for img_file in img_dir.iterdir():
        if img_file.suffix.lower() in IMAGE_SUFFIXES:
            label_file = lbl_dir / (img_file.stem + ".txt")
            if label_file.exists():
                all_image_label_pairs.append((img_file, label_file))

# Shuffle
random.seed(42)
random.shuffle(all_image_label_pairs)

# Feste Splits: 78% train, 12% valid, 10% test
n_total = len(all_image_label_pairs)
n_train = int(n_total * 0.78)
n_val = int(n_total * 0.12)
n_test = n_total - n_train - n_val

splits = {
    "train": all_image_label_pairs[:n_train],
    "valid": all_image_label_pairs[n_train:n_train + n_val],
    "test": all_image_label_pairs[n_train + n_val:]
}

# Kopiere Bilder und Labels in neue Ordnerstruktur
for split, pairs in splits.items():
    for img_path, lbl_path in tqdm(pairs, desc=f"Kopiere {split}"):
        shutil.copy(img_path, NEW_SPLIT_DIR / split / "images" / img_path.name)
        shutil.copy(lbl_path, NEW_SPLIT_DIR / split / "labels" / lbl_path.name)

# Kopiere data.yaml in neuen Ordner
original_yaml = BASE_DIR / "data.yaml"
if original_yaml.exists():
    shutil.copy(original_yaml, NEW_SPLIT_DIR / "data.yaml")
    print("data.yaml wurde erfolgreich kopiert.")
else:
    print("Warnung: data.yaml im Originalordner nicht gefunden.")

print("Split abgeschlossen im Verhältnis 78/12/10.")
print("Neuer Datensatzordner:", NEW_SPLIT_DIR)
