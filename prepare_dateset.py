import os
import shutil
import yaml
import pandas as pd
from pathlib import Path
from roboflow import Roboflow
from sklearn.model_selection import StratifiedShuffleSplit

# === CONFIGURATION ===
API_KEY = "UAKZhKjdRo7DK00tJJfB"
WORKSPACE = "joseph-nelson"
PROJECT = "plantdoc"
VERSION = 3
YOLO_FORMAT = "yolov12"

CLASSES_TO_REMOVE = [
    "Potato leaf",
    "Soybean leaf",
    "Tomato two spotted spider mites leaf"
]

NUM_SPLITS = 3
VAL_RATIO = 0.12  # approx. 80/11/9 ratio (train/val/test)

# === DOWNLOAD DATASET ===
rf = Roboflow(api_key=API_KEY)
project = rf.workspace(WORKSPACE).project(PROJECT)
version = project.version(VERSION)
dataset = version.download(YOLO_FORMAT)
dataset_path = Path(dataset.location)

images_dir = dataset_path / "train/images"
labels_dir = dataset_path / "train/labels"
yaml_path = dataset_path / "data.yaml"

# === UPDATE YAML: REMOVE CLASSES ===
with open(yaml_path, "r") as f:
    data_yaml = yaml.safe_load(f)

original_names = data_yaml["names"]
filtered_names = [n for n in original_names if n not in CLASSES_TO_REMOVE]

# Remap old class indices to new ones
name2id_new = {name: i for i, name in enumerate(filtered_names)}
remap_ids = {i: name2id_new[name] for i, name in enumerate(original_names) if name not in CLASSES_TO_REMOVE}

data_yaml["names"] = filtered_names
data_yaml["nc"] = len(filtered_names)

# === CLEAN TRAINING LABELS & REWRITE ===
new_image_list = []
new_label_list = []
new_labels = []

for file in sorted(os.listdir(labels_dir)):
    label_path = labels_dir / file
    image_file = file.replace(".txt", ".jpg")
    image_path = images_dir / image_file

    keep = True
    new_lines = []

    with open(label_path, "r") as lf:
        lines = lf.readlines()

    for line in lines:
        if not line.strip():
            continue
        parts = line.strip().split()
        cls_id = int(parts[0])
        class_name = original_names[cls_id]
        if class_name in CLASSES_TO_REMOVE:
            keep = False
            break
        new_cls_id = remap_ids[cls_id]
        new_line = " ".join([str(new_cls_id)] + parts[1:]) + "\n"
        new_lines.append(new_line)

    if keep and new_lines:
        with open(label_path, "w") as f:
            f.writelines(new_lines)
        new_image_list.append(image_file)
        new_label_list.append(file)
        new_labels.append(int(new_lines[0].split()[0]))
    else:
        if image_path.exists():
            image_path.unlink()
        if label_path.exists():
            label_path.unlink()

# === STRATIFIED SPLIT ===
df = pd.DataFrame({
    "image": new_image_list,
    "label_file": new_label_list,
    "label": new_labels
})
df = df.drop_duplicates(subset="image").reset_index(drop=True)

splitter = StratifiedShuffleSplit(n_splits=NUM_SPLITS, test_size=VAL_RATIO, random_state=42)
splits = list(splitter.split(df["image"], df["label"]))

# === REMAP TEST LABELS ===
test_images_dir = dataset_path / "test/images"
test_labels_dir = dataset_path / "test/labels"

remapped_test_dir = dataset_path / "test_remapped"
test_remap_images = remapped_test_dir / "images"
test_remap_labels = remapped_test_dir / "labels"
test_remap_images.mkdir(parents=True, exist_ok=True)
test_remap_labels.mkdir(parents=True, exist_ok=True)

for lbl_file in test_labels_dir.glob("*.txt"):
    img_file = lbl_file.with_suffix(".jpg").name
    new_lines = []
    keep = True

    with open(lbl_file, "r") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()
        if not parts:
            continue
        cls_id = int(parts[0])
        class_name = original_names[cls_id]
        if class_name in CLASSES_TO_REMOVE:
            keep = False
            break
        new_cls_id = remap_ids[cls_id]
        new_line = " ".join([str(new_cls_id)] + parts[1:]) + "\n"
        new_lines.append(new_line)

    if keep and new_lines:
        with open(test_remap_labels / lbl_file.name, "w") as f:
            f.writelines(new_lines)
        shutil.copy(test_images_dir / img_file, test_remap_images / img_file)

# === CREATE SPLIT FOLDERS ===
splits_root = dataset_path / "splits"
splits_root.mkdir(exist_ok=True)

for i, (train_idx, val_idx) in enumerate(splits):
    split_dir = splits_root / f"split{i}"
    for split in ["train", "val", "test"]:
        for sub in ["images", "labels"]:
            (split_dir / split / sub).mkdir(parents=True, exist_ok=True)

    # Copy train and val data
    for subset, idxs in [("train", train_idx), ("val", val_idx)]:
        for idx in idxs:
            row = df.iloc[idx]
            shutil.copy(images_dir / row["image"], split_dir / subset / "images" / row["image"])
            shutil.copy(labels_dir / row["label_file"], split_dir / subset / "labels" / row["label_file"])

    # Copy remapped test data
    for test_img in test_remap_images.glob("*"):
        shutil.copy(test_img, split_dir / "test/images" / test_img.name)
    for test_lbl in test_remap_labels.glob("*"):
        shutil.copy(test_lbl, split_dir / "test/labels" / test_lbl.name)

    # Write data.yaml for each split
    data_yaml_split = {
        "train": str((split_dir / "train").resolve()),
        "val": str((split_dir / "val").resolve()),
        "test": str((split_dir / "test").resolve()),
        "nc": data_yaml["nc"],
        "names": data_yaml["names"]
    }
    with open(split_dir / "data.yaml", "w") as f:
        yaml.safe_dump(data_yaml_split, f)

    # Check for data leakage between train and val
    train_images = set(os.listdir(split_dir / "train/images"))
    val_images = set(os.listdir(split_dir / "val/images"))
    intersection = train_images & val_images

    if intersection:
        print(f"Data leakage in split{i}: {len(intersection)} duplicate images in train and val.")
        for img in sorted(intersection)[:5]:
            print(f"  - {img}")
        raise ValueError(f"Split{i} contains duplicate images between train and val.")
    else:
        print(f"Split{i} OK: no duplicate images between train and val.")

# === CLEANUP ===
for folder in ["test_remapped"]:
    path_to_delete = dataset_path / folder
    if path_to_delete.exists():
        shutil.rmtree(path_to_delete)
        print(f"Deleted folder: {path_to_delete}")

yaml_main = dataset_path / "data.yaml"
if yaml_main.exists():
    yaml_main.unlink()
    print(f"Deleted file: {yaml_main}")

print("Finished. Only 'splits/' directory remains.")
