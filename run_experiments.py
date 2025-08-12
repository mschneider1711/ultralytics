import pandas as pd
from pathlib import Path
from ultralytics import YOLO
import os

# === CONFIG ===
BASE_DIR = Path(__file__).resolve().parent
CUSTOM_CONFIGS_DIR = BASE_DIR / "ultralytics/cfg/models/v8_costum2"
SPLITS_DIR = BASE_DIR / "PlantDoc-3" / "splits"
RESULTS = []

# === COLLECT CONFIGS ===

# Official YOLOv8 (untrained YAML + pretrained PT)
yolo_sizes = ["n", "s", "m", "l", "x"]
official_yaml_configs = [Path(f"yolov8{size}.yaml") for size in yolo_sizes]
official_pretrained_models = [Path(f"yolov8{size}.pt") for size in yolo_sizes]

# Custom configs
custom_configs = sorted(CUSTOM_CONFIGS_DIR.glob("*.yaml"))

# Combine all
config_files = custom_configs + official_yaml_configs + official_pretrained_models
config_files = custom_configs

# === FIND SPLITS ===
split_data_files = sorted(SPLITS_DIR.glob("split*/data.yaml"))

print("Found config files:", config_files)
print("Found splits:", split_data_files)

if not split_data_files or not config_files:
    raise FileNotFoundError("No splits or config files found.")

# === LOOP OVER CONFIGS AND SPLITS ===
for config_path in config_files:
    model_name = config_path.stem
    source_type = (
        "pretrained" if config_path.suffix == ".pt"
        else "official" if config_path.name.startswith("yolov8")
        else "custom"
    )

    for idx, data_yaml in enumerate(split_data_files):

        split_name = data_yaml.parent.name
        print(f"\n--- TRAINING {model_name} ({source_type}) ON {split_name} ---")

        try:
            # Load initial model (pretrained or from config)
            model = YOLO(config_path)

            project_dir = BASE_DIR / "PlantDocBackbone2"
            name = f"{model_name}_{source_type}_{split_name}" 

            # === TRAIN ===
            model.train(
                data=str(data_yaml),
                epochs=300,
                imgsz=640,
                name=name,
                project=str(project_dir),
                exist_ok=True,
                verbose=False,
                batch=16,
                device="cuda"
            )

            # === LOAD best.pt ===
            best_model_path = project_dir / name / "weights" / "best.pt"
            model = YOLO(str(best_model_path))

            # === VALIDATION ===
            val_metrics = model.val(data=str(data_yaml), split='val', verbose=False)
            test_metrics = model.val(data=str(data_yaml), split='test', verbose=False)

            # === METRICS COLLECT ===
            RESULTS.append({
                "model": model_name,
                "source": source_type,
                "split": split_name,
                "val/map50": val_metrics.box.map50,
                "val/map": val_metrics.box.map,
                "val/precision": val_metrics.box.mp,
                "val/recall": val_metrics.box.mr,
                "test/map50": test_metrics.box.map50,
                "test/map": test_metrics.box.map,
                "test/precision": test_metrics.box.mp,
                "test/recall": test_metrics.box.mr,
            })

            results = model.benchmark(
                imgsz=640,
                data=data_yaml,
                device="cuda:0",
                verbose=True,
                half=False,
                format="-",
            )

        except Exception as e:
            print(f"❌ ERROR training {model_name} on {split_name}: {e}")

# === SAVE AND PRINT RESULTS ===
results_df = pd.DataFrame(RESULTS)
results_path = BASE_DIR / "experiment_results.csv"
results_df.to_csv(results_path, index=False)

print("\n=== SUMMARY ===")
print(results_df)
print(f"\nSaved to: {results_path}")
