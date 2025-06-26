from ultralytics import YOLO
import glob
import os
import time
import shutil
import torch

# Pfad zu deinen benutzerdefinierten YOLOv8-Konfigurationen
config_path = "/Users/marcschneider/Documents/ultralytics_costum/ultralytics/cfg/models/v8_costum"
configs = glob.glob(f"{config_path}/*.yaml")

print(configs)

# device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# Alle .yaml-Konfigurationen durchlaufen
for config in configs:
    if os.path.basename(config) == "yolov8.yaml":
        config = os.path.join(os.path.dirname(config), "yolov8s.yaml")

    model_name = os.path.basename(config).replace(".yaml", "")
    print(f"\n🚀 Training model with config: {config}")

    project = "yolov8s_modular_experiments"
    name = model_name
    save_dir = os.path.join(project, name)
    os.makedirs(save_dir, exist_ok=True)

    # Aktuelle YAML-Datei in Ergebnisordner kopieren
    shutil.copy(config, os.path.join(save_dir, os.path.basename(config)))

    try:
        time.sleep(2)  # Optional: Startverzögerung zur Trennung von vorherigem Output

        # Modell laden & trainieren
        model = YOLO(config)
        model.train(
            data="/Users/marcschneider/Documents/PlantDoc.v4i.yolov8/data.yaml",
            epochs=300,
            batch=16,
            imgsz=640,
            project=project,
            name=name,
            exist_ok=True,
        )

    except Exception as e:
        print(f" Fehler beim Training von {model_name}: {e}")

    print(f" Finished training model: {model_name}")
    print(f" Ergebnisse gespeichert in: {save_dir}")
