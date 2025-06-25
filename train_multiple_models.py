from ultralytics import YOLO
import glob
import os
import time
import sys
import shutil
from contextlib import redirect_stdout, redirect_stderr

# 📁 Pfad zu deinen benutzerdefinierten YOLOv8-Konfigurationen
config_path = "./ultralytics/cfg/models/v8_costum"
configs = glob.glob(f"{config_path}/*.yaml")

# 🔁 Alle .yaml-Konfigurationen durchlaufen
for config in configs:
    if os.path.basename(config) == "yolov8.yaml":
        config = os.path.join(os.path.dirname(config), "yolov8s.yaml")

    model_name = os.path.basename(config).replace(".yaml", "")
    print(f"\n🚀 Training model with config: {config}")

    project = "yolov8s"
    name = model_name
    save_dir = os.path.join(project, name)
    os.makedirs(save_dir, exist_ok=True)

    # 📝 Aktuelle YAML-Datei in Ergebnisordner kopieren
    shutil.copy(config, os.path.join(save_dir, os.path.basename(config)))

    # 🧾 Log-Datei öffnen
    log_file_path = os.path.join(save_dir, "train_output.txt")
    with open(log_file_path, "w") as f:
        with redirect_stdout(f), redirect_stderr(f):
            sys.stdout = f
            sys.stderr = f

            time.sleep(2)  # Optional: Startverzögerung zur Trennung von vorherigem Log

            try:
                # 🧠 Modell laden & trainieren
                model = YOLO(config)
                model.train(
                    data="/Users/marcschneider/Documents/PlantDoc.v4i.yolov8/data.yaml",
                    epochs=300,
                    batch=16,
                    device="mps",
                    imgsz=640,
                    project=project,
                    name=name,
                    exist_ok=True,
                )
            finally:
                # Standardausgabe wiederherstellen
                sys.stdout = sys.__stdout__
                sys.stderr = sys.__stderr__

    print(f"✅ Finished training model: {model_name}")
    print(f"📄 Log saved to: {log_file_path}")
