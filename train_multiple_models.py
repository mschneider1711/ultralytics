from ultralytics import YOLO
import glob
import os
import sys

# 🔗 Optional: dein Dataset-Link
# dataset url: https://universe.roboflow.com/joseph-nelson/plantdoc/dataset/4

# 📁 Konfigurationspfad
config_path = "./ultralytics/cfg/models/v8_costum"
configs = glob.glob(f"{config_path}/*.yaml")

# 🔁 Über alle Konfigurationen iterieren
for config in configs:
    if os.path.basename(config) == "yolov8.yaml":
        config = os.path.join(os.path.dirname(config), "yolov8s.yaml")

    model_name = os.path.basename(config).replace(".yaml", "")
    print(f"\n🚀 Training model with config: {config}")

    model = YOLO(config)

    # 📁 Pfad zum Output-Ordner (save_dir)
    project = "yolov8s"
    name = model_name
    save_dir = os.path.join(project, name)
    os.makedirs(save_dir, exist_ok=True)

    # 📝 Log-Datei vorbereiten
    log_file_path = os.path.join(save_dir, "train_output.txt")
    with open(log_file_path, "w") as f:
        # 🧭 stdout und stderr umleiten
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = sys.stderr = f

        try:
            # 🏋️ Training starten
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
            # ↩️ Standardausgabe zurücksetzen
            sys.stdout = old_stdout
            sys.stderr = old_stderr

    print(f"✅ Finished training model with config: {config}")
    print(f"📄 Log saved to: {log_file_path}")
