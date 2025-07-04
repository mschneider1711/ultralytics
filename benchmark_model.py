from ultralytics.utils.benchmarks import benchmark
from ultralytics import YOLO

model_path = r"/Users/marcschneider/Desktop/MasterArbeit_Experimente/yolov8s_modular_experiments/yolov8s_modular_experiments/run2/yolov8_P2SwinTrafoCSP_run2/weights/best.pt"
data_path = r"/Users/marcschneider/Documents/PlantDoc.v4i.yolov8_run2/data.yaml"
device = "cpu" 

# Benchmark on GPU
#benchmark(model=model_path, data=data_path, imgsz=640, half=False, device=device)

# Modell laden
model = YOLO(model_path)

# Validierung durchführen
results = model.val(
    data=data_path,
    split="val",
    imgsz=640,
    device=device  # oder z.B. 0 für GPU
)

# Genauigkeit anzeigen
print("mAP@0.5:", results.box.map50)
print("mAP@0.5:0.95:", results.box.map)
