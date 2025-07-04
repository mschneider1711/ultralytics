from ultralytics import YOLO

model = YOLO("/Users/marcschneider/Desktop/MasterArbeit_Experimente/yolov8s_modular_experiments/yolov8s_modular_experiments/run2/yolov8_P3SwinTrafoCSP_run2/weights/best.pt")
for i, layer in enumerate(model.model.model):
    print(f"Layer {i}: {layer.__class__.__name__}")


print(model.model.model[6])
