from ultralytics import YOLO


model_path = r"/Users/marcschneider/Documents/MasterArbeit_Experimente/NEW/backbone_experiments/swinsv2_yolov8_custom_split2/weights/best.pt"

model = YOLO(model_path)
model.info(detailed=True)

model.export(format="engine", half=True, dynamic=False, device=0)
