from ultralytics import YOLO
import torch

# Optional: manuell MPS erzwingen
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f'Using device: {device}')

model = YOLO("/Users/marcschneider/Documents/ultralytics_costum/ultralytics/cfg/models/v8/yolov8.yaml")

model.train(
    data='coco128.yaml',
    device="mps",
    batch=16,
    imgsz=640,
    epochs=100,
    workers=0,
    amp=False,
    lr0=0.001,
    lrf=0.01,
    momentum=0.937,
    weight_decay=0.0005,
    warmup_epochs=3.0,
    warmup_bias_lr=0.1,
    warmup_momentum=0.8,
    optimizer='AdamW',
    name='yolov8-biformer-mps-force'
)
