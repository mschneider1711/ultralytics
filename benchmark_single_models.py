import torch
from pathlib import Path
from ultralytics import YOLO
from torch import nn
import numpy as np


# Custom AdaptiveAvgPool, das Padding und flexible Output-Größe unterstützt
class AdaptiveAvgPool2dCustom(nn.Module):
    def __init__(self, output_size):
        super().__init__()
        # Falls nur eine Zahl angegeben wird, als quadratische Größe interpretieren
        if isinstance(output_size, int):
            output_size = (output_size, output_size)
        self.output_size = np.array(output_size)

    def forward(self, x: torch.Tensor):
        """Einfaches 2D-Average-Pooling mit optionalem Padding auf Zielgröße"""
        shape_x = x.shape
        # Padding am rechten Rand, falls Eingabe kleiner als gewünschte Breite
        if shape_x[-1] < self.output_size[-1]:
            pad_w = self.output_size[-1] - shape_x[-1]
            paddzero = torch.zeros((shape_x[0], shape_x[1], shape_x[2], pad_w), device=x.device)
            x = torch.cat((x, paddzero), axis=-1)

        # Berechne Kernel- und Stride-Größen für exaktes Downsampling
        stride_size = np.floor(np.array(x.shape[-2:]) / self.output_size).astype(np.int32)
        kernel_size = np.array(x.shape[-2:]) - (self.output_size - 1) * stride_size

        avg = nn.AvgPool2d(kernel_size=list(kernel_size), stride=list(stride_size))
        x = avg(x)
        return x


# Pfade und Einstellungen
MODEL_PATH = "/Users/marcschneider/Documents/MasterArbeit_Experimente/NEW/backbone_experiments/pvt2_yolov8_custom_split2/weights/best.pt"
DATA_PATH = Path("/Users/marcschneider/Documents/MasterArbeit_Experimente/NEW/PlantDoc-3/splits/split0/data.yaml")
FORMATS = ["-", "torchscript", "engine"]  # PyTorch, TorchScript, TensorRT
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
IMGSZ = 640
VERBOSE = True

# Modell laden
model = YOLO(MODEL_PATH)

# Ersetze alle Standard-Pooling-Layer durch die Custom-Variante
for name, module in model.model.named_modules():
    if isinstance(module, nn.AdaptiveAvgPool2d):
        print(f"Ersetze: {name}")
        new_pool = AdaptiveAvgPool2dCustom((7, 7))
        parent_name = ".".join(name.split(".")[:-1])
        attr_name = name.split(".")[-1]
        parent = model.model.get_submodule(parent_name) if parent_name else model.model
        setattr(parent, attr_name, new_pool)

# Für jedes Export-Format jeweils einmal in FP32 und FP16 testen
for fmt in FORMATS:
    for half in [False, True]:
        print(f"\n--- Benchmark: format={fmt}, half={half} ---")
        model.benchmark(
            data=DATA_PATH,
            format=fmt,
            half=half,
            device=DEVICE,
            imgsz=IMGSZ,
            verbose=VERBOSE
        )
