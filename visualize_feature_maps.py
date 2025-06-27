import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
from ultralytics import YOLO
from torchvision import transforms
from PIL import Image
import os

# === Modellpfade ===
model_path_A = "/Users/marcschneider/Desktop/MasterArbeit_Experimente/yolov8s_modular_experiments/yolov8s_modular_experiments/yolov8/weights/best.pt"
model_path_B = "/Users/marcschneider/Desktop/MasterArbeit_Experimente/yolov8s_modular_experiments/yolov8s_modular_experiments/yolov8_P2SwinTrafoCSP/weights/best.pt"
image_path = "/Users/marcschneider/Documents/PlantDoc.v4i.yolov8/test/images/why-are-my-pepper-plants-yellow-yellow-pepper-plants-yellow-leaves-green-veins-pepper-plants-yellow-veins_jpg.rf.1b69a7bf01e0ed9eae5fa02c88bed399.jpg"
label_path = image_path.replace("/images/", "/labels/").replace(".jpg", ".txt")

# === Device wählen
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

# === Vorverarbeitung
transform = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor(),
])
img_pil = Image.open(image_path).convert('RGB')
img_tensor = transform(img_pil).unsqueeze(0).to(device)
original_img = np.array(img_pil.resize((640, 640)))

# === Klassenlabels für PlantDoc
class_names = [
    'Apple Scab Leaf', 'Apple leaf', 'Apple rust leaf', 'Bell_pepper leaf spot', 'Bell_pepper leaf',
    'Blueberry leaf', 'Cherry leaf', 'Corn Gray leaf spot', 'Corn leaf blight', 'Corn rust leaf',
    'Peach leaf', 'Potato leaf early blight', 'Potato leaf late blight', 'Potato leaf',
    'Raspberry leaf', 'Soyabean leaf', 'Soybean leaf', 'Squash Powdery mildew leaf', 'Strawberry leaf',
    'Tomato Early blight leaf', 'Tomato Septoria leaf spot', 'Tomato leaf bacterial spot',
    'Tomato leaf late blight', 'Tomato leaf mosaic virus', 'Tomato leaf yellow virus', 'Tomato leaf',
    'Tomato mold leaf', 'Tomato two spotted spider mites leaf', 'grape leaf black rot', 'grape leaf'
]

# === GT Bounding Box + Klassenlabel zeichnen
def draw_ground_truth(img, label_path):
    img = cv2.cvtColor(img.copy(), cv2.COLOR_RGB2BGR) 
    img = img.copy()
    h, w = img.shape[:2]
    if not os.path.exists(label_path):
        return img
    with open(label_path, 'r') as f:
        for line in f.readlines():
            cls, x_c, y_c, bw, bh = map(float, line.strip().split())
            x1 = int((x_c - bw / 2) * w)
            y1 = int((y_c - bh / 2) * h)
            x2 = int((x_c + bw / 2) * w)
            y2 = int((y_c + bh / 2) * h)
            color = (0, 255, 0)
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            label = class_names[int(cls)] if int(cls) < len(class_names) else f"class {int(cls)}"
            cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return img[..., ::-1]  # BGR → RGB


# === Funktion: Vorhersagen visualisieren
def draw_predictions(model_path, img_path):
    model = YOLO(model_path)
    results = model.predict(img_path, conf=0.3, verbose=False)[0]
    return results.plot()[..., ::-1]

# === Feature Maps extrahieren
def get_feature_maps(model_path):
    model = YOLO(model_path)
    model.model.to(device)
    backbone = model.model.model[:10]
    feature_maps = []

    def hook_fn(module, input, output):
        feature_maps.append(output)

    # P3, P4, P5 Hooks
    hooks = [backbone[6].register_forward_hook(hook_fn),
             backbone[8].register_forward_hook(hook_fn),
             backbone[9].register_forward_hook(hook_fn)]

    _ = model.model(img_tensor)

    for h in hooks:
        h.remove()

    return feature_maps

# === Feature Map Overlay
def create_overlay(fmap):
    fmap = fmap.squeeze(0)
    topk = torch.topk(fmap.sum(dim=(1, 2)), k=5).indices
    reduced = fmap[topk].sum(0).detach().cpu().numpy()
    reduced = (reduced - reduced.min()) / (reduced.max() - reduced.min() + 1e-6)
    reduced = cv2.resize(reduced, (640, 640))
    heatmap = (reduced * 255).astype(np.uint8)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(original_img, 0.6, heatmap_color, 0.4, 0)
    return overlay[..., ::-1]

# === Feature Maps berechnen
fmap_A = get_feature_maps(model_path_A)
fmap_B = get_feature_maps(model_path_B)

# === Visualisierung: 2 Zeilen, je 5 Bilder
fig, axes = plt.subplots(2, 5, figsize=(25, 10))

# === Zeile 0: Modell A
axes[0][0].imshow(draw_ground_truth(original_img, label_path))
axes[0][0].set_title("Original + GT A")
axes[0][0].axis("off")
for i in range(3):
    overlay = create_overlay(fmap_A[i])
    axes[0][i + 1].imshow(overlay)
    axes[0][i + 1].set_title(f"Model A - P{i + 3}")
    axes[0][i + 1].axis("off")
axes[0][4].imshow(draw_predictions(model_path_A, image_path))
axes[0][4].set_title("Prediction A")
axes[0][4].axis("off")

# === Zeile 1: Modell B
axes[1][0].imshow(draw_ground_truth(original_img, label_path))
axes[1][0].set_title("Original + GT B")
axes[1][0].axis("off")
for i in range(3):
    overlay = create_overlay(fmap_B[i])
    axes[1][i + 1].imshow(overlay)
    axes[1][i + 1].set_title(f"Model B - P{i + 3}")
    axes[1][i + 1].axis("off")
axes[1][4].imshow(draw_predictions(model_path_B, image_path))
axes[1][4].set_title("Prediction B")
axes[1][4].axis("off")

plt.tight_layout()
plt.show()
