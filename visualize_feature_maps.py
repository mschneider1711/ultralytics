import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
from ultralytics import YOLO
from torchvision import transforms
from PIL import Image
import os
from ultralytics.utils.plotting import colors  # <- YOLO-Farbpalette

# === Modellpfade ===
model_path_A = "/Users/marcschneider/Desktop/MasterArbeit_Experimente/yolov8s_modular_experiments/yolov8s_modular_experiments/run2/yolov8_run2/weights/best.pt"
model_path_B = "/Users/marcschneider/Desktop/MasterArbeit_Experimente/yolov8s_modular_experiments/yolov8s_modular_experiments/run2/yolov8_P3SwinTrafoCSP_run2/weights/best.pt"
image_path = "/Users/marcschneider/Documents/PlantDoc.v4i.yolov8/test/images/B2750109-Late_blight_on_a_potato_plant-SPL_jpg.rf.2ee7b2fe0b7703591b646332c6904cda.jpg"
label_path = image_path.replace("/images/", "/labels/").replace(".jpg", ".txt")

# === Labels (optional)
class_names = [
    'Apple Scab Leaf', 'Apple leaf', 'Apple rust leaf', 'Bell_pepper leaf', 'Bell_pepper leaf spot',
    'Blueberry leaf', 'Cherry leaf', 'Corn Gray leaf spot', 'Corn leaf blight', 'Corn rust leaf',
    'Peach leaf', 'Potato leaf', 'Potato leaf early blight', 'Potato leaf late blight', 'Raspberry leaf',
    'Soyabean leaf', 'Soybean leaf', 'Squash Powdery mildew leaf', 'Strawberry leaf', 'Tomato Early blight leaf',
    'Tomato Septoria leaf spot', 'Tomato leaf', 'Tomato leaf bacterial spot', 'Tomato leaf late blight',
    'Tomato leaf mosaic virus', 'Tomato leaf yellow virus', 'Tomato mold leaf',
    'Tomato two spotted spider mites leaf', 'grape leaf black rot', 'grape leaf'
]


# === Device
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

# === Bild vorbereiten
transform = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor(),
])
img_pil = Image.open(image_path).convert('RGB')
img_tensor = transform(img_pil).unsqueeze(0).to(device)
original_img = np.array(img_pil.resize((640, 640)))

# === Optional: GT zeichnen
def draw_ground_truth(img, label_path):
    img = cv2.cvtColor(img.copy(), cv2.COLOR_RGB2BGR)
    h, w = img.shape[:2]
    if not os.path.exists(label_path):
        return img[..., ::-1]
    with open(label_path, 'r') as f:
        for line in f.readlines():
            cls, x_c, y_c, bw, bh = map(float, line.strip().split())
            x1 = int((x_c - bw / 2) * w)
            y1 = int((y_c - bh / 2) * h)
            x2 = int((x_c + bw / 2) * w)
            y2 = int((y_c + bh / 2) * h)
            
            # === Farbe wie YOLO
            color = colors(int(12), bgr=True)
            
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            label = f"{class_names[int(cls)]}" if int(cls) < len(class_names) else f"class {int(cls)}"
            cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return img[..., ::-1]


# === Prediction-Overlay (optional)
def draw_predictions(model_path, img_path):
    model = YOLO(model_path)
    results = model.predict(img_path, conf=0.3, verbose=False)[0]
    pred_img = results.plot()[..., ::-1]
    return cv2.resize(pred_img, (640, 640))

# === Feature Maps extrahieren (z. B. Layer 6, 8, 9)
def get_feature_maps(model_path):
    model = YOLO(model_path)
    model.model.to(device)
    layers = model.model.model[:10]
    feature_maps = []

    def hook_fn(module, input, output):
        feature_maps.append(output)

    hooks = [layers[i].register_forward_hook(hook_fn) for i in [4, 6, 9]]
    _ = model.model(img_tensor)
    for h in hooks: h.remove()

    return feature_maps

# === Overlay erzeugen: Mittelwert über Kanäle
def create_overlay(fmap):
    fmap = fmap.squeeze(0)  # [C, H, W]
    fmap_mean = fmap.sum(dim=0).detach().cpu().numpy()  # [H, W]

    # Normalisieren
    fmap_norm = (fmap_mean - fmap_mean.min()) / (fmap_mean.max() - fmap_mean.min() + 1e-6)
    heatmap = np.uint8(255 * fmap_norm)
    heatmap = cv2.resize(heatmap, (640, 640))
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(original_img, 0.6, heatmap_color, 0.4, 0)
    return overlay[..., ::-1]

# === Feature Maps
fmap_A = get_feature_maps(model_path_A)
fmap_B = get_feature_maps(model_path_B)

# === Predictions
pred_img_A = draw_predictions(model_path_A, image_path)
pred_img_B = draw_predictions(model_path_B, image_path)

# === Plot
fig, axes = plt.subplots(2, 5, figsize=(25, 10))

# === Zeile A
axes[0][0].imshow(draw_ground_truth(original_img, label_path)); axes[0][0].set_title("GT A"); axes[0][0].axis("off")
for i in range(3):
    axes[0][i+1].imshow(create_overlay(fmap_A[i]))
    axes[0][i+1].set_title(f"Feature Mean A - P{i+3}")
    axes[0][i+1].axis("off")
axes[0][4].imshow(pred_img_A); axes[0][4].set_title("Prediction A"); axes[0][4].axis("off")

# === Zeile B
axes[1][0].imshow(draw_ground_truth(original_img, label_path)); axes[1][0].set_title("GT B"); axes[1][0].axis("off")
for i in range(3):
    axes[1][i+1].imshow(create_overlay(fmap_B[i]))
    axes[1][i+1].set_title(f"Feature Mean B - P{i+3}")
    axes[1][i+1].axis("off")
axes[1][4].imshow(pred_img_B); axes[1][4].set_title("Prediction B"); axes[1][4].axis("off")

plt.tight_layout()
plt.show()
