import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from ultralytics import YOLO
from YOLOv8_Explainer import yolov8_heatmap, display_images
import cv2
import os

# === Modellpfade ===
model_path_A = "/Users/marcschneider/Desktop/MasterArbeit_Experimente/yolov8s_modular_experiments/yolov8s_modular_experiments/run2/yolov8_run2/weights/best.pt"
model_path_B = "/Users/marcschneider/Desktop/MasterArbeit_Experimente/yolov8s_modular_experiments/yolov8s_modular_experiments/run2/yolov8_P3SwinTrafoCSP_run2/weights/best.pt"
image_path = "/Users/marcschneider/Documents/PlantDoc.v4i.yolov8/test/images/B2750109-Late_blight_on_a_potato_plant-SPL_jpg.rf.2ee7b2fe0b7703591b646332c6904cda.jpg"
label_path = image_path.replace("/images/", "/labels/").replace(".jpg", ".txt")

device = torch.device("cpu")

# Klassen aus Model A laden
modelA = YOLO(model_path_A)
class_names = [modelA.names[i] for i in range(len(modelA.names))]

# Ground Truth zeichnen (aus deinem Skript)
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
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = class_names[int(cls)] if int(cls) < len(class_names) else f"class {int(cls)}"
            cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return img[..., ::-1]

# Prediction zeichnen
def draw_predictions(model_path, img_path):
    model = YOLO(model_path)
    results = model.predict(img_path, conf=0.3, verbose=False)[0]
    pred_img = results.plot()
    return cv2.cvtColor(pred_img, cv2.COLOR_BGR2RGB)

# Layer-Indizes für Grad-CAM
layer_indices = [4, 6, 9]

# Funktion um Grad-CAM Bilder für alle Layer eines Modells zu erzeugen
def generate_gradcam_images(model_path):
    gradcam_images = []
    for layer_idx in layer_indices:
        heatmap_model = yolov8_heatmap(
            weight=model_path,
            conf_threshold=0.5,
            method="GradCAM",
            layer=[layer_idx],  # nur einen Layer pro Instanz
            ratio=0.02,
            show_box=True,
            renormalize=False,
            device=device
        )
        imgs = heatmap_model(img_path=image_path)  # gibt Liste mit einem PIL Image zurück
        gradcam_images.append(imgs[0])
    return gradcam_images

# Grad-CAM Bilder erzeugen
gradcam_A = generate_gradcam_images(model_path_A)
gradcam_B = generate_gradcam_images(model_path_B)

# Prediction Bilder erzeugen
pred_img_A = draw_predictions(model_path_A, image_path)
pred_img_B = draw_predictions(model_path_B, image_path)

# Originalbild laden und Ground Truth zeichnen
original_img = Image.open(image_path).convert("RGB")
original_img = np.array(original_img.resize((640, 640)))
gt_img = draw_ground_truth(original_img, label_path)

# Plot erzeugen
fig, axes = plt.subplots(2, 5, figsize=(25, 10))

# Reihe 1: Model A
axes[0][0].imshow(gt_img)
axes[0][0].set_title("Ground Truth")
axes[0][0].axis("off")

for i, img in enumerate(gradcam_A):
    axes[0][i + 1].imshow(img)
    axes[0][i + 1].set_title(f"Grad-CAM Layer {layer_indices[i]}")
    axes[0][i + 1].axis("off")

axes[0][4].imshow(pred_img_A)
axes[0][4].set_title("Prediction")
axes[0][4].axis("off")

# Reihe 2: Model B
axes[1][0].imshow(gt_img)
axes[1][0].set_title("Ground Truth")
axes[1][0].axis("off")

for i, img in enumerate(gradcam_B):
    axes[1][i + 1].imshow(img)
    axes[1][i + 1].set_title(f"Grad-CAM Layer {layer_indices[i]}")
    axes[1][i + 1].axis("off")

axes[1][4].imshow(pred_img_B)
axes[1][4].set_title("Prediction")
axes[1][4].axis("off")

plt.tight_layout()
plt.show()
