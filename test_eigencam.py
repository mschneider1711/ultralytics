import torch
import matplotlib.pyplot as plt
import numpy as np
import cv2
from ultralytics import YOLO
from PIL import Image
import os

# === Eigene yolo_cam importieren
from yolo_cam.eigen_cam import EigenCAM
from yolo_cam.utils.image import show_cam_on_image

# === Modellpfade
model_path_A = "/Users/marcschneider/Desktop/MasterArbeit_Experimente/yolov8s_modular_experiments/yolov8s_modular_experiments/run2/yolov8_run2/weights/best.pt"
model_path_B = "/Users/marcschneider/Desktop/MasterArbeit_Experimente/yolov8s_modular_experiments/yolov8s_modular_experiments/run2/yolov8_P3SwinTrafoCSP_run2/weights/best.pt"
image_path = "/Users/marcschneider/Documents/PlantDoc.v4i.yolov8/test/images/B2750109-Late_blight_on_a_potato_plant-SPL_jpg.rf.2ee7b2fe0b7703591b646332c6904cda.jpg"
label_path = image_path.replace("/images/", "/labels/").replace(".jpg", ".txt")

# === Bild vorbereiten
img_pil = Image.open(image_path).convert('RGB')
original_img = np.array(img_pil.resize((640, 640)))
rgb_img = original_img / 255.0  # Normiert für CAM

# === Ground Truth zeichnen
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
            cv2.putText(img, f"class {int(cls)}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return img[..., ::-1]

# === Prediction Overlay
def draw_predictions(model_path, img_path):
    model = YOLO(model_path)
    results = model.predict(img_path, conf=0.3, verbose=False)[0]
    pred_img = results.plot()[..., ::-1]
    return cv2.resize(pred_img, (640, 640))

# === EigenCAM Wrapper
def get_eigencam_overlays(model_path, rgb_img):
    model = YOLO(model_path)
    model.model.to("cpu").eval()

    # Layer: -2, -3, -4
    target_layers = [
        model.model.model[15],
        model.model.model[18],
        model.model.model[21]
    ]

    overlays = []
    for layer in target_layers:
        cam = EigenCAM(model, target_layers=[layer], task="od")
        grayscale_cam = cam(rgb_img)[0, :, :]
        cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
        overlays.append(cam_image)
    return overlays

# === Ergebnisse generieren
eigencam_A = get_eigencam_overlays(model_path_A, rgb_img)
eigencam_B = get_eigencam_overlays(model_path_B, rgb_img)
prediction_img_A = draw_predictions(model_path_A, image_path)
prediction_img_B = draw_predictions(model_path_B, image_path)

# === Plotten
fig, axes = plt.subplots(2, 5, figsize=(25, 10))

# Zeile A
axes[0][0].imshow(draw_ground_truth(original_img, label_path))
axes[0][0].set_title("Ground Truth A")
axes[0][0].axis("off")
for i in range(3):
    axes[0][i+1].imshow(eigencam_A[i])
    axes[0][i+1].set_title(f"EigenCAM A - Layer {-2 - i}")
    axes[0][i+1].axis("off")
axes[0][4].imshow(prediction_img_A)
axes[0][4].set_title("Prediction A")
axes[0][4].axis("off")

# Zeile B
axes[1][0].imshow(draw_ground_truth(original_img, label_path))
axes[1][0].set_title("Ground Truth B")
axes[1][0].axis("off")
for i in range(3):
    axes[1][i+1].imshow(eigencam_B[i])
    axes[1][i+1].set_title(f"EigenCAM B - Layer {-2 - i}")
    axes[1][i+1].axis("off")
axes[1][4].imshow(prediction_img_B)
axes[1][4].set_title("Prediction B")
axes[1][4].axis("off")

plt.tight_layout()
plt.show()
