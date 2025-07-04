import cv2
from ultralytics import solutions

# Heatmap-Instanz mit Custom-Modell
heatmap = solutions.Heatmap(
    model="/Users/marcschneider/Desktop/MasterArbeit_Experimente/yolov8s_modular_experiments/yolov8s_modular_experiments/run2/yolov8_P3SwinTrafoCSP_run2/weights/best.pt",
    classes=[13]  # nur Klasse 13
)

# Bild mit OpenCV laden (BGR)
img_path = "/Users/marcschneider/Documents/PlantDoc.v4i.yolov8/test/images/B2750109-Late_blight_on_a_potato_plant-SPL_jpg.rf.2ee7b2fe0b7703591b646332c6904cda.jpg"
img = cv2.imread(img_path)

# Heatmap erzeugen
results = heatmap(img)
import matplotlib.pyplot as plt

plt.imshow(cv2.cvtColor(results.heatmap, cv2.COLOR_BGR2RGB))
plt.title("Heatmap")
plt.axis("off")
plt.show()
