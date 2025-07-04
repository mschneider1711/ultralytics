from YOLOv8_Explainer import yolov8_heatmap, display_images
import torch
device = torch.device("cpu")

model = yolov8_heatmap(
    weight="/Users/marcschneider/Desktop/MasterArbeit_Experimente/yolov8s_modular_experiments/yolov8s_modular_experiments/run2/yolov8_P3SwinTrafoCSP_run2/weights/best.pt",
    conf_threshold=0.5, 
    method = "GradCAMPlusPlus", 
    layer=[15],
    ratio=0.02,
    show_box=True,
    renormalize=False,
    device=device
)

# Zielklasse explizit auf 12 setzen (Late Blight Potato)
imgs = model(
    img_path="/Users/marcschneider/Documents/PlantDoc.v4i.yolov8/train/images/_1030395_JPG_jpg.rf.1240a4816a77c133877bcc83c1d23970.jpg",
)
display_images(imgs)
