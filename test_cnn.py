import torch
import numpy as np
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import cv2

import timm
from ultralytics.nn.modules.pvt.pvt_backbone import PyramidVisionTransformerV2
from ultralytics.nn.modules.biformer.biformer_backbone import BiFormer
from ultralytics.nn.modules.swintransformer.swin_backbone import SwinTransformerV2

# 🔹 Lade & Transformiere Bild
image_path = "/Users/marcschneider/Documents/PlantDoc/train/images/zT5la_jpg.rf.8260e340c1a990bbd1eca36e41ab23ba.jpg"
img = Image.open(image_path).convert('RGB')

transform = T.Compose([
    T.Resize((1280, 1280)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])
img_tensor = transform(img).unsqueeze(0)
img_cv = np.array(img.resize((1280, 1280)))[:, :, ::-1].copy()

# 🔹 Modell-Definitionen
model_configs = [
    ("ConvNeXt Tiny", timm.create_model("convnext_base", pretrained=True, features_only=True)),
    ("PVTv2 B0", PyramidVisionTransformerV2(variant="pvt_v2_b2", pretrained=True)),
    ("BiFormer Base", BiFormer("biformer_base", pretrained=True)),
    ("Swin V2 Base", SwinTransformerV2("swin_base", pretrained=True)),
]

# 🔹 Inferenz & Sammle Features
all_raw_maps = []
all_overlays = []
num_top_channels = 4

def compute_topk_avg(feat, k=4):
    B, C, H, W = feat.shape
    feat = feat[0]  # [C, H, W]
    channel_means = feat.view(C, -1).mean(dim=1)
    topk_indices = torch.topk(channel_means, k).indices
    avg_map = feat[topk_indices].mean(dim=0).cpu().numpy()
    return avg_map

for model_name, model in model_configs:
    print(f"Running: {model_name}")
    model.eval()
    with torch.no_grad():
        features = model(img_tensor)

    raw_stage_maps = []
    overlay_stage_maps = []
    for feat in features:
        avg_map = compute_topk_avg(feat, k=num_top_channels)

        # 🔹 Raw Feature Map speichern
        raw_stage_maps.append(avg_map)

        # 🔹 Heatmap Overlay vorbereiten
        heatmap_resized = cv2.resize(avg_map, (1280, 1280))
        heatmap_norm = cv2.normalize(heatmap_resized, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        heatmap_color = cv2.applyColorMap(heatmap_norm, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(img_cv, 0.6, heatmap_color, 0.4, 0)
        overlay_stage_maps.append(overlay)

    all_raw_maps.append((model_name, raw_stage_maps))
    all_overlays.append((model_name, overlay_stage_maps))

# 🔹 1. Figure – Raw Feature Maps
fig_rows = len(all_raw_maps)
fig_cols = max(len(r[1]) for r in all_raw_maps)
fig, axs = plt.subplots(fig_rows, fig_cols, figsize=(4 * fig_cols, 4 * fig_rows))

for row_idx, (model_name, maps) in enumerate(all_raw_maps):
    for col_idx in range(fig_cols):
        ax = axs[row_idx][col_idx] if fig_rows > 1 else axs[col_idx]
        if col_idx < len(maps):
            ax.imshow(maps[col_idx], cmap='viridis')
            ax.set_title(f"{model_name} - Stage {col_idx}")
        else:
            ax.axis('off')
        ax.axis('off')

plt.tight_layout()
plt.suptitle("Raw Feature Maps (⌀ Top-4 Kanäle pro Stage)", fontsize=16)
plt.subplots_adjust(top=0.92)
plt.show()

# 🔹 2. Figure – Overlay Heatmaps
fig, axs = plt.subplots(fig_rows, fig_cols, figsize=(4 * fig_cols, 4 * fig_rows))

for row_idx, (model_name, overlays) in enumerate(all_overlays):
    for col_idx in range(fig_cols):
        ax = axs[row_idx][col_idx] if fig_rows > 1 else axs[col_idx]
        if col_idx < len(overlays):
            ax.imshow(overlays[col_idx][:, :, ::-1])  # BGR → RGB
            ax.set_title(f"{model_name} - Stage {col_idx}")
        else:
            ax.axis('off')
        ax.axis('off')

plt.tight_layout()
plt.suptitle("Overlay Heatmaps auf Originalbild", fontsize=16)
plt.subplots_adjust(top=0.92)
plt.show()
