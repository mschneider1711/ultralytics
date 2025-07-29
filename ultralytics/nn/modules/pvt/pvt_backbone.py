import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
import torch.hub

from .PyramidVisionTransformerV2 import (
    pvt_v2_b0, pvt_v2_b1, pvt_v2_b2, pvt_v2_b2_li, pvt_v2_b3, pvt_v2_b4, pvt_v2_b5
)

# 🔗 Mapping der Modellnamen zu den offiziellen PVTv2 V2 Pretrained Weights
PVT_V2_URLS = {
    "pvt_v2_b0": "https://github.com/whai362/PVT/releases/download/v2/pvt_v2_b0.pth",
    "pvt_v2_b1": "https://github.com/whai362/PVT/releases/download/v2/pvt_v2_b1.pth",
    "pvt_v2_b2": "https://github.com/whai362/PVT/releases/download/v2/pvt_v2_b2.pth",
    "pvt_v2_b2_li": "https://github.com/whai362/PVT/releases/download/v2/pvt_v2_b2_li.pth",
    "pvt_v2_b3": "https://github.com/whai362/PVT/releases/download/v2/pvt_v2_b3.pth",
    "pvt_v2_b4": "https://github.com/whai362/PVT/releases/download/v2/pvt_v2_b4.pth",
    "pvt_v2_b5": "https://github.com/whai362/PVT/releases/download/v2/pvt_v2_b5.pth",
}


PVT_VARIANTS = {
    "pvt_v2_b0": pvt_v2_b0,
    "pvt_v2_b1": pvt_v2_b1,
    "pvt_v2_b2": pvt_v2_b2,
    "pvt_v2_b2_li": pvt_v2_b2_li,
    "pvt_v2_b3": pvt_v2_b3,
    "pvt_v2_b4": pvt_v2_b4,
    "pvt_v2_b5": pvt_v2_b5,
}

def letterbox_tensor(image, new_shape=(640, 640), color=(114, 114, 114)):
    # image: (B, C, H, W)
    B, C, H, W = image.shape
    new_h, new_w = new_shape

    scale = min(new_w / W, new_h / H)
    resized_h, resized_w = int(round(H * scale)), int(round(W * scale))

    # Resize
    image = F.interpolate(image, size=(resized_h, resized_w), mode='bilinear', align_corners=False)

    # Padding
    pad_top = (new_h - resized_h) // 2
    pad_bottom = new_h - resized_h - pad_top
    pad_left = (new_w - resized_w) // 2
    pad_right = new_w - resized_w - pad_left

    image = F.pad(image, (pad_left, pad_right, pad_top, pad_bottom), value=color[0])
    return image

class PyramidVisionTransformerV2(nn.Module):
    def __init__(self, variant="pvt_v2_b2_li", pretrained=False, input_size=640):
        super().__init__()
        assert variant in PVT_VARIANTS, f"[ERROR] Unknown PVTv2 variant: {variant}"
        self.variant = variant
        self.input_size = input_size
        self.backbone = PVT_VARIANTS[variant](pretrained=False)  # manually loaded
        
        if pretrained:
            self.load_pvt_weights(variant)

    def load_pvt_weights(self, variant, verbose=True):
        assert variant in PVT_V2_URLS, f"No URL found for {variant}"
        url = PVT_V2_URLS[variant]

        if verbose:
            print(f"[INFO] Downloading pretrained weights for '{variant}' from: {url}")

        # 1️⃣ Parameter vor dem Laden (erste 5)
        named_params = list(self.backbone.named_parameters())
        print("\n[DEBUG] 🔍 Comparing first 5 parameters BEFORE vs AFTER:")
        tracked_params = []
        for i, (name, param) in enumerate(named_params[:5]):
            mean = param.data.mean().item()
            std = param.data.std().item()
            tracked_params.append((name, param.clone(), mean, std))

        # 2️⃣ Checkpoint laden
        checkpoint = torch.hub.load_state_dict_from_url(url, map_location="cpu", check_hash=True)

        if "model" in checkpoint:
            checkpoint = checkpoint["model"]
        elif "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]

        missing_keys, unexpected_keys = self.backbone.load_state_dict(checkpoint, strict=False)

        # 3️⃣ Vergleich nach dem Laden
        for i, (name, before_tensor, mean_before, std_before) in enumerate(tracked_params):
            after_tensor = dict(self.backbone.named_parameters())[name]
            mean_after = after_tensor.data.mean().item()
            std_after = after_tensor.data.std().item()
            print(f"  [{i}] {name}")
            print(f"       mean: before={mean_before:.6f} | after={mean_after:.6f}")
            print(f"       std : before={std_before:.6f} | after={std_after:.6f}")

        # 4️⃣ Zusammenfassung
        print(f"\n[INFO] ✅ Pretrained weights loaded for '{variant}'")
        print(f"[INFO] 🔄 Loaded params:       {len(checkpoint)}")
        print(f"[INFO] 🟡 Missing params:      {len(missing_keys)}")
        print(f"[INFO] 🔴 Unexpected in ckpt:  {len(unexpected_keys)}")

        if missing_keys:
            print("\n[DETAIL] 🔍 Missing keys:")
            for k in missing_keys:
                print(f"  - {k}")

        if unexpected_keys:
            print("\n[DETAIL] 🔍 Unexpected keys in checkpoint:")
            for k in unexpected_keys:
                print(f"  - {k}")
            

    def forward(self, x):
        features = self.backbone.forward_features(x)

        return features  # P3, P4, P5

    @property
    def out_channels(self):
        return [128, 320, 512]  # Für pvt_v2_b2 oder b2_li
