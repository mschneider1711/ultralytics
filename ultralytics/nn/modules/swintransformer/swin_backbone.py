import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from .SwinTransformerV2 import swinv2_tiny, swinv2_small, swinv2_base
import os
import torch.hub

SWIN_VARIANTS = {
    "swin_tiny": swinv2_tiny,
    "swin_small": swinv2_small,
    "swin_base": swinv2_base,
}

SWIN_PRETRAINED_URLS = {
    "swin_tiny": "https://github.com/SwinTransformer/storage/releases/download/v2.0.0/swinv2_tiny_patch4_window8_256.pth",
    "swin_small": "https://github.com/SwinTransformer/storage/releases/download/v2.0.0/swinv2_small_patch4_window8_256.pth",
    "swin_base": "https://github.com/SwinTransformer/storage/releases/download/v2.0.0/swinv2_base_patch4_window8_256.pth",
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


class SwinTransformerV2(nn.Module):
    def __init__(self, variant="swin_tiny", pretrained=None, input_size=640):
        super().__init__()
        assert variant in SWIN_VARIANTS, f"Unknown Swin variant: {variant}"
        self.backbone = SWIN_VARIANTS[variant](pretrained=pretrained)
        self.input_size = input_size

        if pretrained:
            self.load_swin_weights(self.backbone, variant)


    def load_swin_weights(self, model, variant, verbose=True):
        assert variant in SWIN_PRETRAINED_URLS, f"No URL found for {variant}"
        url = SWIN_PRETRAINED_URLS[variant]

        if verbose:
            print(f"[INFO] Downloading pretrained weights for '{variant}' from: {url}")

        # 1️⃣ Parameter vor dem Laden (erste 5)
        named_params = list(model.named_parameters())
        print("\n[DEBUG] 🔍 Comparing first 5 parameters BEFORE vs AFTER:")
        tracked_params = []
        for i, (name, param) in enumerate(named_params[:5]):
            mean = param.data.mean().item()
            std = param.data.std().item()
            tracked_params.append((name, param.clone(), mean, std))

        # 2️⃣ Lade Weights
        checkpoint = torch.hub.load_state_dict_from_url(url, map_location="cpu", check_hash=True)

        if "model" in checkpoint:
            checkpoint = checkpoint["model"]
        elif "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]

        missing_keys, unexpected_keys = model.load_state_dict(checkpoint, strict=False)

        # 3️⃣ Vergleich nach dem Laden
        for i, (name, before_tensor, mean_before, std_before) in enumerate(tracked_params):
            after_tensor = dict(model.named_parameters())[name]
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
        if x.shape[2:] != (self.input_size, self.input_size):
            x = letterbox_tensor(x, new_shape=(self.input_size, self.input_size))


        features = self.backbone.forward_features(x)

        return features

    @property
    def out_channels(self):
        # Passe diese Liste ggf. an die Architektur deines Swin-Modells an
        return [192, 384, 768]

class Hook:
    def __init__(self, module):
        self.stored = None
        module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.stored = output
