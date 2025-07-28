import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import cv2
from .biformer import biformer_tiny, biformer_small, biformer_base
import os

BI_FORMER_VARIANTS = {
    "biformer_tiny": biformer_tiny,
    "biformer_small": biformer_small,
    "biformer_base": biformer_base,
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

class BiFormer(nn.Module):
    def __init__(self, variant="biformer_tiny", pretrained=None, input_size=640):
        super().__init__()
        assert variant in BI_FORMER_VARIANTS, f"Unknown BiFormer variant: {variant}"
        self.backbone = BI_FORMER_VARIANTS[variant](pretrained=None)
        self.input_size = input_size

        if pretrained:
            self.load_biformer_weights_from_file(pretrained)

    def load_biformer_weights_from_file(self, weight_path, verbose=True):
        assert os.path.isfile(weight_path), f"[ERROR] File not found: {weight_path}"

        if verbose:
            print(f"[INFO] 📂 Loading pretrained weights from local file: {weight_path}")

        # 1️⃣ Parameter vor dem Laden (erste 5)
        named_params = list(self.backbone.named_parameters())
        print("\n[DEBUG] 🔍 Comparing first 5 parameters BEFORE vs AFTER:")
        tracked_params = []
        for i, (name, param) in enumerate(named_params[:5]):
            mean = param.data.mean().item()
            std = param.data.std().item()
            tracked_params.append((name, param.clone(), mean, std))

        # 2️⃣ Lokales Checkpoint laden
        checkpoint = torch.load(weight_path, map_location="cpu")

        # Extrahiere ggf. state_dict
        if "model" in checkpoint:
            checkpoint = checkpoint["model"]
        elif "state_dict" in checkpoint:
            checkpoint = checkpoint["state_dict"]

        # 3️⃣ Lade Gewichte
        missing_keys, unexpected_keys = self.backbone.load_state_dict(checkpoint, strict=False)

        # 4️⃣ Vergleich nach dem Laden
        for i, (name, before_tensor, mean_before, std_before) in enumerate(tracked_params):
            after_tensor = dict(self.backbone.named_parameters())[name]
            mean_after = after_tensor.data.mean().item()
            std_after = after_tensor.data.std().item()
            print(f"  [{i}] {name}")
            print(f"       mean: before={mean_before:.6f} | after={mean_after:.6f}")
            print(f"       std : before={std_before:.6f} | after={std_after:.6f}")

        # 5️⃣ Zusammenfassung
        print(f"\n[INFO] ✅ Local weights loaded successfully.")
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
            print("applying letter box within backbone")
            x = letterbox_tensor(x, new_shape=(self.input_size, self.input_size))

        features = []
        for i in range(len(self.backbone.downsample_layers)):
            x = self.backbone.downsample_layers[i](x)
            x = self.backbone.stages[i](x)
            if i >= 1:  # Only use last 3 stages (P3, P4, P5)
                features.append(x)
        return features

    @property
    def out_channels(self):
        return [128, 256, 512]


class Hook:
    def __init__(self, module):
        self.stored = None
        module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.stored = output
