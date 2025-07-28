import torch
import torch.nn as nn
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


def letterbox(image, new_shape=(640, 640), color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True, stride=32):
    shape = image.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    ratio = r, r
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]

    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    elif scaleFill:
        dw, dh = 0.0, 0.0
        new_unpad = new_shape
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]

    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return image, ratio, (dw, dh)


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
        if x.shape[2:] != (self.input_size, self.input_size):
            imgs = []
            for xi in x:
                xi_np = xi.permute(1, 2, 0).cpu().numpy()
                padded_img = letterbox(xi_np, new_shape=self.input_size, auto=False)[0]
                padded_img = torch.from_numpy(padded_img).permute(2, 0, 1).to(x.device).float() / 255.0
                imgs.append(padded_img)
            x = torch.stack(imgs)

        features = self.backbone.forward_features(x)
        return features  # P3, P4, P5

    @property
    def out_channels(self):
        return [128, 320, 512]  # Für pvt_v2_b2 oder b2_li
