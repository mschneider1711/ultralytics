import torch
import torch.nn as nn
import numpy as np
import cv2
from .biformer import biformer_tiny, biformer_small, biformer_base
import os

BI_FORMER_VARIANTS = {
    "biformer_tiny": biformer_tiny,
    "biformer_small": biformer_small,
    "biformer_base": biformer_base,
}


def letterbox(image, new_shape=(640, 640), color=(114, 114, 114), auto=False, scaleFill=False, scaleup=True, stride=32):
    shape = image.shape[:2]  # current shape [height, width]

    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # width, height padding

    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    elif scaleFill:
        dw, dh = 0.0, 0.0
        new_unpad = new_shape
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:
        if new_unpad[0] <= 0 or new_unpad[1] <= 0:
            raise ValueError(f"[ERROR] Invalid new_unpad size: {new_unpad}")
        image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)

    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return image, ratio, (dw, dh)

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
            imgs = []
            for xi in x:
                xi_np = xi.permute(1, 2, 0).cpu().numpy()
                padded_img = letterbox(xi_np, new_shape=self.input_size, auto=False)[0]
                padded_img = torch.from_numpy(padded_img).permute(2, 0, 1).to(x.device).float() / 255.0
                imgs.append(padded_img)
            x = torch.stack(imgs)

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
