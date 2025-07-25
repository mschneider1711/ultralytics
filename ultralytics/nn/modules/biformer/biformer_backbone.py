import torch
import torch.nn as nn
from .biformer import biformer_tiny, biformer_small, biformer_base
import torch.nn.functional as F

BI_FORMER_VARIANTS = {
    "biformer_tiny": biformer_tiny,
    "biformer_small": biformer_small,
    "biformer_base": biformer_base,
}

class BiFormerBackbone(nn.Module):
    def __init__(self, variant="biformer_tiny", weight=None):
        super().__init__()
        weight = "/Users/marcschneider/Documents/biformer_tiny_best.pth"
        assert variant in BI_FORMER_VARIANTS, f"Unknown BiFormer variant: {variant}"
        self.backbone = BI_FORMER_VARIANTS[variant](pretrained=False)

        # Pretrained Weights laden (optional)
        if weight:
            print(f"[INFO] Loading pretrained weights from {weight}")
            state_dict = torch.load(weight, map_location="cpu")

            # Für HuggingFace / timm-style
            if "model" in state_dict:
                state_dict = state_dict["model"]

            # Laden der Gewichte mit Debugging
            missing_keys, unexpected_keys = self.backbone.load_state_dict(state_dict, strict=False)
            total_keys = len(state_dict)
            loaded_keys = total_keys - len(unexpected_keys)

            print(f"[INFO] ✅ Pretrained weights loaded: {loaded_keys}/{total_keys} params ({(loaded_keys / total_keys)*100:.1f}%)")

            if missing_keys:
                print(f"[WARNING] 🔸 Missing keys ({len(missing_keys)}):")
                for k in missing_keys:
                    print(f"   - {k}")
            else:
                print("[INFO] No missing keys.")

            if unexpected_keys:
                print(f"[WARNING] 🔹 Unexpected keys ({len(unexpected_keys)}):")
                for k in unexpected_keys:
                    print(f"   - {k}")
            else:
                print("[INFO] No unexpected keys.")
    
    def letterbox(self, x, new_shape=640, color=114):
        B, C, H, W = x.shape
        print(f"[DEBUG] Input shape before padding: {x.shape}")
        scale = min(new_shape / H, new_shape / W)
        nh, nw = int(round(H * scale)), int(round(W * scale))

        # Resize
        x_resized = F.interpolate(x, size=(nh, nw), mode='bilinear', align_corners=False)

        # Compute padding
        pad_h = new_shape - nh
        pad_w = new_shape - nw
        top = pad_h // 2
        bottom = pad_h - top
        left = pad_w // 2
        right = pad_w - left

        x_padded = F.pad(x_resized, (left, right, top, bottom), value=color)
        print(f"[DEBUG] Input shape after padding: {x_padded.shape}")
        return x_padded

    def forward(self, x):
        if x.shape[2:] != (640, 640):
            x = self.letterbox(x, new_shape=640)

        features = []
        for i in range(len(self.backbone.downsample_layers)):
            x = self.backbone.downsample_layers[i](x)
            x = self.backbone.stages[i](x)

            # Nur die letzten 3 Feature Maps sammeln (Stage 1–3)
            if i >= 1:
                features.append(x)
                # print(f"[DEBUG] Collected Feature {i-1}: shape = {x.shape}")
        return features

    @property
    def out_channels(self):
        # Passe das ggf. an deine BiFormer-Variante an
        return [128, 256, 512]  # typisch für tiny / small

class Hook:
    def __init__(self, module):
        self.stored = None
        module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.stored = output
