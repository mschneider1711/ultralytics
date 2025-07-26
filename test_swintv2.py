import torch
from ultralytics.nn.modules.swintransformer_v2 import SwinTransformerV2

# 1. Initialisiere das Modell
model = SwinTransformerV2(
    img_size=640,
    patch_size=4,
    embed_dim=96,
    depths=[2, 2, 6, 2],
    num_heads=[3, 6, 12, 24],
    window_size=8,
    mlp_ratio=4.0,
    qkv_bias=True,
    drop_path_rate=0.1,
    patch_norm=True,
    pretrained_window_sizes=[8, 8, 8, 4]
)

# 2. Lade das Checkpoint-File
ckpt_path = "/Users/marcschneider/Downloads/swinv2_tiny_patch4_window8_256.pth"
ckpt = torch.load(ckpt_path, map_location="cpu")

# 3. Extrahiere das state_dict
if "model" in ckpt:
    state_dict = ckpt["model"]
elif "state_dict" in ckpt:
    state_dict = ckpt["state_dict"]
else:
    state_dict = ckpt  # direktes dict

# 4. Filtere nur passende Parameter
model_state_dict = model.state_dict()
pretrained_dict = {
    k: v for k, v in state_dict.items()
    if k in model_state_dict and v.shape == model_state_dict[k].shape
}

# 5. Debug: Zeige nicht geladene / fehlende Keys
unused_keys = [k for k in state_dict if k not in pretrained_dict]
missing_keys = [k for k in model_state_dict if k not in pretrained_dict]

print(f"[INFO] Nicht geladene Keys aus dem Checkpoint: {len(unused_keys)}")
for k in unused_keys:
    print(f"  ❌ {k}")

print(f"[INFO] Fehlende Keys im Modell: {len(missing_keys)}")
for k in missing_keys:
    print(f"  ⚠️ {k}")

# 6. Update und lade die weights
model_state_dict.update(pretrained_dict)
model.load_state_dict(model_state_dict, strict=False)

print(f"[INFO] Loaded {len(pretrained_dict)} pretrained weights from {ckpt_path}")
