import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2
from .SwinTransformerV2 import swinv2_tiny, swinv2_small, swinv2_base
import os
import torch.hub
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F2

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

class SwinTransformerV2(nn.Module):
    def __init__(self, variant="swin_tiny", pretrained=None, input_size=640):
        super().__init__()
        assert variant in SWIN_VARIANTS, f"Unknown Swin variant: {variant}"
        if pretrained:
            self.backbone = SWIN_VARIANTS[variant](pretrained=pretrained, pretrained_window_sizes=[8,8,8,4])
            self.load_swin_weights(self.backbone, variant)
        else:
            self.backbone = SWIN_VARIANTS[variant](pretrained=pretrained)

    def load_swin_weights(self, model, variant, verbose=True):
        """
        Enhanced Swin weight loading mit intelligentem Norm-Layer Mapping
        """
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

        # 2️⃣ Lade Original Checkpoint
        checkpoint = torch.hub.load_state_dict_from_url(url, map_location="cpu", check_hash=True)

        if "model" in checkpoint:
            state_dict = checkpoint["model"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        # 3️⃣ **NEUER TEIL**: Smart Weight Mapping
        model_dict = model.state_dict()
        mapped_dict = {}
        mapping_log = []
        
        for k, v in state_dict.items():
            # Skip Classification-spezifische Layer
            if any(skip in k for skip in ['head.', 'attn_mask']):
                if verbose:
                    print(f"[SKIP] Skipping classification layer: {k}")
                continue
                
            # **KRITISCH**: Map den finalen norm Layer NUR zur letzten Stage (gleiche Dimension)
            if k in ['norm.weight', 'norm.bias']:
                param_type = k.split('.')[1]  # 'weight' oder 'bias'
                
                # Der finale norm hat die gleiche Dimension wie die letzte Stage (norm3)
                # Typischerweise: norm.weight shape = [768] -> norm3.weight shape = [768]
                final_stage_key = f'norm3.{param_type}'
                if final_stage_key in model_dict:
                    expected_shape = model_dict[final_stage_key].shape
                    if v.shape == expected_shape:
                        mapped_dict[final_stage_key] = v.clone()
                        mapping_log.append(f"  📋 {k} -> {final_stage_key} (shape: {v.shape})")
                    else:
                        mapping_log.append(f"  ⚠️  {k} shape mismatch: {v.shape} vs expected {expected_shape}")
                
                # Die anderen norm0-2 müssen wir selbst initialisieren (sie haben andere Dimensionen)
            else:
                # Alle anderen Layers direkt übernehmen
                if k in model_dict:
                    mapped_dict[k] = v
                else:
                    if verbose:
                        print(f"[WARN] Key {k} not found in model, skipping")

        # 4️⃣ Print Mapping Information
        if mapping_log and verbose:
            print(f"\n[INFO] 🎯 Applied Weight Mappings:")
            for log in mapping_log:
                print(log)

        # 5️⃣ Lade die gemappten Weights
        missing_keys, unexpected_keys = model.load_state_dict(mapped_dict, strict=False)

        # 6️⃣ Initialisiere verbleibende fehlende Parameter
        if missing_keys:
            print(f"\n[INFO] 🔧 Initializing {len(missing_keys)} missing parameters...")
            with torch.no_grad():
                for key in missing_keys:
                    # Get the actual parameter from the model
                    param = dict(model.named_parameters())[key]
                    
                    if 'norm' in key and 'weight' in key:
                        param.fill_(1.0)
                        print(f"  ✅ Initialized {key} (shape: {param.shape}) to 1.0")
                    elif 'norm' in key and 'bias' in key:
                        param.fill_(0.0)
                        print(f"  ✅ Initialized {key} (shape: {param.shape}) to 0.0")
                    elif 'weight' in key and param.dim() >= 2:
                        nn.init.trunc_normal_(param, std=0.02)
                        print(f"  ✅ Initialized {key} (shape: {param.shape}) with trunc_normal")
                    elif 'bias' in key:
                        param.fill_(0.0)
                        print(f"  ✅ Initialized {key} (shape: {param.shape}) to 0.0")

        # 7️⃣ Vergleich nach dem Laden
        print("\n[DEBUG] 📊 Parameter changes:")
        for i, (name, before_tensor, mean_before, std_before) in enumerate(tracked_params):
            after_tensor = dict(model.named_parameters())[name]
            mean_after = after_tensor.data.mean().item()
            std_after = after_tensor.data.std().item()
            
            # Check if actually changed
            changed = not torch.equal(before_tensor, after_tensor.data)
            status = "🔄 CHANGED" if changed else "⚪ UNCHANGED"
            
            print(f"  [{i}] {name} {status}")
            print(f"       mean: {mean_before:.6f} -> {mean_after:.6f}")
            print(f"       std : {std_before:.6f} -> {std_after:.6f}")

        # 8️⃣ Final Summary
        total_model_params = len(model_dict)
        loaded_params = len(mapped_dict)
        coverage = (loaded_params / total_model_params) * 100
        
        print(f"\n[INFO] ✅ Pretrained weights loaded for '{variant}'")
        print(f"[INFO] 📈 Coverage: {loaded_params}/{total_model_params} ({coverage:.1f}%)")
        print(f"[INFO] 🔄 Mapped params:       {len(mapped_dict)}")
        print(f"[INFO] 🟡 Missing params:      {len(missing_keys)}")
        print(f"[INFO] 🔴 Unexpected in ckpt:  {len([k for k in state_dict.keys() if k not in mapped_dict])}")

        if missing_keys and verbose:
            print("\n[DETAIL] 🔍 Remaining missing keys:")
            for k in missing_keys:
                print(f"  - {k}")

        # 9️⃣ Quality Check: Verify critical mappings worked
        critical_checks = ['norm0.weight', 'norm1.weight', 'norm2.weight', 'norm3.weight']
        loaded_from_pretrained = 0
        initialized_fresh = 0
        
        for check in critical_checks:
            if check in mapped_dict:
                loaded_from_pretrained += 1
            elif check not in missing_keys:
                initialized_fresh += 1
        
        print(f"\n[INFO] 🎯 Norm layers status:")
        print(f"  📋 Loaded from pretrained: {loaded_from_pretrained}/4 (typically only norm3)")
        print(f"  🔧 Freshly initialized: {initialized_fresh}/4 (norm0-2 with correct dimensions)")
        
        if (loaded_from_pretrained + initialized_fresh) == 4:
            print("[INFO] 🎉 All norm layers ready! Mix of pretrained + initialized weights.")
        else:
            print("[WARN] ⚠️  Some norm layers missing - check your model architecture")

        return model


    # Alternative: Separater Weight Mapper (falls du mehr Kontrolle willst)
    def create_weight_mapping(state_dict, model_dict, verbose=True):
        """
        Erstellt ein Mapping zwischen pretrained und model state dicts
        """
        mapped_dict = {}
        
        # Automatische Mappings
        mappings = [
            # (source_pattern, target_pattern, condition)
            ('norm.weight', 'norm{}.weight', lambda: True),
            ('norm.bias', 'norm{}.bias', lambda: True),
        ]
        
        for k, v in state_dict.items():
            mapped = False
            
            # Check custom mappings
            for source_pattern, target_pattern, condition in mappings:
                if k == source_pattern and condition():
                    # Map to multiple targets
                    if '{}' in target_pattern:
                        for stage in [0, 1, 2, 3]:
                            target_key = target_pattern.format(stage)
                            if target_key in model_dict:
                                mapped_dict[target_key] = v.clone()
                                if verbose:
                                    print(f"  📋 {k} -> {target_key}")
                                mapped = True
                    else:
                        if target_pattern in model_dict:
                            mapped_dict[target_pattern] = v
                            mapped = True
                    break
            
            # Direct mapping if not custom mapped
            if not mapped and k in model_dict:
                mapped_dict[k] = v
        
        return mapped_dict
    
    def forward(self, x):
        features = self.backbone.forward(x)

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
