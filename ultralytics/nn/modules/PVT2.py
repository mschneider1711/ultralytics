import torch
import torch.nn as nn
from timm.models.layers import DropPath, trunc_normal_

# Basic Conv block (BN + Activation)
class Conv(nn.Module):
    default_act = nn.SiLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        p = p or (k // 2 if isinstance(k, int) else [x // 2 for x in k])
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


# Overlap Patch Embedding (kernel=3, stride=1 → no downsampling)
class OverlapPatchEmbed(nn.Module):
    def __init__(self, in_chans, embed_dim, patch_size=3, stride=1):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size,
                              stride=stride, padding=patch_size // 2)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)  # [B, C, H, W]
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # [B, N, C]
        x = self.norm(x)
        return x, H, W


# Depthwise conv for MLP branch
class DWConv(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None,
                 attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)
        else:
            self.sr = None

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr is not None:
            x_ = x.transpose(1, 2).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).transpose(1, 2)
            x_ = self.norm(x_)
        else:
            x_ = x

        kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class PVTBlock(nn.Module):
    def __init__(self, dim, num_heads=8, mlp_ratio=4., qkv_bias=True,
                 drop=0., attn_drop=0., drop_path=0., sr_ratio=1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads, qkv_bias, None, attn_drop, drop, sr_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), drop=drop)

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x


class C3PVT2(nn.Module):
    def __init__(self, c1, c2, n=1, e=0.5):
        super().__init__()
        self.n = n
        self.c_ = int(c2 * e)

        self.cv1 = Conv(c1, self.c_, 1, 1)
        self.embed = OverlapPatchEmbed(self.c_, self.c_, patch_size=3, stride=1)
        self.cv2 = Conv(c1, self.c_, 1, 1)
        self.cv3 = Conv(2 * self.c_, c2, 1, 1)

        self.blocks = None  # Wird im ersten Forward-Durchlauf erzeugt

    def _get_pvt2_config(self, H):
        if H > 80:
            return dict(sr_ratio=8, num_heads=1, mlp_ratio=8)
        elif H > 40:
            return dict(sr_ratio=4, num_heads=2, mlp_ratio=8)
        elif H > 20:
            return dict(sr_ratio=2, num_heads=4, mlp_ratio=4)
        else:
            return dict(sr_ratio=1, num_heads=8, mlp_ratio=4)

    def build_blocks(self, sr_ratio, num_heads, mlp_ratio):
        """Baue eine Sequenz aus n PVT-Blöcken."""
        return nn.Sequential(*[
            PVTBlock(
                dim=self.c_,
                num_heads=num_heads,
                sr_ratio=sr_ratio,
                mlp_ratio=mlp_ratio
            )
            for _ in range(self.n)
        ])

    def forward(self, x):
        B, _, H, W = x.shape
        cfg = self._get_pvt2_config(H)

        if self.blocks is None:
            # Erzeuge Blöcke beim ersten Forward-Aufruf mit dynamischer Konfiguration
            self.blocks = self.build_blocks(cfg['sr_ratio'], cfg['num_heads'], cfg['mlp_ratio']).to(x.device)

        y1 = self.cv1(x)
        y1, H, W = self.embed(y1)  # ergibt (B, H×W, C)

        for blk in self.blocks:
            y1 = blk(y1, H, W)

        y1 = y1.transpose(1, 2).view(B, -1, H, W)  # zurück zu (B, C, H, W)
        y2 = self.cv2(x)

        return self.cv3(torch.cat((y1, y2), dim=1))
