import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# -----------------------------------
# 1) Depthwise Separable Convolution
# -----------------------------------
class DepthwiseSeparableConv(nn.Module):
    """
    Depthwise separable convolution:
    - Depthwise: per-channel spatial conv (groups=in_channels)
    - Pointwise: 1x1 conv mixes channels
    This reduces params/FLOPs vs. a standard conv.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1):
        super().__init__()
        self.depthwise = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
            groups=in_channels, bias=False
        )
        self.pointwise = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=1, stride=1, padding=0, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        return self.relu(x)


# -------------------------------------------------------------
# 2) Windowed + Shifted Multi-Head Self-Attention (Swin-style)
# -------------------------------------------------------------
class WindowedShiftedSelfAttention(nn.Module):
    """
    Windowed + Shifted (Swin-style) MHSA using PyTorch 2.x
    F.scaled_dot_product_attention for efficiency.

    Input/Output: (B, C, H, W)
    - window_size (ws): window side length (e.g., 8 or 16)
    - shift_size: ws//2 when shift=True to cross window boundaries
    """
    def __init__(
        self,
        channels: int,
        num_heads: int = 4,
        window_size: int = 8,
        shift: bool = True,
        mlp_ratio: float = 4.0,
        attn_dropout: float = 0.0,
        proj_dropout: float = 0.0,
        use_gelu: bool = True,
    ):
        super().__init__()
        assert channels % num_heads == 0, "channels must be divisible by num_heads"
        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.window_size = window_size
        self.shift_size = window_size // 2 if shift else 0

        # Pre-norms
        self.norm1 = nn.LayerNorm(channels)
        self.norm2 = nn.LayerNorm(channels)

        # QKV projections
        self.to_q = nn.Linear(channels, channels, bias=False)
        self.to_k = nn.Linear(channels, channels, bias=False)
        self.to_v = nn.Linear(channels, channels, bias=False)

        # Output projection
        self.to_out = nn.Linear(channels, channels, bias=True)
        self.attn_drop = nn.Dropout(attn_dropout)
        self.proj_drop = nn.Dropout(proj_dropout)

        # MLP
        hidden = int(channels * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.GELU() if use_gelu else nn.SiLU(),
            nn.Dropout(proj_dropout),
            nn.Linear(hidden, channels),
            nn.Dropout(proj_dropout),
        )

        # Stable init (residual ~ identity at start)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.zeros_(self.to_out.weight)
        nn.init.zeros_(self.mlp[-2].weight)

    @staticmethod
    def _partition_windows(x, ws):
        """(B, H, W, C) -> (B*nW, L, C), where L=ws*ws"""
        B, H, W, C = x.shape
        x = x.view(B, H // ws, ws, W // ws, ws, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        nW = (H // ws) * (W // ws)
        x = x.view(B * nW, ws * ws, C)
        return x, nW, (H, W)

    @staticmethod
    def _unpartition_windows(x, ws, shape_hw, B):
        """(B*nW, L, C) -> (B, H, W, C)"""
        H, W = shape_hw
        nWh, nWw = H // ws, W // ws
        C = x.shape[-1]
        x = x.view(B, nWh, nWw, ws, ws, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(B, H, W, C)
        return x

    def _build_attn_mask(self, Hp, Wp, device):
        """
        Build a boolean attention mask for shifted windows so tokens
        from different (shifted) windows cannot attend each other.
        Shape: (nW, 1, L, L), broadcastable to (B*nW, heads, L, L).
        """
        ws, ss = self.window_size, self.shift_size
        if ss == 0:
            return None

        img_mask = torch.zeros((1, 1, Hp, Wp), device=device)  # (1,1,H,W)
        cnt = 0
        h_slices = (slice(0, -ws), slice(-ws, -ss), slice(-ss, None))
        w_slices = (slice(0, -ws), slice(-ws, -ss), slice(-ss, None))
        for h in h_slices:
            for w in w_slices:
                img_mask[:, :, h, w] = cnt
                cnt += 1

        # Apply the same cyclic shift as features
        img_mask = torch.roll(img_mask, shifts=(-ss, -ss), dims=(2, 3))
        # To (B,H,W,C)
        mask_tokens = img_mask.permute(0, 2, 3, 1).contiguous()
        # Partition
        mask_windows, nW, _ = self._partition_windows(mask_tokens, ws)  # (nW, L, 1)
        mask_windows = mask_windows.view(nW, ws * ws)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)  # (nW, L, L)
        attn_mask = attn_mask != 0  # True = disallow
        attn_mask = attn_mask.unsqueeze(1)  # (nW, 1, L, L)
        return attn_mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        ws, ss = self.window_size, self.shift_size

        # Pad to multiples of window size
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        if pad_h or pad_w:
            x = F.pad(x, (0, pad_w, 0, pad_h))
        B, C, Hp, Wp = x.shape

        # Cyclic shift (if enabled)
        if ss > 0:
            x = torch.roll(x, shifts=(-ss, -ss), dims=(2, 3))

        # (B,C,H,W) -> (B,H,W,C) then LN over C
        tokens = x.permute(0, 2, 3, 1).contiguous()
        y = self.norm1(tokens.view(B * Hp * Wp, C)).view(B, Hp, Wp, C)

        # Partition windows
        win_tokens, nW, hw = self._partition_windows(y, ws)  # (B*nW, L, C)
        L = ws * ws

        # QKV
        q = self.to_q(win_tokens)
        k = self.to_k(win_tokens)
        v = self.to_v(win_tokens)

        q = q.view(B * nW, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(B * nW, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(B * nW, L, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        # Attention mask for shifted windows
        attn_mask = self._build_attn_mask(Hp, Wp, q.device)
        if attn_mask is not None:
            attn_mask = attn_mask.repeat(B, 1, 1, 1)  # (B*nW, 1, L, L)

        # --- Scaled dot-product attention ---
        if hasattr(F, "scaled_dot_product_attention"):
            attn = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=attn_mask,
                dropout_p=self.attn_drop.p if self.training else 0.0,
                is_causal=False,
            ) # (B*nW, heads, L, head_dim)
        else:
            # Manual SDP for PyTorch < 2.0 (same math)
            scale = (self.head_dim) ** -0.5
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # (B*nW, heads, L, L)

            if attn_mask is not None:
                # attn_mask: (B*nW, 1, L, L) boolean (True = disallow)
                mask = attn_mask.expand(-1, scores.size(1), -1, -1)
                scores = scores.masked_fill(mask, float("-inf"))

            probs = torch.softmax(scores, dim=-1)
            if self.training and self.attn_drop.p > 0:
                probs = torch.dropout(probs, p=self.attn_drop.p, train=True)

            attn = torch.matmul(probs, v)  # (B*nW, heads, L, head_dim)

        # Back to (B*nW, L, C)
        attn = attn.permute(0, 2, 1, 3).contiguous().view(B * nW, L, C)
        attn = self.to_out(attn)
        attn = self.proj_drop(attn)

        # Residual + MLP
        win_tokens = win_tokens + attn
        y2 = self.norm2(win_tokens)
        mlp_out = self.mlp(y2)
        win_tokens = win_tokens + mlp_out

        # Merge windows back: (B,H',W',C) -> (B,C,H',W')
        out_tokens = self._unpartition_windows(win_tokens, ws, hw, B)
        out = out_tokens.permute(0, 3, 1, 2).contiguous()

        # Inverse shift
        if ss > 0:
            out = torch.roll(out, shifts=(ss, ss), dims=(2, 3))

        # Remove padding
        if pad_h or pad_w:
            out = out[:, :, :H, :W]

        return out


# ------------------------------------------
# 3) Mini-ASPP / PPM "lite" for bottleneck
# ------------------------------------------
class MiniASPP(nn.Module):
    """
    Lightweight context block:
    - 1x1
    - 3x3 (dil=2)
    - 3x3 (dil=4)
    (+ optional global pooling branch)
    Concatenates and fuses via 1x1 to out_channels.
    """
    def __init__(self, in_channels: int, out_channels: int, use_gpool: bool = False):
        super().__init__()
        mid = in_channels // 2

        self.b0 = nn.Sequential(
            nn.Conv2d(in_channels, mid, kernel_size=1, bias=False),
            nn.BatchNorm2d(mid), nn.ReLU(inplace=True)
        )
        # DW-Separable 3x3 branches with dilation
        self.b1 = DepthwiseSeparableConv(
            in_channels, mid, kernel_size=3, stride=1, padding=2, dilation=2
        )
        self.b2 = DepthwiseSeparableConv(
            in_channels, mid, kernel_size=3, stride=1, padding=4, dilation=4
        )

        self.use_gpool = use_gpool
        if use_gpool:
            self.gp = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(in_channels, mid, kernel_size=1, bias=False),
                nn.GroupNorm(num_groups=1, num_channels=mid),
                nn.ReLU(inplace=True)
            )

        fuse_in = mid * (3 + (1 if use_gpool else 0))
        self.fuse = nn.Sequential(
            nn.Conv2d(fuse_in, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        y0 = self.b0(x)
        y1 = self.b1(x)
        y2 = self.b2(x)
        ys = [y0, y1, y2]
        if self.use_gpool:
            yg = self.gp(x)
            yg = F.interpolate(yg, size=x.shape[-2:], mode='bilinear', align_corners=False)
            ys.append(yg)
        y = torch.cat(ys, dim=1)
        return self.fuse(y)


# -------------------------------
# 4) UltraLightFCN (final model)
# -------------------------------
class UltraLightFCN(nn.Module):
    """
    Ultra-light FCN for binary (or multi-class) segmentation.
    - Lightweight encoder with depthwise separable + dilated convs
    - Mini-ASPP for cheap global context
    - Windowed + Shifted MHSA in the bottleneck
    - Bilinear upsampling decoder
    - One shallow skip from block1 fused after upconv7
    """
    def __init__(self, in_channels: int = 3, num_classes: int = 1, params: dict = None):
        super().__init__()
        self.num_classes = num_classes

        if params is None:
            params = {
                # Encoder
                'enc_channels': [16, 16, 32, 32, 64],
                'enc_kernel_sizes': [3, 3, 3, 3, 3],
                'enc_strides': [1, 2, 2, 1, 1],
                'dilations': [2, 4],
                # Decoder
                'dec_channels': [32, 16, 16],
                'dec_kernel_sizes': [3, 3],
                'dec_strides': [1, 1],
                'upscale': [2, 2],
                # Context
                'mini_aspp': True,
                'mini_aspp_gpool': False,
                # Attention
                'use_sa': True,
                'sa_windowed': True,
                'sa_window_size': 8,
                'sa_shifted': True,
                'sa_heads': 4,
                'sa_dropout': 0.0,
            }

        # ----- Encoder -----
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, params['enc_channels'][0],
                      kernel_size=params['enc_kernel_sizes'][0],
                      stride=params['enc_strides'][0],
                      padding=params['enc_kernel_sizes'][0] // 2, bias=False),
            nn.BatchNorm2d(params['enc_channels'][0]),
            nn.ReLU(inplace=True)
        )

        self.dsconv2 = DepthwiseSeparableConv(
            params['enc_channels'][0], params['enc_channels'][1],
            kernel_size=params['enc_kernel_sizes'][1],
            stride=params['enc_strides'][1],
            padding=params['enc_kernel_sizes'][1] // 2
        )

        self.dsconv3 = DepthwiseSeparableConv(
            params['enc_channels'][1], params['enc_channels'][2],
            kernel_size=params['enc_kernel_sizes'][2],
            stride=params['enc_strides'][2],
            padding=params['enc_kernel_sizes'][2] // 2
        )

        self.dilconv4 = nn.Sequential(
            nn.Conv2d(params['enc_channels'][2], params['enc_channels'][3],
                      kernel_size=params['enc_kernel_sizes'][3], stride=1,
                      padding=(params['enc_kernel_sizes'][3] // 2) * params['dilations'][0],
                      dilation=params['dilations'][0], bias=False),
            nn.BatchNorm2d(params['enc_channels'][3]),
            nn.ReLU(inplace=True)
        )

        self.dilconv5 = nn.Sequential(
            nn.Conv2d(params['enc_channels'][3], params['enc_channels'][4],
                      kernel_size=params['enc_kernel_sizes'][4], stride=1,
                      padding=(params['enc_kernel_sizes'][4] // 2) * params['dilations'][1],
                      dilation=params['dilations'][1], bias=False),
            nn.BatchNorm2d(params['enc_channels'][4]),
            nn.ReLU(inplace=True)
        )

        # ----- Bottleneck: Mini-ASPP + (Windowed, Shifted) SA -----
        self.use_mini_aspp = params.get('mini_aspp', True)
        bottleneck_c = params['enc_channels'][4]
        if self.use_mini_aspp:
            self.mini_aspp = MiniASPP(
                bottleneck_c, bottleneck_c,
                use_gpool=params.get('mini_aspp_gpool', False)
            )

        self.use_sa = params.get('use_sa', True)
        if self.use_sa and params.get('sa_windowed', True):
            self.sa = WindowedShiftedSelfAttention(
                channels=bottleneck_c,
                num_heads=params.get('sa_heads', 4),
                window_size=params.get('sa_window_size', 8),
                shift=params.get('sa_shifted', True),
                mlp_ratio=4.0,
                attn_dropout=params.get('sa_dropout', 0.0),
                proj_dropout=params.get('sa_dropout', 0.0),
            )

        # ----- Decoder -----
        self.upconv6 = nn.Sequential(
            nn.Upsample(scale_factor=params['upscale'][0], mode='bilinear', align_corners=False),
            nn.Conv2d(bottleneck_c, params['dec_channels'][0],
                      kernel_size=params['dec_kernel_sizes'][0],
                      stride=params['dec_strides'][0],
                      padding=params['dec_kernel_sizes'][0] // 2, bias=False),
            nn.BatchNorm2d(params['dec_channels'][0]),
            nn.ReLU(inplace=True)
        )

        self.upconv7 = nn.Sequential(
            nn.Upsample(scale_factor=params['upscale'][1], mode='bilinear', align_corners=False),
            nn.Conv2d(params['dec_channels'][0], params['dec_channels'][1],
                      kernel_size=params['dec_kernel_sizes'][1],
                      stride=params['dec_strides'][1],
                      padding=params['dec_kernel_sizes'][1] // 2, bias=False),
            nn.BatchNorm2d(params['dec_channels'][1]),
            nn.ReLU(inplace=True)
        )

        # ----- Shallow skip from block1 -----
        # Align channels of block1 to dec_channels[1] and fuse via concat after upconv7
        self.skip_proj = nn.Conv2d(params['enc_channels'][0], params['dec_channels'][1], kernel_size=1, bias=False)
        self.skip_bn   = nn.BatchNorm2d(params['dec_channels'][1])

        # Fuse (2 * dec_channels[1] -> dec_channels[2])
        self.fuse8 = nn.Sequential(
            nn.Conv2d(params['dec_channels'][1] * 2, params['dec_channels'][2], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(params['dec_channels'][2]),
            nn.ReLU(inplace=True)
        )

        # Final logits
        self.conv9 = nn.Conv2d(params['dec_channels'][2], num_classes, kernel_size=1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder
        x1 = self.block1(x)    # (B, C1, 256, 256)  <-- shallow skip
        x2 = self.dsconv2(x1)  # (B, C2, 128, 128)
        x3 = self.dsconv3(x2)  # (B, C3, 64, 64)
        x4 = self.dilconv4(x3) # (B, C4, 64, 64)
        x5 = self.dilconv5(x4) # (B, C5, 64, 64)

        # Bottleneck
        if self.use_mini_aspp:
            x5 = self.mini_aspp(x5)
        if self.use_sa:
            x5 = self.sa(x5)

        # Decoder
        x6 = self.upconv6(x5)   # (B, dec0, 128, 128)
        x7 = self.upconv7(x6)   # (B, dec1, 256, 256)

        # Shallow skip: project & fuse after upconv7
        skip = self.skip_bn(self.skip_proj(x1))      # (B, dec1, 256, 256)
        x_cat = torch.cat([x7, skip], dim=1)         # (B, 2*dec1, 256, 256)
        x8 = self.fuse8(x_cat)                       # (B, dec2, 256, 256)

        # Logits
        out = self.conv9(x8)                         # (B, num_classes, H, W)
        return out
