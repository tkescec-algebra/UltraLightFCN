import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools

from models.UltraLightFCN_base import DepthwiseSeparableConv, MiniASPP, WindowedShiftedSelfAttention


# -----------------------------------------------------------------------------
# 1) UltraLight Encoder (Shared)
# -----------------------------------------------------------------------------
class UltraLightEncoder(nn.Module):
    """
    Encoder part of UltraLightFCN:
      Conv -> DSConv -> DSConv -> DilatedConv -> DilatedConv
    Produces a feature map of shape (B,64,H/4,W/4)
    """
    def __init__(self, in_channels: int = 3, params: dict | None = None):
        super().__init__()
        if params is None:
            params = {
                # Encoder
                'enc_channels': [16, 16, 32, 32, 64],
                'enc_kernel_sizes': [3, 3, 3, 3, 3],
                'enc_strides': [1, 2, 2, 1, 1],
                'dilations': [2, 4],
                # Context
                'mini_aspp': True,
                'mini_aspp_gpool': False,
                # Attention
                'use_sa': True,
                'sa_windowed': True,
                'sa_window_size': 8,
                'sa_shifted': True,
                'sa_heads': 4,
                'sa_dropout': 0.1,
            }

        self.params = params

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

        self.use_sa = params.get('use_sa', True) and params.get('sa_windowed', True)
        if self.use_sa:
            self.sa = WindowedShiftedSelfAttention(
                channels=bottleneck_c,
                num_heads=params.get('sa_heads', 4),
                window_size=params.get('sa_window_size', 8),
                shift=params.get('sa_shifted', True),
                mlp_ratio=4.0,
                attn_dropout=params.get('sa_dropout', 0.0),
                proj_dropout=params.get('sa_dropout', 0.0),
            )

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

    @property
    def out_channels(self) -> int:
        return self.params['enc_channels'][4]

    @property
    def skip_channels(self) -> int:
        return self.params['enc_channels'][0]

    def forward(self, x: torch.Tensor):
        x1 = self.block1(x)     # skip (B, C1, H, W)
        x2 = self.dsconv2(x1)   # (B, C2, H/2, W/2)
        x3 = self.dsconv3(x2)   # (B, C3, H/4, W/4)
        x4 = self.dilconv4(x3)  # (B, C4, H/4, W/4)
        x5 = self.dilconv5(x4)  # (B, C5, H/4, W/4)

        if self.use_mini_aspp:
            x5 = self.mini_aspp(x5)
        if self.use_sa:
            x5 = self.sa(x5)

        return x5, x1  # deep feat, shallow skip

# -----------------------------------------------------------------------------
# 2) Projection Head for SimCLR
# -----------------------------------------------------------------------------
class ProjectionHead(nn.Module):
    """
    MLP projection head for SimCLR:
      Linear(64 -> 128) -> ReLU -> Linear(128 -> 64)
    """
    def __init__(self, in_dim=64, hidden_dim=128, out_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim, bias=False)
        )
        self._init_weights()  # Initialize weights

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def forward(self, x):
        return self.net(x)

# -----------------------------------------------------------------------------
# 4) SimCLR Model
# -----------------------------------------------------------------------------
class SimCLRModel(nn.Module):
    """
    SimCLR self-supervised model:
      Encoder -> GlobalAvgPool -> ProjectionHead -> normalized embedding
    """
    def __init__(self, encoder: nn.Module, proj_head: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.proj_head = proj_head
    def forward(self, x):
        feat, _ = self.encoder(x)                       # (B,64,H/4,W/4)
        pooled = self.gap(feat).view(x.size(0), -1)     # (B,64)
        z = self.proj_head(pooled)                      # (B,64)
        return F.normalize(z, dim=1)                    # L2 normalize

# -----------------------------------------------------------------------------
# 5) Segmentation Model (fine-tune)
# -----------------------------------------------------------------------------
class UltraLightSegmentation(nn.Module):
    """
    Segmentation model using pretrained encoder:
      Encoder -> Decoder (upsample -> conv blocks) -> output logits
    """
    def __init__(self, encoder: nn.Module, num_classes: int=1, params: dict | None = None):
        super().__init__()
        if params is None:
            params = {
                # Decoder
                'dec_channels': [32, 16, 16],
                'dec_kernel_sizes': [3, 3],
                'dec_strides': [1, 1],
                'upscale': [2, 2],
            }
        # ----- Encoder -----
        self.encoder = encoder

        bottleneck_c = encoder.out_channels
        encoder_ch_0 = encoder.params['enc_channels'][0]

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
        self.skip_proj = nn.Conv2d(encoder_ch_0, params['dec_channels'][1], kernel_size=1, bias=False)
        self.skip_bn = nn.BatchNorm2d(params['dec_channels'][1])

        # Fuse (2 * dec_channels[1] -> dec_channels[2])
        self.conv8  = nn.Sequential(
            nn.Conv2d(params['dec_channels'][1] * 2, params['dec_channels'][2], kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(params['dec_channels'][2]),
            nn.ReLU(inplace=True)
        )

        # Final logits
        self.conv9 = nn.Conv2d(params['dec_channels'][2], num_classes, kernel_size=1)

        # Initialize only decoder weights
        self._init_decoder_weights()

    def _init_decoder_weights(self):
        # Initialize conv and bn in upconv6, upconv7, conv8, conv9 only
        layers = itertools.chain(
            self.upconv6.modules(),
            self.upconv7.modules(),
            self.conv8.modules(),
            [self.conv9]
        )
        for m in layers:
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        deep, skip = self.encoder(x)            # encoder return (x5, x1)
        x = self.upconv6(deep)
        x = self.upconv7(x)
        skip_p = self.skip_bn(self.skip_proj(skip))
        x = torch.cat([x, skip_p], dim=1)       # skip connection
        x = self.conv8(x)
        x = self.conv9(x)

        return x  # logits (B,num_classes,H,W)

