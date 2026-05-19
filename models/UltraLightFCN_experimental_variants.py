import torch
import torch.nn as nn

from models.UltraLightFCN_base import (
    DepthwiseSeparableConv,
    MiniASPP,
    WindowedShiftedSelfAttention,
)


class UltraLightFCNNoShallowSkip(nn.Module):
    """
    UltraLightFCN variant that removes the shallow skip path while preserving the
    encoder, bottleneck, and two-stage decoder layout.
    """

    def __init__(self, in_channels: int = 3, num_classes: int = 1, params: dict | None = None):
        super().__init__()
        self.num_classes = num_classes

        if params is None:
            raise ValueError("params must be provided for UltraLightFCNNoShallowSkip")

        self.params = params

        self.block1 = nn.Sequential(
            nn.Conv2d(
                in_channels,
                params["enc_channels"][0],
                kernel_size=params["enc_kernel_sizes"][0],
                stride=params["enc_strides"][0],
                padding=params["enc_kernel_sizes"][0] // 2,
                bias=False,
            ),
            nn.BatchNorm2d(params["enc_channels"][0]),
            nn.ReLU(inplace=True),
        )

        self.dsconv2 = DepthwiseSeparableConv(
            params["enc_channels"][0],
            params["enc_channels"][1],
            kernel_size=params["enc_kernel_sizes"][1],
            stride=params["enc_strides"][1],
            padding=params["enc_kernel_sizes"][1] // 2,
        )

        self.dsconv3 = DepthwiseSeparableConv(
            params["enc_channels"][1],
            params["enc_channels"][2],
            kernel_size=params["enc_kernel_sizes"][2],
            stride=params["enc_strides"][2],
            padding=params["enc_kernel_sizes"][2] // 2,
        )

        self.dilconv4 = nn.Sequential(
            nn.Conv2d(
                params["enc_channels"][2],
                params["enc_channels"][3],
                kernel_size=params["enc_kernel_sizes"][3],
                stride=1,
                padding=(params["enc_kernel_sizes"][3] // 2) * params["dilations"][0],
                dilation=params["dilations"][0],
                bias=False,
            ),
            nn.BatchNorm2d(params["enc_channels"][3]),
            nn.ReLU(inplace=True),
        )

        self.dilconv5 = nn.Sequential(
            nn.Conv2d(
                params["enc_channels"][3],
                params["enc_channels"][4],
                kernel_size=params["enc_kernel_sizes"][4],
                stride=1,
                padding=(params["enc_kernel_sizes"][4] // 2) * params["dilations"][1],
                dilation=params["dilations"][1],
                bias=False,
            ),
            nn.BatchNorm2d(params["enc_channels"][4]),
            nn.ReLU(inplace=True),
        )

        self.use_mini_aspp = params.get("mini_aspp", True)
        bottleneck_c = params["enc_channels"][4]
        if self.use_mini_aspp:
            self.mini_aspp = MiniASPP(
                bottleneck_c,
                bottleneck_c,
                use_gpool=params.get("mini_aspp_gpool", False),
            )

        self.use_sa = params.get("use_sa", True)
        if self.use_sa and params.get("sa_windowed", True):
            self.sa = WindowedShiftedSelfAttention(
                channels=bottleneck_c,
                num_heads=params.get("sa_heads", 4),
                window_size=params.get("sa_window_size", 8),
                shift=params.get("sa_shifted", True),
                mlp_ratio=4.0,
                attn_dropout=params.get("sa_dropout", 0.0),
                proj_dropout=params.get("sa_dropout", 0.0),
            )

        self.upconv6 = nn.Sequential(
            nn.Upsample(scale_factor=params["upscale"][0], mode="bilinear", align_corners=False),
            nn.Conv2d(
                bottleneck_c,
                params["dec_channels"][0],
                kernel_size=params["dec_kernel_sizes"][0],
                stride=params["dec_strides"][0],
                padding=params["dec_kernel_sizes"][0] // 2,
                bias=False,
            ),
            nn.BatchNorm2d(params["dec_channels"][0]),
            nn.ReLU(inplace=True),
        )

        self.upconv7 = nn.Sequential(
            nn.Upsample(scale_factor=params["upscale"][1], mode="bilinear", align_corners=False),
            nn.Conv2d(
                params["dec_channels"][0],
                params["dec_channels"][1],
                kernel_size=params["dec_kernel_sizes"][1],
                stride=params["dec_strides"][1],
                padding=params["dec_kernel_sizes"][1] // 2,
                bias=False,
            ),
            nn.BatchNorm2d(params["dec_channels"][1]),
            nn.ReLU(inplace=True),
        )

        # The base model concatenates a shallow skip here. This variant removes
        # that path and keeps the final decoder width unchanged by replacing the
        # fusion block with a same-resolution refinement layer.
        self.fuse8 = nn.Sequential(
            nn.Conv2d(
                params["dec_channels"][1],
                params["dec_channels"][2],
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(params["dec_channels"][2]),
            nn.ReLU(inplace=True),
        )

        self.conv9 = nn.Conv2d(params["dec_channels"][2], num_classes, kernel_size=1)

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.block1(x)
        x2 = self.dsconv2(x1)
        x3 = self.dsconv3(x2)
        x4 = self.dilconv4(x3)
        x5 = self.dilconv5(x4)

        if self.use_mini_aspp:
            x5 = self.mini_aspp(x5)
        if self.use_sa:
            x5 = self.sa(x5)

        x6 = self.upconv6(x5)
        x7 = self.upconv7(x6)
        x8 = self.fuse8(x7)
        return self.conv9(x8)
