import torch
import torch.nn.functional as F


@torch.no_grad()
def simclr_alignment(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    """Alignment: E[||z1 - z2||^2] over positive pairs. Lower is better."""
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    return ((z1 - z2).pow(2).sum(dim=1)).mean()


@torch.no_grad()
def simclr_uniformity(z: torch.Tensor, t: float = 2.0) -> torch.Tensor:
    """
    Uniformity: log E exp(-t ||zi - zj||^2), i!=j.
    Lower (more negative) is better.
    """
    z = F.normalize(z, dim=1)
    sim = z @ z.t()
    d2 = 2.0 - 2.0 * sim
    n = z.size(0)
    mask = ~torch.eye(n, device=z.device, dtype=torch.bool)
    v = -t * d2[mask]
    return torch.logsumexp(v, dim=0) - torch.log(torch.tensor(v.numel(), device=z.device, dtype=z.dtype))


def _gaussian_1d(window_size: int, sigma: float, device, dtype) -> torch.Tensor:
    """Create 1D Gaussian kernel normalized to sum=1."""
    coords = torch.arange(window_size, device=device, dtype=dtype) - (window_size - 1) / 2.0
    g = torch.exp(-(coords**2) / (2.0 * sigma**2))
    return g / g.sum()


def _gaussian_window_2d(window_size: int, sigma: float, channels: int, device, dtype) -> torch.Tensor:
    """
    Create 2D Gaussian window for SSIM as conv2d weights of shape [C,1,ws,ws].
    """
    g1 = _gaussian_1d(window_size, sigma, device=device, dtype=dtype)
    g2 = torch.outer(g1, g1)
    g2 = g2 / g2.sum()
    w = g2.view(1, 1, window_size, window_size)
    return w.repeat(channels, 1, 1, 1)  # [C,1,ws,ws] for groups=channels


@torch.no_grad()
def batch_ssim_windowed(
    x: torch.Tensor,
    y: torch.Tensor,
    data_range: float = 1.0,
    window_size: int = 11,
    sigma: float = 1.5,
    k1: float = 0.01,
    k2: float = 0.03,
) -> torch.Tensor:
    """
    Windowed (local) SSIM over a batch using a Gaussian window.

    Standard settings (commonly used in SSIM literature):
      window_size=11, sigma=1.5, k1=0.01, k2=0.03

    Args:
        x,y: [B,C,H,W] expected in [0, data_range]
        data_range: usually 1.0 if images are in [0,1]
    Returns:
        Mean SSIM over batch, channels and spatial dims (scalar).
    """
    if x.size(1) > 3:
        x = x[:, :3]
        y = y[:, :3]

    x = x.float()
    y = y.float()

    B, C, H, W = x.shape
    if window_size % 2 == 0:
        raise ValueError("window_size must be odd.")

    window = _gaussian_window_2d(window_size, sigma, channels=C, device=x.device, dtype=x.dtype)

    pad = window_size // 2
    x_pad = F.pad(x, (pad, pad, pad, pad), mode="reflect")
    y_pad = F.pad(y, (pad, pad, pad, pad), mode="reflect")

    mu_x = F.conv2d(x_pad, window, groups=C)
    mu_y = F.conv2d(y_pad, window, groups=C)

    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y

    sigma_x2 = F.conv2d(x_pad * x_pad, window, groups=C) - mu_x2
    sigma_y2 = F.conv2d(y_pad * y_pad, window, groups=C) - mu_y2
    sigma_xy = F.conv2d(x_pad * y_pad, window, groups=C) - mu_xy

    c1 = (k1 * data_range) ** 2
    c2 = (k2 * data_range) ** 2

    num = (2.0 * mu_xy + c1) * (2.0 * sigma_xy + c2)
    den = (mu_x2 + mu_y2 + c1) * (sigma_x2 + sigma_y2 + c2)

    ssim_map = num / (den + 1e-12)
    return ssim_map.mean()
