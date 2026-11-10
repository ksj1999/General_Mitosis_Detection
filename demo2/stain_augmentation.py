"""HED-space stain augmentation shared with the teammate patch workflow."""

import numpy as np
import torch


STAIN_MODES = ("both", "he_only", "scale_only")
RGB_FROM_HED = torch.tensor(
    [[0.65, 0.70, 0.29], [0.07, 0.99, 0.11], [0.27, 0.57, 0.78]],
    dtype=torch.float32,
)
HED_FROM_RGB = torch.linalg.inv(RGB_FROM_HED)
LOG_ADJ = float(np.log(1e-6))


def hed_jitter(images, sigma, mode="both"):
    """Perturb HED stain channels for an RGB BCHW tensor in the [0, 1] range."""
    if mode not in STAIN_MODES:
        raise ValueError(f"Unknown stain mode {mode!r}; choose from {STAIN_MODES}")
    batch_size = images.shape[0]
    rgb = images.clamp_min(1e-6).permute(0, 2, 3, 1)
    hed = (torch.log(rgb) / LOG_ADJ) @ HED_FROM_RGB.to(images.device)
    hed = hed.clamp_min(0)
    scale = torch.empty(batch_size, 1, 1, 3, device=images.device).uniform_(
        1 - sigma, 1 + sigma
    )
    bias = torch.empty(batch_size, 1, 1, 3, device=images.device).uniform_(-sigma, sigma)
    if mode == "he_only":
        scale[..., 2] = 1.0
        bias[..., 2] = 0.0
    elif mode == "scale_only":
        bias.zero_()
    hed = hed * scale + bias
    rgb = torch.exp(-(hed * -LOG_ADJ) @ RGB_FROM_HED.to(images.device))
    return rgb.clamp(0, 1).permute(0, 3, 1, 2).contiguous()
