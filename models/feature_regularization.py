"""Spatial regularizers for patch-level feature maps."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AnisotropicDiffusion(nn.Module):
    """Perona--Malik diffusion on a four-neighbour patch grid.

    Conductance is computed from L2-normalized features so that ``sigma`` has
    a stable meaning while feature magnitudes change during training. The
    original, unnormalized features are diffused to preserve their scale.
    """

    def __init__(self, num_steps=1, tau=0.2, sigma=1.0, eps=1e-6):
        super().__init__()
        if num_steps < 1:
            raise ValueError("num_steps must be at least 1")
        if not 0.0 <= tau <= 0.25:
            raise ValueError("tau must be in [0, 0.25] for stable 4-neighbour diffusion")
        if sigma <= 0.0:
            raise ValueError("sigma must be positive")
        self.num_steps = int(num_steps)
        self.tau = float(tau)
        self.sigma = float(sigma)
        self.eps = float(eps)

    def _pair_flux(self, x, guidance, dim):
        left = x.narrow(dim, 0, x.shape[dim] - 1)
        right = x.narrow(dim, 1, x.shape[dim] - 1)
        guide_left = guidance.narrow(dim, 0, guidance.shape[dim] - 1)
        guide_right = guidance.narrow(dim, 1, guidance.shape[dim] - 1)
        distance_sq = (guide_right - guide_left).square().sum(dim=1, keepdim=True)
        conductance = torch.exp(-distance_sq / (self.sigma ** 2))
        return conductance * (right - left)

    def forward(self, x):
        if x.ndim != 4:
            raise ValueError(f"expected a BCHW feature map, got shape {tuple(x.shape)}")
        if x.shape[-2] == 0 or x.shape[-1] == 0:
            raise ValueError("feature map must have non-empty spatial dimensions")

        for _ in range(self.num_steps):
            guidance = F.normalize(x, p=2, dim=1, eps=self.eps)
            update = torch.zeros_like(x)
            if x.shape[-1] > 1:
                flux = self._pair_flux(x, guidance, dim=3)
                update[:, :, :, :-1] += flux
                update[:, :, :, 1:] -= flux
            if x.shape[-2] > 1:
                flux = self._pair_flux(x, guidance, dim=2)
                update[:, :, :-1, :] += flux
                update[:, :, 1:, :] -= flux
            x = x + self.tau * update
        return x


class TotalVariation(nn.Module):
    """Anisotropic L1 total variation for BCHW patch feature maps."""

    def __init__(self, reduction="mean"):
        super().__init__()
        if reduction not in {"mean", "sum"}:
            raise ValueError("reduction must be either 'mean' or 'sum'")
        self.reduction = reduction

    def forward(self, x):
        if x.ndim != 4:
            raise ValueError(f"expected a BCHW feature map, got shape {tuple(x.shape)}")
        differences = []
        if x.shape[-2] > 1:
            differences.append(x[:, :, 1:, :] - x[:, :, :-1, :])
        if x.shape[-1] > 1:
            differences.append(x[:, :, :, 1:] - x[:, :, :, :-1])
        if not differences:
            return x.sum() * 0.0
        if self.reduction == "sum":
            return sum(diff.abs().sum() for diff in differences)
        return sum(diff.abs().mean() for diff in differences)
