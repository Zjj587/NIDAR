"""Explicit baseline intrinsic objective, adapted from NIDAR's training losses."""
import math

import torch
import torch.nn.functional as F


DEFAULT_WEIGHTS = {"reconstruction": 1.0, "reflectance_smooth": 0.1,
                   "shading_smooth": 0.05, "reflectance_mean": 0.1, "intensity": 1.0}


def intrinsic_loss(x, r, s, intensity, target, mask, weights, edge_weight=10.0,
                   target_mean=0.5):
    if set(weights) != set(DEFAULT_WEIGHTS):
        raise ValueError("Specify exactly the five documented loss weights")
    if any(not math.isfinite(v) or v < 0 for v in weights.values()):
        raise ValueError("Loss weights must be finite and nonnegative")
    if weights["reconstruction"] <= 0 or weights["intensity"] <= 0:
        raise ValueError("The public recipe requires reconstruction and intensity supervision")
    if not math.isfinite(edge_weight) or edge_weight < 0 or not 0 <= target_mean <= 1:
        raise ValueError("Expected nonnegative finite edge_weight and target_mean in [0, 1]")
    if not all(t.shape == x.shape for t in (r, s, intensity, target, mask)):
        raise ValueError("All loss inputs must have the same shape")
    if mask.dtype != torch.bool or not mask.any():
        raise ValueError("Sparse intensity loss requires a nonempty boolean mask")
    dx = lambda t: t[:, :, :, 1:] - t[:, :, :, :-1]
    dy = lambda t: t[:, :, 1:, :] - t[:, :, :-1, :]
    terms = {
        "reconstruction": F.l1_loss(r * s, x),
        "reflectance_smooth": (dx(r).abs() * torch.exp(-edge_weight * dx(x).abs())).mean()
                              + (dy(r).abs() * torch.exp(-edge_weight * dy(x).abs())).mean(),
        "shading_smooth": F.l1_loss(s, F.avg_pool2d(s, 5, stride=1, padding=2)),
        "reflectance_mean": (r.mean() - target_mean).abs(),
        "intensity": F.l1_loss(intensity[mask], target[mask]),
    }
    total = sum(weights[name] * value for name, value in terms.items())
    return total, terms
