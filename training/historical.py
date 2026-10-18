"""Recovered dense-target recipe; explicit compatibility, not exact run replay.

The retained team implementation references LiDARRT for Gaussian SSIM and
PyRResNet for edge-aware smoothness. Optional perceptual/region objectives are
not part of the saved active configuration. See docs/TRAINING.md for boundaries.
"""
import math

import cv2
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

RECIPE = "historical-dense-v1"
WEIGHTS = {"reconstruction": 1.0, "reflectance_smooth": 0.01,
           "shading_smooth": 0.0, "shading_tv": 0.0,
           "reflectance_mean": 0.01, "intensity": 1.0}


def dense_target(points, intrinsic, extrinsic, world_from_reference, shape):
    """Preserve retained target arithmetic, including its input-X collision order."""
    from src.projection.projection_utils import project_points_to_image

    if points.ndim != 2 or points.shape[1] != 4 or not np.isfinite(points).all():
        raise ValueError("Expected finite XYZ/intensity points")
    if np.any(points[:, 3] < 0):
        raise ValueError("Expected nonnegative intensity")
    h, w = shape
    visible, uv, indices = project_points_to_image(points, intrinsic, extrinsic, w, h,
                                                   world_from_reference)
    values = np.zeros((h, w), np.float32)
    support = np.zeros((h, w), np.float32)
    if not len(indices):
        return values.astype(np.uint8), support.astype(bool)
    u = np.clip(uv[:, 0].astype(np.int32), 0, w - 1)
    v = np.clip(uv[:, 1].astype(np.int32), 0, h - 1)
    # The old projector returns input coordinates, not camera coordinates.
    # Do not silently change this to a camera-depth z-buffer in compatibility mode.
    for index in np.argsort(visible[:, 0])[::-1]:
        values[v[index], u[index]] = visible[index, 3]
        support[v[index], u[index]] = 1.0
    measured_support = support.astype(bool)
    if values.max() > 0:
        samples = values[measured_support]
        low = np.percentile(samples, 1)
        high = np.percentile(samples, 99)
        if high > low:
            values = (values - low) / (high - low) * 255
        else:
            values = values / (values.max() + 1e-6) * 255
    values = np.clip(values, 0, 255)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    values = cv2.GaussianBlur(cv2.dilate(values, kernel), (0, 0), 2.0)
    support = cv2.GaussianBlur(cv2.dilate(support, kernel), (0, 0), 2.0)
    result = np.zeros_like(values)
    active = support > 1e-6
    result[active] = values[active] / support[active]
    return result.astype(np.uint8), measured_support


def pair_transform(nir, target, shape):
    """PIL resize, target-derived row crop, uint8 round trip and full crop mask."""
    h, w = shape
    arrays = []
    for value in (nir, target):
        if value.dtype != np.uint8 or value.ndim != 2:
            raise ValueError("Historical pairs must be single-channel uint8 images")
        image = Image.fromarray(value).resize((w, h), Image.Resampling.BILINEAR
                                             if hasattr(Image, "Resampling") else Image.BILINEAR)
        arrays.append(np.array(image, dtype=np.float32) / 255.0)
    x, y = arrays
    mask = y > 0.02
    rows = np.any(mask, axis=1)
    crop = [0, h]
    if rows.any():
        first = int(np.argmax(rows))
        end = int(len(rows) - np.argmax(rows[::-1]))
        if end - first < h // 2:
            first = max(0, end - h // 2)
        crop = [first, end]
        resized = []
        for value in arrays:
            image = Image.fromarray((value[first:end] * 255).astype(np.uint8))
            image = image.resize((w, h), Image.Resampling.BILINEAR
                                 if hasattr(Image, "Resampling") else Image.BILINEAR)
            resized.append(np.array(image) / 255.0)
        x, y = resized
        mask = np.ones_like(y, dtype=bool)
    return x.astype(np.float32), y.astype(np.float32), mask, crop


def intensity_weight(epoch, maximum=1.0, warmup_epochs=1, rampup_epochs=1):
    if epoch < 0 or warmup_epochs < 0 or rampup_epochs < 0:
        raise ValueError("Epoch and warmup/ramp counts must be nonnegative")
    if epoch < warmup_epochs:
        return 0.0
    if epoch < warmup_epochs + rampup_epochs:
        return maximum * (epoch - warmup_epochs) / rampup_epochs
    return maximum


class DenseObjective(torch.nn.Module):
    """Active saved objective, with Gaussian SSIM and multi-scale masked L1."""
    def __init__(self, weights=None, sigma_color=0.01, target_mean=0.5,
                 intensity_components=(1.0, 0.5, 0.2), scales=(1.0, 0.5, 0.25),
                 warmup_epochs=1, rampup_epochs=1, epoch=0):
        super().__init__()
        self.weights = dict(WEIGHTS if weights is None else weights)
        if set(self.weights) != set(WEIGHTS) or any(
                not math.isfinite(v) or v < 0 for v in self.weights.values()):
            raise ValueError("Specify the six nonnegative historical loss weights")
        if self.weights['reconstruction'] <= 0 or self.weights['intensity'] <= 0:
            raise ValueError("Reconstruction and intensity supervision must be enabled")
        if not math.isfinite(sigma_color) or sigma_color <= 0 or not 0 <= target_mean <= 1:
            raise ValueError("Invalid smoothness or mean configuration")
        if len(intensity_components) != 3 or any(not math.isfinite(v) or v < 0 for v in intensity_components):
            raise ValueError("Expected nonnegative L1/L2/SSIM weights")
        if not scales or any(not math.isfinite(v) or not 0 < v <= 1 for v in scales):
            raise ValueError("Expected downsampling scales in (0, 1]")
        self.sigma_color, self.target_mean = sigma_color, target_mean
        self.components, self.scales = intensity_components, scales
        self.active_intensity_weight = intensity_weight(epoch, self.weights['intensity'],
                                                        warmup_epochs, rampup_epochs)
        coords = torch.arange(11, dtype=torch.float32) - 5
        g = torch.exp(-(coords ** 2) / (2 * 1.5 ** 2))
        g = g / g.sum()
        self.register_buffer('window', (g.unsqueeze(0) * g.unsqueeze(1))[None, None])

    def forward(self, x, r, s, pred, target, mask):
        if any(t.shape != x.shape for t in (r, s, pred, target, mask)) or x.ndim != 4 or x.shape[1] != 1:
            raise ValueError("Expected matching Bx1xHxW tensors")
        if mask.dtype != torch.bool or not mask.any():
            raise ValueError("Expected a nonempty boolean training-support mask")
        dx = lambda t: t[:, :, :, :-1] - t[:, :, :, 1:]
        dy = lambda t: t[:, :, :-1, :] - t[:, :, 1:, :]
        smooth = lambda t: F.l1_loss(t, F.avg_pool2d(t, 5, stride=1, padding=2))
        l1, shading = x.new_zeros(()), x.new_zeros(())
        for scale in self.scales:
            if scale == 1:
                p, y, valid, shading_map = pred, target, mask, s
            else:
                size = (int(x.shape[2] * scale), int(x.shape[3] * scale))
                if min(size) < 1:
                    raise ValueError("Input too small for configured scales")
                p, y, shading_map = [F.interpolate(v, size=size, mode='bilinear', align_corners=False)
                                     for v in (pred, target, s)]
                valid = F.interpolate(mask.float(), size=size, mode='nearest') > 0.5
            if valid.any():
                l1 = l1 + F.l1_loss(p[valid], y[valid])
            shading = shading + smooth(shading_map)
        l1 = l1 / len(self.scales)
        l2 = F.mse_loss(pred[mask], target[mask])
        conv = lambda value: F.conv2d(value, self.window, padding=5)
        mx, my = conv(pred), conv(target)
        vx, vy = conv(pred ** 2) - mx ** 2, conv(target ** 2) - my ** 2
        cov = conv(pred * target) - mx * my
        ssim_map = ((2 * mx * my + 0.01 ** 2) * (2 * cov + 0.03 ** 2)) / (
            (mx ** 2 + my ** 2 + 0.01 ** 2) * (vx + vy + 0.03 ** 2))
        ssim = 1 - (ssim_map * mask).sum() / (mask.sum() + 1e-8)
        terms = dict(reconstruction=F.l1_loss(r * s, x),
            reflectance_smooth=(torch.exp(-dx(x).abs() / self.sigma_color) * dx(r).abs()).mean()
                              + (torch.exp(-dy(x).abs() / self.sigma_color) * dy(r).abs()).mean(),
            shading_smooth=shading / len(self.scales), shading_tv=dx(s).abs().mean() + dy(s).abs().mean(),
            reflectance_mean=(r.mean() - self.target_mean).abs(),
            intensity=self.components[0] * l1 + self.components[1] * l2 + self.components[2] * ssim)
        total = sum((self.active_intensity_weight if name == 'intensity' else weight) * terms[name]
                    for name, weight in self.weights.items())
        terms.update(intensity_l1=l1, intensity_l2=l2, intensity_ssim=ssim)
        return total, terms
