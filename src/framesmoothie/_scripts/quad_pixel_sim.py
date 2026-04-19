"""
quad_pixel_sim.py
-----------------
Synthetic 8-channel quad-pixel simulator for framesmoothie QPAF pre-training.

Channel layout (matches plan):
  0: binned      = TL + TR + BL + BR  (full-sensitivity grayscale equivalent)
  1: h_phase     = (TL + BL) - (TR + BR)  (horizontal phase difference)
  2: v_phase     = (TL + TR) - (BL + BR)  (vertical phase difference)
  3: d1_phase    = TL - BR               (diagonal phase difference 1)
  4: d2_phase    = TR - BL               (diagonal phase difference 2)
  5: defocus_mag = sqrt(h_phase^2 + v_phase^2)
  6: defocus_dir = atan2(v_phase, h_phase)
  7: confidence  = local variance (high texture = high confidence)

Usage:
  from framesmoothie._scripts.quad_pixel_sim import QuadPixelSimulator
  sim = QuadPixelSimulator()
  eight_ch = sim(grayscale_image, depth_map)   # [8, H, W] float32
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


__all__ = ["QuadPixelSimulator", "build_depth_from_segments"]


class QuadPixelSimulator:
    """
    Converts a grayscale image + depth map into an 8-channel quad-pixel tensor.

    The physical model:
    - A single microlens covers a 2×2 pixel group.
    - Each sub-pixel (TL, TR, BL, BR) samples incoming light from a slightly
      different pupil position (±sub_shift pixels in the focal plane).
    - A defocus circle of radius r(d) = |d - d_focus| * sigma_scale blurs the
      image seen by each sub-pixel with a spatially varying Gaussian PSF.
    - Phase differences arise because defocused edges appear shifted between the
      left/right and top/bottom sub-pixel pairs.

    Args:
        sigma_scale: Converts defocus magnitude to Gaussian blur σ.
        sub_shift:   Sub-aperture offset in pixels (default 0.25 px).
        conf_kernel: Half-size of the local variance window for ch7.
        normalize:   If True, normalises each channel to [-1, 1].
    """

    def __init__(
        self,
        sigma_scale: float = 0.5,
        sub_shift: float = 0.25,
        conf_kernel: int = 3,
        normalize: bool = True,
    ) -> None:
        self.sigma_scale = sigma_scale
        self.sub_shift = sub_shift
        self.conf_kernel = conf_kernel
        self.normalize = normalize

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def __call__(
        self,
        gray: torch.Tensor,
        depth: torch.Tensor,
        d_focus: Optional[float] = None,
        rng: Optional[torch.Generator] = None,
    ) -> torch.Tensor:
        """
        Args:
            gray:    [H, W] or [1, H, W] float32, range [0, 1].
            depth:   [H, W] or [1, H, W] float32, range [0, 1].
                     Larger value = farther from camera.
            d_focus: Focus plane depth in [0, 1].  If None, drawn uniformly.
            rng:     Optional torch.Generator for reproducibility.

        Returns:
            Tensor [8, H, W] float32.
        """
        gray = gray.squeeze(0) if gray.ndim == 3 else gray
        depth = depth.squeeze(0) if depth.ndim == 3 else depth
        H, W = gray.shape

        if d_focus is None:
            d_focus = float(torch.rand(1, generator=rng).item())

        # --- sub-aperture images ---
        tl, tr, bl, br = self._make_sub_aperture(gray, depth, d_focus)

        # --- 8 channels ---
        binned = tl + tr + bl + br
        h_phase = (tl + bl) - (tr + br)
        v_phase = (tl + tr) - (bl + br)
        d1 = tl - br
        d2 = tr - bl
        defocus_mag = torch.sqrt(h_phase.pow(2) + v_phase.pow(2) + 1e-8)
        defocus_dir = torch.atan2(v_phase, h_phase)  # [-π, π]
        confidence = self._local_variance(binned, self.conf_kernel)

        out = torch.stack([
            binned, h_phase, v_phase, d1, d2,
            defocus_mag, defocus_dir, confidence,
        ], dim=0)  # [8, H, W]

        if self.normalize:
            out = self._channel_normalize(out)

        return out

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_sub_aperture(
        self,
        gray: torch.Tensor,
        depth: torch.Tensor,
        d_focus: float,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply spatially-varying defocus blur and then sample 4 sub-aperture
        positions by bilinear grid-sampling with ±sub_shift pixel offsets.
        """
        H, W = gray.shape
        defocus = (depth - d_focus).abs() * self.sigma_scale  # [H, W]

        # Apply spatially-varying Gaussian blur at a discrete set of sigma
        # levels and interpolate between neighbours.  We use up to 5 levels.
        blurred = _spatially_varying_blur(gray, defocus, levels=5)

        # Sub-aperture sampling: shift the blurred image by ±sub_shift pixels
        # using grid_sample (sub-pixel accurate).
        s = self.sub_shift
        tl = _shift_image(blurred, dy=-s, dx=-s)
        tr = _shift_image(blurred, dy=-s, dx=+s)
        bl = _shift_image(blurred, dy=+s, dx=-s)
        br = _shift_image(blurred, dy=+s, dx=+s)
        return tl, tr, bl, br

    @staticmethod
    def _local_variance(x: torch.Tensor, k: int) -> torch.Tensor:
        """Compute local variance in a (2k+1)×(2k+1) window."""
        x4 = x.unsqueeze(0).unsqueeze(0)   # [1,1,H,W]
        kernel_size = 2 * k + 1
        pad = k
        mean = F.avg_pool2d(x4, kernel_size, stride=1, padding=pad)
        mean_sq = F.avg_pool2d(x4.pow(2), kernel_size, stride=1, padding=pad)
        var = (mean_sq - mean.pow(2)).clamp(min=0.0).squeeze(0).squeeze(0)
        return var

    @staticmethod
    def _channel_normalize(out: torch.Tensor) -> torch.Tensor:
        """Normalise each channel independently to [-1, 1]."""
        C = out.shape[0]
        result = []
        for c in range(C):
            ch = out[c]
            lo, hi = ch.min(), ch.max()
            denom = (hi - lo).clamp(min=1e-8)
            result.append((ch - lo) / denom * 2.0 - 1.0)
        return torch.stack(result, dim=0)


# ---------------------------------------------------------------------------
# Utility: depth map from segmentation masks
# ---------------------------------------------------------------------------

def build_depth_from_segments(
    sem: torch.Tensor,
    inst_masks: Optional[torch.Tensor] = None,
    num_sem_classes: Optional[int] = None,
    noise_std: float = 0.05,
    rng: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """
    Build a pseudo-depth map from panoptic segmentation labels.

    Heuristic: "things" (instances) are typically closer than "stuff"
    background.  Each segment receives a random depth drawn from a
    class-conditioned prior; a small amount of noise is added.

    Args:
        sem:             [H, W] int64 semantic label map.
        inst_masks:      [N, H, W] bool/float instance masks.  If None, only
                         semantic heuristic is used.
        num_sem_classes: Total number of semantic classes.
        noise_std:       Gaussian noise standard deviation added per-pixel.
        rng:             Optional torch.Generator.

    Returns:
        depth: [H, W] float32 in [0, 1].  0 = closest, 1 = farthest.
    """
    H, W = sem.shape
    depth = torch.ones(H, W, dtype=torch.float32) * 0.8  # default: far

    if inst_masks is not None and inst_masks.numel() > 0:
        masks = inst_masks.float() if inst_masks.dtype != torch.float32 else inst_masks
        N = masks.shape[0]
        # Assign random depths to instances, biased toward near
        inst_depths = torch.rand(N, generator=rng) * 0.5  # [0, 0.5]
        for i in range(N):
            mask_i = masks[i] > 0.5
            depth[mask_i] = inst_depths[i]

    # Add per-pixel noise
    depth = depth + torch.randn(H, W, generator=rng) * noise_std
    return depth.clamp(0.0, 1.0)


# ---------------------------------------------------------------------------
# Low-level blur / shift helpers (module-level, no class state)
# ---------------------------------------------------------------------------

def _gaussian_kernel_1d(sigma: float, truncate: float = 3.0) -> torch.Tensor:
    """1-D Gaussian kernel tensor."""
    radius = max(1, int(math.ceil(truncate * sigma)))
    x = torch.arange(-radius, radius + 1, dtype=torch.float32)
    k = torch.exp(-0.5 * (x / max(sigma, 1e-6)).pow(2))
    return k / k.sum()


def _separable_blur(image: torch.Tensor, sigma: float) -> torch.Tensor:
    """Blur a [H, W] image with a separable Gaussian of the given sigma."""
    if sigma < 1e-3:
        return image
    k1d = _gaussian_kernel_1d(sigma).to(image.device)
    x = image.unsqueeze(0).unsqueeze(0)  # [1, 1, H, W]
    r = k1d.shape[0] // 2
    ky = k1d.view(1, 1, -1, 1)
    kx = k1d.view(1, 1, 1, -1)
    x = F.conv2d(F.pad(x, (0, 0, r, r), mode='reflect'), ky)
    x = F.conv2d(F.pad(x, (r, r, 0, 0), mode='reflect'), kx)
    return x.squeeze(0).squeeze(0)


def _spatially_varying_blur(
    image: torch.Tensor,
    sigma_map: torch.Tensor,
    levels: int = 5,
) -> torch.Tensor:
    """
    Approximate spatially-varying Gaussian blur.

    Blurs the image at `levels` discrete sigma values, then linearly
    interpolates between neighbours based on sigma_map.

    Args:
        image:     [H, W] float32.
        sigma_map: [H, W] float32, per-pixel blur sigma >= 0.
        levels:    Number of discrete blur levels.

    Returns:
        [H, W] float32 blurred image.
    """
    max_sigma = sigma_map.max().item()
    if max_sigma < 1e-3:
        return image.clone()

    sigmas = [i * max_sigma / (levels - 1) for i in range(levels)]
    blurred_levels = [_separable_blur(image, s) for s in sigmas]

    # Vectorised interpolation
    result = torch.zeros_like(image)
    sigma_norm = sigma_map / max(max_sigma, 1e-8)  # [0, 1]
    level_idx = (sigma_norm * (levels - 1)).clamp(0, levels - 1)
    lo_idx = level_idx.floor().long().clamp(0, levels - 2)
    hi_idx = (lo_idx + 1).clamp(0, levels - 1)
    frac = level_idx - lo_idx.float()  # [H, W]

    blurred_stack = torch.stack(blurred_levels, dim=0)  # [levels, H, W]
    H, W = image.shape
    flat_lo = lo_idx.view(-1)
    flat_hi = hi_idx.view(-1)
    flat_frac = frac.view(-1)
    flat_stack = blurred_stack.view(levels, -1)  # [levels, H*W]
    hw = H * W
    idx = torch.arange(hw, device=image.device)
    lo_vals = flat_stack[flat_lo, idx]
    hi_vals = flat_stack[flat_hi, idx]
    result = (lo_vals * (1.0 - flat_frac) + hi_vals * flat_frac).view(H, W)
    return result


def _shift_image(
    image: torch.Tensor,
    dy: float,
    dx: float,
) -> torch.Tensor:
    """
    Sub-pixel shift of a [H, W] image using bilinear grid_sample.

    dy > 0 → shift down, dx > 0 → shift right.
    """
    H, W = image.shape
    # Build sampling grid: normalised coordinates in [-1, 1]
    # grid_sample expects [N, H, W, 2] with (x, y) = (col, row) ordering
    ys = torch.linspace(-1, 1, H, device=image.device)
    xs = torch.linspace(-1, 1, W, device=image.device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')  # [H, W]

    # Convert pixel shift to normalised offset
    dy_norm = dy * 2.0 / H
    dx_norm = dx * 2.0 / W
    grid_x = grid_x - dx_norm   # shift sampling origin → apparent image shift
    grid_y = grid_y - dy_norm

    grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0)  # [1,H,W,2]
    src = image.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
    out = F.grid_sample(src, grid, mode='bilinear', padding_mode='border', align_corners=True)
    return out.squeeze(0).squeeze(0)


# ---------------------------------------------------------------------------
# CLI for quick visualisation
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse
    from pathlib import Path

    ap = argparse.ArgumentParser(description='QuadPixelSimulator smoke-test')
    ap.add_argument('--height', type=int, default=64)
    ap.add_argument('--width', type=int, default=64)
    ap.add_argument('--sigma-scale', type=float, default=0.5)
    ap.add_argument('--sub-shift', type=float, default=0.25)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--visualize', action='store_true')
    args = ap.parse_args()

    rng = torch.Generator()
    rng.manual_seed(args.seed)

    # Synthetic grayscale + gradient depth
    gray = torch.rand(args.height, args.width, generator=rng)
    depth = torch.linspace(0, 1, args.width).unsqueeze(0).expand(args.height, -1)

    sim = QuadPixelSimulator(sigma_scale=args.sigma_scale, sub_shift=args.sub_shift)
    out = sim(gray, depth, rng=rng)

    ch_names = ['binned', 'h_phase', 'v_phase', 'd1_phase', 'd2_phase',
                'defocus_mag', 'defocus_dir', 'confidence']
    print(f'Output shape: {out.shape}')
    for i, name in enumerate(ch_names):
        ch = out[i]
        print(f'  ch{i} {name:15s}: min={ch.min():.3f}  max={ch.max():.3f}  std={ch.std():.3f}')

    if args.visualize:
        try:
            import matplotlib.pyplot as plt  # type: ignore
            fig, axes = plt.subplots(2, 4, figsize=(12, 6))
            for i, (ax, name) in enumerate(zip(axes.flat, ch_names)):
                ax.imshow(out[i].numpy(), cmap='bwr' if i > 0 else 'gray')
                ax.set_title(name)
                ax.axis('off')
            plt.tight_layout()
            plt.savefig('quad_pixel_sim_test.png', dpi=100)
            print('Saved quad_pixel_sim_test.png')
        except ImportError:
            print('matplotlib not available, skipping visualisation')
