from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from framesmoothie.model import S9Stack
from s9.transforms.dost import DOST, IDOST


def make_s9_transform_pair(spatial_dims: int) -> tuple[nn.Module, nn.Module]:
    if spatial_dims <= 0:
        raise ValueError("spatial_dims must be positive")
    return DOST(D=spatial_dims), IDOST(D=spatial_dims)


def infer_encoder_channels(
    *,
    transform_fwd: nn.Module,
    input_channels: int,
    spatial_shape: tuple[int, ...],
) -> int:
    x = torch.zeros((1, input_channels, *spatial_shape), dtype=torch.float32)
    with torch.no_grad():
        z = transform_fwd(x)
    return int(z.shape[1])


def make_small_s9_encoder(*, c_model: int, spatial_dims: int, depth: int = 1, dtype_idx: int = 64) -> nn.Module:
    return S9Stack(
        c_model=c_model,
        spatial_dims=spatial_dims,
        depth=depth,
        eps=1e-6,
        dtype_idx=dtype_idx,
    )


def _draw_circle(h: int, w: int, cy: int, cx: int, r: int) -> torch.Tensor:
    yy, xx = torch.meshgrid(torch.arange(h), torch.arange(w), indexing='ij')
    return ((yy - cy) ** 2 + (xx - cx) ** 2) <= (r ** 2)


def _draw_box(h: int, w: int, y0: int, x0: int, y1: int, x1: int) -> torch.Tensor:
    m = torch.zeros((h, w), dtype=torch.bool)
    m[y0:y1, x0:x1] = True
    return m


def make_toy_panoptic_batch(
    *,
    batch_size: int = 2,
    channels: int = 8,
    height: int = 32,
    width: int = 32,
    num_semantic_classes: int = 4,
    num_instance_classes: int = 2,
    style_shift: bool = False,
) -> dict[str, Any]:
    images = []
    sem_labels = []
    inst_targets = []

    for b in range(batch_size):
        x = torch.randn(channels, height, width) * 0.05
        sem = torch.zeros((height, width), dtype=torch.long)

        masks = []
        labels = []

        # thing 0: circle
        circle = _draw_circle(height, width, cy=height // 3, cx=width // 3, r=max(2, min(height, width) // 8))
        if circle.any():
            sem[circle] = 1
            masks.append(circle)
            labels.append(torch.tensor(0, dtype=torch.long))

        # thing 1: box
        box = _draw_box(height, width, height // 2, width // 2, min(height, height // 2 + height // 4), min(width, width // 2 + width // 4))
        if box.any():
            sem[box] = 2
            masks.append(box)
            labels.append(torch.tensor(min(1, num_instance_classes - 1), dtype=torch.long))

        # stuff class
        sem[(sem == 0)] = min(3, num_semantic_classes - 1)

        # paint channels
        x[0][circle] += 1.0
        x[1][box] += 1.0
        x[2][sem == min(3, num_semantic_classes - 1)] += 0.25

        if style_shift:
            x = 0.8 * x + 0.1 * torch.tanh(2.0 * x)

        if masks:
            masks_tensor = torch.stack([m.to(dtype=torch.float32) for m in masks], dim=0)
            labels_tensor = torch.stack(labels)
        else:
            masks_tensor = torch.zeros((0, height, width), dtype=torch.float32)
            labels_tensor = torch.zeros((0,), dtype=torch.long)

        images.append(x)
        sem_labels.append(sem)
        inst_targets.append({"labels": labels_tensor, "masks": masks_tensor})

    return {
        "image": torch.stack(images, dim=0),
        "sem_labels": torch.stack(sem_labels, dim=0),
        "inst_targets": inst_targets,
    }
