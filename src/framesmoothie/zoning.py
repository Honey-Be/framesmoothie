from __future__ import annotations

import torch
import torch.nn as nn
from typing import Mapping

from s9.base import FPDTypeIdx, get_float_dtype
from s9.activations.real.hglu import HGLU

from framesmoothie.specs import PreBridgeMapSpec


def _make_linear(dtype: torch.dtype, in_features: int, out_features: int) -> nn.Linear:
    return nn.Linear(in_features, out_features, bias=False, dtype=dtype)


class PreBridgeProjector(nn.Module):
    """Project a complex pre-bridge tensor into a real feature tensor.

    Input:
      z_cf: [B, C, *S] complex (channel-first)
    Output:
      x_cf: [B, C_out, *S] real (channel-first)
    """

    def __init__(
        self,
        *,
        map_spec: PreBridgeMapSpec,
        in_channels: int,
        out_channels: int,
        spatial_dims: int,
        dtype_idx: FPDTypeIdx = 64,
    ):
        super().__init__()
        self.map_spec = map_spec
        self.in_channels = int(in_channels)
        self.out_channels = int(out_channels)
        self.spatial_dims = int(spatial_dims)
        self.dtype = get_float_dtype(dtype_idx)

        proj_in = self.in_channels * self.map_spec.num_components
        if self.spatial_dims == 1:
            self.proj = nn.Conv1d(proj_in, self.out_channels, kernel_size=1, bias=False, dtype=self.dtype)
        elif self.spatial_dims == 2:
            self.proj = nn.Conv2d(proj_in, self.out_channels, kernel_size=1, bias=False, dtype=self.dtype)
        elif self.spatial_dims == 3:
            self.proj = nn.Conv3d(proj_in, self.out_channels, kernel_size=1, bias=False, dtype=self.dtype)
        else:
            raise ValueError(f"Unsupported spatial_dims={self.spatial_dims}")

        self.act = HGLU(4.0)

    def forward(self, z_cf: torch.Tensor) -> torch.Tensor:
        if not torch.is_complex(z_cf):
            raise TypeError("PreBridgeProjector expects a complex tensor input.")
        if z_cf.ndim < 3:
            raise ValueError(f"Expected [B,C,*S] complex input, got shape={tuple(z_cf.shape)}")
        if z_cf.shape[1] != self.in_channels:
            raise ValueError(f"Expected input channels={self.in_channels}, got {z_cf.shape[1]}")

        comps = self.map_spec(z_cf)  # tuple of [B,C,*S] real
        x = torch.cat([c.to(dtype=self.dtype) for c in comps], dim=1)  # [B,m*C,*S]
        x = self.proj(x)
        x = self.act(x)
        return x


class _ZoneFuseUnit(nn.Module):
    """One zone fuse unit for one scale.

    Inputs/outputs are channel-last: [B,*S,C]
    """
    def __init__(self, *, c_in_pre: int, c_in_post: int, c_out: int, dtype: torch.dtype):
        super().__init__()
        self.pre_proj = _make_linear(dtype, c_in_pre, c_out)
        self.post_proj = _make_linear(dtype, c_in_post, c_out)
        self.gate = nn.Sequential(
            _make_linear(dtype, c_in_pre + c_in_post, c_out),
            HGLU(4.0),
            _make_linear(dtype, c_out, c_out),
            nn.Sigmoid(),
        )

    def forward(self, pre: torch.Tensor, post: torch.Tensor) -> torch.Tensor:
        h_pre = self.pre_proj(pre)
        h_post = self.post_proj(post)
        alpha = self.gate(torch.cat([pre, post], dim=-1))
        return alpha * h_pre + (1.0 - alpha) * h_post

def _weighted_zone_sum(
    zone_dict: dict[str, torch.Tensor],
    zone_names: tuple[str, ...],
    zone_weights: Mapping[str, float] | None = None,
) -> torch.Tensor:
    if len(zone_names) == 0:
        raise ValueError("zone_names must be non-empty")
    if zone_weights is None:
        return sum(zone_dict[z] for z in zone_names)

    weighted = None
    denom = 0.0
    for z in zone_names:
        w = float(zone_weights.get(z, 1.0))
        if w == 0.0:
            continue
        term = zone_dict[z] * w
        weighted = term if weighted is None else (weighted + term)
        denom += abs(w)
    if weighted is None or denom <= 0.0:
        raise ValueError("All zone weights are zero; at least one non-zero weight is required.")
    return weighted / denom

class DualViewZoningBlock(nn.Module):
    """Fuse pre/post pyramids into typed zone pyramids.

    - pre_pyr:  list of [B,*S,C_pre] real (from pre-bridge projector)
    - post_pyr: list of [B,*S,C_post] real
    Returns:
      {
        "zones": list[dict[str, Tensor]],   # zone pyramid per scale
        "sem_pyr": list[Tensor],            # semantic fused pyramid
        "inst_pyr": list[Tensor],           # instance fused pyramid
      }
    """
    def __init__(
        self,
        *,
        num_scales: int,
        c_pre: int,
        c_post: int,
        c_out: int,
        zone_names: tuple[str, ...] = ("content", "structure", "label", "boundary"),
        semantic_zone_names: tuple[str, ...] = ("content", "structure", "boundary"),
        instance_zone_names: tuple[str, ...] = ("content", "label", "boundary"),
        instance_zone_weights: Mapping[str, float] | None = None,
        semantic_zone_weights: Mapping[str, float] | None = None,
        dtype_idx: FPDTypeIdx = 64,
    ):
        super().__init__()
        self.dtype = get_float_dtype(dtype_idx)
        self.num_scales = int(num_scales)
        self.c_pre = int(c_pre)
        self.c_post = int(c_post)
        self.c_out = int(c_out)
        self.zone_names = tuple(zone_names)
        self.semantic_zone_names = tuple(semantic_zone_names)
        self.instance_zone_names = tuple(instance_zone_names)
        self.semantic_zone_weights = dict(semantic_zone_weights) if semantic_zone_weights is not None else None
        self.instance_zone_weights = dict(instance_zone_weights) if instance_zone_weights is not None else None

        self.units = nn.ModuleList()
        for _ in range(self.num_scales):
            scale_units = nn.ModuleDict({
                z: _ZoneFuseUnit(c_in_pre=self.c_pre, c_in_post=self.c_post, c_out=self.c_out, dtype=self.dtype)
                for z in self.zone_names
            })
            self.units.append(scale_units)

    def forward(self, pre_pyr: list[torch.Tensor], post_pyr: list[torch.Tensor]) -> dict[str, list]:
        if len(pre_pyr) != len(post_pyr):
            raise ValueError("pre_pyr and post_pyr must have the same number of scales")
        if len(pre_pyr) != self.num_scales:
            raise ValueError(f"Expected {self.num_scales} scales, got {len(pre_pyr)}")

        zone_pyr: list[dict[str, torch.Tensor]] = []
        sem_pyr: list[torch.Tensor] = []
        inst_pyr: list[torch.Tensor] = []

        for s in range(self.num_scales):
            pre = pre_pyr[s].to(dtype=self.dtype)
            post = post_pyr[s].to(dtype=self.dtype)
            zone_dict: dict[str, torch.Tensor] = {}
            for z in self.zone_names:
                zone_dict[z] = self.units[s][z](pre, post)
            zone_pyr.append(zone_dict)

            sem = _weighted_zone_sum(zone_dict, self.semantic_zone_names, self.semantic_zone_weights)
            inst = _weighted_zone_sum(zone_dict, self.instance_zone_names, self.instance_zone_weights)
            sem_pyr.append(sem)
            inst_pyr.append(inst)

        return {"zones": zone_pyr, "sem_pyr": sem_pyr, "inst_pyr": inst_pyr}
