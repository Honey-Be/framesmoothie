"""Pluggable Transformer-style encoder blocks for framesmoothie.

This module replaces the legacy ``S9Stack`` (which only stacked SSM
layers without norm/residual/FFN) with a full Transformer-style block:

    Block(z) = z + α₁ · SSM(Norm₁(z))
             → block_out + α₂ · FFN(Norm₂(block_out))

All three components are independently swappable:

* **norm**:  any factory ``Callable[[int], nn.Module]`` returning a
  channel-first complex norm. Recommended choices come from
  :mod:`ypsilon_torch.blocks.normalizations.complex` —
  ``ComplexLayerNorm``, ``ComplexRMSNorm``,
  ``ComplexRobustLayerNorm``, ``ComplexAsinhMeanLayerNorm``.

* **ssm**:  any factory ``Callable[[], nn.Module]`` returning a complex
  channel-first layer. Use :class:`s9.modules.S9Layer`,
  :class:`s9.biaffine_s9_modules.BiaffineS9Layer`,
  :class:`s9.contrib.gated_delta_s9_modules.GatedDeltaS9Layer`, etc.

* **ffn**:  any factory ``Callable[[int], nn.Module]`` taking ``c_model``
  and returning a complex channel-first FFN. Use
  :func:`framesmoothie.complex_ffn_backends.create_complex_ffn`.

Optional :class:`LayerScale`-style channel scalars (ViT-22B/CaiT) dampen
each branch's contribution to the residual stream — crucial for deep
stacks where raw SSM output norm grows ~30× per layer.

Norm registry helper :func:`get_norm_factory` provides one-liner access
to common norms by string name.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import torch
import torch.nn as nn
from torch import Tensor

from ypsilon_torch.blocks.normalizations.complex import (
    ComplexAsinhMeanLayerNorm,
    ComplexLayerNorm,
    ComplexRMSNorm,
    ComplexRobustLayerNorm,
)


# ------------------------------------------------------------------ #
# Norm registry                                                       #
# ------------------------------------------------------------------ #

NORM_REGISTRY: Dict[str, type] = {
    "complex_rms_norm":          ComplexRMSNorm,
    "complex_layer_norm":        ComplexLayerNorm,
    "complex_robust_layer_norm": ComplexRobustLayerNorm,
    "complex_asinh_mean_layer_norm": ComplexAsinhMeanLayerNorm,
}


def get_norm_factory(
    name: str,
    *,
    dtype_idx: int = 64,
    **kwargs: Any,
) -> Callable[[int], nn.Module]:
    """Return a one-arg factory ``c_model -> norm_module`` for the named norm.

    Parameters
    ----------
    name : str
        One of :data:`NORM_REGISTRY` keys.
    dtype_idx : int
        Precision selector. Default 64.
    **kwargs
        Forwarded to the norm constructor.
    """
    if name not in NORM_REGISTRY:
        raise ValueError(
            f"Unknown norm {name!r}. Available: {sorted(NORM_REGISTRY.keys())}"
        )
    cls = NORM_REGISTRY[name]

    def _factory(c_model: int) -> nn.Module:
        # All ypsilon-torch complex norms accept (normalized_shape, ...)
        # ComplexLayerNorm uses positional arg name `d_prime`; others use
        # `normalized_shape`. Both accept positional first arg, so this works.
        # ComplexLayerNorm hardcodes channel-dim=1 (channel-first 4D);
        # the others default to dim=-1 — override to dim=1 for our layout.
        ck: Dict[str, Any] = dict(kwargs)
        if cls is ComplexLayerNorm:
            return cls(c_model, dtype_idx=dtype_idx, **ck)
        ck.setdefault("dim", 1)
        return cls(c_model, dtype_idx=dtype_idx, **ck)

    return _factory


# ------------------------------------------------------------------ #
# Transformer block                                                   #
# ------------------------------------------------------------------ #

class TransformerStyleBlock[
    N: nn.Module,
    S: nn.Module,
    F: nn.Module,
](nn.Module):
    """Pre-norm Transformer block with pluggable norm/ssm/ffn.

    Generic over three component types:

    * ``N`` — norm module type (e.g. ``ComplexRMSNorm``)
    * ``S`` — SSM mixer module type (e.g. ``S9Layer``)
    * ``F`` — FFN module type (e.g. ``ComplexStandardFFN``)

    Pattern::

        z' = z + α₁ · SSM(Norm₁(z))
        z" = z' + α₂ · FFN(Norm₂(z'))

    All sub-modules are passed in pre-constructed; the block does not
    own their factory logic. This keeps the block fully composable.

    Shape contract
    --------------
    Input  : ``(B, C, *spatial)`` complex.
    Output : same shape, complex.

    Parameters
    ----------
    c_model : int
        Channel dimension (preserved through the block).
    spatial_dims : int
        Number of spatial axes (used for LayerScale broadcasting).
    norm1, norm2 : N
        Pre-norm modules for the SSM and FFN branches.
    ssm : S
        SSM mixer (e.g. :class:`s9.modules.S9Layer`). Must accept
        and return ``(B, C, *spatial)`` complex.
    ffn : F
        Channel-only complex FFN. Must accept and return
        ``(B, C, *spatial)`` complex.
    layer_scale : float | None
        Initial LayerScale value (per-channel learnable scalar).
        ``None`` disables LayerScale. Default ``0.1``.
    """

    norm1: N
    ssm: S
    norm2: N
    ffn: F

    def __init__(
        self,
        c_model: int,
        spatial_dims: int,
        *,
        norm1: N,
        ssm: S,
        norm2: N,
        ffn: F,
        layer_scale: Optional[float] = 0.1,
    ) -> None:
        super().__init__()
        self.c_model = int(c_model)
        self.spatial_dims = int(spatial_dims)
        self.norm1 = norm1
        self.ssm = ssm
        self.norm2 = norm2
        self.ffn = ffn

        if layer_scale is not None:
            self.alpha_ssm = nn.Parameter(torch.full((c_model,), float(layer_scale)))
            self.alpha_ffn = nn.Parameter(torch.full((c_model,), float(layer_scale)))
        else:
            self.register_parameter("alpha_ssm", None)
            self.register_parameter("alpha_ffn", None)

    def _scale_view(self, alpha: Optional[nn.Parameter]) -> Tensor | float:
        if alpha is None:
            return 1.0
        # broadcast over (B, C, *spatial): shape (1, C, 1, 1, ...)
        view = (1, -1) + (1,) * self.spatial_dims
        return alpha.view(view)

    def forward(self, z: Tensor) -> Tensor:
        # ----- SSM branch -----
        z_ssm = self.ssm(self.norm1(z))
        z = z + self._scale_view(self.alpha_ssm) * z_ssm
        # ----- FFN branch -----
        z_ffn = self.ffn(self.norm2(z))
        z = z + self._scale_view(self.alpha_ffn) * z_ffn
        return z


# ------------------------------------------------------------------ #
# Stack                                                               #
# ------------------------------------------------------------------ #

class TransformerStack[
    N: nn.Module,
    S: nn.Module,
    F: nn.Module,
](nn.Module):
    """Stack of ``depth`` :class:`TransformerStyleBlock` instances.

    Generic over the same three component types as
    :class:`TransformerStyleBlock` (``N``: norm, ``S``: ssm, ``F``: ffn).

    Each block is independently constructed via the supplied factories,
    so per-layer parameter sharing is **not** done (each layer gets fresh
    weights). For shared weights, instantiate once and wrap in a custom
    block.

    Parameters
    ----------
    depth : int
        Number of blocks.
    c_model : int
        Channel dimension.
    spatial_dims : int
        Number of spatial axes.
    ssm_factory : Callable[[], S]
        Returns a fresh SSM mixer. Must accept ``(B, c_model, *S)``
        complex and return same.
    ffn_factory : Callable[[int], F]
        Returns a fresh FFN given ``c_model``. Must accept ``(B, c_model,
        *S)`` complex and return same.
    norm_factory : Callable[[int], N]
        Returns a fresh complex norm given ``c_model``. Must accept and
        return ``(B, c_model, *S)`` complex (channel dim = 1).
    layer_scale : float | None
        LayerScale init. ``None`` disables. Default ``0.1``.
    """

    blocks: nn.ModuleList  # of TransformerStyleBlock[N, S, F]

    def __init__(
        self,
        depth: int,
        c_model: int,
        spatial_dims: int,
        *,
        ssm_factory: Callable[[], S],
        ffn_factory: Callable[[int], F],
        norm_factory: Callable[[int], N],
        layer_scale: Optional[float] = 0.1,
    ) -> None:
        super().__init__()
        self.depth = int(depth)
        self.c_model = int(c_model)
        self.spatial_dims = int(spatial_dims)

        self.blocks = nn.ModuleList([
            TransformerStyleBlock[N, S, F](
                c_model=c_model,
                spatial_dims=spatial_dims,
                norm1=norm_factory(c_model),
                ssm=ssm_factory(),
                norm2=norm_factory(c_model),
                ffn=ffn_factory(c_model),
                layer_scale=layer_scale,
            )
            for _ in range(depth)
        ])

    def forward(self, z: Tensor) -> Tensor:
        for block in self.blocks:
            z = block(z)
        return z


__all__ = [
    "TransformerStyleBlock",
    "TransformerStack",
    "NORM_REGISTRY",
    "get_norm_factory",
]
