"""
ffn_backends.py
---------------
Pluggable FFN (Feed-Forward Network) backends for framesmoothie decoder blocks.

The existing ``RS9CondMixBlock`` and ``RS9DecoderLayer`` use a fixed
``Linear → HGLU → Dropout → Linear → Dropout`` FFN. This module provides
drop-in replacements that share the same ``(in_dim, out_dim)`` interface
and expose ``wrappable_linears()`` for LRCA adapter wrapping.

Available backends:

- :class:`StandardFFN` — current default (Linear→HGLU→Linear)
- :class:`SparseMoEFFN` — sparse Mixture-of-Experts with top-k routing
- :class:`KAFFFN` — Kolmogorov-Arnold-Fourier Network (arXiv:2502.06018)
- :class:`BilinearFFN` — bilinear MLP (u, v branches → elementwise multiply)

Usage::

    from framesmoothie.ffn_backends import create_ffn
    ffn = create_ffn("sparse_moe", in_dim=64, out_dim=64, hidden_mult=4,
                     num_experts=8, top_k=2, dropout=0.1)
"""
from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from s9.activations.real.hglu import HGLU
from s9.base import FPDTypeIdx, get_float_dtype


# ================================================================== #
# Base protocol                                                       #
# ================================================================== #

class FFNBase(nn.Module):
    """Base class for FFN backends.

    Subclasses must implement :meth:`forward` and :meth:`wrappable_linears`.
    """

    def wrappable_linears(self) -> List[Tuple[str, nn.Linear]]:
        """Return ``(attr_path, module)`` pairs for LRCA wrapping.

        The caller uses these to replace the Linear modules in-place with
        ``adapter.wrap_linear(module, task=...)``. The ``attr_path`` must
        be usable with ``setattr(parent, name, wrapped)``.
        """
        raise NotImplementedError


# ================================================================== #
# Standard FFN (current default)                                      #
# ================================================================== #

class StandardFFN(FFNBase):
    """Linear → HGLU → Dropout → Linear → Dropout.

    This is the existing FFN pattern used in ``RS9CondMixBlock``.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_mult: int = 4,
        dropout: float = 0.0,
        dtype_idx: FPDTypeIdx = 64,
    ) -> None:
        super().__init__()
        dtype = get_float_dtype(dtype_idx)
        hidden = in_dim * hidden_mult
        self.proj_up = nn.Linear(in_dim, hidden, dtype=dtype)
        self.act = HGLU(4.0)
        self.drop1 = nn.Dropout(dropout)
        self.proj_down = nn.Linear(hidden, out_dim, dtype=dtype)
        self.drop2 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop2(self.proj_down(self.drop1(self.act(self.proj_up(x)))))

    def wrappable_linears(self) -> List[Tuple[str, nn.Linear]]:
        return [("proj_up", self.proj_up), ("proj_down", self.proj_down)]


# ================================================================== #
# Sparse MoE FFN                                                      #
# ================================================================== #

class SparseMoEFFN(FFNBase):
    """Sparse Mixture-of-Experts FFN with explicit sparsity control.

    Each token is routed to ``top_k`` out of ``num_experts`` expert FFNs.

    Sparsity = 1 − top_k / num_experts.

    Args:
        num_experts: Total number of expert networks.
        top_k: Number of active experts per token.
        capacity_factor: Expert capacity buffer (1.0 = no extra capacity).
        load_balance_weight: Auxiliary load-balancing loss weight.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_mult: int = 4,
        dropout: float = 0.0,
        dtype_idx: FPDTypeIdx = 64,
        num_experts: int = 8,
        top_k: int = 2,
        capacity_factor: float = 1.25,
        load_balance_weight: float = 0.01,
    ) -> None:
        super().__init__()
        assert 1 <= top_k <= num_experts
        dtype = get_float_dtype(dtype_idx)
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.load_balance_weight = load_balance_weight
        self.sparsity = 1.0 - top_k / num_experts

        hidden = in_dim * hidden_mult
        self._out_dim = out_dim

        # Router: produces expert logits
        self.router = nn.Linear(in_dim, num_experts, dtype=dtype)

        # Shared proj_up / proj_down across experts (for LRCA wrapping);
        # per-expert specialisation via intermediate weights.
        self.expert_up = nn.ModuleList([
            nn.Linear(in_dim, hidden, dtype=dtype) for _ in range(num_experts)
        ])
        self.expert_down = nn.ModuleList([
            nn.Linear(hidden, out_dim, dtype=dtype) for _ in range(num_experts)
        ])
        self.act = HGLU(4.0)
        self.drop = nn.Dropout(dropout)

        # For LRCA: expose the first expert's linears as canonical wrapping targets.
        # All experts share the same adaptation semantics.
        self.proj_up = self.expert_up[0]
        self.proj_down = self.expert_down[0]

        self._aux_loss = torch.tensor(0.0, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., in_dim]
        orig_shape = x.shape
        flat = x.reshape(-1, orig_shape[-1])  # [T, in_dim]
        T = flat.shape[0]

        logits = self.router(flat)  # [T, E]
        probs = F.softmax(logits, dim=-1)
        top_vals, top_idx = probs.topk(self.top_k, dim=-1)  # [T, K]
        top_vals = top_vals / top_vals.sum(dim=-1, keepdim=True)  # renormalise

        # Dispatch
        out = torch.zeros(T, self._out_dim,
                          device=x.device, dtype=x.dtype)
        for k in range(self.top_k):
            indices = top_idx[:, k]  # [T]
            weights = top_vals[:, k:k + 1]  # [T, 1]
            for e in range(self.num_experts):
                mask = (indices == e)
                if not mask.any():
                    continue
                h = self.act(self.expert_up[e](flat[mask]))
                h = self.drop(self.expert_down[e](h))
                out[mask] = out[mask] + weights[mask] * h

        # Load-balancing auxiliary loss (simplified)
        density = probs.mean(dim=0)  # [E]
        uniform = torch.ones_like(density) / self.num_experts
        self._aux_loss = self.load_balance_weight * F.mse_loss(density, uniform)

        return out.reshape(*orig_shape[:-1], -1)

    @property
    def aux_loss(self) -> torch.Tensor:
        return self._aux_loss

    def wrappable_linears(self) -> List[Tuple[str, nn.Linear]]:
        # Expert layers cannot be wrapped by LRCA: SparseMoEFFN dispatches
        # masked token subsets to individual experts, causing a batch dimension
        # mismatch with LRCA's context-based gating.
        # MoE routing itself provides task-specific specialization.
        return []


# ================================================================== #
# KAF (Kolmogorov-Arnold-Fourier) FFN                                 #
# ================================================================== #

class KAFFFN(FFNBase):
    """Kolmogorov-Arnold-Fourier Network.

    Reference: https://doi.org/10.48550/arXiv.2502.06018

    Core ideas:
      - Learnable Random Fourier Features (RFF) with adaptive init
      - Hybrid GELU-Fourier activation that progressively enhances
        frequency representation during training
      - Parameter-efficient alternative to KAN (no dual-matrix explosion)

    Note: KAF ≠ FourierKAN. They are distinct methods.

    Args:
        num_frequencies: Number of learnable Fourier feature pairs
            (each produces ``cos`` and ``sin`` → 2× features).
        rff_scale: Initial scale for the RFF frequency matrix.
        fourier_mix: Mixing ratio for the hybrid GELU-Fourier activation
            (0 = pure GELU, 1 = pure Fourier; default 0.5 = equal mix).
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_mult: int = 4,
        dropout: float = 0.0,
        dtype_idx: FPDTypeIdx = 64,
        num_frequencies: int = 32,
        rff_scale: float = 1.0,
        fourier_mix: float = 0.5,
    ) -> None:
        super().__init__()
        dtype = get_float_dtype(dtype_idx)
        hidden = in_dim * hidden_mult

        # Learnable RFF: ω ∈ R^{in_dim × F}, φ ∈ R^{F}
        self.omega = nn.Parameter(
            torch.randn(in_dim, num_frequencies, dtype=dtype) * rff_scale
        )
        self.phi = nn.Parameter(torch.zeros(num_frequencies, dtype=dtype))

        # Project RFF features (cos + sin → 2F) to hidden
        rff_dim = num_frequencies * 2
        self.proj_up = nn.Linear(rff_dim, hidden, dtype=dtype)

        # Hybrid GELU-Fourier activation
        self.fourier_mix = nn.Parameter(torch.tensor(fourier_mix, dtype=dtype))
        # Learnable Fourier modulation frequencies for the activation
        self.act_freq = nn.Parameter(torch.randn(hidden, dtype=dtype) * 0.1)

        self.drop = nn.Dropout(dropout)
        self.proj_down = nn.Linear(hidden, out_dim, dtype=dtype)

    def _hybrid_activation(self, x: torch.Tensor) -> torch.Tensor:
        """Hybrid GELU-Fourier activation: mix GELU and sin modulation."""
        gelu_part = F.gelu(x)
        fourier_part = x * torch.sin(self.act_freq * x)
        mix = self.fourier_mix.sigmoid()
        return (1 - mix) * gelu_part + mix * fourier_part

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., in_dim]
        # 1) Learnable RFF: z = x @ ω + φ → [cos(z), sin(z)]
        z = x @ self.omega + self.phi  # [..., F]
        rff = torch.cat([torch.cos(z), torch.sin(z)], dim=-1)  # [..., 2F]

        # 2) Project to hidden + hybrid activation
        h = self.proj_up(rff)
        h = self._hybrid_activation(h)
        h = self.drop(h)

        # 3) Project to output
        return self.proj_down(h)

    def wrappable_linears(self) -> List[Tuple[str, nn.Linear]]:
        return [("proj_up", self.proj_up), ("proj_down", self.proj_down)]


# ================================================================== #
# Bilinear FFN                                                        #
# ================================================================== #

class BilinearFFN(FFNBase):
    """Bilinear MLP: two branches → elementwise multiply → project.

    ``y = proj_down(act(proj_u(x)) ⊙ proj_v(x))``

    This is a well-known pattern (e.g. GLU variants, SwiGLU, etc.) that
    provides richer feature interaction than a single-branch FFN at
    comparable parameter cost.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_mult: int = 4,
        dropout: float = 0.0,
        dtype_idx: FPDTypeIdx = 64,
    ) -> None:
        super().__init__()
        dtype = get_float_dtype(dtype_idx)
        hidden = in_dim * hidden_mult
        self.proj_u = nn.Linear(in_dim, hidden, dtype=dtype)
        self.proj_v = nn.Linear(in_dim, hidden, dtype=dtype)
        self.act = HGLU(4.0)
        self.drop = nn.Dropout(dropout)
        self.proj_down = nn.Linear(hidden, out_dim, dtype=dtype)

        # LRCA canonical targets
        self.proj_up = self.proj_u  # alias for wrappable_linears convention

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = self.act(self.proj_u(x))
        v = self.proj_v(x)
        h = u * v
        h = self.drop(h)
        return self.proj_down(h)

    def wrappable_linears(self) -> List[Tuple[str, nn.Linear]]:
        return [
            ("proj_u", self.proj_u),
            ("proj_v", self.proj_v),
            ("proj_down", self.proj_down),
        ]


# ================================================================== #
# Factory                                                             #
# ================================================================== #

FFN_REGISTRY: Dict[str, type] = {
    "standard": StandardFFN,
    "sparse_moe": SparseMoEFFN,
    "kaf": KAFFFN,
    "bilinear": BilinearFFN,
}


def create_ffn(
    name: str,
    in_dim: int,
    out_dim: int,
    hidden_mult: int = 4,
    dropout: float = 0.0,
    dtype_idx: FPDTypeIdx = 64,
    **kwargs: Any,
) -> FFNBase:
    """Factory for FFN backends.

    Args:
        name: One of ``"standard"``, ``"sparse_moe"``, ``"kaf"``, ``"bilinear"``.
        in_dim, out_dim: Input and output feature dimensions.
        hidden_mult: Hidden-layer expansion factor.
        dropout: Dropout probability.
        dtype_idx: Float precision index (32/64/128).
        **kwargs: Backend-specific arguments (e.g. ``num_experts``, ``top_k``
            for sparse_moe; ``num_frequencies`` for kaf).

    Returns:
        An :class:`FFNBase` instance.
    """
    if name not in FFN_REGISTRY:
        raise ValueError(f"Unknown FFN backend '{name}'. Available: {list(FFN_REGISTRY.keys())}")
    cls = FFN_REGISTRY[name]
    return cls(
        in_dim=in_dim,
        out_dim=out_dim,
        hidden_mult=hidden_mult,
        dropout=dropout,
        dtype_idx=dtype_idx,
        **kwargs,
    )


__all__ = [
    "FFNBase",
    "StandardFFN",
    "SparseMoEFFN",
    "KAFFFN",
    "BilinearFFN",
    "create_ffn",
    "FFN_REGISTRY",
]
