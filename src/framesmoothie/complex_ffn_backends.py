"""Complex-domain FFN backends for use in TransformerStack encoder blocks.

The existing :mod:`framesmoothie.ffn_backends` operates on real tensors
``(..., d_in)`` and is used by the decoder. The encoder pipeline operates
on complex tensors in channel-first layout ``(B, C, *S)``. This module
provides equivalent FFN backends that:

- accept channel-first complex input/output
- share the same ``(in_dim, out_dim)`` factory signature
- expose ``wrappable_linears()`` (none currently — LRCA is real-domain)

Available backends:

* :class:`ComplexStandardFFN` — Linear → activation → Linear (mirror of real Standard)
* :class:`ComplexBilinearFFN` — two linear branches → elementwise multiply → project
* :class:`ComplexIdentityFFN` — pass-through (for ablations: SSM-only block)

Use :func:`create_complex_ffn` to construct by name.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Tuple

import torch
import torch.nn as nn
from torch import Tensor

import torch.nn.functional as F

from s9.base import FPDTypeIdx, get_complex_dtype, get_float_dtype
from ypsilon_torch.blocks.activations.complex import StableModReLU


class ComplexFFNBase(nn.Module):
    """Base for channel-first complex FFNs.

    Forward contract: ``(B, in_dim, *S) complex → (B, out_dim, *S) complex``.
    Spatial dimensions are pass-through (FFN is channel-only).
    """

    def wrappable_linears(self) -> List[Tuple[str, nn.Module]]:
        """No real-domain Linear layers to wrap. LRCA wrappers are
        implemented for real Linears only; complex parameter wrapping is
        out of scope for now."""
        return []


# ------------------------------------------------------------------ #
# Standard 2-layer complex FFN                                        #
# ------------------------------------------------------------------ #

class ComplexStandardFFN(ComplexFFNBase):
    """2-layer complex MLP (mirror of :class:`ComplexFFN` from h9).

    ``Z → W1·Z + b1 → activation → dropout → W2·h + b2``

    All weights complex; activation is complex-valued (default
    :class:`StableModReLU`).
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_mult: int = 4,
        dropout: float = 0.0,
        gen_activation: Callable[..., nn.Module] = StableModReLU,
        eps: float = 1e-6,
        dtype_idx: FPDTypeIdx = 64,
    ) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.hidden = int(in_dim * hidden_mult)
        cdtype = get_complex_dtype(dtype_idx)

        self.W_1 = nn.Parameter(torch.empty(self.hidden, self.in_dim, dtype=cdtype))
        self.b_1 = nn.Parameter(torch.empty(self.hidden, dtype=cdtype))
        self.W_2 = nn.Parameter(torch.empty(self.out_dim, self.hidden, dtype=cdtype))
        self.b_2 = nn.Parameter(torch.empty(self.out_dim, dtype=cdtype))

        self.activation = gen_activation(self.hidden, eps, dtype_idx)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.reset_parameters()

    def reset_parameters(self) -> None:
        std1 = 1.0 / math.sqrt(self.in_dim)
        std2 = 1.0 / math.sqrt(self.hidden)
        nn.init.normal_(self.W_1.real, mean=0.0, std=std1)
        nn.init.normal_(self.W_1.imag, mean=0.0, std=std1)
        nn.init.zeros_(self.b_1)
        nn.init.normal_(self.W_2.real, mean=0.0, std=std2)
        nn.init.normal_(self.W_2.imag, mean=0.0, std=std2)
        nn.init.zeros_(self.b_2)

    def forward(self, Z: Tensor) -> Tensor:
        # Z: (B, in_dim, *S) complex. Use einsum across leading dims.
        # We collapse spatial into a single "sequence" via flatten then unflatten.
        spatial = Z.shape[2:]
        B, C = Z.shape[0], Z.shape[1]
        Z_flat = Z.reshape(B, C, -1)  # (B, in_dim, T)

        # Channel mixing: W_1 @ Z (mix over input dim)
        h = torch.einsum("bcn,fc->bfn", Z_flat, self.W_1) + self.b_1.view(1, -1, 1)

        # Activation expects channel-last
        h_cl = h.movedim(1, -1)  # (B, T, hidden)
        h_cl = self.activation(h_cl)
        h = h_cl.movedim(-1, 1)  # back

        h = self.dropout(h)

        out = torch.einsum("bfn,cf->bcn", h, self.W_2) + self.b_2.view(1, -1, 1)
        return out.reshape(B, self.out_dim, *spatial)


# ------------------------------------------------------------------ #
# Complex Bilinear (GLU-style) FFN                                    #
# ------------------------------------------------------------------ #

class ComplexBilinearFFN(ComplexFFNBase):
    """Two complex linear branches → elementwise complex multiply → project.

    ``y = W_down · (act(W_u · Z) ⊙ (W_v · Z))``

    Element-wise complex multiplication preserves phase composition.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_mult: int = 4,
        dropout: float = 0.0,
        gen_activation: Callable[..., nn.Module] = StableModReLU,
        eps: float = 1e-6,
        dtype_idx: FPDTypeIdx = 64,
    ) -> None:
        super().__init__()
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.hidden = int(in_dim * hidden_mult)
        cdtype = get_complex_dtype(dtype_idx)

        self.W_u = nn.Parameter(torch.empty(self.hidden, self.in_dim, dtype=cdtype))
        self.b_u = nn.Parameter(torch.empty(self.hidden, dtype=cdtype))
        self.W_v = nn.Parameter(torch.empty(self.hidden, self.in_dim, dtype=cdtype))
        self.b_v = nn.Parameter(torch.empty(self.hidden, dtype=cdtype))
        self.W_down = nn.Parameter(torch.empty(self.out_dim, self.hidden, dtype=cdtype))
        self.b_down = nn.Parameter(torch.empty(self.out_dim, dtype=cdtype))

        self.activation = gen_activation(self.hidden, eps, dtype_idx)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.reset_parameters()

    def reset_parameters(self) -> None:
        std_in = 1.0 / math.sqrt(self.in_dim)
        std_h = 1.0 / math.sqrt(self.hidden)
        for p in (self.W_u, self.W_v):
            nn.init.normal_(p.real, mean=0.0, std=std_in)
            nn.init.normal_(p.imag, mean=0.0, std=std_in)
        nn.init.normal_(self.W_down.real, mean=0.0, std=std_h)
        nn.init.normal_(self.W_down.imag, mean=0.0, std=std_h)
        for b in (self.b_u, self.b_v, self.b_down):
            nn.init.zeros_(b)

    def forward(self, Z: Tensor) -> Tensor:
        spatial = Z.shape[2:]
        B = Z.shape[0]
        Z_flat = Z.reshape(B, self.in_dim, -1)

        u = torch.einsum("bcn,fc->bfn", Z_flat, self.W_u) + self.b_u.view(1, -1, 1)
        v = torch.einsum("bcn,fc->bfn", Z_flat, self.W_v) + self.b_v.view(1, -1, 1)

        # Activation on u (channel-last)
        u_cl = u.movedim(1, -1)
        u_cl = self.activation(u_cl)
        u = u_cl.movedim(-1, 1)

        h = u * v          # complex elementwise
        h = self.dropout(h)

        out = torch.einsum("bfn,cf->bcn", h, self.W_down) + self.b_down.view(1, -1, 1)
        return out.reshape(B, self.out_dim, *spatial)


# ------------------------------------------------------------------ #
# Sparse MoE (magnitude-routed, complex experts)                      #
# ------------------------------------------------------------------ #

class ComplexSparseMoEFFN(ComplexFFNBase):
    """Sparse Mixture-of-Experts for complex tensors.

    Routing operates on per-token magnitudes (real ``|Z|``) — softmax of
    complex values has no canonical definition, so the gate is decoupled
    from phase. Each expert is a 2-layer complex MLP. Per-token output
    is a probability-weighted sum of the top-k expert outputs.

    Parameters
    ----------
    num_experts : int
        Total number of experts.
    top_k : int
        Active experts per token.
    load_balance_weight : float
        Auxiliary load-balancing loss weight (stored on ``self._aux_loss``;
        not currently aggregated by the panoptic model).
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_mult: int = 4,
        dropout: float = 0.0,
        gen_activation: Callable[..., nn.Module] = StableModReLU,
        eps: float = 1e-6,
        dtype_idx: FPDTypeIdx = 64,
        num_experts: int = 8,
        top_k: int = 2,
        load_balance_weight: float = 0.01,
    ) -> None:
        super().__init__()
        if not (1 <= top_k <= num_experts):
            raise ValueError(f"top_k must be in [1, {num_experts}], got {top_k}")
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.hidden = int(in_dim * hidden_mult)
        self.num_experts = int(num_experts)
        self.top_k = int(top_k)
        self.load_balance_weight = float(load_balance_weight)
        self.sparsity = 1.0 - top_k / num_experts

        cdtype = get_complex_dtype(dtype_idx)
        rdtype = get_float_dtype(dtype_idx)

        # Real-valued router gate (operates on |Z|)
        self.router = nn.Linear(self.in_dim, self.num_experts, dtype=rdtype)

        # Per-expert complex weights
        self.expert_W1 = nn.ParameterList([
            nn.Parameter(torch.empty(self.hidden, self.in_dim, dtype=cdtype))
            for _ in range(self.num_experts)
        ])
        self.expert_b1 = nn.ParameterList([
            nn.Parameter(torch.empty(self.hidden, dtype=cdtype))
            for _ in range(self.num_experts)
        ])
        self.expert_W2 = nn.ParameterList([
            nn.Parameter(torch.empty(self.out_dim, self.hidden, dtype=cdtype))
            for _ in range(self.num_experts)
        ])
        self.expert_b2 = nn.ParameterList([
            nn.Parameter(torch.empty(self.out_dim, dtype=cdtype))
            for _ in range(self.num_experts)
        ])

        self.activation = gen_activation(self.hidden, eps, dtype_idx)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self._aux_loss = torch.tensor(0.0, dtype=rdtype)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        std_in = 1.0 / math.sqrt(self.in_dim)
        std_h = 1.0 / math.sqrt(self.hidden)
        for W1, b1, W2, b2 in zip(
            self.expert_W1, self.expert_b1, self.expert_W2, self.expert_b2,
        ):
            nn.init.normal_(W1.real, mean=0.0, std=std_in)
            nn.init.normal_(W1.imag, mean=0.0, std=std_in)
            nn.init.zeros_(b1)
            nn.init.normal_(W2.real, mean=0.0, std=std_h)
            nn.init.normal_(W2.imag, mean=0.0, std=std_h)
            nn.init.zeros_(b2)

    @property
    def aux_loss(self) -> Tensor:
        return self._aux_loss

    def forward(self, Z: Tensor) -> Tensor:
        # Z: (B, in_dim, *S) complex
        spatial = Z.shape[2:]
        B = Z.shape[0]
        Z_flat = Z.reshape(B, self.in_dim, -1)            # (B, C, T)
        T = Z_flat.shape[-1]

        # Token-major flat: (B*T, C) complex
        Z_tokens = Z_flat.movedim(1, -1).reshape(-1, self.in_dim)
        mag_tokens = Z_tokens.abs()                         # (B*T, C) real

        # Route on magnitude (real)
        logits = self.router(mag_tokens)                    # (B*T, E)
        probs = F.softmax(logits, dim=-1)
        top_vals, top_idx = probs.topk(self.top_k, dim=-1)  # (B*T, K)
        top_vals = top_vals / top_vals.sum(dim=-1, keepdim=True)

        # Dispatch
        out = torch.zeros(
            B * T, self.out_dim, dtype=Z_tokens.dtype, device=Z_tokens.device,
        )
        for k in range(self.top_k):
            indices = top_idx[:, k]
            weights = top_vals[:, k:k + 1]                  # real
            for e in range(self.num_experts):
                mask = (indices == e)
                if not mask.any():
                    continue
                z_sel = Z_tokens[mask]                      # (M, C) complex
                # Expert e: linear → activation (channel-last) → dropout → linear
                h = z_sel @ self.expert_W1[e].T + self.expert_b1[e]
                h = self.activation(h)
                h = self.dropout(h)
                y = h @ self.expert_W2[e].T + self.expert_b2[e]
                # Weight by router prob (real → broadcast complex via .to)
                out[mask] = out[mask] + weights[mask].to(y.dtype) * y

        # Load-balance aux (informational; not aggregated into model loss)
        density = probs.mean(dim=0)
        uniform = torch.ones_like(density) / self.num_experts
        self._aux_loss = self.load_balance_weight * F.mse_loss(density, uniform)

        # (B*T, out_dim) → (B, out_dim, *S)
        return out.reshape(B, T, self.out_dim).movedim(-1, 1).reshape(
            B, self.out_dim, *spatial,
        )


# ------------------------------------------------------------------ #
# Identity (no FFN; SSM-only ablation)                                #
# ------------------------------------------------------------------ #

class ComplexIdentityFFN(ComplexFFNBase):
    """Pass-through. Use to ablate the FFN branch entirely."""

    def __init__(self, in_dim: int, out_dim: int, **_: Any) -> None:
        super().__init__()
        if in_dim != out_dim:
            raise ValueError(
                f"ComplexIdentityFFN requires in_dim == out_dim, got {in_dim} != {out_dim}"
            )

    def forward(self, Z: Tensor) -> Tensor:
        return Z


# ------------------------------------------------------------------ #
# Factory                                                             #
# ------------------------------------------------------------------ #

COMPLEX_FFN_REGISTRY: Dict[str, type] = {
    "standard": ComplexStandardFFN,
    "bilinear": ComplexBilinearFFN,
    "sparse_moe": ComplexSparseMoEFFN,
    "identity": ComplexIdentityFFN,
}


def create_complex_ffn(
    name: str,
    in_dim: int,
    out_dim: int,
    *,
    hidden_mult: int = 4,
    dropout: float = 0.0,
    dtype_idx: FPDTypeIdx = 64,
    **kwargs: Any,
) -> ComplexFFNBase:
    """Build a complex FFN by name.

    Parameters
    ----------
    name : str
        ``"standard"``, ``"bilinear"``, or ``"identity"``.
    in_dim, out_dim : int
        Channel dimensions.
    hidden_mult : int
        FFN expansion (default 4 — ignored for ``"identity"``).
    dropout : float
        Dropout probability after activation.
    dtype_idx : FPDTypeIdx
        Precision selector. Default 64 (complex64).
    **kwargs
        Backend-specific extras (e.g. ``gen_activation``).
    """
    if name not in COMPLEX_FFN_REGISTRY:
        raise ValueError(
            f"Unknown complex FFN backend {name!r}. "
            f"Available: {sorted(COMPLEX_FFN_REGISTRY.keys())}"
        )
    cls = COMPLEX_FFN_REGISTRY[name]
    if cls is ComplexIdentityFFN:
        return cls(in_dim=in_dim, out_dim=out_dim)
    return cls(
        in_dim=in_dim, out_dim=out_dim,
        hidden_mult=hidden_mult, dropout=dropout, dtype_idx=dtype_idx,
        **kwargs,
    )


__all__ = [
    "ComplexFFNBase",
    "ComplexStandardFFN",
    "ComplexBilinearFFN",
    "ComplexSparseMoEFFN",
    "ComplexIdentityFFN",
    "COMPLEX_FFN_REGISTRY",
    "create_complex_ffn",
]
