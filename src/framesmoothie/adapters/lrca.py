import torch
import torch.nn as nn
from typing import Optional

from s9.base import FPDTypeIdx, get_float_dtype
from .base import AdapterHubBase, ModuleAdapterBase


class LRCAHub(AdapterHubBase):
    """Shared LRCA context hub (shared gate)."""
    def __init__(self, ctx_dim: int, rank_shared: int, dtype_idx: FPDTypeIdx = 64):
        super().__init__()
        self.dtype = get_float_dtype(dtype_idx)
        self.ctx_dim = int(ctx_dim)
        self.rank_shared = int(rank_shared)

        self.ws = nn.Linear(self.ctx_dim, self.rank_shared, bias=True, dtype=self.dtype)
        self._ctx: Optional[torch.Tensor] = None
        self._g_s: Optional[torch.Tensor] = None

    def set_context(self, ctx: torch.Tensor) -> None:
        self._ctx = ctx.to(dtype=self.dtype)
        self._g_s = None

    def clear(self) -> None:
        self._ctx = None
        self._g_s = None

    def shared_gate(self) -> torch.Tensor:
        if self._ctx is None:
            raise RuntimeError("LRCAHub context is not set. Call set_context(ctx) before forward.")
        if self._g_s is None:
            self._g_s = torch.sigmoid(self.ws(self._ctx))
        return self._g_s


class LRCALinear(nn.Module):
    """Linear with LRCA shared+private low-rank coadaptation."""
    def __init__(
        self,
        base: nn.Linear,
        hub: LRCAHub,
        *,
        rank_shared: int,
        rank_private: int,
        ctx_dim: int,
        dtype_idx: FPDTypeIdx = 64,
        scale: float = 1.0,
        freeze_base: bool = True,
    ):
        super().__init__()
        self.base = base
        self.hub = hub
        self.dtype = get_float_dtype(dtype_idx)

        self.rank_shared = int(rank_shared)
        self.rank_private = int(rank_private)
        self.ctx_dim = int(ctx_dim)
        self.scale = float(scale)

        d_out = base.out_features
        d_in = base.in_features

        self.U_s = nn.Parameter(torch.zeros(d_out, self.rank_shared, dtype=self.dtype))
        self.V_s = nn.Parameter(torch.randn(self.rank_shared, d_in, dtype=self.dtype) * 0.01)

        self.U_p = nn.Parameter(torch.zeros(d_out, self.rank_private, dtype=self.dtype))
        self.V_p = nn.Parameter(torch.randn(self.rank_private, d_in, dtype=self.dtype) * 0.01)

        self.wp = nn.Linear(self.ctx_dim, self.rank_private, bias=True, dtype=self.dtype)

        if freeze_base:
            for p in self.base.parameters():
                p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base(x)

        g_s = self.hub.shared_gate().to(dtype=self.dtype)
        ctx = self.hub._ctx
        if ctx is None:
            raise RuntimeError("LRCAHub context missing")
        g_p = torch.sigmoid(self.wp(ctx)).to(dtype=self.dtype)

        B = x.shape[0]
        extra = x.ndim - 2
        g_s_view = g_s.view(B, *([1] * extra), self.rank_shared)
        g_p_view = g_p.view(B, *([1] * extra), self.rank_private)

        a_s = torch.matmul(x.to(dtype=self.dtype), self.V_s.t())
        a_s = a_s * g_s_view
        y_s = torch.matmul(a_s, self.U_s.t())

        a_p = torch.matmul(x.to(dtype=self.dtype), self.V_p.t())
        a_p = a_p * g_p_view
        y_p = torch.matmul(a_p, self.U_p.t())

        return y + self.scale * (y_s + y_p)


class LRCAAdapter(ModuleAdapterBase):
    """Swappable LRCA adapter plugin."""
    def __init__(
        self,
        *,
        ctx_dim: int,
        rank_shared: int = 8,
        rank_private: int = 4,
        dtype_idx: FPDTypeIdx = 64,
        scale: float = 1.0,
        freeze_base: bool = True,
    ):
        super().__init__()
        self.ctx_dim = int(ctx_dim)
        self.rank_shared = int(rank_shared)
        self.rank_private = int(rank_private)
        self.dtype_idx = dtype_idx
        self.scale = float(scale)
        self.freeze_base = bool(freeze_base)
        self.hub = LRCAHub(ctx_dim=self.ctx_dim, rank_shared=self.rank_shared, dtype_idx=dtype_idx)

    def set_context(self, ctx: torch.Tensor) -> None:
        self.hub.set_context(ctx)

    def clear(self) -> None:
        self.hub.clear()

    def wrap_linear(self, base: nn.Linear) -> nn.Module:
        return LRCALinear(
            base,
            self.hub,
            rank_shared=self.rank_shared,
            rank_private=self.rank_private,
            ctx_dim=self.ctx_dim,
            dtype_idx=self.dtype_idx,
            scale=self.scale,
            freeze_base=self.freeze_base,
        )
