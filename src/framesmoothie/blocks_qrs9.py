"""QRS9-based conditioned mixing block (mirror of RS9CondMixBlock).

Uses QRS9Layer (quantized variant of RS9Layer) instead of RS9Layer.
Real I/O pipeline is identical to RS9; only the underlying SSM kernel differs.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from s9.qrs9_modules import QRS9Layer
from s9.quantization.bit_budget import QuantConfig
from s9.base import FPDTypeIdx, get_float_dtype
from s9.activations.real.hglu import HGLU
from s9._common.kernel_base import InitMode, Discretization

try:
    # Python 3.12+
    from typing import override, Tuple, Optional, Dict, Any, Callable
except Exception:  # pragma: no cover
    from typing_extensions import override, Tuple, Optional, Dict, Any, Callable

from torchutils.decorators import auxloss
from framesmoothie.base import StabilizedActivationFunctionBase
from framesmoothie.activations import BiasedTeLU
from framesmoothie.fmlm import FMLMFiLM
from framesmoothie.adapters.base import ModuleAdapterBase
from framesmoothie.ffn_backends import FFNBase, create_ffn
from framesmoothie.blocks import (
    _to_channel_first, _to_channel_last,
    _flatten_spatial, _unflatten_spatial,
    BilinearGate, Film,
)


class QRS9CondMixBlock(nn.Module):
    """
    Attention-free, QRS9-based conditioned mixing block.

    Structurally identical to ``RS9CondMixBlock`` but uses QRS9Layer
    (quantized variant of RS9Layer) as the global mixer.

    Returns:
      Q_out: [B, K, Dq]
      mask_logits(optional): [B, K, *S]
    """
    def __init__(
        self,
        c_model: int,
        q_dim: int,
        spatial_dims: int,
        gate_dim: int = 64,
        v_dim: Optional[int] = None,
        ffn_mult: int = 4,
        dropout: float = 0.0,
        eps: float = 1e-6,
        qrs9_eps: float = 1e-6,
        return_masks: bool = True,
        qrs9: Optional[QRS9Layer] = None,
        gen_activation: Callable[[int, float, FPDTypeIdx], StabilizedActivationFunctionBase] = BiasedTeLU,
        dtype_idx: FPDTypeIdx = 64,
        lambda_gate_entropy: float = 0.0,
        lambda_gate_competition: float = 0.0,
        adapter: Optional[ModuleAdapterBase] = None,
        ffn_backend: Optional[str] = None,
        ffn_kwargs: Optional[Dict[str, Any]] = None,
        init_mode: InitMode = "legacy",
        discretization: Discretization = "zoh",
        quant_config: QuantConfig = QuantConfig(),
    ):
        super().__init__()
        self.qrs9: QRS9Layer = qrs9 if qrs9 is not None else QRS9Layer(
            d_model=c_model,
            spatial_dims=spatial_dims,
            gen_activation=gen_activation,
            eps=qrs9_eps,
            dtype_idx=dtype_idx,
            init_mode=init_mode,
            discretization=discretization,
            quant_config=quant_config,
        )
        self.c_model: int = c_model
        self.q_dim: int = q_dim
        self.gate_dim: int = gate_dim
        self.v_dim: int = v_dim if v_dim is not None else q_dim
        self.eps: float = eps
        self.return_masks: bool = return_masks
        self.adapter: Optional[ModuleAdapterBase] = adapter
        self.dtype: torch.dtype = get_float_dtype(dtype_idx)

        # Gate from X and Q
        self.gate = BilinearGate(c_in=c_model, q_dim=q_dim, gate_dim=gate_dim, dtype=self.dtype)

        # Values
        self.vx = nn.Linear(c_model, self.v_dim, bias=False, dtype=self.dtype)
        self.vh = nn.Linear(c_model, self.v_dim, bias=False, dtype=self.dtype)
        ctx_dim = getattr(adapter, 'ctx_dim', None) if adapter is not None else None
        self.film = FMLMFiLM(q_dim=q_dim, d=self.v_dim, ctx_dim=ctx_dim, rank=8, eta=0.1, alpha=0.1, dtype_idx=dtype_idx)

        # Query update
        self.ln = nn.LayerNorm(self.v_dim, dtype=self.dtype)
        if ffn_backend is not None:
            _fkw = ffn_kwargs or {}
            self.ffn: nn.Module = create_ffn(
                ffn_backend,
                in_dim=self.v_dim,
                out_dim=q_dim,
                hidden_mult=ffn_mult,
                dropout=dropout,
                dtype_idx=dtype_idx,
                **_fkw,
            )
        else:
            self.ffn = nn.Sequential(
                nn.Linear(self.v_dim, ffn_mult * self.v_dim, dtype=self.dtype),
                HGLU(4.0),
                nn.Dropout(dropout),
                nn.Linear(ffn_mult * self.v_dim, q_dim, dtype=self.dtype),
                nn.Dropout(dropout),
            )

        # Adapter hooks
        if self.adapter is not None:
            self.gate.wx = self.adapter.wrap_linear(self.gate.wx, task="instance")
            self.gate.wq = self.adapter.wrap_linear(self.gate.wq, task="instance")

            self.vx = self.adapter.wrap_linear(self.vx, task="instance")
            self.vh = self.adapter.wrap_linear(self.vh, task="instance")

            if isinstance(self.ffn, FFNBase):
                for attr_path, linear in self.ffn.wrappable_linears():
                    parts = attr_path.split(".")
                    parent = self.ffn
                    for p in parts[:-1]:
                        parent = getattr(parent, p) if not p.isdigit() else parent[int(p)]
                    setattr(parent, parts[-1], self.adapter.wrap_linear(linear, task="instance"))
            else:
                self.ffn[0] = self.adapter.wrap_linear(self.ffn[0], task="instance")
                self.ffn[3] = self.adapter.wrap_linear(self.ffn[3], task="instance")

        self._reg_loss: torch.Tensor = torch.tensor(0.0, dtype=self.dtype)
        self.lambda_gate_entropy: float = lambda_gate_entropy
        self.lambda_gate_competition: float = lambda_gate_competition

    def forward(self, x: torch.Tensor, q: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        x: [B, *S, C]
        q: [B, K, Dq]
        """
        if (x.ndim - 2) != self.qrs9.base.spatial_dims:
            raise ValueError(f"spatial_dims mismatch: x has {x.ndim-2}, qrs9 has {self.qrs9.base.spatial_dims}")
        x_cf = _to_channel_first(x)
        h_cf = self.qrs9(x_cf)
        h = _to_channel_last(h_cf)

        x_flat, spatial = _flatten_spatial(x)
        h_flat, _ = _flatten_spatial(h)
        B, N, C = x_flat.shape
        K = q.shape[1]

        g_logits = self.gate(x_flat, q)
        g = torch.sigmoid(g_logits)

        reg = torch.tensor(0.0, dtype=self.dtype, device=g.device)

        if self.lambda_gate_entropy != 0.0:
            p = g.clamp(min=self.eps, max=1.0 - self.eps)
            ent = -(p * torch.log(p) + (1.0 - p) * torch.log(1.0 - p))
            reg = reg + (self.lambda_gate_entropy * ent.mean())

        if self.lambda_gate_competition != 0.0:
            overlap = g.sum(dim=1)
            comp = F.relu(overlap - 1.0)
            reg = reg + (self.lambda_gate_competition * comp.mean())

        self._reg_loss = reg

        vx = self.vx(x_flat)
        vh = self.vh(h_flat)
        ctx = self.adapter.hub.get_context() if (self.adapter is not None and hasattr(self.adapter, 'hub')) else None
        vh_film = self.film(q, vh, ctx=ctx)
        v = vx.unsqueeze(1) + vh_film

        w = g.unsqueeze(-1)
        num = (w * v).sum(dim=2)
        den = w.sum(dim=2).clamp_min(self.eps)
        read = num / den

        read = self.ln(read)
        dq = self.ffn(read)
        q_out = q + dq

        out: Dict[str, torch.Tensor] = {"q": q_out}

        if self.return_masks:
            mask_logits = g_logits.reshape(B, K, *spatial)
            out["mask_logits"] = mask_logits

        return out

    @auxloss
    def reg_loss(self) -> torch.Tensor:
        return self._reg_loss

    @reg_loss.setter
    def reg_loss(self, loss: torch.Tensor):
        self._reg_loss = loss.to(dtype=self.dtype).reshape(())

    @reg_loss.collector
    def reg_loss(
        self,
        aggregate: Callable[[torch.Tensor], torch.Tensor] = torch.sum,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        losses = []
        for m in self.modules():
            if hasattr(m, "_reg_loss"):
                rl = getattr(m, "_reg_loss")
                if isinstance(rl, torch.Tensor):
                    losses.append(rl.reshape(()))
        out = torch.tensor(0.0, dtype=self.dtype) if not losses else aggregate(torch.stack(losses))
        if dtype is not None:
            out = out.to(dtype=dtype)
        if device is not None:
            out = out.to(device=device)
        return out.reshape(())

    @reg_loss.resetter
    @torch.no_grad()
    def reg_loss(self, value: float):
        for m in self.modules():
            if hasattr(m, "_reg_loss"):
                rl = getattr(m, "_reg_loss")
                if isinstance(rl, torch.Tensor):
                    setattr(m, "_reg_loss", rl.detach().new_tensor(value).reshape(()))
