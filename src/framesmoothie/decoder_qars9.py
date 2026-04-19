"""QARS9 decoder layers (mirror of ARS9DecoderLayer / ARS9Decoder).

Uses QARS9CondMixBlock for the cross-attention-free mixing step.
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List, Callable, Tuple

from framesmoothie.blocks_qars9 import QARS9CondMixBlock
from framesmoothie.base import StabilizedActivationFunctionBase
from framesmoothie.activations import BiasedTeLU
from framesmoothie.adapters.base import ModuleAdapterBase
from s9.activations.real.hglu import HGLU
from s9.qars9_modules import QARS9Layer
from s9.quantization.bit_budget import QuantConfig
from s9.base import FPDTypeIdx
from s9._common.kernel_base import InitMode, Discretization
from framesmoothie.ffn_backends import FFNBase, create_ffn


class QARS9DecoderLayer(nn.Module):
    """
    One QARS9 decoder layer:
      q -> QARS9CondMixBlock(x, q) -> q'
      optional: extra FFN on q (slot-wise)
    """
    def __init__(
        self,
        *,
        c_model: int,
        q_dim: int,
        spatial_dims: int,
        gate_dim: int = 64,
        v_dim: Optional[int] = None,
        ffn_mult: int = 4,
        dropout: float = 0.0,
        eps: float = 1e-6,
        qars9_eps: float = 1e-6,
        return_masks: bool = True,
        qars9: Optional[QARS9Layer] = None,
        gen_activation: Callable[[int, float, FPDTypeIdx], StabilizedActivationFunctionBase] = BiasedTeLU,
        dtype_idx: FPDTypeIdx = 64,
        lambda_gate_entropy: float = 0.0,
        lambda_gate_competition: float = 0.0,
        adapter: Optional[ModuleAdapterBase] = None,
        post_ffn: bool = True,
        post_ffn_mult: int = 4,
        ffn_backend: Optional[str] = None,
        ffn_kwargs: Optional[Dict[str, Any]] = None,
        init_mode: InitMode = "legacy",
        discretization: Discretization = "zoh",
        quant_config: QuantConfig = QuantConfig(),
    ):
        super().__init__()

        self.cross = QARS9CondMixBlock(
            c_model=c_model,
            q_dim=q_dim,
            spatial_dims=spatial_dims,
            gate_dim=gate_dim,
            v_dim=v_dim,
            ffn_mult=ffn_mult,
            dropout=dropout,
            eps=eps,
            qars9_eps=qars9_eps,
            return_masks=return_masks,
            qars9=qars9,
            gen_activation=gen_activation,
            dtype_idx=dtype_idx,
            lambda_gate_entropy=lambda_gate_entropy,
            lambda_gate_competition=lambda_gate_competition,
            adapter=adapter,
            ffn_backend=ffn_backend,
            ffn_kwargs=ffn_kwargs,
            init_mode=init_mode,
            discretization=discretization,
            quant_config=quant_config,
        )

        self.post_ffn_enabled = post_ffn
        if post_ffn:
            self.q_ln = nn.LayerNorm(q_dim)
            if ffn_backend is not None:
                _fkw = ffn_kwargs or {}
                self.q_ffn: nn.Module = create_ffn(
                    ffn_backend,
                    in_dim=q_dim,
                    out_dim=q_dim,
                    hidden_mult=post_ffn_mult,
                    dropout=dropout,
                    dtype_idx=dtype_idx,
                    **_fkw,
                )
            else:
                self.q_ffn = nn.Sequential(
                    nn.Linear(q_dim, post_ffn_mult * q_dim),
                    HGLU(4.0),
                    nn.Dropout(dropout),
                    nn.Linear(post_ffn_mult * q_dim, q_dim),
                    nn.Dropout(dropout),
                )
        else:
            self.q_ln = None
            self.q_ffn = None

    def forward(self, x: torch.Tensor, q: torch.Tensor) -> Dict[str, torch.Tensor]:
        out = self.cross(x, q)
        q = out["q"]

        if self.post_ffn_enabled:
            dq = self.q_ffn(self.q_ln(q))
            q = q + dq

        out["q"] = q
        return out


class QARS9Decoder(nn.Module):
    """
    Stack of QARS9DecoderLayer.

    Returns:
      - final q
      - list of mask_logits per layer (if return_masks=True)
    """
    def __init__(self, layers: List[QARS9DecoderLayer]):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor, q: torch.Tensor) -> Dict[str, Any]:
        mask_logits_all = []
        for layer in self.layers:
            out = layer(x, q)
            q = out["q"]
            if "mask_logits" in out:
                mask_logits_all.append(out["mask_logits"])
        return {"q": q, "mask_logits_all": mask_logits_all}
