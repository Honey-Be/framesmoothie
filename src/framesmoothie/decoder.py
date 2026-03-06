import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List

from framesmoothie.blocks import RS9CondMixBlock
from s9.activations.real.hglu import HGLU


class RS9DecoderLayer(nn.Module):
    """
    One decoder layer:
      q -> RS9CondMixBlock(x, q) -> q'
      optional: extra FFN on q (slot-wise)
    """
    def __init__(
        self,
        *,
        c_model: int,
        q_dim: int,
        spatial_dims: int,
        # RS9CondMixBlock params
        gate_dim: int = 64,
        v_dim: Optional[int] = None,
        ffn_mult: int = 4,
        dropout: float = 0.0,
        eps: float = 1e-6,
        rs9_eps: float = 1e-6,
        return_masks: bool = True,
        rs9=None,
        gen_activation=None,
        dtype_idx=64,
        lambda_gate_entropy: float = 0.0,
        lambda_gate_competition: float = 0.0,
        # extra slot FFN (post)
        post_ffn: bool = True,
        post_ffn_mult: int = 4,
    ):
        super().__init__()

        self.cross = RS9CondMixBlock(
            c_model=c_model,
            q_dim=q_dim,
            spatial_dims=spatial_dims,
            gate_dim=gate_dim,
            v_dim=v_dim,
            ffn_mult=ffn_mult,
            dropout=dropout,
            eps=eps,
            rs9_eps=rs9_eps,
            return_masks=return_masks,
            rs9=rs9,
            gen_activation=gen_activation,
            dtype_idx=dtype_idx,
            lambda_gate_entropy=lambda_gate_entropy,
            lambda_gate_competition=lambda_gate_competition,
        )

        self.post_ffn_enabled = post_ffn
        if post_ffn:
            # Slot-wise FFN on q: [B,K,Dq] -> [B,K,Dq]
            # Note: LayerNorm supports (B,K,D) with normalized_shape=D
            self.q_ln = nn.LayerNorm(q_dim)
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
        """
        x: [B,*S,C]
        q: [B,K,Dq]
        returns:
          q: [B,K,Dq]
          mask_logits (optional): [B,K,*S]
        """
        out = self.cross(x, q)
        q = out["q"]

        if self.post_ffn_enabled:
            dq = self.q_ffn(self.q_ln(q))
            q = q + dq

        out["q"] = q
        return out


class RS9Decoder(nn.Module):
    """
    Stack of RS9DecoderLayer.

    Returns:
      - final q
      - list of mask_logits per layer (if return_masks=True)
    """
    def __init__(self, layers: List[RS9DecoderLayer]):
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