import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any

from s9.base import FPDTypeIdx, get_float_dtype
from s9.activations.real.hglu import HGLU

def _flatten_spatial_cl(x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, ...]]:
    """
    Channel-last flatten.
    x: [B, *S, C] -> xf: [B, N, C], spatial: (*S)
    """
    if x.ndim < 3:
        raise ValueError(f"Expected [B,*S,C], got {tuple(x.shape)}")
    B = x.shape[0]
    C = x.shape[-1]
    spatial = tuple(x.shape[1:-1])
    N = 1
    for s in spatial:
        N *= s
    xf = x.reshape(B, N, C)
    return xf, spatial


class RS9InstanceHead(nn.Module):
    """
    Instance head for RS9Decoder outputs.

    Inputs:
      x: [B,*S,C]  (backbone/decoder pixel features; channel-last)
      q: [B,K,Dq]  (decoder slots/queries)
      mask_logits_gate_last (optional): [B,K,*S]  (e.g., last decoder layer mask logits)

    Outputs:
      class_logits: [B,K,num_classes+1]  (include "no-object" at last index)
      scores:       [B,K]               (optional confidence head)
      mask_logits:  [B,K,*S]
    """
    def __init__(
        self,
        *,
        c_model: int,
        q_dim: int,
        num_classes: int,
        mask_dim: int = 256,
        dtype_idx: FPDTypeIdx = 64,
        dropout: float = 0.0,
        gate_fuse: float = 0.0,  # weight for adding gate logits
        normalize_embeddings: bool = True,
        with_scores: bool = True,
    ):
        super().__init__()
        self.dtype = get_float_dtype(dtype_idx)
        self.num_classes = num_classes
        self.gate_fuse = float(gate_fuse)
        self.normalize_embeddings = bool(normalize_embeddings)
        self.with_scores = bool(with_scores)

        # --- classification head (slot-wise) ---
        self.cls_ln = nn.LayerNorm(q_dim, dtype=self.dtype)
        self.cls_mlp = nn.Sequential(
            nn.Linear(q_dim, q_dim, dtype=self.dtype),
            HGLU(4.0),
            nn.Dropout(dropout),
            nn.Linear(q_dim, num_classes + 1, dtype=self.dtype),  # +1 for no-object
        )

        # optional score head (slot-wise)
        if self.with_scores:
            self.score_ln = nn.LayerNorm(q_dim, dtype=self.dtype)
            self.score_head = nn.Sequential(
                nn.Linear(q_dim, q_dim, dtype=self.dtype),
                HGLU(4.0),
                nn.Dropout(dropout),
                nn.Linear(q_dim, 1, dtype=self.dtype),
            )
        else:
            self.score_ln = None
            self.score_head = None

        # --- mask head (pixel/slot embeddings) ---
        self.pixel_proj = nn.Linear(c_model, mask_dim, bias=False, dtype=self.dtype)
        self.slot_proj = nn.Linear(q_dim, mask_dim, bias=False, dtype=self.dtype)

    def forward(
        self,
        x: torch.Tensor,
        q: torch.Tensor,
        *,
        mask_logits_gate_last: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        # ---- class logits ----
        class_logits = self.cls_mlp(self.cls_ln(q))  # [B,K,C+1]

        if self.with_scores:
            scores = self.score_head(self.score_ln(q)).squeeze(-1)  # [B,K]
        else:
            scores = torch.zeros(q.shape[0], q.shape[1], dtype=self.dtype, device=q.device)

        # ---- mask logits via dot product ----
        x_flat, spatial = _flatten_spatial_cl(x)  # [B,N,C]
        pix = self.pixel_proj(x_flat)             # [B,N,Dm]
        slot = self.slot_proj(q)                  # [B,K,Dm]

        if self.normalize_embeddings:
            pix = F.normalize(pix, dim=-1)
            slot = F.normalize(slot, dim=-1)

        # mask_logits_proto: [B,K,N]
        mask_logits = torch.einsum("bkd,bnd->bkn", slot, pix)

        # reshape to [B,K,*S]
        mask_logits = mask_logits.reshape(q.shape[0], q.shape[1], *spatial)

        # optionally fuse last gate logits (from RS9CondMixBlock)
        if (mask_logits_gate_last is not None) and (self.gate_fuse != 0.0):
            # Assume same spatial shape
            mask_logits = mask_logits + (self.gate_fuse * mask_logits_gate_last.to(dtype=self.dtype))

        return {
            "class_logits": class_logits,
            "scores": scores,
            "mask_logits": mask_logits,
        }