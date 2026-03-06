import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union, Dict

from s9.base import FPDTypeIdx, get_float_dtype
from s9.activations.real.hglu import HGLU


def _to_channel_first(x: torch.Tensor) -> torch.Tensor:
    # x: [B,*S,C] -> [B,C,*S]
    spatial_dims = x.ndim - 2
    perm = [0, spatial_dims + 1] + list(range(1, 1 + spatial_dims))
    return x.permute(*perm)


def _to_channel_last(x: torch.Tensor) -> torch.Tensor:
    # x: [B,C,*S] -> [B,*S,C]
    spatial_dims = x.ndim - 2
    perm = [0] + list(range(2, 2 + spatial_dims)) + [1]
    return x.permute(*perm)


def _infer_interp_mode(spatial_dims: int) -> str:
    if spatial_dims == 1:
        return "linear"
    if spatial_dims == 2:
        return "bilinear"
    if spatial_dims == 3:
        return "trilinear"
    raise ValueError(f"Unsupported spatial_dims={spatial_dims}")


class SemanticFPNHead(nn.Module):
    """
    Semantic head (FPN-like) for panoptic UDA.

    Input:
      - features: Tensor [B,*S,C] or List[Tensor] multi-scale (channel-last).
        * The first element is assumed highest resolution.

    Output:
      - sem_logits: [B, num_classes, *S0] (channel-first)
    """
    def __init__(
        self,
        *,
        num_classes: int,
        c_model: int,
        spatial_dims: int,
        hidden_dim: int = 256,
        dropout: float = 0.0,
        dtype_idx: FPDTypeIdx = 64,
    ):
        super().__init__()
        self.dtype = get_float_dtype(dtype_idx)
        self.num_classes = num_classes
        self.c_model = c_model
        self.hidden_dim = hidden_dim
        self.spatial_dims = spatial_dims
        self.mode = _infer_interp_mode(spatial_dims)

        # per-scale projection (assume same c_model across scales)
        self.proj = nn.ModuleList([
            nn.Conv2d(c_model, hidden_dim, 1, bias=False, dtype=self.dtype) if spatial_dims == 2 else
            nn.Conv3d(c_model, hidden_dim, 1, bias=False, dtype=self.dtype) if spatial_dims == 3 else
            nn.Conv1d(c_model, hidden_dim, 1, bias=False, dtype=self.dtype)
            for _ in range(4)  # safe default; if fewer scales passed, only first N used
        ])
        self.act = HGLU(4.0)
        self.drop = nn.Dropout(dropout)

        self.fuse = (
            nn.Conv2d(hidden_dim, hidden_dim, 3, padding=1, bias=False, dtype=self.dtype) if spatial_dims == 2 else
            nn.Conv3d(hidden_dim, hidden_dim, 3, padding=1, bias=False, dtype=self.dtype) if spatial_dims == 3 else
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1, bias=False, dtype=self.dtype)
        )
        self.classifier = (
            nn.Conv2d(hidden_dim, num_classes, 1, bias=True, dtype=self.dtype) if spatial_dims == 2 else
            nn.Conv3d(hidden_dim, num_classes, 1, bias=True, dtype=self.dtype) if spatial_dims == 3 else
            nn.Conv1d(hidden_dim, num_classes, 1, bias=True, dtype=self.dtype)
        )

    def forward(self, features: Union[torch.Tensor, List[torch.Tensor]]) -> torch.Tensor:
        if isinstance(features, torch.Tensor):
            feats = [features]
        else:
            feats = features

        if len(feats) == 0:
            raise ValueError("features must be non-empty")

        # target resolution from highest-res feature (first)
        x0 = feats[0]
        target_spatial = tuple(x0.shape[1:-1])  # *S0

        # aggregate
        agg_cf = None
        for i, x in enumerate(feats):
            if i >= len(self.proj):
                break
            x_cf = _to_channel_first(x).to(dtype=self.dtype)  # [B,C,*S]
            y = self.proj[i](x_cf)  # [B,Hd,*S_i]
            y = self.act(y)
            y = self.drop(y)

            # upsample to target spatial
            if tuple(y.shape[2:]) != target_spatial:
                y = F.interpolate(y, size=target_spatial, mode=self.mode, align_corners=False if self.spatial_dims >= 2 else None)

            agg_cf = y if agg_cf is None else (agg_cf + y)

        agg_cf = self.fuse(agg_cf)
        agg_cf = self.act(agg_cf)
        agg_cf = self.drop(agg_cf)
        sem_logits = self.classifier(agg_cf)  # [B,num_classes,*S0]
        return sem_logits


class RS9InstanceHead(nn.Module):
    """
    Prototype dot-product instance head (attention-free).

    Inputs:
      x: [B,*S,C] (real features, channel-last)
      q: [B,K,Dq]
      mask_logits_gate_last(optional): [B,K,*S]
    Outputs:
      class_logits: [B,K,num_classes+1]
      scores: [B,K] (optional)
      mask_logits: [B,K,*S]
    """
    def __init__(
        self,
        *,
        c_model: int,
        q_dim: int,
        num_classes: int,
        mask_dim: int = 256,
        dropout: float = 0.0,
        dtype_idx: FPDTypeIdx = 64,
        gate_fuse: float = 0.0,
        normalize_embeddings: bool = True,
        with_scores: bool = True,
    ):
        super().__init__()
        self.dtype = get_float_dtype(dtype_idx)
        self.num_classes = num_classes
        self.gate_fuse = float(gate_fuse)
        self.normalize_embeddings = bool(normalize_embeddings)
        self.with_scores = bool(with_scores)

        self.act = HGLU(4.0)

        self.cls_ln = nn.LayerNorm(q_dim, dtype=self.dtype)
        self.cls_fc1 = nn.Linear(q_dim, q_dim, dtype=self.dtype)
        self.cls_fc2 = nn.Linear(q_dim, num_classes + 1, dtype=self.dtype)
        self.drop = nn.Dropout(dropout)

        if self.with_scores:
            self.score_ln = nn.LayerNorm(q_dim, dtype=self.dtype)
            self.score_fc1 = nn.Linear(q_dim, q_dim, dtype=self.dtype)
            self.score_fc2 = nn.Linear(q_dim, 1, dtype=self.dtype)
        else:
            self.score_ln = None
            self.score_fc1 = None
            self.score_fc2 = None

        self.pixel_proj = nn.Linear(c_model, mask_dim, bias=False, dtype=self.dtype)
        self.slot_proj = nn.Linear(q_dim, mask_dim, bias=False, dtype=self.dtype)

    def forward(
        self,
        x: torch.Tensor,
        q: torch.Tensor,
        *,
        mask_logits_gate_last: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        # class logits
        h = self.cls_ln(q)
        h = self.drop(self.act(self.cls_fc1(h)))
        class_logits = self.cls_fc2(h)

        if self.with_scores:
            s = self.score_ln(q)
            s = self.drop(self.act(self.score_fc1(s)))
            scores = self.score_fc2(s).squeeze(-1)
        else:
            scores = torch.zeros(q.shape[0], q.shape[1], dtype=self.dtype, device=q.device)

        # masks
        B = x.shape[0]
        K = q.shape[1]
        spatial = tuple(x.shape[1:-1])
        N = 1
        for sdim in spatial:
            N *= sdim

        x_flat = x.reshape(B, N, x.shape[-1]).to(dtype=self.dtype)  # [B,N,C]
        pix = self.pixel_proj(x_flat)  # [B,N,Dm]
        slot = self.slot_proj(q)       # [B,K,Dm]

        if self.normalize_embeddings:
            pix = F.normalize(pix, dim=-1)
            slot = F.normalize(slot, dim=-1)

        mask_logits = torch.einsum("bkd,bnd->bkn", slot, pix).reshape(B, K, *spatial)

        if (mask_logits_gate_last is not None) and (self.gate_fuse != 0.0):
            mask_logits = mask_logits + self.gate_fuse * mask_logits_gate_last.to(dtype=self.dtype)

        return {"class_logits": class_logits, "scores": scores, "mask_logits": mask_logits}