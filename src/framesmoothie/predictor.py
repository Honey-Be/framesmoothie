from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Any, Optional, Mapping

import torch
import torch.nn as nn

from s9.base import FPDTypeIdx, get_float_dtype
from s9.activations.real.hglu import HGLU


def edge_key(srcs: tuple[str, ...], tgt: str) -> str:
    return "+".join(srcs) + "->" + tgt

SSLProjectorLossesType = Literal[
    "simsiam",
    "symmetric_simsiam",
    "sym_simsiam",
    "barlow",
    "vicreg"
]

SSL_PROJECTOR_LOSSES: set[SSLProjectorLossesType] = {
    "simsiam",
    "symmetric_simsiam",
    "sym_simsiam",
    "barlow",
    "vicreg"
}

@dataclass(frozen=True, slots=True)
class ZoneEdgeSpec:
    srcs: tuple[str, ...]
    tgt: str
    weight: float = 1.0
    name: str | None = None
    
    # predictor
    predictor_kind: str = "mlp"
    hidden_mult: float = 1.0
    
    # loss
    loss_type: str = "smooth_l1"
    loss_kwargs: Optional[Mapping[str, float]] = None
    detach_target: Optional[bool] = None

    # projector path(s) for representation-style losses
    projector_kind: Optional[str] = None
    projector_dim: Optional[int] = None
    projector_hidden_mult: float = 1.0

    target_projector_kind: Optional[str] = None
    target_projector_dim: Optional[int] = None
    target_projector_hidden_mult: Optional[float] = None

    # projector sharing
    projector_share_mode: str = "none"    # "none" | "shared"

    # projector regularization (primarily for Barlow/VICReg)
    projector_reg_type: Optional[str] = None  # "variance" | "covariance" | "var_cov" | "l2"
    projector_reg_weight: float = 0.0
    projector_reg_kwargs: Optional[Mapping[str, float]] = None

    def key(self) -> str:
        return self.name if self.name is not None else edge_key(self.srcs, self.tgt)


EdgeSpecLike = (
    ZoneEdgeSpec
    | tuple[tuple[str, ...], str]
    | tuple[tuple[str, ...], str, float]
    | tuple[tuple[str, ...], str, float, str]
)


def normalize_edge_specs(specs: Sequence[EdgeSpecLike]) -> tuple[ZoneEdgeSpec, ...]:
    out: list[ZoneEdgeSpec] = []
    for spec in specs:
        if isinstance(spec, ZoneEdgeSpec):
            out.append(spec)
        elif isinstance(spec, tuple) and len(spec) == 2:
            srcs, tgt = spec
            out.append(ZoneEdgeSpec(tuple(srcs), tgt))
        elif isinstance(spec, tuple) and len(spec) == 3:
            srcs, tgt, weight = spec
            out.append(ZoneEdgeSpec(tuple(srcs), tgt, float(weight)))
        elif isinstance(spec, tuple) and len(spec) == 4:
            srcs, tgt, weight, predictor_kind = spec
            out.append(ZoneEdgeSpec(tuple(srcs), tgt, float(weight), name=None, predictor_kind=str(predictor_kind)))
        else:
            raise TypeError(f"Unsupported zone edge spec: {spec!r}")
    return tuple(out)

def _to_channel_first(x: torch.Tensor) -> torch.Tensor:
    spatial_dims = x.ndim - 2
    perm = [0, spatial_dims + 1] + list(range(1, 1 + spatial_dims))
    return x.permute(*perm)


def _to_channel_last(x: torch.Tensor) -> torch.Tensor:
    spatial_dims = x.ndim - 2
    perm = [0] + list(range(2, 2 + spatial_dims)) + [1]
    return x.permute(*perm)

class _LinearPredictor(nn.Module):
    """Per-scale predictor for one zone edge, operating on channel-last tensors."""
    def __init__(self, *, c_in: int, c_out: int, dtype: torch.dtype):
        super().__init__()
        self.net = nn.Linear(c_in, c_out, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _MLPPredictor(nn.Module):
    def __init__(self, *, c_in: int, c_out: int, hidden: int, depth: int, dtype: torch.dtype):
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = c_in
        for _ in range(max(depth - 1, 1)):
            layers.append(nn.Linear(in_dim, hidden, dtype=dtype))
            layers.append(HGLU(4.0))
            in_dim = hidden
        layers.append(nn.Linear(in_dim, c_out, dtype=dtype))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class _ResidualMLPPredictor(nn.Module):
    def __init__(self, *, c_in: int, c_out: int, hidden: int, dtype: torch.dtype):
        super().__init__()
        self.proj_in = nn.Linear(c_in, c_out, dtype=dtype)
        self.ffn = nn.Sequential(
            nn.Linear(c_out, hidden, dtype=dtype),
            HGLU(4.0),
            nn.Linear(hidden, c_out, dtype=dtype),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.proj_in(x)
        return y + self.ffn(y)


class _BilinearMLPPredictor(nn.Module):
    """
    Bilinear MLP / bilinear GLU without element-wise nonlinearity:
      y = W_o((W_u x) ⊙ (W_v x))
    """
    def __init__(self, *, c_in: int, c_out: int, hidden: int, dtype: torch.dtype):
        super().__init__()
        self.u = nn.Linear(c_in, hidden, dtype=dtype)
        self.v = nn.Linear(c_in, hidden, dtype=dtype)
        self.out = nn.Linear(hidden, c_out, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(self.u(x) * self.v(x))


class _LightConvPredictor(nn.Module):
    """
    Lightweight spatial predictor on channel-last tensors.

    kind:
      - conv1x1: 1x1 -> HGLU -> 1x1
      - dwsep_conv: pw -> HGLU -> depthwise 3x3 -> HGLU -> pw
    """
    def __init__(self, *, c_in: int, c_out: int, hidden: int, dtype: torch.dtype, kind: str):
        super().__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.hidden = hidden
        self.dtype = dtype
        self.kind = kind
        self.net: Optional[nn.Module] = None

    def _build(self, spatial_dims: int):
        if spatial_dims == 1:
            Conv = nn.Conv1d
        elif spatial_dims == 2:
            Conv = nn.Conv2d
        elif spatial_dims == 3:
            Conv = nn.Conv3d
        else:
            raise ValueError(f"Unsupported spatial_dims={spatial_dims}")

        if self.kind == "conv1x1":
            self.net = nn.Sequential(
                Conv(self.c_in, self.hidden, kernel_size=1, bias=True, dtype=self.dtype),
                HGLU(4.0),
                Conv(self.hidden, self.c_out, kernel_size=1, bias=True, dtype=self.dtype),
            )
        elif self.kind == "dwsep_conv":
            self.net = nn.Sequential(
                Conv(self.c_in, self.hidden, kernel_size=1, bias=True, dtype=self.dtype),
                HGLU(4.0),
                Conv(self.hidden, self.hidden, kernel_size=3, padding=1, groups=self.hidden, bias=True, dtype=self.dtype),
                HGLU(4.0),
                Conv(self.hidden, self.c_out, kernel_size=1, bias=True, dtype=self.dtype),
            )
        else:
            raise ValueError(f"Unsupported conv predictor kind={self.kind!r}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        spatial_dims = x.ndim - 2
        if self.net is None:
            self._build(spatial_dims)
        x_cf = _to_channel_first(x)
        y_cf = self.net(x_cf)  # type: ignore[arg-type]
        return _to_channel_last(y_cf)


def build_edge_module(
    *,
    kind: str,
    c_in: int,
    c_out: int,
    hidden_mult: float,
    dtype: torch.dtype,
) -> nn.Module:
    hidden = max(1, int(round(hidden_mult * c_out)))
    kind = kind.lower()
    if kind in {"identity", "none"}:
        if c_in != c_out:
            return _LinearPredictor(c_in=c_in, c_out=c_out, dtype=dtype)
        return nn.Identity()
    if kind == "linear":
        return _LinearPredictor(c_in=c_in, c_out=c_out, dtype=dtype)
    if kind == "mlp":
        return _MLPPredictor(c_in=c_in, c_out=c_out, hidden=hidden, depth=2, dtype=dtype)
    if kind == "bottleneck_mlp":
        return _MLPPredictor(c_in=c_in, c_out=c_out, hidden=hidden, depth=3, dtype=dtype)
    if kind == "res_mlp":
        return _ResidualMLPPredictor(c_in=c_in, c_out=c_out, hidden=hidden, dtype=dtype)
    if kind == "bilinear_mlp":
        return _BilinearMLPPredictor(c_in=c_in, c_out=c_out, hidden=hidden, dtype=dtype)
    if kind in {"conv1x1", "dwsep_conv"}:
        return _LightConvPredictor(c_in=c_in, c_out=c_out, hidden=hidden, dtype=dtype, kind=kind)
    raise ValueError(f"Unsupported predictor/projector kind={kind!r}")


class ZonePredictiveGraph(nn.Module):
    """Predictive graph over zone pyramid with optional per-edge projector paths."""
    def __init__(
        self,
        *,
        num_scales: int,
        c_model: int,
        edge_specs: Sequence[EdgeSpecLike] = (
            (("structure",), "boundary"),
            (("content", "structure"), "label"),
        ),
        dtype_idx: FPDTypeIdx = 64,
    ):
        super().__init__()
        self.dtype = get_float_dtype(dtype_idx)
        self.num_scales = int(num_scales)
        self.c_model = int(c_model)
        self.edge_specs = normalize_edge_specs(edge_specs)

        self.predictors = nn.ModuleList()
        self.online_projectors = nn.ModuleList()
        self.target_projectors = nn.ModuleList()
        for _ in range(self.num_scales):
            scale_pred = nn.ModuleDict()
            scale_proj_online = nn.ModuleDict()
            scale_proj_target = nn.ModuleDict()
            for spec in self.edge_specs:
                k = spec.key()
                scale_pred[k] = build_edge_module(
                    kind=spec.predictor_kind,
                    c_in=len(spec.srcs) * self.c_model,
                    c_out=self.c_model,
                    hidden_mult=spec.hidden_mult,
                    dtype=self.dtype,
                )
                need_projector = (spec.projector_kind is not None) or (spec.loss_type.lower() in SSL_PROJECTOR_LOSSES)
                if need_projector:
                    p_dim = int(spec.projector_dim) if spec.projector_dim is not None else self.c_model
                    p_kind = spec.projector_kind or "mlp"
                    p_hidden = spec.projector_hidden_mult

                    tp_kind = spec.target_projector_kind or p_kind
                    tp_dim = int(spec.target_projector_dim) if spec.target_projector_dim is not None else p_dim
                    tp_hidden = spec.target_projector_hidden_mult if spec.target_projector_hidden_mult is not None else p_hidden
                    if tp_dim != p_dim:
                        raise ValueError(f"Projector dims must match for edge {k}: got online={p_dim}, target={tp_dim}")

                    scale_proj_online[k] = build_edge_module(
                        kind=p_kind,
                        c_in=self.c_model,
                        c_out=p_dim,
                        hidden_mult=p_hidden,
                        dtype=self.dtype,
                    )
                    scale_proj_target[k] = build_edge_module(
                        kind=tp_kind,
                        c_in=self.c_model,
                        c_out=tp_dim,
                        hidden_mult=tp_hidden,
                        dtype=self.dtype,
                    )

            self.predictors.append(scale_pred)
            self.online_projectors.append(scale_proj_online)
            self.target_projectors.append(scale_proj_target)

    def forward(self, zone_pyr: list[dict[str, torch.Tensor]]) -> dict[str, object]:
        if len(zone_pyr) != self.num_scales:
            raise ValueError(f"Expected {self.num_scales} scales, got {len(zone_pyr)}")

        preds: list[dict[str, torch.Tensor]] = []
        proj_preds: list[dict[str, torch.Tensor]] = []
        proj_tgts: list[dict[str, torch.Tensor]] = []
        edge_meta: dict[str, dict[str, object]] = {}
        for s in range(self.num_scales):
            scale_preds: dict[str, torch.Tensor] = {}
            scale_proj_preds: dict[str, torch.Tensor] = {}
            scale_proj_tgts: dict[str, torch.Tensor] = {}
            zones = zone_pyr[s]
            for spec in self.edge_specs:
                k = spec.key()
                edge_meta[k] = {
                    "srcs": spec.srcs,
                    "tgt": spec.tgt,
                    "weight": spec.weight,
                    "name": spec.name,
                    "predictor_kind": spec.predictor_kind,
                    "hidden_mult": spec.hidden_mult,
                    "loss_type": spec.loss_type,
                    "loss_kwargs": dict(spec.loss_kwargs or {}),
                    "detach_target": spec.detach_target,
                    "edge_norm": spec.edge_norm,
                    "projector_norm": spec.projector_norm,
                    "target_projector_norm": spec.target_projector_norm,
                    "projector_kind": spec.projector_kind,
                    "projector_dim": spec.projector_dim,
                    "projector_hidden_mult": spec.projector_hidden_mult,
                    "target_projector_kind": spec.target_projector_kind,
                    "target_projector_dim": spec.target_projector_dim,
                    "target_projector_hidden_mult": spec.target_projector_hidden_mult,
                    "projector_share_mode": spec.projector_share_mode,
                    "projector_reg_type": spec.projector_reg_type,
                    "projector_reg_weight": spec.projector_reg_weight,
                    "projector_reg_kwargs": dict(spec.projector_reg_kwargs or {}),
                }
                x = torch.cat([zones[n] for n in spec.srcs], dim=-1)
                pred = self.predictors[s][k](x)
                scale_preds[k] = pred
                if k in self.online_projectors[s]:
                    scale_proj_preds[k] = self.online_projectors[s][k](pred)
                if k in self.target_projectors[s]:
                    scale_proj_tgts[k] = self.target_projectors[s][k](zones[spec.tgt])
            preds.append(scale_preds)
            proj_preds.append(scale_proj_preds)
            proj_tgts.append(scale_proj_tgts)
        return {"preds": preds, "proj_preds": proj_preds, "proj_tgts": proj_tgts, "edge_meta": edge_meta}
