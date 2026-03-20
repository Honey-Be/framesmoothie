import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union, Callable

from torchutils.decorators import auxloss
from framesmoothie.heads import SemanticFPNHead, RS9InstanceHead
from framesmoothie.decoder import RS9Decoder, RS9DecoderLayer
from framesmoothie.panoptic import panoptic_fusion
from framesmoothie.hmc import HMCCalibrator
from framesmoothie.adapters.base import ModuleAdapterBase
from framesmoothie.adapters.lrca import LRCAAdapter
from framesmoothie.specs import PreBridgeMapSpec, resolve_pre_bridge_map
from framesmoothie.zoning import PreBridgeProjector, DualViewZoningBlock
from framesmoothie.predictor import ZonePredictiveGraph, ZoneEdgeSpec

from s9.base import FPDTypeIdx, get_float_dtype
from s9.rs9_modules import RS9Layer
from s9.activations.real.hglu import HGLU

from s9.activations.complex.stable_cardioid import StableComplexCardioid
from s9.modules import S9Layer


def _to_channel_last(x_cf: torch.Tensor) -> torch.Tensor:
    # [B,C,*S] -> [B,*S,C]
    spatial_dims = x_cf.ndim - 2
    perm = [0] + list(range(2, 2 + spatial_dims)) + [1]
    return x_cf.permute(*perm)

def _to_channel_first(x_cl: torch.Tensor) -> torch.Tensor:
    # [B,*S,C] -> [B,C,*S]
    spatial_dims = x_cl.ndim - 2
    perm = [0, spatial_dims + 1] + list(range(1, 1 + spatial_dims))
    return x_cl.permute(*perm)

def _avg_pool_nd(x_cf: torch.Tensor, factor: int) -> torch.Tensor:
    # x_cf: [B,C,*S]
    sd = x_cf.ndim - 2
    if sd == 1:
        return F.avg_pool1d(x_cf, kernel_size=factor, stride=factor)
    if sd == 2:
        return F.avg_pool2d(x_cf, kernel_size=factor, stride=factor)
    if sd == 3:
        return F.avg_pool3d(x_cf, kernel_size=factor, stride=factor)
    raise ValueError(f"Unsupported spatial_dims={sd}")

def _interp_nd(x: torch.Tensor, size: Tuple[int, ...], spatial_dims: int) -> torch.Tensor:
    if spatial_dims == 1:
        return F.interpolate(x, size=size, mode="linear", align_corners=False)
    if spatial_dims == 2:
        return F.interpolate(x, size=size, mode="bilinear", align_corners=False)
    if spatial_dims == 3:
        return F.interpolate(x, size=size, mode="trilinear", align_corners=False)
    raise ValueError(f"Unsupported spatial_dims={spatial_dims}")

def _resize_masks_nearest(masks: torch.Tensor, size: Tuple[int, ...]) -> torch.Tensor:
    # masks: [N,*S] or [B,N,*S] bool/float -> resized same dtype float
    if masks.ndim < 3:
        raise ValueError("masks must be at least 3D")
    sd = masks.ndim - 2  # after [B,N,...] or [N,...] ??? we'll handle
    if masks.ndim >= 4:
        # treat as [B,N,*S]
        B, N = masks.shape[:2]
        x = masks.float().reshape(B * N, 1, *masks.shape[2:])
        x = _interp_nd(x, size=size, spatial_dims=len(size))
        x = x.reshape(B, N, *size)
        return x
    else:
        # [N,*S]
        N = masks.shape[0]
        x = masks.float().reshape(N, 1, *masks.shape[1:])
        x = _interp_nd(x, size=size, spatial_dims=len(size))
        x = x.reshape(N, *size)
        return x

def infer_inverse_transform(T: nn.Module, **kwargs) -> nn.Module:
    """
    Convenience helper: infer inverse class by name convention 'Inverse' + forward_class_name
    in the same python module as T's class.
    If you prefer explicit wiring, pass transform_inv directly.
    """
    mod = __import__(T.__class__.__module__, fromlist=[T.__class__.__name__])
    inv_name = "Inverse" + T.__class__.__name__
    if not hasattr(mod, inv_name):
        raise AttributeError(f"Could not find inverse class {inv_name} in module {mod.__name__}")
    inv_cls = getattr(mod, inv_name)
    return inv_cls(**kwargs)


class S9Stack(nn.Module):
    """
    Minimal complex-valued S9 encoder stack.
    Builds a stack of S9Layer blocks lazily on first forward based on the input spatial shape.
    """
    def __init__(
        self,
        *,
        c_model: int,
        spatial_dims: int,
        depth: int = 4,
        eps: float = 1e-6,
        dtype_idx: FPDTypeIdx = 64,
        gen_activation: Callable[[int, float, FPDTypeIdx], nn.Module] = StableComplexCardioid,
    ):
        super().__init__()
        self.c_model = c_model
        self.depth = depth
        self.eps = eps
        self.dtype_idx = dtype_idx
        self.gen_activation = gen_activation
        self.layers: nn.ModuleList = nn.ModuleList([
            S9Layer(
                d_model=self.c_model,
                spatial_dims=spatial_dims,
                gen_activation=self.gen_activation,
                eps=self.eps,
                dtype_idx=self.dtype_idx,
            )
            for _ in range(self.depth)
        ])

    def forward(self, z_cf: torch.Tensor) -> torch.Tensor:
        # z_cf: [B,C,*S] complex
        out = z_cf
        for layer in self.layers:  # type: ignore[union-attr]
            out = layer(out)
        return out


class FrameSmoothiePanopticModel(nn.Module):
    """
    framesmoothie end-to-end model:
      real x -> T (real->complex) -> complex S9 encoder -> T^{-1} (complex->real) -> dual heads

    Inputs:
      x_real_cf: [B,C,*S] real (channel-first)

    Outputs (dict):
      sem_logits: [B,C_sem,*S0] channel-first (S0 is top resolution)
      inst: dict with class_logits/mask_logits (+ upsampled for fusion), etc.
    """
    def __init__(
        self,
        *,
        # transforms (explicit modules; you can import from s9.transforms.* directly)
        transform_fwd: nn.Module,
        transform_inv: Optional[nn.Module] = None,
        # encoder
        encoder: Optional[nn.Module] = None,
        c_model: int,
        enc_c_model: Optional[int] = None,
        spatial_dims: int,
        # semantic
        num_semantic_classes: int,
        semantic_hidden_dim: int = 256,
        # instance
        num_instance_classes: int,
        q_dim: int,
        num_queries: int = 100,
        decoder_layers: int = 4,
        instance_mask_dim: int = 256,
        instance_level: int = 0,  # 0 = highest-res; >0 uses downsampled pyramid level
        # rs9 decoder params
        gate_dim: int = 64,
        v_dim: Optional[int] = None,
        ffn_mult: int = 4,
        dropout: float = 0.0,
        eps: float = 1e-6,
        rs9_eps: float = 1e-6,
        dtype_idx: FPDTypeIdx = 64,
        lambda_gate_entropy: float = 0.0,
        lambda_gate_competition: float = 0.0,
        # gate fusion in head
        gate_fuse: float = 0.0,
        # aux-loss aggregation
                aux_weight_default: float = 1.0,
        # LRCA adapter (optional)
        use_lrca: bool = False,
        adapter: Optional[ModuleAdapterBase] = None,
        lrca_rank_shared: int = 8,
        lrca_rank_private: int = 4,
        lrca_fmlm_rank: int = 8,
        lrca_fmlm_eta: float = 0.1,
        lrca_domain_dim: int = 1,
        lrca_task_id_dim: int = 1,
        return_diag_default: bool = False,
        # pre-bridge projector (optional)
        pre_bridge_map: Optional[str | PreBridgeMapSpec] = None,
        pre_bridge_channels: Optional[int] = None,
        # dual-view zoning (optional)
        use_zoning: bool = False,
        zone_names: tuple[str, ...] = ("content", "structure", "label", "boundary"),
        semantic_zone_names: tuple[str, ...] = ("content", "structure", "boundary"),
        instance_zone_names: tuple[str, ...] = ("content", "label", "boundary"),
        semantic_zone_weights: Optional[dict[str, float]] = None,
        instance_zone_weights: Optional[dict[str, float]] = None,
        # zone-wise predictive loss (optional)
        use_zone_prediction: bool = False,
        zone_pred_edges: tuple[ZoneEdgeSpec | tuple[tuple[str, ...], str] | tuple[tuple[str, ...], str, float], ...] = ((("structure",), "boundary"), (("content", "structure"), "label")),

    ):
        super().__init__()
        self.dtype = get_float_dtype(dtype_idx)
        self.c_model = int(c_model)                # decoder / post-bridge channel dim
        self.enc_c_model = int(enc_c_model) if enc_c_model is not None else int(c_model)  # encoder / pre-bridge channel dim
        self.spatial_dims = spatial_dims
        self.instance_level = int(instance_level)
        self.aux_weight_default = float(aux_weight_default)
        self.return_diag_default = bool(return_diag_default)
        self.adapter: Optional[ModuleAdapterBase] = adapter
        if use_lrca and self.adapter is None:
            # context = GAP(real_feature) ⊕ domain_id
            ctx_dim = int(c_model + lrca_domain_dim)
            self.adapter = LRCAAdapter(ctx_dim=ctx_dim, task_id_dim=lrca_task_id_dim, rank_shared=lrca_rank_shared, rank_private=lrca_rank_private, fmlm_rank=lrca_fmlm_rank, fmlm_eta=lrca_fmlm_eta, dtype_idx=dtype_idx)

        self.T = transform_fwd
        self.T_inv = transform_inv if transform_inv is not None else infer_inverse_transform(transform_fwd)

        # build pyramid levels by pooling in real domain
        self.num_pyr_levels = 4  # 1,2,4,8

        self.encoder = encoder if encoder is not None else S9Stack(c_model=self.enc_c_model, depth=4, eps=eps, dtype_idx=dtype_idx, spatial_dims=spatial_dims)

        # Optional pre-bridge realification of zL for future dual-view zoning
        self.pre_bridge_map = resolve_pre_bridge_map(pre_bridge_map)
        self.pre_bridge_channels = int(pre_bridge_channels) if pre_bridge_channels is not None else int(c_model)
        self.pre_bridge_projector: Optional[PreBridgeProjector]
        if self.pre_bridge_map is not None:
            self.pre_bridge_projector = PreBridgeProjector(
                map_spec=self.pre_bridge_map,
                in_channels=self.enc_c_model,
                out_channels=self.pre_bridge_channels,
                spatial_dims=spatial_dims,
                dtype_idx=dtype_idx,
            )
        else:
            self.pre_bridge_projector = None

        self.use_zoning = bool(use_zoning)
        if self.use_zoning:
            if self.pre_bridge_projector is None:
                raise ValueError("use_zoning=True requires a pre_bridge_map / pre_bridge_projector.")
            self.zoning_block = DualViewZoningBlock(
                num_scales=self.num_pyr_levels,
                c_pre=self.pre_bridge_channels,
                c_post=self.c_model,
                c_out=self.c_model,
                zone_names=zone_names,
                semantic_zone_names=semantic_zone_names,
                instance_zone_names=instance_zone_names,
                semantic_zone_weights=semantic_zone_weights,
                instance_zone_weights=instance_zone_weights,
                dtype_idx=dtype_idx,
            )
        else:
            self.zoning_block = None

        self.use_zone_prediction = bool(use_zone_prediction)
        if self.use_zone_prediction:
            if not self.use_zoning:
                raise ValueError("use_zone_prediction=True requires use_zoning=True.")
            self.zone_predictor = ZonePredictiveGraph(
                num_scales=self.num_pyr_levels,
                c_model=self.c_model,
                edge_specs=zone_pred_edges,
                dtype_idx=dtype_idx,
            )
        else:
            self.zone_predictor = None

        self.semantic_head = SemanticFPNHead(
            num_classes=num_semantic_classes,
            c_model=self.c_model,
            spatial_dims=spatial_dims,
            hidden_dim=semantic_hidden_dim,
            dropout=dropout,
            dtype_idx=dtype_idx,
        )

        # decoder stack
        layers = []
        for _ in range(decoder_layers):
            layers.append(RS9DecoderLayer(
                c_model=c_model,
                q_dim=q_dim,
                spatial_dims=spatial_dims,
                gate_dim=gate_dim,
                v_dim=v_dim,
                ffn_mult=ffn_mult,
                dropout=dropout,
                eps=eps,
                rs9_eps=rs9_eps,
                return_masks=True,
                rs9=None,
                dtype_idx=dtype_idx,
                lambda_gate_entropy=lambda_gate_entropy,
                lambda_gate_competition=lambda_gate_competition,
                    adapter=self.adapter,
                post_ffn=True,
            ))
        self.decoder = RS9Decoder(layers)

        self.instance_head = RS9InstanceHead(
            c_model=self.c_model,
            q_dim=q_dim,
            num_classes=num_instance_classes,
            mask_dim=instance_mask_dim,
            dropout=dropout,
            dtype_idx=dtype_idx,
            gate_fuse=gate_fuse,
            normalize_embeddings=True,
            with_scores=True,
                adapter=self.adapter,
        )

        self.query_embed = nn.Parameter(torch.randn(num_queries, q_dim, dtype=self.dtype))

        # aux-loss aggregation root
        self._reg_loss = torch.tensor(0.0, dtype=self.dtype)

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
        losses: List[torch.Tensor] = []
        for m in self.modules():
            if hasattr(m, "_reg_loss"):
                rl = getattr(m, "_reg_loss")
                if isinstance(rl, torch.Tensor):
                    losses.append(rl.reshape(()))
        out = torch.tensor(0.0, dtype=self.dtype, device=self._reg_loss.device) if not losses else aggregate(torch.stack(losses))
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
    
    

    def _make_pyramid(self, x_cf: torch.Tensor) -> List[torch.Tensor]:
        # x_cf: [B,C,*S] real
        feats_cf = [x_cf]
        cur = x_cf
        for i in range(1, self.num_pyr_levels):
            cur = _avg_pool_nd(cur, factor=2)
            feats_cf.append(cur)
        feats_cl = [_to_channel_last(f) for f in feats_cf]  # [B,*S,C]
        return feats_cl

    def forward(
        self,
        x_real_cf: torch.Tensor,
        *,
        q0: Optional[torch.Tensor] = None,
        domain_id: Optional[torch.Tensor] = None,
        return_panoptic: bool = False,
        return_diag: Optional[bool] = None,
        panoptic_cfg: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        # 1) real -> complex
        z0 = self.T(x_real_cf)              # [B,C,*S] complex
        # 2) complex encoder
        zL = self.encoder(z0)               # [B,C,*S] complex
        # 3) complex -> real via T^{-1}
        x_dec_cf = self.T_inv(zL)           # [B,C,*S] real

        # adapter context (for LRCA gates / FMLM FiLM)
        if self.adapter is not None:
            B = x_dec_cf.shape[0]
            # GAP over spatial dims -> [B,C]
            gap = x_dec_cf.flatten(2).mean(dim=-1)  # [B,C]
            if domain_id is None:
                d = torch.zeros((B, 1), dtype=gap.dtype, device=gap.device)
            else:
                d = domain_id.reshape(B, -1).to(dtype=gap.dtype, device=gap.device)
            ctx = torch.cat([gap, d], dim=-1)
            self.adapter.set_context(ctx)


        # 4) pyramids
        pre_pyr_cl = None
        if self.pre_bridge_projector is not None:
            pre0_cf = self.pre_bridge_projector(zL)   # [B,C_pre,*S] real
            pre_pyr_cl = self._make_pyramid(pre0_cf)  # list of [B,*S_i,C_pre]

        pyr_cl = self._make_pyramid(x_dec_cf)  # list of [B,*S_i,C]

        zone_out = None
        sem_pyr_cl = pyr_cl
        inst_pyr_cl = pyr_cl
        if self.zoning_block is not None:
            if pre_pyr_cl is None:
                raise RuntimeError("zoning_block requires pre_pyr_cl, but pre_pyr_cl is None")
            zone_out = self.zoning_block(pre_pyr_cl, pyr_cl)
            sem_pyr_cl = zone_out["sem_pyr"]
            inst_pyr_cl = zone_out["inst_pyr"]

        zone_pred = None
        if self.zone_predictor is not None:
            if zone_out is None:
                raise RuntimeError("zone_predictor requires zone_out, but zone_out is None")
            zone_pred = self.zone_predictor(zone_out["zones"])

        sem_logits = self.semantic_head(sem_pyr_cl)  # [B,C_sem,*S0] (channel-first)

        # instance branch feature level
        lvl = max(0, min(self.instance_level, len(inst_pyr_cl) - 1))
        x_inst_cl = inst_pyr_cl[lvl]
        spatial_inst = tuple(x_inst_cl.shape[1:-1])

        # init queries
        B = x_inst_cl.shape[0]
        if q0 is None:
            q = self.query_embed.unsqueeze(0).expand(B, -1, -1)
        else:
            q = q0

        dec = self.decoder(x_inst_cl, q)
        q_final = dec["q"]
        mask_gate_last = dec["mask_logits_all"][-1] if len(dec["mask_logits_all"]) > 0 else None

        inst = self.instance_head(x_inst_cl, q_final, mask_logits_gate_last=mask_gate_last)
        inst["q"] = q_final
        inst["mask_logits_all"] = dec["mask_logits_all"]

        # upsample instance masks to semantic resolution (for fusion/HMC), if needed
        spatial_sem = tuple(sem_logits.shape[2:])
        if spatial_inst != spatial_sem:
            # upsample logits (not sigmoid)
            x = inst["mask_logits"].unsqueeze(2)  # [B,K,1,*S_inst]
            x = x.reshape(B * x.shape[1], 1, *spatial_inst)
            x_up = _interp_nd(x, size=spatial_sem, spatial_dims=self.spatial_dims)
            x_up = x_up.reshape(B, -1, *spatial_sem)
            inst["mask_logits_up"] = x_up
        else:
            inst["mask_logits_up"] = inst["mask_logits"]

        out: Dict[str, Any] = {"sem_logits": sem_logits, "inst": inst, "pyr": pyr_cl, "pre_pyr": pre_pyr_cl, "zones": zone_out, "zone_pred": zone_pred}

        # Surface highest-resolution boundary zone for HMC.
        if isinstance(zone_out, dict) and "zones" in zone_out and len(zone_out["zones"]) > 0:
            z0 = zone_out["zones"][0]
            if isinstance(z0, dict) and "boundary" in z0:
                out["boundary_features"] = z0["boundary"]

        rd = self.return_diag_default if return_diag is None else bool(return_diag)
        if rd and self.adapter is not None:
            out["diag"] = self.adapter.hub.get_diagnostics()


        if return_panoptic:
            cfg = panoptic_cfg or {}
            out["panoptic"] = panoptic_fusion(
                sem_logits=sem_logits,
                inst_class_logits=inst["class_logits"],
                inst_mask_logits=inst["mask_logits_up"],
                inst_scores=inst.get("scores", None),
                **cfg,
            )

        return out


def make_ema_teacher(student: nn.Module) -> nn.Module:
    teacher = copy.deepcopy(student)
    for p in teacher.parameters():
        p.requires_grad_(False)
    teacher.eval()
    return teacher


@torch.no_grad()
def ema_update(teacher: nn.Module, student: nn.Module, beta: float = 0.999):
    # teacher = beta*teacher + (1-beta)*student
    for tp, sp in zip(teacher.parameters(), student.parameters()):
        tp.data.mul_(beta).add_(sp.data, alpha=(1.0 - beta))