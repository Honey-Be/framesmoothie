from __future__ import annotations

import torch
import torch.nn.functional as F
from typing import Any, Optional

def _flatten_features(x: torch.Tensor) -> torch.Tensor:
    # [B,*S,C] -> [N,C]
    if x.ndim < 2:
        raise ValueError("Expected at least 2D tensor")
    return x.reshape(-1, x.shape[-1])

def _apply_norm(x: torch.Tensor, kind: str, eps: float = 1e-8) -> torch.Tensor:
    k = (kind or "none").lower()
    if k in {"none", ""}:
        return x
    if k == "l2":
        return F.normalize(x, dim=-1, eps=eps)
    if k == "layernorm":
        return F.layer_norm(x, (x.shape[-1],))
    if k == "standardize":
        mu = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True, unbiased=False).clamp_min(eps)
        return (x - mu) / std
    raise ValueError(f"Unsupported normalization kind={kind!r}")

def _kl_div_channel_last(pred: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
    return F.kl_div(
        F.log_softmax(pred, dim=-1),
        F.softmax(tgt, dim=-1),
        reduction="none",
    ).sum(dim=-1).mean()

def _cosine_loss(pred: torch.Tensor, tgt: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    cos = F.cosine_similarity(pred, tgt, dim=-1, eps=eps)
    return (1.0 - cos).mean()

def _negative_cosine_loss(pred: torch.Tensor, tgt: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    cos = F.cosine_similarity(pred, tgt, dim=-1, eps=eps)
    return (-cos).mean()

def _symmetric_simsiam_loss(pred: torch.Tensor, tgt: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    # symmetric variant: both sides predict each other with stop-grad on the opposite branch
    return 0.5 * (_negative_cosine_loss(pred, tgt.detach(), eps=eps) + _negative_cosine_loss(tgt, pred.detach(), eps=eps))

def _off_diagonal(x: torch.Tensor) -> torch.Tensor:
    n, m = x.shape
    if n != m:
        raise ValueError("Expected square matrix")
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def _barlow_twins_loss(pred: torch.Tensor, tgt: torch.Tensor, lambda_offdiag: float = 5e-3, eps: float = 1e-4) -> torch.Tensor:
    z1 = _flatten_features(pred)
    z2 = _flatten_features(tgt)
    if z1.shape[0] < 2:
        return F.mse_loss(pred, tgt)
    z1 = (z1 - z1.mean(dim=0, keepdim=True)) / (z1.std(dim=0, keepdim=True, unbiased=False) + eps)
    z2 = (z2 - z2.mean(dim=0, keepdim=True)) / (z2.std(dim=0, keepdim=True, unbiased=False) + eps)
    n = z1.shape[0]
    c = (z1.T @ z2) / n
    on_diag = torch.diagonal(c).add_(-1.0).pow_(2).sum()
    off_diag = _off_diagonal(c).pow_(2).sum()
    return on_diag + lambda_offdiag * off_diag


def _vicreg_loss(
    pred: torch.Tensor,
    tgt: torch.Tensor,
    sim_coeff: float = 25.0,
    std_coeff: float = 25.0,
    cov_coeff: float = 1.0,
    gamma: float = 1.0,
    eps: float = 1e-4,
) -> torch.Tensor:
    x = _flatten_features(pred)
    y = _flatten_features(tgt)
    if x.shape[0] < 2:
        return F.mse_loss(pred, tgt)

    repr_loss = F.mse_loss(x, y)

    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)

    std_x = torch.sqrt(x.var(dim=0, unbiased=False) + eps)
    std_y = torch.sqrt(y.var(dim=0, unbiased=False) + eps)
    std_loss = torch.mean(F.relu(gamma - std_x)) + torch.mean(F.relu(gamma - std_y))

    n, d = x.shape
    cov_x = (x.T @ x) / max(n - 1, 1)
    cov_y = (y.T @ y) / max(n - 1, 1)
    cov_loss = _off_diagonal(cov_x).pow_(2).sum().div(d) + _off_diagonal(cov_y).pow_(2).sum().div(d)

    return sim_coeff * repr_loss + std_coeff * std_loss + cov_coeff * cov_loss

def _variance_reg(x: torch.Tensor, gamma: float = 1.0, eps: float = 1e-4) -> torch.Tensor:
    z = _flatten_features(x)
    if z.shape[0] < 2:
        return torch.tensor(0.0, device=x.device, dtype=x.dtype)
    z = z - z.mean(dim=0, keepdim=True)
    std = torch.sqrt(z.var(dim=0, unbiased=False) + eps)
    return torch.mean(F.relu(gamma - std))


def _covariance_reg(x: torch.Tensor) -> torch.Tensor:
    z = _flatten_features(x)
    if z.shape[0] < 2:
        return torch.tensor(0.0, device=x.device, dtype=x.dtype)
    z = z - z.mean(dim=0, keepdim=True)
    n, d = z.shape
    cov = (z.T @ z) / max(n - 1, 1)
    return _off_diagonal(cov).pow_(2).sum().div(d)


def _projector_regularization(
    pred: torch.Tensor,
    tgt: torch.Tensor,
    *,
    reg_type: str,
    reg_kwargs: dict[str, float] | None = None,
) -> torch.Tensor:
    reg_kwargs = reg_kwargs or {}
    rt = reg_type.lower()
    if rt == "variance":
        gamma = float(reg_kwargs.get("gamma", 1.0))
        eps = float(reg_kwargs.get("eps", 1e-4))
        return _variance_reg(pred, gamma=gamma, eps=eps) + _variance_reg(tgt, gamma=gamma, eps=eps)
    if rt == "covariance":
        return _covariance_reg(pred) + _covariance_reg(tgt)
    if rt == "var_cov":
        gamma = float(reg_kwargs.get("gamma", 1.0))
        eps = float(reg_kwargs.get("eps", 1e-4))
        return (
            _variance_reg(pred, gamma=gamma, eps=eps)
            + _variance_reg(tgt, gamma=gamma, eps=eps)
            + _covariance_reg(pred)
            + _covariance_reg(tgt)
        )
    if rt == "l2":
        return pred.pow(2).mean() + tgt.pow(2).mean()
    raise ValueError(f"Unsupported projector_reg_type={reg_type!r}")


def compute_edge_loss(
    pred: torch.Tensor,
    tgt: torch.Tensor,
    *,
    loss_type: str,
    reduction: str = "mean",
    loss_kwargs: dict[str, float] | None = None,
) -> torch.Tensor:
    loss_kwargs = loss_kwargs or {}
    lt = loss_type.lower()

    if lt == "smooth_l1":
        beta = float(loss_kwargs.get("beta", 1.0))
        return F.smooth_l1_loss(pred, tgt, reduction=reduction, beta=beta)
    if lt == "mse":
        return F.mse_loss(pred, tgt, reduction=reduction)
    if lt == "l1":
        return F.l1_loss(pred, tgt, reduction=reduction)
    if lt == "huber":
        delta = float(loss_kwargs.get("delta", 1.0))
        return F.huber_loss(pred, tgt, reduction=reduction, delta=delta)
    if lt == "cosine":
        eps = float(loss_kwargs.get("eps", 1e-8))
        return _cosine_loss(pred, tgt, eps=eps)
    if lt == "neg_cosine":
        eps = float(loss_kwargs.get("eps", 1e-8))
        return _negative_cosine_loss(pred, tgt, eps=eps)
    if lt == "simsiam":
        eps = float(loss_kwargs.get("eps", 1e-8))
        return _negative_cosine_loss(pred, tgt, eps=eps)
    if lt in {"symmetric_simsiam", "sym_simsiam"}:
        eps = float(loss_kwargs.get("eps", 1e-8))
        return _symmetric_simsiam_loss(pred, tgt, eps=eps)
    if lt == "kl_div":
        return _kl_div_channel_last(pred, tgt)
    if lt == "barlow":
        lambda_offdiag = float(loss_kwargs.get("lambda_offdiag", 5e-3))
        eps = float(loss_kwargs.get("eps", 1e-4))
        return _barlow_twins_loss(pred, tgt, lambda_offdiag=lambda_offdiag, eps=eps)
    if lt == "vicreg":
        return _vicreg_loss(
            pred,
            tgt,
            sim_coeff=float(loss_kwargs.get("sim_coeff", 25.0)),
            std_coeff=float(loss_kwargs.get("std_coeff", 25.0)),
            cov_coeff=float(loss_kwargs.get("cov_coeff", 1.0)),
            gamma=float(loss_kwargs.get("gamma", 1.0)),
            eps=float(loss_kwargs.get("eps", 1e-4)),
        )
    raise ValueError(f"Unsupported loss_type={loss_type!r}")



def compute_zone_predictive_loss(
    *,
    zone_pred: dict[str, Any] | None,
    zones: list[dict[str, torch.Tensor]] | None,
    detach_target: bool = True,
    reduction: str = "mean",
    use_edge_weights: bool = True,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if zone_pred is None or zones is None:
        return torch.tensor(0.0), {}

    preds = zone_pred.get("preds", None)
    proj_preds = zone_pred.get("proj_preds", None)
    proj_tgts = zone_pred.get("proj_tgts", None)
    edge_meta = zone_pred.get("edge_meta", None)
    if preds is None or edge_meta is None:
        return torch.tensor(0.0), {}

    if proj_preds is None:
        proj_preds = [{} for _ in range(len(preds))]
    if proj_tgts is None:
        proj_tgts = [{} for _ in range(len(preds))]

    projector_losses = {"simsiam", "symmetric_simsiam", "sym_simsiam", "barlow", "vicreg"}

    per_edge: dict[str, list[torch.Tensor]] = {}
    total_terms: list[torch.Tensor] = []
    total_weights: list[torch.Tensor] = []

    for s, scale_preds in enumerate(preds):
        zone_dict = zones[s]
        scale_proj_preds = proj_preds[s] if proj_preds is not None else {}
        scale_proj_tgts = proj_tgts[s] if proj_tgts is not None else {}

        for k, pred_raw in scale_preds.items():
            meta = edge_meta[k]
            tgt_name = meta["tgt"]
            tgt_raw = zone_dict[tgt_name]

            loss_type = str(meta.get("loss_type", "smooth_l1"))
            loss_kwargs = dict(meta.get("loss_kwargs", {}) or {})

            use_projector_space = (
                loss_type.lower() in {"simsiam", "symmetric_simsiam", "sym_simsiam", "barlow", "vicreg"}
                and k in scale_proj_preds
                and k in scale_proj_tgts
            )

            if use_projector_space:
                pred = scale_proj_preds[k]
                tgt = scale_proj_tgts[k]

                pred = _apply_norm(pred, str(meta.get("projector_norm", "none")), eps=float(loss_kwargs.get("eps", 1e-8)))
                target_norm = meta.get("target_projector_norm", None)
                if target_norm is None:
                    target_norm = meta.get("projector_norm", "none")
                tgt = _apply_norm(tgt, str(target_norm), eps=float(loss_kwargs.get("eps", 1e-8)))
            else:
                pred = _apply_norm(pred_raw, str(meta.get("edge_norm", "none")), eps=float(loss_kwargs.get("eps", 1e-8)))
                tgt = _apply_norm(tgt_raw, str(meta.get("edge_norm", "none")), eps=float(loss_kwargs.get("eps", 1e-8)))

            detach_this = meta.get("detach_target", None)
            lt = loss_type.lower()
            if detach_this is None:
                if lt == "simsiam":
                    detach_flag = True
                elif lt in {"symmetric_simsiam", "sym_simsiam", "barlow", "vicreg"}:
                    detach_flag = False
                else:
                    detach_flag = detach_target
            else:
                detach_flag = bool(detach_this)

            if detach_flag:
                 tgt = tgt.detach()

            loss_main = compute_edge_loss(
                pred,
                tgt,
                loss_type=loss_type,
                reduction=reduction,
                loss_kwargs=loss_kwargs,
            )
            loss_total = loss_main

            # projector regularization: primarily for Barlow/VICReg projector paths
            reg_weight = float(meta.get("projector_reg_weight", 0.0))
            reg_type = meta.get("projector_reg_type", None)
            if use_projector_space and reg_weight > 0.0 and reg_type not in {None, ""} and lt in {"barlow", "vicreg"}:
                reg = _projector_regularization(
                    pred,
                    tgt,
                    reg_type=str(reg_type),
                    reg_kwargs=dict(meta.get("projector_reg_kwargs", {}) or {}),
                )
                loss_total = loss_total + reg_weight * reg
                per_edge.setdefault(f"{k}/reg", []).append(reg)

            per_edge.setdefault(k, []).append(loss_total)
            w = float(meta.get("weight", 1.0)) if use_edge_weights else 1.0
            total_terms.append(loss_total * w)
            total_weights.append(torch.tensor(w, device=loss_total.device, dtype=loss_total.dtype))
 
    if not total_terms:
        dev = None
        for z in zones:
            for t in z.values():
                dev = t.device
                break
            if dev is not None:
                break
        return torch.tensor(0.0, device=dev if dev is not None else "cpu"), {}

    denom = torch.stack(total_weights).sum().clamp_min(1e-8)
    total = torch.stack(total_terms).sum() / denom

    per_edge_mean = {k: torch.stack(v).mean() for k, v in per_edge.items()}
    return total, per_edge_mean
