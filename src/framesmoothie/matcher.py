import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

from framesmoothie.utils import linear_sum_assignment_torch as linear_sum_assignment


def _flatten_masks(mask: torch.Tensor) -> torch.Tensor:
    """
    mask: [N, *S] or [B,K,*S] -> flatten last dims
    returns [N, P] or [B,K,P]
    """
    return mask.flatten(start_dim=mask.ndim - (mask.ndim - 1)) if mask.ndim == 1 else mask.flatten(start_dim=mask.ndim - (mask.ndim - 1))


def flatten_spatial(x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, ...]]:
    """
    x: [B, K, *S] -> [B, K, P], spatial shape
    """
    spatial = tuple(x.shape[2:])
    return x.flatten(start_dim=2), spatial


def sigmoid_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    inputs: logits [..., P]
    targets: {0,1} [..., P]
    """
    prob = torch.sigmoid(inputs)
    ce = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p_t = prob * targets + (1 - prob) * (1 - targets)
    loss = ce * ((1 - p_t) ** gamma)
    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    return loss


def dice_loss(inputs: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    inputs: logits [..., P]
    targets: {0,1} [..., P]
    """
    probs = torch.sigmoid(inputs)
    num = 2 * (probs * targets).sum(dim=-1)
    den = (probs + targets).sum(dim=-1).clamp_min(eps)
    loss = 1 - (num / den)
    return loss.mean()


def sample_points_uncertainty(
    mask_logits: torch.Tensor,
    num_points: int,
    oversample_ratio: int = 3,
    importance_ratio: float = 0.75,
) -> torch.Tensor:
    """
    mask_logits: [N, *S] (no batch) or [B,K,*S] (we'll flatten outside).
    Returns indices into flattened spatial P: [N, num_points].
    Strategy: sample oversample_ratio*num_points random, pick top uncertain = |p-0.5| small.
    """
    # flatten
    flat = mask_logits.flatten(start_dim=1)  # [N, P]
    N, P = flat.shape
    num_sampled = min(P, num_points * oversample_ratio)
    rand_idx = torch.randint(0, P, (N, num_sampled), device=flat.device)

    sampled = flat.gather(1, rand_idx)  # [N, num_sampled]
    prob = sampled.sigmoid()
    uncertainty = -(prob - 0.5).abs()  # higher is more uncertain

    num_imp = int(num_points * importance_ratio)
    num_rand = num_points - num_imp

    topk = torch.topk(uncertainty, k=min(num_imp, num_sampled), dim=1).indices
    imp_idx = rand_idx.gather(1, topk)

    if num_rand > 0:
        rand_idx2 = torch.randint(0, P, (N, num_rand), device=flat.device)
        out = torch.cat([imp_idx, rand_idx2], dim=1)
    else:
        out = imp_idx

    # if oversampled < required, pad by random
    if out.shape[1] < num_points:
        pad = torch.randint(0, P, (N, num_points - out.shape[1]), device=flat.device)
        out = torch.cat([out, pad], dim=1)

    return out[:, :num_points]


@dataclass
class MatcherWeights:
    cls: float = 1.0
    focal: float = 1.0
    dice: float = 1.0


class HungarianMatcher:
    """
    Per-sample Hungarian matcher for (class + mask) predictions.
    """
    def __init__(
        self,
        weights: MatcherWeights = MatcherWeights(),
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        num_points: int = 12544,
    ):
        self.w = weights
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.num_points = num_points

    @torch.no_grad()
    def __call__(
        self,
        class_logits: torch.Tensor,  # [B,K,C+1]
        mask_logits: torch.Tensor,   # [B,K,*S]
        targets: List[Dict[str, torch.Tensor]],  # per batch: {"labels":[M], "masks":[M,*S]}
    ) -> List[Tuple[torch.Tensor, torch.Tensor]]:

        B, K, Cp1 = class_logits.shape
        out = []

        # flatten masks to [B,K,P]
        mask_flat, _ = flatten_spatial(mask_logits)

        for b in range(B):
            tgt_labels = targets[b]["labels"]   # [M]
            tgt_masks = targets[b]["masks"]     # [M,*S]
            M = tgt_labels.shape[0]
            if M == 0:
                out.append((torch.empty(0, dtype=torch.long), torch.empty(0, dtype=torch.long)))
                continue

            # class cost: -log prob of target class
            prob = class_logits[b].softmax(dim=-1)          # [K,C+1]
            cls_cost = -prob[:, tgt_labels]                 # [K,M]

            # mask cost on sampled points
            # pick points using predicted masks (use average over K? simplest: per GT use pred uncertainty from first pass)
            # We'll sample points from predicted mask logits of all K against all GT using a shared point set:
            # Use uncertainty from max over K (cheap-ish):
            pred_for_sampling = mask_flat[b].max(dim=0).values.unsqueeze(0)  # [1,P]
            # fake N=1 input expected by sampler; returns [1,num_points]
            point_idx = sample_points_uncertainty(pred_for_sampling, self.num_points)[0]  # [num_points]

            # gather predicted logits at points: [K, num_points]
            pred_pts = mask_flat[b].gather(1, point_idx.unsqueeze(0).expand(K, -1))
            # gather GT masks at points: [M, num_points]
            tgt_flat = tgt_masks.flatten(start_dim=1)  # [M,P]
            tgt_pts = tgt_flat.gather(1, point_idx.unsqueeze(0).expand(M, -1))

            # focal/dice costs: produce [K,M]
            # Expand to [K,M,num_points] via broadcasting
            pred_k = pred_pts.unsqueeze(1)      # [K,1,Pp]
            tgt_m = tgt_pts.unsqueeze(0)        # [1,M,Pp]

            # focal cost (mean over points)
            prob_k = pred_k.sigmoid()
            ce = F.binary_cross_entropy_with_logits(pred_k, tgt_m, reduction="none")
            p_t = prob_k * tgt_m + (1 - prob_k) * (1 - tgt_m)
            focal = ce * ((1 - p_t) ** self.focal_gamma)
            alpha_t = self.focal_alpha * tgt_m + (1 - self.focal_alpha) * (1 - tgt_m)
            focal = (alpha_t * focal).mean(dim=-1)  # [K,M]

            # dice cost
            probs = prob_k
            num = 2 * (probs * tgt_m).sum(dim=-1)
            den = (probs + tgt_m).sum(dim=-1).clamp_min(1e-6)
            dice = 1 - (num / den)  # [K,M]

            cost = self.w.cls * cls_cost + self.w.focal * focal + self.w.dice * dice
            cost = cost.detach().cpu()

            row_ind, col_ind = linear_sum_assignment(cost)
            out.append((torch.as_tensor(row_ind, dtype=torch.long), torch.as_tensor(col_ind, dtype=torch.long)))

        return out


class SetCriterion(nn.Module):
    """
    Computes losses given predictions and matched targets (DETR-style).
    """
    def __init__(
        self,
        num_classes: int,
        matcher: HungarianMatcher,
        eos_coef: float = 0.1,
        loss_weights: Dict[str, float] = None,
        focal_alpha: float = 0.25,
        focal_gamma: float = 2.0,
        num_points: int = 12544,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.matcher = matcher
        self.eos_coef = eos_coef
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
        self.num_points = num_points
        self.loss_weights = loss_weights or {"loss_ce": 1.0, "loss_mask": 1.0, "loss_dice": 1.0}

        # weight for no-object in CE
        empty_weight = torch.ones(num_classes + 1)
        empty_weight[num_classes] = eos_coef
        self.register_buffer("empty_weight", empty_weight)

    def loss_labels(self, class_logits: torch.Tensor, targets: List[Dict[str, torch.Tensor]], indices):
        B, K, Cp1 = class_logits.shape
        target_classes = torch.full((B, K), self.num_classes, dtype=torch.long, device=class_logits.device)

        for b, (src_idx, tgt_idx) in enumerate(indices):
            if src_idx.numel() == 0:
                continue
            target_classes[b, src_idx] = targets[b]["labels"][tgt_idx].to(device=class_logits.device)

        loss_ce = F.cross_entropy(class_logits.transpose(1, 2), target_classes, weight=self.empty_weight)
        return {"loss_ce": loss_ce}

    def loss_masks(self, mask_logits: torch.Tensor, targets: List[Dict[str, torch.Tensor]], indices):
        # mask_logits: [B,K,*S]
        B, K = mask_logits.shape[:2]
        mask_flat, _ = flatten_spatial(mask_logits)  # [B,K,P]
        losses = {"loss_mask": torch.tensor(0.0, device=mask_logits.device), "loss_dice": torch.tensor(0.0, device=mask_logits.device)}

        total_pairs = 0
        for b, (src_idx, tgt_idx) in enumerate(indices):
            if src_idx.numel() == 0:
                continue
            pred = mask_flat[b, src_idx]                 # [n_match, P]
            tgt = targets[b]["masks"][tgt_idx].flatten(start_dim=1).to(pred.device)  # [n_match, P]

            # point sampling (per matched mask)
            point_idx = sample_points_uncertainty(pred, num_points=self.num_points)  # [n_match, num_points]

            pred_pts = pred.gather(1, point_idx)
            tgt_pts = tgt.gather(1, point_idx)

            losses["loss_mask"] = losses["loss_mask"] + sigmoid_focal_loss(
                pred_pts, tgt_pts, alpha=self.focal_alpha, gamma=self.focal_gamma, reduction="mean"
            )
            losses["loss_dice"] = losses["loss_dice"] + dice_loss(pred_pts, tgt_pts)

            total_pairs += 1

        if total_pairs > 0:
            losses["loss_mask"] = losses["loss_mask"] / total_pairs
            losses["loss_dice"] = losses["loss_dice"] / total_pairs

        return losses

    def forward(self, outputs: Dict[str, torch.Tensor], targets: List[Dict[str, torch.Tensor]]):
        """
        outputs must include:
          - class_logits: [B,K,C+1]
          - mask_logits:  [B,K,*S]
        """
        class_logits = outputs["class_logits"]
        mask_logits = outputs["mask_logits"]

        indices = self.matcher(class_logits, mask_logits, targets)

        losses = {}
        losses.update(self.loss_labels(class_logits, targets, indices))
        losses.update(self.loss_masks(mask_logits, targets, indices))

        # apply weights
        weighted = {}
        total = torch.tensor(0.0, device=class_logits.device)
        for k, v in losses.items():
            w = self.loss_weights.get(k, 1.0)
            weighted[k] = v * w
            total = total + weighted[k]
        weighted["main_loss"] = total
        return weighted, indices