import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, Optional, Tuple


def _infer_spatial_dims(x: torch.Tensor) -> int:
    # x: [B, ...]
    return x.ndim - 2  # for [B,C,*S]


def _avg_pool_nd(x: torch.Tensor, k: int, spatial_dims: int) -> torch.Tensor:
    if spatial_dims == 2:
        return F.avg_pool2d(x, kernel_size=k, stride=k)
    if spatial_dims == 3:
        return F.avg_pool3d(x, kernel_size=k, stride=k)
    raise ValueError("Only 2D/3D supported")


def _upsample_nd(x: torch.Tensor, size: Tuple[int, ...], spatial_dims: int) -> torch.Tensor:
    if spatial_dims == 2:
        return F.interpolate(x, size=size, mode="nearest")
    if spatial_dims == 3:
        return F.interpolate(x, size=size, mode="nearest")
    raise ValueError("Only 2D/3D supported")


class HMCCalibrator(nn.Module):
    """
    HMC-like calibrator (Region / Superpixel / Pixel), UniDAformer-inspired.

    Inputs (teacher outputs):
      - sem_logits: [B, C_sem, *S] (channel-first)
      - inst_class_logits: [B, K, C_thing+1] (no-object last)
      - inst_mask_logits: [B, K, *S]
      - inst_scores(optional): [B,K]

    Outputs (pseudo targets for student):
      - pseudo_instances: {"labels":[B,M], "masks":[B,M,*S], "scores":[B,M]}
      - pseudo_semantic:  {"labels":[B,*S], "conf":[B,*S]}

    Notes:
      - superpixel step uses grid blocks (no external deps)
      - pixel step enforces semantic consistency
    """
    def __init__(
        self,
        *,
        thing_classes: Tuple[int, ...],
        score_thresh: float = 0.6,
        mask_thresh: float = 0.5,
        min_area: int = 64,
        sem_conf_thresh: float = 0.7,
        # region calibration
        region_sem_power: float = 1.0,
        # superpixel calibration (grid)
        sp_block: int = 16,
        sp_thresh: float = 0.5,
        # overlap handling
        overlap_thresh: float = 0.5,
    ):
        super().__init__()
        self.thing_classes = tuple(int(x) for x in thing_classes)
        self.score_thresh = float(score_thresh)
        self.mask_thresh = float(mask_thresh)
        self.min_area = int(min_area)
        self.sem_conf_thresh = float(sem_conf_thresh)
        self.region_sem_power = float(region_sem_power)
        self.sp_block = int(sp_block)
        self.sp_thresh = float(sp_thresh)
        self.overlap_thresh = float(overlap_thresh)

    @torch.no_grad()
    def forward(
        self,
        *,
        sem_logits: torch.Tensor,
        inst_class_logits: torch.Tensor,
        inst_mask_logits: torch.Tensor,
        inst_scores: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        device = sem_logits.device
        B, C_sem = sem_logits.shape[:2]
        spatial = tuple(sem_logits.shape[2:])
        spatial_dims = len(spatial)

        sem_prob = F.softmax(sem_logits, dim=1)         # [B,C_sem,*S]
        sem_conf, sem_pred = sem_prob.max(dim=1)        # [B,*S]

        # semantic pseudo (confidence masking)
        sem_pseudo = sem_pred.clone()
        sem_pseudo_conf = sem_conf.clone()

        # instance base selection
        cls_prob = F.softmax(inst_class_logits, dim=-1)  # [B,K,C_thing+1]
        cls_prob_thing = cls_prob[:, :, :-1]
        cls_score, cls_id = cls_prob_thing.max(dim=-1)   # [B,K], [B,K]

        if inst_scores is not None:
            cls_score = cls_score * torch.sigmoid(inst_scores)

        thing_tensor = torch.tensor(self.thing_classes, device=device, dtype=torch.long)
        in_thing = (cls_id.unsqueeze(-1) == thing_tensor.unsqueeze(0).unsqueeze(0)).any(dim=-1)
        keep = in_thing & (cls_score >= self.score_thresh)

        # prepare outputs
        all_labels = []
        all_masks = []
        all_scores = []

        for b in range(B):
            idx = torch.nonzero(keep[b], as_tuple=False).squeeze(1)
            if idx.numel() == 0:
                all_labels.append(torch.empty((0,), dtype=torch.long, device=device))
                all_masks.append(torch.empty((0, *spatial), dtype=torch.bool, device=device))
                all_scores.append(torch.empty((0,), dtype=torch.float32, device=device))
                continue

            # sort by score
            s = cls_score[b, idx]
            s_sorted, order = torch.sort(s, descending=True)
            slots = idx[order]

            # masks prob
            mprob = torch.sigmoid(inst_mask_logits[b, slots])  # [M,*S]
            labels = cls_id[b, slots]                          # [M]
            scores = cls_score[b, slots]                       # [M]

            # ---- Region-wise calibration: semantic consistency re-score ----
            if self.region_sem_power != 0.0:
                # per-instance: mean sem_prob[label] over mask region
                # loop is ok (M small)
                new_scores = []
                for i in range(mprob.shape[0]):
                    mask = mprob[i] > self.mask_thresh
                    area = int(mask.sum().item())
                    if area < self.min_area:
                        new_scores.append(torch.tensor(0.0, device=device))
                        continue
                    c = int(labels[i].item())
                    sem_in = sem_prob[b, c][mask].mean() if area > 0 else torch.tensor(0.0, device=device)
                    new_scores.append(scores[i] * (sem_in.clamp_min(0.0) ** self.region_sem_power))
                scores = torch.stack(new_scores)

            # filter again
            ok = scores >= self.score_thresh
            mprob = mprob[ok]
            labels = labels[ok]
            scores = scores[ok]

            if mprob.numel() == 0:
                all_labels.append(torch.empty((0,), dtype=torch.long, device=device))
                all_masks.append(torch.empty((0, *spatial), dtype=torch.bool, device=device))
                all_scores.append(torch.empty((0,), dtype=torch.float32, device=device))
                continue

            # ---- Superpixel-wise calibration: grid blocks snap ----
            # treat each instance prob as [1,1,*S] for pooling
            # 2D: [N,1,H,W], 3D: [N,1,T,H,W]
            if spatial_dims == 2:
                mp = mprob.unsqueeze(1)  # [M,1,H,W]
            elif spatial_dims == 3:
                mp = mprob.unsqueeze(1)  # [M,1,T,H,W]
            else:
                raise ValueError("Only 2D/3D supported")

            pooled = _avg_pool_nd(mp, self.sp_block, spatial_dims)  # [M,1,*S']
            snapped = _upsample_nd(pooled, spatial, spatial_dims)   # [M,1,*S]
            snapped = snapped.squeeze(1)                            # [M,*S]
            mprob = torch.where(snapped > self.sp_thresh, mprob, mprob * 0.5)

            # ---- Pixel-wise calibration: enforce semantic consistency ----
            # multiply by semantic prob of predicted label
            mcal = []
            for i in range(mprob.shape[0]):
                c = int(labels[i].item())
                mcal.append(mprob[i] * sem_prob[b, c])
            mprob = torch.stack(mcal)

            # binarize
            masks = mprob > self.mask_thresh

            # remove small + resolve overlaps greedily by score
            areas = masks.flatten(1).sum(dim=1)
            valid = areas >= self.min_area
            masks = masks[valid]
            labels = labels[valid]
            scores = scores[valid]

            if masks.numel() == 0:
                all_labels.append(torch.empty((0,), dtype=torch.long, device=device))
                all_masks.append(torch.empty((0, *spatial), dtype=torch.bool, device=device))
                all_scores.append(torch.empty((0,), dtype=torch.float32, device=device))
                continue

            # sort again
            s_sorted, order = torch.sort(scores, descending=True)
            masks = masks[order]
            labels = labels[order]
            scores = scores[order]

            taken = torch.zeros(spatial, dtype=torch.bool, device=device)
            kept_masks = []
            kept_labels = []
            kept_scores = []

            for i in range(masks.shape[0]):
                m = masks[i]
                area = int(m.sum().item())
                if area < self.min_area:
                    continue
                inter = (m & taken).sum().item()
                if inter / max(area, 1) > self.overlap_thresh:
                    continue
                m2 = m & (~taken)
                if int(m2.sum().item()) < self.min_area:
                    continue
                kept_masks.append(m2)
                kept_labels.append(labels[i])
                kept_scores.append(scores[i])
                taken |= m2

            if len(kept_masks) == 0:
                all_labels.append(torch.empty((0,), dtype=torch.long, device=device))
                all_masks.append(torch.empty((0, *spatial), dtype=torch.bool, device=device))
                all_scores.append(torch.empty((0,), dtype=torch.float32, device=device))
                continue

            all_labels.append(torch.stack(kept_labels))
            all_masks.append(torch.stack(kept_masks))
            all_scores.append(torch.stack(kept_scores).to(dtype=torch.float32))

        # pack outputs (ragged per batch)
        return {
            "pseudo_semantic": {"labels": sem_pseudo, "conf": sem_pseudo_conf},
            "pseudo_instances": {"labels": all_labels, "masks": all_masks, "scores": all_scores},
        }