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


def _binary_erode(mask: torch.Tensor) -> torch.Tensor:
    """Binary erosion for 2D/3D masks.

    mask: [H,W] or [T,H,W] bool
    returns bool mask of same shape
    """
    if mask.ndim == 2:
        x = mask.float().unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
        # erode = 1 - dilate(1-mask)
        y = 1.0 - F.max_pool2d(1.0 - x, kernel_size=3, stride=1, padding=1)
        return (y.squeeze(0).squeeze(0) > 0.5)
    if mask.ndim == 3:
        x = mask.float().unsqueeze(0).unsqueeze(0)  # [1,1,T,H,W]
        y = 1.0 - F.max_pool3d(1.0 - x, kernel_size=3, stride=1, padding=1)
        return (y.squeeze(0).squeeze(0) > 0.5)
    raise ValueError("Only 2D/3D supported")


def _mask_edge(mask: torch.Tensor) -> torch.Tensor:
    er = _binary_erode(mask)
    return mask & (~er)


def _boundary_strength_map(boundary_features: torch.Tensor) -> torch.Tensor:
    """Convert boundary zone features into a scalar map in [0,1].

    Expected input:
      - channel-last: [B,*S,C]
    Output:
      - [B,*S]
    """
    if boundary_features.ndim < 4:
        raise ValueError("boundary_features must be channel-last [B,*S,C]")
    # scalar strength via RMS over channels
    x = boundary_features.to(dtype=torch.float32)
    m = torch.sqrt(torch.mean(torch.square(x), dim=-1) + 1e-6)  # [B,*S]
    # normalize per-sample
    B = m.shape[0]
    flat = m.reshape(B, -1)
    mu = flat.mean(dim=1).view(B, *([1] * (m.ndim - 1)))
    sd = flat.std(dim=1, unbiased=False).clamp_min(1e-6).view(B, *([1] * (m.ndim - 1)))
    z = (m - mu) / sd
    return torch.sigmoid(z)


class HMCCalibrator(nn.Module):
    """
    HMC-like calibrator (Region / Superpixel / Pixel), UniDAformer-inspired.

    Inputs (teacher outputs):
      - sem_logits: [B, C_sem, *S] (channel-first)
      - inst_class_logits: [B, K, C_thing+1] (no-object last)
      - inst_mask_logits: [B, K, *S]
      - inst_scores(optional): [B,K]
      - boundary_features(optional): [B,*S,Cb] channel-last boundary zone features

    Outputs (pseudo targets for student):
      - pseudo_instances: {"labels":[B,M], "masks":[B,M,*S], "scores":[B,M]}
      - pseudo_semantic:  {"labels":[B,*S], "conf":[B,*S]}

    Notes:
      - superpixel step uses grid blocks (no external deps)
      - pixel step enforces semantic consistency
      - boundary_features, when provided, act as an edge prior
        * region-wise: edge alignment boosts instance score
        * superpixel-wise: suppression is weakened near boundaries
        * pixel-wise: sem-consistent boundary pixels are emphasized
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
        lambda_region_boundary: float = 0.25,
        # superpixel calibration (grid)
        sp_block: int = 16,
        sp_thresh: float = 0.5,
        lambda_superpixel_boundary: float = 0.5,
        # pixel calibration
        lambda_pixel_boundary: float = 0.5,
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
        self.lambda_region_boundary = float(lambda_region_boundary)
        self.sp_block = int(sp_block)
        self.sp_thresh = float(sp_thresh)
        self.lambda_superpixel_boundary = float(lambda_superpixel_boundary)
        self.lambda_pixel_boundary = float(lambda_pixel_boundary)
        self.overlap_thresh = float(overlap_thresh)

    @torch.no_grad()
    def forward(
        self,
        *,
        sem_logits: torch.Tensor,
        inst_class_logits: torch.Tensor,
        inst_mask_logits: torch.Tensor,
        inst_scores: Optional[torch.Tensor] = None,
        boundary_features: Optional[torch.Tensor] = None,
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

        # optional boundary map
        bmap = None
        if boundary_features is not None:
            bmap = _boundary_strength_map(boundary_features)  # [B,*S]

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
            _, order = torch.sort(s, descending=True)
            slots = idx[order]

            # masks prob
            mprob = torch.sigmoid(inst_mask_logits[b, slots])  # [M,*S]
            labels = cls_id[b, slots]                          # [M]
            scores = cls_score[b, slots]                       # [M]

            # ---- Region-wise calibration: semantic consistency + boundary edge alignment ----
            if self.region_sem_power != 0.0 or (bmap is not None and self.lambda_region_boundary != 0.0):
                new_scores = []
                for i in range(mprob.shape[0]):
                    mask = mprob[i] > self.mask_thresh
                    area = int(mask.sum().item())
                    if area < self.min_area:
                        new_scores.append(torch.tensor(0.0, device=device))
                        continue

                    c = int(labels[i].item())
                    sem_in = sem_prob[b, c][mask].mean() if area > 0 else torch.tensor(0.0, device=device)
                    sc = scores[i] * (sem_in.clamp_min(0.0) ** self.region_sem_power)

                    if bmap is not None and self.lambda_region_boundary != 0.0:
                        edge = _mask_edge(mask)
                        if int(edge.sum().item()) > 0:
                            balign = bmap[b][edge].mean()
                            sc = sc * (1.0 + self.lambda_region_boundary * balign)
                    new_scores.append(sc)
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

            # ---- Superpixel-wise calibration: grid blocks snap, but preserve strong boundaries ----
            if spatial_dims == 2:
                mp = mprob.unsqueeze(1)  # [M,1,H,W]
            elif spatial_dims == 3:
                mp = mprob.unsqueeze(1)  # [M,1,T,H,W]
            else:
                raise ValueError("Only 2D/3D supported")

            pooled = _avg_pool_nd(mp, self.sp_block, spatial_dims)  # [M,1,*S']
            snapped = _upsample_nd(pooled, spatial, spatial_dims).squeeze(1)  # [M,*S]

            if bmap is None or self.lambda_superpixel_boundary == 0.0:
                mprob = torch.where(snapped > self.sp_thresh, mprob, mprob * 0.5)
            else:
                # near boundaries => weaker suppression (preserve sharp mask detail)
                preserve = 0.5 + self.lambda_superpixel_boundary * bmap[b].clamp(0.0, 1.0)
                preserve = preserve.clamp(0.5, 1.0)
                mprob = torch.where(snapped > self.sp_thresh, mprob, mprob * preserve)

            # ---- Pixel-wise calibration: semantic consistency + boundary emphasis ----
            mcal = []
            for i in range(mprob.shape[0]):
                c = int(labels[i].item())
                x = mprob[i] * sem_prob[b, c]
                if bmap is not None and self.lambda_pixel_boundary != 0.0:
                    x = x * (1.0 + self.lambda_pixel_boundary * bmap[b]).clamp(1.0, 2.0)
                mcal.append(x)
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

            _, order = torch.sort(scores, descending=True)
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

        return {
            "pseudo_semantic": {"labels": sem_pseudo, "conf": sem_pseudo_conf},
            "pseudo_instances": {"labels": all_labels, "masks": all_masks, "scores": all_scores},
        }
