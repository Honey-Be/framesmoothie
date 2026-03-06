import torch
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Sequence, Tuple


def panoptic_fusion(
    *,
    sem_logits: torch.Tensor,          # [B,C_sem,*S] channel-first
    inst_class_logits: torch.Tensor,   # [B,K,C_thing+1] (no-object last)
    inst_mask_logits: torch.Tensor,    # [B,K,*S]
    inst_scores: Optional[torch.Tensor] = None,  # [B,K] optional
    thing_classes: Sequence[int] = (),
    stuff_classes: Optional[Sequence[int]] = None,
    score_thresh: float = 0.5,
    mask_thresh: float = 0.5,
    overlap_thresh: float = 0.5,
    min_inst_area: int = 64,
    min_stuff_area: int = 64,
    void_id: int = 0,
) -> List[Dict[str, Any]]:
    """
    Greedy panoptic fusion baseline.

    Returns list length B:
      {
        "panoptic_id": Tensor[*S] int32,
        "segments_info": List[dict]
      }
    """
    B, C_sem = sem_logits.shape[:2]
    spatial = tuple(sem_logits.shape[2:])
    device = sem_logits.device

    sem_prob = F.softmax(sem_logits, dim=1)
    sem_pred = sem_prob.argmax(dim=1)  # [B,*S]

    if stuff_classes is None:
        # default: all semantic classes not in thing_classes
        thing_set = set(int(x) for x in thing_classes)
        stuff_classes = [c for c in range(C_sem) if c not in thing_set]

    thing_tensor = torch.tensor(list(thing_classes), device=device, dtype=torch.long) if len(thing_classes) > 0 else None

    out_all: List[Dict[str, Any]] = []

    for b in range(B):
        pan = torch.full(spatial, void_id, dtype=torch.int32, device=device)
        segments: List[Dict[str, Any]] = []
        next_id = 1

        # --- instances ---
        cls_prob = F.softmax(inst_class_logits[b], dim=-1)  # [K,C_thing+1]
        cls_prob_thing = cls_prob[:, :-1]  # exclude no-object
        cls_score, cls_id = cls_prob_thing.max(dim=-1)      # [K], [K]

        if inst_scores is not None:
            # if inst_scores is a logit-like score, you may want sigmoid; if already prob, set directly upstream
            cls_score = cls_score * torch.sigmoid(inst_scores[b])

        keep = cls_score >= score_thresh

        if thing_tensor is not None and thing_tensor.numel() > 0:
            # keep only predicted labels in thing_classes
            # torch.isin exists in newer pytorch; implement manual:
            in_thing = (cls_id.unsqueeze(-1) == thing_tensor.unsqueeze(0)).any(dim=-1)
            keep = keep & in_thing

        idx = torch.nonzero(keep, as_tuple=False).squeeze(1)
        if idx.numel() > 0:
            scores_sorted, order = torch.sort(cls_score[idx], descending=True)
            slots = idx[order]

            taken = pan.ne(void_id)

            for si in slots.tolist():
                prob = torch.sigmoid(inst_mask_logits[b, si])
                mask = prob > mask_thresh
                area = int(mask.sum().item())
                if area < min_inst_area:
                    continue

                inter = mask & taken
                inter_area = int(inter.sum().item())
                if inter_area / max(area, 1) > overlap_thresh:
                    continue

                mask2 = mask & (~taken)
                area2 = int(mask2.sum().item())
                if area2 < min_inst_area:
                    continue

                pan[mask2] = next_id
                segments.append({
                    "id": int(next_id),
                    "category_id": int(cls_id[si].item()),
                    "isthing": True,
                    "score": float(cls_score[si].item()),
                    "area": int(area2),
                })
                next_id += 1
                taken = pan.ne(void_id)

        # --- stuff fill ---
        unassigned = pan.eq(void_id)
        for c in stuff_classes:
            m = (sem_pred[b] == int(c)) & unassigned
            area = int(m.sum().item())
            if area < min_stuff_area:
                continue
            pan[m] = next_id
            segments.append({
                "id": int(next_id),
                "category_id": int(c),
                "isthing": False,
                "score": float(sem_prob[b, int(c)][m].mean().item()) if area > 0 else 0.0,
                "area": int(area),
            })
            next_id += 1
            unassigned = pan.eq(void_id)

        out_all.append({"panoptic_id": pan, "segments_info": segments})

    return out_all