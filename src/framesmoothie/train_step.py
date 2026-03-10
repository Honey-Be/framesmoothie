import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Tuple

from framesmoothie.matcher import SetCriterion
from framesmoothie.hmc import HMCCalibrator
from framesmoothie.panoptic import panoptic_fusion
from framesmoothie.model import ema_update


def _resize_targets_masks(
    targets: List[Dict[str, torch.Tensor]],
    size: Tuple[int, ...],
    spatial_dims: int,
) -> List[Dict[str, torch.Tensor]]:
    """
    Resize GT masks to match prediction resolution.
    targets[b]["masks"]: [M,*S] -> [M,*size]
    """
    out = []
    for t in targets:
        labels = t["labels"]
        masks = t["masks"].float()
        if tuple(masks.shape[1:]) != size:
            m = masks.unsqueeze(1)  # [M,1,*S]
            if spatial_dims == 2:
                m = F.interpolate(m, size=size, mode="nearest")
            elif spatial_dims == 3:
                m = F.interpolate(m, size=size, mode="nearest")
            else:
                raise ValueError("Only 2D/3D supported")
            masks = m.squeeze(1)
        out.append({"labels": labels, "masks": masks})
    return out


def semantic_ce_loss(
    sem_logits: torch.Tensor,  # [B,C,*S]
    sem_labels: torch.Tensor,  # [B,*S]
    *,
    ignore_index: int = -100,
    mask: Optional[torch.Tensor] = None,  # [B,*S] bool
) -> torch.Tensor:
    if mask is None:
        return F.cross_entropy(sem_logits, sem_labels, ignore_index=ignore_index)
    # masked CE
    # flatten
    B, C = sem_logits.shape[:2]
    logits = sem_logits.flatten(2).transpose(1, 2).reshape(-1, C)   # [B*P,C]
    labels = sem_labels.flatten().reshape(-1)                       # [B*P]
    m = mask.flatten().reshape(-1)
    logits = logits[m]
    labels = labels[m]
    if logits.numel() == 0:
        return torch.tensor(0.0, device=sem_logits.device)
    return F.cross_entropy(logits, labels, ignore_index=ignore_index)


class FrameSmoothieTrainStep:
    """
    One-step training orchestration for UDA panoptic:
      - supervised on source
      - pseudo-label on target using EMA teacher + panoptic + HMC
      - aux loss via model.reg_loss.collect(model)

    Assumptions:
      - semantic class IDs align with instance class IDs for thing classes
      - instance SetCriterion consumes mask_logits at its own resolution
    """
    def __init__(
        self,
        *,
        criterion_inst: SetCriterion,
        hmc: HMCCalibrator,
        thing_classes: List[int],
        lambda_tgt: float = 1.0,
        lambda_aux: float = 1.0,
        sem_ignore_index: int = -100,
        pseudo_sem_conf_thresh: float = 0.7,
        panoptic_cfg: Optional[Dict[str, Any]] = None,
    ):
        self.criterion_inst = criterion_inst
        self.hmc = hmc
        self.thing_classes = thing_classes
        self.lambda_tgt = float(lambda_tgt)
        self.lambda_aux = float(lambda_aux)
        self.sem_ignore_index = int(sem_ignore_index)
        self.pseudo_sem_conf_thresh = float(pseudo_sem_conf_thresh)
        self.panoptic_cfg = panoptic_cfg or {}

    def __call__(
        self,
        *,
        student: nn.Module,
        teacher: nn.Module,
        src: Dict[str, Any],
        tgt: Dict[str, Any],
        ema_beta: float = 0.999,
        aux_weight: Optional[float] = None,
    ) -> Dict[str, Any]:
        device = next(student.parameters()).device
        aux_weight = float(aux_weight) if aux_weight is not None else getattr(student, "aux_weight_default", 1.0)

        # -------- source supervised --------
        out_s = student(src["image"].to(device))
        sem_logits_s = out_s["sem_logits"]                     # [B,C,*S]
        inst_s = out_s["inst"]
        spatial_inst = tuple(inst_s["mask_logits"].shape[2:])
        spatial_dims = len(spatial_inst)

        sem_loss_s = semantic_ce_loss(
            sem_logits_s,
            src["sem_labels"].to(device),
            ignore_index=self.sem_ignore_index,
        )

        targets_inst_s = _resize_targets_masks(src["inst_targets"], spatial_inst, spatial_dims)
        inst_losses_s, _ = self.criterion_inst(
            {"class_logits": inst_s["class_logits"], "mask_logits": inst_s["mask_logits"]},
            targets_inst_s,
        )
        inst_loss_s = inst_losses_s["main_loss"]

        # -------- target pseudo via teacher --------
        with torch.no_grad():
            out_t_teacher = teacher(tgt["image"].to(device), return_panoptic=True, panoptic_cfg=self.panoptic_cfg)
            sem_logits_t = out_t_teacher["sem_logits"]
            inst_t = out_t_teacher["inst"]

            # HMC calibration (uses upsampled masks for pixel-level semantics)
            pseudo = self.hmc(
                sem_logits=sem_logits_t,
                inst_class_logits=inst_t["class_logits"],
                inst_mask_logits=inst_t["mask_logits_up"],
                inst_scores=inst_t.get("scores", None),
            )

        # student forward on target for pseudo losses
        out_t_student = student(tgt["image"].to(device))
        sem_logits_ts = out_t_student["sem_logits"]
        inst_ts = out_t_student["inst"]
        spatial_inst_ts = tuple(inst_ts["mask_logits"].shape[2:])
        spatial_dims_ts = len(spatial_inst_ts)

        # semantic pseudo loss (confidence masked)
        sem_pseudo = pseudo["pseudo_semantic"]["labels"].to(device)  # [B,*S]
        sem_conf = pseudo["pseudo_semantic"]["conf"].to(device)
        sem_mask = sem_conf >= self.pseudo_sem_conf_thresh
        sem_loss_t = semantic_ce_loss(
            sem_logits_ts,
            sem_pseudo,
            ignore_index=self.sem_ignore_index,
            mask=sem_mask,
        )

        # instance pseudo loss
        pseudo_inst = pseudo["pseudo_instances"]
        targets_inst_t: List[Dict[str, torch.Tensor]] = []
        B = sem_logits_ts.shape[0]
        for b in range(B):
            labels_b = pseudo_inst["labels"][b].to(device)
            masks_b = pseudo_inst["masks"][b].to(device).float()
            # resize to student instance resolution
            if masks_b.numel() > 0 and tuple(masks_b.shape[1:]) != spatial_inst_ts:
                m = masks_b.unsqueeze(1)
                if spatial_dims_ts == 2:
                    m = F.interpolate(m, size=spatial_inst_ts, mode="nearest")
                elif spatial_dims_ts == 3:
                    m = F.interpolate(m, size=spatial_inst_ts, mode="nearest")
                else:
                    raise ValueError("Only 2D/3D supported")
                masks_b = m.squeeze(1)
            targets_inst_t.append({"labels": labels_b, "masks": masks_b})

        inst_losses_t, _ = self.criterion_inst(
            {"class_logits": inst_ts["class_logits"], "mask_logits": inst_ts["mask_logits"]},
            targets_inst_t,
        )
        inst_loss_t = inst_losses_t["main_loss"]

        # -------- aux loss --------
        aux = student.reg_loss.collect(student) if hasattr(student, "reg_loss") else torch.tensor(0.0, device=device)

        # total
        src_total = sem_loss_s + inst_loss_s
        tgt_total = sem_loss_t + inst_loss_t
        total = src_total + self.lambda_tgt * tgt_total + self.lambda_aux * aux_weight * aux

        # EMA update (do this after optimizer step in your outer loop; providing here as helper)
        # ema_update(teacher, student, beta=ema_beta)

        return {
            "loss": total,
            "loss_src_sem": sem_loss_s.detach(),
            "loss_src_inst": inst_loss_s.detach(),
            "loss_tgt_sem": sem_loss_t.detach(),
            "loss_tgt_inst": inst_loss_t.detach(),
            "loss_aux": aux.detach(),
            "out_teacher_panoptic": out_t_teacher.get("panoptic", None),
            "pseudo": pseudo,
        }
