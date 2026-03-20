import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Any, Dict, List, Optional, Tuple, Protocol

from torchutils.decorators import auxloss
from framesmoothie.matcher import SetCriterion
from framesmoothie.hmc import HMCCalibrator
from framesmoothie.diag_meter import DiagMeter
from framesmoothie.zone_losses import compute_zone_predictive_loss
from framesmoothie.model import FrameSmoothiePanopticModel

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
        if tuple(masks.shape[1:]) != size and masks.numel() > 0:
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

    B, C = sem_logits.shape[:2]
    logits = sem_logits.flatten(2).transpose(1, 2).reshape(-1, C)  # [B*P,C]
    labels = sem_labels.flatten().reshape(-1)
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
      - pseudo-label on target using EMA teacher + HMC
      - aux loss via model.reg_loss.collect(model)

    Diagnostics:
      - corr_sem_inst is logged ONLY for target (teacher/student),
        while source corr is maintained as EMA reference baseline.
      - Optional DiagMeter can be provided to accumulate scalar diagnostics
        without returning full raw dict every step.
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
        corr_ema_beta: float = 0.99,
        lambda_pred: float = 0.1,
        pred_detach_target: bool = True,
        use_edge_weights: bool = True,
    ):
        self.criterion_inst = criterion_inst
        self.hmc = hmc
        self.thing_classes = thing_classes
        self.lambda_tgt = float(lambda_tgt)
        self.lambda_aux = float(lambda_aux)
        self.sem_ignore_index = int(sem_ignore_index)
        self.pseudo_sem_conf_thresh = float(pseudo_sem_conf_thresh)
        self.panoptic_cfg = panoptic_cfg or {}
        self.corr_ema_beta = float(corr_ema_beta)
        self.lambda_pred = float(lambda_pred)
        self.pred_detach_target = bool(pred_detach_target)
        self.use_edge_weights = bool(use_edge_weights)
        self._src_corr_ema: Optional[torch.Tensor] = None  # CPU scalar

    @staticmethod
    def _extract_corr(diag: Dict[str, Any]) -> Optional[torch.Tensor]:
        if isinstance(diag, dict) and "corr_sem_inst" in diag and isinstance(diag["corr_sem_inst"], torch.Tensor):
            return diag["corr_sem_inst"].detach().float().cpu().reshape(())
        return None

    def __call__(
        self,
        *,
        student: FrameSmoothiePanopticModel,
        teacher: nn.Module,
        src: Dict[str, Any],
        tgt: Dict[str, Any],
        ema_beta: float = 0.999,
        aux_weight: Optional[float] = None,
        return_diag: bool = False,
        diag_meter: Optional[DiagMeter] = None,
    ) -> Dict[str, Any]:
        device = next(student.parameters()).device
        aux_weight = float(aux_weight) if aux_weight is not None else getattr(student, "aux_weight_default", 1.0)

        # We only need raw diag dict if caller wants it or meter needs it.
        need_diag = bool(return_diag or (diag_meter is not None))

        # -------- source supervised --------
        B_src = src["image"].shape[0]
        out_s = student(
            src["image"].to(device),
            domain_id=torch.zeros((B_src, 1), device=device),
            return_diag=need_diag,
        )
        sem_logits_s = out_s["sem_logits"]
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
            B_tgt = tgt["image"].shape[0]
            out_t_teacher = teacher(
                tgt["image"].to(device),
                domain_id=torch.ones((B_tgt, 1), device=device),
                return_panoptic=True,
                return_diag=need_diag,
                panoptic_cfg=self.panoptic_cfg,
            )
            sem_logits_t = out_t_teacher["sem_logits"]
            inst_t = out_t_teacher["inst"]

            pseudo = self.hmc(
                sem_logits=sem_logits_t,
                inst_class_logits=inst_t["class_logits"],
                inst_mask_logits=inst_t["mask_logits_up"],
                inst_scores=inst_t.get("scores", None),
                boundary_features=out_t_teacher.get("boundary_features", None),
            )

        # student forward on target for pseudo losses
        out_t_student = student(
            tgt["image"].to(device),
            domain_id=torch.ones((B_tgt, 1), device=device),
            return_diag=need_diag,
        )
        sem_logits_ts = out_t_student["sem_logits"]
        inst_ts = out_t_student["inst"]
        spatial_inst_ts = tuple(inst_ts["mask_logits"].shape[2:])
        spatial_dims_ts = len(spatial_inst_ts)

        # semantic pseudo loss (confidence masked)
        sem_pseudo = pseudo["pseudo_semantic"]["labels"].to(device)
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
        for b in range(B_tgt):
            labels_b = pseudo_inst["labels"][b].to(device)
            masks_b = pseudo_inst["masks"][b].to(device).float()
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

        # -------- zone-wise predictive loss (L_pred) --------
        pred_loss_s, pred_loss_s_dict = compute_zone_predictive_loss(
            zone_pred=out_s.get("zone_pred", None),
            zones=(out_s["zones"]["zones"] if isinstance(out_s.get("zones", None), dict) else None),
            detach_target=self.pred_detach_target,
            use_edge_weights=self.use_edge_weights,
        )
        pred_loss_t, pred_loss_t_dict = compute_zone_predictive_loss(
            zone_pred=out_t_student.get("zone_pred", None),
            zones=(out_t_student["zones"]["zones"] if isinstance(out_t_student.get("zones", None), dict) else None),
            detach_target=self.pred_detach_target,
            use_edge_weights=self.use_edge_weights,
        )

        # -------- aux loss --------
        aux = FrameSmoothiePanopticModel.reg_loss.collect(student, aggregate=torch.sum)

        # total
        src_total = sem_loss_s + inst_loss_s + self.lambda_pred * pred_loss_s
        tgt_total = sem_loss_t + inst_loss_t + self.lambda_pred * pred_loss_t
        total = src_total + self.lambda_tgt * tgt_total + self.lambda_aux * aux_weight * aux

        # -------- diagnostics (C) --------
        diag_src = out_s.get("diag", {}) if need_diag else {}
        src_corr = self._extract_corr(diag_src)
        if src_corr is not None:
            self._src_corr_ema = src_corr if self._src_corr_ema is None else (self._src_corr_ema * self.corr_ema_beta + src_corr * (1.0 - self.corr_ema_beta))

        diag_tgt_teacher = out_t_teacher.get("diag", {}) if need_diag else {}
        diag_tgt_student = out_t_student.get("diag", {}) if need_diag else {}

        corr_ref = self._src_corr_ema
        corr_tgt_teacher = self._extract_corr(diag_tgt_teacher)
        corr_tgt_student = self._extract_corr(diag_tgt_student)

        # Meter update: target-only full diag + ref corr
        if diag_meter is not None:
            if corr_ref is not None:
                diag_meter.update({"corr_ref_src_ema": corr_ref})
            diag_meter.update(diag_tgt_teacher, prefix="tgt_teacher/")
            diag_meter.update(diag_tgt_student, prefix="tgt_student/")
            if corr_tgt_teacher is not None and corr_ref is not None:
                diag_meter.update({"corr_tgt_teacher_delta": (corr_tgt_teacher - corr_ref)})
            if corr_tgt_student is not None and corr_ref is not None:
                diag_meter.update({"corr_tgt_student_delta": (corr_tgt_student - corr_ref)})

        # Return payload: by default, do NOT return full diag dicts unless return_diag=True
        out: Dict[str, Any] = {
            "loss": total,
            "loss_src_sem": sem_loss_s.detach(),
            "loss_src_inst": inst_loss_s.detach(),
            "loss_tgt_sem": sem_loss_t.detach(),
            "loss_tgt_inst": inst_loss_t.detach(),
            "loss_aux": aux.detach(),
            "loss_pred_src": pred_loss_s.detach() if isinstance(pred_loss_s, torch.Tensor) else pred_loss_s,
            "loss_pred_tgt": pred_loss_t.detach() if isinstance(pred_loss_t, torch.Tensor) else pred_loss_t,
            "corr_ref_src_ema": corr_ref,
            "corr_tgt_teacher": corr_tgt_teacher,
            "corr_tgt_student": corr_tgt_student,
            "corr_tgt_teacher_delta": (corr_tgt_teacher - corr_ref) if (corr_tgt_teacher is not None and corr_ref is not None) else None,
            "corr_tgt_student_delta": (corr_tgt_student - corr_ref) if (corr_tgt_student is not None and corr_ref is not None) else None,
            "out_teacher_panoptic": out_t_teacher.get("panoptic", None),
            "pseudo": pseudo,
            "pred_loss_src_dict": pred_loss_s_dict,
            "pred_loss_tgt_dict": pred_loss_t_dict,
        }
        if return_diag:
            out["diag_tgt_teacher"] = diag_tgt_teacher
            out["diag_tgt_student"] = diag_tgt_student
        return out
