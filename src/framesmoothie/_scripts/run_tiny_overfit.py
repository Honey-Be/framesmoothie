from __future__ import annotations

import argparse
import csv
import dataclasses
from datetime import datetime
import json
import math
from pathlib import Path
import shutil
import sys
from typing import Any, Dict, Optional, Literal

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn

from framesmoothie.checkpointing import (
    export_state_dict_artifact,
    load_training_checkpoint,
    save_training_checkpoint,
)
from framesmoothie.diag_meter import DiagMeter
from framesmoothie.fp import curry
from framesmoothie.hmc import HMCCalibrator
from framesmoothie.matcher import HungarianMatcher, SetCriterion
from framesmoothie.model import FrameSmoothiePanopticModel, S9Stack, ema_update, make_ema_teacher
from framesmoothie.repro import (
    DeterminismConfig,
    RngRegistry,
    RngSnapshot,
    SeedPath,
    choose_multiprocessing_context,
    configure_determinism,
    rng_digest,
    restore_rng_snapshot,
)
from framesmoothie.safetensors_io import load_object
from framesmoothie.train_step import FrameSmoothieTrainStep
from framesmoothie.trainer_runtime import RuntimeEnv, RuntimePolicy, RuntimeState, make_step_program
from framesmoothie._scripts.presets import get_preset, list_presets
from s9.transforms.dost import DOST, IDOST
from s9.base import FPDTypeIdx

def _to_jsonable(obj: Any) -> Any:
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _to_jsonable(v) for k, v in obj.items()}
    if dataclasses.is_dataclass(obj):
        return _to_jsonable(dataclasses.asdict(obj))
    if isinstance(obj, torch.dtype):
        return str(obj)
    if isinstance(obj, torch.device):
        return str(obj)
    if isinstance(obj, nn.Module):
        return {
            "__module_class__": obj.__class__.__name__,
            "__module_path__": f"{obj.__class__.__module__}.{obj.__class__.__qualname__}",
        }
    if callable(obj):
        return getattr(obj, "__name__", repr(obj))
    return repr(obj)


@curry
def _call_train_step(
    stepper: FrameSmoothieTrainStep,
    student: FrameSmoothiePanopticModel,
    teacher: nn.Module,
    diag_meter: Optional[DiagMeter],
    return_diag: bool,
    src: Dict[str, Any],
    tgt: Dict[str, Any],
) -> Dict[str, Any]:
    return stepper(
        student=student,
        teacher=teacher,
        src=src,
        tgt=tgt,
        diag_meter=diag_meter,
        return_diag=return_diag,
    )


class EmaDecayScheduler:
    """Simple stateful EMA decay scheduler for teacher update."""

    def __init__(
        self,
        *,
        kind: str = "constant",
        beta_start: float = 0.99,
        beta_end: float = 0.999,
        total_steps: int = 100,
        last_step: int = 0,
    ):
        self.kind = str(kind).lower()
        self.beta_start = float(beta_start)
        self.beta_end = float(beta_end)
        self.total_steps = max(1, int(total_steps))
        self.last_step = int(last_step)

    def _interp(self, t: int) -> float:
        if self.kind == "constant":
            return self.beta_start
        u = min(max(float(t) / float(self.total_steps), 0.0), 1.0)
        if self.kind == "linear":
            return self.beta_start + (self.beta_end - self.beta_start) * u
        if self.kind == "cosine":
            c = 0.5 * (1.0 - math.cos(math.pi * u))
            return self.beta_start + (self.beta_end - self.beta_start) * c
        raise ValueError(f"Unknown ema schedule kind: {self.kind}")

    def step(self) -> float:
        self.last_step += 1
        return self._interp(self.last_step)

    def get_last_beta(self) -> float:
        return self._interp(self.last_step)

    def state_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "beta_start": self.beta_start,
            "beta_end": self.beta_end,
            "total_steps": self.total_steps,
            "last_step": self.last_step,
        }

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.kind = str(state.get("kind", self.kind))
        self.beta_start = float(state.get("beta_start", self.beta_start))
        self.beta_end = float(state.get("beta_end", self.beta_end))
        self.total_steps = int(state.get("total_steps", self.total_steps))
        self.last_step = int(state.get("last_step", self.last_step))



def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    images = torch.stack([b["image"] for b in batch])
    sem = torch.stack([b["sem_labels"] for b in batch])
    inst_targets = [b["inst_targets"] for b in batch]
    return {"image": images, "sem_labels": sem, "inst_targets": inst_targets}



def _make_run_dir(output_root: Path, preset: str, run_name: Optional[str]) -> Path:
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    name = run_name if run_name is not None else f"{preset}_{ts}"
    run_dir = output_root / name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir



def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")



def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows



def _select_key(row: dict[str, Any], metric_name: str) -> Optional[float]:
    v = row.get(metric_name, None)
    if v is None:
        return None
    try:
        return float(v)
    except Exception:
        return None



def _is_better(candidate: float, best: float, mode: str, min_delta: float = 0.0) -> bool:
    if mode == "max":
        return candidate > (best + min_delta)
    return candidate < (best - min_delta)



def _save_metrics_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = sorted({k for r in rows for k in r.keys()}) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        if not fieldnames:
            f.write("")
            return
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)



def _build_scheduler(opt: torch.optim.Optimizer, *, scheduler_kind: str, steps: int, step_size: int, gamma: float):
    kind = scheduler_kind.lower()
    if kind == "none":
        return None
    if kind == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, steps))
    if kind == "step":
        return torch.optim.lr_scheduler.StepLR(opt, step_size=max(1, step_size), gamma=gamma)
    raise ValueError(f"Unknown scheduler_kind: {scheduler_kind}")



def _load_sample_file(path: Path) -> dict[str, Any]:
    if path.suffix == ".safetensors":
        sample = load_object(path, device="cpu")
    else:
        sample = torch.load(path, map_location="cpu")
    if not isinstance(sample, dict):
        raise TypeError(f"Expected sample payload to be a dict, got {type(sample)!r}")
    return sample



def _load_selected_domain(dir_path: Path, *, limit: int) -> list[dict[str, Any]]:
    files = sorted([
        *dir_path.glob("*.pt"),
        *dir_path.glob("*.pth"),
        *dir_path.glob("*.safetensors"),
    ])
    if not files:
        raise FileNotFoundError(f"No sample files found under {dir_path}")
    selected = files[: min(limit, len(files))]
    return [_load_sample_file(path) for path in selected]



def _load_overfit_batches(dataset: Path, *, sample_limit: int) -> tuple[dict[str, Any], dict[str, Any], str]:
    if dataset.is_dir():
        src_dir = dataset / "src"
        tgt_dir = dataset / "tgt"
        if not src_dir.exists() or not tgt_dir.exists():
            raise FileNotFoundError(
                f"Directory datasets must contain 'src/' and 'tgt/' subdirectories: {dataset}"
            )
        src = collate(_load_selected_domain(src_dir, limit=sample_limit))
        tgt = collate(_load_selected_domain(tgt_dir, limit=sample_limit))
        return src, tgt, "directory"

    data = torch.load(dataset, map_location="cpu")
    src = collate(data["src"][:sample_limit])
    tgt = collate(data["tgt"][:sample_limit])
    return src, tgt, "torch_file"



def _capture_stepper_state(stepper: FrameSmoothieTrainStep) -> dict[str, Any]:
    src_corr = stepper._src_corr_ema
    if isinstance(src_corr, torch.Tensor):
        src_corr = src_corr.detach().clone()
    return {"src_corr_ema": src_corr}



def _restore_stepper_state(stepper: FrameSmoothieTrainStep, state: dict[str, Any]) -> None:
    src_corr = state.get("src_corr_ema", None)
    if isinstance(src_corr, torch.Tensor):
        src_corr = src_corr.detach().clone()
    stepper._src_corr_ema = src_corr



def _save_checkpoint(
    path: Path,
    *,
    step: int,
    preset: str,
    model: nn.Module,
    teacher: nn.Module,
    opt: torch.optim.Optimizer,
    scheduler: Optional[torch.optim.lr_scheduler.LRScheduler],
    ema_scheduler: EmaDecayScheduler,
    meter: DiagMeter,
    stepper: FrameSmoothieTrainStep,
    row: dict[str, Any],
    best_metric: Optional[float],
    topk: list[dict[str, Any]],
    global_step: int,
    runtime_state: RuntimeState,
    rng_registry: RngRegistry,
) -> Path:
    rng_snapshot = rng_registry.snapshot()
    payload = {
        "step": int(step),
        "global_step": int(global_step),
        "preset": preset,
        "model_state_dict": model.state_dict(),
        "teacher_state_dict": teacher.state_dict(),
        "optimizer_state_dict": opt.state_dict(),
        "scheduler_state_dict": None if scheduler is None else scheduler.state_dict(),
        "ema_scheduler_state_dict": ema_scheduler.state_dict(),
        "diag_meter_state_dict": meter.state_dict(),
        "stepper_state_dict": _capture_stepper_state(stepper),
        "row": row,
        "best_metric": best_metric,
        "topk": topk,
        "runtime_state": dataclasses.asdict(runtime_state),
        "rng_snapshot": dataclasses.asdict(rng_snapshot),
    }
    return save_training_checkpoint(
        path,
        payload,
        metadata={
            "preset": preset,
            "global_step": str(global_step),
            "rng_digest": rng_digest(rng_snapshot),
        },
    )



def _update_topk(
    *,
    ckpt_path: Path,
    metric_val: Optional[float],
    step: int,
    topk: list[dict[str, Any]],
    topk_limit: int,
    mode: str,
) -> list[dict[str, Any]]:
    if metric_val is None or topk_limit <= 0:
        return topk
    topk = [x for x in topk if Path(x["path"]).exists()]
    topk.append({"path": str(ckpt_path), "metric": float(metric_val), "step": int(step)})
    reverse = mode == "max"
    topk = sorted(topk, key=lambda x: x["metric"], reverse=reverse)
    kept = topk[:topk_limit]
    removed = topk[topk_limit:]
    for item in removed:
        p = Path(item["path"])
        if p.exists():
            p.unlink()
    return kept



def run_overfit(
    *,
    dataset: Path,
    preset: str,
    steps: int = 100,
    lr: float = 1e-3,
    device: str = "cpu",
    output_root: Path = Path("artifacts/runs"),
    run_name: Optional[str] = None,
    log_every: int = 10,
    save_checkpoints: bool = True,
    select_metric: str = "loss",
    select_mode: str = "min",
    resume_from: Optional[Path] = None,
    topk: int = 1,
    early_stop_metric: Optional[str] = None,
    early_stop_mode: Optional[str] = None,
    early_stop_patience: int = 0,
    early_stop_min_delta: float = 0.0,
    scheduler_kind: str = "none",
    scheduler_step_size: int = 50,
    scheduler_gamma: float = 0.5,
    ema_schedule_kind: str = "constant",
    ema_beta_start: float = 0.99,
    ema_beta_end: float = 0.999,
    seed: int = 0,
    dtype_bits: Literal[16, 32, 64] = 32,
    microbatch_size: Optional[int] = None,
    cpu_fallback_on_cuda_oom: bool = True,
    max_oom_retries: int = 16,
    sample_limit: int = 4,
    export_final_artifacts: bool = True,
) -> Dict[str, Any]:
    dataset = Path(dataset)
    output_root = Path(output_root)
    preset_cfg = get_preset(preset)
    mp_context = choose_multiprocessing_context()
    determinism = DeterminismConfig(
        master_seed=seed,
        multiprocessing_context=mp_context,
    )
    environment = configure_determinism(determinism)
    seed_root = SeedPath(seed).child("run_overfit", preset)
    rng_registry = RngRegistry()

    src, tgt, dataset_format = _load_overfit_batches(dataset, sample_limit=sample_limit)

    dev = torch.device(device)
    c_model = int(src["image"].shape[1])
    transform_fwd = DOST(D=2)
    transform_inv = IDOST(D=2)
    with torch.no_grad():
        z_probe = transform_fwd(src["image"][:1])
    enc_c_model = int(z_probe.shape[1])

    dtype_idx: FPDTypeIdx = dtype_bits * 2

    model_kwargs = dict(
        transform_fwd=transform_fwd,
        transform_inv=transform_inv,
        encoder=S9Stack(c_model=enc_c_model, depth=1, spatial_dims=2, dtype_idx=dtype_idx),
        c_model=c_model,
        enc_c_model=enc_c_model,
        spatial_dims=2,
        num_semantic_classes=4,
        num_instance_classes=2,
        q_dim=c_model,
        num_queries=4,
        decoder_layers=1,
        dtype_idx=dtype_idx,
        pre_bridge_map=preset_cfg.get("pre_bridge_map", "log1p_arsinh"),
        pre_bridge_channels=preset_cfg.get("pre_bridge_channels", c_model) or c_model,
        use_zoning=preset_cfg.get("use_zoning", False),
        zone_names=preset_cfg.get("zone_names", ("content", "structure", "label", "boundary")),
        semantic_zone_names=preset_cfg.get("semantic_zone_names", ("content", "structure", "boundary")),
        instance_zone_names=preset_cfg.get("instance_zone_names", ("content", "label", "boundary")),
        semantic_zone_weights=preset_cfg.get("semantic_zone_weights", None),
        instance_zone_weights=preset_cfg.get("instance_zone_weights", None),
        use_zone_prediction=preset_cfg.get("use_zone_prediction", False),
        zone_pred_edges=preset_cfg.get("zone_pred_edges", ()),
        use_lrca=preset_cfg.get("use_lrca", False),
        lrca_rank_shared=preset_cfg.get("lrca_rank_shared", 4),
        lrca_rank_private=preset_cfg.get("lrca_rank_private", 2),
        lrca_fmlm_rank=preset_cfg.get("lrca_fmlm_rank", 4),
        lrca_fmlm_eta=preset_cfg.get("lrca_fmlm_eta", 0.1),
    )
    model = FrameSmoothiePanopticModel(**model_kwargs).to(dev)
    teacher = make_ema_teacher(model).to(dev)

    matcher = HungarianMatcher(num_points=256)
    criterion = SetCriterion(num_classes=2, matcher=matcher, num_points=128)
    hmc_kwargs = {"thing_classes": (0, 1), "min_area": 8}
    hmc_kwargs.update(preset_cfg.get("hmc_kwargs", {}))
    hmc = HMCCalibrator(**hmc_kwargs)
    stepper = FrameSmoothieTrainStep(
        criterion_inst=criterion,
        hmc=hmc,
        thing_classes=[0, 1],
        lambda_pred=preset_cfg.get("lambda_pred", 0.0),
        pred_detach_target=True,
        use_edge_weights=True,
    )
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=lr)
    scheduler = _build_scheduler(opt, scheduler_kind=scheduler_kind, steps=steps, step_size=scheduler_step_size, gamma=scheduler_gamma)
    ema_scheduler = EmaDecayScheduler(kind=ema_schedule_kind, beta_start=ema_beta_start, beta_end=ema_beta_end, total_steps=steps, last_step=0)
    meter = DiagMeter()

    runtime_policy = RuntimePolicy(
        device=str(dev),
        microbatch_size=microbatch_size,
        cpu_fallback_on_cuda_oom=cpu_fallback_on_cuda_oom,
        max_oom_retries=max_oom_retries,
        empty_cuda_cache_on_retry=True,
        empty_cuda_cache_every=0,
        grad_clip_norm=None,
    )
    runtime_state = RuntimeState(effective_device=str(dev), microbatch_size=microbatch_size)

    if resume_from is not None:
        resume_from = Path(resume_from)
        ckpt = load_training_checkpoint(resume_from, device="cpu")
        run_dir = resume_from.parent.parent if resume_from.parent.name == "checkpoints" else resume_from.parent
        model.load_state_dict(ckpt["model_state_dict"])
        teacher.load_state_dict(ckpt["teacher_state_dict"])
        model.to(dev)
        teacher.to(dev)
        opt.load_state_dict(ckpt["optimizer_state_dict"])
        for state in opt.state.values():
            for key, value in list(state.items()):
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(dev)
        if scheduler is not None and ckpt.get("scheduler_state_dict", None) is not None:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if ckpt.get("diag_meter_state_dict", None) is not None:
            meter.load_state_dict(ckpt["diag_meter_state_dict"])
        if ckpt.get("ema_scheduler_state_dict", None) is not None:
            ema_scheduler.load_state_dict(ckpt["ema_scheduler_state_dict"])
        if ckpt.get("stepper_state_dict", None) is not None:
            _restore_stepper_state(stepper, ckpt["stepper_state_dict"])
        if ckpt.get("runtime_state", None) is not None:
            runtime_state = RuntimeState(**ckpt["runtime_state"])
        start_step = int(ckpt.get("global_step", ckpt.get("step", 0)))
        best_metric: Optional[float] = ckpt.get("best_metric", None)
        topk_manifest: list[dict[str, Any]] = ckpt.get("topk", [])
        rows: list[dict[str, Any]] = _read_jsonl(run_dir / "metrics.jsonl")
        runtime_events: list[dict[str, Any]] = _read_jsonl(run_dir / "runtime_events.jsonl")
        resumed_from = str(resume_from)

        rng_snapshot_dict = ckpt.get("rng_snapshot", None)
        if rng_snapshot_dict is not None:
            restore_rng_snapshot(
                snapshot=RngSnapshot(**rng_snapshot_dict),
                numpy_generators=rng_registry.numpy_generators,
                torch_generators=rng_registry.torch_generators,
            )
    else:
        run_dir = _make_run_dir(output_root, preset, run_name)
        start_step = 0
        best_metric = None
        topk_manifest = []
        rows = []
        runtime_events = []
        resumed_from = None
        config_payload = _to_jsonable({
            "preset": preset,
            "steps": steps,
            "lr": lr,
            "device": device,
            "dataset": dataset,
            "dataset_format": dataset_format,
            "sample_limit": sample_limit,
            "scheduler_kind": scheduler_kind,
            "scheduler_step_size": scheduler_step_size,
            "scheduler_gamma": scheduler_gamma,
            "ema_schedule_kind": ema_schedule_kind,
            "ema_beta_start": ema_beta_start,
            "ema_beta_end": ema_beta_end,
            "seed": seed,
            "dtype_bits": dtype_bits,
            "microbatch_size": microbatch_size,
            "cpu_fallback_on_cuda_oom": cpu_fallback_on_cuda_oom,
            "max_oom_retries": max_oom_retries,
            "determinism": determinism,
            "runtime_policy": runtime_policy,
            "environment": environment,
            "model_kwargs": {k: str(v) if k == "zone_pred_edges" else v for k, v in model_kwargs.items()},
        })
        (run_dir / "config.json").write_text(
            json.dumps(config_payload, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    last_out: Dict[str, Any] = {}
    ckpt_dir = run_dir / "checkpoints"
    if save_checkpoints:
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    es_metric_name = early_stop_metric or select_metric
    es_mode = early_stop_mode or select_mode
    es_best: Optional[float] = None
    es_bad_count = 0
    early_stopped = False
    stopped_step: Optional[int] = None
    best_ckpt: Optional[Path] = Path(topk_manifest[0]["path"]) if topk_manifest else None

    if rows and early_stop_patience > 0:
        for r in rows:
            v = _select_key(r, es_metric_name)
            if v is None:
                continue
            if es_best is None or _is_better(v, es_best, es_mode, early_stop_min_delta):
                es_best = v
                es_bad_count = 0
            else:
                es_bad_count += 1

    step_fn = _call_train_step(stepper)(model)(teacher)(meter)(False)
    runtime_env = RuntimeEnv(
        student=model,
        teacher=teacher,
        optimizer=opt,
        step_fn=step_fn,
        policy=runtime_policy,
        diag_meter=meter,
        capture_ephemeral_state=lambda: _capture_stepper_state(stepper),
        restore_ephemeral_state=lambda state: _restore_stepper_state(stepper, state),
    )

    for it in range(start_step, steps):
        runtime_state, (out, runtime_report), program_log = make_step_program(src, tgt).execute(runtime_env, runtime_state)
        if runtime_policy.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=float(runtime_policy.grad_clip_norm))
        opt.step()
        if scheduler is not None:
            scheduler.step()
        ema_beta = ema_scheduler.step()
        ema_update(teacher, model, beta=ema_beta)
        last_out = out
        runtime_events.extend(dict(event) for event in program_log.events)

        if (it + 1) % log_every == 0 or (it + 1) == steps:
            means = meter.compute_means()
            current_rng_snapshot = rng_registry.snapshot()
            row = {
                "step": it + 1,
                "loss": float(out["loss"].detach().cpu().item()),
                "loss_src_sem": float(out["loss_src_sem"].cpu().item()),
                "loss_src_inst": float(out["loss_src_inst"].cpu().item()),
                "loss_tgt_sem": float(out["loss_tgt_sem"].cpu().item()),
                "loss_tgt_inst": float(out["loss_tgt_inst"].cpu().item()),
                "loss_aux": float(out["loss_aux"].cpu().item()),
                "corr_ref_src_ema": means.get("corr_ref_src_ema"),
                "corr_tgt_teacher": means.get("tgt_teacher/corr_sem_inst"),
                "corr_tgt_student": means.get("tgt_student/corr_sem_inst"),
                "lr": float(opt.param_groups[0]["lr"]),
                "ema_beta": float(ema_beta),
                "effective_device": runtime_report.effective_device,
                "effective_microbatch_size": int(runtime_report.effective_microbatch_size),
                "num_microbatches": int(runtime_report.num_microbatches),
                "oom_retries": int(runtime_report.oom_retries),
                "cpu_fallback_used": bool(runtime_report.cpu_fallback_used),
                "rng_digest": rng_digest(current_rng_snapshot),
            }
            rows.append(row)

            metric_val = _select_key(row, select_metric)
            if metric_val is not None:
                if best_metric is None or _is_better(metric_val, best_metric, select_mode):
                    best_metric = metric_val

            if save_checkpoints:
                ckpt_path = ckpt_dir / f"step_{it + 1:06d}.safetensors"
                topk_manifest = _update_topk(
                    ckpt_path=ckpt_path,
                    metric_val=metric_val,
                    step=it + 1,
                    topk=topk_manifest,
                    topk_limit=max(1, int(topk)),
                    mode=select_mode,
                )
                ckpt_path = _save_checkpoint(
                    ckpt_path,
                    step=it + 1,
                    preset=preset,
                    model=model,
                    teacher=teacher,
                    opt=opt,
                    scheduler=scheduler,
                    ema_scheduler=ema_scheduler,
                    meter=meter,
                    stepper=stepper,
                    row=row,
                    best_metric=best_metric,
                    topk=topk_manifest,
                    global_step=it + 1,
                    runtime_state=runtime_state,
                    rng_registry=rng_registry,
                )
                if topk_manifest:
                    best_ckpt = Path(topk_manifest[0]["path"])
                    best_link = run_dir / "best_checkpoint.safetensors"
                    if best_ckpt.exists():
                        shutil.copy2(best_ckpt, best_link)
                    manifest_path = run_dir / "topk_checkpoints.json"
                    manifest_path.write_text(json.dumps(topk_manifest, indent=2, ensure_ascii=False), encoding="utf-8")

            if early_stop_patience > 0:
                es_val = _select_key(row, es_metric_name)
                if es_val is not None:
                    if es_best is None or _is_better(es_val, es_best, es_mode, early_stop_min_delta):
                        es_best = es_val
                        es_bad_count = 0
                    else:
                        es_bad_count += 1
                        if es_bad_count >= early_stop_patience:
                            early_stopped = True
                            stopped_step = it + 1

            corr_ref_text = ""
            if row["corr_ref_src_ema"] is not None:
                corr_ref_text = f" corr_ref={row['corr_ref_src_ema']:.4f}"
            print(
                f"[{preset}] step={it + 1:04d} "
                f"loss={row['loss']:.4f}"
                f" mb={row['effective_microbatch_size']}"
                f" dev={row['effective_device']}"
                f" oom={row['oom_retries']}"
                f"{corr_ref_text}"
            )
            meter.reset()

            if early_stopped:
                print(f"[{preset}] early stopping at step={stopped_step} on metric={es_metric_name}")
                break

    _write_jsonl(run_dir / "metrics.jsonl", rows)
    _save_metrics_csv(run_dir / "metrics.csv", rows)
    _write_jsonl(run_dir / "runtime_events.jsonl", runtime_events)

    final_student_artifact = None
    final_teacher_artifact = None
    if export_final_artifacts:
        final_student_artifact = export_state_dict_artifact(
            run_dir / "student_final.safetensors",
            model,
            metadata={
                "preset": preset,
                "steps_executed": str(rows[-1]["step"] if rows else start_step),
                "seed": str(seed),
            },
        )
        final_teacher_artifact = export_state_dict_artifact(
            run_dir / "teacher_final.safetensors",
            teacher,
            metadata={
                "preset": preset,
                "steps_executed": str(rows[-1]["step"] if rows else start_step),
                "seed": str(seed),
            },
        )

    summary = {
        "preset": preset,
        "run_dir": str(run_dir),
        "steps_requested": steps,
        "steps_executed": rows[-1]["step"] if rows else start_step,
        "device_requested": device,
        "device_effective": runtime_state.effective_device,
        "resumed_from": resumed_from,
        "dataset": str(dataset),
        "dataset_format": dataset_format,
        "sample_limit": int(sample_limit),
        "scheduler_kind": scheduler_kind,
        "scheduler_step_size": int(scheduler_step_size),
        "scheduler_gamma": float(scheduler_gamma),
        "ema_schedule_kind": ema_schedule_kind,
        "ema_beta_start": float(ema_beta_start),
        "ema_beta_end": float(ema_beta_end),
        "final_ema_beta": float(ema_scheduler.get_last_beta()),
        "save_checkpoints": save_checkpoints,
        "select_metric": select_metric,
        "select_mode": select_mode,
        "topk": int(topk),
        "best_metric": best_metric,
        "best_checkpoint": str(best_ckpt) if best_ckpt is not None else None,
        "topk_manifest": topk_manifest,
        "early_stop_metric": es_metric_name if early_stop_patience > 0 else None,
        "early_stop_mode": es_mode if early_stop_patience > 0 else None,
        "early_stop_patience": int(early_stop_patience),
        "early_stop_min_delta": float(early_stop_min_delta),
        "early_stopped": bool(early_stopped),
        "stopped_step": stopped_step,
        "final_loss": float(last_out["loss"].detach().cpu().item()) if last_out else None,
        "final_loss_src_sem": float(last_out["loss_src_sem"].cpu().item()) if last_out else None,
        "final_loss_src_inst": float(last_out["loss_src_inst"].cpu().item()) if last_out else None,
        "final_loss_tgt_sem": float(last_out["loss_tgt_sem"].cpu().item()) if last_out else None,
        "final_loss_tgt_inst": float(last_out["loss_tgt_inst"].cpu().item()) if last_out else None,
        "rows_logged": len(rows),
        "runtime_events_logged": len(runtime_events),
        "final_lr": float(opt.param_groups[0]["lr"]),
        "seed": int(seed),
        "dtype_bits": int(dtype_bits),
        "microbatch_size_requested": microbatch_size,
        "microbatch_size_effective": runtime_state.last_effective_microbatch_size,
        "cpu_fallback_on_cuda_oom": bool(cpu_fallback_on_cuda_oom),
        "oom_events": runtime_state.oom_events,
        "final_rng_digest": rng_digest(rng_registry.snapshot()),
        "student_final_artifact": str(final_student_artifact) if final_student_artifact is not None else None,
        "teacher_final_artifact": str(final_teacher_artifact) if final_teacher_artifact is not None else None,
    }
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, required=True)
    ap.add_argument("--preset", type=str, default="pred", choices=sorted(list_presets()))
    ap.add_argument("--steps", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--output-root", type=Path, default=Path("artifacts/runs"))
    ap.add_argument("--run-name", type=str, default=None)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--save-checkpoints", action="store_true")
    ap.add_argument("--select-metric", type=str, default="loss")
    ap.add_argument("--select-mode", type=str, default="min", choices=["min", "max"])
    ap.add_argument("--resume-from", type=Path, default=None)
    ap.add_argument("--topk", type=int, default=1)
    ap.add_argument("--early-stop-metric", type=str, default=None)
    ap.add_argument("--early-stop-mode", type=str, default=None, choices=[None, "min", "max"])
    ap.add_argument("--early-stop-patience", type=int, default=0)
    ap.add_argument("--early-stop-min-delta", type=float, default=0.0)
    ap.add_argument("--scheduler-kind", type=str, default="none", choices=["none", "cosine", "step"])
    ap.add_argument("--scheduler-step-size", type=int, default=50)
    ap.add_argument("--scheduler-gamma", type=float, default=0.5)
    ap.add_argument("--ema-schedule-kind", type=str, default="constant", choices=["constant", "linear", "cosine"])
    ap.add_argument("--ema-beta-start", type=float, default=0.99)
    ap.add_argument("--ema-beta-end", type=float, default=0.999)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype-bits", type=int, default=32, choices=[16, 32, 64])
    ap.add_argument("--microbatch-size", type=int, default=None)
    ap.add_argument("--sample-limit", type=int, default=4)
    ap.add_argument("--no-cpu-fallback-on-cuda-oom", action="store_true")
    ap.add_argument("--max-oom-retries", type=int, default=16)
    ap.add_argument("--no-export-final-artifacts", action="store_true")
    args = ap.parse_args()

    run_overfit(
        dataset=args.dataset,
        preset=args.preset,
        steps=args.steps,
        lr=args.lr,
        device=args.device,
        output_root=args.output_root,
        run_name=args.run_name,
        log_every=args.log_every,
        save_checkpoints=args.save_checkpoints,
        select_metric=args.select_metric,
        select_mode=args.select_mode,
        resume_from=args.resume_from,
        topk=args.topk,
        early_stop_metric=args.early_stop_metric,
        early_stop_mode=args.early_stop_mode,
        early_stop_patience=args.early_stop_patience,
        early_stop_min_delta=args.early_stop_min_delta,
        scheduler_kind=args.scheduler_kind,
        scheduler_step_size=args.scheduler_step_size,
        scheduler_gamma=args.scheduler_gamma,
        ema_schedule_kind=args.ema_schedule_kind,
        ema_beta_start=args.ema_beta_start,
        ema_beta_end=args.ema_beta_end,
        seed=args.seed,
        dtype_bits=args.dtype_bits,
        microbatch_size=args.microbatch_size,
        cpu_fallback_on_cuda_oom=not args.no_cpu_fallback_on_cuda_oom,
        max_oom_retries=args.max_oom_retries,
        sample_limit=args.sample_limit,
        export_final_artifacts=not args.no_export_final_artifacts,
    )


if __name__ == "__main__":
    main()
