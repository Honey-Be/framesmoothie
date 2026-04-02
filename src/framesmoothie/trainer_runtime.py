from __future__ import annotations

"""Memory-safe training-step runtime with deterministic state transitions."""

from dataclasses import dataclass, field
import math
from typing import Any, Callable, Mapping, MutableMapping, Optional

import torch
from torch import nn

from framesmoothie.fp import Run, RunLog


Batch = dict[str, Any]
StepFn = Callable[[Batch, Batch], dict[str, Any]]
CaptureStateFn = Callable[[], Any]
RestoreStateFn = Callable[[Any], None]


@dataclass(frozen=True)
class RuntimePolicy:
    device: str = "cpu"
    microbatch_size: int | None = None
    cpu_fallback_on_cuda_oom: bool = True
    max_oom_retries: int = 16
    empty_cuda_cache_on_retry: bool = True
    empty_cuda_cache_every: int = 0
    grad_clip_norm: float | None = None


@dataclass
class RuntimeState:
    effective_device: str
    microbatch_size: int | None = None
    oom_events: list[dict[str, Any]] = field(default_factory=list)
    steps_completed: int = 0
    last_effective_microbatch_size: int | None = None


@dataclass
class RuntimeEnv:
    student: nn.Module
    teacher: nn.Module
    optimizer: torch.optim.Optimizer
    step_fn: StepFn
    policy: RuntimePolicy
    diag_meter: Any | None = None
    capture_ephemeral_state: CaptureStateFn | None = None
    restore_ephemeral_state: RestoreStateFn | None = None


@dataclass(frozen=True)
class AdaptiveStepReport:
    effective_device: str
    effective_microbatch_size: int
    num_microbatches: int
    oom_retries: int
    cpu_fallback_used: bool


def _is_oom(exc: BaseException) -> bool:
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    text = str(exc).lower()
    return "out of memory" in text and ("cuda" in text or "memory" in text)


def _clear_cuda_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _move_optimizer_state_(optimizer: torch.optim.Optimizer, device: str | torch.device) -> None:
    tgt = torch.device(device)
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device=tgt)


def move_training_state_(student: nn.Module, teacher: nn.Module, optimizer: torch.optim.Optimizer, device: str | torch.device) -> None:
    student.to(device)
    teacher.to(device)
    _move_optimizer_state_(optimizer, device)


def infer_batch_size(batch: Batch) -> int:
    for value in batch.values():
        if isinstance(value, torch.Tensor):
            return int(value.shape[0])
        if isinstance(value, (list, tuple)):
            return len(value)
    raise ValueError("Unable to infer batch size from batch payload")


def slice_batch(batch: Batch, start: int, end: int) -> Batch:
    sliced: Batch = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            sliced[key] = value[start:end]
        elif isinstance(value, list):
            sliced[key] = value[start:end]
        elif isinstance(value, tuple):
            sliced[key] = value[start:end]
        else:
            sliced[key] = value
    return sliced


def _iter_microbatches(src: Batch, tgt: Batch, microbatch_size: int) -> list[tuple[Batch, Batch, float]]:
    src_bs = infer_batch_size(src)
    tgt_bs = infer_batch_size(tgt)
    full_bs = min(src_bs, tgt_bs)
    if full_bs <= 0:
        raise ValueError("Empty batches are not supported")
    chunks: list[tuple[Batch, Batch, float]] = []
    for start in range(0, full_bs, microbatch_size):
        end = min(start + microbatch_size, full_bs)
        weight = float(end - start) / float(full_bs)
        chunks.append((slice_batch(src, start, end), slice_batch(tgt, start, end), weight))
    return chunks


def _aggregate_step_outputs(outputs: list[tuple[dict[str, Any], float]]) -> dict[str, Any]:
    if not outputs:
        return {}
    aggregate: dict[str, Any] = {}
    scalar_keys = {
        key
        for out, _weight in outputs
        for key, value in out.items()
        if isinstance(value, torch.Tensor) and value.ndim == 0
    }
    for key in scalar_keys:
        acc: torch.Tensor | None = None
        for out, weight in outputs:
            if key not in out:
                continue
            value = out[key].detach().reshape(())
            scaled = value * weight
            acc = scaled if acc is None else acc + scaled
        if acc is not None:
            aggregate[key] = acc

    # carry last non-scalar payload for debugging / summaries
    last_out = outputs[-1][0]
    for key, value in last_out.items():
        if key not in aggregate:
            aggregate[key] = value
    return aggregate


def _run_backward_with_microbatches(
    env: RuntimeEnv,
    *,
    src: Batch,
    tgt: Batch,
    microbatch_size: int,
) -> tuple[dict[str, Any], AdaptiveStepReport]:
    outputs: list[tuple[dict[str, Any], float]] = []
    oom_retries = 0
    cpu_fallback_used = False
    current_device = next(env.student.parameters()).device.type
    effective_microbatch = microbatch_size

    meter_state = env.diag_meter.state_dict() if env.diag_meter is not None and hasattr(env.diag_meter, "state_dict") else None
    ephemeral_state = env.capture_ephemeral_state() if env.capture_ephemeral_state is not None else None

    while True:
        env.optimizer.zero_grad(set_to_none=True)
        outputs.clear()
        if meter_state is not None and env.diag_meter is not None and hasattr(env.diag_meter, "load_state_dict"):
            env.diag_meter.load_state_dict(meter_state)
        if ephemeral_state is not None and env.restore_ephemeral_state is not None:
            env.restore_ephemeral_state(ephemeral_state)
        try:
            microbatches = _iter_microbatches(src, tgt, effective_microbatch)
            for src_mb, tgt_mb, weight in microbatches:
                out = env.step_fn(src_mb, tgt_mb)
                loss = out["loss"] * weight
                loss.backward()
                outputs.append((out, weight))
            aggregate = _aggregate_step_outputs(outputs)
            return aggregate, AdaptiveStepReport(
                effective_device=current_device,
                effective_microbatch_size=effective_microbatch,
                num_microbatches=len(microbatches),
                oom_retries=oom_retries,
                cpu_fallback_used=cpu_fallback_used,
            )
        except RuntimeError as exc:
            if not _is_oom(exc):
                raise
            oom_retries += 1
            env.optimizer.zero_grad(set_to_none=True)
            if env.policy.empty_cuda_cache_on_retry:
                _clear_cuda_cache()
            if oom_retries > env.policy.max_oom_retries:
                raise
            if effective_microbatch > 1:
                effective_microbatch = max(1, math.ceil(effective_microbatch / 2))
                continue
            if current_device != "cpu" and env.policy.cpu_fallback_on_cuda_oom:
                move_training_state_(env.student, env.teacher, env.optimizer, "cpu")
                current_device = "cpu"
                cpu_fallback_used = True
                effective_microbatch = 1
                continue
            raise


def make_step_program(src: Batch, tgt: Batch) -> Run[RuntimeEnv, RuntimeState, tuple[dict[str, Any], AdaptiveStepReport]]:
    def _program(env: RuntimeEnv, state: RuntimeState) -> tuple[RuntimeState, tuple[dict[str, Any], AdaptiveStepReport], RunLog]:
        desired_device = state.effective_device or env.policy.device
        current_device = next(env.student.parameters()).device.type
        if current_device != torch.device(desired_device).type:
            move_training_state_(env.student, env.teacher, env.optimizer, desired_device)

        full_batch = min(infer_batch_size(src), infer_batch_size(tgt))
        start_microbatch = state.microbatch_size or env.policy.microbatch_size or full_batch
        aggregate, report = _run_backward_with_microbatches(
            env,
            src=src,
            tgt=tgt,
            microbatch_size=max(1, int(start_microbatch)),
        )
        new_state = RuntimeState(
            effective_device=report.effective_device,
            microbatch_size=report.effective_microbatch_size,
            oom_events=state.oom_events + ([{
                "event": "oom_recovery",
                "from_device": desired_device,
                "to_device": report.effective_device,
                "effective_microbatch_size": report.effective_microbatch_size,
                "oom_retries": report.oom_retries,
                "cpu_fallback_used": report.cpu_fallback_used,
            }] if report.oom_retries > 0 else []),
            steps_completed=state.steps_completed + 1,
            last_effective_microbatch_size=report.effective_microbatch_size,
        )
        events: list[dict[str, Any]] = [
            {
                "event": "train_step",
                "step_index": new_state.steps_completed,
                "effective_device": report.effective_device,
                "effective_microbatch_size": report.effective_microbatch_size,
                "num_microbatches": report.num_microbatches,
                "oom_retries": report.oom_retries,
                "cpu_fallback_used": report.cpu_fallback_used,
            }
        ]
        if report.oom_retries > 0:
            events.extend(new_state.oom_events[-1:])
        return new_state, (aggregate, report), RunLog(tuple(events))

    return Run(_program)


__all__ = [
    "AdaptiveStepReport",
    "RuntimeEnv",
    "RuntimePolicy",
    "RuntimeState",
    "infer_batch_size",
    "make_step_program",
    "move_training_state_",
    "slice_batch",
]
