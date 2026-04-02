from __future__ import annotations

"""Checkpoint helpers built on the local safetensors implementation."""

from pathlib import Path
from typing import Any, Mapping

import torch
from torch import nn

from framesmoothie.safetensors_io import load_object, load_state_dict, save_object, save_state_dict


def save_training_checkpoint(path: str | Path, payload: Mapping[str, Any], *, metadata: Mapping[str, str] | None = None) -> Path:
    path = Path(path)
    if path.suffix != ".safetensors":
        path = path.with_suffix(".safetensors")
    return save_object(path, dict(payload), metadata=metadata)


def load_training_checkpoint(path: str | Path, *, device: str | torch.device = "cpu") -> dict[str, Any]:
    payload = load_object(path, device=device)
    if not isinstance(payload, dict):
        raise TypeError("Training checkpoint payload must decode to a dictionary")
    return payload


def export_state_dict_artifact(
    path: str | Path,
    module: nn.Module,
    *,
    metadata: Mapping[str, str] | None = None,
) -> Path:
    path = Path(path)
    if path.suffix != ".safetensors":
        path = path.with_suffix(".safetensors")
    return save_state_dict(path, module.state_dict(), metadata=metadata)


def load_state_dict_artifact(path: str | Path, *, device: str | torch.device = "cpu") -> dict[str, torch.Tensor]:
    return load_state_dict(path, device=device)


__all__ = [
    "export_state_dict_artifact",
    "load_state_dict_artifact",
    "load_training_checkpoint",
    "save_training_checkpoint",
]
