"""Checkpoint migration utilities for framesmoothie v0.3.x → v0.4.0.

Thin re-export of ``s9.migration`` with framesmoothie-level convenience
wrappers. The underlying rescaling logic pattern-matches on
``...kernels.{i}.{param}`` prefixes, so it works seamlessly with any
framesmoothie model state_dict regardless of the outer module hierarchy.

Usage::

    from framesmoothie.migration import migrate_state_dict_zoh

    # v0.3.x checkpoint → v0.4.0
    new_sd = migrate_state_dict_zoh(old_state_dict)
    model.load_state_dict(new_sd)

    # Roundtrip: v0.4.0 → v0.3.x (lossless)
    from framesmoothie.migration import migrate_state_dict_from_zoh
    old_sd_back = migrate_state_dict_from_zoh(new_sd)
"""

from __future__ import annotations

from typing import Any, Optional

from s9.migration import (
    migrate_state_dict_zoh as _migrate_state_dict_zoh,
    migrate_state_dict_from_zoh as _migrate_state_dict_from_zoh,
)

__all__ = [
    "migrate_state_dict_zoh",
    "migrate_state_dict_from_zoh",
    "migrate_framesmoothie_checkpoint",
]


# Re-exports (keep the s9 signatures intact)
migrate_state_dict_zoh = _migrate_state_dict_zoh
migrate_state_dict_from_zoh = _migrate_state_dict_from_zoh


def migrate_framesmoothie_checkpoint(
    checkpoint: dict,
    *,
    direction: str = "to_zoh",
    state_dict_key: Optional[str] = None,
) -> dict:
    """Migrate a framesmoothie checkpoint dict in place-equivalent style.

    Args:
        checkpoint: Output of ``torch.load``. May be either a raw state_dict
            or a dict containing one under ``state_dict_key``.
        direction: ``"to_zoh"`` (v0.3.x → v0.4.0) or ``"from_zoh"`` (reverse).
        state_dict_key: Nested key if the state_dict is wrapped. If ``None``,
            tries ``"state_dict"`` → ``"model"`` → treat the whole dict as the
            state_dict (same heuristic as s9's CLI).

    Returns:
        A new checkpoint dict with the SSM ``B`` parameters rescaled. The
        migration is lossless: forward outputs are preserved under the matched
        discretization.
    """
    migrate_fn = migrate_state_dict_zoh if direction == "to_zoh" else migrate_state_dict_from_zoh

    key: Optional[str] = state_dict_key
    if key is not None:
        sd = checkpoint[key]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        sd = checkpoint["state_dict"]
        key = "state_dict"
    elif isinstance(checkpoint, dict) and "model" in checkpoint:
        sd = checkpoint["model"]
        key = "model"
    else:
        sd = checkpoint
        key = None

    new_sd = migrate_fn(sd)

    if key is None:
        return new_sd

    new_checkpoint: dict[str, Any] = dict(checkpoint)
    new_checkpoint[key] = new_sd
    return new_checkpoint
