"""Test framesmoothie migration wrappers."""

import copy
import torch

from framesmoothie.blocks import RS9CondMixBlock
from framesmoothie.blocks_ars9 import ARS9CondMixBlock
from framesmoothie.migration import (
    migrate_state_dict_zoh,
    migrate_state_dict_from_zoh,
    migrate_framesmoothie_checkpoint,
)


def _forward_approx(block_cls, init_kwargs):
    block = block_cls(
        c_model=16, q_dim=8, spatial_dims=1,
        gate_dim=8, ffn_mult=2, dtype_idx=64,
        discretization="approx",
        **init_kwargs,
    )
    block.eval()
    x = torch.randn(2, 6, 16)
    q = torch.randn(2, 3, 8)
    with torch.no_grad():
        out = block(x, q)
    return block, out["q"], x, q


def test_rs9_forward_equivalence_after_migration():
    """approx model → migrate → zoh model: forward outputs match."""
    torch.manual_seed(0)
    block_approx, y_approx, x, q = _forward_approx(RS9CondMixBlock, {})

    sd_approx = block_approx.state_dict()
    sd_zoh = migrate_state_dict_zoh(sd_approx)

    torch.manual_seed(0)
    block_zoh = RS9CondMixBlock(
        c_model=16, q_dim=8, spatial_dims=1,
        gate_dim=8, ffn_mult=2, dtype_idx=64,
        discretization="zoh",
    )
    block_zoh.load_state_dict(sd_zoh)
    block_zoh.eval()

    with torch.no_grad():
        y_zoh = block_zoh(x, q)["q"]

    torch.testing.assert_close(y_approx, y_zoh, atol=1e-5, rtol=1e-5)


def test_ars9_forward_equivalence_after_migration():
    """ARS9 variant also migrates losslessly."""
    torch.manual_seed(0)
    block_approx, y_approx, x, q = _forward_approx(ARS9CondMixBlock, {})

    sd_approx = block_approx.state_dict()
    sd_zoh = migrate_state_dict_zoh(sd_approx)

    torch.manual_seed(0)
    block_zoh = ARS9CondMixBlock(
        c_model=16, q_dim=8, spatial_dims=1,
        gate_dim=8, ffn_mult=2, dtype_idx=64,
        discretization="zoh",
    )
    block_zoh.load_state_dict(sd_zoh)
    block_zoh.eval()

    with torch.no_grad():
        y_zoh = block_zoh(x, q)["q"]

    torch.testing.assert_close(y_approx, y_zoh, atol=1e-5, rtol=1e-5)


def test_roundtrip_preserves_state_dict():
    """approx → zoh → approx yields identical parameters."""
    block = RS9CondMixBlock(
        c_model=16, q_dim=8, spatial_dims=1,
        gate_dim=8, ffn_mult=2, dtype_idx=64,
        discretization="approx",
    )
    sd_orig = block.state_dict()
    sd_zoh = migrate_state_dict_zoh(sd_orig)
    sd_back = migrate_state_dict_from_zoh(sd_zoh)

    for key in sd_orig:
        torch.testing.assert_close(
            sd_orig[key], sd_back[key], atol=1e-6, rtol=1e-6,
            msg=f"Roundtrip mismatch on {key}",
        )


def test_checkpoint_wrapper_handles_nested_dict():
    """migrate_framesmoothie_checkpoint handles {'model': sd, 'optimizer': ...} style."""
    block = RS9CondMixBlock(
        c_model=16, q_dim=8, spatial_dims=1,
        gate_dim=8, ffn_mult=2, dtype_idx=64,
        discretization="approx",
    )
    checkpoint = {"model": block.state_dict(), "epoch": 42}

    migrated = migrate_framesmoothie_checkpoint(checkpoint)

    assert "model" in migrated
    assert migrated["epoch"] == 42
    # B parameters should be different
    orig_B = checkpoint["model"]["rs9.kernels.0.B"]
    new_B = migrated["model"]["rs9.kernels.0.B"]
    assert not torch.equal(orig_B, new_B)


def test_checkpoint_wrapper_bare_state_dict():
    """migrate_framesmoothie_checkpoint handles bare state_dict too."""
    block = RS9CondMixBlock(
        c_model=16, q_dim=8, spatial_dims=1,
        gate_dim=8, ffn_mult=2, dtype_idx=64,
        discretization="approx",
    )
    sd = block.state_dict()
    migrated = migrate_framesmoothie_checkpoint(sd)
    assert set(migrated.keys()) == set(sd.keys())
