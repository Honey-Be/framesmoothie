"""Smoke tests for ARS9CondMixBlock / ARS9DecoderLayer / ARS9Decoder."""

import pytest
import torch

from framesmoothie.blocks_ars9 import ARS9CondMixBlock
from framesmoothie.decoder_ars9 import ARS9DecoderLayer, ARS9Decoder


def _forward_shapes(block_cls, spatial_dims: int):
    B, K = 2, 3
    c_model = 16
    q_dim = 8
    spatial = (6,) * spatial_dims

    block = block_cls(
        c_model=c_model,
        q_dim=q_dim,
        spatial_dims=spatial_dims,
        gate_dim=8,
        ffn_mult=2,
        return_masks=True,
        dtype_idx=64,
    )

    x = torch.randn(B, *spatial, c_model)
    q = torch.randn(B, K, q_dim)

    out = block(x, q)
    assert out["q"].shape == (B, K, q_dim)
    assert "mask_logits" in out
    assert out["mask_logits"].shape == (B, K, *spatial)


@pytest.mark.parametrize("spatial_dims", [1, 2])
def test_ars9_cond_mix_block(spatial_dims):
    _forward_shapes(ARS9CondMixBlock, spatial_dims)


def test_ars9_decoder_layer_and_stack():
    B, K = 2, 3
    c_model = 16
    q_dim = 8
    spatial_dims = 2
    spatial = (6, 6)

    layer = ARS9DecoderLayer(
        c_model=c_model,
        q_dim=q_dim,
        spatial_dims=spatial_dims,
        gate_dim=8,
        ffn_mult=2,
        dtype_idx=64,
    )
    decoder = ARS9Decoder([layer])

    x = torch.randn(B, *spatial, c_model)
    q = torch.randn(B, K, q_dim)

    out = decoder(x, q)
    assert out["q"].shape == (B, K, q_dim)
    assert len(out["mask_logits_all"]) == 1
    assert out["mask_logits_all"][0].shape == (B, K, *spatial)


def test_ars9_block_with_hippo_n_init():
    """ARS9 supports init_mode='hippo_n'."""
    block = ARS9CondMixBlock(
        c_model=16,
        q_dim=8,
        spatial_dims=1,
        gate_dim=8,
        ffn_mult=2,
        dtype_idx=64,
        init_mode="hippo_n",
    )
    x = torch.randn(2, 6, 16)
    q = torch.randn(2, 3, 8)
    out = block(x, q)
    assert out["q"].shape == (2, 3, 8)
