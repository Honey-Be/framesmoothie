"""Smoke tests for Q-prefixed framesmoothie blocks/decoders (QRS9/QARS9)."""

import pytest
import torch

from s9.quantization import QuantConfig

from framesmoothie.blocks_qrs9 import QRS9CondMixBlock
from framesmoothie.decoder_qrs9 import QRS9DecoderLayer, QRS9Decoder
from framesmoothie.blocks_qars9 import QARS9CondMixBlock
from framesmoothie.decoder_qars9 import QARS9DecoderLayer, QARS9Decoder


def _quant_config():
    # Relax stability assertion for small random kernels in tests.
    return QuantConfig(enforce_stability=False)


@pytest.mark.parametrize("block_cls", [QRS9CondMixBlock, QARS9CondMixBlock])
def test_q_cond_mix_blocks_forward(block_cls):
    B, K = 2, 3
    c_model = 16
    q_dim = 8
    spatial = (6, 6)

    block = block_cls(
        c_model=c_model,
        q_dim=q_dim,
        spatial_dims=2,
        gate_dim=8,
        ffn_mult=2,
        return_masks=True,
        dtype_idx=64,
        quant_config=_quant_config(),
    )

    x = torch.randn(B, *spatial, c_model)
    q = torch.randn(B, K, q_dim)
    out = block(x, q)
    assert out["q"].shape == (B, K, q_dim)
    assert out["mask_logits"].shape == (B, K, *spatial)


@pytest.mark.parametrize("layer_cls,decoder_cls", [
    (QRS9DecoderLayer, QRS9Decoder),
    (QARS9DecoderLayer, QARS9Decoder),
])
def test_q_decoder_stack(layer_cls, decoder_cls):
    B, K = 2, 3
    c_model = 16
    q_dim = 8
    spatial = (6, 6)

    layer = layer_cls(
        c_model=c_model,
        q_dim=q_dim,
        spatial_dims=2,
        gate_dim=8,
        ffn_mult=2,
        dtype_idx=64,
        quant_config=_quant_config(),
    )
    decoder = decoder_cls([layer])

    x = torch.randn(B, *spatial, c_model)
    q = torch.randn(B, K, q_dim)

    out = decoder(x, q)
    assert out["q"].shape == (B, K, q_dim)
    assert len(out["mask_logits_all"]) == 1


def test_q_blocks_backward_pass():
    """Gradients flow through Q blocks (STE on fake quant)."""
    block = QARS9CondMixBlock(
        c_model=16, q_dim=8, spatial_dims=1,
        gate_dim=8, ffn_mult=2, dtype_idx=64,
        quant_config=_quant_config(),
    )
    x = torch.randn(2, 6, 16, requires_grad=True)
    q = torch.randn(2, 3, 8, requires_grad=True)
    out = block(x, q)
    out["q"].sum().backward()
    assert x.grad is not None
    assert q.grad is not None


def test_q_block_with_hippo_n_init():
    """QARS9 accepts init_mode='hippo_n' (inherited from ARS9)."""
    block = QARS9CondMixBlock(
        c_model=16, q_dim=8, spatial_dims=1,
        gate_dim=8, ffn_mult=2, dtype_idx=64,
        init_mode="hippo_n",
        quant_config=_quant_config(),
    )
    x = torch.randn(2, 6, 16)
    q = torch.randn(2, 3, 8)
    out = block(x, q)
    assert out["q"].shape == (2, 3, 8)
