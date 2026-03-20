import pytest
import torch
from torch.autograd import gradcheck

from framesmoothie.fmlm import FactorizedFMLM, FMLMGateGenerator


@pytest.mark.slow

def test_fmlm_gate_generator_gradcheck():
    mod = FMLMGateGenerator(gate_dim=3, cond_dims=[2], rank=2, dtype_idx=128)
    u = torch.randn(4, 2, dtype=torch.double, requires_grad=True)

    def func(inp):
        return mod([inp])

    assert gradcheck(func, (u,), eps=1e-6, atol=1e-4, rtol=1e-3)


@pytest.mark.slow

def test_factorized_fmlm_gradcheck():
    mod = FactorizedFMLM(d=3, cond_dims=[2], rank=2, dtype_idx=128)
    h = torch.randn(4, 3, dtype=torch.double, requires_grad=True)
    u = torch.randn(4, 2, dtype=torch.double, requires_grad=True)

    def func(h_inp, u_inp):
        out, _, _ = mod(h_inp, [u_inp])
        return out

    assert gradcheck(func, (h, u), eps=1e-6, atol=1e-4, rtol=1e-3)
