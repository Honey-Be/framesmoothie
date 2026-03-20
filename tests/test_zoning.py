import pytest
import torch

from framesmoothie.specs import resolve_pre_bridge_map
from framesmoothie.zoning import PreBridgeProjector, DualViewZoningBlock


def test_pre_bridge_projector_shapes():
    spec = resolve_pre_bridge_map('log1p_arsinh')
    proj = PreBridgeProjector(map_spec=spec, in_channels=4, out_channels=8, spatial_dims=2)
    z = torch.randn(2, 4, 16, 16, dtype=torch.cfloat)
    out = proj(z)
    assert out.shape == (2, 8, 16, 16)
    assert not torch.is_complex(out)


def test_dual_view_zoning_shapes():
    block = DualViewZoningBlock(num_scales=2, c_pre=8, c_post=8, c_out=8)
    pre = [torch.randn(2, 16, 16, 8), torch.randn(2, 8, 8, 8)]
    post = [torch.randn(2, 16, 16, 8), torch.randn(2, 8, 8, 8)]
    out = block(pre, post)
    assert set(out.keys()) == {'zones', 'sem_pyr', 'inst_pyr'}
    assert len(out['zones']) == 2
    assert out['sem_pyr'][0].shape == (2, 16, 16, 8)
    assert out['inst_pyr'][1].shape == (2, 8, 8, 8)
    assert set(out['zones'][0].keys()) == {'content', 'structure', 'label', 'boundary'}
