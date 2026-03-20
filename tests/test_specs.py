import torch

from framesmoothie.specs import PREDEFINED_PRE_BRIDGE_MAPS, PreBridgeMapSpec, resolve_pre_bridge_map


def test_predefined_maps_return_expected_components():
    z = torch.randn(2, 3, 8, 8, dtype=torch.cfloat)
    for name, spec in PREDEFINED_PRE_BRIDGE_MAPS.items():
        out = spec(z)
        assert len(out) == spec.num_components
        for t in out:
            assert t.shape == z.shape
            assert not torch.is_complex(t)


def test_resolve_pre_bridge_map_accepts_name_and_spec():
    spec = resolve_pre_bridge_map('log1p_arsinh')
    assert isinstance(spec, PreBridgeMapSpec)
    assert resolve_pre_bridge_map(spec) is spec
    assert resolve_pre_bridge_map(None) is None
