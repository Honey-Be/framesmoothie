import pytest
import torch

from framesmoothie.model import FrameSmoothiePanopticModel
from framesmoothie.predictor import ZoneEdgeSpec
from tests.helpers import make_s9_transform_pair, make_small_s9_encoder, infer_encoder_channels


def test_model_forward_smoke():
    t_fwd, t_inv = make_s9_transform_pair(2)
    enc_c_model = infer_encoder_channels(transform_fwd=t_fwd, input_channels=8, spatial_shape=(32, 32))
    model = FrameSmoothiePanopticModel(
        transform_fwd=t_fwd,
        transform_inv=t_inv,
        encoder=make_small_s9_encoder(c_model=enc_c_model, spatial_dims=2, depth=1),
        c_model=8,
        enc_c_model=enc_c_model,
        spatial_dims=2,
        num_semantic_classes=4,
        num_instance_classes=2,
        q_dim=8,
        num_queries=4,
        decoder_layers=1,
        pre_bridge_map='log1p_arsinh',
        pre_bridge_channels=8,
        use_zoning=True,
        use_zone_prediction=True,
        zone_pred_edges=(
            ZoneEdgeSpec(srcs=('structure',), tgt='boundary', predictor_kind='linear', loss_type='smooth_l1'),
        ),
    )
    x = torch.randn(2, 8, 32, 32)
    out = model(x, domain_id=torch.zeros((2, 1)), return_diag=True)
    assert 'sem_logits' in out and 'inst' in out and 'zones' in out
    assert out['sem_logits'].shape[:2] == (2, 4)
    assert out['inst']['class_logits'].shape[:2] == (2, 4)
    # assert 'diag' in out
