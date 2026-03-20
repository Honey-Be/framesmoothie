import pytest
import torch

from framesmoothie.model import FrameSmoothiePanopticModel, make_ema_teacher
from framesmoothie.matcher import HungarianMatcher, SetCriterion
from framesmoothie.hmc import HMCCalibrator
from framesmoothie.train_step import FrameSmoothieTrainStep
from framesmoothie.predictor import ZoneEdgeSpec
from framesmoothie.diag_meter import DiagMeter
from tests.helpers import make_s9_transform_pair, make_small_s9_encoder, make_toy_panoptic_batch, infer_encoder_channels


@pytest.mark.slow

def test_single_step_smoke_train():
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
            ZoneEdgeSpec(srcs=('structure',), tgt='boundary', predictor_kind='dwsep_conv', loss_type='smooth_l1'),
            ZoneEdgeSpec(srcs=('content', 'structure'), tgt='label', predictor_kind='bilinear_mlp', projector_kind='bnfree_mlp', projector_dim=16, loss_type='simsiam'),
        ),
    )
    teacher = make_ema_teacher(model)
    matcher = HungarianMatcher(num_points=256)
    criterion = SetCriterion(num_classes=2, matcher=matcher, num_points=128)
    hmc = HMCCalibrator(thing_classes=(0, 1), min_area=8)
    step = FrameSmoothieTrainStep(
        criterion_inst=criterion,
        hmc=hmc,
        thing_classes=[0, 1],
        lambda_pred=0.1,
        pred_detach_target=True,
        use_edge_weights=True,
    )

    src = make_toy_panoptic_batch(batch_size=2, channels=8, height=32, width=32, style_shift=False)
    tgt = make_toy_panoptic_batch(batch_size=2, channels=8, height=32, width=32, style_shift=True)

    meter = DiagMeter()
    out = step(student=model, teacher=teacher, src=src, tgt=tgt, diag_meter=meter, return_diag=False)
    loss = out['loss']
    assert torch.isfinite(loss)
    loss.backward()

    # At least one gradient should flow
    grads = [p.grad for p in model.parameters() if p.requires_grad and p.grad is not None]
    assert len(grads) > 0
    assert any(torch.isfinite(g).all() for g in grads)

    means = meter.compute_means()
    assert isinstance(means, dict)
