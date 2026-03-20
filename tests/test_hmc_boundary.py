import torch

from framesmoothie.hmc import HMCCalibrator


def test_hmc_forward_with_boundary_features():
    B, K, Csem, H, W = 2, 4, 4, 32, 32
    sem_logits = torch.randn(B, Csem, H, W)
    inst_class_logits = torch.randn(B, K, 3)  # 2 thing + no-object
    inst_mask_logits = torch.randn(B, K, H, W)
    boundary_features = torch.randn(B, H, W, 8)

    hmc = HMCCalibrator(thing_classes=(0, 1), min_area=8)
    out = hmc(
        sem_logits=sem_logits,
        inst_class_logits=inst_class_logits,
        inst_mask_logits=inst_mask_logits,
        boundary_features=boundary_features,
    )
    assert 'pseudo_semantic' in out
    assert 'pseudo_instances' in out
    assert len(out['pseudo_instances']['labels']) == B
