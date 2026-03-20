import math

import torch

from framesmoothie.zone_losses import compute_edge_loss, compute_zone_predictive_loss


ALL_LOSSES = [
    'smooth_l1', 'mse', 'l1', 'huber', 'cosine', 'neg_cosine',
    'simsiam', 'symmetric_simsiam', 'sym_simsiam', 'kl_div', 'barlow', 'vicreg'
]


def test_compute_edge_loss_supported_variants():
    pred = torch.randn(2, 8, 16)
    tgt = torch.randn(2, 8, 16)
    for lt in ALL_LOSSES:
        kwargs = {}
        if lt == 'huber':
            kwargs = {'delta': 0.5}
        elif lt == 'barlow':
            kwargs = {'lambda_offdiag': 1e-2}
        elif lt == 'vicreg':
            kwargs = {'sim_coeff': 10.0, 'std_coeff': 10.0, 'cov_coeff': 1.0, 'gamma': 1.0}
        loss = compute_edge_loss(pred, tgt, loss_type=lt, loss_kwargs=kwargs)
        assert loss.ndim == 0
        assert torch.isfinite(loss)


def _fake_zone_pred(loss_type: str):
    zones = [
        {
            'boundary': torch.randn(2, 4, 8),
            'label': torch.randn(2, 4, 8),
            'structure': torch.randn(2, 4, 8),
            'content': torch.randn(2, 4, 8),
        }
    ]
    zone_pred = {
        'preds': [
            {
                'e0': torch.randn(2, 4, 8),
            }
        ],
        'proj_preds': [
            {
                'e0': torch.randn(2, 4, 8),
            }
        ],
        'proj_tgts': [
            {
                'e0': torch.randn(2, 4, 8),
            }
        ],
        'edge_meta': {
            'e0': {
                'tgt': 'boundary',
                'weight': 1.0,
                'loss_type': loss_type,
                'loss_kwargs': {'lambda_offdiag': 1e-2} if loss_type == 'barlow' else {},
                'projector_reg_type': 'var_cov',
                'projector_reg_weight': 0.01,
                'projector_post_norm': 'l2',
                'projector_temperature': 0.5,
            }
        }
    }
    return zone_pred, zones


def test_projector_regularization_is_generalized():
    for lt in ['smooth_l1', 'mse', 'neg_cosine', 'simsiam', 'barlow', 'vicreg']:
        zone_pred, zones = _fake_zone_pred(lt)
        total, per_edge = compute_zone_predictive_loss(zone_pred=zone_pred, zones=zones)
        assert total.ndim == 0
        assert torch.isfinite(total)
        assert 'e0' in per_edge
        assert 'e0/reg' in per_edge
