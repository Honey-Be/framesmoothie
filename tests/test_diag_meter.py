import torch

from framesmoothie.diag_meter import DiagMeter


def test_diag_meter_update_and_compute():
    meter = DiagMeter(device=torch.device('cpu'))
    meter.update({'a': torch.tensor(1.0), 'b': 2.0})
    meter.update({'a': torch.tensor(3.0)})
    out = meter.compute()
    assert set(out.keys()) == {'a', 'b'}
    assert abs(float(out['a']['mean']) - 2.0) < 1e-6
    assert abs(float(out['b']['mean']) - 2.0) < 1e-6


def test_diag_meter_merge_and_state_dict_roundtrip():
    m1 = DiagMeter()
    m2 = DiagMeter()
    m1.update({'x': 1.0})
    m2.update({'x': 3.0, 'y': 5.0})
    m1.merge(m2)
    means = m1.compute_means()
    assert abs(means['x'] - 2.0) < 1e-6
    assert abs(means['y'] - 5.0) < 1e-6

    state = m1.state_dict()
    m3 = DiagMeter()
    m3.load_state_dict(state)
    means3 = m3.compute_means()
    assert means3 == means


def test_diag_meter_filtered_views():
    meter = DiagMeter()
    meter.update({'tgt_teacher/corr': 0.5, 'tgt_student/corr': 0.25, 'other': 1.0})
    means = meter.compute_means_filtered(prefix='tgt_teacher/')
    assert set(means.keys()) == {'tgt_teacher/corr'}
