import os
import sys
import random
from pathlib import Path

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'src'
for candidate in (ROOT, SRC):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))


@pytest.fixture(autouse=True)
def _set_seed():
    seed = 0
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    yield


@pytest.fixture(scope='session')
def cpu_device():
    return torch.device('cpu')


@pytest.fixture(scope='session')
def cuda_device():
    if torch.cuda.is_available():
        return torch.device('cuda')
    return None


def pytest_configure(config):
    config.addinivalue_line('markers', 'slow: marks tests as slow')
    config.addinivalue_line('markers', 'gpu: requires CUDA')
    config.addinivalue_line('markers', 'requires_s9: requires installed s9 package')
