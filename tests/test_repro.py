import random

import numpy as np
import torch

from framesmoothie.repro import (
    DeterminismConfig,
    capture_rng_snapshot,
    configure_determinism,
    derive_seed,
    restore_rng_snapshot,
)


def test_derive_seed_is_stable_and_namespaced():
    seed_a = derive_seed(123, "loader", 0, 1, 2, 3)
    seed_b = derive_seed(123, "loader", 0, 1, 2, 3)
    seed_c = derive_seed(123, "model", 0, 1, 2, 3)
    assert seed_a == seed_b
    assert seed_a != seed_c



def test_rng_snapshot_roundtrip_restores_sequences():
    configure_determinism(DeterminismConfig(master_seed=7))
    snapshot = capture_rng_snapshot()

    seq1 = (
        random.random(),
        np.random.rand(),
        torch.rand(3),
    )

    restore_rng_snapshot(snapshot)
    seq2 = (
        random.random(),
        np.random.rand(),
        torch.rand(3),
    )

    assert seq1[0] == seq2[0]
    assert seq1[1] == seq2[1]
    assert torch.equal(seq1[2], seq2[2])
