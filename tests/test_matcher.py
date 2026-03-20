import torch

from framesmoothie.matcher import HungarianMatcher


def test_matcher_pairwise_mask_cost_no_bce_shape_error():
    matcher = HungarianMatcher(num_points=64)

    # B=1, K=4 queries, C+1=3 classes (2 fg + no-object), HxW=16x16
    class_logits = torch.randn(1, 4, 3)
    mask_logits = torch.randn(1, 4, 16, 16)

    targets = [{
        "labels": torch.tensor([0, 1], dtype=torch.long),
        "masks": torch.randint(0, 2, (2, 16, 16), dtype=torch.float32),
    }]

    indices = matcher(class_logits, mask_logits, targets)
    assert isinstance(indices, list)
    assert len(indices) == 1
    row_ind, col_ind = indices[0]
    assert row_ind.dtype == torch.long
    assert col_ind.dtype == torch.long
    assert row_ind.numel() == col_ind.numel() == 2
