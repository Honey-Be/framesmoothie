import math
from typing import Sequence, Tuple, Union

import torch

TensorLike = Union[torch.Tensor, Sequence[Sequence[float]]]

@torch.no_grad()
def linear_sum_assignment_torch(
    cost_matrix: TensorLike,
    maximize: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    PyTorch port of SciPy's linear_sum_assignment / rectangular_lsap logic.

    Parameters
    ----------
    cost_matrix:
        2-D cost matrix. May be a torch.Tensor or any array-like 2-D object.
    maximize:
        If True, compute a maximum-weight matching.

    Returns
    -------
    row_ind, col_ind:
        int64 torch tensors on the same device as the input tensor (if the input
        was a tensor). row_ind is sorted, matching SciPy's contract.

    Notes
    -----
    - This is a portability-first implementation, not a speed-first one.
    - Internal state is kept in Python lists/scalars to minimize backend-specific
      requirements.
    - Like SciPy:
        * only 2-D inputs are accepted
        * the working dtype is float64
        * NaN and -inf are rejected as invalid numeric entries
        * +inf is allowed, but may make the matrix infeasible
    """
    if not isinstance(cost_matrix, torch.Tensor):
        cost_matrix = torch.as_tensor(cost_matrix)

    if cost_matrix.ndim != 2:
        raise ValueError(f"expected 2-D cost matrix, got {cost_matrix.ndim}-D")

    if cost_matrix.dtype.is_complex:
        raise TypeError("complex cost matrices are not supported")

    device = cost_matrix.device
    nr0, nc0 = cost_matrix.shape
    dim = min(nr0, nc0)

    if dim == 0:
        empty = torch.empty((0,), dtype=torch.int64, device=device)
        return empty, empty

    cost = cost_matrix.to(dtype=torch.float64)

    transpose = nc0 < nr0
    if transpose:
        cost = cost.transpose(0, 1).contiguous()
        nr, nc = nc0, nr0
    else:
        cost = cost.contiguous()
        nr, nc = nr0, nc0

    if maximize:
        cost = -cost

    # NaN / -inf reject (SciPy-compatible)
    if torch.logical_or(torch.isnan(cost).any(), torch.isneginf(cost).any()).item():
        raise ValueError("matrix contains invalid numeric entries")

    # duals + matching
    u = torch.zeros(nr, dtype=torch.float64, device=device)
    v = torch.zeros(nc, dtype=torch.float64, device=device)
    col4row = torch.full((nr,), -1, dtype=torch.int64, device=device)
    row4col = torch.full((nc,), -1, dtype=torch.int64, device=device)

    # work buffers
    shortest = torch.empty(nc, dtype=torch.float64, device=device)
    path = torch.empty(nc, dtype=torch.int64, device=device)
    SR = torch.empty(nr, dtype=torch.bool, device=device)
    SC = torch.empty(nc, dtype=torch.bool, device=device)

    INF = float("inf")

    for cur_row in range(nr):
        shortest.fill_(INF)
        path.fill_(-1)
        SR.zero_()
        SC.zero_()

        min_val = cost.new_tensor(0.0)
        i = cur_row
        sink = -1

        while sink < 0:
            SR[i] = True

            # Vectorized reduced costs for row i over all columns
            r = min_val + cost[i] - u[i] - v  # [nc]

            improved = r < shortest
            if improved.any():
                shortest[improved] = r[improved]
                path[improved] = i

            # choose next column among unscanned
            spc = shortest.masked_fill(SC, INF)
            min_spc = spc.min()
            if torch.isinf(min_spc).item():
                raise ValueError("cost matrix is infeasible")

            cand = torch.eq(spc, min_spc)  # already excludes SC via INF
            # tie-break: prefer unmatched columns if possible
            unmatched = row4col.eq(-1)
            cand_u = torch.logical_and(cand, unmatched)

            if cand_u.any():
                j = cand_u.nonzero(as_tuple=False)[-1, 0].item()  # last index tie-break
            else:
                j = cand.nonzero(as_tuple=False)[-1, 0].item()

            min_val = min_spc
            SC[j] = True

            if row4col[j].item() == -1:
                sink = j
            else:
                i = row4col[j].item()

        # Dual updates (vectorized where possible)
        u[cur_row] = u[cur_row] + min_val

        sr_mask = SR.clone()
        sr_mask[cur_row] = False
        if sr_mask.any():
            cols = col4row[sr_mask]         # cols matched to visited rows
            valid = cols.ne(-1)
            if valid.any():
                rows = sr_mask.nonzero(as_tuple=False).squeeze(1)[valid]
                cols = cols[valid]
                u[rows] = u[rows] + (min_val - shortest[cols])

        if SC.any():
            v[SC] = v[SC] - (min_val - shortest[SC])

        # Augment along the found path
        j = sink
        while True:
            i = path[j].item()
            row4col[j] = i
            old_j = col4row[i].item()
            col4row[i] = j
            j = old_j
            if i == cur_row:
                break

    if transpose:
        # convert solution of transposed problem back to original indexing
        row_ind, col_ind = torch.sort(col4row)  # row_ind: original row, col_ind: original col
        return row_ind.to(dtype=torch.int64), col_ind.to(dtype=torch.int64)

    row_ind = torch.arange(nr, device=device, dtype=torch.int64)
    col_ind = col4row.to(dtype=torch.int64)
    return row_ind, col_ind
