"""H9 encoder wrapper for FrameSmoothiePanopticModel.

Replaces the ``transform_fwd + encoder + transform_inv`` pipeline with
H9's unified spectral-adaptive architecture:

    real input → stem(skip) → WarpedDOST → HSSBlocks → IDOST → real output

Usage with ``FrameSmoothiePanopticModel``::

    from framesmoothie.encoder_h9 import H9Encoder

    h9_enc = H9Encoder(d_model=32, n_layers=4, n_per_axis=2)
    model = FrameSmoothiePanopticModel(
        transform_fwd=nn.Identity(),
        transform_inv=nn.Identity(),
        encoder=h9_enc,
        c_model=32,
        enc_c_model=32,  # same as d_model — H9Encoder is real-in/real-out
        ...
    )
    # calibrate before first forward
    h9_enc.calibrate(sample_batch)

Requires ``s9[h9]`` (v0.6.0+).
"""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor, nn

try:
    # Wheel-installed: s9 v0.6.0 maps src/h9 → s9/h9
    from s9.h9.hss_block import HSSBlock
except ModuleNotFoundError:
    # Editable install: src/h9 is exposed as the top-level `h9` package
    from h9.hss_block import HSSBlock
from ypsilon_torch import FPDTypeIdx
from ypsilon_torch.blocks.transforms.real_complex.warped_dost import (
    InverseWarpedDOST,
    WarpedDOST,
)
from ypsilon_torch.blocks.activations.complex import StableModReLU


class _H9EncoderFitter:
    """Streaming calibration fitter for H9Encoder."""

    def __init__(self, encoder: H9Encoder) -> None:
        self._enc = encoder
        self._dost_fitter = encoder.dost.fitter

    def accumulate(self, x: Tensor) -> None:
        """Accumulate calibration statistics.

        Parameters
        ----------
        x : Tensor
            Real input ``(B, d_model, *S)``. Must already be at ``d_model``
            channels (post-stem in FrameSmoothiePanopticModel).
        """
        with torch.no_grad():
            self._dost_fitter.accumulate(x)

    def finalize(self) -> None:
        self._dost_fitter.finalize()


class H9Encoder(nn.Module):
    """Real-in / real-out H9 encoder for FrameSmoothiePanopticModel.

    Internally performs WarpedDOST → HSSBlock stack → InverseWarpedDOST.
    When used with ``FrameSmoothiePanopticModel``, set
    ``transform_fwd=nn.Identity()`` and ``transform_inv=nn.Identity()``
    so that the model's own transform pipeline is bypassed.

    Parameters
    ----------
    d_model : int
        Channel dimension of the real input (= ``c_model`` in the panoptic
        model). HSS blocks operate on ``d_prime = d_model * n_per_axis^D``.
    n_layers : int
        Number of HSSBlock layers.
    n_per_axis : int
        Warped DOST band count per spatial axis. Default 2.
    spatial_dims : int
        Spatial dimensionality. Default 2.
    d_ff_mult : int
        FFN expansion multiplier in each HSSBlock. Default 4.
    init_mode : str
        Initialization scheme. Default ``"gaussian"``.
    dropout : float
        Dropout in FFN. Default 0.0.
    eps : float
        Numerical epsilon. Default 1e-8.
    dtype_idx : int
        Precision selector (32 or 64). Default 64.
    """

    def __init__(
        self,
        d_model: int,
        n_layers: int,
        n_per_axis: int = 4,
        spatial_dims: int = 2,
        gen_activation: Callable[[int, float, FPDTypeIdx], ComplexActivationFunctionBase] = StableModReLU,
        gen_gate_activation: Callable[[], nn.Module] = nn.Sigmoid,
        d_ff_mult: int = 4,
        init_mode: Literal["gaussian"] = "gaussian",
        dropout: float = 0.0,
        eps: float = 1e-8,
        dtype_idx: FPDTypeIdx = 64,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_per_axis = n_per_axis
        self.spatial_dims = spatial_dims

        self.dost = WarpedDOST(D=spatial_dims, n_per_axis=n_per_axis)

        self.hss_blocks = nn.ModuleList([
            HSSBlock(
                d_model=d_model,
                n_per_axis=n_per_axis,
                spatial_dims=spatial_dims,
                gen_activation=gen_activation,
                gen_gate_activation=gen_gate_activation,
                d_ff_mult=d_ff_mult,
                init_mode=init_mode,
                dropout=dropout,
                eps=eps,
                dtype_idx=dtype_idx,
            )
            for _ in range(n_layers)
        ])

    def calibrate(self, x: Tensor) -> None:
        """One-shot DOST calibration.

        Parameters
        ----------
        x : Tensor
            Real input ``(B, d_model, *S)``.
        """
        with torch.no_grad():
            self.dost.fit(x)

    @property
    def fitter(self) -> _H9EncoderFitter:
        """Streaming calibration fitter."""
        return _H9EncoderFitter(self)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass: real → WarpedDOST → HSSBlocks → IDOST → real.

        Parameters
        ----------
        x : Tensor
            Real input ``(B, d_model, *S)``.

        Returns
        -------
        Tensor
            Real output ``(B, d_model, *S)``, same shape as input.
        """
        z = self.dost(x)                                   # complex
        for block in self.hss_blocks:
            z = block(z)
        inv: InverseWarpedDOST = self.dost.get_inverse_transform()
        y = inv(z)                                          # real
        return y
