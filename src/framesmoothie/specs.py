from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch


PreBridgeMapFn = Callable[[torch.Tensor], tuple[torch.Tensor, ...]]


@dataclass(frozen=True, slots=True)
class PreBridgeMapSpec:
    """Specification of a complex-to-real pre-bridge mapping φ: C -> R^m.

    The mapping is applied elementwise to a complex tensor and must return a fixed number
    of real-valued components, each with the same shape as the input tensor.
    """

    name: str
    fn: PreBridgeMapFn
    num_components: int

    def __call__(self, z: torch.Tensor) -> tuple[torch.Tensor, ...]:
        if not torch.is_complex(z):
            raise TypeError("PreBridgeMapSpec expects a complex tensor input.")

        out = self.fn(z)

        if len(out) != self.num_components:
            raise ValueError(
                f"{self.name}: expected {self.num_components} components, got {len(out)}"
            )

        for i, t in enumerate(out):
            if t.shape != z.shape:
                raise ValueError(
                    f"{self.name}: component {i} has shape {tuple(t.shape)}, expected {tuple(z.shape)}"
                )
            if torch.is_complex(t):
                raise TypeError(f"{self.name}: component {i} must be real-valued.")

        return out

    @staticmethod
    def log1p_arsinh(z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.log1p(torch.abs(z)),
            torch.asinh(torch.real(z)),
            torch.asinh(torch.imag(z)),
        )

    @staticmethod
    def arsinh_arsinh(z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.asinh(torch.abs(z)),
            torch.asinh(torch.real(z)),
            torch.asinh(torch.imag(z)),
        )

    @staticmethod
    def abs_phase(z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        theta = torch.angle(z)
        return (
            torch.abs(z),
            torch.cos(theta),
            torch.sin(theta),
        )

    @staticmethod
    def re_im_abs(z: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return (
            torch.real(z),
            torch.imag(z),
            torch.abs(z),
        )


PREDEFINED_PRE_BRIDGE_MAPS: dict[str, PreBridgeMapSpec] = {
    "log1p_arsinh": PreBridgeMapSpec("log1p_arsinh", PreBridgeMapSpec.log1p_arsinh, 3),
    "arsinh_arsinh": PreBridgeMapSpec("arsinh_arsinh", PreBridgeMapSpec.arsinh_arsinh, 3),
    "abs_phase": PreBridgeMapSpec("abs_phase", PreBridgeMapSpec.abs_phase, 3),
    "re_im_abs": PreBridgeMapSpec("re_im_abs", PreBridgeMapSpec.re_im_abs, 3),
}


def resolve_pre_bridge_map(spec: str | PreBridgeMapSpec | None) -> PreBridgeMapSpec | None:
    if spec is None:
        return None
    if isinstance(spec, PreBridgeMapSpec):
        return spec
    try:
        return PREDEFINED_PRE_BRIDGE_MAPS[spec]
    except KeyError as e:
        raise KeyError(
            f"Unknown pre-bridge map '{spec}'. Available: {sorted(PREDEFINED_PRE_BRIDGE_MAPS.keys())}"
        ) from e
