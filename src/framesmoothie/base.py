import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import override, Tuple, Optional, Dict, Any, Callable, TypeVar, Generic, Self

from s9.rs9_modules import RS9Layer
from s9.base import FPDTypeIdx, get_float_dtype

from abc import ABC, abstractmethod

class StabilizedActivationFunctionBase(nn.Module, ABC):
    @abstractmethod
    def __init__(self, features: int, eps: float = 1e-6, dtype_idx: FPDTypeIdx = 64):
        super().__init__()

        self.dtype: torch.dtype = get_float_dtype(dtype_idx)
        self.features: int = features
        self.eps: float = eps
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

class StabilizedBiasedActivationFunctionBase(StabilizedActivationFunctionBase, ABC):
    @abstractmethod
    @override
    def __init__(self, features: int, eps: float = 1e-6, dtype_idx: FPDTypeIdx = 64):
        super().__init__(features, eps, dtype_idx)

        self.bias: nn.Parameter = nn.Parameter(torch.zeros(self.features, dtype=self.dtype))
        self.eps: float = eps
