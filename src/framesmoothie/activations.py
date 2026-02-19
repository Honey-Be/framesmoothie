import math
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    # Python 3.12+
    from typing import override, Tuple, Optional, Dict, Any, Callable
except Exception:  # pragma: no cover
    from typing_extensions import override, Tuple, Optional, Dict, Any, Callable

from s9.rs9_modules import RS9Layer
from s9.base import FPDTypeIdx, get_float_dtype

from framesmoothie.base import StabilizedActivationFunctionBase, StabilizedBiasedActivationFunctionBase, auxloss

class HybridVeLU(StabilizedActivationFunctionBase):
    def __init__(self, features: int, eps: float = 1e-6, dtype_idx: FPDTypeIdx = 64, *, alpha: float = 0.1, beta1: float = 0.1, beta2: float = 0.1, gamma: float = 0.1, momentum: float = 0.9, 
                 lambda_ot_global: float = 0.05, lambda_ot_channel: float = 0.05):
        super().__init__(features, eps, dtype_idx)
        self.alpha: float = alpha
        self.beta1: float = beta1
        self.beta2: float = beta2
        self.gamma: float = gamma
        self.momentum: float = momentum
        self.lambda_ot_global: float = lambda_ot_global
        self.lambda_ot_channel: float = lambda_ot_channel
        
        
        # 채널 수 C를 활용하여 각 채널별로 독립적인 활성화 스케일을 학습하도록 벡터 형태로 선언
        self.lambda_ = nn.Parameter(torch.ones(self.features, dtype=self.dtype))
        
        self._reg_loss = torch.tensor(0.0,dtype=self.dtype)

    @auxloss
    def reg_loss(self) -> torch.Tensor:
        return self._reg_loss
    
    @reg_loss.setter
    def reg_loss(self, loss: torch.Tensor):
        self._reg_loss = loss.to(dtype=self.dtype).reshape(())
    
    @reg_loss.collector
    def reg_loss(self, aggregate: Callable[[torch.Tensor], torch.Tensor] = torch.sum,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None
    ) -> torch.Tensor:
        losses = []
        for m in self.modules():
            if hasattr(m, "_reg_loss"):
                rl = getattr(m, "_reg_loss")
                if isinstance(rl, torch.Tensor):
                    losses.append(rl)
        if not losses:
            out = torch.tensor(0.0, dtype = self.dtype)
        else:
            out = aggregate(torch.stack([l.reshape(()) for l in losses]))
        
        if dtype is not None:
            out = out.to(dtype=dtype)
        if device is not None:
            out = out.to(device=device)

        return out.reshape(())

    @reg_loss.resetter
    @torch.no_grad()
    def reg_loss(self, value: float):
        for m in self.modules():
            if hasattr(m, "_reg_loss"):
                rl = getattr(m, "_reg_loss")
                if isinstance(rl, torch.Tensor):
                    setattr(m, "_reg_loss", rl.detach().new_tensor(value).reshape(()))

    def wasserstein_distance_hybrid(self, x):
        target_mean = 0.0
        target_std = 1.0

        # 1. 전역(Global) Wasserstein 거리 계산
        global_mean = torch.mean(x)
        global_std = torch.std(x, unbiased=False)
        w_dist_global = torch.square(global_mean - target_mean) + torch.square(global_std - target_std)

        # 2. 채널별(Channel-wise) Wasserstein 거리 계산
        dims_to_reduce = tuple(range(x.dim() - 1))
        
        channel_mean = torch.mean(x, dim=dims_to_reduce)
        channel_var = torch.var(x, dim=dims_to_reduce, unbiased=False)
        
        # 분산이 0일 때의 NaN 그래디언트를 방지하기 위해 epsilon 적용
        channel_std = torch.sqrt(channel_var + self.eps)
        
        w_dist_channel = torch.mean(
            torch.square(channel_mean - target_mean) + torch.square(channel_std - target_std)
        )

        total_w_dist = (self.lambda_ot_global * w_dist_global) + (self.lambda_ot_channel * w_dist_channel)
        
        return total_w_dist

    def forward(self, x):
        # 적응형 스케일링을 위한 분산 계산
        variance = torch.mean(torch.square(x), dim=-1, keepdim=True)
        
        # 0으로 나누는 오류 및 극단적 스케일링 폭발을 방지하기 위해 epsilon 적용
        std_dev = torch.rsqrt(variance + self.eps)

        adapt_factor = 1.0 + self.gamma * torch.tanh(std_dev * self.momentum)

        clipped = torch.clamp(self.beta2 * x, min=-1.0, max=1.0)
        mix = torch.atan(self.beta1 * x) + torch.asin(clipped)

        activation = x * torch.sigmoid(self.alpha * mix) * adapt_factor

        self.reg_loss = self.wasserstein_distance_hybrid(activation)

        # self.lambda_는 크기가 (C,)인 벡터이므로 (B, ..., C) 형태의 텐서에 자동으로 브로드캐스팅됨
        return self.lambda_ * activation


class BiasedTeLU(StabilizedBiasedActivationFunctionBase):
    """
    TeLU(x) = x * (tanh(exp(x + bias)) + eps)
    """
    @override
    def __init__(self, features: int, eps: float = 0.0, dtype_idx: FPDTypeIdx = 64):
        super().__init__(features, eps, dtype_idx)
    
    @override
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, D1, D2, ..., C) real

        view_shape = [1] * (x.ndim - 1) + [-1]
        b = self.bias.view(*view_shape)

        act = torch.tanh(torch.exp(x + b)) + self.eps

        return x * act

