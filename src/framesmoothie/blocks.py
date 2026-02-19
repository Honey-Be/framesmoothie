import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from s9.rs9_modules import RS9Layer
from s9.base import FPDTypeIdx, get_float_dtype

try:
    # Python 3.12+
    from typing import override, Tuple, Optional, Dict, Any, Callable
except Exception:  # pragma: no cover
    from typing_extensions import override, Tuple, Optional, Dict, Any, Callable


from framesmoothie.base import StabilizedActivationFunctionBase, auxloss
from framesmoothie.activations import BiasedTeLU


def _to_channel_first(x: torch.Tensor) -> torch.Tensor:
    # x: [B, *S, C] -> [B, C, *S]
    spatial_dims = x.ndim - 2
    perm = [0, spatial_dims + 1] + list(range(1, 1 + spatial_dims))
    return x.permute(*perm)

def _to_channel_last(x: torch.Tensor) -> torch.Tensor:
    # x: [B, C, *S] -> [B, *S, C]
    spatial_dims = x.ndim - 2
    perm = [0] + list(range(2, 2 + spatial_dims)) + [1]
    return x.permute(*perm)

def _flatten_spatial(x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, ...]]:
    """
    x: [B, *S, C]  (channel-last)
    return:
      xf: [B, N, C], spatial: (*S)
    """
    if x.ndim < 3:
        raise ValueError(f"Expected [B,*S,C], got shape={tuple(x.shape)}")
    B = x.shape[0]
    C = x.shape[-1]
    spatial = tuple(x.shape[1:-1])
    N = 1
    for s in spatial:
        N *= s
    xf = x.reshape(B, N, C)
    return xf, spatial


def _unflatten_spatial(xf: torch.Tensor, spatial: Tuple[int, ...]) -> torch.Tensor:
    """
    xf: [B, N, C] -> [B, *S, C]
    """
    B, N, C = xf.shape
    return xf.reshape(B, *spatial, C)


class BilinearGate(nn.Module):
    """
    Gate logits via elementwise product in a shared gate-dim space:
      g_logits[b,k,n] = < Wx(x)[b,n,:], Wq(q)[b,k,:] > / sqrt(dg)
    """
    def __init__(self, c_in: int, q_dim: int, gate_dim: int, dtype: torch.dtype):
        super().__init__()
        self.wx = nn.Linear(c_in, gate_dim, bias=False, dtype=dtype)
        self.wq = nn.Linear(q_dim, gate_dim, bias=False, dtype=dtype)
        self.bias = nn.Parameter(torch.zeros(1, dtype=dtype))
        self.scale = 1.0 / math.sqrt(gate_dim)

    def forward(self, x_flat: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        """
        x_flat: [B, N, C]
        q:      [B, K, Dq]
        returns g_logits: [B, K, N]
        """
        xg = self.wx(x_flat)          # [B, N, G]
        qg = self.wq(q)               # [B, K, G]
        # Broadcast multiply then sum over G:
        # (B,K,N,G) = (B,1,N,G) * (B,K,1,G)
        g_logits = (xg.unsqueeze(1) * qg.unsqueeze(2)).sum(dim=-1) * self.scale
        g_logits = g_logits + self.bias
        return g_logits


class Film(nn.Module):
    """
    Query-conditioned FiLM: for a value vector v in R^D:
      v' = gamma(q) * v + beta(q)
    """
    def __init__(self, q_dim: int, d: int, dtype: torch.dtype):
        super().__init__()
        self.proj = nn.Linear(q_dim, 2 * d, bias=True, dtype=dtype)

    def forward(self, q: torch.Tensor, v_flat: torch.Tensor) -> torch.Tensor:
        """
        q:      [B, K, Dq]
        v_flat: [B, N, D]
        return: v_film: [B, K, N, D]
        """
        B, N, D = v_flat.shape
        K = q.shape[1]
        gb = self.proj(q)             # [B, K, 2D]
        gamma, beta = gb.chunk(2, dim=-1)  # [B,K,D], [B,K,D]
        # broadcast over N
        v = v_flat.unsqueeze(1)       # [B,1,N,D]
        gamma = gamma.unsqueeze(2)    # [B,K,1,D]
        beta  = beta.unsqueeze(2)     # [B,K,1,D]
        return v * gamma + beta


class RS9CondMixBlock(nn.Module):
    """
    Attention-free, RS9-based conditioned mixing block.

    - Runs RS9 once on X to get global-mixed H.
    - Builds per-slot gates g(b,k,n) from X and Q.
    - Pools slot readouts from values V (X/H + FiLM), weighted by g.
    - Updates Q via LN + FFN.

    Returns:
      Q_out: [B, K, Dq]
      mask_logits(optional): [B, K, *S]
    """
    def __init__(
        self,
        c_model: int,
        q_dim: int,
        spatial_dims: int,
        gate_dim: int = 64,
        v_dim: Optional[int] = None,  # internal value dim; default=q_dim
        ffn_mult: int = 4,
        dropout: float = 0.0,
        eps: float = 1e-6,
        rs9_eps: float = 1e-6,
        return_masks: bool = True,
        rs9: Optional[RS9Layer] = None,
        gen_activation: Callable[[int, float, FPDTypeIdx], StabilizedActivationFunctionBase] = BiasedTeLU,
        dtype_idx: FPDTypeIdx = 64,
        lambda_gate_entropy: float = 0.0,
        lambda_gate_competition: float = 0.0,
    ):
        super().__init__()
        self.rs9: RS9Layer = rs9 if rs9 is not None else RS9Layer(
            d_model=c_model,
            spatial_dims=spatial_dims,
            # RS9는 real activation을 받는 구조라면 그에 맞게
            gen_activation=gen_activation,  # 예시 (네 RS9 시그니처에 맞춰 조정)
            eps=rs9_eps,
            dtype_idx=dtype_idx,
        )
        self.c_model: int = c_model
        self.q_dim: int = q_dim
        self.gate_dim: int = gate_dim
        self.v_dim: int = v_dim if v_dim is not None else q_dim
        self.eps: float = eps
        self.return_masks: bool = return_masks
        self.dtype: torch.dtype = get_float_dtype(dtype_idx)

        # Gate from X and Q
        self.gate = BilinearGate(c_in=c_model, q_dim=q_dim, gate_dim=gate_dim, dtype=self.dtype)

        # Values: project X and H into v_dim, apply FiLM(H) conditioned on Q, then sum with X
        self.vx = nn.Linear(c_model, self.v_dim, bias=False, dtype=self.dtype)
        self.vh = nn.Linear(c_model, self.v_dim, bias=False, dtype=self.dtype)
        self.film = Film(q_dim=q_dim, d=self.v_dim, dtype=self.dtype)

        # Query update: LN + FFN
        self.ln = nn.LayerNorm(self.v_dim, dtype=self.dtype)
        self.ffn = nn.Sequential(
            nn.Linear(self.v_dim, ffn_mult * self.v_dim, dtype=self.dtype),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_mult * self.v_dim, q_dim, dtype=self.dtype),
            nn.Dropout(dropout),
        )

        self._reg_loss: torch.Tensor = torch.tensor(0.0, dtype = self.dtype)
        self.lambda_gate_entropy: float = lambda_gate_entropy
        self.lambda_gate_competition: float = lambda_gate_competition

    def forward(self, x: torch.Tensor, q: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        x: [B, *S, C]
        q: [B, K, Dq]
        """
        # 1) RS9 global mixing (RS9 expects channel-first: [B,C,*S])
        if (x.ndim - 2) != self.rs9.spatial_dims:
            raise ValueError(f"spatial_dims mismatch: x has {x.ndim-2}, rs9 has {self.rs9.spatial_dims}")
        x_cf = _to_channel_first(x)     # [B,C,*S]
        h_cf = self.rs9(x_cf)           # [B,C,*S]
        h = _to_channel_last(h_cf)      # [B,*S,C]

        # 2) flatten spatial for efficient gating/pooling
        x_flat, spatial = _flatten_spatial(x)   # [B,N,C]
        h_flat, _ = _flatten_spatial(h)         # [B,N,C]
        B, N, C = x_flat.shape
        K = q.shape[1]

        # 3) gate logits and gate weights
        g_logits = self.gate(x_flat, q)         # [B,K,N]
        g = torch.sigmoid(g_logits)             # [B,K,N]

        reg = torch.tensor(0.0, dtype=self.dtype, device=g.device)

        if self.lambda_gate_entropy != 0.0:
            # entropy of Bernoulli gate: -p log p -(1-p) log(1-p)
            p = g.clamp(min=self.eps, max=1.0 - self.eps)
            ent = -(p * torch.log(p) + (1.0 - p) * torch.log(1.0 - p))  # [B,K,N]
            reg = reg + (self.lambda_gate_entropy * ent.mean())

        if self.lambda_gate_competition != 0.0:
            # competition: penalize sum_k g_{k,n} exceeding 1
            # shape: [B,N]
            overlap = g.sum(dim=1)  # sum over K
            comp = F.relu(overlap - 1.0)
            reg = reg + (self.lambda_gate_competition * comp.mean())

        self.reg_loss = reg

        # 4) values
        vx = self.vx(x_flat)                    # [B,N,V]
        vh = self.vh(h_flat)                    # [B,N,V]
        vh_film = self.film(q, vh)              # [B,K,N,V]
        # Note: avoid materializing Y=[B,K,N,C]; use values V=[B,K,N,V]
        v = vx.unsqueeze(1) + vh_film           # [B,K,N,V]

        # 5) gate-weighted mean pool over N
        # read[b,k,:] = sum_n g[b,k,n]*v[b,k,n,:] / (sum_n g[b,k,n] + eps)
        w = g.unsqueeze(-1)                     # [B,K,N,1]
        num = (w * v).sum(dim=2)                # [B,K,V]
        den = w.sum(dim=2).clamp_min(self.eps)  # [B,K,1]
        read = num / den                        # [B,K,V]

        # 6) query update
        read = self.ln(read)
        dq = self.ffn(read)                     # [B,K,Dq]
        q_out = q + dq

        out: Dict[str, torch.Tensor] = {"q": q_out}

        if self.return_masks:
            # reshape g_logits to [B,K,*S] as mask logits
            mask_logits = g_logits.reshape(B, K, *spatial)
            out["mask_logits"] = mask_logits

        return out

    @auxloss
    def reg_loss(self) -> torch.Tensor:
        return self._reg_loss

    @reg_loss.setter
    def reg_loss(self, loss: torch.Tensor):
        self._reg_loss = loss.to(dtype=self.dtype).reshape(())

    @reg_loss.collector
    def reg_loss(
        self,
        aggregate: Callable[[torch.Tensor], torch.Tensor] = torch.sum,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        losses = []
        for m in self.modules():
            if hasattr(m, "_reg_loss"):
                rl = getattr(m, "_reg_loss")
                if isinstance(rl, torch.Tensor):
                    losses.append(rl.reshape(()))
        out = torch.tensor(0.0, dtype=self.dtype) if not losses else aggregate(torch.stack(losses))
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