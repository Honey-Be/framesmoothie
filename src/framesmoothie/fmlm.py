import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, List, Optional, Sequence, Tuple

from s9.base import FPDTypeIdx, get_float_dtype


def default_f(x: torch.Tensor) -> torch.Tensor:
    # range (-1,1) for stability assumptions in the notes
    return torch.tanh(x)


class FactorizedFMLM(nn.Module):
    """
    General-$f$ stable Rank-$R$ Factorized FMLM (Definition 4.1 in the notes).

    Given conditions u_i and a feature h, compute:
      g_{ir} = f(a_{ir}^T u_i + b_{ir})
      ell_{ir} = 1 + eta * g_{ir}
      s_r = Π_i ell_{ir}
      Δγ = Σ_i G_i u_i + P^(γ) s
      β  = Σ_i B_i u_i + P^(β) s
      γ  = 1 + α * f(Δγ)
      h' = γ ⊙ h + β

    Shapes:
      - h: [..., d]
      - u_i: broadcastable to h's leading dims, last dim = m_i
    """

    def __init__(
        self,
        *,
        d: int,
        cond_dims: Sequence[int],
        rank: int = 8,
        eta: float = 0.1,
        alpha: float = 0.1,
        f: Callable[[torch.Tensor], torch.Tensor] = default_f,
        dtype_idx: FPDTypeIdx = 64,
    ):
        super().__init__()
        self.dtype = get_float_dtype(dtype_idx)
        self.d = int(d)
        self.cond_dims = [int(x) for x in cond_dims]
        self.rank = int(rank)
        self.eta = float(eta)
        self.alpha = float(alpha)
        self.f = f

        # interaction parameters (A_i, b_i) per condition
        self.A = nn.ParameterList([
            nn.Parameter(torch.randn(self.rank, m, dtype=self.dtype) * 0.02)
            for m in self.cond_dims
        ])
        self.b = nn.ParameterList([
            nn.Parameter(torch.zeros(self.rank, dtype=self.dtype))
            for _ in self.cond_dims
        ])

        # linear heads for Δγ and β
        self.G = nn.ParameterList([
            nn.Parameter(torch.randn(self.d, m, dtype=self.dtype) * 0.02)
            for m in self.cond_dims
        ])
        self.B = nn.ParameterList([
            nn.Parameter(torch.randn(self.d, m, dtype=self.dtype) * 0.02)
            for m in self.cond_dims
        ])
        self.P_gamma = nn.Parameter(torch.randn(self.d, self.rank, dtype=self.dtype) * 0.02)
        self.P_beta  = nn.Parameter(torch.randn(self.d, self.rank, dtype=self.dtype) * 0.02)

    def _interaction_scores(self, conds: List[torch.Tensor]) -> torch.Tensor:
        # returns s: [..., R]
        s = None
        for u, A, b in zip(conds, self.A, self.b):
            u = u.to(dtype=self.dtype)
            # [..., R] = [..., m] @ [m,R]
            g = self.f(torch.matmul(u, A.t()) + b)  # range (-1,1)
            ell = 1.0 + self.eta * g               # >0 if eta in (0,1)
            s = ell if s is None else (s * ell)
        return s

    def forward(self, h: torch.Tensor, conds: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # returns (h', gamma, beta)
        h = h.to(dtype=self.dtype)
        s = self._interaction_scores(conds)  # [..., R]

        # Δγ, β
        delta_gamma = 0.0
        beta = 0.0
        for u, G, B in zip(conds, self.G, self.B):
            u = u.to(dtype=self.dtype)
            delta_gamma = delta_gamma + torch.matmul(u, G.t())
            beta = beta + torch.matmul(u, B.t())

        delta_gamma = delta_gamma + torch.matmul(s, self.P_gamma.t())
        beta = beta + torch.matmul(s, self.P_beta.t())

        gamma = 1.0 + self.alpha * self.f(delta_gamma)
        out = gamma * h + beta
        return out, gamma, beta


class FMLMGateGenerator(nn.Module):
    """
    Gate generator using the same rank-R interaction mechanism as FMLM.
    Produces a (0,1) gate via sigmoid over an FMLM-style 'beta' head.
    """
    def __init__(
        self,
        *,
        gate_dim: int,
        cond_dims: Sequence[int],
        rank: int = 8,
        eta: float = 0.1,
        f: Callable[[torch.Tensor], torch.Tensor] = default_f,
        dtype_idx: FPDTypeIdx = 64,
    ):
        super().__init__()
        self.dtype = get_float_dtype(dtype_idx)
        self.gate_dim = int(gate_dim)
        self.cond_dims = [int(x) for x in cond_dims]
        self.rank = int(rank)
        self.eta = float(eta)
        self.f = f

        self.A = nn.ParameterList([
            nn.Parameter(torch.randn(self.rank, m, dtype=self.dtype) * 0.02)
            for m in self.cond_dims
        ])
        self.b = nn.ParameterList([
            nn.Parameter(torch.zeros(self.rank, dtype=self.dtype))
            for _ in self.cond_dims
        ])
        self.W = nn.ParameterList([
            nn.Parameter(torch.randn(self.gate_dim, m, dtype=self.dtype) * 0.02)
            for m in self.cond_dims
        ])
        self.P = nn.Parameter(torch.randn(self.gate_dim, self.rank, dtype=self.dtype) * 0.02)
        self.bias = nn.Parameter(torch.zeros(self.gate_dim, dtype=self.dtype))

    def forward(self, conds: List[torch.Tensor]) -> torch.Tensor:
        # returns gate: [..., gate_dim] in (0,1)
        s = None
        for u, A, b in zip(conds, self.A, self.b):
            u = u.to(dtype=self.dtype)
            g = self.f(torch.matmul(u, A.t()) + b)
            ell = 1.0 + self.eta * g
            s = ell if s is None else (s * ell)

        out = self.bias
        for u, W in zip(conds, self.W):
            u = u.to(dtype=self.dtype)
            out = out + torch.matmul(u, W.t())
        out = out + torch.matmul(s, self.P.t())
        return torch.sigmoid(out)


class FMLMFiLM(nn.Module):
    """
    FiLM replacement using FactorizedFMLM.
    Produces (gamma,beta) per-slot to modulate value vectors.

    q:   [B,K,Dq]
    ctx: [B,Dc] (optional), will be broadcast to [B,1,Dc]
    v:   [B,N,D] (value to be modulated), broadcast to [B,1,N,D]
    """
    def __init__(
        self,
        *,
        q_dim: int,
        d: int,
        ctx_dim: Optional[int] = None,
        rank: int = 8,
        eta: float = 0.1,
        alpha: float = 0.1,
        f: Callable[[torch.Tensor], torch.Tensor] = default_f,
        dtype_idx: FPDTypeIdx = 64,
    ):
        super().__init__()
        self.dtype = get_float_dtype(dtype_idx)
        self.q_dim = int(q_dim)
        self.d = int(d)
        self.ctx_dim = int(ctx_dim) if ctx_dim is not None else None

        cond_dims = [self.q_dim] + ([self.ctx_dim] if self.ctx_dim is not None else [])
        self.fmlm = FactorizedFMLM(d=self.d, cond_dims=cond_dims, rank=rank, eta=eta, alpha=alpha, f=f, dtype_idx=dtype_idx)

    def forward(self, q: torch.Tensor, v_flat: torch.Tensor, ctx: Optional[torch.Tensor] = None) -> torch.Tensor:
        # q: [B,K,Dq], v_flat: [B,N,D]
        B, N, D = v_flat.shape
        K = q.shape[1]
        q = q.to(dtype=self.dtype)
        v_flat = v_flat.to(dtype=self.dtype)

        conds = [q]
        if self.ctx_dim is not None:
            if ctx is None:
                raise ValueError("ctx is required for this FMLMFiLM (ctx_dim was set)")
            ctx = ctx.to(dtype=self.dtype).unsqueeze(1).expand(B, K, -1)  # [B,K,Dc]
            conds.append(ctx)

        # modulate a per-slot 'base' h by setting h = 0 and using beta as additive,
        # but for FiLM we need gamma/beta applied to v (broadcast over N).
        # We'll compute gamma/beta from FMLM using a dummy h of zeros.
        h0 = torch.zeros(B, K, D, dtype=self.dtype, device=q.device)
        _, gamma, beta = self.fmlm(h0, conds)  # [B,K,D] each

        v = v_flat.unsqueeze(1)            # [B,1,N,D]
        gamma = gamma.unsqueeze(2)         # [B,K,1,D]
        beta = beta.unsqueeze(2)           # [B,K,1,D]
        return v * gamma + beta
