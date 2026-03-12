import torch
from typing import Any, Dict, Optional

try:
    import torch.distributed as dist
except Exception:  # pragma: no cover
    dist = None  # type: ignore

class _ScalarStats:
    __slots__ = ("sum", "sum_sq", "count")

    def __init__(self, device: torch.device):
        self.sum = torch.zeros((), dtype=torch.float64, device=device)
        self.sum_sq = torch.zeros((), dtype=torch.float64, device=device)
        self.count = torch.zeros((), dtype=torch.float64, device=device)

    def update(self, x: torch.Tensor, weight: float = 1.0):
        x = x.detach().reshape(()).to(dtype=torch.float64, device=self.sum.device)
        w = torch.tensor(weight, dtype=torch.float64, device=self.sum.device)
        self.sum += w * x
        self.sum_sq += w * (x * x)
        self.count += w

    def mean(self) -> torch.Tensor:
        return self.sum / self.count.clamp_min(1.0)

    def var(self) -> torch.Tensor:
        m = self.mean()
        return (self.sum_sq / self.count.clamp_min(1.0)) - (m * m)

    def std(self) -> torch.Tensor:
        return torch.sqrt(torch.clamp(self.var(), min=0.0))


class DiagMeter:
    """
    TorchMetrics-style scalar accumulator for diagnostics dictionaries.

    - update(diag_dict, prefix=..., weight=...)
    - compute() -> {key: {"mean":..., "std":..., "count":...}}
    - compute_means() -> {key: mean}
    - reset()

    Notes:
      - Only scalar tensors / python numbers are accepted as values.
      - Values are detached; internal accumulation device is configurable via `device=`.
    """

    def __init__(self, *, device: torch.device = torch.device("cpu"), allow_new_keys: bool = True):
        self.device = device
        self.allow_new_keys = bool(allow_new_keys)
        self._stats: Dict[str, _ScalarStats] = {}

    def to(self, device: torch.device) -> "DiagMeter":
        """Move accumulator state to a new device (torchmetrics-style)."""
        if torch.device(device) == torch.device(self.device):
            return self
        self.device = torch.device(device)
        for st in self._stats.values():
            st.sum = st.sum.to(device=self.device)
            st.sum_sq = st.sum_sq.to(device=self.device)
            st.count = st.count.to(device=self.device)
        return self

    def reset(self):
        self._stats.clear()

    def merge(self, other: "DiagMeter") -> "DiagMeter":
        """Merge another meter into this one (sums stats)."""
        for k, ost in other._stats.items():
            if k not in self._stats:
                if not self.allow_new_keys:
                    continue
                self._stats[k] = _ScalarStats(self.device)
            st = self._stats[k]
            st.sum += ost.sum.to(device=self.device)
            st.sum_sq += ost.sum_sq.to(device=self.device)
            st.count += ost.count.to(device=self.device)
        return self

    def sync(self, group: Optional[object] = None) -> "DiagMeter":
        """
        DDP sync: all-reduce sums/sumsq/count across ranks.
        Robust to rank-local key differences by unioning keys.
        """
        if dist is None or not dist.is_available() or not dist.is_initialized():
            return self

        world = dist.get_world_size(group=group)
        if world <= 1:
            return self

        # Union keys across ranks
        keys_local = sorted(self._stats.keys())
        keys_all: list[list[str]] = [None for _ in range(world)]  # type: ignore
        dist.all_gather_object(keys_all, keys_local, group=group)
        keys_union = sorted({k for ks in keys_all for k in ks})

        # Ensure all keys exist locally
        for k in keys_union:
            if k not in self._stats:
                self._stats[k] = _ScalarStats(self.device)

        # Pack and all-reduce (3 scalars per key)
        buf = []
        for k in keys_union:
            st = self._stats[k]
            buf.extend([st.sum, st.sum_sq, st.count])
        packed = torch.stack([t.to(device=self.device, dtype=torch.float64).reshape(()) for t in buf])
        dist.all_reduce(packed, op=dist.ReduceOp.SUM, group=group)

        # Unpack
        it = iter(packed)
        for k in keys_union:
            st = self._stats[k]
            st.sum = next(it).reshape(())
            st.sum_sq = next(it).reshape(())
            st.count = next(it).reshape(())
        return self

    def state_dict(self) -> Dict[str, Any]:
        # Minimal state for checkpointing
        out = {}
        for k, st in self._stats.items():
            out[k] = {
                "sum": st.sum.detach().cpu(),
                "sum_sq": st.sum_sq.detach().cpu(),
                "count": st.count.detach().cpu(),
            }
        return {"device": str(self.device), "stats": out}

    def load_state_dict(self, state: Dict[str, Any]):
        self.reset()
        stats = state.get("stats", {})
        for k, v in stats.items():
            st = _ScalarStats(self.device)
            st.sum = v["sum"].to(device=self.device, dtype=torch.float64).reshape(())
            st.sum_sq = v["sum_sq"].to(device=self.device, dtype=torch.float64).reshape(())
            st.count = v["count"].to(device=self.device, dtype=torch.float64).reshape(())
            self._stats[k] = st

    def update(self, diag: Dict[str, Any], *, prefix: str = "", weight: float = 1.0):
        for k, v in diag.items():
            key = f"{prefix}{k}" if prefix else k
            if isinstance(v, torch.Tensor):
                if v.ndim != 0:
                    v = v.reshape(())
                x = v
            elif isinstance(v, (float, int)):
                x = torch.tensor(float(v), device=self.device, dtype=torch.float64)
            else:
                # ignore non-scalars / None
                continue

            if key not in self._stats:
                if not self.allow_new_keys:
                    continue
                self._stats[key] = _ScalarStats(self.device)
            self._stats[key].update(x, weight=weight)

    def compute(self) -> Dict[str, Dict[str, torch.Tensor]]:
        out: Dict[str, Dict[str, torch.Tensor]] = {}
        for k, st in self._stats.items():
            out[k] = {
                "mean": st.mean().detach().cpu(),
                "std": st.std().detach().cpu(),
                "count": st.count.detach().cpu(),
            }
        return out

    def compute_means(self) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for k, st in self._stats.items():
            out[k] = float(st.mean().detach().cpu().item())
        return out
