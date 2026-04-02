from __future__ import annotations

"""Deterministic runtime and RNG-state utilities."""

from dataclasses import dataclass, field
import hashlib
import os
import platform
import random
import sys
from typing import Any, Mapping, MutableMapping, Optional

import numpy as np
import torch

from framesmoothie.fp import curry


def _stable_bytes(parts: tuple[Any, ...]) -> bytes:
    h = hashlib.blake2b(digest_size=16)
    for part in parts:
        h.update(repr(part).encode("utf-8"))
        h.update(b"\0")
    return h.digest()


@curry
def derive_seed(
    master_seed: int,
    namespace: str,
    rank: int = 0,
    worker: int | None = None,
    epoch: int | None = None,
    step: int | None = None,
) -> int:
    """Derive a stable 64-bit child seed from a master seed and coordinates."""

    return int.from_bytes(
        _stable_bytes((int(master_seed), namespace, rank, worker, epoch, step))[:8],
        byteorder="little",
        signed=False,
    )


@dataclass(frozen=True)
class SeedPath:
    """Hierarchical seed namespace.

    ``SeedPath`` lets the caller model the preset as a tree of RNG streams. This
    makes the assignment position explicit without having to log every sampled
    scalar.
    """

    master_seed: int
    parts: tuple[str, ...] = ()

    def child(self, *parts: Any) -> "SeedPath":
        return SeedPath(self.master_seed, self.parts + tuple(str(p) for p in parts))

    def seed(self) -> int:
        return derive_seed(self.master_seed, "/".join(self.parts), 0, None, None, None)

    def torch_generator(self, *, device: str | torch.device = "cpu") -> torch.Generator:
        gen = torch.Generator(device=str(device))
        gen.manual_seed(self.seed())
        return gen

    def numpy_generator(self) -> np.random.Generator:
        return np.random.default_rng(self.seed())

    def python_random(self) -> random.Random:
        rng = random.Random()
        rng.seed(self.seed())
        return rng


@dataclass(frozen=True)
class DeterminismConfig:
    master_seed: int = 0
    deterministic_algorithms: bool = True
    cudnn_benchmark: bool = False
    cudnn_deterministic: bool = True
    allow_tf32: bool = False
    matmul_precision: str = "highest"
    cublas_workspace_config: str | None = ":4096:2"
    python_hash_seed: int | None = None
    multiprocessing_context: str | None = None

    def with_context(self, ctx: str) -> "DeterminismConfig":
        return DeterminismConfig(
            master_seed=self.master_seed,
            deterministic_algorithms=self.deterministic_algorithms,
            cudnn_benchmark=self.cudnn_benchmark,
            cudnn_deterministic=self.cudnn_deterministic,
            allow_tf32=self.allow_tf32,
            matmul_precision=self.matmul_precision,
            cublas_workspace_config=self.cublas_workspace_config,
            python_hash_seed=self.python_hash_seed,
            multiprocessing_context=ctx,
        )


@dataclass
class RngSnapshot:
    python_state: Any
    numpy_global_state: Any
    numpy_named_states: dict[str, Any] = field(default_factory=dict)
    torch_cpu_state: torch.Tensor | None = None
    torch_cuda_states: list[torch.Tensor] = field(default_factory=list)
    torch_named_states: dict[str, torch.Tensor] = field(default_factory=dict)


class RngRegistry:
    """Explicit registry of named RNG streams.

    The registry is small but important: checkpoints only need to serialize the
    state of the registered generators plus the global framework RNGs.
    """

    def __init__(self) -> None:
        self.numpy_generators: dict[str, np.random.Generator] = {}
        self.torch_generators: dict[str, torch.Generator] = {}

    def register_numpy(self, name: str, generator: np.random.Generator) -> np.random.Generator:
        self.numpy_generators[name] = generator
        return generator

    def register_torch(self, name: str, generator: torch.Generator) -> torch.Generator:
        self.torch_generators[name] = generator
        return generator

    def snapshot(self) -> RngSnapshot:
        return capture_rng_snapshot(
            numpy_generators=self.numpy_generators,
            torch_generators=self.torch_generators,
        )

    def restore(self, snapshot: RngSnapshot) -> None:
        restore_rng_snapshot(
            snapshot,
            numpy_generators=self.numpy_generators,
            torch_generators=self.torch_generators,
        )


def choose_multiprocessing_context() -> str:
    if sys.platform == "win32":
        return "spawn"
    if sys.platform == "darwin":
        return "spawn"
    if sys.version_info >= (3, 14):
        return "forkserver"
    return "fork"


def configure_determinism(config: DeterminismConfig) -> dict[str, Any]:
    """Apply deterministic runtime settings and return an environment report."""

    desired_hash_seed = str(config.python_hash_seed if config.python_hash_seed is not None else config.master_seed)
    existing_hash_seed = os.environ.get("PYTHONHASHSEED")
    os.environ.setdefault("PYTHONHASHSEED", desired_hash_seed)

    if config.cublas_workspace_config:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", config.cublas_workspace_config)

    random.seed(config.master_seed)
    np.random.seed(config.master_seed)
    torch.manual_seed(config.master_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.master_seed)

    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(config.deterministic_algorithms)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.benchmark = bool(config.cudnn_benchmark)
        torch.backends.cudnn.deterministic = bool(config.cudnn_deterministic)
        torch.backends.cudnn.allow_tf32 = bool(config.allow_tf32)
    if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
        torch.backends.cuda.matmul.allow_tf32 = bool(config.allow_tf32)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision(config.matmul_precision)

    return {
        "python_version": platform.python_version(),
        "sys_version": sys.version,
        "platform": platform.platform(),
        "master_seed": int(config.master_seed),
        "python_hash_seed_requested": desired_hash_seed,
        "python_hash_seed_env_before": existing_hash_seed,
        "python_hash_seed_env_after": os.environ.get("PYTHONHASHSEED"),
        "python_hash_seed_effective_for_current_process": existing_hash_seed == desired_hash_seed,
        "multiprocessing_context": config.multiprocessing_context or choose_multiprocessing_context(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if hasattr(torch.backends, "cudnn") else None,
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()) if hasattr(torch, "are_deterministic_algorithms_enabled") else None,
        "cudnn_benchmark": bool(torch.backends.cudnn.benchmark) if hasattr(torch.backends, "cudnn") else None,
        "cudnn_deterministic": bool(torch.backends.cudnn.deterministic) if hasattr(torch.backends, "cudnn") else None,
        "allow_tf32": bool(config.allow_tf32),
        "matmul_precision": torch.get_float32_matmul_precision() if hasattr(torch, "get_float32_matmul_precision") else None,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "torch_config": torch.__config__.show(),
    }


@curry
def make_worker_init_fn(master_seed: int, namespace: str, rank: int = 0):
    """Return a DataLoader ``worker_init_fn`` with explicit namespace seeding.

    PyTorch already seeds workers from the DataLoader generator; we additionally
    seed Python and NumPy from the derived worker seed so that every randomness
    source follows the same deterministic tree.
    """

    def _init(worker_id: int) -> None:
        worker_seed = derive_seed(master_seed, namespace, rank, worker_id, None, None) % (2**32)
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    return _init


def capture_rng_snapshot(
    *,
    numpy_generators: Mapping[str, np.random.Generator] | None = None,
    torch_generators: Mapping[str, torch.Generator] | None = None,
) -> RngSnapshot:
    numpy_generators = dict(numpy_generators or {})
    torch_generators = dict(torch_generators or {})
    return RngSnapshot(
        python_state=random.getstate(),
        numpy_global_state=np.random.get_state(),
        numpy_named_states={name: generator.bit_generator.state for name, generator in numpy_generators.items()},
        torch_cpu_state=torch.get_rng_state(),
        torch_cuda_states=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        torch_named_states={name: generator.get_state() for name, generator in torch_generators.items()},
    )


def restore_rng_snapshot(
    snapshot: RngSnapshot,
    *,
    numpy_generators: MutableMapping[str, np.random.Generator] | None = None,
    torch_generators: MutableMapping[str, torch.Generator] | None = None,
) -> None:
    numpy_generators = numpy_generators or {}
    torch_generators = torch_generators or {}

    random.setstate(snapshot.python_state)
    np.random.set_state(snapshot.numpy_global_state)

    for name, generator in numpy_generators.items():
        if name in snapshot.numpy_named_states:
            generator.bit_generator.state = snapshot.numpy_named_states[name]

    if snapshot.torch_cpu_state is not None:
        torch.set_rng_state(snapshot.torch_cpu_state)
    if torch.cuda.is_available() and snapshot.torch_cuda_states:
        torch.cuda.set_rng_state_all(snapshot.torch_cuda_states)

    for name, generator in torch_generators.items():
        if name in snapshot.torch_named_states:
            generator.set_state(snapshot.torch_named_states[name])


def _hash_tensor(h: "hashlib._Hash", tensor: torch.Tensor) -> None:
    t = tensor.detach().cpu().contiguous().resolve_conj().resolve_neg().view(torch.uint8).reshape(-1)
    h.update(memoryview(t.numpy()))


def rng_digest(snapshot: RngSnapshot) -> str:
    h = hashlib.blake2b(digest_size=16)
    h.update(repr(snapshot.python_state).encode("utf-8"))
    h.update(repr(snapshot.numpy_global_state).encode("utf-8"))
    for name in sorted(snapshot.numpy_named_states):
        h.update(name.encode("utf-8"))
        h.update(repr(snapshot.numpy_named_states[name]).encode("utf-8"))
    if snapshot.torch_cpu_state is not None:
        _hash_tensor(h, snapshot.torch_cpu_state)
    for state in snapshot.torch_cuda_states:
        _hash_tensor(h, state)
    for name in sorted(snapshot.torch_named_states):
        h.update(name.encode("utf-8"))
        _hash_tensor(h, snapshot.torch_named_states[name])
    return h.hexdigest()


__all__ = [
    "DeterminismConfig",
    "RngRegistry",
    "RngSnapshot",
    "SeedPath",
    "capture_rng_snapshot",
    "choose_multiprocessing_context",
    "configure_determinism",
    "derive_seed",
    "make_worker_init_fn",
    "restore_rng_snapshot",
    "rng_digest",
]
