from __future__ import annotations

"""Minimal safetensors-compatible I/O.

The repository requirement is strict: checkpoints and exported model instances
must be stored in safetensors format. To keep the codebase self-contained, this
module implements the small subset of the safetensors specification that the
training runtime needs.

Supported tensor dtypes cover the common optimizer/model states used by this
project, including complex tensors.
"""

from dataclasses import dataclass
import json
from pathlib import Path
import struct
from typing import Any, Iterable, Mapping, MutableMapping

import numpy as np
import torch


_TORCH_TO_SAFE_DTYPE: dict[torch.dtype, str] = {
    torch.bool: "BOOL",
    torch.uint8: "U8",
    torch.int8: "I8",
    torch.int16: "I16",
    torch.int32: "I32",
    torch.int64: "I64",
    torch.float16: "F16",
    torch.bfloat16: "BF16",
    torch.float32: "F32",
    torch.float64: "F64",
    torch.complex64: "C64",
    torch.complex128: "C128",
}

if hasattr(torch, "uint16"):
    _TORCH_TO_SAFE_DTYPE[torch.uint16] = "U16"
if hasattr(torch, "uint32"):
    _TORCH_TO_SAFE_DTYPE[torch.uint32] = "U32"
if hasattr(torch, "uint64"):
    _TORCH_TO_SAFE_DTYPE[torch.uint64] = "U64"

_SAFE_TO_TORCH_DTYPE: dict[str, torch.dtype] = {code: dtype for dtype, code in _TORCH_TO_SAFE_DTYPE.items()}

_NUMPY_SAFE_CAST: dict[str, np.dtype] = {
    "BOOL": np.dtype(np.bool_),
    "U8": np.dtype(np.uint8),
    "I8": np.dtype(np.int8),
    "I16": np.dtype(np.int16),
    "I32": np.dtype(np.int32),
    "I64": np.dtype(np.int64),
    "U16": np.dtype(np.uint16),
    "U32": np.dtype(np.uint32),
    "U64": np.dtype(np.uint64),
    "F16": np.dtype(np.float16),
    "BF16": np.dtype(np.uint16),
    "F32": np.dtype(np.float32),
    "F64": np.dtype(np.float64),
    "C64": np.dtype(np.complex64),
    "C128": np.dtype(np.complex128),
}


@dataclass(frozen=True)
class SafeTensorsFile:
    tensors: dict[str, torch.Tensor]
    metadata: dict[str, str]


class SafeTensorFormatError(ValueError):
    pass


def _normalize_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.dtype not in _TORCH_TO_SAFE_DTYPE:
        raise TypeError(f"Unsupported dtype for safetensors serialization: {tensor.dtype}")
    return tensor.detach().cpu().contiguous().resolve_conj().resolve_neg()


def _tensor_nbytes(tensor: torch.Tensor) -> int:
    return int(tensor.numel()) * int(tensor.element_size())


def _tensor_bytes_view(tensor: torch.Tensor) -> torch.Tensor:
    normalized = _normalize_tensor(tensor)
    base = normalized.reshape(1) if normalized.ndim == 0 else normalized
    return base.view(torch.uint8).reshape(-1)


def _encode_header(tensors: Mapping[str, torch.Tensor], metadata: Mapping[str, str] | None = None) -> bytes:
    header: dict[str, Any] = {}
    offset = 0
    for name, tensor in tensors.items():
        if name == "__metadata__":
            raise SafeTensorFormatError("'__metadata__' is reserved in safetensors headers")
        normalized = _normalize_tensor(tensor)
        nbytes = _tensor_nbytes(normalized)
        header[name] = {
            "dtype": _TORCH_TO_SAFE_DTYPE[normalized.dtype],
            "shape": list(normalized.shape),
            "data_offsets": [offset, offset + nbytes],
        }
        offset += nbytes
    if metadata:
        meta = {str(k): str(v) for k, v in metadata.items()}
        header["__metadata__"] = meta
    return json.dumps(header, ensure_ascii=False, separators=(",", ":")).encode("utf-8")


def save_tensors(path: str | Path, tensors: Mapping[str, torch.Tensor], metadata: Mapping[str, str] | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = _encode_header(tensors, metadata=metadata)
    with path.open("wb") as f:
        f.write(struct.pack("<Q", len(header)))
        f.write(header)
        for name, tensor in tensors.items():
            normalized = _normalize_tensor(tensor)
            if normalized.numel() == 0:
                continue
            raw = _tensor_bytes_view(normalized).numpy()
            f.write(memoryview(raw))
    return path


def _empty_tensor(shape: Iterable[int], dtype: torch.dtype, device: str | torch.device = "cpu") -> torch.Tensor:
    return torch.empty(tuple(int(x) for x in shape), dtype=dtype, device=device)


def load_tensors(path: str | Path, *, device: str | torch.device = "cpu") -> SafeTensorsFile:
    path = Path(path)
    with path.open("rb") as f:
        header_len_raw = f.read(8)
        if len(header_len_raw) != 8:
            raise SafeTensorFormatError("Invalid safetensors file: missing header length")
        header_len = struct.unpack("<Q", header_len_raw)[0]
        header_raw = f.read(header_len)
        if len(header_raw) != header_len:
            raise SafeTensorFormatError("Invalid safetensors file: truncated header")
        header = json.loads(header_raw.decode("utf-8"))

        file_size = path.stat().st_size
        data_start = 8 + header_len
        tensors: dict[str, torch.Tensor] = {}
        metadata = {str(k): str(v) for k, v in header.get("__metadata__", {}).items()}
        for name, spec in header.items():
            if name == "__metadata__":
                continue
            dtype_code = spec["dtype"]
            shape = tuple(int(x) for x in spec["shape"])
            offsets = spec["data_offsets"]
            if dtype_code not in _SAFE_TO_TORCH_DTYPE:
                raise SafeTensorFormatError(f"Unsupported dtype code in safetensors file: {dtype_code}")
            dtype = _SAFE_TO_TORCH_DTYPE[dtype_code]
            start = int(offsets[0])
            end = int(offsets[1])
            if end < start:
                raise SafeTensorFormatError(f"Invalid data offsets for tensor {name!r}")
            abs_start = data_start + start
            abs_end = data_start + end
            if abs_end > file_size:
                raise SafeTensorFormatError(f"Tensor {name!r} exceeds safetensors file bounds")
            if end == start:
                tensor = _empty_tensor(shape, dtype=dtype, device="cpu")
            else:
                f.seek(abs_start)
                raw = bytearray(f.read(end - start))
                if len(raw) != (end - start):
                    raise SafeTensorFormatError(f"Tensor {name!r} is truncated in safetensors payload")
                raw_u8 = torch.frombuffer(raw, dtype=torch.uint8, count=(end - start)).clone()
                tensor = raw_u8.view(dtype).reshape(shape)
            if str(device) != "cpu":
                tensor = tensor.to(device=device)
            tensors[name] = tensor
    return SafeTensorsFile(tensors=tensors, metadata=metadata)


def save_state_dict(path: str | Path, state_dict: Mapping[str, torch.Tensor], metadata: Mapping[str, str] | None = None) -> Path:
    ordered = {str(name): tensor for name, tensor in state_dict.items()}
    return save_tensors(path, ordered, metadata=metadata)


def load_state_dict(path: str | Path, *, device: str | torch.device = "cpu") -> dict[str, torch.Tensor]:
    return load_tensors(path, device=device).tensors


# ---------------------------------------------------------------------------
# Structured-object checkpointing (nested dict/list/tuple/scalars + tensors)
# ---------------------------------------------------------------------------


def _tensor_placeholder_name(index: int) -> str:
    return f"tensor_{index:06d}"


def _encode_object(obj: Any, tensors: MutableMapping[str, torch.Tensor]) -> Any:
    if isinstance(obj, torch.Tensor):
        name = _tensor_placeholder_name(len(tensors))
        tensors[name] = obj
        return {"kind": "torch_tensor", "name": name}
    if isinstance(obj, np.ndarray):
        name = _tensor_placeholder_name(len(tensors))
        tensors[name] = torch.from_numpy(np.ascontiguousarray(obj))
        return {
            "kind": "numpy_array",
            "name": name,
            "dtype": str(obj.dtype),
            "shape": list(obj.shape),
        }
    if isinstance(obj, np.generic):
        return {
            "kind": "numpy_scalar",
            "dtype": str(obj.dtype),
            "value": obj.item(),
        }
    if isinstance(obj, Path):
        return {"kind": "path", "value": str(obj)}
    if isinstance(obj, torch.device):
        return {"kind": "torch_device", "value": str(obj)}
    if isinstance(obj, torch.dtype):
        return {"kind": "torch_dtype", "value": str(obj)}
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return {"kind": "literal", "value": obj}
    if isinstance(obj, dict):
        return {
            "kind": "dict",
            "items": [
                [_encode_object(key, tensors), _encode_object(value, tensors)]
                for key, value in obj.items()
            ],
        }
    if isinstance(obj, list):
        return {"kind": "list", "items": [_encode_object(item, tensors) for item in obj]}
    if isinstance(obj, tuple):
        return {"kind": "tuple", "items": [_encode_object(item, tensors) for item in obj]}
    raise TypeError(f"Unsupported object type for safetensors checkpoint manifest: {type(obj)!r}")


def _tensor_to_numpy(tensor: torch.Tensor, dtype_name: str, shape: tuple[int, ...]) -> np.ndarray:
    safe_code = _TORCH_TO_SAFE_DTYPE[tensor.dtype]
    u8 = _tensor_bytes_view(tensor).numpy()
    if safe_code == "BF16":
        base = np.frombuffer(u8.tobytes(), dtype=np.uint16).reshape(shape)
        return base.view(np.dtype(dtype_name))
    arr = np.frombuffer(u8.tobytes(), dtype=np.dtype(dtype_name))
    return arr.reshape(shape)


def _decode_object(node: Any, tensors: Mapping[str, torch.Tensor]) -> Any:
    kind = node["kind"]
    if kind == "torch_tensor":
        return tensors[node["name"]]
    if kind == "numpy_array":
        tensor = tensors[node["name"]]
        return _tensor_to_numpy(tensor, node["dtype"], tuple(int(x) for x in node["shape"]))
    if kind == "numpy_scalar":
        return np.dtype(node["dtype"]).type(node["value"])
    if kind == "path":
        return Path(node["value"])
    if kind == "torch_device":
        return torch.device(node["value"])
    if kind == "torch_dtype":
        value = node["value"]
        if not value.startswith("torch."):
            raise SafeTensorFormatError(f"Unsupported torch dtype encoding: {value}")
        dtype_name = value.split(".", 1)[1]
        return getattr(torch, dtype_name)
    if kind == "literal":
        return node["value"]
    if kind == "list":
        return [_decode_object(item, tensors) for item in node["items"]]
    if kind == "tuple":
        return tuple(_decode_object(item, tensors) for item in node["items"])
    if kind == "dict":
        out: dict[Any, Any] = {}
        for key_node, value_node in node["items"]:
            out[_decode_object(key_node, tensors)] = _decode_object(value_node, tensors)
        return out
    raise SafeTensorFormatError(f"Unknown manifest node kind: {kind}")


def save_object(path: str | Path, obj: Any, *, metadata: Mapping[str, str] | None = None) -> Path:
    tensors: dict[str, torch.Tensor] = {}
    manifest = _encode_object(obj, tensors)
    meta = {"framesmoothie_manifest": json.dumps(manifest, ensure_ascii=False, separators=(",", ":"))}
    if metadata:
        meta.update({str(k): str(v) for k, v in metadata.items()})
    return save_tensors(path, tensors, metadata=meta)


def load_object(path: str | Path, *, device: str | torch.device = "cpu") -> Any:
    loaded = load_tensors(path, device=device)
    manifest_raw = loaded.metadata.get("framesmoothie_manifest")
    if manifest_raw is None:
        raise SafeTensorFormatError("Missing framesmoothie manifest in safetensors metadata")
    manifest = json.loads(manifest_raw)
    return _decode_object(manifest, loaded.tensors)


__all__ = [
    "SafeTensorFormatError",
    "SafeTensorsFile",
    "load_object",
    "load_state_dict",
    "load_tensors",
    "save_object",
    "save_state_dict",
    "save_tensors",
]
