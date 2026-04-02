from pathlib import Path

import numpy as np
import torch

from framesmoothie.safetensors_io import load_object, load_state_dict, load_tensors, save_object, save_state_dict, save_tensors


def test_tensor_roundtrip(tmp_path: Path):
    path = tmp_path / "weights.safetensors"
    tensors = {
        "linear.weight": torch.arange(12, dtype=torch.float32).reshape(3, 4),
        "linear.bias": torch.tensor([1.0, 2.0, 3.0], dtype=torch.float64),
    }
    save_tensors(path, tensors, metadata={"kind": "unit-test"})
    loaded = load_tensors(path)
    assert loaded.metadata["kind"] == "unit-test"
    assert torch.equal(loaded.tensors["linear.weight"], tensors["linear.weight"])
    assert torch.equal(loaded.tensors["linear.bias"], tensors["linear.bias"])



def test_state_dict_roundtrip(tmp_path: Path):
    path = tmp_path / "state.safetensors"
    state_dict = {
        "module.weight": torch.randn(2, 2, dtype=torch.float32),
        "module.bias": torch.randn(2, dtype=torch.float16),
    }
    save_state_dict(path, state_dict)
    restored = load_state_dict(path)
    for key in state_dict:
        assert torch.equal(restored[key], state_dict[key])



def test_nested_object_roundtrip(tmp_path: Path):
    path = tmp_path / "checkpoint.safetensors"
    obj = {
        "step": 7,
        "optimizer_state_dict": {
            "state": {
                0: {
                    "momentum_buffer": torch.arange(4, dtype=torch.float32),
                    "exp_avg_sq": torch.arange(4, dtype=torch.float64),
                }
            },
            "param_groups": [{"lr": 0.01, "params": [0]}],
        },
        "rng": {
            "numpy_state": np.arange(8, dtype=np.uint32),
            "python_state": (1, 2, None),
        },
    }
    save_object(path, obj, metadata={"format": "checkpoint"})
    restored = load_object(path)
    assert restored["step"] == 7
    assert torch.equal(
        restored["optimizer_state_dict"]["state"][0]["momentum_buffer"],
        obj["optimizer_state_dict"]["state"][0]["momentum_buffer"],
    )
    assert torch.equal(
        restored["optimizer_state_dict"]["state"][0]["exp_avg_sq"],
        obj["optimizer_state_dict"]["state"][0]["exp_avg_sq"],
    )
    assert np.array_equal(restored["rng"]["numpy_state"], obj["rng"]["numpy_state"])
    assert restored["rng"]["python_state"] == obj["rng"]["python_state"]


def test_actual_optimizer_state_dict_roundtrip(tmp_path: Path):
    model = torch.nn.Linear(2, 2)
    opt = torch.optim.AdamW(model.parameters(), lr=0.01)
    loss = model(torch.randn(1, 2)).sum()
    loss.backward()
    opt.step()

    state_dict = opt.state_dict()
    path = tmp_path / "optimizer.safetensors"
    save_object(path, {"optimizer_state_dict": state_dict})
    restored = load_object(path)["optimizer_state_dict"]

    assert restored["param_groups"] == state_dict["param_groups"]
    for param_idx, slot in state_dict["state"].items():
        restored_slot = restored["state"][param_idx]
        for key, value in slot.items():
            if isinstance(value, torch.Tensor):
                assert torch.equal(restored_slot[key], value)
            else:
                assert restored_slot[key] == value
