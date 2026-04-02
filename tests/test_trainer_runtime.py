import torch
import torch.nn as nn

from framesmoothie.trainer_runtime import RuntimeEnv, RuntimePolicy, RuntimeState, make_step_program


class DummyStep:
    def __init__(self):
        self.calls = []

    def __call__(self, src, tgt):
        batch_size = src["image"].shape[0]
        self.calls.append(batch_size)
        if batch_size > 1:
            raise RuntimeError("CUDA out of memory")
        loss = (src["image"].sum() + tgt["image"].sum()) * 0.0
        loss = loss + torch.tensor(1.0, requires_grad=True)
        return {
            "loss": loss,
            "loss_src_sem": torch.tensor(0.1),
            "loss_src_inst": torch.tensor(0.2),
            "loss_tgt_sem": torch.tensor(0.3),
            "loss_tgt_inst": torch.tensor(0.4),
            "loss_aux": torch.tensor(0.0),
        }



def test_make_step_program_reduces_microbatch_on_oom():
    student = nn.Linear(2, 2)
    teacher = nn.Linear(2, 2)
    opt = torch.optim.SGD(student.parameters(), lr=0.1)
    step = DummyStep()
    env = RuntimeEnv(
        student=student,
        teacher=teacher,
        optimizer=opt,
        step_fn=step,
        policy=RuntimePolicy(device="cpu", microbatch_size=4, cpu_fallback_on_cuda_oom=False),
        diag_meter=None,
    )
    state = RuntimeState(effective_device="cpu", microbatch_size=4)
    batch = {"image": torch.randn(4, 2)}

    new_state, (out, report), log = make_step_program(batch, batch).execute(env, state)

    assert report.effective_microbatch_size == 1
    assert new_state.last_effective_microbatch_size == 1
    assert report.oom_retries >= 1
    assert any(event["event"] == "oom_recovery" for event in log.events)
    assert out["loss"].shape == ()
