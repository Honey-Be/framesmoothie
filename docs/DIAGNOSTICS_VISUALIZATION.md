# Framesmoothie Diagnostics Visualization How‑To

이 문서는 `DiagMeter`를 이용해 LRCA/FMLM/HMC 진단 값을 누적하고, TensorBoard / matplotlib / W&B / DDP 환경에서 시각화하는 방법을 설명한다.

## 1) 무엇을 볼 것인가

권장 지표 예시:

- `tgt_teacher/g_global_var`
- `tgt_teacher/corr_sem_inst`
- `tgt_student/corr_sem_inst`
- `corr_ref_src_ema`
- `corr_tgt_teacher_delta`
- `corr_tgt_student_delta`

해석 예시:

- `g_global_var ↓` : global gate가 거의 상수화됨
- `corr_sem_inst → 1` : semantic/instance gate 분화 부족
- `corr_tgt_*_delta` 편차 큼 : target에서 source 기준선과 다른 gate 관계

## 2) 기본 사용법

```python
from framesmoothie.diag_meter import DiagMeter

meter = DiagMeter(device="cpu")

out = step(
    student=model,
    teacher=teacher,
    src=src_batch,
    tgt=tgt_batch,
    diag_meter=meter,
    return_diag=False,
)

# 주기적으로
means = meter.compute_means()
print(means.get("tgt_teacher/corr_sem_inst", None))
meter.reset()
```

## 3) prefix 단위 집계

`DiagMeter`는 prefix 필터링을 지원한다.

```python
teacher_stats = meter.compute_filtered(prefix="tgt_teacher/")
student_means = meter.compute_means_filtered(prefix="tgt_student/")
```

## 4) TensorBoard

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter("runs/framesmoothie")
means = meter.compute_means()
for k, v in means.items():
    writer.add_scalar(k, v, global_step)
writer.flush()
```

## 5) Matplotlib

```python
import matplotlib.pyplot as plt

history = {"step": [], "corr_teacher": [], "gvar_teacher": []}

means = meter.compute_means()
history["step"].append(global_step)
history["corr_teacher"].append(means.get("tgt_teacher/corr_sem_inst", float("nan")))
history["gvar_teacher"].append(means.get("tgt_teacher/g_global_var", float("nan")))

plt.figure()
plt.plot(history["step"], history["corr_teacher"])
plt.xlabel("step")
plt.ylabel("corr_sem_inst")
plt.title("Target teacher corr_sem_inst")
plt.show()
```

## 6) Weights & Biases

```python
import wandb
wandb.init(project="framesmoothie")
wandb.log(meter.compute_means(), step=global_step)
```

## 7) DDP

DDP에서 로깅 전에는 `sync()`를 호출해 rank 간 통계를 맞춘다.

```python
meter.sync()
if rank == 0:
    means = meter.compute_means()
```

권장 패턴:

- `sync()`는 로깅 주기마다 호출
- 실제 출력/시각화는 `rank == 0`에서만 수행

## 8) 저장/복원

```python
state = meter.state_dict()
# checkpoint에 저장

meter2 = DiagMeter()
meter2.load_state_dict(state)
```

## 9) 주의사항

- `DiagMeter.update()`는 **스칼라 텐서 / 숫자만** 누적한다.
- `return_diag=True`는 raw dict를 직접 반환하므로 오버헤드가 있다.
- 실전 학습은 `diag_meter` 누적 방식을 기본으로 두고, 필요한 step만 `return_diag=True`로 확인하는 것을 권장한다.
