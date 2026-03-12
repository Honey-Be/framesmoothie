# Framesmoothie Diagnostics Visualization How‑To

이 문서는 `DiagMeter`(torchmetrics 스타일 누적/리셋 가능한 진단 메터)로 수집한 LRCA/FMLM/HMC 진단 값을 **로깅/시각화**하는 실전 가이드를 제공한다.

## 1) 무엇을 시각화할까?

권장 진단 키(예시):

- `tgt_teacher/g_global_var`: global gate 분산
- `tgt_teacher/corr_sem_inst`: semantic vs instance gate 상관
- `tgt_student/corr_sem_inst_delta`: (target corr − source EMA baseline)
- `tgt_student/g_semantic_mean`, `tgt_student/g_instance_mean`

목표:

- global gate가 두 task에서 **공통 요인으로** 작동하는지(분산/드리프트)
- semantic/instance local shared gate가 **적절히 분리**되는지(상관/델타)
- teacher→student에서 gate 통계가 어떻게 이동하는지

## 2) 코드: DiagMeter 누적

```python
from framesmoothie.diag_meter import DiagMeter

meter = DiagMeter(device="cpu")

# train loop
out = step(
    student=model,
    teacher=teacher,
    src=src_batch,
    tgt=tgt_batch,
    diag_meter=meter,     # ✅ 누적
    return_diag=False,
)

# N step마다 요약
if (global_step + 1) % 100 == 0:
    means = meter.compute_means()
    # 예: print 또는 logger로 전송
    print(global_step, means.get("tgt_teacher/corr_sem_inst", None))
    meter.reset()
```

## 3) TensorBoard 로깅

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir="runs/framesmoothie")

# ... inside periodic logging block
means = meter.compute_means()
for k, v in means.items():
    writer.add_scalar(k, v, global_step)
writer.flush()
```

## 4) Matplotlib로 타임시리즈 플롯

훈련 중 매 100 step마다 `means`를 리스트에 저장해두었다가 종료 후 플롯한다.

```python
import matplotlib.pyplot as plt

history = {"step": [], "corr": [], "gvar": []}

# periodic:
means = meter.compute_means()
history["step"].append(global_step)
history["corr"].append(means.get("tgt_teacher/corr_sem_inst", float("nan")))
history["gvar"].append(means.get("tgt_teacher/g_global_var", float("nan")))

# after training:
plt.figure()
plt.plot(history["step"], history["corr"])
plt.xlabel("step")
plt.ylabel("corr_sem_inst (teacher)")
plt.title("Target teacher corr_sem_inst")
plt.show()

plt.figure()
plt.plot(history["step"], history["gvar"])
plt.xlabel("step")
plt.ylabel("g_global_var (teacher)")
plt.title("Target teacher g_global variance")
plt.show()
```

## 5) W&B 로깅(선택)

```python
import wandb
wandb.init(project="framesmoothie")

means = meter.compute_means()
wandb.log({k: v for k, v in means.items()}, step=global_step)
```

## 6) DDP에서의 진단 집계

DDP 환경에서 각 rank가 별도로 meter를 업데이트한다면, 로그를 내기 전에 `sync()`를 호출해 global 평균으로 만든다.

```python
meter.sync()            # all-reduce sum/sumsq/count
means = meter.compute_means()
```

권장 패턴:

- `rank==0`에서만 출력/로깅
- `sync()`는 로깅 주기마다(예: 100 step마다) 호출

## 7) 해석 가이드

- `g_global_var`가 0에 수렴: global gate가 거의 상수(학습이 죽었거나 과도하게 정규화됨)
- `corr_sem_inst`가 1에 수렴: semantic/instance local gate가 사실상 동일(분리 부족)
- `corr_sem_inst_delta`가 크게 음수/양수로 치우침: target에서 gate 관계가 source 기준선과 다름
  - HMC pseudo 품질/threshold 조정 포인트

## 8) 체크리스트

- `DiagMeter.update()`는 **스칼라만 누적**한다. 벡터/텐서가 들어오면 무시됨.
- `return_diag=True`는 디버그용(메모리/오버헤드 증가). 실전은 `diag_meter` 사용 권장.
