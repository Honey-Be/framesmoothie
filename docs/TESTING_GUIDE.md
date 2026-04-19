# Framesmoothie v0.4.0 Testing Guide

이 문서는 현재 코드베이스에서 **실전 검증을 시작하기 위한 최소 테스트 루틴**을 정리한다.

> **v0.4.0 변경사항**: s9 v0.4.0 업그레이드에 따라 ARS9 파생 블록, Q 계열(양자화) 파생 블록, 체크포인트 마이그레이션 테스트가 추가되었다. 자세한 내용은 `docs/MIGRATION_v0.4.0.md` 와 `docs/LAYER_GUIDE.md`를 참조한다.


## 0) 자동 실행 러너

반복 실행을 줄이기 위해 루트에 아래 두 파일이 포함되어 있다.

- `Makefile`
- `justfile`

### Makefile 예시

```bash
make compile
make test-fast
make test-s9
make toy-data
make tiny-overfit
```

### justfile 예시

```bash
just compile
just test-fast
just test-s9
just toy-data
just tiny-overfit cpu
```

## 1) 빠른 정적 검증

```bash
python -m py_compile $(find src/framesmoothie -name '*.py')
pytest -q tests/test_specs.py tests/test_diag_meter.py tests/test_zone_losses.py
```

## 2) zoning / HMC / smoke test

`s9`가 설치되어 있다면 아래를 권장한다.

```bash
pytest -q tests/test_zoning.py tests/test_hmc_boundary.py tests/test_model_smoke.py
```

## 3) gradcheck (slow)

```bash
pytest -q tests/test_gradcheck_fmlm.py -m slow
```

## 4) single-step smoke train

```bash
pytest -q tests/test_smoke_train.py -m slow
```

## 5) tiny overfit script

먼저 toy dataset 생성:

```bash
python -m framesmoothie._scripts.make_toy_panoptic --output artifacts/toy_panoptic.pt --num-samples 16 --target-style-shift
```

그 다음 tiny overfit 실행:

```bash
python -m framesmoothie._scripts.run_tiny_overfit --dataset artifacts/toy_panoptic.pt --steps 100 --device cpu
```

## 6) 추천 확인 지표

- `loss_src_sem`, `loss_src_inst`
- `loss_pred_src`, `loss_pred_tgt`
- `loss_aux`
- `corr_ref_src_ema`
- `tgt_teacher/corr_sem_inst`
- `tgt_student/corr_sem_inst`

## 7) v0.4.0 신규 테스트

### ARS9 블록 테스트
```bash
pytest -q tests/test_ars9_blocks.py
```
ARS9CondMixBlock / ARS9DecoderLayer / ARS9Decoder 의 1D/2D forward + HiPPO-N 초기화를 검증한다.

### 양자화 블록 테스트
```bash
pytest -q tests/test_quantization_blocks.py
```
QRS9CondMixBlock / QARS9CondMixBlock 및 해당 Decoder stack 의 forward/backward, STE gradient 흐름, HiPPO-N+양자화 조합을 검증한다.

### 체크포인트 마이그레이션 테스트
```bash
pytest -q tests/test_migration.py
```
v0.3.x (approx 이산화) → v0.4.0 (exact ZOH) 전환 시 RS9/ARS9 블록의 forward 동치성, roundtrip lossless, 중첩 dict 체크포인트 지원을 검증한다.

## 8) 주의

- `tests/test_zoning.py`, `tests/test_model_smoke.py`, `tests/test_smoke_train.py`, `tests/test_gradcheck_fmlm.py`는 `s9`가 설치되지 않으면 자동 skip되도록 작성했다.
- `tests/test_ars9_blocks.py`, `tests/test_quantization_blocks.py`, `tests/test_migration.py` 도 동일하게 `s9` v0.4.0이 필요하다.
- DDP 검증은 현재 `DiagMeter.sync()`를 직접 호출하는 별도 스크립트/실험 루프에서 확인하는 것을 권장한다.


## 9) preset별 실행 / 로그 디렉토리 / 결과 요약

### 단일 preset 실행
```bash
python -m framesmoothie._scripts.run_tiny_overfit --dataset artifacts/toy_panoptic.pt --preset pred --steps 100 --device cpu
```

기본적으로 결과는 `artifacts/runs/<preset>_<timestamp>/` 아래에 저장된다.

저장 파일:
- `config.json`
- `metrics.jsonl`
- `metrics.csv`
- `summary.json`

선택적으로 checkpoint와 best-model 선택을 같이 쓰려면:

```bash
python -m framesmoothie._scripts.run_tiny_overfit \
  --dataset artifacts/toy_panoptic.pt \
  --preset pred \
  --steps 100 \
  --device cpu \
  --save-checkpoints \
  --select-metric loss \
  --select-mode min
```

추가 저장 파일:
- `checkpoints/step_*.pt`
- `best_checkpoint.pt`

### preset suite 실행
```bash
python -m framesmoothie._scripts.run_preset_suite --dataset artifacts/toy_panoptic.pt --presets baseline zoning pred full full_lite_instance_friendly full_lite_target_instance_friendly full_lite_target_instance_friendly_v3 full_lite_target_instance_friendly_v4 --steps 100 --device cpu
```

suite 결과는 `artifacts/suites/<suite_name>/` 아래에 저장된다.

저장 파일:
- 각 preset별 run subdir
- `suite_summary.json`
- `suite_summary.csv`
- `plot_final_loss.png`
- `plot_best_metric.png`
- `plot_final_loss_src_sem.png`
- `plot_final_loss_src_inst.png`
- `plot_final_loss_tgt_sem.png`
- `plot_final_loss_tgt_inst.png`

즉, suite를 돌리면 결과 요약과 기본 플롯이 한 번에 만들어진다.

현재 기본 suite에는 아래 preset이 포함된다.

- `baseline`
- `zoning`
- `pred`
- `full`
- `full_lite_instance_friendly`
- `full_lite_target_instance_friendly`

특히 마지막 두 preset은 instance branch를 덜 희생하는 방향을 보기 위한 비교용이다.

- `full_lite_instance_friendly`
  - predictive label pressure 완화
  - instance branch에 `structure` zone 재도입
  - HMC boundary prior 완화
- `full_lite_target_instance_friendly`
  - 위 조정 위에 `lambda_pred`와 label edge influence를 한 번 더 낮춰
    **target instance** 회복을 더 노리는 버전


플롯만 다시 그리고 싶다면:

```bash
python -m framesmoothie._scripts.plot_suite_summary --suite-dir artifacts/suites/<suite_name>
```

또는:

```bash
make plots SUITE_DIR=artifacts/suites/<suite_name>
just plot-suite artifacts/suites/<suite_name>
```

### Makefile / justfile
```bash
make list-presets
make tiny-overfit-preset PRESET=pred
make tiny-overfit-suite
```

```bash
just list-presets
just tiny-overfit-preset pred cpu
just tiny-overfit-suite cpu
```


## 10) resume / early stopping / top-k checkpoints

### 단일 run에서 top-k 저장
```bash
python -m framesmoothie._scripts.run_tiny_overfit \
  --dataset artifacts/toy_panoptic.pt \
  --preset pred \
  --steps 200 \
  --device cpu \
  --save-checkpoints \
  --select-metric loss \
  --select-mode min \
  --topk 3
```

이 경우 run 디렉토리 아래에:
- `checkpoints/step_*.pt`
- `best_checkpoint.pt`
- `topk_checkpoints.json`

이 함께 저장된다. `topk_checkpoints.json`에는 현재 유지 중인 top-k checkpoint와 metric 값이 들어 있다.

### early stopping
```bash
python -m framesmoothie._scripts.run_tiny_overfit \
  --dataset artifacts/toy_panoptic.pt \
  --preset pred \
  --steps 500 \
  --device cpu \
  --save-checkpoints \
  --select-metric loss \
  --select-mode min \
  --early-stop-patience 5 \
  --early-stop-metric loss \
  --early-stop-mode min \
  --early-stop-min-delta 0.0
```

주의:
- patience는 **log event 기준**이다. 즉 `log_every=10`이면 10 step마다 한 번 개선 여부를 본다.
- `summary.json`에 `early_stopped`, `stopped_step`, `steps_executed`가 기록된다.

### resume from checkpoint
```bash
python -m framesmoothie._scripts.run_tiny_overfit \
  --dataset artifacts/toy_panoptic.pt \
  --preset pred \
  --steps 500 \
  --device cpu \
  --resume-from artifacts/runs/<run>/checkpoints/step_000100.pt \
  --save-checkpoints
```

resume 시:
- model / teacher / optimizer state를 복원
- 기존 `metrics.jsonl`을 읽어서 이어붙임
- `run_dir`는 checkpoint가 속한 기존 run 디렉토리를 계속 사용

### Makefile / justfile 예시

```bash
make tiny-overfit-preset PRESET=pred TOPK=3 EARLY_STOP=5
make tiny-overfit-resume RESUME=artifacts/runs/<run>/checkpoints/step_000100.pt
```

```bash
just tiny-overfit-preset pred cpu 200 3 5 loss min 0.0
just tiny-overfit-resume artifacts/runs/<run>/checkpoints/step_000100.pt pred cpu 500 3
```


## Resume / scheduler / DiagMeter state

`run_tiny_overfit.py` checkpoint now stores and restores:

- model state
- teacher state
- optimizer state
- **scheduler state**
- **DiagMeter state**
- best metric / top-k manifest

### Resume example

```bash
python -m framesmoothie._scripts.run_tiny_overfit \
  --dataset artifacts/toy_panoptic.pt \
  --resume-from artifacts/runs/<run>/checkpoints/step_000100.pt \
  --save-checkpoints
```

### Scheduler examples

Cosine scheduler:
```bash
python -m framesmoothie._scripts.run_tiny_overfit \
  --dataset artifacts/toy_panoptic.pt \
  --preset pred \
  --steps 200 \
  --scheduler-kind cosine
```

Step scheduler:
```bash
python -m framesmoothie._scripts.run_tiny_overfit \
  --dataset artifacts/toy_panoptic.pt \
  --preset pred \
  --steps 200 \
  --scheduler-kind step \
  --scheduler-step-size 50 \
  --scheduler-gamma 0.5
```

### Makefile examples

```bash
make tiny-overfit-preset PRESET=pred SCHEDULER=cosine
make tiny-overfit-preset PRESET=pred SCHEDULER=step SCHED_STEP=50 SCHED_GAMMA=0.5
make tiny-overfit-resume RESUME=artifacts/runs/<run>/checkpoints/step_000100.pt
```

### just examples

```bash
just tiny-overfit-preset pred cpu 200 3 "" "" "" "" cosine 50 0.5
just tiny-overfit-resume artifacts/runs/<run>/checkpoints/step_000100.pt cpu 200 3 "" "" "" "" step 50 0.5
```


## Resume / scheduler / DiagMeter / EMA schedule

`run_tiny_overfit.py` checkpoint now saves and restores:

- model state
- teacher state
- optimizer state
- scheduler state
- `DiagMeter` state
- **EMA decay scheduler state**

### EMA schedule options

Supported CLI flags:

- `--ema-schedule-kind {constant,linear,cosine}`
- `--ema-beta-start`
- `--ema-beta-end`

Examples:

```bash
python -m framesmoothie._scripts.run_tiny_overfit \
  --dataset artifacts/toy_panoptic.pt \
  --preset pred \
  --steps 200 \
  --device cpu \
  --save-checkpoints \
  --ema-schedule-kind cosine \
  --ema-beta-start 0.99 \
  --ema-beta-end 0.999
```

Resume:

```bash
python -m framesmoothie._scripts.run_tiny_overfit \
  --dataset artifacts/toy_panoptic.pt \
  --steps 200 \
  --device cpu \
  --resume-from artifacts/runs/<run>/checkpoints/step_000100.pt
```

The checkpoint restores the EMA schedule state automatically, so teacher decay continues from the
previous step rather than restarting from the initial beta.

### Makefile

```bash
make tiny-overfit-preset PRESET=pred EMA_SCHEDULER=cosine EMA_BETA_START=0.99 EMA_BETA_END=0.999
make tiny-overfit-resume RESUME=artifacts/runs/<run>/checkpoints/step_000100.pt EMA_SCHEDULER=cosine
```

### justfile

```bash
just tiny-overfit-preset pred cpu 200 3 "" "" "" "" cosine 0.99 0.999
just tiny-overfit-resume artifacts/runs/<run>/checkpoints/step_000100.pt cpu 200 3 "" "" "" "" cosine 0.99 0.999
```

## Design notes (not implemented yet)

### Multiple schedulers support

A clean way to extend the current runner is to introduce a scheduler registry and a small scheduler spec,
for example:

- optimizer scheduler spec (LR)
- EMA scheduler spec
- optional predictor/projector scheduler spec

Then `run_tiny_overfit.py` can build and checkpoint a dict of schedulers:

```python
{
    "lr": lr_scheduler,
    "ema": ema_scheduler,
    "predictor": predictor_scheduler,
}
```

and save/load:

```python
{
    "scheduler_state_dicts": {name: sch.state_dict() for name, sch in schedulers.items()}
}
```

This keeps the current API compatible while allowing multiple independent schedules.

### Metric-specific top-k management

The current runner keeps top-k checkpoints for a single metric. To support multiple metrics cleanly,
the most maintainable approach is to make the manifest metric-aware, e.g.

```json
{
  "loss": [...],
  "corr_tgt_teacher": [...],
  "best_metric_name": "loss"
}
```

and update `_update_topk(...)` to operate on one metric namespace at a time.
The user-facing API can then be something like:

- `--topk-metrics loss,corr_tgt_teacher`
- `--topk-modes min,max`
- `--topk-limit 3`

This is best added only after the single-metric flow is stable.

## `full_lite_target_instance_friendly_v4`

`full_lite_target_instance_friendly_v4`는 v3에서 target-instance recovery에 성공했던 전역 설정을 유지하면서,
instance weighted fusion만 미세 조정한 preset이다.

핵심 차이:
- `structure`: `1.25 -> 1.15`
- `label`: `0.50 -> 0.65`
- `boundary`: `0.75 -> 0.85`

목표는 v3의 target-instance 이득을 최대한 유지하면서 source-instance fitting을 조금 회복하는 것이다.