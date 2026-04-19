# Framesmoothie v0.3.0 → v0.4.0 Migration Guide

이 문서는 framesmoothie v0.3.0에서 v0.4.0으로 업그레이드할 때 필요한 변경 사항을 정리한다.

## 의존성 변경

s9 v0.3.0 → v0.4.0. `pyproject.toml`의 s9 rev가 `v0.4.0`으로 변경되었다.

## Breaking Change: Exact ZOH 이산화 (default 변경)

s9 v0.4.0에서 SSM 커널 이산화 기본값이 변경되었다:

| 버전 | 기본 이산화 | 수식 |
|------|-----------|------|
| v0.3.x | `approx` | $\bar B = B \cdot \Delta t$ |
| v0.4.0 | `zoh` | $\bar B = \frac{e^{A \cdot \Delta t} - 1}{A} \cdot B$ |

### 영향 범위

`RS9CondMixBlock`, `RS9DecoderLayer`, `S9Stack` 모두 s9 커널을 내부적으로 사용한다. 새 코드에서 이 클래스들을 생성하면 자동으로 exact ZOH가 적용된다.

### 기존 동작 유지

모든 해당 생성자에 `discretization="approx"`를 명시하면 v0.3.x와 동일하게 동작한다:

```python
block = RS9CondMixBlock(
    c_model=64, q_dim=32, spatial_dims=2,
    discretization="approx",  # v0.3.x 호환
)
```

### 기존 체크포인트 마이그레이션

v0.3.x에서 학습한 체크포인트를 v0.4.0 (exact ZOH) 환경에서 그대로 사용하려면 SSM 파라미터 재매핑이 필요하다. 이 변환은 **lossless**(수학적으로 정확)하며, 변환 전후의 forward 출력이 동일하다.

#### 라이브러리 API

```python
from framesmoothie.migration import migrate_framesmoothie_checkpoint
import torch

# 단순 state_dict
old_sd = torch.load("checkpoint_v0.3.pt", map_location="cpu")
new_sd = migrate_framesmoothie_checkpoint(old_sd)
model.load_state_dict(new_sd)

# 중첩 dict ({"model": sd, "optimizer": ..., "epoch": ...})
checkpoint = torch.load("checkpoint_v0.3.pt", map_location="cpu")
migrated = migrate_framesmoothie_checkpoint(checkpoint)
model.load_state_dict(migrated["model"])
```

#### s9 CLI

```bash
python scripts/migrate_checkpoint.py --in old.pt --out new.pt
```

> **Note**: s9의 `scripts/migrate_checkpoint.py`를 직접 사용할 수도 있다. framesmoothie의 state_dict 내부에 포함된 RS9/ARS9 커널은 s9의 표준 패턴(`kernels.{i}.{param}`)을 따르므로 자동 감지된다.

#### 역방향 마이그레이션

v0.4.0 → v0.3.x로 돌아가야 하는 경우:

```python
from framesmoothie.migration import migrate_framesmoothie_checkpoint

reverted = migrate_framesmoothie_checkpoint(
    checkpoint, direction="from_zoh"
)
```

## 신규 초기화 옵션

모든 SSM 레이어 생성자에 `init_mode` kwarg가 추가되었다:

| `init_mode` | 대상 | 설명 |
|---|---|---|
| `"legacy"` (기본) | 전체 | v0.3.x 호환 초기화 |
| `"hippo_n"` | S9, ARS9 계열 | HiPPO-N 대각화. 장거리 의존성 학습에 유리 |
| `"s4d_real"` | RS9 계열 | S4D-Real log-spaced decay. 장거리 실수 동역학 |

```python
block = RS9CondMixBlock(
    c_model=64, q_dim=32, spatial_dims=2,
    init_mode="s4d_real",   # RS9 전용 장거리 초기화
)
```

## 신규 레이어 계열

### ARS9 (Advanced RS9)

RS9의 순실수 상태를 **복소 conjugate-pair**로 교체하여 진동 모드를 지원한다. I/O는 실수로 유지되므로 DOST가 불필요하다.

| 모듈 | 클래스 |
|---|---|
| `framesmoothie.blocks_ars9` | `ARS9CondMixBlock` |
| `framesmoothie.decoder_ars9` | `ARS9DecoderLayer`, `ARS9Decoder` |

```python
from framesmoothie.blocks_ars9 import ARS9CondMixBlock

block = ARS9CondMixBlock(
    c_model=64, q_dim=32, spatial_dims=2,
    init_mode="hippo_n",  # ARS9는 hippo_n 지원
)
```

### QRS9 / QARS9 (양자화 계열)

Q-S5의 per-component sensitivity 분석에 기반한 양자화 레이어. `QuantConfig`로 비트 할당을 제어한다.

| 모듈 | 클래스 |
|---|---|
| `framesmoothie.blocks_qrs9` | `QRS9CondMixBlock` |
| `framesmoothie.decoder_qrs9` | `QRS9DecoderLayer`, `QRS9Decoder` |
| `framesmoothie.blocks_qars9` | `QARS9CondMixBlock` |
| `framesmoothie.decoder_qars9` | `QARS9DecoderLayer`, `QARS9Decoder` |

```python
from s9.quantization import QuantConfig
from framesmoothie.blocks_qars9 import QARS9CondMixBlock

config = QuantConfig(
    w_bits_B=8,           # SSM B 행렬: int8
    w_bits_output=4,      # 출력 선형층: int4
    a_bits_input=8,       # 입력 활성: int8
    enforce_stability=True,
)

block = QARS9CondMixBlock(
    c_model=64, q_dim=32, spatial_dims=2,
    init_mode="hippo_n",
    quant_config=config,
)
```

## 신규 테스트 파일

| 테스트 파일 | 검증 대상 |
|---|---|
| `tests/test_ars9_blocks.py` | ARS9CondMixBlock, ARS9DecoderLayer, ARS9Decoder |
| `tests/test_quantization_blocks.py` | QRS9/QARS9 blocks, backward STE, HiPPO-N+양자화 |
| `tests/test_migration.py` | forward 동치성, roundtrip lossless, 중첩 dict 처리 |

실행:

```bash
pytest tests/test_ars9_blocks.py tests/test_quantization_blocks.py tests/test_migration.py -q
```
