# Framesmoothie Layer Family Guide

이 문서는 framesmoothie가 사용하는 SSM 레이어 계열(RS9 / ARS9 / QRS9 / QARS9)의 관계와 선택 기준을 정리한다.

## 계열 관계도

```
              ┌─ RS9CondMixBlock ─── RS9DecoderLayer ─── RS9Decoder
              │   (실수 I/O, 실수 상태)
              │
              ├─ ARS9CondMixBlock ── ARS9DecoderLayer ── ARS9Decoder
              │   (실수 I/O, 복소 conjugate-pair 상태)
SSM 커널 ─────┤
              ├─ QRS9CondMixBlock ── QRS9DecoderLayer ── QRS9Decoder
              │   (RS9 + 양자화)
              │
              └─ QARS9CondMixBlock ─ QARS9DecoderLayer ─ QARS9Decoder
                  (ARS9 + 양자화)
```

별도로, 복소 도메인 인코더:
```
S9Stack  (복소 I/O, DOST 전처리 필수)
```

## 선택 기준

### RS9 vs ARS9

| | RS9 | ARS9 |
|---|---|---|
| 내부 상태 도메인 | $\mathbb{R}$ (순실수) | $\mathbb{C}$ (복소 conjugate-pair) |
| 표현 가능 동역학 | 지수 감쇠만 | 지수 감쇠 + **진동** |
| DOST 필요 여부 | 불필요 | 불필요 |
| 파라미터 수 | $N$ 상태 | $N/2$ 쌍 × 실부/허부 → 동등 |
| 적합한 경우 | 단조 감쇠가 지배적인 신호 | 주기적 패턴, 텍스처, 음성 등 진동이 중요한 신호 |

**권장**: 특별한 이유가 없으면 **ARS9를 기본으로** 사용하라. RS9가 표현 가능한 동역학은 ARS9의 부분집합이다(허수부를 0으로 학습하면 RS9와 동등). 초기화 모드는 `"hippo_n"`을 추천한다.

### Q 계열 (QRS9 / QARS9)

| | 비양자화 (RS9/ARS9) | 양자화 (QRS9/QARS9) |
|---|---|---|
| 학습 시 | 풀정밀 fp32 | QAT (fake quantization + STE) |
| 추론 시 | 풀정밀 | 커널 캐시 + int8/int4 정적 양자화 |
| A, Δ 정밀도 | fp32 | **fp32 유지** (양자화하지 않음) |
| B, C 정밀도 | fp32 | int8 (기본) |
| 출력 선형층 | fp32 | int4 (기본) |
| 극점 안정성 검증 | 없음 | `|A_bar| < 1 - ε` 자동 체크 |

**권장**: 학습 단계에서는 비양자화 계열(RS9/ARS9)로 우선 수렴을 확인한 뒤, 배포 시 Q 계열로 전환하라. `QuantConfig`의 bit-budget을 필요에 따라 조정할 수 있다.

## 초기화 모드 (`init_mode`)

| `init_mode` | 사용 가능 계열 | 설명 |
|---|---|---|
| `"legacy"` (기본) | 전체 | v0.3.x 호환. $\text{Re}(\lambda) = -0.5$, $\text{Im}(\lambda) = \pi k / N$ |
| `"hippo_n"` | ARS9, QARS9, S9 | HiPPO-N 대각화. $\text{Im}(\lambda_n) = \pi(n + \frac{1}{2})$, $B_n = \sqrt{2n+1}$ |
| `"s4d_real"` | RS9, QRS9 | S4D-Real. $A_n = -(n+1)/2$. 장거리 모노토닉 decay |

잘못된 조합(예: RS9에 `"hippo_n"`)은 `ValueError`로 차단된다.

## 이산화 모드 (`discretization`)

| `discretization` | 수식 | 비고 |
|---|---|---|
| `"zoh"` (기본) | $\bar B = \frac{e^{A \Delta t} - 1}{A} B$ | Exact Zero-Order Hold. 해상도/스케일 강건 |
| `"approx"` | $\bar B = B \cdot \Delta t$ | v0.3.x 호환. 1차 근사 |

v0.3.x 체크포인트를 v0.4.0에서 사용하려면 `framesmoothie.migration`을 통해 파라미터를 재매핑하거나, `discretization="approx"`를 명시해야 한다. 자세한 내용은 `docs/MIGRATION_v0.4.0.md`를 참조.

## 사용 예시

### 기본 ARS9 블록 (권장)

```python
from framesmoothie.blocks_ars9 import ARS9CondMixBlock

block = ARS9CondMixBlock(
    c_model=64,
    q_dim=32,
    spatial_dims=2,
    gate_dim=32,
    init_mode="hippo_n",
    # discretization="zoh" 는 기본값이므로 생략 가능
)

x = torch.randn(B, H, W, 64)  # channel-last
q = torch.randn(B, K, 32)
out = block(x, q)
# out["q"]: [B, K, 32]
# out["mask_logits"]: [B, K, H, W]
```

### 양자화된 ARS9 디코더

```python
from s9.quantization import QuantConfig
from framesmoothie.decoder_qars9 import QARS9DecoderLayer, QARS9Decoder

layers = [
    QARS9DecoderLayer(
        c_model=64,
        q_dim=32,
        spatial_dims=2,
        gate_dim=32,
        init_mode="hippo_n",
        quant_config=QuantConfig(w_bits_B=8, w_bits_output=4),
    )
    for _ in range(4)
]
decoder = QARS9Decoder(layers)
```

### v0.3.x 호환 모드

```python
from framesmoothie.blocks import RS9CondMixBlock

block = RS9CondMixBlock(
    c_model=64,
    q_dim=32,
    spatial_dims=2,
    init_mode="legacy",
    discretization="approx",
)
```

## 모듈 위치 요약

| 파일 | 클래스 |
|---|---|
| `framesmoothie.blocks` | `RS9CondMixBlock` |
| `framesmoothie.decoder` | `RS9DecoderLayer`, `RS9Decoder` |
| `framesmoothie.blocks_ars9` | `ARS9CondMixBlock` |
| `framesmoothie.decoder_ars9` | `ARS9DecoderLayer`, `ARS9Decoder` |
| `framesmoothie.blocks_qrs9` | `QRS9CondMixBlock` |
| `framesmoothie.decoder_qrs9` | `QRS9DecoderLayer`, `QRS9Decoder` |
| `framesmoothie.blocks_qars9` | `QARS9CondMixBlock` |
| `framesmoothie.decoder_qars9` | `QARS9DecoderLayer`, `QARS9Decoder` |
| `framesmoothie.model` | `S9Stack` (복소 인코더) |
| `framesmoothie.migration` | `migrate_framesmoothie_checkpoint` 등 |
