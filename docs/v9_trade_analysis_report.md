# v9.0 전체 트레이딩 기록 분석 & 전략 재설계 리포트

**작성일**: 2026-03-29
**분석 범위**: 214건 (v4.3 LIVE 45건 + v8.0+ Demo 169건)
**결론**: 2전략 체제(CVD Extreme + Cascade Fade), SL floor 0.40%, 고정 TP, 일일 20건 cap

---

## 1. 데이터 소스 및 범위

### 1.1 데이터 파일 위치

| 소스 | 기간 | 건수 | 모드 | 파일 |
|------|------|------|------|------|
| v4.3 LIVE | 03-18 ~ 03-24 | 45건 | LIVE ($65) | `trading_result/events.jsonl` |
| v8.0+ Demo | 03-27 ~ 03-29 | 169건 | Demo ($4,609) | `data/reports/multi_strategy/trades.jsonl` |
| GitHub | Gyofy/ru_trading_newest | 동기화 완료 | HEAD: `2d3a666` |

### 1.2 데이터 공백
- 03-25 ~ 03-26: 거래 없음 (v7.0→v8.0 설계 기간)
- btc_spike_paper: 8건 PnL $0 (미작동)
- tsmom_paper: 빈 파일

---

## 2. 전략별 성적 분석 (v8.0+ 169건)

### 2.1 전체 요약

```
총 거래: 169건 (3일)
Gross PnL: +$32.20
수수료: $1,360.42
Net PnL: -$1,328.22
승률: 36.1% (61/169)
```

### 2.2 전략별 상세

| 전략 | 건수 | WR | Gross PnL | 수수료 | Net PnL | Sharpe | PF |
|------|------|-----|-----------|--------|---------|--------|-----|
| **liquidation_fade** | 35 | 57.1% | +$234.50 | $221.85 | **+$12.65** | 2.14 | 1.28 |
| asymmetric_sniper | 38 | 44.7% | +$7.51 | $613.71 | -$606.20 | 0.83 | 1.04 |
| cvd_spike | 84 | 27.4% | -$132.41 | $490.58 | -$622.99 | -4.74 | 0.55 |
| momentum_breakout | 12 | 8.3% | -$77.40 | $34.27 | -$111.68 | -20.67 | 0.00 |

### 2.3 TP/SL 분석

| 전략 | TP 히트 | SL 히트 | 기타 | TP 비율 |
|------|---------|---------|------|---------|
| liquidation_fade | **13건** | 11건 | 11건 | **37%** |
| asymmetric_sniper | 0건 | 26건 | 12건 | 0% |
| cvd_spike | **0건** | 80건 | 4건 | **0%** |
| momentum_breakout | 0건 | 12건 | 0건 | 0% |

**핵심**: cvd_spike 84건, sniper 38건에서 TP 도달 0건. 고정 TP를 쓴 liq_fade만 37% TP 히트.

### 2.4 SL 거리 분석

| 전략 | SL 평균 | SL 중앙값 | 수수료(0.18%) 대비 |
|------|---------|-----------|-------------------|
| asymmetric_sniper | 0.100% | 0.081% | **0.56배 (수수료 미만!)** |
| cvd_spike | 0.167% | 0.129% | 0.93배 (거의 동일) |
| liquidation_fade | 0.377% | 0.356% | 2.10배 (적정) |
| momentum_breakout | 0.296% | 0.276% | 1.64배 |

---

## 3. MFE/MAE 심층 분석

### 3.1 전략별 MFE/MAE

| 전략 | avg MFE | avg MAE | MFE>|MAE| 비율 | WIN MFE | LOSS MFE |
|------|---------|---------|----------------|---------|----------|
| liquidation_fade | **0.475%** | -0.268% | **60.0%** | 0.675% | 0.208% |
| cvd_spike | 0.195% | -0.134% | 48.8% | **0.407%** | 0.115% |
| asymmetric_sniper | 0.075% | -0.063% | 47.4% | 0.159% | **0.007%** |
| momentum_breakout | 0.113% | -0.277% | 25.0% | 0.659% | 0.063% |

### 3.2 해석

- **sniper 패배 MFE 0.007%**: 가격이 거의 유리하게 움직이지 않고 즉시 SL → SL이 noise 범위 내
- **cvd_spike WIN MFE 0.407%**: 방향은 맞지만 trailing이 breakeven으로 당겨진 후 noise에 청산
- **liq_fade**: 유일하게 MFE>|MAE| 60%+ → 진입 후 유리한 방향으로 더 많이 움직임

### 3.3 보유시간 분석

| 전략 | WIN avg bars | LOSS avg bars | 해석 |
|------|-------------|---------------|------|
| liquidation_fade | 73.9 | 36.1 | 승리 거래가 2배 오래 보유 (정상) |
| cvd_spike | 17.8 | 9.5 | 패배 거래가 빨리 끝남 (SL tight) |
| asymmetric_sniper | 5.9 | 3.7 | 전체적으로 너무 짧음 |

---

## 4. 시간대/방향/코인 분석

### 4.1 시간대 (UTC)

| 시간대 | 건수 | WR | PnL | 판정 |
|--------|------|-----|-----|------|
| 01:00~08:00 | 13 | 84.6% | +$103 | 수익 구간 |
| 11:00~16:00 | 126 | 28.6% | -$257 | **대량 손실 구간** |
| 17:00~18:00 | 19 | 57.9% | +$111 | 수익 구간 |

### 4.2 방향 편향

| 전략 | BUY WR | BUY PnL | SELL WR | SELL PnL |
|------|--------|---------|---------|----------|
| asymmetric_sniper | **71.4%** | +$33 | 38.7% | -$45 |
| cvd_spike | 18.2% | **-$125** | 37.5% | -$65 |
| liquidation_fade | 58.8% | +$178 | 55.6% | +$65 |

- sniper SHORT 편향 손실, cvd_spike LONG 역추세 특히 나쁨

### 4.3 코인 상위/하위

| 코인 | 건수 | WR | PnL | 핵심 전략 |
|------|------|-----|-----|----------|
| **AVAX** | 9 | 55.6% | **+$162** | liq_fade 3건 +$136 |
| DOT | 14 | 42.9% | +$59 | 다전략 |
| ADA | 11 | 54.5% | +$22 | sniper +$15 |
| LINK | 10 | **0.0%** | **-$118** | 전 전략 전패 |
| DOGE | 21 | 28.6% | -$77 | cvd_spike -$43 |
| ARB | 14 | 28.6% | -$76 | cvd_spike -$39 |

### 4.4 중복/이상 거래
- 동일 시각 동일 가격 중복 거래: OP cvd_spike 2건, AVAX liq_fade 2건
- 3초 이내 동시 진입: 2건 감지
- cvd_spike 동일 코인 최소 갭: 0.0분 (OP), 1.3분 (ADA) → 쿨다운 부재

---

## 5. v4.3 LIVE 거래 분석 (45건)

### 5.1 코인별

| 코인 | 건수 | avg PnL% | 주요 사유 |
|------|------|----------|----------|
| ADA | 5 | -5.49% | SL_HIT 4, EXCHANGE_CLOSED 1 |
| SOL | 32 | +0.10% | SL_HIT 22, TP_HIT 4 |
| DOT | 5 | -0.37% | SL_HIT 3 |
| LINK | 1 | -1.16% | SL_HIT 1 |
| XRP | 2 | -0.003% | SL_HIT 1, KILL_SWITCH 1 |

### 5.2 핵심 교훈
- ADA SL 0.10% 거리 → 4연패 → KILL_SWITCH
- SOL 03-23: 20건 소액 반복매매 → TP 4건/SL 16건 = WR 20% (과거래)
- TP 미설정 포지션 존재 → v8.0에서 FOR POSITION 필수로 해결

---

## 6. 구조적 문제 진단 (3라운드 비판적 사고)

### Round 1 발견
1. trailing 자체가 아닌 **breakeven 진입 조건이 너무 빠름**이 근본 원인
2. 시간대 필터는 3일 표본으로 통계적 무의미
3. cvd_spike WIN MFE 0.407% = 신호는 유효, TP 구조만 문제

### Round 2 발견
4. **1분봉 ATR이 너무 작음** = SL 절대값이 0.1~0.2%밖에 안 되는 진짜 이유
5. cvd_spike와 sniper는 같은 CVD 신호 → **중복 전략, 수수료 2배**
6. R:R 0.75:1이면 WR 60% 필요 → 비현실적 → **R:R 2.0+ 필수**

### Round 3 발견
7. 15m ATR로 SL 계산하면 보유시간이 자연스럽게 증가 → **position size 조정 필요**
8. liq_fade 50% 배분은 AVAX 3건(56% 수익) 의존 → margin 얇음
9. **100건+ 검증 후 live 승급**이 안전 (50건은 부족)

---

## 7. v9.0 설계 및 구현

### 7.1 전략 구조

```
v9.0 Multi-Strategy Bot (2전략 체제)
├── CVD Extreme (50%, $2,500, 3x)
│   ├── 신호: CVD Q95 + Z-score 2.0σ (듀얼 조건)
│   ├── SL: 15m ATR × 2.0 (min 0.40%)
│   ├── TP: SL × 2.5 (고정, trailing 없음)
│   ├── 쿨다운: 전략 3분, 코인 SL 2회→1시간
│   └── 일일 10건 cap
│
└── Cascade Fade (50%, $2,500, 5x)
    ├── 신호: OI 급감 + Taker spike (검증된 설정 유지)
    ├── SL: 5m ATR × 3.5 (min 0.40%)
    ├── TP: ATR × 7.5 (고정)
    └── 파라미터 변경 없음
```

### 7.2 코드 변경 요약

| 파일 | 변경 |
|------|------|
| `src/strategies/cvd_extreme.py` | **신규** — 통합 전략 (190 LOC) |
| `src/strategies/base.py` | SL floor: `max(fee×2.5, 0.40%)` |
| `src/strategies/asymmetric_sniper.py` | 동일 SL floor (비활성이지만 정합성) |
| `config/multi_strategy.yaml` | v9.0 2전략 체제 |
| `run_multi_strategy.py` | CVDExtreme 등록, daily cap 20, checkpoint, SL callback |

### 7.3 SL Floor 메커니즘

```python
# base.py line 242-253
SL_FLOOR_PCT = 0.004   # 0.40% absolute minimum
SL_FEE_MULT = 2.5      # SL >= 2.5× round-trip fee
min_sl_dist = max(price * ROUND_TRIP_FEE_RATE * SL_FEE_MULT,  # 0.45%
                  price * SL_FLOOR_PCT)                         # 0.40%
# → 실질 SL floor = 0.45% (fee×2.5가 0.40%보다 큼)
```

### 7.4 CVD Extreme 15m ATR 스케일링

```python
# cvd_extreme.py compute_barriers()
atr_15m = atr_1m * (15 ** 0.5)  # √15 ≈ 3.87
sl_dist = atr_15m * sl_mult     # 2.0 × ~3.87 × ATR_1m
sl_dist = max(sl_dist, price * 0.004)  # 0.40% floor
tp_dist = sl_dist * tp_rr       # 2.5 × SL (fixed)
```

### 7.5 EV 시뮬레이션

**CVD Extreme** (보수적):
```
WR=45%, SL=0.45%, TP=1.13% (R:R 2.5:1), Fee=0.18%
Win: +1.13% - 0.18% = +0.95%
Loss: -0.45% - 0.18% = -0.63%
EV = 0.45 × 0.95 - 0.55 × 0.63 = +0.081%/trade ✓
```

**Cascade Fade** (실측):
```
WR=50%, SL=0.38%, TP=0.75% (R:R ~2:1), Fee=0.18%
Win: +0.75% - 0.18% = +0.57%
Loss: -0.38% - 0.18% = -0.56%
EV = 0.50 × 0.57 - 0.50 × 0.56 = +0.005%/trade ✓ (얇음)
```

---

## 8. 검증 계획

| 마일스톤 | 기준 | 액션 |
|----------|------|------|
| 50건 | 자동 checkpoint | 전략별 WR/PF 초기 검증 |
| 100건 | Live 승급 검토 가능 | Net PnL > 0, MDD < 30% |
| 150건 | 파라미터 미세 조정 | TP R:R, SL mult 조정 |
| 200건 | 확정 판단 | Live 승급 또는 v10.0 재설계 |

### 핵심 KPI
- CVD Extreme WR >= 45%
- 양 전략 Net PnL > 0
- Bot daily trades <= 20
- MDD < 30%

---

*v9.0 리포트 작성: 2026-03-29 | 분석 기반: 214건 전체 거래 기록*
