# v5.1 브레인스토밍 — 데이터 기반 다음 단계

> 2026-03-24 | 실패 인정 후 재설계

---

## 1. 현재 상태: 숫자로 보는 진실

### 실전 결과
```
v4.3 실거래:  41건, WR 17%  → 실패 (미검증 1m봉 적용)
v5.1 paper:   0건            → 시그널 미발생 (필터 과다)
v4.4 spike:   8건, avg +0.11% → 약함
```

### 백테스트 vs 현실

| 전략 | 백테스트 | 현실 | 괴리 |
|------|---------|------|------|
| v4.3 ML | WR 54% | WR 17% | **-37%p** |
| v5.1 5중필터 | WR 60% | **0건** | **거래 불가** |

### 핵심 문제: 필터를 쌓을수록 Sharpe는 올라가지만 거래가 사라진다

```
필터 수    시그널 비율    거래/일    Sharpe    현실에서 가능?
1개        93.6%         42건       0.59     거래 多, edge 無
2개        70.9%         40건       1.28     거래 多, edge 약
3개         9.8%         10건       4.21     거래 적당, edge 有
4개         9.1%          9건       4.88     ← 최적점
5개(현재)   5.9%          6건       4.35     거래 부족
```

**4개 필터 (TSMOM 7d + RSI + CVD + OI)가 최적.** 5번째 필터(dual 28d agree)를 추가하면 Sharpe가 오히려 하락하고 거래가 40% 감소.

---

## 2. 왜 지금 시그널이 0건인가

```
10개 코인 전부:
  7일 TSMOM  = SHORT (-1)
  28일 TSMOM = ZERO (0)  ← 28일 전 가격 ≈ 현재 가격
  → dual_disagree → 전부 차단

28일 return이 0 근처에서 진동 중
→ TSMOM sign()이 0으로 나옴
→ dual agree가 절대 성립 안 됨
```

**이건 dual lookback의 구조적 약점.** 28일 return이 0 근처에 있으면 direction이 결정되지 않아 영원히 대기.

---

## 3. 데이터가 말하는 최적 전략

### 3.1 필터별 성과 비교 (OOS, 10코인)

| Config | 거래 | WR | Avg PnL | Sharpe | 일평균 거래 |
|--------|------|-----|---------|--------|------------|
| TSMOM only | 458 | 34.9% | +0.13% | 0.59 | 42 |
| + RSI | 432 | 38.0% | +0.30% | 1.28 | 40 |
| **+ RSI + CVD** | **104** | **54.8%** | **+2.33%** | **4.21** | **10** |
| **+ RSI + CVD + OI** | **96** | **58.3%** | **+2.84%** | **4.88** | **9** |
| + RSI + CVD + OI + dual | 61 | 60.7% | +3.57% | 4.35 | 6 |

**TSMOM 7d + RSI + CVD + OI (4중 필터)가 Sharpe 4.88로 최고.**
5번째 필터(dual)를 빼면 거래가 96건 → 61건 대비 57% 더 많고 Sharpe도 더 높음.

### 3.2 CVD Quantile 감도

| CVD Q | 거래 | WR | Avg PnL | Sharpe |
|-------|------|-----|---------|--------|
| 0.55 | 108 | 50.0% | +1.84% | 3.26 |
| 0.60 | 92 | 53.3% | +2.18% | 3.51 |
| **0.65** | **82** | **56.1%** | **+2.50%** | **3.77** |
| 0.70 | 72 | 58.3% | +2.80% | 3.80 |
| 0.75 | 61 | 60.7% | +3.57% | 4.35 |
| 0.80 | 42 | 66.7% | +4.16% | 4.17 |

CVD Q65로 완화하면 거래 82건 (34% 증가), Sharpe 3.77 (여전히 높음).

### 3.3 Single vs Dual TSMOM

| Config | 거래 | WR | Avg PnL | Sharpe |
|--------|------|-----|---------|--------|
| **single 7d + cq0.75** | **96** | **58.3%** | **+2.84%** | **4.88** |
| single 7d + cq0.65 | 143 | 48.3% | +1.53% | 3.42 |
| dual 7+28 + cq0.75 | 61 | 60.7% | +3.57% | 4.35 |

**single 7d가 dual보다 Sharpe 높고 거래 57% 많음.**

---

## 4. 브레인스토밍: 3가지 방향

### Option A: 4중 필터로 단순화 (가장 현실적)

```
변경: dual lookback 제거 → single 7d TSMOM
유지: RSI + CVD Q75 + OI
결과: 96건 거래 (vs 61건), Sharpe 4.88 (vs 4.35)
```

**장점:**
- 28d return = 0 문제 해소 → 시그널 즉시 발생
- OOS Sharpe가 오히려 상승 (4.35 → 4.88)
- 일 9건 시그널 → 충분한 거래 빈도

**단점:**
- 7d lookback은 노이즈에 민감
- 단기 whipsaw에 취약

### Option B: CVD 완화 (거래 빈도 극대화)

```
변경: CVD Q75 → Q65
유지: single 7d + RSI + OI
결과: 143건 거래, WR 48.3%, Sharpe 3.42
```

**장점:**
- 가장 많은 거래 (143건)
- 빠른 paper trading 검증 (50건 = ~5일)

**단점:**
- WR 48% (약간 아래)
- Sharpe 3.42 (낮아지지만 여전히 좋음)

### Option C: 현상 유지 + 대기

```
변경: 없음
대기: 28d return이 음수로 전환될 때까지
예상: 며칠~1주
```

**장점:**
- 백테스트 최적 config 유지

**단점:**
- 그 "며칠"이 "몇 주"가 될 수 있음
- 실전 검증 지연 → 시간 낭비

---

## 5. 냉정한 판단

### Option A가 정답인 이유

1. **데이터가 증명:** single 7d + RSI + CVD + OI = OOS Sharpe **4.88** (전체 최고)
2. **dual 28d는 해(害):** Sharpe를 낮추고 (4.88→4.35), 거래를 줄이고 (96→61), 현재 시그널을 차단
3. **dual을 넣은 이유가 "추세 전환점 방어"였는데**, 실제로는 추세 전환점에서 아무것도 못 하게 됨
4. **즉시 실행 가능:** paper bot config 한 줄 변경

### 왜 처음부터 single이 나았는데 dual을 선택했나

이전에 dual이 avg PnL +3.46% vs single +2.84%로 per-trade 수익이 높았기 때문.
하지만 **Sharpe는 거래 수를 반영**하므로 실제로는 single이 우월.
"per-trade PnL 최대화"에 매몰되어 "포트폴리오 Sharpe 최대화"를 놓친 오류.

---

## 6. 실행 계획

### 즉시 (오늘)

```
1. paper bot config: dual_lookback = False, lookback_days = 7
2. state 리셋, 단일 프로세스 재시작
3. git push
```

### 1주 내

```
4. 50건 paper trade 축적 (일 ~9건 × 6일)
5. 중간 WR/PnL 점검 (25건 시점)
6. WR > 40% + avg > 0 → 계속
   WR < 30% → 전략 재검토
```

### 2주 내

```
7. 50건 달성 → live promotion 평가
8. RL signal 200건 → offline training
9. 실거래 전환 여부 결정
```

---

## 7. 최종 Config (v5.1r — revised)

```yaml
strategy: TSMOM_v5.1_revised
lookback_days: 7           # single (dual 제거)
dual_lookback: false       # 핵심 변경
cvd_quantile: 0.75         # 유지 (Q65로 완화 가능)
cvd_roll_window: 120
k_upper: 5.0               # TP = 5 × ATR
k_lower: 1.5               # SL = 1.5 × ATR
max_hold_bars: 24
use_oi: true
oi_zscore_max: 2.0
leverage: 2
coins: 10                  # BTC ETH SOL XRP ADA DOT LINK DOGE AVAX BNB

Expected (OOS):
  trades: ~96 / 3.5months = ~1 trade/day
  WR: 58.3%
  avg PnL: +2.84%/trade
  Sharpe: 4.88
```
