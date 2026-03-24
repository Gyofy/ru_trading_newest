# v6.1 Parameter Optimization Report

> 2026-03-24 | 전체 파라미터 검토 + 과대평가 보정

## 테스트된 파라미터

| 파라미터 | 범위 | 최적 | 근거 |
|----------|------|------|------|
| **ku (TP)** | 3, 4, 5, 7 | **5.0** | v5.1 grid 6,480개 검색 결과 |
| **kl (SL)** | 1.0, 1.5, 2.0, 2.5, 3.0 | **2.0** | WR 64%, DOGE SL 문제 해결 |
| **mh (TTL)** | 18, 24, 36, 48, inf | **24** (96 1h) | TTL 6건 전부 양수, 수익의 57% |
| **dfd (DIR_FLIP)** | 3, 6, 12, 24, off | **6** | ptS 0.211→0.251 (+19%) |
| **cq (CVD Q)** | 0.55~0.90 | **0.75** | 거래 빈도/품질 균형 |
| Trailing stop | tested | 불채택 | 4h 노이즈에 역효과 |
| Scale-out | tested | 불채택 | avg 감소 |
| GARCH SL | tested | 불채택 | ATR과 동일 |
| ML filter | tested | 불채택 | CV 0.543 (near random) |
| RS sizing | tested | 불채택 | equal이 최적 |

## DIR_FLIP delay 상세

| dfd | OOS 거래 | WR | ptS | DIR_FLIP 건수 |
|-----|---------|-----|-----|-------------|
| 3 (이전) | 42 | 50.0% | 0.211 | 20 |
| **6 (적용)** | **38** | **47.4%** | **0.251** | **16** |
| 12 | 37 | 45.9% | 0.161 | 10 |
| off | 35 | 51.4% | 0.195 | 0 |

dfd=6: 불필요한 DIR_FLIP 4건 제거 → ptS +19%

## TTL 기여도

```
TP:       7건 (17%), avg +9.81%, 총 +68.7%
SL:       9건 (21%), avg -3.93%, 총 -35.4%
DIR_FLIP: 20건 (48%), avg -0.69%, 총 -13.8%
TTL:      6건 (14%), avg +4.35%, 총 +26.1%  ← 전부 양수
```

TTL 제거 시 avg +1.09% → +0.54% (절반)

## 과대평가 보정

| 항목 | 보정 전 | 보정 후 |
|------|--------|--------|
| 10코인 상관관계 | Sharpe 4.88 | **ptS 0.208** (8.6x 보정) |
| 다중비교 (6,480 config) | best ptS 0.504 | Bonferroni 보정 필요 |
| IS/OOS 일관성 | v5.3 IS 음수 | **v6 IS+OOS 둘 다 양수** |

## 최종 config (v6.1)

```yaml
lookback_bars: 168      # 7d × 24 (1h)
cvd_quantile: 0.75
cvd_roll_window: 480    # 120 4h × 4
k_upper: 5.0
k_lower: 2.0
max_hold_bars: 96       # 24 4h × 4
dir_flip_delay: 6       # 6 1h-bars (이전 3에서 변경)
oi_zscore_max: 2.0
leverage: 2
max_positions: 2
```
