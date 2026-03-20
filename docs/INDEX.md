# CLAUDE_CRYPTO_AGENT -- Development Memory Index

This directory serves as the persistent memory for development history,
checkpoints, and brainstorming notes. Every agent session should read
this index before starting work.

## Directory Structure

```
docs/
  INDEX.md              <-- THIS FILE (always read first)
  devlog/               <-- chronological development logs
  checkpoints/          <-- decision checkpoints & state snapshots
  brainstorm/           <-- ideas, experiments, design notes
```

## Current State (2026-03-20)

### CRITICAL: Feature Leakage Discovered (2026-03-20)
- STL decomposition, Ichimoku .shift(26), SVD interpolation → 미래 데이터 누수
- **S2 방향 예측: 0.687 → 0.518 (누수 제거 후 거의 랜덤)**
- **v4.0~v4.3 모든 백테스트 결과 무효**
- 15,120 조합 전수 검색 → BTC spike만 유일한 edge

### Active Strategy: BTC Spike → Alt Follow (v4.4)
- **Bot**: `run_btc_spike_paper.py` (paper trading, 백그라운드 실행 중)
- **Config**: `config/frozen_params_v4_3.yaml` (누수 수정 + rf=2%)
- **Coins**: SOL, ETH, XRP, ADA
- **Trigger**: BTC 1h |ret| > 1.2% + alt confirmation
- **Evidence**: 180d, n=276, WR 48.6%, avg +0.11% (하락장 -38.5%)

### Suspended
- v4.3 ML bot (`run_live_bot_v2.py`) — 누수 수정 적용됨, ML 방향 예측 신뢰 불가
- RL meta-layer — shadow mode 유지, 데이터 축적 중
- Mega Search v3 결과 — 누수 feature 기반이므로 무효

## Devlog Files

| Date | File | Summary |
|------|------|---------|
| **2026-03-20** | **[devlog/2026-03-20.md](devlog/2026-03-20.md)** | **Feature leakage discovery, strategy pivot, BTC spike bot** |
| 2026-03-18 | [devlog/2026-03-18.md](devlog/2026-03-18.md) | v4.1 config, Mega Search v1 results, Frozen OOS framework |
| 2026-03-17 | [devlog/2026-03-17.md](devlog/2026-03-17.md) | v3.1->v3.4 evolution, OOS validation, strategy diagnosis |

## Checkpoints

| ID | File | Description |
|----|------|-------------|
| CP-001 | [checkpoints/cp001_v3_4_frozen.md](checkpoints/cp001_v3_4_frozen.md) | v3.4 final config frozen |

## Brainstorm

| Topic | File | Status |
|-------|------|--------|
| v4 Production Blueprint | [brainstorm/v4_production_blueprint.md](brainstorm/v4_production_blueprint.md) | APPROVED |

## Key Reports

| Report | Path |
|--------|------|
| Mega Search v1 Results | `experiments/tabpfn_test/results/mega_search_final.json` |
| v3.1 Final Report | `data/reports/v3_1_netev_final_report.md` |
| Development History | `data/reports/DEVELOPMENT_HISTORY.md` |
| Frozen OOS v3.4 | `data/reports/frozen_oos_frozen_params_v3_4/` |
| Strategy Diagnosis | `data/reports/strategy_diagnosis/` |
| Paper Sim (5M KRW) | Run `run_paper_sim_5m.py` |

## Config History

| Version | File | Key Changes |
|---------|------|-------------|
| v4.1 | `config/frozen_params_v4_1.yaml` | 5-coin, ET primary, model combos, microstructure ON |
| v4.0 | `config/frozen_params_v4_0.yaml` | WF-v3.1 net-EV optimization, threshold tuning |
| v3.4 | `config/frozen_params_v3_4.yaml` | Production baseline (DOT+ADA) |
| v3.3 | `config/frozen_params_v3_3.yaml` | Intermediate |
| v3.2 | `config/frozen_params_v3_2.yaml` | Intermediate |

## How to Use

1. **New session**: Read `docs/INDEX.md` first
2. **After changes**: Update relevant devlog + INDEX.md
3. **Major decisions**: Create checkpoint in `docs/checkpoints/`
4. **Ideas/experiments**: Write to `docs/brainstorm/`
