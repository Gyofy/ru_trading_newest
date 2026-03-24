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

## Current State (2026-03-24)

### Active: v4.3-1m ML Bot — LIVE 실거래 중
- **Bot**: `run_live_bot_v2.py` (LIVE 모드, 실거래 진행 중)
- **Config**: `config/frozen_params_v4_3_1m.yaml`
- **Coins**: BTC, ETH, SOL, XRP, ADA, DOT, LINK (7코인)
- **Timeframe**: 1분봉, 1000 bars
- **Equity**: ~65 USDT (Binance USDT-M Futures)
- **Settings**: 일일 손실 -10% halt, 포지션당 자본 10%, Post-Only→Limit 폴백

### 2026-03-24 주요 변경사항
- Binance 공개 데이터(OI/L&S/Taker/Funding) 피처 통합
- Post-Only -5022 에러 시 limit 주문 자동 폴백
- 7코인 확장 (SOL 단일 → BTC/ETH/SOL/XRP/ADA/DOT/LINK)
- Discord 봇 종료 알림 추가

### Suspended
- v5.1 TSMOM paper bot (`run_tsmom_paper.py`) — 별도 전략
- RL meta-layer — shadow mode 유지, 데이터 축적 중

## Devlog Files

| Date | File | Summary |
|------|------|---------|
| **2026-03-24** | *(README.md 참조)* | **7코인 LIVE 실거래, Binance 피처 통합, Post-Only 폴백** |
| 2026-03-20 | [devlog/2026-03-20.md](devlog/2026-03-20.md) | Feature leakage discovery, strategy pivot, BTC spike bot |
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
