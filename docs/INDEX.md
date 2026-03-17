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

## Current State (2026-03-17)

### Active Config
- **Production params**: `config/frozen_params_v3_4.yaml` (FROZEN)
- **Live promotion criteria**: `config/live_promotion_criteria.yaml`
- **Evaluation**: trade-level bar-by-bar simulation (not summary EV)
- **Feature policy**: `src/utils/feature_policy.py`

### Active Coins
- **DOT**: PASS (OOS +18.72%, Sharpe 8.6)
- **ADA**: PASS (OOS +9.26%, cost 8.8%)
- **XRP**: SUSPENDED (S1 signal scarcity)

### Running Processes
- Paper Trading v3.4: DOT+ADA, 2026-03-17 ~ 03-31

### Paper Sim Results (500M KRW, 8w)
- 41 trades, +29.48% (+1,474,153 KRW), MDD 2.94%

## Devlog Files

| Date | File | Summary |
|------|------|---------|
| 2026-03-17 | [devlog/2026-03-17.md](devlog/2026-03-17.md) | v3.1->v3.4 evolution, OOS validation, strategy diagnosis |

## Checkpoints

| ID | File | Description |
|----|------|-------------|
| CP-001 | [checkpoints/cp001_v3_4_frozen.md](checkpoints/cp001_v3_4_frozen.md) | v3.4 final config frozen |

## Brainstorm

| Topic | File | Status |
|-------|------|--------|
| (none yet) | | |

## Key Reports

| Report | Path |
|--------|------|
| v3.1 Final Report | `data/reports/v3_1_netev_final_report.md` |
| Development History | `data/reports/DEVELOPMENT_HISTORY.md` |
| Frozen OOS v3.4 | `data/reports/frozen_oos_frozen_params_v3_4/` |
| Strategy Diagnosis | `data/reports/strategy_diagnosis/` |
| Paper Sim (5M KRW) | Run `run_paper_sim_5m.py` |

## How to Use

1. **New session**: Read `docs/INDEX.md` first
2. **After changes**: Update relevant devlog + INDEX.md
3. **Major decisions**: Create checkpoint in `docs/checkpoints/`
4. **Ideas/experiments**: Write to `docs/brainstorm/`
