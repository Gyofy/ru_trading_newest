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

## Current State (2026-03-18)

### Active Config
- **Production params**: `config/frozen_params_v4_1.yaml` (v4.1 -- 5-coin expansion)
- **Previous params**: `config/frozen_params_v4_0.yaml` (v4.0, superseded)
- **Live promotion criteria**: `config/live_promotion_criteria.yaml`
- **Evaluation**: trade-level bar-by-bar simulation (not summary EV)
- **Feature policy**: `src/utils/feature_policy.py`

### Active Coins (v4.1 -- 5 coins)
- **DOT**: ET solo, th=0.45 (Mega Search v1 confirmed, +27.15% total PnL)
- **ADA**: ET+TabPFN 70/30, th=0.40 (Mega Search v1 confirmed, +21.87% total PnL)
- **XRP**: ET+TabM 70/30, th=0.45 (Mega Search v1 best, +51.54% total PnL)
- **SOL**: ET solo (default), th=0.50 (pending Mega Search v2 results)
- **LINK**: ET solo (default), th=0.50 (pending Mega Search v2 results)

### Running Processes
- **Mega Search v2**: SOL/LINK model combo optimization (in progress)
- **Frozen OOS v4.1**: validation framework prepared, awaiting v2 completion

### Key v4.1 Changes (from v4.0)
- ExtraTrees as primary base model (Mega Search v1 finding)
- Coin-specific model combos (ET solo / ET+TabPFN / ET+TabM)
- 5-coin expansion (DOT, ADA, XRP + SOL, LINK)
- Microstructure features enabled (CVD filter, OFI timing)

### Paper Sim Results (500M KRW, 8w, v3.4 baseline)
- 41 trades, +29.48% (+1,474,153 KRW), MDD 2.94%

## Devlog Files

| Date | File | Summary |
|------|------|---------|
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
