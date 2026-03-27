# CLAUDE_CRYPTO_AGENT -- Development Memory Index

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

- **Bot**: `run_live_bot_v2.py` (LIVE 모드)
- **Config**: `config/frozen_params_v4_3_1m.yaml`
- **Coins**: SOL, XRP, ADA, DOT (4코인 — 최소주문 5 USDT 이하만)
- **Timeframe**: 1분봉, 1000 bars
- **Equity**: ~65 USDT (Binance USDT-M Futures)
- **Position**: 자본의 10% / 일일 손실 -10% halt

### 2026-03-24 주요 변경사항

| 변경 | 내용 |
|------|------|
| Binance 공개 데이터 통합 | OI/L&S/Taker/Funding → compute_features() |
| Post-Only 폴백 | -5022 → limit 주문 자동 재시도 |
| 거래 코인 | SOL 단일 → SOL/XRP/ADA/DOT 4코인 |
| 포지션 크기 | 자본 10% notional cap |
| 일일 손실 한도 | -10% halt |
| Discord 종료 알림 | 봇 종료 시 Discord embed 발송 |
| --yes 플래그 | 백그라운드 실행 시 확인 우회 |

### Parallel: v5.1 TSMOM Paper Bot (별도 전략)

- **Bot**: `run_tsmom_paper.py` (paper trading)
- **Strategy**: Dual TSMOM(7d+28d) + RSI + CVD + OI 필터
- **Coins**: 10코인 (BTC ETH SOL XRP ADA DOT LINK DOGE AVAX BNB)
- **OOS Sharpe**: 4.03 (permutation p=0.006)
- **Status**: 별도 운영 (v4.3-1m LIVE와 독립)

---

## Devlog Files

| Date | File | Summary |
|------|------|---------|
| **2026-03-24** | *(README.md 참조)* | 7코인→4코인 LIVE, Binance 피처 통합, Post-Only 폴백 |
| 2026-03-23 | *(README.md v5.1 섹션)* | v5.1 TSMOM 10코인, OOS Sharpe 4.03 |
| 2026-03-20 | [devlog/2026-03-20.md](devlog/2026-03-20.md) | Feature leakage discovery, BTC spike bot |
| 2026-03-18 | [devlog/2026-03-18.md](devlog/2026-03-18.md) | v4.1 config, Mega Search v1, Frozen OOS |
| 2026-03-17 | [devlog/2026-03-17.md](devlog/2026-03-17.md) | v3.1→v3.4 evolution, OOS validation |

## Config History

| Version | File | Key Changes |
|---------|------|-------------|
| **v4.3-1m** | `config/frozen_params_v4_3_1m.yaml` | **현재 운영**, 1분봉, 4코인 |
| v4.2 | `config/frozen_params_v4_2.yaml` | Mega Search v2 best, 5코인 |
| v4.1 | `config/frozen_params_v4_1.yaml` | 5코인, ET primary, microstructure ON |
| v3.4 | `config/frozen_params_v3_4.yaml` | Production baseline (DOT+ADA) |

---

## ⛔ 실행 절대 원칙: TP/SL FOR POSITION 필수

**어떤 포지션도 SL과 TP 없이 존재할 수 없다. 예외 없음.**

- SL: `STOP_MARKET` + `closePosition=True`
- TP: `TAKE_PROFIT_MARKET` + `closePosition=True`
- fill 후 반드시 **1초 대기** 후 등록 (Binance position propagation delay — -4509 방지)
- 5회 retry 실패 시 **즉시 강제청산** (naked position = 절대 금지)
- `positions.json`의 `sl_exchange_id` / `tp_exchange_id` 둘 다 비어있으면 위반

> 상세: `CLAUDE.md` → "⛔ 절대 원칙: 모든 포지션에 TP/SL FOR POSITION 필수"

---

## How to Use

1. **New session**: Read `docs/INDEX.md` first
2. **After changes**: Update relevant devlog + INDEX.md
3. **Major decisions**: Create checkpoint in `docs/checkpoints/`
4. **Ideas/experiments**: Write to `docs/brainstorm/`
