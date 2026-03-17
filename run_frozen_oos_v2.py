"""Frozen OOS v2 -- YAML 기반 동결 파라미터 + Trade-level Replay.

v1 대비 변경:
1. frozen_params_v3_2.yaml에서 파라미터 로드 (하드코딩 제거)
2. 완전 신규 OOS 기간 분리 (v1 OOS와 겹치지 않음)
3. 코인별 + 합산 EV, MDD, 비용 분해
4. 피처 제거/레짐 필터 YAML 기반 적용
"""

import sys
sys.path.insert(0, "C:/Users/RJ/Desktop/CLAUDE_CRYPTO_AGENT")
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None

import json
import time
import warnings
import numpy as np
import pandas as pd
import yaml
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, asdict

warnings.filterwarnings("ignore")

from src.data.crawlers.crypto_ohlcv import fetch_all_top10
from src.data.crawlers.macro_commodity_crawler import crawl_all_macro_data
from src.models.masking_loop import (
    create_labels_triple_barrier, compute_extended_metrics,
    LABEL_MAP, HORIZONS,
)
from src.models.enhanced_ensemble import EnhancedEnsemble
from src.execution.cost_model import CostModel, FeeSchedule, FundingConfig, MissFillConfig, ExitType
from src.utils.config import bar_minutes as cfg_bar_minutes
from sklearn.feature_selection import mutual_info_classif

# ==================== YAML Config ====================

YAML_PATH = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("config/frozen_params_v3_2.yaml")
REPORT_DIR = Path(f"data/reports/frozen_oos_{YAML_PATH.stem}")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

with open(YAML_PATH, encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

VERSION = CFG["version"]
COMMON = CFG["common"]
COIN_PARAMS = CFG["coins"]
COST_CFG = CFG["cost_model"]
EXCLUDED_KW = CFG["excluded_feature_keywords"]
BLOCKED_REGIMES = CFG["blocked_regimes"]
ACTIVE_COINS = list(COIN_PARAMS.keys())

BM = cfg_bar_minutes()
MAX_HORIZON = COMMON["max_horizon"]
RISK_FRAC = COMMON["risk_frac"]
K_UPPER = COMMON["k_upper"]
K_LOWER = COMMON["k_lower"]
MIN_BARRIER_PCT = COMMON["min_barrier_pct"]
N_JOBS = 6

COST_MODEL = CostModel(
    fee_schedule=FeeSchedule(
        maker_fee=COST_CFG["maker_fee"],
        taker_fee=COST_CFG["taker_fee"],
        slippage_entry=COST_CFG["slippage_entry"],
        slippage_exit_limit=COST_CFG["slippage_exit_limit"],
        slippage_exit_market=COST_CFG["slippage_exit_market"],
    ),
    funding_config=FundingConfig(
        interval_hours=COST_CFG["funding_interval_hours"],
        default_rate=COST_CFG["funding_default_rate"],
    ),
    miss_fill_config=MissFillConfig(
        reject_prob=COST_CFG["miss_fill_reject_prob"],
        missed_ev_pct=COST_CFG["miss_fill_missed_ev"],
    ),
)

# OOS 기간 설정
# v1 OOS: 2026-02-03 ~ 2026-03-17 (42일)
# v2 OOS: 가장 최근 8주를 2개 블록으로 분리
#   Block A (validation): -8w ~ -4w (이전 OOS 일부 포함 가능하나 별도 검증)
#   Block B (holdout):    -4w ~ now  (완전 신규)
OOS_BLOCK_A_DAYS = 28  # 4주 (validation)
OOS_BLOCK_B_DAYS = 28  # 4주 (holdout, 완전 신규)
PURGE_BARS = MAX_HORIZON * 2
EMBARGO_BARS = 6


# ==================== Trade Record ====================

@dataclass
class Trade:
    coin: str
    side: str
    entry_idx: int
    entry_time: str
    entry_price: float
    tp_price: float
    sl_price: float
    atr: float
    exit_idx: int = -1
    exit_time: str = ""
    exit_price: float = 0.0
    exit_type: str = ""
    holding_bars: int = 0
    gross_pnl_pct: float = 0.0
    gross_pnl_eq: float = 0.0
    # Cost breakdown
    cost_entry_fee: float = 0.0
    cost_exit_fee: float = 0.0
    cost_slippage: float = 0.0
    cost_funding: float = 0.0
    cost_miss_fill: float = 0.0
    cost_total: float = 0.0
    net_pnl_eq: float = 0.0
    regime: str = "UNKNOWN"
    block: str = ""  # "A" or "B"


# ==================== Core Functions ====================

def fetch_data():
    print(f"\n{'='*70}")
    print(f"  DATA COLLECTION")
    print(f"{'='*70}")
    ohlcv = fetch_all_top10("365d", "1h")
    first = list(ohlcv.values())[0]
    macro = crawl_all_macro_data(first.index)
    macro_al = macro.get("aligned", pd.DataFrame())

    data = {}
    for coin in ACTIVE_COINS:
        if coin not in ohlcv:
            continue
        df = ohlcv[coin].copy()
        if len(macro_al) > 0:
            for col in macro_al.columns:
                df[col] = macro_al[col].reindex(df.index).ffill().bfill().fillna(0)
        data[coin] = df
        print(f"  {coin}: {len(df)} bars, {len(df.columns)} cols")
    return data


def split_blocks(df):
    """Split into train / OOS-A / OOS-B."""
    idx = df.index
    end = idx[-1]
    b_start = end - timedelta(days=OOS_BLOCK_B_DAYS)
    a_start = b_start - timedelta(days=OOS_BLOCK_A_DAYS)
    purge_td = timedelta(hours=(PURGE_BARS + EMBARGO_BARS) * (BM // 60))
    train_end = a_start - purge_td

    train = df[idx <= train_end]
    block_a = df[(idx >= a_start) & (idx < b_start)]
    block_b = df[idx >= b_start]

    return train, block_a, block_b


def is_excluded(col):
    col_l = col.lower()
    return any(kw in col_l for kw in EXCLUDED_KW)


def get_regime(df, i):
    """Simple regime at bar i."""
    if i < 42:
        return "UNKNOWN"
    close = df["close"].values[max(0, i-41):i+1]
    ema_f = pd.Series(close).ewm(span=10).mean().iloc[-1]
    ema_s = pd.Series(close).ewm(span=30).mean().iloc[-1]
    ret_std = np.std(np.diff(close) / close[:-1]) if len(close) > 1 else 0

    if ema_f > ema_s * 1.005:
        return "TREND_UP"
    elif ema_f < ema_s * 0.995:
        return "TREND_DOWN"
    else:
        return "RANGE_HIGH" if ret_std > 0.02 else "RANGE_LOW"


def _coin_k(coin, key):
    """Get coin-specific k override or common default."""
    params = COIN_PARAMS[coin]
    override_key = f"{key}_override"
    if override_key in params:
        return params[override_key]
    return COMMON[key]


def train_model(train_df, coin):
    """Train S1+S2 with frozen params from YAML."""
    params = COIN_PARAMS[coin]
    h = HORIZONS[-1]
    h_label = f"label_{h*BM}min"
    coin_k_upper = _coin_k(coin, "k_upper")
    coin_k_lower = _coin_k(coin, "k_lower")

    labeled = create_labels_triple_barrier(
        train_df.copy(), h, k_upper_override=coin_k_upper,
        k_lower_override=coin_k_lower, verbose=False)

    if h_label not in labeled.columns:
        return None

    exclude = {"label", "future_return", "open", "high", "low", "close", "volume"}
    for hh in HORIZONS:
        exclude.add(f"label_{hh*BM}min")
        exclude.add(f"return_{hh*BM}min")
    leak_kw = ["future", "target", "label_", "return_", "fwd_", "forward_"]

    feature_cols = [c for c in labeled.columns
                    if c not in exclude
                    and not any(kw in c.lower() for kw in leak_kw)
                    and not is_excluded(c)
                    and labeled[c].dtype in [np.float64, np.float32, np.int64, np.int32, float, int]]

    max_feat = params["max_features"]
    clean = labeled.replace([np.inf, -np.inf], np.nan).ffill().bfill()
    X = clean[feature_cols].fillna(0).values
    y = clean[h_label].fillna(1).values.astype(int)

    if len(feature_cols) > max_feat:
        n_mi = min(2000, len(X))
        mi = mutual_info_classif(X[:n_mi], y[:n_mi],
                                 discrete_features=False, random_state=42, n_neighbors=5)
        top_idx = np.argsort(mi)[-max_feat:]
        feature_cols = [feature_cols[i] for i in sorted(top_idx)]
        X = clean[feature_cols].fillna(0).values

    # S1
    y_s1 = (y != LABEL_MAP["HOLD"]).astype(int)
    if len(np.unique(y_s1)) < 2:
        return None
    s1_counts = np.bincount(y_s1, minlength=2)
    s1_sw = np.where(s1_counts > 0, len(y_s1) / (2 * s1_counts + 1e-10), 1.0)[y_s1]
    s1 = EnhancedEnsemble(n_classes=2, use_stacking=True, n_jobs=N_JOBS, verbose=False)
    s1.fit(X, y_s1, sample_weight=s1_sw)

    # S2
    trade_mask = y != LABEL_MAP["HOLD"]
    s2 = None
    if trade_mask.sum() >= 30:
        y_s2 = (y[trade_mask] == LABEL_MAP["UP"]).astype(int)
        if len(np.unique(y_s2)) >= 2:
            s2_counts = np.bincount(y_s2, minlength=2)
            s2_sw = np.where(s2_counts > 0, len(y_s2) / (2 * s2_counts + 1e-10), 1.0)[y_s2]
            s2 = EnhancedEnsemble(n_classes=2, use_stacking=True, n_jobs=N_JOBS, verbose=False)
            s2.fit(X[trade_mask], y_s2, sample_weight=s2_sw)

    return {"s1": s1, "s2": s2, "features": feature_cols, "n_train": len(X)}


def simulate_block(oos_df, model, coin, block_name):
    """Bar-by-bar trade-level replay on OOS block."""
    params = COIN_PARAMS[coin]
    s1_model = model["s1"]
    s2_model = model["s2"]
    feature_cols = model["features"]

    oos_clean = oos_df.replace([np.inf, -np.inf], np.nan).ffill().bfill()
    X = oos_clean[feature_cols].fillna(0).values

    s1_probs = s1_model.predict_proba(X)
    s1_pred = (s1_probs[:, 1] >= params["stage1_threshold"]).astype(int)

    s2_pred = np.zeros(len(X), dtype=int)
    s2_prob = np.full(len(X), 0.5)
    if s2_model is not None:
        s2_probs = s2_model.predict_proba(X)
        s2_pred = np.argmax(s2_probs, axis=1)
        s2_prob = s2_probs[:, 1]

    close = oos_df["close"].values
    high = oos_df["high"].values
    low = oos_df["low"].values
    times = oos_df.index
    n = len(close)

    if "atr_14" in oos_df.columns:
        atr = oos_df["atr_14"].values
    else:
        tr = np.maximum(high - low,
                        np.maximum(np.abs(high - np.roll(close, 1)),
                                   np.abs(low - np.roll(close, 1))))
        tr[0] = high[0] - low[0]
        atr = pd.Series(tr).rolling(14, min_periods=1).mean().values

    trades = []
    next_avail = 0

    for i in range(n - MAX_HORIZON):
        if i < next_avail:
            continue
        if s1_pred[i] != 1:
            continue

        # Regime filter
        regime = get_regime(oos_df, i)
        coin_blocked = COIN_PARAMS.get(coin, {}).get("blocked_regimes_override", BLOCKED_REGIMES)
        if regime in coin_blocked:
            continue

        entry = close[i]
        side = "BUY" if s2_pred[i] == 1 else "SELL"
        cur_atr = atr[i] if not np.isnan(atr[i]) else entry * 0.01

        coin_ku = _coin_k(coin, "k_upper")
        coin_kl = _coin_k(coin, "k_lower")
        u_dist = max(coin_ku * cur_atr, MIN_BARRIER_PCT * entry)
        l_dist = max(coin_kl * cur_atr, MIN_BARRIER_PCT * entry)

        if side == "BUY":
            tp, sl = entry + u_dist, entry - l_dist
        else:
            tp, sl = entry - u_dist, entry + l_dist

        # Barrier scan
        exit_type, exit_bar, exit_price = None, -1, 0.0
        for j in range(i + 1, min(i + MAX_HORIZON + 1, n)):
            if side == "BUY":
                h_tp, h_sl = high[j] >= tp, low[j] <= sl
            else:
                h_tp, h_sl = low[j] <= tp, high[j] >= sl

            if h_tp and h_sl:
                exit_type, exit_bar, exit_price = "stop_loss", j, sl
                break
            elif h_tp:
                exit_type, exit_bar, exit_price = "take_profit", j, tp
                break
            elif h_sl:
                exit_type, exit_bar, exit_price = "stop_loss", j, sl
                break

        if exit_type is None:
            exit_bar = min(i + MAX_HORIZON, n - 1)
            exit_type, exit_price = "time_stop", close[exit_bar]

        hold = exit_bar - i

        # PnL
        if side == "BUY":
            gpnl_pct = (exit_price - entry) / entry
        else:
            gpnl_pct = (entry - exit_price) / entry

        stop_dist_pct = max(l_dist / entry, 0.003)
        not_ratio = RISK_FRAC / stop_dist_pct
        gpnl_eq = gpnl_pct * not_ratio

        # Cost breakdown
        exit_enum = {"take_profit": ExitType.TAKE_PROFIT,
                     "stop_loss": ExitType.STOP_LOSS,
                     "time_stop": ExitType.TIME_STOP}[exit_type]

        cost = COST_MODEL.estimate_trade_cost(
            entry_price=entry, sl_price=sl, tp_price=tp,
            risk_frac=RISK_FRAC, exit_type=exit_enum,
            holding_bars=hold, bar_minutes=BM, entry_is_maker=True,
        )

        trades.append(Trade(
            coin=coin, side=side, entry_idx=i,
            entry_time=str(times[i]), entry_price=round(entry, 6),
            tp_price=round(tp, 6), sl_price=round(sl, 6),
            atr=round(cur_atr, 6),
            exit_idx=exit_bar, exit_time=str(times[exit_bar]),
            exit_price=round(exit_price, 6), exit_type=exit_type,
            holding_bars=hold,
            gross_pnl_pct=round(gpnl_pct, 6),
            gross_pnl_eq=round(gpnl_eq, 6),
            cost_entry_fee=round(cost.entry_fee_eq, 6),
            cost_exit_fee=round(cost.exit_fee_eq, 6),
            cost_slippage=round(cost.slippage_eq, 6),
            cost_funding=round(cost.funding_eq, 6),
            cost_miss_fill=round(cost.miss_fill_eq, 6),
            cost_total=round(cost.total_eq, 6),
            net_pnl_eq=round(gpnl_eq - cost.total_eq, 6),
            regime=regime, block=block_name,
        ))
        next_avail = exit_bar + 1

    return trades


def compute_metrics(trades, label=""):
    """Compute comprehensive metrics from trade list."""
    n = len(trades)
    if n == 0:
        return {"label": label, "trades": 0, "verdict": "NO TRADES"}

    pnls = [t.net_pnl_eq for t in trades]
    gross = [t.gross_pnl_eq for t in trades]
    costs = [t.cost_total for t in trades]
    holds = [t.holding_bars for t in trades]

    tp = sum(1 for t in trades if t.exit_type == "take_profit")
    sl = sum(1 for t in trades if t.exit_type == "stop_loss")
    ts = sum(1 for t in trades if t.exit_type == "time_stop")

    avg_net = np.mean(pnls)
    total_net = np.sum(pnls)
    total_gross = np.sum(gross)
    total_cost = np.sum(costs)

    # MDD
    eq = np.cumsum(pnls)
    pk = np.maximum.accumulate(eq)
    mdd = abs(np.min(eq - pk))

    # Sharpe
    if n > 1 and np.std(pnls) > 0:
        avg_h = np.mean(holds)
        sharpe = np.mean(pnls) / np.std(pnls) * np.sqrt(252 * 6 / max(avg_h, 1))
    else:
        sharpe = 0

    # CI
    se = np.std(pnls) / np.sqrt(n)
    ci_lo, ci_hi = avg_net - 1.96 * se, avg_net + 1.96 * se

    # Cost breakdown totals
    cost_entry = sum(t.cost_entry_fee for t in trades)
    cost_exit = sum(t.cost_exit_fee for t in trades)
    cost_slip = sum(t.cost_slippage for t in trades)
    cost_fund = sum(t.cost_funding for t in trades)
    cost_miss = sum(t.cost_miss_fill for t in trades)

    # Direction
    longs = sum(1 for t in trades if t.side == "BUY")
    shorts = n - longs

    # Regime
    regime_stats = {}
    for r in set(t.regime for t in trades):
        rt = [t for t in trades if t.regime == r]
        rp = [t.net_pnl_eq for t in rt]
        regime_stats[r] = {
            "n": len(rt), "avg": round(np.mean(rp), 6), "total": round(np.sum(rp), 6)
        }

    verdict = "PASS" if avg_net > 0 and ci_lo > -0.002 and n >= 10 else \
              "PASS (low N)" if avg_net > 0 and n >= 5 else \
              "MARGINAL" if avg_net > 0 else "FAIL"

    return {
        "label": label,
        "trades": n, "longs": longs, "shorts": shorts,
        "tp": tp, "sl": sl, "ts": ts,
        "win_rate": round(tp / n, 4),
        "avg_net_pnl": round(avg_net, 6),
        "total_net_pnl": round(total_net, 6),
        "total_gross_pnl": round(total_gross, 6),
        "mdd": round(mdd, 6),
        "sharpe": round(sharpe, 2),
        "ci_lo": round(ci_lo, 6), "ci_hi": round(ci_hi, 6),
        "avg_hold": round(np.mean(holds), 1),
        "cost_total": round(total_cost, 6),
        "cost_entry_fee": round(cost_entry, 6),
        "cost_exit_fee": round(cost_exit, 6),
        "cost_slippage": round(cost_slip, 6),
        "cost_funding": round(cost_fund, 6),
        "cost_miss_fill": round(cost_miss, 6),
        "cost_share": round(total_cost / (abs(total_gross) + 1e-10), 4),
        "regime": regime_stats,
        "verdict": verdict,
    }


def generate_report(all_results):
    """Generate markdown report."""
    lines = [
        f"# Frozen OOS v2 Report",
        f"",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Params: {YAML_PATH} (v{VERSION})",
        f"OOS Block A: validation ({OOS_BLOCK_A_DAYS}d), Block B: holdout ({OOS_BLOCK_B_DAYS}d)",
        f"Features excluded: {len(EXCLUDED_KW)} keywords",
        f"Regimes blocked: {BLOCKED_REGIMES}",
        f"",
        "---",
        "",
        "## 1. Per-Coin Results",
        "",
    ]

    for coin in ACTIVE_COINS:
        cr = all_results.get(coin, {})
        lines.append(f"### {coin}")
        lines.append("")
        lines.append("| Block | Trades | Win% | Avg Net | Total | MDD | Sharpe | 95% CI | Verdict |")
        lines.append("|-------|--------|------|---------|-------|-----|--------|--------|---------|")
        for block in ["A", "B", "A+B"]:
            m = cr.get(block, {})
            if m.get("trades", 0) == 0:
                lines.append(f"| {block} | 0 | - | - | - | - | - | - | NO TRADES |")
            else:
                lines.append(f"| {block} | {m['trades']} | {m['win_rate']:.1%} | "
                             f"{m['avg_net_pnl']:+.4%} | {m['total_net_pnl']:+.4%} | "
                             f"{m['mdd']:.4%} | {m['sharpe']:.1f} | "
                             f"[{m['ci_lo']:+.4%},{m['ci_hi']:+.4%}] | {m['verdict']} |")
        lines.append("")

        # Cost breakdown
        m = cr.get("A+B", {})
        if m.get("trades", 0) > 0:
            lines.append(f"**Cost breakdown**: entry={m['cost_entry_fee']:.4%} "
                         f"exit={m['cost_exit_fee']:.4%} slip={m['cost_slippage']:.4%} "
                         f"fund={m['cost_funding']:.4%} miss={m['cost_miss_fill']:.4%} "
                         f"| total={m['cost_total']:.4%} ({m['cost_share']:.1%} of gross)")
            lines.append(f"**Direction**: {m['longs']}L / {m['shorts']}S")
            lines.append("")

    # Portfolio
    lines += ["## 2. Portfolio (All Coins Combined)", ""]
    lines.append("| Block | Trades | Avg Net | Total | MDD | Sharpe | Verdict |")
    lines.append("|-------|--------|---------|-------|-----|--------|---------|")
    for block in ["A", "B", "A+B"]:
        m = all_results.get("portfolio", {}).get(block, {})
        if m.get("trades", 0) == 0:
            lines.append(f"| {block} | 0 | - | - | - | - | NO TRADES |")
        else:
            lines.append(f"| {block} | {m['trades']} | {m['avg_net_pnl']:+.4%} | "
                         f"{m['total_net_pnl']:+.4%} | {m['mdd']:.4%} | "
                         f"{m['sharpe']:.1f} | {m['verdict']} |")
    lines.append("")

    # Regime
    lines += ["## 3. Regime Analysis (All Coins, A+B)", ""]
    pm = all_results.get("portfolio", {}).get("A+B", {})
    rs = pm.get("regime", {})
    if rs:
        lines.append("| Regime | Trades | Avg Net | Total |")
        lines.append("|--------|--------|---------|-------|")
        for r, s in sorted(rs.items(), key=lambda x: -x[1]["total"]):
            lines.append(f"| {r} | {s['n']} | {s['avg']:+.4%} | {s['total']:+.4%} |")

    report = "\n".join(lines)
    with open(REPORT_DIR / "frozen_oos_v2_report.md", "w", encoding="utf-8") as f:
        f.write(report)
    return report


# ==================== MAIN ====================

def main():
    start = datetime.now()
    print(f"\n{'='*70}")
    print(f"  FROZEN OOS v2 -- YAML-based Trade-level Replay")
    print(f"  {start.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Params: {YAML_PATH} (v{VERSION})")
    print(f"  Coins: {ACTIVE_COINS}")
    print(f"  OOS: Block A ({OOS_BLOCK_A_DAYS}d) + Block B ({OOS_BLOCK_B_DAYS}d)")
    print(f"  Excluded features: {len(EXCLUDED_KW)} keywords")
    print(f"  Blocked regimes: {BLOCKED_REGIMES}")
    print(f"{'='*70}")

    feature_data = fetch_data()

    all_results = {}
    all_trades = []

    for coin in ACTIVE_COINS:
        if coin not in feature_data:
            continue

        print(f"\n{'='*50}")
        print(f"  {coin}")
        print(f"{'='*50}")

        df = feature_data[coin]
        train, block_a, block_b = split_blocks(df)
        print(f"  Train:   {len(train)} bars ({train.index[0]} ~ {train.index[-1]})")
        print(f"  Block A: {len(block_a)} bars ({block_a.index[0]} ~ {block_a.index[-1]})")
        print(f"  Block B: {len(block_b)} bars ({block_b.index[0]} ~ {block_b.index[-1]})")

        t0 = time.time()
        model = train_model(train, coin)
        if model is None:
            print(f"  [FAIL] Training failed")
            continue
        print(f"  Trained in {time.time()-t0:.1f}s ({model['n_train']} samples, "
              f"{len(model['features'])} features)")

        # Simulate both blocks
        trades_a = simulate_block(block_a, model, coin, "A")
        trades_b = simulate_block(block_b, model, coin, "B")
        trades_ab = trades_a + trades_b

        metrics_a = compute_metrics(trades_a, f"{coin}_A")
        metrics_b = compute_metrics(trades_b, f"{coin}_B")
        metrics_ab = compute_metrics(trades_ab, f"{coin}_A+B")

        all_results[coin] = {"A": metrics_a, "B": metrics_b, "A+B": metrics_ab}
        all_trades.extend(trades_ab)

        # Print summary
        for block, m in [("A", metrics_a), ("B", metrics_b), ("A+B", metrics_ab)]:
            if m["trades"] > 0:
                print(f"  Block {block}: {m['trades']} trades, "
                      f"avg={m['avg_net_pnl']:+.4%}, total={m['total_net_pnl']:+.4%}, "
                      f"MDD={m['mdd']:.4%}, verdict={m['verdict']}")
            else:
                print(f"  Block {block}: 0 trades")

    # Portfolio metrics
    print(f"\n{'='*50}")
    print(f"  PORTFOLIO (ALL COINS)")
    print(f"{'='*50}")

    portfolio = {}
    for block in ["A", "B", "A+B"]:
        block_trades = [t for t in all_trades if (block == "A+B" or t.block == block)]
        pm = compute_metrics(block_trades, f"portfolio_{block}")
        portfolio[block] = pm
        if pm["trades"] > 0:
            print(f"  {block}: {pm['trades']} trades, avg={pm['avg_net_pnl']:+.4%}, "
                  f"total={pm['total_net_pnl']:+.4%}, MDD={pm['mdd']:.4%}, "
                  f"Sharpe={pm['sharpe']:.1f}, verdict={pm['verdict']}")

    all_results["portfolio"] = portfolio

    # Save
    with open(REPORT_DIR / "frozen_oos_v2_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Trade CSVs
    for coin in ACTIVE_COINS:
        coin_trades = [t for t in all_trades if t.coin == coin]
        if coin_trades:
            pd.DataFrame([asdict(t) for t in coin_trades]).to_csv(
                REPORT_DIR / f"trades_{coin}.csv", index=False)

    # All trades combined
    if all_trades:
        pd.DataFrame([asdict(t) for t in all_trades]).to_csv(
            REPORT_DIR / "trades_all.csv", index=False)

    # Report
    generate_report(all_results)

    elapsed = (datetime.now() - start).total_seconds() / 60
    print(f"\n{'='*70}")
    print(f"  COMPLETE ({elapsed:.1f} min)")
    print(f"  Reports: {REPORT_DIR}/")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
