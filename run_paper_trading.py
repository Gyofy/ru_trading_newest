"""Paper Trading -- DOT + XRP with Frozen Params.

4h bar close마다:
1. 최신 OHLCV fetch
2. Frozen model로 S1/S2 prediction
3. Open position 관리 (TP/SL/TTL check)
4. Trade log 기록

Parameters: FROZEN (v3.1_netev R38) -- 절대 수정 금지.
XRP threshold=0.6은 OOS에서 0건이었으나 동결 유지.
"""

import sys
sys.path.insert(0, "C:/Users/RJ/Desktop/CLAUDE_CRYPTO_AGENT")
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None

import json
import time
import warnings
import traceback
import numpy as np
import pandas as pd
from datetime import datetime, timedelta, timezone
from pathlib import Path
from dataclasses import dataclass, field, asdict

warnings.filterwarnings("ignore")

from src.data.crawlers.crypto_ohlcv import fetch_all_top10
from src.data.crawlers.macro_commodity_crawler import crawl_all_macro_data
from src.models.masking_loop import (
    create_labels_triple_barrier, LABEL_MAP, HORIZONS,
)
from src.models.enhanced_ensemble import EnhancedEnsemble
from src.execution.cost_model import CostModel, FeeSchedule, FundingConfig, MissFillConfig, ExitType
from src.utils.config import load_settings, bar_minutes as cfg_bar_minutes
from sklearn.feature_selection import mutual_info_classif
from src.utils.feature_policy import is_excluded_feature, is_blocked_regime

# ==================== 설정 ====================
REPORT_DIR = Path("data/reports/paper_trading")
REPORT_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = REPORT_DIR / "paper_trading_log.jsonl"

BM = cfg_bar_minutes()
MAX_HORIZON = max(HORIZONS)
RISK_FRAC = 0.005
N_JOBS = 6
INITIAL_EQUITY = 10000  # $10,000 USD

PAPER_COINS = ["DOT", "XRP"]
POLL_MINUTES = 30  # 30분마다 체크
PAPER_DURATION_DAYS = 14  # 2주

COST_MODEL = CostModel(
    fee_schedule=FeeSchedule(
        maker_fee=0.0002, taker_fee=0.00055,
        slippage_entry=0.0003, slippage_exit_limit=0.0001,
        slippage_exit_market=0.0005,
    ),
    funding_config=FundingConfig(interval_hours=8.0, default_rate=0.0001),
    miss_fill_config=MissFillConfig(reject_prob=0.15, missed_ev_pct=0.0015),
)

# 동결 파라미터 (v3.1_netev R38)
FROZEN_PARAMS = {
    "XRP": {
        "k_upper": 3.0, "k_lower": 0.6, "stage1_threshold": 0.46,
        "max_features": 120, "num_leaves": 47, "learning_rate": 0.02,
        "n_estimators": 100, "max_depth_tree": 6, "subsample": 0.8,
        "colsample": 0.6, "min_child_samples": 30,
    },
    "DOT": {
        "k_upper": 3.0, "k_lower": 0.6, "stage1_threshold": 0.5,
        "max_features": 80, "num_leaves": 47, "learning_rate": 0.1,
        "n_estimators": 300, "max_depth_tree": 8, "subsample": 0.7,
        "colsample": 0.8, "min_child_samples": 10,
    },
}


# ==================== Position Tracking ====================

@dataclass
class PaperPosition:
    coin: str
    side: str  # BUY or SELL
    entry_price: float
    entry_time: str
    tp_price: float
    sl_price: float
    atr: float
    ttl_bars: int
    bars_held: int = 0
    status: str = "OPEN"  # OPEN, CLOSED_TP, CLOSED_SL, CLOSED_TTL

@dataclass
class PaperTrade:
    coin: str
    side: str
    entry_price: float
    entry_time: str
    exit_price: float
    exit_time: str
    exit_type: str
    holding_bars: int
    gross_pnl_pct: float
    net_pnl_eq: float
    cost_eq: float
    s1_prob: float
    s2_prob: float


class PaperTrader:
    def __init__(self):
        self.equity = INITIAL_EQUITY
        self.positions = {}  # coin -> PaperPosition
        self.trades = []
        self.signals_total = {c: 0 for c in PAPER_COINS}
        self.signals_triggered = {c: 0 for c in PAPER_COINS}
        self.models = {}  # coin -> (s1_model, s2_model, feature_cols)
        self.last_train_time = None
        self.bars_seen = 0

    def log(self, event: dict):
        """Append event to JSONL log."""
        event["timestamp"] = datetime.now().isoformat()
        event["equity"] = round(self.equity, 2)
        with open(LOG_FILE, "a") as f:
            f.write(json.dumps(event, default=str) + "\n")

    def train_models(self, feature_data: dict):
        """Train frozen models for each coin."""
        for coin in PAPER_COINS:
            if coin not in feature_data:
                continue
            params = FROZEN_PARAMS[coin]
            df = feature_data[coin]

            h = HORIZONS[-1]
            h_label = f"label_{h*BM}min"

            labeled = create_labels_triple_barrier(
                df.copy(), h, k_upper_override=params["k_upper"],
                k_lower_override=params["k_lower"], verbose=False)

            if h_label not in labeled.columns:
                print(f"    [SKIP] {coin}: labeling failed")
                continue

            # Feature selection
            exclude = {"label", "future_return", "open", "high", "low", "close", "volume"}
            for hh in HORIZONS:
                exclude.add(f"label_{hh*BM}min")
                exclude.add(f"return_{hh*BM}min")
            leak_kw = ["future", "target", "label_", "return_", "fwd_", "forward_"]

            feature_cols = [c for c in labeled.columns
                           if c not in exclude
                           and not any(kw in c.lower() for kw in leak_kw)
                           and not is_excluded_feature(c)
                           and labeled[c].dtype in [np.float64, np.float32, np.int64, np.int32, float, int]]

            max_feat = params["max_features"]
            if len(feature_cols) > max_feat:
                X_mi = labeled[feature_cols].replace([np.inf, -np.inf], 0).fillna(0).values
                y_mi = labeled[h_label].values
                n_mi = min(2000, len(X_mi))
                mi = mutual_info_classif(X_mi[:n_mi], y_mi[:n_mi],
                                         discrete_features=False, random_state=42, n_neighbors=5)
                top_idx = np.argsort(mi)[-max_feat:]
                feature_cols = [feature_cols[i] for i in sorted(top_idx)]

            clean = labeled.replace([np.inf, -np.inf], np.nan).ffill().bfill()
            X = clean[feature_cols].fillna(0).values
            y_3c = clean[h_label].fillna(1).values.astype(int)

            # S1
            y_s1 = (y_3c != LABEL_MAP["HOLD"]).astype(int)
            if len(np.unique(y_s1)) < 2:
                continue
            s1_counts = np.bincount(y_s1, minlength=2)
            s1_sw = np.where(s1_counts > 0, len(y_s1) / (2 * s1_counts + 1e-10), 1.0)[y_s1]

            ens_s1 = EnhancedEnsemble(n_classes=2, use_stacking=True, n_jobs=N_JOBS, verbose=False)
            ens_s1.fit(X, y_s1, sample_weight=s1_sw)

            # S2
            trade_mask = y_3c != LABEL_MAP["HOLD"]
            ens_s2 = None
            if trade_mask.sum() >= 30:
                y_s2 = (y_3c[trade_mask] == LABEL_MAP["UP"]).astype(int)
                if len(np.unique(y_s2)) >= 2:
                    s2_counts = np.bincount(y_s2, minlength=2)
                    s2_sw = np.where(s2_counts > 0, len(y_s2) / (2 * s2_counts + 1e-10), 1.0)[y_s2]
                    ens_s2 = EnhancedEnsemble(n_classes=2, use_stacking=True, n_jobs=N_JOBS, verbose=False)
                    ens_s2.fit(X[trade_mask], y_s2, sample_weight=s2_sw)

            self.models[coin] = (ens_s1, ens_s2, feature_cols)
            print(f"    {coin}: model trained ({len(X)} samples, {len(feature_cols)} features)")

        self.last_train_time = datetime.now()

    def check_signal(self, coin: str, latest_bar: pd.Series, all_features: pd.DataFrame) -> dict:
        """Check if current bar generates a trade signal."""
        if coin not in self.models:
            return None

        params = FROZEN_PARAMS[coin]
        s1_model, s2_model, feature_cols = self.models[coin]

        # Extract features for latest bar
        bar_features = all_features[feature_cols].iloc[-1:].replace([np.inf, -np.inf], 0).fillna(0).values

        # S1 prediction
        s1_probs = s1_model.predict_proba(bar_features)
        s1_prob = float(s1_probs[0, 1])
        s1_pred = 1 if s1_prob >= params["stage1_threshold"] else 0

        self.signals_total[coin] += 1

        if s1_pred == 0:
            return None

        # S2 prediction
        s2_prob = 0.5
        s2_pred = 0
        if s2_model is not None:
            s2_probs = s2_model.predict_proba(bar_features)
            s2_pred = int(np.argmax(s2_probs[0]))
            s2_prob = float(s2_probs[0, 1])

        self.signals_triggered[coin] += 1

        return {
            "s1_prob": s1_prob,
            "s2_pred": s2_pred,
            "s2_prob": s2_prob,
            "side": "BUY" if s2_pred == 1 else "SELL",
        }

    def open_position(self, coin: str, signal: dict, bar: pd.Series):
        """Open a new paper position."""
        params = FROZEN_PARAMS[coin]
        entry_price = bar["close"]

        atr = bar.get("atr_14", entry_price * 0.01)
        if np.isnan(atr):
            atr = entry_price * 0.01

        upper_dist = max(params["k_upper"] * atr, 0.002 * entry_price)
        lower_dist = max(params["k_lower"] * atr, 0.002 * entry_price)

        if signal["side"] == "BUY":
            tp = entry_price + upper_dist
            sl = entry_price - lower_dist
        else:
            tp = entry_price - upper_dist
            sl = entry_price + lower_dist

        pos = PaperPosition(
            coin=coin, side=signal["side"],
            entry_price=entry_price,
            entry_time=str(bar.name),
            tp_price=tp, sl_price=sl,
            atr=atr, ttl_bars=MAX_HORIZON,
        )
        self.positions[coin] = pos

        self.log({
            "event": "OPEN", "coin": coin, "side": signal["side"],
            "entry": entry_price, "tp": round(tp, 6), "sl": round(sl, 6),
            "s1_prob": signal["s1_prob"], "s2_prob": signal["s2_prob"],
        })
        print(f"      [{coin}] OPEN {signal['side']} @ {entry_price:.4f} "
              f"TP={tp:.4f} SL={sl:.4f} s1={signal['s1_prob']:.2f} s2={signal['s2_prob']:.2f}")

    def check_exits(self, coin: str, bar: pd.Series):
        """Check if open position should be closed."""
        if coin not in self.positions:
            return
        pos = self.positions[coin]
        if pos.status != "OPEN":
            return

        pos.bars_held += 1
        high = bar["high"]
        low = bar["low"]
        close = bar["close"]

        exit_type = None
        exit_price = 0.0

        if pos.side == "BUY":
            hit_tp = high >= pos.tp_price
            hit_sl = low <= pos.sl_price
        else:
            hit_tp = low <= pos.tp_price
            hit_sl = high >= pos.sl_price

        if hit_tp and hit_sl:
            exit_type = "stop_loss"  # conservative
            exit_price = pos.sl_price
        elif hit_tp:
            exit_type = "take_profit"
            exit_price = pos.tp_price
        elif hit_sl:
            exit_type = "stop_loss"
            exit_price = pos.sl_price
        elif pos.bars_held >= pos.ttl_bars:
            exit_type = "time_stop"
            exit_price = close

        if exit_type:
            self._close_position(coin, exit_type, exit_price, str(bar.name))

    def _close_position(self, coin: str, exit_type: str, exit_price: float, exit_time: str):
        pos = self.positions[coin]
        params = FROZEN_PARAMS[coin]

        if pos.side == "BUY":
            gross_pnl_pct = (exit_price - pos.entry_price) / pos.entry_price
        else:
            gross_pnl_pct = (pos.entry_price - exit_price) / pos.entry_price

        stop_dist_pct = max(params["k_lower"] * pos.atr / pos.entry_price, 0.003)
        notional_ratio = RISK_FRAC / stop_dist_pct
        gross_pnl_eq = gross_pnl_pct * notional_ratio

        exit_type_enum = {
            "take_profit": ExitType.TAKE_PROFIT,
            "stop_loss": ExitType.STOP_LOSS,
            "time_stop": ExitType.TIME_STOP,
        }[exit_type]

        cost = COST_MODEL.estimate_trade_cost(
            entry_price=pos.entry_price, sl_price=pos.sl_price,
            tp_price=pos.tp_price, risk_frac=RISK_FRAC,
            exit_type=exit_type_enum, holding_bars=pos.bars_held,
            bar_minutes=BM, entry_is_maker=True,
        )
        net_pnl_eq = gross_pnl_eq - cost.total_eq
        self.equity *= (1 + net_pnl_eq)

        trade = PaperTrade(
            coin=coin, side=pos.side,
            entry_price=pos.entry_price, entry_time=pos.entry_time,
            exit_price=exit_price, exit_time=exit_time,
            exit_type=exit_type, holding_bars=pos.bars_held,
            gross_pnl_pct=round(gross_pnl_pct, 6),
            net_pnl_eq=round(net_pnl_eq, 6),
            cost_eq=round(cost.total_eq, 6),
            s1_prob=0, s2_prob=0,
        )
        self.trades.append(trade)
        pos.status = f"CLOSED_{exit_type.upper()}"
        del self.positions[coin]

        emoji = "W" if net_pnl_eq > 0 else "L"
        self.log({
            "event": "CLOSE", "coin": coin, "exit_type": exit_type,
            "exit_price": exit_price, "holding_bars": pos.bars_held,
            "gross_pnl_pct": round(gross_pnl_pct, 6),
            "net_pnl_eq": round(net_pnl_eq, 6),
            "cost_eq": round(cost.total_eq, 6),
        })
        print(f"      [{coin}] CLOSE {exit_type} @ {exit_price:.4f} "
              f"hold={pos.bars_held} net={net_pnl_eq:+.4%} [{emoji}] eq=${self.equity:.2f}")

    def summary(self) -> dict:
        """Generate paper trading summary."""
        n = len(self.trades)
        if n == 0:
            return {
                "total_trades": 0,
                "equity": self.equity,
                "return_pct": 0,
                "signals_total": self.signals_total,
                "signals_triggered": self.signals_triggered,
            }

        pnls = [t.net_pnl_eq for t in self.trades]
        wins = sum(1 for p in pnls if p > 0)

        by_coin = {}
        for coin in PAPER_COINS:
            coin_trades = [t for t in self.trades if t.coin == coin]
            coin_pnls = [t.net_pnl_eq for t in coin_trades]
            by_coin[coin] = {
                "trades": len(coin_trades),
                "wins": sum(1 for p in coin_pnls if p > 0),
                "total_pnl": round(sum(coin_pnls), 6) if coin_pnls else 0,
                "avg_pnl": round(np.mean(coin_pnls), 6) if coin_pnls else 0,
            }

        return {
            "total_trades": n,
            "wins": wins,
            "win_rate": round(wins / n, 4),
            "equity": round(self.equity, 2),
            "return_pct": round((self.equity / INITIAL_EQUITY - 1) * 100, 4),
            "max_equity": round(max(INITIAL_EQUITY, self.equity), 2),
            "by_coin": by_coin,
            "signals_total": self.signals_total,
            "signals_triggered": self.signals_triggered,
        }


# ==================== Regime Helper ====================

def _get_current_regime(df: pd.DataFrame) -> str:
    """Simple regime classification for latest bar."""
    if len(df) < 42:
        return "UNKNOWN"
    close = df["close"].values
    segment = close[-42:]
    ema_fast = pd.Series(segment).ewm(span=10).mean().iloc[-1]
    ema_slow = pd.Series(segment).ewm(span=30).mean().iloc[-1]
    ret_std = np.std(np.diff(segment) / segment[:-1])
    median_std = 0.02

    if ema_fast > ema_slow * 1.005:
        return "TREND_UP"
    elif ema_fast < ema_slow * 0.995:
        return "TREND_DOWN"
    else:
        return "RANGE_HIGH" if ret_std > median_std else "RANGE_LOW"


# ==================== Main Loop ====================

def fetch_and_prepare():
    """Fetch latest data and prepare features."""
    ohlcv = fetch_all_top10("365d", "1h")
    first_coin = list(ohlcv.values())[0]
    macro = crawl_all_macro_data(first_coin.index)
    macro_aligned = macro.get("aligned", pd.DataFrame())

    feature_data = {}
    for coin in PAPER_COINS:
        if coin not in ohlcv:
            continue
        df = ohlcv[coin].copy()
        if len(macro_aligned) > 0:
            for col in macro_aligned.columns:
                df[col] = macro_aligned[col].reindex(df.index).ffill().bfill().fillna(0)
        feature_data[coin] = df

    return feature_data


def main():
    start_time = datetime.now()
    end_time = start_time + timedelta(days=PAPER_DURATION_DAYS)

    print(f"\n{'='*70}")
    print(f"  PAPER TRADING -- DOT + XRP (Frozen Params)")
    print(f"  Started: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  End: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Initial equity: ${INITIAL_EQUITY:,.2f}")
    print(f"  Poll interval: {POLL_MINUTES} min")
    print(f"  XRP threshold=0.46 (adjusted from 0.60)")
    print(f"  Regime filter: RANGE_LOW blocked")
    print(f"{'='*70}")

    trader = PaperTrader()

    # Initial data + model training
    print(f"\n  [INIT] Fetching data and training models...")
    feature_data = fetch_and_prepare()
    trader.train_models(feature_data)
    trader.log({"event": "INIT", "coins": PAPER_COINS, "initial_equity": INITIAL_EQUITY})

    last_bar_time = {}
    retrain_interval = timedelta(hours=24)  # Retrain daily (data refresh)

    cycle = 0
    while datetime.now() < end_time:
        cycle += 1
        now = datetime.now()

        # Retrain daily (data refresh, same frozen params)
        if trader.last_train_time and (now - trader.last_train_time) > retrain_interval:
            print(f"\n  [RETRAIN] Daily data refresh...")
            try:
                feature_data = fetch_and_prepare()
                trader.train_models(feature_data)
            except Exception as e:
                print(f"  [ERR] Retrain failed: {e}")

        # Check each coin
        for coin in PAPER_COINS:
            if coin not in feature_data:
                continue

            df = feature_data[coin]
            latest_time = df.index[-1]

            # Only process new bars
            if coin in last_bar_time and latest_time <= last_bar_time[coin]:
                continue

            last_bar_time[coin] = latest_time
            latest_bar = df.iloc[-1]

            # Check exits first
            trader.check_exits(coin, latest_bar)

            # Check for new signal (only if no open position)
            if coin not in trader.positions:
                # Regime filter: block RANGE_LOW
                regime = _get_current_regime(df)
                if is_blocked_regime(regime):
                    continue

                signal = trader.check_signal(coin, latest_bar, df)
                if signal:
                    trader.open_position(coin, signal, latest_bar)

        # Periodic status (every 10 cycles)
        if cycle % 10 == 0:
            summary = trader.summary()
            elapsed_h = (now - start_time).total_seconds() / 3600
            remaining_h = (end_time - now).total_seconds() / 3600
            print(f"\n  [{now.strftime('%H:%M')}] cycle={cycle} "
                  f"elapsed={elapsed_h:.1f}h remain={remaining_h:.1f}h "
                  f"trades={summary['total_trades']} "
                  f"equity=${summary['equity']:,.2f} "
                  f"return={summary['return_pct']:+.2f}%")
            for coin in PAPER_COINS:
                bc = summary.get("by_coin", {}).get(coin, {})
                st = summary["signals_total"].get(coin, 0)
                sg = summary["signals_triggered"].get(coin, 0)
                print(f"    {coin}: {bc.get('trades', 0)} trades, "
                      f"signals {sg}/{st}, "
                      f"total_pnl={bc.get('total_pnl', 0):+.4%}")

        # Wait
        time.sleep(POLL_MINUTES * 60)

    # Final summary
    summary = trader.summary()
    print(f"\n{'='*70}")
    print(f"  PAPER TRADING COMPLETE")
    print(f"  Duration: {PAPER_DURATION_DAYS} days")
    print(f"  Total trades: {summary['total_trades']}")
    print(f"  Final equity: ${summary['equity']:,.2f}")
    print(f"  Return: {summary['return_pct']:+.2f}%")
    print(f"{'='*70}")

    for coin in PAPER_COINS:
        bc = summary.get("by_coin", {}).get(coin, {})
        print(f"  {coin}: {bc.get('trades', 0)} trades, "
              f"avg_pnl={bc.get('avg_pnl', 0):+.4%}, "
              f"total_pnl={bc.get('total_pnl', 0):+.4%}")

    # Save final report
    with open(REPORT_DIR / "paper_trading_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Save trade details
    if trader.trades:
        trades_df = pd.DataFrame([asdict(t) for t in trader.trades])
        trades_df.to_csv(REPORT_DIR / "paper_trades.csv", index=False)

    print(f"\n  Reports: {REPORT_DIR}/")


if __name__ == "__main__":
    main()
