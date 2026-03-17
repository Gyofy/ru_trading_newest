"""Paper Trading v3.4 -- DOT + ADA Forward Validation.

frozen_params_v3_4.yaml 기반.
코인별 레짐 정책, 피처 제거, live 승급 기준 사전 고정.

기록 항목:
- signal count / actual fill / reject
- realized slippage / cost share
- net EV/trade, MDD
- regime별 거래 빈도
- live_promotion_criteria 자동 판정
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
from collections import Counter

warnings.filterwarnings("ignore")

from src.data.crawlers.crypto_ohlcv import fetch_all_top10
from src.data.crawlers.macro_commodity_crawler import crawl_all_macro_data
from src.models.masking_loop import (
    create_labels_triple_barrier, LABEL_MAP, HORIZONS,
)
from src.models.enhanced_ensemble import EnhancedEnsemble
from src.execution.cost_model import CostModel, FeeSchedule, FundingConfig, MissFillConfig, ExitType
from src.utils.config import bar_minutes as cfg_bar_minutes
from sklearn.feature_selection import mutual_info_classif

# ==================== Config ====================
YAML_PATH = Path("config/frozen_params_v3_4.yaml")
CRITERIA_PATH = Path("config/live_promotion_criteria.yaml")
REPORT_DIR = Path("data/reports/paper_v3_4")
REPORT_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = REPORT_DIR / "paper_log.jsonl"

with open(YAML_PATH, encoding="utf-8") as f:
    CFG = yaml.safe_load(f)
with open(CRITERIA_PATH, encoding="utf-8") as f:
    CRITERIA = yaml.safe_load(f)

COMMON = CFG["common"]
COIN_PARAMS = CFG["coins"]
COST_CFG = CFG["cost_model"]
EXCLUDED_KW = CFG["excluded_feature_keywords"]
GLOBAL_BLOCKED = CFG["blocked_regimes"]
ACTIVE_COINS = list(COIN_PARAMS.keys())

BM = cfg_bar_minutes()
MAX_HORIZON = COMMON["max_horizon"]
RISK_FRAC = COMMON["risk_frac"]
N_JOBS = 6
INITIAL_EQUITY = 10000
POLL_MINUTES = 30
PAPER_DAYS = 14

COST_MODEL = CostModel(
    fee_schedule=FeeSchedule(
        maker_fee=COST_CFG["maker_fee"], taker_fee=COST_CFG["taker_fee"],
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


def coin_k(coin, key):
    override = f"{key}_override"
    if override in COIN_PARAMS.get(coin, {}):
        return COIN_PARAMS[coin][override]
    return COMMON[key]


def coin_blocked_regimes(coin):
    params = COIN_PARAMS.get(coin, {})
    return params.get("blocked_regimes_override", GLOBAL_BLOCKED)


def is_excluded(col):
    cl = col.lower()
    return any(kw in cl for kw in EXCLUDED_KW)


def get_regime(df, i):
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


# ==================== Tracker ====================

@dataclass
class PaperTrade:
    coin: str
    side: str
    entry_price: float
    entry_time: str
    exit_price: float = 0.0
    exit_time: str = ""
    exit_type: str = ""
    holding_bars: int = 0
    gross_pnl_eq: float = 0.0
    cost_eq: float = 0.0
    net_pnl_eq: float = 0.0
    regime: str = ""
    s1_prob: float = 0.0
    s2_prob: float = 0.0


class PaperEngine:
    def __init__(self):
        self.equity = INITIAL_EQUITY
        self.trades = []
        self.positions = {}
        self.models = {}
        self.last_train = None
        self.signal_count = Counter()
        self.signal_triggered = Counter()
        self.signal_blocked_regime = Counter()
        self.regime_count = Counter()

    def log(self, event):
        event["ts"] = datetime.now().isoformat()
        event["eq"] = round(self.equity, 2)
        with open(LOG_FILE, "a") as f:
            f.write(json.dumps(event, default=str) + "\n")

    def train(self, feature_data):
        for coin in ACTIVE_COINS:
            if coin not in feature_data:
                continue
            params = COIN_PARAMS[coin]
            df = feature_data[coin]
            h = HORIZONS[-1]
            h_label = f"label_{h*BM}min"
            ku, kl = coin_k(coin, "k_upper"), coin_k(coin, "k_lower")

            labeled = create_labels_triple_barrier(
                df.copy(), h, k_upper_override=ku, k_lower_override=kl, verbose=False)
            if h_label not in labeled.columns:
                continue

            exclude = {"label", "future_return", "open", "high", "low", "close", "volume"}
            for hh in HORIZONS:
                exclude.add(f"label_{hh*BM}min")
                exclude.add(f"return_{hh*BM}min")
            leak_kw = ["future", "target", "label_", "return_", "fwd_", "forward_"]

            fcols = [c for c in labeled.columns
                     if c not in exclude
                     and not any(kw in c.lower() for kw in leak_kw)
                     and not is_excluded(c)
                     and labeled[c].dtype in [np.float64, np.float32, np.int64, np.int32, float, int]]

            mf = params["max_features"]
            clean = labeled.replace([np.inf, -np.inf], np.nan).ffill().bfill()
            X = clean[fcols].fillna(0).values
            y = clean[h_label].fillna(1).values.astype(int)

            if len(fcols) > mf:
                nmi = min(2000, len(X))
                mi = mutual_info_classif(X[:nmi], y[:nmi], discrete_features=False,
                                         random_state=42, n_neighbors=5)
                top = np.argsort(mi)[-mf:]
                fcols = [fcols[i] for i in sorted(top)]
                X = clean[fcols].fillna(0).values

            y_s1 = (y != LABEL_MAP["HOLD"]).astype(int)
            if len(np.unique(y_s1)) < 2:
                continue
            s1c = np.bincount(y_s1, minlength=2)
            s1w = np.where(s1c > 0, len(y_s1) / (2 * s1c + 1e-10), 1.0)[y_s1]
            s1 = EnhancedEnsemble(n_classes=2, use_stacking=True, n_jobs=N_JOBS, verbose=False)
            s1.fit(X, y_s1, sample_weight=s1w)

            s2 = None
            tm = y != LABEL_MAP["HOLD"]
            if tm.sum() >= 30:
                y_s2 = (y[tm] == LABEL_MAP["UP"]).astype(int)
                if len(np.unique(y_s2)) >= 2:
                    s2c = np.bincount(y_s2, minlength=2)
                    s2w = np.where(s2c > 0, len(y_s2) / (2 * s2c + 1e-10), 1.0)[y_s2]
                    s2 = EnhancedEnsemble(n_classes=2, use_stacking=True, n_jobs=N_JOBS, verbose=False)
                    s2.fit(X[tm], y_s2, sample_weight=s2w)

            self.models[coin] = (s1, s2, fcols)
            print(f"    {coin}: trained ({len(X)} samples, {len(fcols)} features)")
        self.last_train = datetime.now()

    def on_bar(self, coin, bar, df):
        if coin not in self.models:
            return
        params = COIN_PARAMS[coin]
        s1m, s2m, fcols = self.models[coin]

        # Check exits
        if coin in self.positions:
            self._check_exit(coin, bar)
            return  # no new entry while position open

        # Regime
        idx = df.index.get_loc(bar.name)
        regime = get_regime(df, idx)
        self.regime_count[f"{coin}_{regime}"] += 1

        # Signal
        xf = df[fcols].iloc[-1:].replace([np.inf, -np.inf], 0).fillna(0).values
        s1p = s1m.predict_proba(xf)
        s1_prob = float(s1p[0, 1])
        self.signal_count[coin] += 1

        if s1_prob < params["stage1_threshold"]:
            return

        # Regime block
        blocked = coin_blocked_regimes(coin)
        if regime in blocked:
            self.signal_blocked_regime[coin] += 1
            self.log({"event": "BLOCKED", "coin": coin, "regime": regime, "s1_prob": round(s1_prob, 3)})
            return

        self.signal_triggered[coin] += 1

        s2_prob = 0.5
        side = "SELL"
        if s2m is not None:
            s2p = s2m.predict_proba(xf)
            s2_prob = float(s2p[0, 1])
            side = "BUY" if np.argmax(s2p[0]) == 1 else "SELL"

        # Open
        entry = bar["close"]
        ku, kl = coin_k(coin, "k_upper"), coin_k(coin, "k_lower")
        atr = bar.get("atr_14", entry * 0.01)
        if np.isnan(atr):
            atr = entry * 0.01

        ud = max(ku * atr, COMMON["min_barrier_pct"] * entry)
        ld = max(kl * atr, COMMON["min_barrier_pct"] * entry)
        tp = entry + ud if side == "BUY" else entry - ud
        sl = entry - ld if side == "BUY" else entry + ld

        self.positions[coin] = {
            "side": side, "entry": entry, "tp": tp, "sl": sl,
            "atr": atr, "time": str(bar.name), "bars": 0,
            "regime": regime, "s1_prob": s1_prob, "s2_prob": s2_prob,
        }
        self.log({"event": "OPEN", "coin": coin, "side": side,
                  "entry": round(entry, 6), "tp": round(tp, 6), "sl": round(sl, 6),
                  "regime": regime, "s1": round(s1_prob, 3), "s2": round(s2_prob, 3)})
        print(f"      [{coin}] OPEN {side} @{entry:.4f} tp={tp:.4f} sl={sl:.4f} "
              f"regime={regime} s1={s1_prob:.2f}")

    def _check_exit(self, coin, bar):
        pos = self.positions[coin]
        pos["bars"] += 1
        h, l, c = bar["high"], bar["low"], bar["close"]

        et, ep = None, 0.0
        if pos["side"] == "BUY":
            ht, hs = h >= pos["tp"], l <= pos["sl"]
        else:
            ht, hs = l <= pos["tp"], h >= pos["sl"]

        if ht and hs:
            et, ep = "stop_loss", pos["sl"]
        elif ht:
            et, ep = "take_profit", pos["tp"]
        elif hs:
            et, ep = "stop_loss", pos["sl"]
        elif pos["bars"] >= MAX_HORIZON:
            et, ep = "time_stop", c

        if et:
            self._close(coin, et, ep, str(bar.name))

    def _close(self, coin, exit_type, exit_price, exit_time):
        pos = self.positions.pop(coin)
        kl = coin_k(coin, "k_lower")

        if pos["side"] == "BUY":
            gpnl = (exit_price - pos["entry"]) / pos["entry"]
        else:
            gpnl = (pos["entry"] - exit_price) / pos["entry"]

        sd = max(kl * pos["atr"] / pos["entry"], 0.003)
        nr = RISK_FRAC / sd
        gpnl_eq = gpnl * nr

        ee = {"take_profit": ExitType.TAKE_PROFIT, "stop_loss": ExitType.STOP_LOSS,
              "time_stop": ExitType.TIME_STOP}[exit_type]
        cost = COST_MODEL.estimate_trade_cost(
            entry_price=pos["entry"], sl_price=pos["sl"], tp_price=pos["tp"],
            risk_frac=RISK_FRAC, exit_type=ee, holding_bars=pos["bars"],
            bar_minutes=BM, entry_is_maker=True)

        net = gpnl_eq - cost.total_eq
        self.equity *= (1 + net)

        trade = PaperTrade(
            coin=coin, side=pos["side"], entry_price=pos["entry"],
            entry_time=pos["time"], exit_price=exit_price, exit_time=exit_time,
            exit_type=exit_type, holding_bars=pos["bars"],
            gross_pnl_eq=round(gpnl_eq, 6), cost_eq=round(cost.total_eq, 6),
            net_pnl_eq=round(net, 6), regime=pos["regime"],
            s1_prob=pos["s1_prob"], s2_prob=pos["s2_prob"])
        self.trades.append(trade)

        tag = "W" if net > 0 else "L"
        self.log({"event": "CLOSE", "coin": coin, "exit": exit_type,
                  "net": round(net, 6), "hold": pos["bars"], "eq": round(self.equity, 2)})
        print(f"      [{coin}] CLOSE {exit_type} @{exit_price:.4f} "
              f"hold={pos['bars']} net={net:+.4%} [{tag}] eq=${self.equity:.2f}")

    def evaluate_promotion(self):
        """Check live promotion criteria."""
        n = len(self.trades)
        results = {}

        # Min trades
        results["min_trades_total"] = n >= CRITERIA["min_trades_total"]
        for coin in ACTIVE_COINS:
            ct = sum(1 for t in self.trades if t.coin == coin)
            results[f"min_trades_{coin}"] = ct >= CRITERIA["min_trades_per_coin"]

        if n == 0:
            return {k: False for k in results}, "FAIL: 0 trades"

        pnls = [t.net_pnl_eq for t in self.trades]
        avg_pnl = np.mean(pnls)
        total_pnl = np.sum(pnls)

        # Net EV
        results["net_ev_positive"] = avg_pnl > CRITERIA["net_ev_per_trade_min"]

        # MDD
        eq = np.cumsum(pnls)
        pk = np.maximum.accumulate(eq)
        mdd = abs(np.min(eq - pk))
        results["mdd_ok"] = mdd < CRITERIA["max_mdd_pct"]

        # Cost share
        tc = sum(t.cost_eq for t in self.trades)
        tg = sum(abs(t.gross_pnl_eq) for t in self.trades)
        cs = tc / (tg + 1e-10)
        results["cost_share_ok"] = cs < CRITERIA["max_cost_share"]

        # Concentration
        if total_pnl > 0:
            max_contrib = max(t.net_pnl_eq for t in self.trades) / total_pnl
            results["concentration_ok"] = max_contrib < CRITERIA["max_single_trade_contribution"]
        else:
            results["concentration_ok"] = True

        # Consecutive losses
        max_consec = 0
        cur = 0
        for p in pnls:
            if p < 0:
                cur += 1
                max_consec = max(max_consec, cur)
            else:
                cur = 0
        results["consec_loss_ok"] = max_consec < CRITERIA["max_consecutive_losses"]

        # Regime diversity
        regimes = set(t.regime for t in self.trades)
        results["regime_diversity"] = len(regimes) >= CRITERIA["min_regime_diversity"]

        passed = all(results.values())
        verdict = "PASS -- LIVE ELIGIBLE" if passed else "FAIL"
        return results, verdict

    def summary(self):
        n = len(self.trades)
        if n == 0:
            return {"trades": 0, "equity": self.equity}

        pnls = [t.net_pnl_eq for t in self.trades]
        eq = np.cumsum(pnls)
        pk = np.maximum.accumulate(eq)
        mdd = abs(np.min(eq - pk))
        tc = sum(t.cost_eq for t in self.trades)
        tg = sum(abs(t.gross_pnl_eq) for t in self.trades)

        by_coin = {}
        for coin in ACTIVE_COINS:
            ct = [t for t in self.trades if t.coin == coin]
            cp = [t.net_pnl_eq for t in ct]
            by_coin[coin] = {
                "trades": len(ct),
                "avg_pnl": round(np.mean(cp), 6) if cp else 0,
                "total_pnl": round(np.sum(cp), 6) if cp else 0,
            }

        promo_checks, promo_verdict = self.evaluate_promotion()

        return {
            "trades": n, "equity": round(self.equity, 2),
            "return_pct": round((self.equity / INITIAL_EQUITY - 1) * 100, 2),
            "avg_pnl": round(np.mean(pnls), 6),
            "total_pnl": round(np.sum(pnls), 6),
            "mdd": round(mdd, 6),
            "cost_share": round(tc / (tg + 1e-10), 4),
            "by_coin": by_coin,
            "signals": dict(self.signal_count),
            "triggered": dict(self.signal_triggered),
            "blocked_regime": dict(self.signal_blocked_regime),
            "regime_freq": dict(self.regime_count),
            "promotion_checks": promo_checks,
            "promotion_verdict": promo_verdict,
        }


# ==================== Main ====================

def fetch_data():
    ohlcv = fetch_all_top10("365d", "1h")
    first = list(ohlcv.values())[0]
    macro = crawl_all_macro_data(first.index)
    ma = macro.get("aligned", pd.DataFrame())
    data = {}
    for coin in ACTIVE_COINS:
        if coin not in ohlcv:
            continue
        df = ohlcv[coin].copy()
        if len(ma) > 0:
            for col in ma.columns:
                df[col] = ma[col].reindex(df.index).ffill().bfill().fillna(0)
        data[coin] = df
    return data


def main():
    start = datetime.now()
    end = start + timedelta(days=PAPER_DAYS)

    print(f"\n{'='*70}")
    print(f"  PAPER TRADING v3.4 -- DOT + ADA")
    print(f"  {start.strftime('%Y-%m-%d %H:%M')}")
    print(f"  End: {end.strftime('%Y-%m-%d %H:%M')}")
    print(f"  Config: {YAML_PATH}")
    print(f"  Criteria: {CRITERIA_PATH}")
    print(f"  Coins: {ACTIVE_COINS}")
    print(f"  Equity: ${INITIAL_EQUITY:,}")
    print(f"{'='*70}")

    engine = PaperEngine()

    print(f"\n  [INIT] Training models...")
    data = fetch_data()
    engine.train(data)
    engine.log({"event": "INIT", "coins": ACTIVE_COINS})

    last_bar = {}
    cycle = 0

    while datetime.now() < end:
        cycle += 1

        # Retrain daily
        if engine.last_train and (datetime.now() - engine.last_train) > timedelta(hours=24):
            print(f"\n  [RETRAIN] Daily refresh...")
            try:
                data = fetch_data()
                engine.train(data)
            except Exception as e:
                print(f"  [ERR] {e}")

        for coin in ACTIVE_COINS:
            if coin not in data:
                continue
            df = data[coin]
            lt = df.index[-1]
            if coin in last_bar and lt <= last_bar[coin]:
                continue
            last_bar[coin] = lt
            engine.on_bar(coin, df.iloc[-1], df)

        if cycle % 10 == 0:
            s = engine.summary()
            elapsed = (datetime.now() - start).total_seconds() / 3600
            remain = (end - datetime.now()).total_seconds() / 3600
            print(f"\n  [{datetime.now().strftime('%H:%M')}] "
                  f"elapsed={elapsed:.1f}h remain={remain:.1f}h "
                  f"trades={s['trades']} eq=${s['equity']:,.2f} "
                  f"ret={s['return_pct']:+.2f}%")
            for coin in ACTIVE_COINS:
                bc = s["by_coin"].get(coin, {})
                sig = s["signals"].get(coin, 0)
                trig = s["triggered"].get(coin, 0)
                blk = s["blocked_regime"].get(coin, 0)
                print(f"    {coin}: {bc.get('trades',0)}T sig={trig}/{sig} "
                      f"blocked={blk} pnl={bc.get('total_pnl',0):+.4%}")

        time.sleep(POLL_MINUTES * 60)

    # Final
    s = engine.summary()
    print(f"\n{'='*70}")
    print(f"  PAPER TRADING COMPLETE")
    print(f"  Trades: {s['trades']} | Equity: ${s['equity']:,.2f} | Return: {s['return_pct']:+.2f}%")
    print(f"  Avg PnL: {s['avg_pnl']:+.4%} | MDD: {s['mdd']:.4%} | Cost: {s['cost_share']:.1%}")
    print(f"  Regime frequency: {s['regime_freq']}")
    print(f"  Promotion: {s['promotion_verdict']}")
    for k, v in s["promotion_checks"].items():
        print(f"    {k}: {'PASS' if v else 'FAIL'}")
    print(f"{'='*70}")

    with open(REPORT_DIR / "paper_summary.json", "w") as f:
        json.dump(s, f, indent=2, default=str)
    if engine.trades:
        pd.DataFrame([asdict(t) for t in engine.trades]).to_csv(
            REPORT_DIR / "paper_trades.csv", index=False)
    print(f"  Reports: {REPORT_DIR}/")


if __name__ == "__main__":
    main()
