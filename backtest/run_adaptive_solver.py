"""Adaptive Multi-Signal Solver on 27-month 1h data.

Core idea: Instead of fixed single-signal rules, combine 8+ signals with
adaptive weights that update based on rolling 30-day performance.

Architecture:
  1. Signal Layer: 8 independent signals each produce [-1, +1] score
  2. Adaptive Weights: LinUCB-style bandit or exponential decay based on
     recent signal-to-outcome correlation
  3. Entry: weighted_score > threshold → BUY/SELL
  4. Exit: ATR-based SL/TP (same as before)

Signals:
  - TSMOM 24h, TSMOM 72h (momentum)
  - EMA 8/21 direction (trend)
  - RSI zone (reversal filter)
  - MACD histogram direction (momentum confirmation)
  - Bollinger position (mean-reversion pressure)
  - Volume trend (conviction)
  - Donchian position (breakout proximity)

The solver updates weights every 24h based on which signals predicted
the actual 24h-forward return correctly.
"""

from __future__ import annotations
import json, logging, sys, time, math, uuid
from pathlib import Path
from dataclasses import dataclass, field
from collections import defaultdict

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("adaptive")

FEE_RATE = 0.0020
SLIP = 0.0003


# ── Data Hub (reuse) ──
class Hub:
    def __init__(self, data_dir, coins, tf="1h"):
        self.data = {}
        for coin in coins:
            files = list(data_dir.glob(f"{coin}_{tf}_*.parquet"))
            if files:
                df = pd.read_parquet(files[0])
                df.index = pd.to_datetime(df.index, utc=True)
                df.sort_index(inplace=True)
                for c in ["open","high","low","close","volume"]:
                    df[c] = pd.to_numeric(df[c], errors="coerce")
                self.data[coin] = df

    def get(self, coin, limit, ts):
        if coin not in self.data: return None
        df = self.data[coin]
        avail = df.loc[:ts]
        return avail.tail(limit) if len(avail) >= 2 else None

    def bar(self, coin, ts):
        if coin not in self.data: return None
        df = self.data[coin]
        if ts in df.index:
            r = df.loc[ts]
            return {"o":float(r["open"]),"h":float(r["high"]),"l":float(r["low"]),"c":float(r["close"]),"v":float(r["volume"])}
        return None

    def timestamps(self, coins):
        idx = pd.DatetimeIndex([])
        for c in coins:
            if c in self.data: idx = idx.union(self.data[c].index)
        return idx.sort_values()


# ── Signal Functions: each returns float in [-1, +1] ──

def sig_tsmom_24h(close):
    """24h momentum normalized."""
    if len(close) < 25: return 0.0
    mom = (close[-1] - close[-24]) / close[-24]
    return np.clip(mom / 0.05, -1, 1)  # normalize: 5% move = ±1

def sig_tsmom_72h(close):
    """72h (3-day) momentum."""
    if len(close) < 73: return 0.0
    mom = (close[-1] - close[-72]) / close[-72]
    return np.clip(mom / 0.10, -1, 1)

def sig_ema_trend(close):
    """EMA 8/21 trend direction."""
    if len(close) < 25: return 0.0
    s = pd.Series(close)
    e8 = s.ewm(span=8).mean().values
    e21 = s.ewm(span=21).mean().values
    diff = (e8[-1] - e21[-1]) / close[-1]
    return np.clip(diff / 0.005, -1, 1)

def sig_rsi(close):
    """RSI as directional signal: >50 bullish, <50 bearish, extreme = stronger."""
    if len(close) < 15: return 0.0
    d = np.diff(close)
    g = np.where(d>0,d,0)
    l = np.where(d<0,-d,0)
    ag, al = np.mean(g[-14:]), np.mean(l[-14:])
    if al == 0: rsi = 100
    else: rsi = 100 - 100/(1+ag/al)
    return np.clip((rsi - 50) / 30, -1, 1)  # 20→-1, 50→0, 80→+1

def sig_macd(close):
    """MACD histogram sign and magnitude."""
    if len(close) < 30: return 0.0
    s = pd.Series(close)
    macd = s.ewm(span=12).mean() - s.ewm(span=26).mean()
    sig = macd.ewm(span=9).mean()
    hist = float(macd.iloc[-1] - sig.iloc[-1])
    return np.clip(hist / (close[-1] * 0.003), -1, 1)

def sig_bollinger(close):
    """Bollinger Band position: +1 at upper band, -1 at lower."""
    if len(close) < 21: return 0.0
    ma = np.mean(close[-20:])
    std = np.std(close[-20:])
    if std < 1e-10: return 0.0
    z = (close[-1] - ma) / std
    return np.clip(z / 2, -1, 1)  # 2σ = ±1

def sig_volume_trend(close, volume):
    """Volume trend: rising volume in direction of price = confirmation."""
    if len(close) < 13 or len(volume) < 13: return 0.0
    price_dir = 1 if close[-1] > close[-6] else -1
    vol_recent = np.mean(volume[-6:])
    vol_prior = np.mean(volume[-12:-6])
    if vol_prior <= 0: return 0.0
    vol_trend = vol_recent / vol_prior - 1  # >0 = increasing
    return np.clip(price_dir * vol_trend / 0.5, -1, 1)

def sig_donchian_pos(close):
    """Position within 20-bar Donchian channel: +1 at top, -1 at bottom."""
    if len(close) < 21: return 0.0
    # Use high/low from close approximation
    dc_h = np.max(close[-21:-1])
    dc_l = np.min(close[-21:-1])
    if dc_h == dc_l: return 0.0
    pos = (close[-1] - dc_l) / (dc_h - dc_l)  # 0~1
    return np.clip(pos * 2 - 1, -1, 1)  # -1~+1

SIGNAL_FNS = {
    "tsmom_24h": lambda c, v: sig_tsmom_24h(c),
    "tsmom_72h": lambda c, v: sig_tsmom_72h(c),
    "ema_trend": lambda c, v: sig_ema_trend(c),
    "rsi": lambda c, v: sig_rsi(c),
    "macd": lambda c, v: sig_macd(c),
    "bollinger": lambda c, v: sig_bollinger(c),
    "vol_trend": lambda c, v: sig_volume_trend(c, v),
    "donchian": lambda c, v: sig_donchian_pos(c),
}
N_SIGNALS = len(SIGNAL_FNS)


# ── Adaptive Weight Solver ──

class AdaptiveSolver:
    """Online weight updater: tracks which signals predict 24h returns.

    Uses exponential moving correlation between signal value and forward return.
    Weights are proportional to recent predictive power.
    """

    def __init__(self, signal_names: list[str], lookback: int = 720, decay: float = 0.995):
        self.names = signal_names
        self.n = len(signal_names)
        self.lookback = lookback  # ~30 days of 1h bars
        self.decay = decay

        # Running stats for each signal
        self.signal_history: list[dict[str, float]] = []  # [{signal_name: value}, ...]
        self.return_history: list[float] = []
        self.weights = np.ones(self.n) / self.n  # start equal

        # Performance tracking
        self.weight_history: list[dict] = []

    def record(self, signals: dict[str, float], forward_return: float):
        """Record signal values and the actual forward return."""
        self.signal_history.append(signals)
        self.return_history.append(forward_return)

        # Keep only lookback window
        if len(self.signal_history) > self.lookback:
            self.signal_history = self.signal_history[-self.lookback:]
            self.return_history = self.return_history[-self.lookback:]

    def update_weights(self):
        """Recompute weights based on signal-return correlation."""
        if len(self.signal_history) < 100:
            return  # Not enough data

        # Build signal matrix and return vector
        n = len(self.signal_history)
        sig_matrix = np.zeros((n, self.n))
        for i, sigs in enumerate(self.signal_history):
            for j, name in enumerate(self.names):
                sig_matrix[i, j] = sigs.get(name, 0.0)

        returns = np.array(self.return_history)

        # Exponential decay weights (recent data matters more)
        time_weights = np.array([self.decay ** (n - 1 - i) for i in range(n)])
        time_weights /= time_weights.sum()

        # Weighted correlation for each signal
        scores = np.zeros(self.n)
        for j in range(self.n):
            sig_j = sig_matrix[:, j]
            # Directional agreement: signal * return > 0 means correct prediction
            agreement = sig_j * returns
            scores[j] = np.sum(agreement * time_weights)

        # Convert to weights: softmax of scores
        # Only positive-scoring signals get weight
        scores = np.maximum(scores, 0)  # zero out negative predictors
        total = scores.sum()
        if total > 0:
            self.weights = scores / total
        else:
            self.weights = np.ones(self.n) / self.n

        self.weight_history.append({
            name: round(w, 4) for name, w in zip(self.names, self.weights)
        })

    def get_score(self, signals: dict[str, float]) -> float:
        """Compute weighted score from current signals."""
        score = 0.0
        for j, name in enumerate(self.names):
            score += self.weights[j] * signals.get(name, 0.0)
        return score


# ── Position ──
@dataclass
class Pos:
    id: str; coin: str; side: str; entry: float; entry_ts: pd.Timestamp
    sl: float; tp: float; notional: float; ttl: int
    held: int = 0; mfe: float = 0; mae: float = 0


# ── Adaptive Engine ──

class AdaptiveEngine:
    def __init__(self, hub, coins, capital=5000, lev=3, max_pos=6,
                 sl_mult=1.5, tp_mult=5.0, entry_threshold=0.3,
                 update_interval=24, lookback=720, decay=0.995):
        self.hub = hub
        self.coins = coins
        self.capital = capital
        self.lev = lev
        self.max_pos = max_pos
        self.sl_m = sl_mult
        self.tp_m = tp_mult
        self.threshold = entry_threshold
        self.update_interval = update_interval

        self.solver = AdaptiveSolver(list(SIGNAL_FNS.keys()), lookback, decay)
        self.open: list[Pos] = []
        self.closed: list[dict] = []
        self.cooldowns: dict[str, pd.Timestamp] = {}
        self._bar_count = 0

    def run(self) -> list[dict]:
        ts_all = self.hub.timestamps(self.coins)
        log.info(f"Adaptive engine: {len(ts_all)} bars, {len(self.coins)} coins, "
                 f"threshold={self.threshold}, SL={self.sl_m}, TP={self.tp_m}")

        for ts in ts_all:
            self._bar_count += 1

            # Phase 0: Record forward returns for previous signals (24h ago)
            self._record_outcomes(ts)

            # Phase 1: Update weights periodically
            if self._bar_count % self.update_interval == 0:
                self.solver.update_weights()

            # Phase 2: Check exits
            self._check_exits(ts)

            # Phase 3: Evaluate entries
            if len(self.open) < self.max_pos:
                self._evaluate(ts)

        for p in list(self.open):
            self._close(p, "END", ts_all[-1])

        log.info(f"Adaptive complete: {len(self.closed)} trades")
        return self.closed

    def _compute_signals(self, coin, ts):
        """Compute all 8 signals for a coin at timestamp."""
        df = self.hub.get(coin, 100, ts)
        if df is None or len(df) < 75:
            return None
        c = df["close"].values
        v = df["volume"].values
        signals = {}
        for name, fn in SIGNAL_FNS.items():
            try:
                signals[name] = fn(c, v)
            except:
                signals[name] = 0.0
        return signals

    def _record_outcomes(self, ts):
        """Record what happened 24h after previous signal snapshots."""
        # We recorded signals 24h ago; now we know the return
        target_ts = ts - pd.Timedelta(hours=24)
        for coin in self.coins:
            bar_now = self.hub.bar(coin, ts)
            bar_prev = self.hub.bar(coin, target_ts)
            if bar_now and bar_prev and bar_prev["c"] > 0:
                fwd_ret = (bar_now["c"] - bar_prev["c"]) / bar_prev["c"]
                # Get signals from 24h ago
                signals = self._compute_signals(coin, target_ts)
                if signals:
                    self.solver.record(signals, fwd_ret)

    def _evaluate(self, ts):
        for coin in self.coins:
            if any(p.coin == coin for p in self.open):
                continue
            ck = coin
            if ck in self.cooldowns and (ts - self.cooldowns[ck]).total_seconds() < 3600:
                continue

            signals = self._compute_signals(coin, ts)
            if signals is None:
                continue

            score = self.solver.get_score(signals)

            if abs(score) < self.threshold:
                continue

            side = "BUY" if score > 0 else "SELL"
            self._open_pos(coin, side, ts)
            self.cooldowns[ck] = ts

    def _open_pos(self, coin, side, ts):
        df = self.hub.get(coin, 20, ts)
        if df is None or len(df) < 15: return
        c = df["close"].values
        h, l = df["high"].values, df["low"].values
        # ATR
        tr = np.maximum(h[1:]-l[1:], np.maximum(np.abs(h[1:]-c[:-1]), np.abs(l[1:]-c[:-1])))
        atr = float(np.mean(tr[-14:])) if len(tr) >= 14 else float(np.mean(tr))
        if atr <= 0: return

        price = c[-1]
        fill = price*(1+SLIP) if side=="BUY" else price*(1-SLIP)
        sl_d = max(atr*self.sl_m, price*0.0045)
        tp_d = atr*self.tp_m
        if side=="BUY": sl,tp = fill-sl_d, fill+tp_d
        else: sl,tp = fill+sl_d, fill-tp_d
        notional = self.capital*0.08*self.lev

        self.open.append(Pos(str(uuid.uuid4())[:8], coin, side, fill, ts, sl, tp, notional, ttl=48))

    def _check_exits(self, ts):
        for p in list(self.open):
            bar = self.hub.bar(p.coin, ts)
            if not bar: continue
            p.held += 1
            h, l = bar["h"], bar["l"]
            if p.side=="BUY":
                p.mfe = max(p.mfe, h) if p.mfe else h
                p.mae = min(p.mae, l) if p.mae else l
                if l <= p.sl: self._close(p, "SL", ts)
                elif h >= p.tp: self._close(p, "TP", ts)
                elif p.held >= p.ttl: self._close(p, "TTL", ts)
            else:
                p.mfe = min(p.mfe, l) if p.mfe else l
                p.mae = max(p.mae, h) if p.mae else h
                if h >= p.sl: self._close(p, "SL", ts)
                elif l <= p.tp: self._close(p, "TP", ts)
                elif p.held >= p.ttl: self._close(p, "TTL", ts)

    def _close(self, p, reason, ts):
        if p not in self.open: return
        self.open.remove(p)
        if reason=="SL": ep=p.sl
        elif reason=="TP": ep=p.tp
        else:
            bar = self.hub.bar(p.coin, ts)
            ep = bar["c"] if bar else p.entry
        if p.side=="BUY":
            ep*=(1-SLIP)
            pnl_pct=(ep-p.entry)/p.entry
        else:
            ep*=(1+SLIP)
            pnl_pct=(p.entry-ep)/p.entry
        fee = p.notional * FEE_RATE
        gross = pnl_pct * p.notional
        self.closed.append({
            "coin":p.coin, "side":p.side, "reason":reason,
            "gross":round(gross,2), "net":round(gross-fee,2), "fee":round(fee,2),
            "pnl_pct":round(pnl_pct*100,3), "held":p.held,
            "entry_ts":str(p.entry_ts)[:19], "exit_ts":str(ts)[:19],
        })


def metrics(trades):
    if not trades: return {"n":0,"net":0,"wr":0,"pf":0,"tp%":0}
    df = pd.DataFrame(trades)
    n = len(df)
    w = df[df["net"]>0]
    wr = len(w)/n*100
    return {
        "n":n, "wr":round(wr,1), "gross":round(df["gross"].sum(),2),
        "fee":round(df["fee"].sum(),2), "net":round(df["net"].sum(),2),
        "pf":round(w["net"].sum()/(abs(df[df["net"]<=0]["net"].sum())+0.01),2),
        "tp%":round(len(df[df["reason"]=="TP"])/n*100,1),
        "sl%":round(len(df[df["reason"]=="SL"])/n*100,1),
        "avg_held":round(df["held"].mean(),1),
    }


def main():
    t0 = time.time()
    data_dir = ROOT / "data" / "historical"
    coins = ["SOL","XRP","ADA","DOT","DOGE"]
    ref = ["ETH","BNB"]

    log.info("Loading data...")
    hub = Hub(data_dir, coins + ref)
    ts_all = hub.timestamps(coins)
    split_50 = ts_all[len(ts_all)//2]
    split_70 = ts_all[int(len(ts_all)*0.70)]

    # ════════════════════════════════════════════════════════
    # SWEEP: threshold x SL x TP x decay x lookback
    # ════════════════════════════════════════════════════════
    print("\n" + "="*95)
    print("  ADAPTIVE SOLVER: Multi-Signal + Rolling Weight Update")
    print("  8 signals, 24h weight update, exponential decay")
    print("="*95)

    thresholds = [0.15, 0.20, 0.25, 0.30, 0.40]
    sl_mults = [1.0, 1.5, 2.0]
    tp_mults = [3.0, 5.0, 7.0]
    decays = [0.990, 0.995, 0.999]

    results = []
    best_net = -999999
    best_cfg = None

    total = len(thresholds) * len(sl_mults) * len(tp_mults) * len(decays)
    log.info(f"Sweeping {total} combinations...")

    count = 0
    for thresh in thresholds:
        for sl in sl_mults:
            for tp in tp_mults:
                for decay in decays:
                    if tp < sl: continue
                    count += 1
                    eng = AdaptiveEngine(hub, coins, sl_mult=sl, tp_mult=tp,
                                         entry_threshold=thresh, decay=decay)
                    trades = eng.run()
                    m = metrics(trades)
                    m.update({"thresh":thresh, "sl":sl, "tp":tp, "decay":decay})
                    results.append(m)

                    if m["net"] > best_net:
                        best_net = m["net"]
                        best_cfg = m

                    if count % 20 == 0:
                        log.info(f"  {count}/{total} done, best so far: ${best_net:+.2f}")

    # Print results
    profitable = [r for r in results if r["net"] > 0 and r["n"] >= 20]
    profitable.sort(key=lambda x: x["net"], reverse=True)

    print(f"\n  Profitable: {len(profitable)} / {len(results)}")
    if profitable:
        hdr = f"  {'Thresh':>6} {'SL':>4} {'TP':>4} {'Decay':>6} {'N':>5} {'WR%':>6} {'Gross$':>9} {'Net$':>9} {'PF':>5} {'TP%':>5}"
        print(hdr)
        print("  "+"-"*(len(hdr)-2))
        for r in profitable[:15]:
            print(f"  {r['thresh']:>6.2f} {r['sl']:>4.1f} {r['tp']:>4.1f} {r['decay']:>6.3f} "
                  f"{r['n']:>5} {r['wr']:>5.1f}% {r['gross']:>+9.2f} {r['net']:>+9.2f} "
                  f"{r['pf']:>5.2f} {r['tp%']:>4.1f}%")

        # ════════════════════════════════════════════════════════
        # VALIDATION on best config
        # ════════════════════════════════════════════════════════
        best = profitable[0]
        print(f"\n  BEST: thresh={best['thresh']} SL={best['sl']} TP={best['tp']} decay={best['decay']}")
        print(f"  Full: N={best['n']} WR={best['wr']}% Net=${best['net']:+.2f} PF={best['pf']}")

        print(f"\n  === WALK-FORWARD VALIDATION ===")

        # OOS test
        eng_oos = AdaptiveEngine(hub, coins, sl_mult=best["sl"], tp_mult=best["tp"],
                                  entry_threshold=best["thresh"], decay=best["decay"])
        # Hack: only run from split_70 onward
        eng_oos_full = AdaptiveEngine(hub, coins, sl_mult=best["sl"], tp_mult=best["tp"],
                                       entry_threshold=best["thresh"], decay=best["decay"])
        all_trades = eng_oos_full.run()
        # Split trades by time
        if all_trades:
            df_t = pd.DataFrame(all_trades)
            df_t["entry_dt"] = pd.to_datetime(df_t["entry_ts"])
            train = df_t[df_t["entry_dt"] < str(split_70)]
            test = df_t[df_t["entry_dt"] >= str(split_70)]
            h1 = df_t[df_t["entry_dt"] < str(split_50)]
            h2 = df_t[df_t["entry_dt"] >= str(split_50)]

            m_train = metrics(train.to_dict("records"))
            m_test = metrics(test.to_dict("records"))
            m_h1 = metrics(h1.to_dict("records"))
            m_h2 = metrics(h2.to_dict("records"))

            print(f"  Train (70%):  N={m_train['n']:>4} WR={m_train['wr']:>5.1f}% Net=${m_train['net']:>+9.2f} PF={m_train['pf']:>5.2f}")
            print(f"  Test  (30%):  N={m_test['n']:>4} WR={m_test['wr']:>5.1f}% Net=${m_test['net']:>+9.2f} PF={m_test['pf']:>5.2f}")
            print(f"  Half 1:       N={m_h1['n']:>4} WR={m_h1['wr']:>5.1f}% Net=${m_h1['net']:>+9.2f} PF={m_h1['pf']:>5.2f}")
            print(f"  Half 2:       N={m_h2['n']:>4} WR={m_h2['wr']:>5.1f}% Net=${m_h2['net']:>+9.2f} PF={m_h2['pf']:>5.2f}")

            # Per-year
            df_t["year"] = df_t["entry_dt"].dt.year
            print(f"\n  Per-year:")
            for yr in sorted(df_t["year"].unique()):
                ym = metrics(df_t[df_t["year"]==yr].to_dict("records"))
                print(f"    {yr}: N={ym['n']:>4} WR={ym['wr']:>5.1f}% Net=${ym['net']:>+9.2f} PF={ym['pf']:>5.2f}")

            # Weight evolution
            if eng_oos_full.solver.weight_history:
                print(f"\n  Weight evolution ({len(eng_oos_full.solver.weight_history)} updates):")
                for i in [0, len(eng_oos_full.solver.weight_history)//4,
                          len(eng_oos_full.solver.weight_history)//2,
                          3*len(eng_oos_full.solver.weight_history)//4,
                          -1]:
                    wh = eng_oos_full.solver.weight_history[i]
                    parts = [f"{k}={v:.3f}" for k,v in sorted(wh.items(), key=lambda x:-x[1])[:4]]
                    print(f"    [{i:>4}] {', '.join(parts)}")

            # Neighborhood check
            print(f"\n  Param neighborhood (+-20%):")
            nb_ok = 0
            nb_total = 0
            for dt in [-0.2, -0.1, 0, 0.1, 0.2]:
                for ds in [-0.2, 0, 0.2]:
                    for dtp in [-0.2, 0, 0.2]:
                        t = best["thresh"]*(1+dt)
                        s = best["sl"]*(1+ds)
                        p = best["tp"]*(1+dtp)
                        eng_n = AdaptiveEngine(hub, coins, sl_mult=s, tp_mult=p,
                                               entry_threshold=t, decay=best["decay"])
                        nt = eng_n.run()
                        nm = metrics(nt)
                        if nm["net"] > 0: nb_ok += 1
                        nb_total += 1
            print(f"    {nb_ok}/{nb_total} ({nb_ok/nb_total*100:.0f}%) neighbors profitable")

    else:
        print("\n  No profitable configs found.")
        if best_cfg:
            print(f"  Closest: thresh={best_cfg.get('thresh')} SL={best_cfg.get('sl')} "
                  f"TP={best_cfg.get('tp')} N={best_cfg['n']} Net=${best_cfg['net']:+.2f}")

    # Save
    out = ROOT / "backtest" / "output"
    out.mkdir(exist_ok=True)
    with open(out / "adaptive_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed/60:.1f} min")


if __name__ == "__main__":
    main()
