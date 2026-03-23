"""Download Binance metrics data + integrate with TSMOM strategy."""

import sys, os, io, zipfile, time
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
import numpy as np
import pandas as pd
from datetime import date, timedelta
from pathlib import Path
from itertools import product
import warnings
warnings.filterwarnings("ignore")

COINS_MAP = {
    "BTC": "BTCUSDT", "ETH": "ETHUSDT", "SOL": "SOLUSDT",
    "XRP": "XRPUSDT", "ADA": "ADAUSDT", "DOT": "DOTUSDT", "LINK": "LINKUSDT",
}
BASE_URL = "https://data.binance.vision/data/futures/um/daily/metrics"
OUT_DIR = Path("data/raw/binance_public/metrics")
COST = 0.0020

# ══════════════════════════════════════════════════════════
# 1. Download Binance metrics
# ══════════════════════════════════════════════════════════

def download_metrics(days=365):
    """Download metrics (OI, long/short ratio, taker ratio) for all coins."""
    session = requests.Session()
    today = date.today()
    dates = [today - timedelta(days=i) for i in range(2, days + 2)]

    for coin, symbol in COINS_MAP.items():
        sym_dir = OUT_DIR / symbol
        sym_dir.mkdir(parents=True, exist_ok=True)

        existing = list(sym_dir.glob("*.csv"))
        if len(existing) >= days * 0.8:
            print(f"  [SKIP] {coin}: {len(existing)} CSVs already exist")
            continue

        ok, fail = 0, 0
        for d in dates:
            csv_path = sym_dir / f"{symbol}-metrics-{d}.csv"
            if csv_path.exists():
                ok += 1
                continue

            url = f"{BASE_URL}/{symbol}/{symbol}-metrics-{d}.zip"
            try:
                resp = session.get(url, timeout=15)
                if resp.status_code == 404:
                    fail += 1
                    continue
                resp.raise_for_status()

                zip_path = sym_dir / f"{symbol}-metrics-{d}.zip"
                zip_path.write_bytes(resp.content)

                with zipfile.ZipFile(zip_path, 'r') as z:
                    z.extractall(sym_dir)
                zip_path.unlink()
                ok += 1
            except Exception as e:
                fail += 1

        print(f"  [DL] {coin}: {ok} ok, {fail} missing")

    return True


def load_metrics(coin: str) -> pd.DataFrame:
    """Load all metrics CSVs for a coin into a single DataFrame."""
    symbol = COINS_MAP[coin]
    sym_dir = OUT_DIR / symbol
    csvs = sorted(sym_dir.glob("*.csv"))

    if not csvs:
        return pd.DataFrame()

    dfs = []
    for f in csvs:
        try:
            df = pd.read_csv(f)
            dfs.append(df)
        except Exception:
            continue

    if not dfs:
        return pd.DataFrame()

    merged = pd.concat(dfs, ignore_index=True)
    merged["create_time"] = pd.to_datetime(merged["create_time"])
    merged = merged.sort_values("create_time").drop_duplicates("create_time")
    merged = merged.set_index("create_time")

    return merged


def resample_metrics_4h(metrics: pd.DataFrame) -> pd.DataFrame:
    """Resample 5-min metrics to 4h bars."""
    if metrics.empty:
        return metrics

    rules = {}
    for col in metrics.columns:
        if col == "symbol":
            continue
        rules[col] = "last"

    # Also compute mean for some columns
    resampled = metrics.drop(columns=["symbol"], errors="ignore").resample("4h").agg(rules)
    resampled = resampled.dropna(how="all")

    return resampled


# ══════════════════════════════════════════════════════════
# 2. Feature Engineering from Binance data
# ══════════════════════════════════════════════════════════

def add_binance_features(df_4h: pd.DataFrame, metrics_4h: pd.DataFrame) -> pd.DataFrame:
    """Add Binance-derived features to OHLCV dataframe."""
    if metrics_4h.empty:
        return df_4h

    # Align metrics to OHLCV index (handle timezone mismatch)
    if df_4h.index.tz is not None and metrics_4h.index.tz is None:
        metrics_4h.index = metrics_4h.index.tz_localize(df_4h.index.tz)
    elif df_4h.index.tz is None and metrics_4h.index.tz is not None:
        metrics_4h.index = metrics_4h.index.tz_localize(None)
    metrics_aligned = metrics_4h.reindex(df_4h.index, method="ffill")

    # OI features
    if "sum_open_interest_value" in metrics_aligned.columns:
        oi = metrics_aligned["sum_open_interest_value"].astype(float)
        df_4h["oi_value"] = oi
        df_4h["oi_change_pct"] = oi.pct_change()
        df_4h["oi_ma_24"] = oi.rolling(24, min_periods=6).mean()
        df_4h["oi_ratio"] = oi / df_4h["oi_ma_24"]
        df_4h["oi_zscore"] = (oi - oi.rolling(48, min_periods=12).mean()) / oi.rolling(48, min_periods=12).std()

        # OI divergence: OI rising + price falling = short buildup
        price_chg = df_4h["close"].pct_change(6)
        oi_chg = oi.pct_change(6)
        df_4h["oi_price_div"] = np.sign(oi_chg) * -np.sign(price_chg)  # +1 = divergence

    # Long/Short Ratio features
    if "count_long_short_ratio" in metrics_aligned.columns:
        lsr = metrics_aligned["count_long_short_ratio"].astype(float)
        df_4h["lsr"] = lsr
        df_4h["lsr_ma_12"] = lsr.rolling(12, min_periods=3).mean()
        df_4h["lsr_zscore"] = (lsr - lsr.rolling(48, min_periods=12).mean()) / lsr.rolling(48, min_periods=12).std()
        # Extreme long = contrarian short signal
        df_4h["lsr_extreme_long"] = (df_4h["lsr_zscore"] > 1.5).astype(int)
        df_4h["lsr_extreme_short"] = (df_4h["lsr_zscore"] < -1.5).astype(int)

    # Top Trader L/S Ratio
    if "sum_toptrader_long_short_ratio" in metrics_aligned.columns:
        ttr = metrics_aligned["sum_toptrader_long_short_ratio"].astype(float)
        df_4h["top_trader_lsr"] = ttr
        df_4h["top_trader_lsr_zscore"] = (ttr - ttr.rolling(48, min_periods=12).mean()) / ttr.rolling(48, min_periods=12).std()

    # Taker Buy/Sell Volume Ratio
    if "sum_taker_long_short_vol_ratio" in metrics_aligned.columns:
        tvr = metrics_aligned["sum_taker_long_short_vol_ratio"].astype(float)
        df_4h["taker_vol_ratio"] = tvr
        df_4h["taker_vol_ma_6"] = tvr.rolling(6, min_periods=2).mean()
        df_4h["taker_buy_pressure"] = (tvr > 1.0).astype(int)

    return df_4h


# ══════════════════════════════════════════════════════════
# 3. Enhanced Strategy with Binance data
# ══════════════════════════════════════════════════════════

def compute_cvd_ratio(df, window=24):
    hr = (df["high"] - df["low"]).replace(0, np.nan)
    buy_frac = ((df["close"] - df["low"]) / hr).fillna(0.5).clip(0, 1)
    vd = (2 * buy_frac - 1) * df["volume"]
    cvd = vd.cumsum()
    cvd_ma = cvd.rolling(window, min_periods=max(6, window // 4)).mean()
    return ((cvd - cvd_ma) / cvd_ma.abs().replace(0, np.nan)).fillna(0)


def generate_enhanced_signals(df, lookback_days=5, volume_weighted=True,
                                cvd_quantile=0.75, cvd_roll_window=120,
                                use_oi=False, use_lsr=False, use_taker=False,
                                oi_div_filter=False, lsr_contrarian=False):
    """Config3 base + optional Binance feature filters."""

    lookback_bars = lookback_days * 6

    # TSMOM
    if volume_weighted and "volume" in df.columns:
        ret = df["close"].pct_change()
        vol_w = df["volume"] / df["volume"].rolling(lookback_bars, min_periods=1).mean()
        weighted_ret = (ret * vol_w).rolling(lookback_bars, min_periods=lookback_bars).sum()
        tsmom = np.sign(weighted_ret)
    else:
        tsmom = np.sign(df["close"].pct_change(lookback_bars))

    # RSI filter
    rsi = df.get("rsi_14", pd.Series(50, index=df.index))
    rsi_ok = ((tsmom == 1) & (rsi > 50)) | ((tsmom == -1) & (rsi < 50))

    # CVD timing
    cvd_ratio = compute_cvd_ratio(df, window=24)
    q_hi = cvd_ratio.rolling(cvd_roll_window, min_periods=30).quantile(cvd_quantile)
    q_lo = cvd_ratio.rolling(cvd_roll_window, min_periods=30).quantile(1 - cvd_quantile)
    cvd_ok = ((tsmom == -1) & (cvd_ratio > q_hi)) | ((tsmom == 1) & (cvd_ratio < q_lo))

    # Base filter
    mask = rsi_ok & cvd_ok

    # OI filter: avoid entering when OI is extremely high (crowded)
    if use_oi and "oi_zscore" in df.columns:
        oi_ok = df["oi_zscore"].abs() < 2.0  # Avoid extreme positioning
        mask = mask & oi_ok

    # OI divergence filter
    if oi_div_filter and "oi_price_div" in df.columns:
        # Only enter when OI divergence confirms our direction
        oi_div_ok = (
            ((tsmom == -1) & (df["oi_price_div"] > 0)) |  # SHORT + OI rising while price falling
            ((tsmom == 1) & (df["oi_price_div"] < 0)) |   # LONG + OI falling while price rising
            (df["oi_price_div"] == 0)  # no divergence = neutral
        )
        mask = mask & oi_div_ok

    # LSR contrarian: extreme long positioning = short signal boost
    if lsr_contrarian and "lsr_extreme_long" in df.columns:
        lsr_ok = (
            ((tsmom == -1) & (df["lsr_extreme_long"] == 1)) |  # SHORT + crowd is long
            ((tsmom == 1) & (df["lsr_extreme_short"] == 1)) |  # LONG + crowd is short
            ((df["lsr_extreme_long"] == 0) & (df["lsr_extreme_short"] == 0))  # neutral = allow
        )
        mask = mask & lsr_ok

    # Taker volume: confirm direction
    if use_taker and "taker_buy_pressure" in df.columns:
        taker_ok = (
            ((tsmom == 1) & (df["taker_buy_pressure"] == 1)) |
            ((tsmom == -1) & (df["taker_buy_pressure"] == 0)) |
            True  # Don't filter if feature missing
        )
        mask = mask & taker_ok

    signals = tsmom.copy()
    signals[~mask] = 0
    return signals.fillna(0).astype(int)


def run_backtest(df, signals, k_upper=4.0, k_lower=1.5, max_hold=24):
    """Fast barrier backtest."""
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values

    if "atr_14" in df.columns:
        atr = df["atr_14"].values
    else:
        tr = np.maximum(high - low,
                        np.maximum(np.abs(high - np.roll(close, 1)),
                                   np.abs(low - np.roll(close, 1))))
        atr = pd.Series(tr).rolling(14, min_periods=1).mean().values

    sig = signals.values if hasattr(signals, 'values') else signals
    trades = []
    next_avail = 0

    for i in range(len(df) - max_hold):
        if i < next_avail or sig[i] == 0 or np.isnan(atr[i]) or atr[i] <= 0:
            continue

        side = int(sig[i])
        entry = close[i]
        a = atr[i]
        tp_d = max(k_upper * a, entry * 0.002)
        sl_d = max(k_lower * a, entry * 0.002)

        tp = entry + tp_d * side
        sl = entry - sl_d * side

        exit_p = close[min(i + max_hold, len(df) - 1)]
        exit_bar = min(i + max_hold, len(df) - 1)

        for j in range(i + 1, min(i + max_hold + 1, len(df))):
            if side == 1:
                if low[j] <= sl:
                    exit_p, exit_bar = sl, j; break
                if high[j] >= tp:
                    exit_p, exit_bar = tp, j; break
            else:
                if high[j] >= sl:
                    exit_p, exit_bar = sl, j; break
                if low[j] <= tp:
                    exit_p, exit_bar = tp, j; break

        pnl = ((exit_p - entry) / entry) * side
        trades.append(pnl - COST)
        next_avail = exit_bar + 1

    return np.array(trades)


def calc_metrics(pnls):
    if len(pnls) == 0:
        return {"n": 0, "wr": 0, "avg": 0, "sharpe": 0, "mdd": 0, "pf": 0, "total": 0}
    arr = np.array(pnls)
    eq = np.cumsum(arr)
    dd = eq - np.maximum.accumulate(eq)
    wins = sum(p for p in arr if p > 0)
    losses = abs(sum(p for p in arr if p < 0))
    return {
        "n": len(arr), "wr": np.mean(arr > 0), "avg": np.mean(arr),
        "sharpe": np.mean(arr) / np.std(arr) * np.sqrt(len(arr)) if np.std(arr) > 0 else 0,
        "mdd": np.min(dd) if len(dd) > 0 else 0,
        "pf": wins / losses if losses > 0 else float("inf"),
        "total": np.sum(arr),
    }


# ══════════════════════════════════════════════════════════
# 4. Main: Download, Integrate, Test, Optimize
# ══════════════════════════════════════════════════════════

def main():
    t0 = time.time()

    # Step 1: Download
    print("=" * 90)
    print("STEP 1: Download Binance metrics data")
    print("=" * 90)
    download_metrics(days=365)

    # Step 2: Load OHLCV
    print("\n" + "=" * 90)
    print("STEP 2: Load OHLCV + Technical Indicators + Microstructure")
    print("=" * 90)
    from src.data.crawlers.crypto_ohlcv import fetch_ohlcv, resample_to_4h, add_technical_indicators, TOP10_YAHOO
    from src.data.crawlers.microstructure_rollup import add_microstructure_rollup

    data = {}
    for coin in COINS_MAP:
        sym = TOP10_YAHOO.get(coin)
        if not sym:
            continue
        df = fetch_ohlcv(coin, sym, period="365d", interval="1h")
        if df.empty:
            continue
        df = resample_to_4h(df)
        df = add_technical_indicators(df)
        df = add_microstructure_rollup(df)

        # Integrate Binance metrics
        metrics = load_metrics(coin)
        if not metrics.empty:
            metrics_4h = resample_metrics_4h(metrics)
            df = add_binance_features(df, metrics_4h)
            n_binance = sum(1 for c in df.columns if c.startswith(("oi_", "lsr", "top_trader", "taker_")))
            print(f"  [OK] {coin}: {len(df)} bars, +{n_binance} Binance features")
        else:
            print(f"  [OK] {coin}: {len(df)} bars, no Binance data")

        data[coin] = df

    has_binance = any("oi_value" in data[c].columns for c in data)
    print(f"\n  Binance data available: {has_binance}")

    # Step 3: Config3 baseline
    print("\n" + "=" * 90)
    print("STEP 3: Config3 Baseline (TSMOM + RSI + CVD)")
    print("=" * 90)

    configs = [
        {"name": "C3_base", "lb": 5, "vw": True, "cq": 0.75, "cw": 120,
         "ku": 4.0, "kl": 1.5, "mh": 24,
         "use_oi": False, "use_lsr": False, "use_taker": False,
         "oi_div_filter": False, "lsr_contrarian": False},
    ]

    # Step 4: Generate enhanced configs with Binance data
    if has_binance:
        extra_configs = [
            {"name": "C3+OI_filter", "use_oi": True},
            {"name": "C3+OI_div", "oi_div_filter": True},
            {"name": "C3+LSR_contra", "lsr_contrarian": True},
            {"name": "C3+taker", "use_taker": True},
            {"name": "C3+OI+LSR", "use_oi": True, "lsr_contrarian": True},
            {"name": "C3+OI+taker", "use_oi": True, "use_taker": True},
            {"name": "C3+ALL", "use_oi": True, "lsr_contrarian": True, "use_taker": True, "oi_div_filter": True},
            {"name": "C3+OI_div+LSR", "oi_div_filter": True, "lsr_contrarian": True},
        ]
        for ec in extra_configs:
            cfg = configs[0].copy()
            cfg.update(ec)
            configs.append(cfg)

    # Step 5: Also test different TSMOM lookbacks with best Binance combo
    for lb in [5, 7, 10, 14, 21, 28]:
        for cq in [0.65, 0.75, 0.85]:
            cfg = configs[0].copy()
            cfg["name"] = f"lb{lb}_cq{cq}"
            cfg["lb"] = lb
            cfg["cq"] = cq
            if has_binance:
                cfg["use_oi"] = True
                cfg["lsr_contrarian"] = True
            configs.append(cfg)

    print(f"  Total configs to test: {len(configs)}")

    # Step 6: Run all configs
    print("\n" + "=" * 90)
    print("STEP 4: Test all configs")
    print("=" * 90)

    all_results = []
    for cfg in configs:
        all_pnls = []
        coin_results = {}

        for coin in COINS_MAP:
            if coin not in data:
                continue
            df = data[coin]

            sigs = generate_enhanced_signals(
                df, lookback_days=cfg["lb"], volume_weighted=cfg["vw"],
                cvd_quantile=cfg["cq"], cvd_roll_window=cfg["cw"],
                use_oi=cfg.get("use_oi", False), use_lsr=cfg.get("use_lsr", False),
                use_taker=cfg.get("use_taker", False),
                oi_div_filter=cfg.get("oi_div_filter", False),
                lsr_contrarian=cfg.get("lsr_contrarian", False),
            )

            pnls = run_backtest(df, sigs, k_upper=cfg["ku"], k_lower=cfg["kl"], max_hold=cfg["mh"])
            all_pnls.extend(pnls.tolist())
            coin_results[coin] = calc_metrics(pnls)

        portfolio = calc_metrics(all_pnls)
        portfolio["name"] = cfg["name"]
        all_results.append(portfolio)

        if portfolio["n"] >= 15 and portfolio["avg"] > 0:
            print(f"  {cfg['name']:25s} | n={portfolio['n']:3d} WR={portfolio['wr']:.1%} "
                  f"avg={portfolio['avg']:+.3%} Sharpe={portfolio['sharpe']:.2f} "
                  f"MDD={portfolio['mdd']:.2%} PF={portfolio['pf']:.2f}")

    # Step 7: Sort and report
    print("\n" + "=" * 90)
    print("STEP 5: Rankings")
    print("=" * 90)

    rdf = pd.DataFrame(all_results).sort_values("sharpe", ascending=False)
    valid = rdf[rdf["n"] >= 15]

    print("\n  TOP 10 by Sharpe:")
    for _, r in valid.head(10).iterrows():
        print(f"    {r['name']:25s} | n={r['n']:3.0f} WR={r['wr']:.1%} avg={r['avg']:+.3%} "
              f"Sharpe={r['sharpe']:.2f} MDD={r['mdd']:.2%} PF={r['pf']:.2f}")

    # Step 8: Walk-forward on top 3
    print("\n" + "=" * 90)
    print("STEP 6: Walk-Forward OOS (top 3)")
    print("=" * 90)

    top3_names = valid.head(3)["name"].tolist()
    top3_configs = [c for c in configs if c["name"] in top3_names]

    for cfg in top3_configs:
        print(f"\n  Config: {cfg['name']}")
        all_oos = []

        for coin in COINS_MAP:
            if coin not in data:
                continue
            df = data[coin]
            n = len(df)
            n_windows = 5
            ws = n // n_windows

            coin_oos = []
            for w in range(n_windows):
                s, e = w * ws, min((w + 1) * ws, n)
                if e - s < 60:
                    continue
                df_w = df.iloc[s:e].copy()
                sigs = generate_enhanced_signals(
                    df_w, lookback_days=cfg["lb"], volume_weighted=cfg["vw"],
                    cvd_quantile=cfg["cq"], cvd_roll_window=cfg["cw"],
                    use_oi=cfg.get("use_oi", False), use_lsr=cfg.get("use_lsr", False),
                    use_taker=cfg.get("use_taker", False),
                    oi_div_filter=cfg.get("oi_div_filter", False),
                    lsr_contrarian=cfg.get("lsr_contrarian", False),
                )
                pnls = run_backtest(df_w, sigs, k_upper=cfg["ku"], k_lower=cfg["kl"], max_hold=cfg["mh"])
                coin_oos.extend(pnls.tolist())

            if coin_oos:
                m = calc_metrics(coin_oos)
                all_oos.extend(coin_oos)
                print(f"    {coin}: n={m['n']:3d} WR={m['wr']:.1%} avg={m['avg']:+.3%} Sharpe={m['sharpe']:.2f}")

        if all_oos:
            m = calc_metrics(all_oos)
            print(f"    TOTAL: n={m['n']:3d} WR={m['wr']:.1%} avg={m['avg']:+.3%} "
                  f"Sharpe={m['sharpe']:.2f} MDD={m['mdd']:.2%}")

    # Step 9: Bootstrap best
    print("\n" + "=" * 90)
    print("STEP 7: Bootstrap stability (best config)")
    print("=" * 90)

    if top3_configs:
        best = top3_configs[0]
        all_pnls = []
        for coin in COINS_MAP:
            if coin not in data:
                continue
            df = data[coin]
            sigs = generate_enhanced_signals(
                df, lookback_days=best["lb"], volume_weighted=best["vw"],
                cvd_quantile=best["cq"], cvd_roll_window=best["cw"],
                use_oi=best.get("use_oi", False), use_lsr=best.get("use_lsr", False),
                use_taker=best.get("use_taker", False),
                oi_div_filter=best.get("oi_div_filter", False),
                lsr_contrarian=best.get("lsr_contrarian", False),
            )
            pnls = run_backtest(df, sigs, k_upper=best["ku"], k_lower=best["kl"], max_hold=best["mh"])
            all_pnls.extend(pnls.tolist())

        arr = np.array(all_pnls)
        rng = np.random.RandomState(42)
        boot_means = [np.mean(rng.choice(arr, len(arr), replace=True)) for _ in range(2000)]
        boot_sharpes = []
        for _ in range(2000):
            s = rng.choice(arr, len(arr), replace=True)
            if np.std(s) > 0:
                boot_sharpes.append(np.mean(s) / np.std(s) * np.sqrt(len(s)))

        bm, bs = np.array(boot_means), np.array(boot_sharpes)
        print(f"  Config: {best['name']}")
        print(f"  Original: n={len(arr)} avg={np.mean(arr):+.3%} Sharpe={calc_metrics(arr)['sharpe']:.2f}")
        print(f"  Bootstrap avg: {np.percentile(bm,5):+.3%} / {np.percentile(bm,50):+.3%} / {np.percentile(bm,95):+.3%}")
        print(f"  Bootstrap Sharpe: {np.percentile(bs,5):.2f} / {np.percentile(bs,50):.2f} / {np.percentile(bs,95):.2f}")
        print(f"  P(avg>0): {np.mean(bm>0):.1%}  P(Sharpe>1): {np.mean(bs>1):.1%}")

    # Save
    out_path = "data/reports/tsmom_binance_enhanced.csv"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    rdf.to_csv(out_path, index=False)
    print(f"\n  Results saved to {out_path}")

    elapsed = time.time() - t0
    print(f"\n{'=' * 90}")
    print(f"Total: {elapsed:.0f}s ({elapsed/60:.1f}min)")
    print(f"{'=' * 90}")


if __name__ == "__main__":
    main()
