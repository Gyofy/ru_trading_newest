"""
Binance Public Data Feature Builder
-------------------------------------
data/raw/binance_public/ 에 저장된 CSV를 읽어
OHLCV DataFrame에 실거래소 마이크로스트럭처 피처를 추가합니다.

추가 피처:
  OI  : oi_sum, oi_value, oi_chg_{3,6,12,24}, oi_z_24
  L/S : ls_ratio, ls_top_ratio, ls_diverge
  Taker: taker_ratio, taker_ratio_chg_{3,6,12}, taker_ratio_z_24
  Funding: funding_rate, funding_cum_{12,24,48}, funding_sign

호출 방식:
  df = add_binance_public_features(df, coin="SOL")

데이터 없음 → 조용히 원본 df 반환 (graceful degradation).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger("live_bot.features.binance_public")

# 프로젝트 루트 기준 데이터 경로
_DEFAULT_DATA_DIR = Path(__file__).resolve().parents[3] / "data" / "raw" / "binance_public"

# coin 단축명 → Binance symbol 매핑
_SYMBOL_MAP = {
    "BTC": "BTCUSDT", "ETH": "ETHUSDT", "SOL": "SOLUSDT",
    "XRP": "XRPUSDT", "ADA": "ADAUSDT", "DOT": "DOTUSDT",
    "LINK": "LINKUSDT", "BNB": "BNBUSDT", "DOGE": "DOGEUSDT", "AVAX": "AVAXUSDT",
}


# ── 내부 로더 ──────────────────────────────────────────────────────────────

def _load_metrics(symbol: str, start: pd.Timestamp, end: pd.Timestamp,
                  data_dir: Path) -> pd.DataFrame:
    """metrics CSV(5분봉)를 로드해 DatetimeIndex DataFrame 반환."""
    metrics_dir = data_dir / "metrics" / symbol
    if not metrics_dir.exists():
        return pd.DataFrame()

    dates = pd.date_range(start.date(), end.date(), freq="D")
    frames = []
    for d in dates:
        fp = metrics_dir / f"{symbol}-metrics-{d.strftime('%Y-%m-%d')}.csv"
        if fp.exists():
            try:
                df = pd.read_csv(fp, parse_dates=["create_time"])
                frames.append(df)
            except Exception as e:
                logger.debug("metrics load error %s: %s", fp, e)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values("create_time").drop_duplicates("create_time")
    out = out.set_index("create_time")
    out.index = out.index.tz_localize(None)  # naive UTC
    return out[["sum_open_interest", "sum_open_interest_value",
                "count_long_short_ratio", "count_toptrader_long_short_ratio",
                "sum_taker_long_short_vol_ratio"]]


def _load_funding(symbol: str, start: pd.Timestamp, end: pd.Timestamp,
                  data_dir: Path) -> pd.DataFrame:
    """fundingRate CSV(8h)를 로드해 DatetimeIndex DataFrame 반환."""
    fund_dir = data_dir / "funding_rate" / symbol
    if not fund_dir.exists():
        return pd.DataFrame()

    months = pd.date_range(
        start.to_period("M").to_timestamp(),
        end.to_period("M").to_timestamp(),
        freq="MS",
    )
    frames = []
    for m in months:
        fp = fund_dir / f"{symbol}-fundingRate-{m.strftime('%Y-%m')}.csv"
        if fp.exists():
            try:
                df = pd.read_csv(fp)
                frames.append(df)
            except Exception as e:
                logger.debug("funding load error %s: %s", fp, e)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    out["ts"] = pd.to_datetime(out["calc_time"], unit="ms", utc=True).dt.tz_localize(None)
    out = out.sort_values("ts").drop_duplicates("ts").set_index("ts")
    return out[["last_funding_rate"]]


# ── 피처 계산 ──────────────────────────────────────────────────────────────

def _merge_to_df(df: pd.DataFrame, ext: pd.DataFrame,
                 col_map: dict[str, str]) -> pd.DataFrame:
    """외부 시계열을 df(1m 바)에 merge_asof로 조인 후 ffill."""
    if ext.empty:
        return df

    df_idx = df.index
    if df_idx.tz is not None:
        df_idx_naive = df_idx.tz_localize(None)
    else:
        df_idx_naive = df_idx

    # 기준 프레임: df 인덱스를 열로 만들어 merge_asof (dtype을 ns로 통일)
    left = pd.DataFrame({"_ts": df_idx_naive.astype("datetime64[ns]")})
    right = ext.reset_index().rename(columns={ext.index.name: "_ts"})
    right["_ts"] = right["_ts"].astype("datetime64[ns]")
    right = right.rename(columns=col_map)

    merged = pd.merge_asof(left.sort_values("_ts"),
                           right[["_ts"] + list(col_map.values())].sort_values("_ts"),
                           on="_ts", direction="backward")
    merged = merged.set_index("_ts")
    merged.index = df_idx  # 원래 인덱스(tz 포함) 복원

    for new_col in col_map.values():
        df[new_col] = merged[new_col].values
    return df


def _add_oi_features(df: pd.DataFrame) -> pd.DataFrame:
    oi = df["oi_sum"].replace(0, np.nan)
    for w in [3, 6, 12, 24]:
        df[f"oi_chg_{w}"] = oi.pct_change(w)
    roll24 = oi.rolling(24, min_periods=6)
    df["oi_z_24"] = (oi - roll24.mean()) / (roll24.std() + 1e-10)
    return df


def _add_ls_features(df: pd.DataFrame) -> pd.DataFrame:
    # 롱숏 비율: >1=롱 우세, <1=숏 우세
    df["ls_diverge"] = df["ls_ratio"] - df["ls_top_ratio"]  # 개미 vs 고래 gap
    return df


def _add_taker_features(df: pd.DataFrame) -> pd.DataFrame:
    tr = df["taker_ratio"].replace(0, np.nan)
    for w in [3, 6, 12]:
        df[f"taker_ratio_chg_{w}"] = tr.pct_change(w)
    roll24 = tr.rolling(24, min_periods=6)
    df["taker_ratio_z_24"] = (tr - roll24.mean()) / (roll24.std() + 1e-10)
    return df


def _add_funding_features(df: pd.DataFrame) -> pd.DataFrame:
    fr = df["funding_rate"]
    for w in [12, 24, 48]:
        df[f"funding_cum_{w}"] = fr.rolling(w, min_periods=1).sum()
    df["funding_sign"] = np.sign(fr)
    return df


# ── 공개 API ───────────────────────────────────────────────────────────────

def add_binance_public_features(
    df: pd.DataFrame,
    coin: str,
    data_dir: str | Path | None = None,
) -> pd.DataFrame:
    """OHLCV DataFrame에 Binance public data 피처를 추가한다.

    Parameters
    ----------
    df   : OHLCV + 기존 피처 DataFrame (DatetimeIndex)
    coin : 'SOL', 'BTC' 등 단축명
    data_dir : data/raw/binance_public 경로 (None=자동)

    Returns
    -------
    피처가 추가된 DataFrame (데이터 없으면 원본 반환)
    """
    symbol = _SYMBOL_MAP.get(coin.upper(), coin.upper() + "USDT")
    data_dir = Path(data_dir) if data_dir else _DEFAULT_DATA_DIR

    if df.empty or not isinstance(df.index, pd.DatetimeIndex):
        return df

    start = df.index[0]
    end   = df.index[-1]

    # ── metrics 로드 & 조인 ────────────────────────────────────────────────
    metrics = _load_metrics(symbol, start, end, data_dir)
    if not metrics.empty:
        df = _merge_to_df(df, metrics, {
            "sum_open_interest":            "oi_sum",
            "sum_open_interest_value":      "oi_value",
            "count_long_short_ratio":       "ls_ratio",
            "count_toptrader_long_short_ratio": "ls_top_ratio",
            "sum_taker_long_short_vol_ratio":   "taker_ratio",
        })
        df = _add_oi_features(df)
        df = _add_ls_features(df)
        df = _add_taker_features(df)
        logger.debug("[BinancePublic] %s: metrics 피처 추가 완료", coin)
    else:
        logger.debug("[BinancePublic] %s: metrics 없음, 스킵", coin)

    # ── funding_rate 로드 & 조인 ───────────────────────────────────────────
    funding = _load_funding(symbol, start, end, data_dir)
    if not funding.empty:
        df = _merge_to_df(df, funding, {"last_funding_rate": "funding_rate"})
        df = _add_funding_features(df)
        logger.debug("[BinancePublic] %s: funding 피처 추가 완료", coin)
    else:
        logger.debug("[BinancePublic] %s: funding 없음, 스킵", coin)

    return df
