"""OHLCV 데이터 수집기 (yfinance 기반).

상위 10개 가상화폐의 5분봉/1시간봉 OHLCV 데이터를 수집합니다.
Yahoo Finance API 사용 (무료, 인증 불필요).

Tier 1 피처: VWAP, cross-asset correlation, 시간 피처, return distribution
성분분해: STL(가용시) / EMA 경량 분해 (trend, seasonal, residual) + SVD latent factors
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import json
import ta
from scipy.linalg import svd as scipy_svd
from src.utils.config import get_tactical
from src.data.crawlers.signal_features import add_signal_features

# ---------------------------------------------------------------------------
# Tier 1 + 성분분해 유틸리티
# ---------------------------------------------------------------------------


def _add_vwap(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Rolling VWAP (Volume-Weighted Average Price) + deviation.

    Uses rolling window instead of cumulative to maintain signal quality
    over long series. Cumulative VWAP converges to long-run mean and
    becomes useless after ~100 bars.
    """
    vp = df["close"] * df["volume"]
    roll_vp = vp.rolling(window, min_periods=1).sum()
    roll_vol = df["volume"].rolling(window, min_periods=1).sum()
    df["vwap"] = roll_vp / roll_vol.replace(0, np.nan)
    df["vwap_deviation"] = (df["close"] - df["vwap"]) / df["vwap"].replace(0, np.nan)
    return df


def _add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """시간 피처: hour_of_day, day_of_week → sin/cos 인코딩.

    크립토 아시아/유럽/미국 세션별 패턴 포착.
    """
    idx = df.index
    hour = idx.hour + idx.minute / 60.0
    dow = idx.dayofweek

    df["hour_sin"] = np.sin(2 * np.pi * hour / 24.0)
    df["hour_cos"] = np.cos(2 * np.pi * hour / 24.0)
    df["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    df["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
    return df


def _add_return_distribution(df: pd.DataFrame, ret_col: str, window: int = 20) -> pd.DataFrame:
    """Rolling return distribution: skewness, kurtosis.

    정규분포 이탈 = 꼬리 리스크 / 비대칭 모멘텀 신호.
    """
    r = df[ret_col]
    df["ret_skew"] = r.rolling(window, min_periods=window // 2).skew()
    df["ret_kurtosis"] = r.rolling(window, min_periods=window // 2).kurt()
    return df


def _add_garman_klass_vol(df: pd.DataFrame, window: int = 20) -> pd.DataFrame:
    """Garman-Klass volatility: OHLC 4개 모두 사용, close-to-close 대비 ~5배 효율."""
    log_hl = np.log(df["high"] / df["low"].replace(0, np.nan)) ** 2
    log_co = np.log(df["close"] / df["open"].replace(0, np.nan)) ** 2
    gk = 0.5 * log_hl - (2.0 * np.log(2.0) - 1.0) * log_co
    df["gk_volatility"] = gk.rolling(window, min_periods=window // 2).mean().apply(np.sqrt)
    return df


def _add_decomposition(df: pd.DataFrame, period: int, min_bars_factor: int = 2) -> pd.DataFrame:
    """시계열 성분분해: trend / seasonal / residual.

    STL 가용하면 사용, 아니면 EMA 경량 분해 fallback.
    period: 1일 주기 바 수 (5분봉=288, 1시간봉=24).
    """
    close = df["close"].copy()
    n = len(close)

    if n < period * min_bars_factor:
        return _add_ema_decomposition(df, period)

    try:
        from statsmodels.tsa.seasonal import STL
        stl = STL(close, period=period, robust=True)
        result = stl.fit()
        df["decomp_trend"] = (result.trend / close).fillna(0)
        df["decomp_seasonal"] = (result.seasonal / close).fillna(0)
        df["decomp_residual"] = (result.resid / close).fillna(0)
    except Exception:
        return _add_ema_decomposition(df, period)

    df["trend_slope"] = df["decomp_trend"].diff(3)
    s_std = df["decomp_seasonal"].rolling(period, min_periods=period // 2).std()
    r_std = df["decomp_residual"].rolling(period, min_periods=period // 2).std()
    df["seasonal_strength"] = s_std / (s_std + r_std + 1e-10)
    return df


def _add_ema_decomposition(df: pd.DataFrame, period: int) -> pd.DataFrame:
    """EMA 경량 분해 (STL fallback)."""
    close = df["close"]
    trend = close.ewm(span=period, adjust=False).mean()
    detrended = close - trend
    seasonal = detrended.rolling(period, min_periods=period // 2, center=False).mean().fillna(0)
    residual = close - trend - seasonal

    df["decomp_trend"] = (trend / close).fillna(0)
    df["decomp_seasonal"] = (seasonal / close).fillna(0)
    df["decomp_residual"] = (residual / close).fillna(0)
    df["trend_slope"] = df["decomp_trend"].diff(3)
    s_std = df["decomp_seasonal"].rolling(period, min_periods=period // 2).std()
    r_std = df["decomp_residual"].rolling(period, min_periods=period // 2).std()
    df["seasonal_strength"] = s_std / (s_std + r_std + 1e-10)
    return df


def _add_svd_features(df: pd.DataFrame, window: int = 48, n_components: int = 3) -> pd.DataFrame:
    """SVD latent factors: rolling OHLCV window → 상위 성분 추출.

    노이즈 제거 + 잠재 구조(공통 변동 축) 포착.
    stride로 연산 최적화 → 중간 바는 보간.
    """
    cols = [c for c in ["close", "high", "low"] if c in df.columns]
    if len(cols) < 2:
        return df

    # volume은 0→inf 문제가 있어 log-ratio로 대체
    ret_df = df[cols].pct_change().fillna(0)
    if "volume" in df.columns:
        vol = df["volume"].replace(0, np.nan)
        ret_df["volume"] = np.log1p(vol).diff().fillna(0)
    ret_matrix = ret_df.replace([np.inf, -np.inf], 0)
    n = len(ret_matrix)

    if n < window + 10:
        for i in range(n_components):
            df[f"svd_factor_{i}"] = 0.0
        df["svd_explained_ratio"] = 0.0
        return df

    stride = max(1, window // 6)
    factors = np.full((n, n_components), np.nan)
    explained = np.full(n, np.nan)
    values = ret_matrix.values

    # stride 적용하되 마지막 바는 반드시 포함
    indices = list(range(window, n, stride))
    if indices and indices[-1] != n - 1 and n - 1 >= window:
        indices.append(n - 1)

    for i in indices:
        chunk = values[i - window:i]
        with np.errstate(invalid="ignore"):
            std = np.nanstd(chunk, axis=0)
        std[std < 1e-10] = 1.0
        chunk_norm = (chunk - np.nanmean(chunk, axis=0)) / std
        try:
            U, S, _ = scipy_svd(chunk_norm, full_matrices=False)
            total_var = (S ** 2).sum()
            if total_var > 1e-10:
                for k in range(min(n_components, len(S))):
                    factors[i, k] = U[-1, k] * S[k]
                explained[i] = (S[:n_components] ** 2).sum() / total_var
        except Exception:
            pass

    for k in range(n_components):
        col = f"svd_factor_{k}"
        df[col] = factors[:, k]
        df[col] = df[col].interpolate(method="linear", limit_direction="forward").fillna(0)

    df["svd_explained_ratio"] = explained
    df["svd_explained_ratio"] = df["svd_explained_ratio"].interpolate(
        method="linear", limit_direction="forward"
    ).fillna(0)
    return df


def _add_cross_asset_correlation(
    all_data: dict[str, pd.DataFrame], target: str, window: int = 20
) -> pd.DataFrame:
    """Cross-asset rolling correlation: BTC/ETH/SPX 대비 상관 변화.

    all_data에 다른 코인 데이터가 있으면 활용,
    없으면 스킵 (파이프라인 후반에 별도 호출 가능).
    """
    df = all_data[target].copy()
    target_ret = df["close"].pct_change()

    ref_pairs = {"BTC": "corr_btc", "ETH": "corr_eth"}
    for ref, col_name in ref_pairs.items():
        if ref == target or ref not in all_data:
            df[col_name] = 0.0
            continue
        ref_df = all_data[ref]
        ref_ret = ref_df["close"].pct_change().reindex(df.index, method="ffill")
        df[col_name] = target_ret.rolling(window, min_periods=window // 2).corr(ref_ret).fillna(0)

    return df


# 상위 10개 코인 (Yahoo Finance 심볼)
TOP10_YAHOO = {
    "BTC": "BTC-USD",
    "ETH": "ETH-USD",
    "SOL": "SOL-USD",
    "XRP": "XRP-USD",
    "ADA": "ADA-USD",
    "DOGE": "DOGE-USD",
    "AVAX": "AVAX-USD",
    "DOT": "DOT-USD",
    "LINK": "LINK-USD",
    "BNB": "BNB-USD",
}


def resample_to_4h(df: pd.DataFrame) -> pd.DataFrame:
    """Resample 1h bars to 4h bars."""
    ohlcv_rules = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
    }
    # Only resample columns that exist
    rules = {k: v for k, v in ohlcv_rules.items() if k in df.columns}
    resampled = df.resample('4h').agg(rules).dropna()
    return resampled


def fetch_ohlcv(ticker: str, yahoo_symbol: str, period: str = "365d", interval: str = "1h") -> pd.DataFrame:
    """단일 심볼의 OHLCV 데이터를 가져옵니다."""
    df = yf.download(yahoo_symbol, period=period, interval=interval, progress=False)

    if df.empty:
        return pd.DataFrame()

    # MultiIndex 컬럼 처리
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.rename(columns={
        "Open": "open", "High": "high", "Low": "low",
        "Close": "close", "Volume": "volume",
    })

    # Adj Close가 있으면 제거
    if "Adj Close" in df.columns:
        df.drop(columns=["Adj Close"], inplace=True)

    df = df[["open", "high", "low", "close", "volume"]]
    return df


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """기술적 지표를 추가합니다."""
    if len(df) < 50:
        return df

    # 추세
    df["sma_7"] = ta.trend.sma_indicator(df["close"], window=7)
    df["sma_20"] = ta.trend.sma_indicator(df["close"], window=20)
    df["sma_50"] = ta.trend.sma_indicator(df["close"], window=50)
    df["ema_12"] = ta.trend.ema_indicator(df["close"], window=12)
    df["ema_26"] = ta.trend.ema_indicator(df["close"], window=26)

    # MACD
    macd_obj = ta.trend.MACD(df["close"])
    df["macd"] = macd_obj.macd()
    df["macd_signal"] = macd_obj.macd_signal()
    df["macd_hist"] = macd_obj.macd_diff()

    # RSI
    df["rsi_14"] = ta.momentum.rsi(df["close"], window=14)

    # 볼린저 밴드
    bb = ta.volatility.BollingerBands(df["close"], window=20, window_dev=2)
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["close"]

    # ATR
    df["atr_14"] = ta.volatility.average_true_range(df["high"], df["low"], df["close"], window=14)

    # 거래량 지표
    df["volume_sma_20"] = df["volume"].rolling(20).mean()
    df["volume_ratio"] = df["volume"] / df["volume_sma_20"].replace(0, np.nan)

    # 수익률 / 모멘텀 (ffill 후 계산으로 NaN 최소화)
    df["returns_1h"] = df["close"].pct_change()
    df["returns_4h"] = df["close"].pct_change(4)
    df["returns_24h"] = df["close"].pct_change(24)
    df["momentum_3"] = df["close"].pct_change(3)
    df["momentum_6"] = df["close"].pct_change(6)
    df["momentum_12"] = df["close"].pct_change(12)

    # Forward fill로 초기 NaN 최소화
    df.ffill(inplace=True)
    df.fillna(0, inplace=True)  # no bfill: prevents future data leakage into early bars

    # 변동성
    df["volatility_12h"] = df["returns_1h"].rolling(12).std()
    df["volatility_24h"] = df["returns_1h"].rolling(24).std()

    # 거래량 모멘텀
    df["vol_momentum_6"] = df["volume"].pct_change(6)
    df["vol_momentum_24"] = df["volume"].pct_change(24)

    # 가격 위치 (24h 범위 내 0~1)
    rolling_high = df["high"].rolling(24).max()
    rolling_low = df["low"].rolling(24).min()
    df["price_position"] = (df["close"] - rolling_low) / (rolling_high - rolling_low + 1e-10)

    # --- Tier 1: VWAP, 시간, 수익률 분포, Garman-Klass ---
    df = _add_vwap(df)
    df = _add_time_features(df)
    df = _add_return_distribution(df, "returns_1h", window=20)
    df = _add_garman_klass_vol(df, window=20)

    # --- 성분분해: STL (period=24 = 1일 주기) + SVD ---
    df = _add_decomposition(df, period=24)
    df = _add_svd_features(df, window=48, n_components=3)

    # NaN 정리
    df.ffill(inplace=True)
    df.fillna(0, inplace=True)  # no bfill: prevents future data leakage into early bars
    df.replace([np.inf, -np.inf], 0, inplace=True)

    return df


def add_technical_indicators_5m(df: pd.DataFrame) -> pd.DataFrame:
    """5분봉 전용 기술적 지표를 추가합니다."""
    if len(df) < 50:
        return df

    # 추세 (5분봉 기준 window 조정)
    df["sma_12"] = ta.trend.sma_indicator(df["close"], window=12)   # 1시간
    df["sma_48"] = ta.trend.sma_indicator(df["close"], window=48)   # 4시간
    df["sma_144"] = ta.trend.sma_indicator(df["close"], window=144) # 12시간
    df["ema_12"] = ta.trend.ema_indicator(df["close"], window=12)
    df["ema_26"] = ta.trend.ema_indicator(df["close"], window=26)

    # MACD
    macd_obj = ta.trend.MACD(df["close"])
    df["macd"] = macd_obj.macd()
    df["macd_signal"] = macd_obj.macd_signal()
    df["macd_hist"] = macd_obj.macd_diff()

    # RSI
    df["rsi_14"] = ta.momentum.rsi(df["close"], window=14)

    # 볼린저 밴드
    bb = ta.volatility.BollingerBands(df["close"], window=20, window_dev=2)
    df["bb_upper"] = bb.bollinger_hband()
    df["bb_lower"] = bb.bollinger_lband()
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / df["close"]

    # ATR
    df["atr_14"] = ta.volatility.average_true_range(df["high"], df["low"], df["close"], window=14)

    # 거래량 지표
    df["volume_sma_20"] = df["volume"].rolling(20).mean()
    df["volume_ratio"] = df["volume"] / df["volume_sma_20"].replace(0, np.nan)

    # 수익률 / 모멘텀 (5분봉 기준)
    df["returns_1bar"] = df["close"].pct_change()       # 5분
    df["returns_6bar"] = df["close"].pct_change(6)      # 30분
    df["returns_12bar"] = df["close"].pct_change(12)    # 60분
    df["returns_24bar"] = df["close"].pct_change(24)    # 2시간
    df["momentum_3"] = df["close"].pct_change(3)        # 15분
    df["momentum_6"] = df["close"].pct_change(6)        # 30분
    df["momentum_12"] = df["close"].pct_change(12)      # 60분

    df.ffill(inplace=True)
    df.fillna(0, inplace=True)  # no bfill: prevents future data leakage into early bars

    # 변동성
    df["volatility_12bar"] = df["returns_1bar"].rolling(12).std()   # 60분 변동성
    df["volatility_24bar"] = df["returns_1bar"].rolling(24).std()   # 2시간 변동성

    # 거래량 모멘텀
    df["vol_momentum_6"] = df["volume"].pct_change(6)
    df["vol_momentum_12"] = df["volume"].pct_change(12)

    # 가격 위치 (12bar = 1시간 범위 내 0~1)
    rolling_high = df["high"].rolling(24).max()
    rolling_low = df["low"].rolling(24).min()
    df["price_position"] = (df["close"] - rolling_low) / (rolling_high - rolling_low + 1e-10)

    # --- Tier 1: VWAP, 시간, 수익률 분포, Garman-Klass ---
    df = _add_vwap(df)
    df = _add_time_features(df)
    df = _add_return_distribution(df, "returns_1bar", window=24)   # 24bar=2h
    df = _add_garman_klass_vol(df, window=24)

    # --- 성분분해: STL (period=288 = 5분봉 1일 주기) + SVD ---
    df = _add_decomposition(df, period=288)
    df = _add_svd_features(df, window=48, n_components=3)

    # NaN 정리
    df.ffill(inplace=True)
    df.fillna(0, inplace=True)  # no bfill: prevents future data leakage into early bars
    df.replace([np.inf, -np.inf], 0, inplace=True)

    return df


def fetch_all_top10(period: str = "365d", interval: str = "1h") -> dict[str, pd.DataFrame]:
    """상위 10개 코인의 OHLCV + 기술지표 + Tier1 피처를 수집합니다.

    v4: 1h fetch → 4h resample → 기술지표 계산.
    """
    tactical = get_tactical()
    bar_min = tactical.get("bar_minutes", 240)
    decomp_period = tactical.get("decomposition_period", 42)

    result = {}
    for ticker, yahoo_sym in TOP10_YAHOO.items():
        try:
            df = fetch_ohlcv(ticker, yahoo_sym, period, interval)
            if not df.empty:
                # Resample 1h → 4h if bar_minutes >= 240
                if bar_min >= 240 and interval in ("1h", "60m"):
                    df = resample_to_4h(df)
                    print(f"  [RESAMPLE] {ticker}: 1h → 4h, {len(df)} bars")

                # Use generic technical indicators (not 5m-specific)
                df = add_technical_indicators(df)

                # Override decomposition with config period
                df = _add_decomposition(df, period=decomp_period)

                # Signal analysis features (wavelet, FFT, Hilbert, entropy, etc.)
                df = add_signal_features(df, verbose=True)

                result[ticker] = df
                print(f"  [OK] {ticker}: {len(df)} bars (4h), {len(df.columns)} cols, latest ${df['close'].iloc[-1]:,.2f}")
            else:
                print(f"  [EMPTY] {ticker}")
        except Exception as e:
            print(f"  [FAIL] {ticker}: {e}")

    # Cross-asset correlation (모든 코인 수집 후 계산)
    if len(result) >= 2:
        for ticker in list(result.keys()):
            try:
                result[ticker] = _add_cross_asset_correlation(result, ticker, window=20)
            except Exception:
                pass

    return result


def save_ohlcv_data(data: dict[str, pd.DataFrame], base_dir: str = "data") -> Path:
    """OHLCV 데이터를 저장합니다."""
    date_str = datetime.now().strftime("%Y%m%d")
    save_dir = Path(base_dir) / "raw" / date_str
    save_dir.mkdir(parents=True, exist_ok=True)

    for ticker, df in data.items():
        path = save_dir / f"{ticker}_ohlcv_1h.parquet"
        df.to_parquet(path)

    # 요약 JSON
    summary = {}
    for ticker, df in data.items():
        if len(df) > 0:
            latest = df.iloc[-1]
            summary[ticker] = {
                "price": float(latest["close"]),
                "change_24h_pct": float(latest.get("returns_24h", 0)) * 100 if pd.notna(latest.get("returns_24h")) else 0,
                "rsi": float(latest["rsi_14"]) if pd.notna(latest.get("rsi_14")) else None,
                "volume": float(latest["volume"]),
                "bars": len(df),
            }

    with open(save_dir / "ohlcv_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    return save_dir


if __name__ == "__main__":
    print("=== Top 10 Crypto OHLCV (yfinance, 1h → 4h resample) ===")
    data = fetch_all_top10("365d", "1h")
    save_dir = save_ohlcv_data(data)
    print(f"\nSaved to: {save_dir}")
