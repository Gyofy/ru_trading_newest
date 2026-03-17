"""시계열 신호분석 + 파형분석 피처 엔지니어링.

기존 기술지표(50개)에 아래 카테고리를 추가:
1. Wavelet 분해 (다중 스케일 에너지/패턴)
2. FFT 스펙트럼 (주파수 도메인)
3. Hilbert 변환 (순간 진폭/위상/주파수)
4. 엔트로피 (시장 불확실성/복잡도)
5. Hurst 지수 + 자기상관 (추세 지속성)
6. 추가 기술지표 (Ichimoku, ADX, Stochastic, OBV, MFI, CMF)
7. 미시구조 (High-Low 패턴, 가격 분포, Z-score)
8. 변화점 감지 (CUSUM)

사용법:
    df = add_signal_features(df)  # OHLCV DataFrame에 ~120개 피처 추가
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
import pywt
from scipy.signal import hilbert
from scipy.fft import fft
from scipy.stats import entropy as scipy_entropy

warnings.filterwarnings("ignore", module="pywt")


# ══════════════════════════════════════════════════════════════
# 1. WAVELET DECOMPOSITION (다중 스케일 에너지)
# ══════════════════════════════════════════════════════════════

def _add_wavelet_features(df: pd.DataFrame, col: str = "close",
                          wavelet: str = "db4", max_level: int = 4) -> pd.DataFrame:
    """Discrete Wavelet Transform → 스케일별 에너지 + 계수 통계.

    Level 1 = 고주파(노이즈), Level 4 = 저주파(추세).
    4h bar 기준: L1≈8h, L2≈16h, L3≈32h(1.3일), L4≈64h(2.7일)
    """
    series = df[col].values.astype(np.float64)
    n = len(series)

    # 전체 시계열 DWT
    coeffs = pywt.wavedec(series, wavelet, level=max_level)
    # coeffs[0] = approximation (trend), coeffs[1..4] = detail (noise→trend)

    # Rolling wavelet: 최근 64 bars 윈도우에서 계산
    window = min(64, n // 2)
    for level in range(1, max_level + 1):
        energy_col = f"wt_energy_L{level}"
        ratio_col = f"wt_ratio_L{level}"
        df[energy_col] = 0.0
        df[ratio_col] = 0.0

    df["wt_trend_energy"] = 0.0
    df["wt_noise_ratio"] = 0.0

    # 윈도우 크기에 맞는 최대 레벨 계산
    safe_level = min(max_level, pywt.dwt_max_level(window, pywt.Wavelet(wavelet).dec_len))

    for i in range(window, n):
        segment = series[i - window:i]
        try:
            c = pywt.wavedec(segment, wavelet, level=safe_level)
            total_energy = sum(np.sum(d ** 2) for d in c) + 1e-12

            for level in range(1, max_level + 1):
                if level < len(c):
                    detail = c[level]
                    e = np.sum(detail ** 2)
                    df.iloc[i, df.columns.get_loc(f"wt_energy_L{level}")] = e
                    df.iloc[i, df.columns.get_loc(f"wt_ratio_L{level}")] = e / total_energy

            # 추세 에너지 (approximation)
            trend_e = np.sum(c[0] ** 2)
            df.iloc[i, df.columns.get_loc("wt_trend_energy")] = trend_e / total_energy

            # 노이즈 비율 (L1 / total)
            noise_e = np.sum(c[1] ** 2) if len(c) > 1 else 0
            df.iloc[i, df.columns.get_loc("wt_noise_ratio")] = noise_e / total_energy
        except Exception:
            continue

    return df


# ══════════════════════════════════════════════════════════════
# 2. FFT SPECTRAL FEATURES (주파수 도메인)
# ══════════════════════════════════════════════════════════════

def _add_fft_features(df: pd.DataFrame, col: str = "close",
                      window: int = 64) -> pd.DataFrame:
    """Rolling FFT → 주파수 도메인 특성.

    - 지배 주파수 (dominant frequency)
    - 스펙트럼 엔트로피 (spectral entropy)
    - 저주파/고주파 에너지 비율
    - 스펙트럼 엣지 주파수
    """
    series = df[col].pct_change().fillna(0).values.astype(np.float64)
    n = len(series)

    cols = ["fft_dominant_freq", "fft_spectral_entropy", "fft_low_high_ratio",
            "fft_spectral_edge", "fft_spectral_centroid"]
    for c in cols:
        df[c] = 0.0

    half_w = window // 2

    for i in range(window, n):
        segment = series[i - window:i]
        # Hanning window로 스펙트럼 누출 방지
        windowed = segment * np.hanning(window)
        spectrum = np.abs(fft(windowed))[:half_w]
        freqs = np.fft.fftfreq(window, d=1.0)[:half_w]

        power = spectrum ** 2
        total_power = np.sum(power) + 1e-12

        # 지배 주파수
        dom_idx = np.argmax(power[1:]) + 1  # DC 제외
        df.iloc[i, df.columns.get_loc("fft_dominant_freq")] = freqs[dom_idx] if dom_idx < len(freqs) else 0

        # 스펙트럼 엔트로피
        p_norm = power / total_power
        p_norm = p_norm[p_norm > 0]
        df.iloc[i, df.columns.get_loc("fft_spectral_entropy")] = scipy_entropy(p_norm)

        # 저주파/고주파 비율 (50% split)
        mid = half_w // 2
        low_e = np.sum(power[:mid]) + 1e-12
        high_e = np.sum(power[mid:]) + 1e-12
        df.iloc[i, df.columns.get_loc("fft_low_high_ratio")] = low_e / high_e

        # 스펙트럼 엣지 (95% 에너지 포인트)
        cum_power = np.cumsum(power) / total_power
        edge_idx = np.searchsorted(cum_power, 0.95)
        df.iloc[i, df.columns.get_loc("fft_spectral_edge")] = freqs[min(edge_idx, len(freqs) - 1)]

        # 스펙트럼 중심 (centroid)
        centroid = np.sum(freqs * power) / total_power
        df.iloc[i, df.columns.get_loc("fft_spectral_centroid")] = centroid

    return df


# ══════════════════════════════════════════════════════════════
# 3. HILBERT TRANSFORM (순간 진폭/위상/주파수)
# ══════════════════════════════════════════════════════════════

def _add_hilbert_features(df: pd.DataFrame, col: str = "close",
                          window: int = 48) -> pd.DataFrame:
    """Hilbert 변환 → 해석적 신호(analytic signal) 특성.

    - 순간 진폭 (envelope): 변동성의 실시간 측정
    - 순간 위상: 시장 사이클 위치
    - 순간 주파수: 사이클 속도 변화
    """
    returns = df[col].pct_change().fillna(0).values.astype(np.float64)
    n = len(returns)

    df["ht_amplitude"] = 0.0
    df["ht_phase"] = 0.0
    df["ht_inst_freq"] = 0.0
    df["ht_amplitude_trend"] = 0.0

    for i in range(window, n):
        segment = returns[i - window:i]
        try:
            analytic = hilbert(segment)
            amplitude = np.abs(analytic)
            phase = np.angle(analytic)

            df.iloc[i, df.columns.get_loc("ht_amplitude")] = amplitude[-1]
            df.iloc[i, df.columns.get_loc("ht_phase")] = phase[-1]

            # 순간 주파수 = d(phase)/dt
            inst_freq = np.diff(np.unwrap(phase))
            df.iloc[i, df.columns.get_loc("ht_inst_freq")] = inst_freq[-1] if len(inst_freq) > 0 else 0

            # 진폭 추세 (최근 vs 이전)
            half = len(amplitude) // 2
            recent = np.mean(amplitude[half:])
            earlier = np.mean(amplitude[:half]) + 1e-12
            df.iloc[i, df.columns.get_loc("ht_amplitude_trend")] = recent / earlier - 1
        except Exception:
            continue

    return df


# ══════════════════════════════════════════════════════════════
# 4. ENTROPY FEATURES (시장 복잡도/불확실성)
# ══════════════════════════════════════════════════════════════

def _sample_entropy(series: np.ndarray, m: int = 2, r_mult: float = 0.2) -> float:
    """Sample Entropy — 시계열 규칙성 측정. 낮을수록 규칙적."""
    n = len(series)
    if n < m + 2:
        return 0.0
    r = r_mult * np.std(series)
    if r < 1e-12:
        return 0.0

    def _count_matches(template_len):
        count = 0
        templates = np.array([series[i:i + template_len] for i in range(n - template_len)])
        for i in range(len(templates)):
            for j in range(i + 1, len(templates)):
                if np.max(np.abs(templates[i] - templates[j])) <= r:
                    count += 1
        return count

    a = _count_matches(m + 1)
    b = _count_matches(m)
    if b == 0:
        return 0.0
    return -np.log((a + 1e-12) / (b + 1e-12))


def _add_entropy_features(df: pd.DataFrame, col: str = "close",
                          window: int = 48) -> pd.DataFrame:
    """다양한 엔트로피 측정.

    - Shannon Entropy: 수익률 분포의 정보량
    - Sample Entropy: 시계열 복잡도/예측가능성
    - Permutation Entropy: 순서 패턴 복잡도
    """
    returns = df[col].pct_change().fillna(0).values.astype(np.float64)
    n = len(returns)

    df["ent_shannon"] = 0.0
    df["ent_sample"] = 0.0
    df["ent_permutation"] = 0.0

    for i in range(window, n):
        segment = returns[i - window:i]

        # Shannon Entropy (히스토그램 기반)
        hist, _ = np.histogram(segment, bins=10, density=True)
        hist = hist[hist > 0]
        df.iloc[i, df.columns.get_loc("ent_shannon")] = scipy_entropy(hist)

        # Sample Entropy (매 6바마다만 — 연산량 절약)
        if i % 6 == 0:
            se = _sample_entropy(segment, m=2, r_mult=0.2)
            df.iloc[i, df.columns.get_loc("ent_sample")] = se
        elif i > window:
            df.iloc[i, df.columns.get_loc("ent_sample")] = df.iloc[i - 1, df.columns.get_loc("ent_sample")]

        # Permutation Entropy (order pattern complexity)
        order = 3
        pe_count = {}
        for j in range(len(segment) - order + 1):
            pattern = tuple(np.argsort(segment[j:j + order]))
            pe_count[pattern] = pe_count.get(pattern, 0) + 1
        total = sum(pe_count.values())
        pe_probs = np.array([v / total for v in pe_count.values()])
        df.iloc[i, df.columns.get_loc("ent_permutation")] = scipy_entropy(pe_probs)

    return df


# ══════════════════════════════════════════════════════════════
# 5. HURST EXPONENT + AUTOCORRELATION (추세 지속성)
# ══════════════════════════════════════════════════════════════

def _hurst_exponent(series: np.ndarray) -> float:
    """R/S 방법으로 Hurst 지수 추정.
    H > 0.5: 추세 지속, H < 0.5: 평균회귀, H ≈ 0.5: 랜덤워크
    """
    n = len(series)
    if n < 20:
        return 0.5

    max_k = min(n // 2, 50)
    lags = range(10, max_k + 1, 5)
    rs_list = []
    lag_list = []

    for lag in lags:
        subseries = [series[i:i + lag] for i in range(0, n - lag + 1, lag)]
        rs_vals = []
        for s in subseries:
            if len(s) < 2:
                continue
            mean = np.mean(s)
            dev = np.cumsum(s - mean)
            r = np.max(dev) - np.min(dev)
            std = np.std(s)
            if std > 1e-12:
                rs_vals.append(r / std)
        if rs_vals:
            rs_list.append(np.log(np.mean(rs_vals)))
            lag_list.append(np.log(lag))

    if len(rs_list) < 2:
        return 0.5

    # Linear regression: log(R/S) = H * log(lag)
    coeffs = np.polyfit(lag_list, rs_list, 1)
    return np.clip(coeffs[0], 0.0, 1.0)


def _add_hurst_and_autocorr(df: pd.DataFrame, col: str = "close",
                             window: int = 96) -> pd.DataFrame:
    """Rolling Hurst Exponent + Autocorrelation 피처.

    - hurst: 추세/평균회귀 판별
    - acf_lag_1~6: 자기상관 (단기 기억 구조)
    - acf_decay: 자기상관 감소 속도
    """
    returns = df[col].pct_change().fillna(0).values.astype(np.float64)
    n = len(returns)

    df["hurst"] = 0.5
    for lag in [1, 3, 6, 12, 24]:
        df[f"acf_lag_{lag}"] = 0.0
    df["acf_decay_rate"] = 0.0

    step = 6  # Hurst는 매 6바마다 계산 (비용 절약)

    for i in range(window, n):
        segment = returns[i - window:i]

        # Hurst (매 6바마다)
        if i % step == 0:
            h = _hurst_exponent(segment)
            df.iloc[i, df.columns.get_loc("hurst")] = h
        elif i > window:
            df.iloc[i, df.columns.get_loc("hurst")] = df.iloc[i - 1, df.columns.get_loc("hurst")]

        # Autocorrelation at multiple lags
        std = np.std(segment)
        if std < 1e-12:
            continue
        mean = np.mean(segment)
        centered = segment - mean

        acf_values = []
        for lag in [1, 3, 6, 12, 24]:
            if lag < len(segment):
                acf = np.sum(centered[lag:] * centered[:-lag]) / (len(centered) * std ** 2 + 1e-12)
                df.iloc[i, df.columns.get_loc(f"acf_lag_{lag}")] = acf
                acf_values.append(abs(acf))

        # ACF 감소 속도
        if len(acf_values) >= 2:
            decay = acf_values[0] - acf_values[-1]
            df.iloc[i, df.columns.get_loc("acf_decay_rate")] = decay

    return df


# ══════════════════════════════════════════════════════════════
# 6. ADDITIONAL TECHNICAL INDICATORS
# ══════════════════════════════════════════════════════════════

def _add_advanced_technicals(df: pd.DataFrame) -> pd.DataFrame:
    """기존에 없는 추가 기술지표."""
    close = df["close"]
    high = df["high"]
    low = df["low"]
    volume = df["volume"]

    # --- ADX (Average Directional Index) ---
    period = 14
    plus_dm = high.diff().clip(lower=0)
    minus_dm = (-low.diff()).clip(lower=0)
    # 양방향 중 큰 쪽만 유지
    mask = plus_dm > minus_dm
    plus_dm = plus_dm.where(mask, 0)
    minus_dm = minus_dm.where(~mask, 0)

    tr = pd.concat([
        high - low,
        (high - close.shift(1)).abs(),
        (low - close.shift(1)).abs()
    ], axis=1).max(axis=1)

    atr = tr.rolling(period).mean()
    plus_di = 100 * (plus_dm.rolling(period).mean() / (atr + 1e-10))
    minus_di = 100 * (minus_dm.rolling(period).mean() / (atr + 1e-10))
    dx = 100 * ((plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10))
    df["adx_14"] = dx.rolling(period).mean()
    df["plus_di_14"] = plus_di
    df["minus_di_14"] = minus_di
    df["di_diff"] = plus_di - minus_di

    # --- Stochastic Oscillator ---
    period_k = 14
    low_min = low.rolling(period_k).min()
    high_max = high.rolling(period_k).max()
    df["stoch_k"] = 100 * (close - low_min) / (high_max - low_min + 1e-10)
    df["stoch_d"] = df["stoch_k"].rolling(3).mean()
    df["stoch_kd_diff"] = df["stoch_k"] - df["stoch_d"]

    # --- Williams %R ---
    df["williams_r"] = -100 * (high_max - close) / (high_max - low_min + 1e-10)

    # --- OBV (On Balance Volume) ---
    obv = pd.Series(0.0, index=df.index)
    obv_sign = np.sign(close.diff().fillna(0))
    obv = (volume * obv_sign).cumsum()
    df["obv"] = obv
    df["obv_ema_12"] = obv.ewm(span=12).mean()
    df["obv_divergence"] = obv / (obv.ewm(span=12).mean() + 1e-10) - 1

    # --- MFI (Money Flow Index) ---
    typical_price = (high + low + close) / 3
    mf = typical_price * volume
    pos_mf = mf.where(typical_price > typical_price.shift(1), 0).rolling(14).sum()
    neg_mf = mf.where(typical_price <= typical_price.shift(1), 0).rolling(14).sum()
    mfr = pos_mf / (neg_mf + 1e-10)
    df["mfi_14"] = 100 - (100 / (1 + mfr))

    # --- CMF (Chaikin Money Flow) ---
    clv = ((close - low) - (high - close)) / (high - low + 1e-10)
    df["cmf_20"] = (clv * volume).rolling(20).sum() / (volume.rolling(20).sum() + 1e-10)

    # --- Ichimoku Cloud (간소화) ---
    tenkan = (high.rolling(9).max() + low.rolling(9).min()) / 2
    kijun = (high.rolling(26).max() + low.rolling(26).min()) / 2
    df["ichi_tenkan"] = (tenkan - close) / (close + 1e-10)  # 정규화
    df["ichi_kijun"] = (kijun - close) / (close + 1e-10)
    df["ichi_tk_diff"] = (tenkan - kijun) / (close + 1e-10)
    senkou_a = (tenkan + kijun) / 2
    senkou_b = (high.rolling(52).max() + low.rolling(52).min()) / 2
    df["ichi_cloud_width"] = (senkou_a - senkou_b) / (close + 1e-10)
    df["ichi_above_cloud"] = (close > senkou_a.shift(26)).astype(float)

    # --- Keltner Channel ---
    ema_20 = close.ewm(span=20).mean()
    atr_10 = tr.rolling(10).mean()
    df["keltner_upper_dist"] = (ema_20 + 2 * atr_10 - close) / (close + 1e-10)
    df["keltner_lower_dist"] = (close - (ema_20 - 2 * atr_10)) / (close + 1e-10)
    df["keltner_position"] = (close - (ema_20 - 2 * atr_10)) / (4 * atr_10 + 1e-10)

    # --- Donchian Channel ---
    dc_high = high.rolling(20).max()
    dc_low = low.rolling(20).min()
    df["donchian_position"] = (close - dc_low) / (dc_high - dc_low + 1e-10)
    df["donchian_width"] = (dc_high - dc_low) / (close + 1e-10)

    return df


# ══════════════════════════════════════════════════════════════
# 7. MICROSTRUCTURE FEATURES (미시구조)
# ══════════════════════════════════════════════════════════════

def _add_microstructure(df: pd.DataFrame) -> pd.DataFrame:
    """가격 미시구조 피처."""
    close = df["close"]
    high = df["high"]
    low = df["low"]
    open_ = df["open"]

    # Bar 내부 구조
    bar_range = high - low
    df["bar_range_pct"] = bar_range / (close + 1e-10)
    df["upper_shadow"] = (high - pd.concat([close, open_], axis=1).max(axis=1)) / (bar_range + 1e-10)
    df["lower_shadow"] = (pd.concat([close, open_], axis=1).min(axis=1) - low) / (bar_range + 1e-10)
    df["body_ratio"] = (close - open_).abs() / (bar_range + 1e-10)

    # Close position in bar
    df["close_position_in_bar"] = (close - low) / (bar_range + 1e-10)

    # Rolling Z-score
    for w in [12, 24, 48]:
        roll_mean = close.rolling(w).mean()
        roll_std = close.rolling(w).std()
        df[f"zscore_{w}"] = (close - roll_mean) / (roll_std + 1e-10)

    # True Range 정규화
    df["norm_atr"] = df.get("atr_14", pd.Series(0, index=df.index)) / (close + 1e-10)

    # 연속 양봉/음봉 카운트
    is_up = (close > open_).astype(int)
    streak = is_up.copy()
    for i in range(1, len(streak)):
        if is_up.iloc[i] == is_up.iloc[i - 1]:
            streak.iloc[i] = streak.iloc[i - 1] + 1
        else:
            streak.iloc[i] = 1
    df["bar_streak"] = streak * (2 * is_up - 1)  # 양수=연속양봉, 음수=연속음봉

    # Gap (전일 종가 vs 당일 시가)
    df["gap_pct"] = (open_ - close.shift(1)) / (close.shift(1) + 1e-10)

    # 변동성 클러스터링 (GARCH-like feature)
    sq_returns = (close.pct_change() ** 2).fillna(0)
    df["vol_cluster"] = sq_returns.ewm(span=12).mean()
    df["vol_cluster_ratio"] = df["vol_cluster"] / (sq_returns.ewm(span=48).mean() + 1e-10)

    return df


# ══════════════════════════════════════════════════════════════
# 8. CUSUM CHANGE POINT DETECTION
# ══════════════════════════════════════════════════════════════

def _add_cusum(df: pd.DataFrame, col: str = "close",
               threshold: float = 2.0) -> pd.DataFrame:
    """CUSUM (Cumulative Sum) — 변화점 감지.

    - S+/S-: 상방/하방 누적편차
    - cusum_signal: 변화 강도
    """
    returns = df[col].pct_change().fillna(0).values
    n = len(returns)

    s_pos = np.zeros(n)
    s_neg = np.zeros(n)
    mean_ret = 0.0

    for i in range(1, n):
        # Adaptive mean (EMA)
        mean_ret = 0.95 * mean_ret + 0.05 * returns[i]
        s_pos[i] = max(0, s_pos[i - 1] + returns[i] - mean_ret)
        s_neg[i] = min(0, s_neg[i - 1] + returns[i] - mean_ret)

    df["cusum_pos"] = s_pos
    df["cusum_neg"] = s_neg
    df["cusum_signal"] = s_pos + s_neg  # net signal

    # 변화점 이벤트 (threshold 초과 시)
    std_ret = pd.Series(returns).rolling(48).std().fillna(0.01).values
    df["cusum_event"] = ((s_pos > threshold * std_ret) | (s_neg < -threshold * std_ret)).astype(float)

    return df


# ══════════════════════════════════════════════════════════════
# 9. MULTI-TIMEFRAME FEATURES (다중 타임프레임)
# ══════════════════════════════════════════════════════════════

def _add_multi_timeframe(df: pd.DataFrame) -> pd.DataFrame:
    """다중 타임프레임 파생 피처.

    4h 기본에서 12h, 24h, 48h 관점의 피처 생성.
    """
    close = df["close"]
    volume = df["volume"]

    for period in [12, 24, 48]:  # 2일, 4일, 8일
        label = f"mtf_{period}"

        # 수익률
        df[f"{label}_return"] = close.pct_change(period)

        # 변동성 (Parkinson)
        rolling_high = df["high"].rolling(period).max()
        rolling_low = df["low"].rolling(period).min()
        df[f"{label}_parkinson_vol"] = np.log(rolling_high / (rolling_low + 1e-10)) / (2 * np.sqrt(np.log(2)))

        # 가격 위치 (0~1)
        df[f"{label}_price_pos"] = (close - rolling_low) / (rolling_high - rolling_low + 1e-10)

        # 볼륨 트렌드
        vol_now = volume.rolling(period // 2).mean()
        vol_before = volume.shift(period // 2).rolling(period // 2).mean()
        df[f"{label}_vol_trend"] = vol_now / (vol_before + 1e-10) - 1

        # RSI at different timeframes
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(period).mean()
        loss = (-delta.clip(upper=0)).rolling(period).mean()
        rs = gain / (loss + 1e-10)
        df[f"{label}_rsi"] = 100 - (100 / (1 + rs))

    return df


# ══════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════

def add_signal_features(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """모든 신호분석 피처를 DataFrame에 추가.

    Args:
        df: OHLCV DataFrame (open, high, low, close, volume 필수)
        verbose: 진행상황 출력 여부

    Returns:
        피처가 추가된 DataFrame (~120개 신규 피처)
    """
    n_before = len(df.columns)

    if verbose:
        print(f"  [Signal Features] Starting with {n_before} columns, {len(df)} rows")

    # 1. Wavelet (10 features)
    df = _add_wavelet_features(df, "close", wavelet="db4", max_level=4)
    if verbose:
        print(f"  [+] Wavelet: {len(df.columns) - n_before} new")
    n1 = len(df.columns)

    # 2. FFT (5 features)
    df = _add_fft_features(df, "close", window=64)
    if verbose:
        print(f"  [+] FFT: {len(df.columns) - n1} new")
    n2 = len(df.columns)

    # 3. Hilbert Transform (4 features)
    df = _add_hilbert_features(df, "close", window=48)
    if verbose:
        print(f"  [+] Hilbert: {len(df.columns) - n2} new")
    n3 = len(df.columns)

    # 4. Entropy (3 features)
    df = _add_entropy_features(df, "close", window=48)
    if verbose:
        print(f"  [+] Entropy: {len(df.columns) - n3} new")
    n4 = len(df.columns)

    # 5. Hurst + Autocorrelation (7 features)
    df = _add_hurst_and_autocorr(df, "close", window=96)
    if verbose:
        print(f"  [+] Hurst/ACF: {len(df.columns) - n4} new")
    n5 = len(df.columns)

    # 6. Advanced Technicals (~25 features)
    df = _add_advanced_technicals(df)
    if verbose:
        print(f"  [+] Technicals: {len(df.columns) - n5} new")
    n6 = len(df.columns)

    # 7. Microstructure (~15 features)
    df = _add_microstructure(df)
    if verbose:
        print(f"  [+] Microstructure: {len(df.columns) - n6} new")
    n7 = len(df.columns)

    # 8. CUSUM (4 features)
    df = _add_cusum(df, "close")
    if verbose:
        print(f"  [+] CUSUM: {len(df.columns) - n7} new")
    n8 = len(df.columns)

    # 9. Multi-timeframe (15 features)
    df = _add_multi_timeframe(df)
    if verbose:
        print(f"  [+] Multi-TF: {len(df.columns) - n8} new")

    # NaN 정리
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    df.replace([np.inf, -np.inf], 0, inplace=True)

    n_after = len(df.columns)
    if verbose:
        print(f"  [Signal Features] Done: {n_before} → {n_after} columns (+{n_after - n_before} signal features)")

    return df
