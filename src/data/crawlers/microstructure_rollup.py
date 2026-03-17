"""Microstructure Roll-up Features — CVD, OFI, VPIN, Roll Spread, Amihud.

OHLCV 데이터만으로 오더플로우 미시구조 피처를 근사 계산합니다.
(L2 호가창 없이 BVC 기법으로 buy/sell volume 분리)

─────────────────────────────────────────────
Feature Groups:
  1. CVD  (Cumulative Volume Delta)       — 누적 매수/매도 볼륨 불균형
  2. OFI  (Order Flow Imbalance)          — 바 내 주문흐름 강도
  3. VPIN (Vol-sync Prob of Informed Trading) — 정보거래 확률
  4. Roll Spread                          — 추정 bid-ask spread
  5. Amihud Illiquidity                   — 가격충격 유동성 지수
─────────────────────────────────────────────

Usage:
    from src.data.crawlers.microstructure_rollup import add_microstructure_rollup
    df = add_microstructure_rollup(df)       # 37+ features 추가
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# 집계 윈도우 (4h bar 기준)
WINDOWS = [3, 6, 12, 24]   # 12h, 24h, 48h, 96h


# ══════════════════════════════════════════════════════════════
# 1. CVD — Cumulative Volume Delta (BVC 기법)
# ══════════════════════════════════════════════════════════════

def _add_cvd(df: pd.DataFrame) -> pd.DataFrame:
    """Cumulative Volume Delta 피처.

    BVC (Bulk Volume Classification):
        buy_frac = (close - low) / (high - low + ε)
        buy_vol  = buy_frac * volume
        sell_vol = (1 - buy_frac) * volume
        vd       = buy_vol - sell_vol  (단일 바 볼륨 델타)
        cvd      = cumsum(vd)          (누적)

    Alpha hypothesis:
        CVD 상승 + 가격 하락 → bullish divergence (매수세 축적)
        CVD 하락 + 가격 상승 → bearish divergence (매도세 분산)
    """
    high  = df["high"].values
    low   = df["low"].values
    close = df["close"].values
    vol   = df["volume"].values

    bar_range = high - low
    buy_frac  = np.where(bar_range > 0, (close - low) / (bar_range + 1e-10), 0.5)
    buy_frac  = np.clip(buy_frac, 0.0, 1.0)

    buy_vol  = buy_frac * vol
    sell_vol = (1.0 - buy_frac) * vol
    vd       = buy_vol - sell_vol        # Volume Delta (단일 바)
    cvd      = np.cumsum(vd)             # Cumulative VD

    # BVC 기반 CVD
    df["vd"]       = vd
    df["cvd"]      = cvd
    df["buy_frac"] = buy_frac

    # Tick-rule 기반 CVD (보조 — 가격 변화 방향만으로 판단)
    tick_sign  = np.sign(np.diff(close, prepend=close[0]))
    tick_vd    = tick_sign * vol
    df["cvd_tick"] = np.cumsum(tick_vd)

    s_vd      = pd.Series(vd, index=df.index)
    s_tick_vd = pd.Series(tick_vd, index=df.index)
    s_close   = df["close"]
    s_vol     = df["volume"]

    for w in WINDOWS:
        df[f"cvd_{w}"]       = s_vd.rolling(w).sum()
        df[f"cvd_tick_{w}"]  = s_tick_vd.rolling(w).sum()          # tick-rule 버전
        df[f"cvd_ma_{w}"]    = s_vd.rolling(w).mean()
        df[f"cvd_ratio_{w}"] = (
            s_vd.rolling(w).sum() / (s_vol.rolling(w).sum() + 1e-10)
        )
        # 다이버전스: cvd 방향 ≠ close 방향
        cvd_sign   = np.sign(df[f"cvd_{w}"])
        price_sign = np.sign(s_close.diff(w))
        df[f"cvd_div_{w}"] = (cvd_sign != price_sign).astype(float)
        # cvd slope / price slope 비율 (연속형 divergence 강도)
        cvd_slope   = s_vd.rolling(w).mean()
        price_slope = s_close.pct_change(w)
        df[f"cvd_divstrength_{w}"] = (
            cvd_slope / (s_vol.rolling(w).mean() + 1e-10)
        ) - price_slope

    return df


# ══════════════════════════════════════════════════════════════
# 2. OFI — Order Flow Imbalance
# ══════════════════════════════════════════════════════════════

def _add_ofi(df: pd.DataFrame) -> pd.DataFrame:
    """Order Flow Imbalance 피처.

    바 내 주문흐름 강도:
        ofi_raw = (close - open) / (high - low + ε) * volume

    해석:
        +1 → 강한 매수 (open 대비 close 상단, 최대 범위)
        -1 → 강한 매도
        0  → 결정 없음 (도지, 상하 균형)

    Alpha hypothesis:
        ofi_ma 상승 추세 → 기관 누적매수 가능성
        ofi_std 급등    → 방향 불확실 (변동성 확대)
    """
    close  = df["close"]
    open_  = df["open"]
    high   = df["high"]
    low    = df["low"]
    vol    = df["volume"]

    bar_range = high - low
    body      = close - open_

    ofi_raw   = body / (bar_range + 1e-10) * vol   # 원시 OFI

    # ATR 정규화
    atr = df.get("atr_14", pd.Series(np.nan, index=df.index))
    atr = atr.fillna(bar_range.rolling(14).mean())
    ofi_norm = ofi_raw / (atr * vol.rolling(14).mean() + 1e-10)

    df["ofi_raw"]  = ofi_raw
    df["ofi_norm"] = ofi_norm

    s_ofi = pd.Series(ofi_raw.values, index=df.index)
    for w in WINDOWS:
        df[f"ofi_sum_{w}"]   = s_ofi.rolling(w).sum()
        df[f"ofi_mean_{w}"]  = s_ofi.rolling(w).mean()
        df[f"ofi_std_{w}"]   = s_ofi.rolling(w).std()
        # 정규화 (전체 볼륨 대비)
        total_vol = vol.rolling(w).sum()
        df[f"ofi_pct_{w}"]   = s_ofi.rolling(w).sum() / (total_vol + 1e-10)

    return df


# ══════════════════════════════════════════════════════════════
# 3. VPIN — Volume-synchronized Prob of Informed Trading
# ══════════════════════════════════════════════════════════════

def _add_vpin(df: pd.DataFrame,
              n_buckets: int = 50,
              tau_window: int = 50) -> pd.DataFrame:
    """VPIN (Easley et al., 2012) OHLCV 근사.

    원본 알고리즘:
        1. 전체 거래량을 n_buckets개 동등 bucket으로 분할
        2. 각 bucket에서 |V_buy - V_sell| / V_bucket 계산
        3. tau_window 구간 평균 → VPIN

    OHLCV 근사:
        buy_vol = buy_frac(BVC) * volume
        bucket_vol = rolling_sum(volume, w) / n_buckets
        VPIN = |sum(vd, w)| / sum(volume, w)

    범위: [0, 1]
        0 → 완전 균형 (정보거래 없음)
        1 → 극도 불균형 (정보거래자 지배)

    Alpha hypothesis:
        VPIN 급등 → 정보 비대칭 확대 → 방향성 돌파 임박
        VPIN > 0.7 + CVD 상승 → 강한 매수 압력
    """
    # BVC 기반 buy fraction (이미 _add_cvd에서 계산됨)
    bvc_frac = df.get("buy_frac", (df["close"] - df["low"]) / (df["high"] - df["low"] + 1e-10))
    bvc_frac = bvc_frac.clip(0, 1)

    # Easley et al. (2012) — 정규분포 CDF 기반 buy fraction
    dp    = df["close"].diff().fillna(0)
    sigma = dp.rolling(20, min_periods=5).std().replace(0, np.nan).ffill().fillna(0.001)
    z     = dp / (sigma + 1e-10)
    # erf 기반 CDF (scipy 불필요)
    cdf_frac = 0.5 * (1.0 + np.sign(z) * (1 - np.exp(-0.7071 * z.abs())))
    cdf_frac = cdf_frac.clip(0, 1).fillna(0.5)

    vol = df["volume"]

    for w in [12, 24, 48]:
        vol_sum = vol.rolling(w).sum()

        # BVC 기반 VPIN
        vd_bvc = (2 * bvc_frac - 1) * vol
        df[f"vpin_{w}"] = (
            vd_bvc.rolling(w).sum().abs() / (vol_sum + 1e-10)
        ).clip(0, 1)

        # CDF 기반 VPIN (Easley et al. 원본 근사)
        vd_cdf = (2 * cdf_frac - 1) * vol
        df[f"vpin_cdf_{w}"] = (
            vd_cdf.rolling(w).sum().abs() / (vol_sum + 1e-10)
        ).clip(0, 1)

        # VPIN 변화율
        df[f"vpin_chg_{w}"]  = df[f"vpin_{w}"].diff(3)

        # 레짐 플래그 (75th percentile 초과)
        pct75 = df[f"vpin_{w}"].rolling(72, min_periods=12).quantile(0.75)
        df[f"vpin_regime_{w}"] = (df[f"vpin_{w}"] > pct75).astype(float)

        # 방향성 정보 — buy fraction 평균
        df[f"vpin_buyfrac_{w}"] = bvc_frac.rolling(w).mean()

    return df


# ══════════════════════════════════════════════════════════════
# 4. Roll Spread Estimator
# ══════════════════════════════════════════════════════════════

def _add_roll_spread(df: pd.DataFrame, windows: list[int] = None) -> pd.DataFrame:
    """Roll (1984) 추정 bid-ask spread.

    Roll spread = 2 * sqrt(max(-Cov(Δp_t, Δp_{t-1}), 0))

    log return 기반으로 계산 (가격 스케일 독립):
        Δp_t = log(close_t / close_{t-1})
        cov  = rolling Cov(Δp_t, Δp_{t-1})
        roll_spread = 2 * sqrt(max(-cov, 0))

    해석:
        높은 spread → 유동성 낮음, 진입비용 높음
        낮은 spread → 타이트한 마켓

    Alpha hypothesis:
        spread 급등 + VPIN 높음 → 유동성 위기 → 변동성 확대 경고
    """
    if windows is None:
        windows = [12, 24, 48]

    log_ret = np.log(df["close"] / (df["close"].shift(1) + 1e-10))
    lag_ret = log_ret.shift(1)

    for w in windows:
        cov = log_ret.rolling(w).cov(lag_ret)
        # cov < 0: mean-reversion (bid-ask bounce) → spread 계산 가능
        # cov > 0: trending market → Roll estimator 무의미 → NaN 후 ffill
        trending = cov >= 0
        roll_sq  = (-cov).clip(lower=0)
        spread   = 2.0 * np.sqrt(roll_sq)
        spread   = spread.where(~trending, other=np.nan).ffill()
        df[f"roll_spread_{w}"]     = spread
        df[f"roll_spread_bps_{w}"] = spread * 10000
        # 공분산 부호: +1=추세(모멘텀), -1=평균회귀
        df[f"roll_cov_sign_{w}"]   = np.sign(cov)

    # 단기/장기 spread 비율 (유동성 충격 지수)
    if "roll_spread_12" in df.columns and "roll_spread_48" in df.columns:
        df["roll_spread_ratio"] = (
            df["roll_spread_12"] / (df["roll_spread_48"] + 1e-10)
        )

    return df


# ══════════════════════════════════════════════════════════════
# 5. Amihud Illiquidity
# ══════════════════════════════════════════════════════════════

def _add_amihud(df: pd.DataFrame, windows: list[int] = None) -> pd.DataFrame:
    """Amihud (2002) 비유동성 지수.

    ILLIQ_t = |r_t| / (volume_t × price_t) × 10^6

    해석:
        높은 ILLIQ → 단위 거래량당 가격충격 큼 → 비유동
        낮은 ILLIQ → 깊은 시장

    Alpha hypothesis:
        ILLIQ 급등 → 유동성 공급자 철수 → 슬리피지 위험
        ILLIQ 하락 추세 → 기관 자금 유입 (시장 깊이 증가)
    """
    if windows is None:
        windows = WINDOWS

    abs_ret     = df["close"].pct_change().abs()
    dollar_vol  = df["volume"] * df["close"]   # dollar volume (USD 기준)
    illiq_raw   = abs_ret / (dollar_vol + 1e-10) * 1e6

    df["illiq_raw"] = illiq_raw

    s_illiq = pd.Series(illiq_raw.values, index=df.index)
    for w in windows:
        df[f"illiq_{w}"]       = s_illiq.rolling(w).mean()
        df[f"illiq_std_{w}"]   = s_illiq.rolling(w).std()
        # Z-score (anomaly detection)
        mu  = s_illiq.rolling(w).mean()
        std = s_illiq.rolling(w).std()
        df[f"illiq_z_{w}"]     = (illiq_raw - mu) / (std + 1e-10)

    # 단기/장기 비율
    if "illiq_6" in df.columns and "illiq_24" in df.columns:
        df["illiq_ratio"] = df["illiq_6"] / (df["illiq_24"] + 1e-10)

    return df


# ══════════════════════════════════════════════════════════════
# COMPOSITE SIGNALS
# ══════════════════════════════════════════════════════════════

def _add_composite_signals(df: pd.DataFrame) -> pd.DataFrame:
    """CVD × OFI × VPIN 결합 복합 신호.

    개별 지표 조합으로 더 강한 알파 생성.
    """
    # 매수 압력 종합 스코어 (z-score 합산)
    cols_long  = [c for c in df.columns if c.startswith("cvd_ratio_") or c.startswith("ofi_pct_")]
    cols_vpin  = [c for c in df.columns if c.startswith("vpin_") and "chg" not in c and "ratio" not in c]

    if cols_long:
        # 정규화 후 합산
        z_scores = []
        for col in cols_long[:4]:  # 최대 4개
            s = pd.Series(df[col].values, index=df.index)
            z = (s - s.rolling(96).mean()) / (s.rolling(96).std() + 1e-10)
            z_scores.append(z)
        if z_scores:
            df["ms_flow_score"] = sum(z_scores) / len(z_scores)

    # VPIN 기반 시장 불안 지수
    if cols_vpin:
        vpin_cols = cols_vpin[:3]
        df["ms_informed_score"] = df[vpin_cols].mean(axis=1)

    # 유동성 위기 신호 (spread + illiq 동시 급등)
    if "roll_spread_12" in df.columns and "illiq_6" in df.columns:
        rs_z = (df["roll_spread_12"] - df["roll_spread_12"].rolling(48).mean()) / \
               (df["roll_spread_12"].rolling(48).std() + 1e-10)
        il_z = df.get("illiq_z_6", pd.Series(0, index=df.index))
        df["ms_liquidity_stress"] = ((rs_z + il_z) / 2).clip(-3, 3)

    # 통합 마이크로스트럭처 신호 [-1, +1]
    if "ms_flow_score" in df.columns and "ms_informed_score" in df.columns:
        fs = df["ms_flow_score"].clip(-3, 3) / 3
        info = df["ms_informed_score"].clip(0, 1)
        stress = df.get("ms_liquidity_stress", pd.Series(0, index=df.index)).clip(-3, 3) / 3
        df["ms_composite"] = (fs * 0.5 + info * 0.3 - stress * 0.2)

    return df


# ══════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ══════════════════════════════════════════════════════════════

def add_microstructure_rollup(df: pd.DataFrame, verbose: bool = True) -> pd.DataFrame:
    """CVD + OFI + VPIN + Roll Spread + Amihud 피처를 DataFrame에 추가.

    Args:
        df   : OHLCV DataFrame (open, high, low, close, volume 필수)
        verbose : 피처 추가 현황 출력

    Returns:
        피처 추가된 DataFrame (~37+ 신규 피처)

    Feature Count (approximate):
        CVD     : 1 + 1 + 1 + 4×4 = 19
        OFI     : 2 + 4×4 = 18
        VPIN    : 3×3  = 9
        Roll    : 3×2 + 1 = 7
        Amihud  : 1 + 4×3 + 1 = 14
        Composite: 4
        Total   : ~71
    """
    n0 = len(df.columns)

    df = _add_cvd(df)
    n1 = len(df.columns)
    if verbose:
        print(f"  [MS] CVD: +{n1 - n0}")

    df = _add_ofi(df)
    n2 = len(df.columns)
    if verbose:
        print(f"  [MS] OFI: +{n2 - n1}")

    df = _add_vpin(df)
    n3 = len(df.columns)
    if verbose:
        print(f"  [MS] VPIN: +{n3 - n2}")

    df = _add_roll_spread(df)
    n4 = len(df.columns)
    if verbose:
        print(f"  [MS] Roll Spread: +{n4 - n3}")

    df = _add_amihud(df)
    n5 = len(df.columns)
    if verbose:
        print(f"  [MS] Amihud: +{n5 - n4}")

    df = _add_composite_signals(df)
    n6 = len(df.columns)
    if verbose:
        print(f"  [MS] Composite: +{n6 - n5}")
        print(f"  [MS] Total: {n0} → {n6} cols (+{n6 - n0})")

    return df


# ══════════════════════════════════════════════════════════════
# FEATURE NAMES (for reference)
# ══════════════════════════════════════════════════════════════

CVD_FEATURES = (
    ["vd", "cvd", "buy_frac"]
    + [f"cvd_{w}" for w in WINDOWS]
    + [f"cvd_ma_{w}" for w in WINDOWS]
    + [f"cvd_ratio_{w}" for w in WINDOWS]
    + [f"cvd_div_{w}" for w in WINDOWS]
)

OFI_FEATURES = (
    ["ofi_raw", "ofi_norm"]
    + [f"ofi_sum_{w}" for w in WINDOWS]
    + [f"ofi_mean_{w}" for w in WINDOWS]
    + [f"ofi_std_{w}" for w in WINDOWS]
    + [f"ofi_pct_{w}" for w in WINDOWS]
)

VPIN_FEATURES = (
    [f"vpin_{w}" for w in [12, 24, 48]]
    + [f"vpin_chg_{w}" for w in [12, 24, 48]]
    + [f"vpin_price_ratio_{w}" for w in [12, 24, 48]]
)

ROLL_FEATURES = (
    [f"roll_spread_{w}" for w in [12, 24, 48]]
    + [f"roll_spread_bps_{w}" for w in [12, 24, 48]]
    + ["roll_spread_ratio"]
)

AMIHUD_FEATURES = (
    ["illiq_raw"]
    + [f"illiq_{w}" for w in WINDOWS]
    + [f"illiq_std_{w}" for w in WINDOWS]
    + [f"illiq_z_{w}" for w in WINDOWS]
    + ["illiq_ratio"]
)

COMPOSITE_FEATURES = ["ms_flow_score", "ms_informed_score",
                       "ms_liquidity_stress", "ms_composite"]

ALL_MS_FEATURES = (CVD_FEATURES + OFI_FEATURES + VPIN_FEATURES
                   + ROLL_FEATURES + AMIHUD_FEATURES + COMPOSITE_FEATURES)
