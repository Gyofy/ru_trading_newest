"""Short-term Profit Scorer - 거래량+변동성 기반 단기수익 가능성 평가.

핵심 질문: "이 코인이 지금 단기 트레이딩으로 수익을 낼 수 있는가?"

5개 점수 (거래량·변동성 중심):
1. Volume Power      (25%) - 거래량 크기 + 최근 서지(surge) + 안정성
2. Volatility Quality (25%) - 수익 가능 변동성 (너무 낮지도 높지도 않은 sweet spot)
3. Profit Ratio       (20%) - 실제 봉 진폭 vs 왕복비용 (돈이 되는가?)
4. Momentum Clarity   (20%) - 방향성 모멘텀 강도 + 지속성
5. Risk Penalty       (10%) - 위험 신호 감점
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field


@dataclass
class CoinScore:
    coin: str
    price: float = 0.0

    # 5대 점수 (0~100)
    volume_power: float = 0.0
    volatility_quality: float = 0.0
    profit_ratio: float = 0.0
    momentum_clarity: float = 0.0

    # 종합
    profit_score: float = 0.0    # 단기수익 가능성 (0~100)

    # 방향성
    direction: str = "neutral"   # LONG, SHORT, neutral
    direction_strength: float = 0.0  # 0~1

    # 세부 지표 (리포트용)
    vol_24h_pct: float = 0.0         # 24h 실현 변동성 %
    vol_7d_pct: float = 0.0          # 7d 실현 변동성 %
    daily_volume_usd: float = 0.0    # 일평균 거래대금
    volume_surge: float = 0.0        # 최근 거래량 / 평균 (1.0 = 평균)
    avg_bar_range_pct: float = 0.0   # 평균 봉 진폭 %
    cost_ratio: float = 0.0          # 비용/진폭 비율
    adx: float = 0.0                 # ADX (추세 강도)
    ret_4h: float = 0.0              # 최근 4h 수익률 %
    ret_24h: float = 0.0             # 최근 24h 수익률 %
    ret_72h: float = 0.0             # 최근 72h 수익률 %

    # 5분봉 세부
    vol_5m_burst: float = 0.0        # 5분봉 거래량 버스트 비율

    # 리스크
    risk_flags: list = field(default_factory=list)
    risk_deduction: float = 0.0      # 리스크 감점 (0~)


def _compute_adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> float:
    """ADX 계산."""
    if len(close) < period + 2:
        return 0.0

    tr = np.maximum(
        high[1:] - low[1:],
        np.maximum(np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])),
    )
    plus_dm = np.where(
        (high[1:] - high[:-1]) > (low[:-1] - low[1:]),
        np.maximum(high[1:] - high[:-1], 0), 0,
    )
    minus_dm = np.where(
        (low[:-1] - low[1:]) > (high[1:] - high[:-1]),
        np.maximum(low[:-1] - low[1:], 0), 0,
    )

    atr = pd.Series(tr).ewm(span=period, adjust=False).mean().values
    plus_di = 100 * pd.Series(plus_dm).ewm(span=period, adjust=False).mean().values / (atr + 1e-10)
    minus_di = 100 * pd.Series(minus_dm).ewm(span=period, adjust=False).mean().values / (atr + 1e-10)

    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
    adx = pd.Series(dx).ewm(span=period, adjust=False).mean().values
    return float(adx[-1]) if len(adx) > 0 else 0.0


def score_coin(coin: str, df_1h: pd.DataFrame, df_5m: pd.DataFrame = None) -> CoinScore:
    """단일 코인의 단기수익 가능성 점수 계산."""
    s = CoinScore(coin=coin)

    if len(df_1h) < 24:
        s.risk_flags.append("insufficient_data")
        s.risk_deduction = 100
        return s

    close = df_1h["Close"].astype(float)
    high = df_1h["High"].astype(float)
    low = df_1h["Low"].astype(float)
    volume = df_1h["Volume"].astype(float) if "Volume" in df_1h.columns else pd.Series(0, index=df_1h.index)

    s.price = float(close.iloc[-1])
    returns = close.pct_change().dropna()

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 1. VOLUME POWER (0~100)
    #    거래량이 클수록 + 최근 증가할수록 + 안정적일수록 높음
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    avg_hourly_vol = float(volume.mean())
    daily_vol_usd = avg_hourly_vol * s.price * 24
    s.daily_volume_usd = round(daily_vol_usd, 0)

    # (a) 절대 거래량 점수 (0~50)
    if daily_vol_usd >= 1_000_000_000:
        vol_abs = 50
    elif daily_vol_usd >= 200_000_000:
        vol_abs = 35 + (daily_vol_usd - 200_000_000) / 800_000_000 * 15
    elif daily_vol_usd >= 50_000_000:
        vol_abs = 20 + (daily_vol_usd - 50_000_000) / 150_000_000 * 15
    elif daily_vol_usd >= 10_000_000:
        vol_abs = 10 + (daily_vol_usd - 10_000_000) / 40_000_000 * 10
    else:
        vol_abs = max(0, daily_vol_usd / 10_000_000 * 10)

    # (b) 거래량 서지: 최근 6h vs 전체 평균 (0~30)
    recent_6h_vol = float(volume.tail(6).mean())
    vol_surge_ratio = recent_6h_vol / (avg_hourly_vol + 1e-10)
    s.volume_surge = round(vol_surge_ratio, 2)

    if vol_surge_ratio >= 3.0:
        vol_surge_score = 30
    elif vol_surge_ratio >= 2.0:
        vol_surge_score = 20 + (vol_surge_ratio - 2.0) * 10
    elif vol_surge_ratio >= 1.2:
        vol_surge_score = 5 + (vol_surge_ratio - 1.2) / 0.8 * 15
    elif vol_surge_ratio >= 0.8:
        vol_surge_score = 5  # 평균 수준
    else:
        vol_surge_score = max(0, vol_surge_ratio / 0.8 * 5)  # 급감 페널티

    # (c) 안정성: 시간대별 CV가 낮을수록 (0~20)
    vol_cv = float(volume.std() / (volume.mean() + 1e-10))
    vol_stability = max(0, 20 - vol_cv * 8)

    s.volume_power = min(100, vol_abs + vol_surge_score + vol_stability)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 2. VOLATILITY QUALITY (0~100)
    #    sweet spot: 2~8% (24h) → 단기 트레이딩에 적합한 변동성
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    vol_24h = float(returns.tail(24).std() * np.sqrt(24) * 100)
    vol_7d = float(returns.std() * np.sqrt(24) * 100)
    s.vol_24h_pct = round(vol_24h, 2)
    s.vol_7d_pct = round(vol_7d, 2)

    # 24h 변동성 기반 (0~60) - sweet spot 2~8%
    if vol_24h < 0.5:
        vq_24h = 5
    elif vol_24h < 1.5:
        vq_24h = 5 + (vol_24h - 0.5) * 15  # 5~20
    elif vol_24h < 2.0:
        vq_24h = 20 + (vol_24h - 1.5) * 40  # 20~40
    elif vol_24h <= 5.0:
        vq_24h = 40 + (min(vol_24h, 5.0) - 2.0) / 3.0 * 20  # 40~60
    elif vol_24h <= 8.0:
        vq_24h = 60  # 최고 구간
    elif vol_24h <= 12.0:
        vq_24h = max(20, 60 - (vol_24h - 8.0) * 10)  # 60~20
    else:
        vq_24h = max(5, 20 - (vol_24h - 12.0) * 3)  # 너무 높으면 위험

    # 변동성 확대 추세 보너스: 최근 24h > 7d 평균 (0~25)
    if vol_7d > 0:
        vol_expansion = vol_24h / vol_7d
    else:
        vol_expansion = 1.0

    if vol_expansion >= 1.5:
        vq_expansion = 25  # 변동성 확대 중 = 기회
    elif vol_expansion >= 1.1:
        vq_expansion = 10 + (vol_expansion - 1.1) / 0.4 * 15
    elif vol_expansion >= 0.8:
        vq_expansion = 10  # 안정적
    else:
        vq_expansion = max(0, vol_expansion / 0.8 * 10)  # 변동성 수축

    # 봉 내 방향성: (Close-Open)/Range가 클수록 실체가 큰 봉 = 트레이딩 유리 (0~15)
    open_price = df_1h["Open"].astype(float)
    body = (close - open_price).abs()
    wick = (high - low).replace(0, np.nan)
    body_ratio = (body / wick).dropna().tail(24)
    avg_body_ratio = float(body_ratio.mean()) if len(body_ratio) > 0 else 0.5
    vq_body = min(15, avg_body_ratio * 20)

    s.volatility_quality = min(100, vq_24h + vq_expansion + vq_body)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 3. PROFIT RATIO (0~100)
    #    실제 봉 진폭 vs 왕복 거래비용 → 돈이 되는가?
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    bar_range = ((high - low) / (close + 1e-10))
    avg_range = float(bar_range.tail(48).mean() * 100)  # 최근 2일 평균 %
    s.avg_bar_range_pct = round(avg_range, 3)

    # 왕복 비용: 편도 0.20% × 2 = 0.40%
    # 편도 = (maker 0.10% + taker 0.10%) / 2 + slippage 0.05% + buffer 0.05%
    round_trip_cost = 0.40
    cost_ratio = round_trip_cost / (avg_range + 1e-10)
    s.cost_ratio = round(cost_ratio, 3)

    # 비용 대비 진폭이 클수록 좋음
    # cost_ratio < 0.2 → 진폭이 비용의 5배 이상 (매우 좋음)
    if cost_ratio < 0.1:
        s.profit_ratio = 100
    elif cost_ratio < 0.2:
        s.profit_ratio = 80 + (0.2 - cost_ratio) / 0.1 * 20
    elif cost_ratio < 0.3:
        s.profit_ratio = 60 + (0.3 - cost_ratio) / 0.1 * 20
    elif cost_ratio < 0.5:
        s.profit_ratio = 30 + (0.5 - cost_ratio) / 0.2 * 30
    elif cost_ratio < 1.0:
        s.profit_ratio = 5 + (1.0 - cost_ratio) / 0.5 * 25
    else:
        s.profit_ratio = max(0, 5 - (cost_ratio - 1.0) * 5)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 4. MOMENTUM CLARITY (0~100)
    #    방향성이 명확할수록 진입 타이밍 잡기 쉬움
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    ret_4h = float((close.iloc[-1] / close.iloc[-4] - 1) * 100) if len(close) > 4 else 0
    ret_24h = float((close.iloc[-1] / close.iloc[-24] - 1) * 100) if len(close) > 24 else 0
    ret_72h = float((close.iloc[-1] / close.iloc[-min(72, len(close)-1)] - 1) * 100) if len(close) > 2 else 0
    s.ret_4h = round(ret_4h, 2)
    s.ret_24h = round(ret_24h, 2)
    s.ret_72h = round(ret_72h, 2)

    # ADX (추세 강도)
    adx_val = _compute_adx(high.values, low.values, close.values, 14)
    s.adx = round(adx_val, 1)

    # (a) ADX 기반 (0~40): ADX > 25이면 추세 있음
    if adx_val >= 40:
        mc_adx = 40
    elif adx_val >= 25:
        mc_adx = 20 + (adx_val - 25) / 15 * 20
    elif adx_val >= 15:
        mc_adx = 5 + (adx_val - 15) / 10 * 15
    else:
        mc_adx = max(0, adx_val / 15 * 5)

    # (b) 단기 모멘텀 크기 (0~35): 최근 움직임이 클수록
    mom_abs = abs(ret_4h) * 0.5 + abs(ret_24h) * 0.3 + abs(ret_72h) * 0.2
    mc_mom = min(35, mom_abs * 5)

    # (c) 방향 일관성: 4h, 24h, 72h가 같은 방향이면 보너스 (0~25)
    signs = [
        1 if ret_4h > 0.1 else (-1 if ret_4h < -0.1 else 0),
        1 if ret_24h > 0.2 else (-1 if ret_24h < -0.2 else 0),
        1 if ret_72h > 0.3 else (-1 if ret_72h < -0.3 else 0),
    ]
    nonzero = [x for x in signs if x != 0]
    if len(nonzero) >= 2 and all(x == nonzero[0] for x in nonzero):
        mc_consistency = 25  # 일관된 방향
    elif len(nonzero) >= 2:
        mc_consistency = 5   # 혼재
    else:
        mc_consistency = 10  # 약한 움직임

    s.momentum_clarity = min(100, mc_adx + mc_mom + mc_consistency)

    # ── Direction (방향성 판단) ──
    weighted_mom = ret_4h * 0.5 + ret_24h * 0.3 + ret_72h * 0.2
    if weighted_mom > 0.5:
        s.direction = "LONG"
        s.direction_strength = min(1.0, abs(weighted_mom) / 5.0)
    elif weighted_mom < -0.5:
        s.direction = "SHORT"
        s.direction_strength = min(1.0, abs(weighted_mom) / 5.0)
    else:
        s.direction = "neutral"
        s.direction_strength = abs(weighted_mom) / 5.0

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 5. RISK FLAGS + 감점
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 거래량 급감
    if len(volume) > 48:
        recent_vol = float(volume.tail(6).mean())
        avg_vol_2d = float(volume.tail(48).mean())
        if avg_vol_2d > 0 and recent_vol / avg_vol_2d < 0.2:
            s.risk_flags.append("vol_collapse")
            s.risk_deduction += 15

    # 변동성 급등 (최근 4h vs 전체)
    if len(returns) > 24:
        recent_std = float(returns.tail(4).std())
        overall_std = float(returns.std())
        if overall_std > 0 and recent_std / overall_std > 3.0:
            s.risk_flags.append("vol_spike")
            s.risk_deduction += 10

    # 24h 급등/급락 (15% 이상)
    if abs(ret_24h) > 15:
        s.risk_flags.append("extreme_move")
        s.risk_deduction += 12

    # 일평균 거래대금 $1M 미만 (유동성 위험)
    if daily_vol_usd < 1_000_000:
        s.risk_flags.append("low_liquidity")
        s.risk_deduction += 20

    # 연속 6봉 같은 방향 (과열)
    if len(returns) >= 6:
        last_6 = returns.tail(6)
        if all(last_6 > 0) or all(last_6 < 0):
            s.risk_flags.append("overextended")
            s.risk_deduction += 8

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # 5분봉 보너스 (있으면)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    if df_5m is not None and len(df_5m) > 24:
        vol_5m = df_5m["Volume"].astype(float) if "Volume" in df_5m.columns else None
        if vol_5m is not None and float(vol_5m.mean()) > 0:
            # 5분봉 거래량 버스트: 상위 10% 봉의 거래량 / 평균
            top_10pct = vol_5m.quantile(0.9)
            s.vol_5m_burst = round(float(top_10pct / vol_5m.mean()), 2)

    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    # PROFIT SCORE (종합)
    # ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
    raw = (
        s.volume_power * 0.25
        + s.volatility_quality * 0.25
        + s.profit_ratio * 0.20
        + s.momentum_clarity * 0.20
    )
    # 5분봉 버스트 보너스 (최대 +5)
    burst_bonus = min(5, max(0, (s.vol_5m_burst - 2.0) * 2.5)) if s.vol_5m_burst > 2.0 else 0

    s.profit_score = round(max(0, min(100, raw + burst_bonus - s.risk_deduction)), 1)
    return s


def score_universe(ohlcv_1h: dict, ohlcv_5m: dict = None, top_n: int = 15) -> list[CoinScore]:
    """전체 유니버스 점수 계산 후 profit_score 내림차순 정렬, 상위 N개 반환."""
    print(f"\n[Scorer] 단기수익 가능성 평가 ({len(ohlcv_1h)} coins)...")
    all_scores = []

    for coin, df in ohlcv_1h.items():
        df_5m = ohlcv_5m.get(coin) if ohlcv_5m else None
        s = score_coin(coin, df, df_5m)
        all_scores.append(s)

    all_scores.sort(key=lambda x: x.profit_score, reverse=True)

    # 콘솔 요약
    for i, s in enumerate(all_scores[:top_n]):
        risk = f" [{','.join(s.risk_flags)}]" if s.risk_flags else ""
        print(f"  {i+1:2d}. {s.coin:6s} | Score:{s.profit_score:5.1f} | "
              f"VP:{s.volume_power:4.0f} VQ:{s.volatility_quality:4.0f} "
              f"PR:{s.profit_ratio:4.0f} MC:{s.momentum_clarity:4.0f} | "
              f"{s.direction:>7s}{risk}")

    below = len(all_scores) - top_n
    if below > 0:
        print(f"  ... +{below} coins below top {top_n}")

    return all_scores
