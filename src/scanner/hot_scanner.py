"""Hot Coin Discovery - 지금 돈이 몰리는 코인을 찾는다.

Phase 1: DISCOVERY - 거래량 폭발 + 가격 이탈 + 변동성 급등 감지
Phase 2: VALIDATION - 차트 분석 + 미디어 감성 + 거래량-가격 정합성
Phase 3: SCORING - 단기수익 가능성 종합 평가

모든 가중치/임계값은 params dict로 외부 주입 가능 → 최적화 루프에서 사용.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from src.scanner.universe_scanner import UNIVERSE, scan_ohlcv


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Default Parameters (최적화 대상)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

DEFAULT_PARAMS = {
    # ── Opportunity Score 가중치 (합=1.0) ── [Optimized 2026-03-12, Score +22.17]
    "w_heat": 0.347,
    "w_chart": 0.257,
    "w_media": 0.009,
    "w_vp": 0.100,
    "w_cost": 0.288,

    # ── Direction 신호 가중치 (mean-reversion 최적화) ──
    "dw_trend": -0.226,
    "dw_macd": -0.922,
    "dw_rsi": -0.198,      # 역추세
    "dw_momentum": -1.033,  # 역추세
    "dw_bb": -1.862,        # 역추세
    "dw_media": -0.561,

    # ── Direction 임계값 (보수적: 확실할 때만 방향 콜) ──
    "dir_threshold": 0.365,
    "rsi_oversold": 39,
    "rsi_overbought": 59,

    # ── Heat Score 가중치 ──
    "hw_vol_surge_6h": 29.8,
    "hw_vol_spike_1h": 2.8,
    "hw_price_break": 27.9,
    "hw_vol_expand": 8.7,
    "hw_momentum_4h": 4.3,
    "hw_momentum_24h": 2.2,
    "hw_micro_burst": 1.5,
    "hw_micro_trend": 8.0,
}


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Data Classes
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class Anomaly:
    coin: str
    price: float = 0.0
    heat_score: float = 0.0
    volume_surge_6h: float = 0.0
    volume_surge_1h: float = 0.0
    price_break_pct: float = 0.0
    volatility_expansion: float = 0.0
    momentum_4h: float = 0.0
    momentum_24h: float = 0.0
    momentum_72h: float = 0.0
    micro_burst_count: int = 0
    micro_trend_strength: float = 0.0
    anomaly_reasons: list = field(default_factory=list)


@dataclass
class HotCoin:
    coin: str
    price: float = 0.0
    heat_score: float = 0.0
    anomaly_reasons: list = field(default_factory=list)
    rsi: float = 0.0
    macd_signal: str = ""
    bb_position: str = ""
    trend_direction: str = ""
    adx: float = 0.0
    support: float = 0.0
    resistance: float = 0.0
    chart_score: float = 0.0
    media_sentiment: str = ""
    media_score: float = 0.0
    media_buzz_count: int = 0
    top_headlines: list = field(default_factory=list)
    media_validation_score: float = 0.0
    vp_aligned: bool = False
    vp_score: float = 0.0
    opportunity_score: float = 0.0
    direction: str = "neutral"
    direction_confidence: float = 0.0
    expected_range_pct: float = 0.0
    risk_reward: float = 0.0
    catalyst: str = ""
    volume_surge: float = 0.0
    vol_24h_pct: float = 0.0
    ret_4h: float = 0.0
    ret_24h: float = 0.0
    daily_volume_usd: float = 0.0
    cost_ratio: float = 0.0


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Phase 1: DISCOVERY
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def detect_anomalies(
    ohlcv_1h: dict[str, pd.DataFrame],
    ohlcv_5m: dict[str, pd.DataFrame] = None,
    top_n: int = 15,
    params: dict = None,
    quiet: bool = False,
) -> list[Anomaly]:
    p = {**DEFAULT_PARAMS, **(params or {})}
    if not quiet:
        print(f"\n[Phase 1] Anomaly Detection ({len(ohlcv_1h)} coins)...")
    anomalies = []

    for coin, df in ohlcv_1h.items():
        a = _detect_single(coin, df, ohlcv_5m.get(coin) if ohlcv_5m else None, p)
        if a is not None:
            anomalies.append(a)

    anomalies.sort(key=lambda x: x.heat_score, reverse=True)

    if not quiet:
        for i, a in enumerate(anomalies[:top_n]):
            reasons = ", ".join(a.anomaly_reasons) if a.anomaly_reasons else "baseline"
            print(f"  {i+1:2d}. {a.coin:6s} Heat:{a.heat_score:5.1f} | "
                  f"VolSurge:{a.volume_surge_6h:.1f}x PriceBreak:{a.price_break_pct:+.2f} "
                  f"VolExpand:{a.volatility_expansion:.1f}x | {reasons}")

    return anomalies[:top_n]


def _detect_single(coin, df_1h, df_5m, p):
    if len(df_1h) < 48:
        return None

    a = Anomaly(coin=coin)
    close = df_1h["Close"].astype(float)
    high = df_1h["High"].astype(float)
    low = df_1h["Low"].astype(float)
    volume = df_1h["Volume"].astype(float) if "Volume" in df_1h.columns else pd.Series(0, index=df_1h.index)

    a.price = float(close.iloc[-1])
    returns = close.pct_change().dropna()

    avg_vol_7d = float(volume.mean())
    avg_vol_24h = float(volume.tail(24).mean())
    vol_6h = float(volume.tail(6).mean())
    vol_1h = float(volume.iloc[-1])

    a.volume_surge_6h = round(vol_6h / (avg_vol_7d + 1e-10), 2)
    a.volume_surge_1h = round(vol_1h / (avg_vol_24h + 1e-10), 2)

    high_7d = float(high.max())
    low_7d = float(low.min())
    range_7d = high_7d - low_7d
    a.price_break_pct = round((a.price - low_7d) / (range_7d + 1e-10) * 2 - 1, 2)

    if len(returns) > 24:
        vol_6h_std = float(returns.tail(6).std())
        vol_7d_std = float(returns.std())
        a.volatility_expansion = round(vol_6h_std / (vol_7d_std + 1e-10), 2)

    a.momentum_4h = round(float((close.iloc[-1] / close.iloc[-4] - 1) * 100), 2) if len(close) > 4 else 0
    a.momentum_24h = round(float((close.iloc[-1] / close.iloc[-24] - 1) * 100), 2) if len(close) > 24 else 0
    n72 = min(72, len(close) - 1)
    a.momentum_72h = round(float((close.iloc[-1] / close.iloc[-n72] - 1) * 100), 2) if n72 > 0 else 0

    if df_5m is not None and len(df_5m) > 24:
        vol_5m = df_5m["Volume"].astype(float) if "Volume" in df_5m.columns else None
        if vol_5m is not None and float(vol_5m.mean()) > 0:
            avg_5m = float(vol_5m.mean())
            recent_24bars = vol_5m.tail(24)
            a.micro_burst_count = int((recent_24bars > avg_5m * 3).sum())
            close_5m = df_5m["Close"].astype(float)
            ret_5m = close_5m.pct_change().dropna().tail(24)
            if len(ret_5m) > 0:
                pos = (ret_5m > 0).sum()
                neg = (ret_5m < 0).sum()
                a.micro_trend_strength = round(abs(pos - neg) / (pos + neg + 1e-10), 2)

    # Anomaly reasons
    if a.volume_surge_6h >= 2.0:
        a.anomaly_reasons.append(f"vol_surge_6h({a.volume_surge_6h:.1f}x)")
    if a.volume_surge_1h >= 3.0:
        a.anomaly_reasons.append(f"vol_spike_1h({a.volume_surge_1h:.1f}x)")
    if abs(a.price_break_pct) > 0.8:
        a.anomaly_reasons.append(f"price_extreme({'high' if a.price_break_pct > 0 else 'low'})")
    if a.volatility_expansion >= 2.0:
        a.anomaly_reasons.append(f"vol_expand({a.volatility_expansion:.1f}x)")
    if abs(a.momentum_4h) >= 3.0:
        a.anomaly_reasons.append(f"mom_4h({a.momentum_4h:+.1f}%)")
    if abs(a.momentum_24h) >= 8.0:
        a.anomaly_reasons.append(f"mom_24h({a.momentum_24h:+.1f}%)")
    if a.micro_burst_count >= 3:
        a.anomaly_reasons.append(f"micro_bursts({a.micro_burst_count})")

    # Heat Score (파라미터화)
    heat = 0.0
    heat += min(20, max(0, (a.volume_surge_6h - 1.0) * p["hw_vol_surge_6h"]))
    heat += min(15, max(0, (a.volume_surge_1h - 1.0) * p["hw_vol_spike_1h"]))
    heat += min(20, abs(a.price_break_pct) * p["hw_price_break"])
    heat += min(15, max(0, (a.volatility_expansion - 1.0) * p["hw_vol_expand"]))
    heat += min(10, abs(a.momentum_4h) * p["hw_momentum_4h"])
    heat += min(10, abs(a.momentum_24h) * p["hw_momentum_24h"])
    heat += min(5, a.micro_burst_count * p["hw_micro_burst"])
    heat += min(5, a.micro_trend_strength * p["hw_micro_trend"])

    a.heat_score = round(min(100, heat), 1)
    return a


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Phase 2: VALIDATION
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def validate_chart(coin: str, df_1h: pd.DataFrame, params: dict = None) -> dict:
    p = {**DEFAULT_PARAMS, **(params or {})}
    close = df_1h["Close"].astype(float)
    high = df_1h["High"].astype(float)
    low = df_1h["Low"].astype(float)
    volume = df_1h["Volume"].astype(float) if "Volume" in df_1h.columns else pd.Series(0, index=df_1h.index)
    result = {}

    # RSI
    delta = close.diff()
    gain = delta.where(delta > 0, 0).ewm(span=14, adjust=False).mean()
    loss = (-delta.where(delta < 0, 0)).ewm(span=14, adjust=False).mean()
    rs = gain / (loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    result["rsi"] = round(float(rsi.iloc[-1]), 1)

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    hist = macd_line - signal_line
    if len(hist) >= 2:
        if float(hist.iloc[-1]) > 0 and float(hist.iloc[-2]) <= 0:
            result["macd_signal"] = "bullish_cross"
        elif float(hist.iloc[-1]) < 0 and float(hist.iloc[-2]) >= 0:
            result["macd_signal"] = "bearish_cross"
        elif float(hist.iloc[-1]) > 0:
            result["macd_signal"] = "bullish"
        else:
            result["macd_signal"] = "bearish"
    else:
        result["macd_signal"] = "neutral"

    # Bollinger Bands
    sma20 = close.rolling(20).mean()
    std20 = close.rolling(20).std()
    bb_upper = sma20 + 2 * std20
    bb_lower = sma20 - 2 * std20
    current = float(close.iloc[-1])

    if current > float(bb_upper.iloc[-1]):
        result["bb_position"] = "above_upper"
        result["bb_signal"] = -1  # 과매수 역추세
    elif current < float(bb_lower.iloc[-1]):
        result["bb_position"] = "below_lower"
        result["bb_signal"] = 1   # 과매도 역추세
    else:
        bb_range = float(bb_upper.iloc[-1]) - float(bb_lower.iloc[-1])
        bb_pct = (current - float(bb_lower.iloc[-1])) / (bb_range + 1e-10)
        result["bb_position"] = f"inside({bb_pct:.0%})"
        result["bb_signal"] = 0

    # ADX
    high_v, low_v, close_v = high.values, low.values, close.values
    if len(close_v) > 16:
        tr = np.maximum(high_v[1:] - low_v[1:],
                        np.maximum(np.abs(high_v[1:] - close_v[:-1]),
                                   np.abs(low_v[1:] - close_v[:-1])))
        plus_dm = np.where((high_v[1:] - high_v[:-1]) > (low_v[:-1] - low_v[1:]),
                           np.maximum(high_v[1:] - high_v[:-1], 0), 0)
        minus_dm = np.where((low_v[:-1] - low_v[1:]) > (high_v[1:] - high_v[:-1]),
                            np.maximum(low_v[:-1] - low_v[1:], 0), 0)
        atr = pd.Series(tr).ewm(span=14, adjust=False).mean().values
        plus_di = 100 * pd.Series(plus_dm).ewm(span=14, adjust=False).mean().values / (atr + 1e-10)
        minus_di = 100 * pd.Series(minus_dm).ewm(span=14, adjust=False).mean().values / (atr + 1e-10)
        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        adx_arr = pd.Series(dx).ewm(span=14, adjust=False).mean().values
        result["adx"] = round(float(adx_arr[-1]), 1)
        result["plus_di"] = round(float(plus_di[-1]), 1)
        result["minus_di"] = round(float(minus_di[-1]), 1)
        result["trend_direction"] = "up" if float(plus_di[-1]) > float(minus_di[-1]) else "down"
    else:
        result["adx"] = 0
        result["trend_direction"] = "sideways"

    # Support / Resistance
    result["resistance"] = round(float(high.tail(48).max()), 4)
    result["support"] = round(float(low.tail(48).min()), 4)

    # ATR
    tr_s = pd.Series(np.maximum(
        high.values[1:] - low.values[1:],
        np.maximum(np.abs(high.values[1:] - close.values[:-1]),
                   np.abs(low.values[1:] - close.values[:-1]))))
    atr_val = float(tr_s.tail(14).mean())
    result["atr_pct"] = round(atr_val / (current + 1e-10) * 100, 2)

    # Volume-Price alignment
    ret_24h = float((close.iloc[-1] / close.iloc[-24] - 1)) if len(close) > 24 else 0
    vol_24h = float(volume.tail(24).mean())
    vol_prev = float(volume.tail(48).head(24).mean()) if len(volume) > 48 else vol_24h
    vol_change = vol_24h / (vol_prev + 1e-10)
    if abs(ret_24h) > 0.01:
        result["vp_aligned"] = (vol_change > 1.0)
    else:
        result["vp_aligned"] = True

    # Chart Score
    cs = 50
    if result["rsi"] < p["rsi_oversold"]:
        cs += 15
    elif result["rsi"] > p["rsi_overbought"]:
        cs += 15
    elif 40 <= result["rsi"] <= 60:
        cs -= 5
    if "cross" in result["macd_signal"]:
        cs += 15
    elif result["macd_signal"] in ("bullish", "bearish"):
        cs += 5
    if result["adx"] > 30:
        cs += 10
    elif result["adx"] > 25:
        cs += 5
    if result["bb_position"] in ("above_upper", "below_lower"):
        cs += 10
    if result["vp_aligned"]:
        cs += 5
    else:
        cs -= 10
    result["chart_score"] = min(100, max(0, cs))
    return result


def validate_media(coin: str) -> dict:
    """미디어 검증 (live scan에서만 사용, 백테스트에서는 기본값)."""
    from src.data.crawlers.sentiment_analyzer import (
        analyze_batch, aggregate_sentiment, filter_by_ticker
    )
    result = {
        "media_sentiment": "neutral", "media_score": 0.0,
        "media_buzz_count": 0, "top_headlines": [], "media_validation_score": 50,
    }
    all_items = []
    try:
        from src.data.crawlers.google_news_crawler import fetch_google_news
        all_items.extend(fetch_google_news(f"{coin} crypto", num=8))
    except Exception:
        pass
    try:
        from src.data.crawlers.reddit_crawler import fetch_subreddit_posts
        for sub in ["cryptocurrency", "CryptoMarkets"]:
            try:
                posts = fetch_subreddit_posts(sub, sort="hot", limit=10)
                all_items.extend(filter_by_ticker(posts, coin))
            except Exception:
                pass
    except Exception:
        pass
    try:
        from src.data.crawlers.x_crawler import fetch_x_trends_via_google
        all_items.extend(fetch_x_trends_via_google(f"{coin} crypto", num=5))
    except Exception:
        pass
    if not all_items:
        return result
    analyzed = analyze_batch(all_items, text_field="title")
    agg = aggregate_sentiment(analyzed)
    result["media_sentiment"] = agg.get("overall", "neutral")
    result["media_score"] = round(agg.get("avg_score", 0), 3)
    result["media_buzz_count"] = len(all_items)
    scored = sorted([i for i in analyzed if "sentiment" in i],
                    key=lambda x: abs(x["sentiment"].get("score", 0)), reverse=True)
    result["top_headlines"] = [
        {"title": i.get("title", "")[:80], "sentiment": i["sentiment"]["sentiment"],
         "score": i["sentiment"]["score"], "source": i.get("source", "")}
        for i in scored[:5]
    ]
    ms = 50
    buzz = len(all_items)
    ms += 15 if buzz >= 15 else (8 if buzz >= 8 else (-10 if buzz <= 2 else 0))
    score_abs = abs(result["media_score"])
    ms += 15 if score_abs > 0.3 else (8 if score_abs > 0.15 else 0)
    bull_c = agg.get("bullish_count", 0)
    bear_c = agg.get("bearish_count", 0)
    total = bull_c + bear_c
    if total > 3 and max(bull_c, bear_c) / total > 0.7:
        ms += 10
    result["media_validation_score"] = min(100, max(0, ms))
    return result


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Phase 3: SCORING (파라미터화)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def build_hot_coin(
    anomaly: Anomaly,
    chart: dict,
    media: dict,
    df_1h: pd.DataFrame,
    params: dict = None,
) -> HotCoin:
    p = {**DEFAULT_PARAMS, **(params or {})}
    close = df_1h["Close"].astype(float)
    high = df_1h["High"].astype(float)
    low = df_1h["Low"].astype(float)
    volume = df_1h["Volume"].astype(float) if "Volume" in df_1h.columns else pd.Series(0, index=df_1h.index)

    h = HotCoin(coin=anomaly.coin, price=anomaly.price)
    h.heat_score = anomaly.heat_score
    h.anomaly_reasons = anomaly.anomaly_reasons
    h.volume_surge = anomaly.volume_surge_6h
    h.ret_4h = anomaly.momentum_4h
    h.ret_24h = anomaly.momentum_24h

    avg_hourly = float(volume.mean())
    h.daily_volume_usd = round(avg_hourly * h.price * 24, 0)
    returns = close.pct_change().dropna()
    h.vol_24h_pct = round(float(returns.tail(24).std() * np.sqrt(24) * 100), 2)
    bar_range = ((high - low) / (close + 1e-10)).tail(48)
    avg_range = float(bar_range.mean() * 100)
    h.cost_ratio = round(0.25 / (avg_range + 1e-10), 3)

    h.rsi = chart.get("rsi", 50)
    h.macd_signal = chart.get("macd_signal", "neutral")
    h.bb_position = chart.get("bb_position", "inside")
    h.trend_direction = chart.get("trend_direction", "sideways")
    h.adx = chart.get("adx", 0)
    h.support = chart.get("support", 0)
    h.resistance = chart.get("resistance", 0)
    h.chart_score = chart.get("chart_score", 50)
    h.vp_aligned = chart.get("vp_aligned", False)
    h.vp_score = 70 if chart.get("vp_aligned", False) else 30

    h.media_sentiment = media.get("media_sentiment", "neutral")
    h.media_score = media.get("media_score", 0)
    h.media_buzz_count = media.get("media_buzz_count", 0)
    h.top_headlines = media.get("top_headlines", [])
    h.media_validation_score = media.get("media_validation_score", 50)

    # ── Direction 판단 (가중치 기반, 역추세 반영) ──
    weighted_sum = 0.0
    weight_total = 0.0

    # 1. ADX 추세 방향
    w = abs(p["dw_trend"])
    if chart.get("trend_direction") == "up":
        weighted_sum += p["dw_trend"] * 1
    elif chart.get("trend_direction") == "down":
        weighted_sum += p["dw_trend"] * (-1)
    weight_total += w

    # 2. MACD
    w = abs(p["dw_macd"])
    if "bullish" in chart.get("macd_signal", ""):
        weighted_sum += p["dw_macd"] * 1
    elif "bearish" in chart.get("macd_signal", ""):
        weighted_sum += p["dw_macd"] * (-1)
    weight_total += w

    # 3. RSI 역추세: 과매도→LONG(+1), 과매수→SHORT(-1)
    w = abs(p["dw_rsi"])
    rsi_val = chart.get("rsi", 50)
    if rsi_val < p["rsi_oversold"]:
        weighted_sum += p["dw_rsi"] * 1    # 과매도 → LONG
    elif rsi_val > p["rsi_overbought"]:
        weighted_sum += p["dw_rsi"] * (-1)  # 과매수 → SHORT
    weight_total += w

    # 4. 모멘텀 (음수 가중치 = 역추세)
    w = abs(p["dw_momentum"])
    if anomaly.momentum_4h > 1.0:
        weighted_sum += p["dw_momentum"] * 1   # dw_momentum이 음수면 역추세
    elif anomaly.momentum_4h < -1.0:
        weighted_sum += p["dw_momentum"] * (-1)
    weight_total += w

    # 5. Bollinger Band
    w = abs(p["dw_bb"])
    bb_sig = chart.get("bb_signal", 0)
    if bb_sig != 0:
        weighted_sum += p["dw_bb"] * bb_sig
    weight_total += w

    # 6. 미디어
    w = abs(p["dw_media"])
    if media.get("media_score", 0) > 0.15:
        weighted_sum += p["dw_media"] * 1
    elif media.get("media_score", 0) < -0.15:
        weighted_sum += p["dw_media"] * (-1)
    weight_total += w

    # 정규화
    if weight_total > 0:
        dir_signal = weighted_sum / weight_total
    else:
        dir_signal = 0

    if dir_signal > p["dir_threshold"]:
        h.direction = "LONG"
    elif dir_signal < -p["dir_threshold"]:
        h.direction = "SHORT"
    else:
        h.direction = "neutral"
    h.direction_confidence = round(min(1.0, abs(dir_signal) * 2), 2)

    # Expected Range
    h.expected_range_pct = round(chart.get("atr_pct", 0) * 4, 2)

    # Risk/Reward
    if 0 < h.cost_ratio < 5:
        h.risk_reward = round(1.0 / h.cost_ratio, 1)

    # Catalyst
    if h.top_headlines:
        h.catalyst = h.top_headlines[0]["title"]
    elif h.anomaly_reasons:
        h.catalyst = "; ".join(h.anomaly_reasons[:2])
    else:
        h.catalyst = "No clear catalyst"

    # ── Opportunity Score (가중치 기반) ──
    cost_score = max(0, min(100, 100 - h.cost_ratio * 100))
    h.opportunity_score = round(
        h.heat_score * p["w_heat"]
        + h.chart_score * p["w_chart"]
        + h.media_validation_score * p["w_media"]
        + h.vp_score * p["w_vp"]
        + cost_score * p["w_cost"],
        1,
    )
    return h


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Full Pipeline
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def run_hot_scan(top_n=15, media_top_n=10, params=None):
    from src.scanner.universe_scanner import scan_all
    print("\n" + "=" * 60)
    print("  HOT COIN DISCOVERY PIPELINE")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    data = scan_all()
    ohlcv_1h = data["ohlcv_1h"]
    ohlcv_5m = data["ohlcv_5m"]
    anomalies = detect_anomalies(ohlcv_1h, ohlcv_5m, top_n=top_n, params=params)
    print(f"\n[Phase 2] Chart + Media Validation (top {min(media_top_n, len(anomalies))})...")
    hot_coins = []
    for i, a in enumerate(anomalies):
        if a.coin not in ohlcv_1h:
            continue
        chart = validate_chart(a.coin, ohlcv_1h[a.coin], params)
        if i < media_top_n:
            print(f"  [{a.coin}] fetching media...")
            media = validate_media(a.coin)
        else:
            media = {"media_validation_score": 50}
        hc = build_hot_coin(a, chart, media, ohlcv_1h[a.coin], params)
        hot_coins.append(hc)
    hot_coins.sort(key=lambda x: x.opportunity_score, reverse=True)
    print(f"\n[Phase 3] Final Ranking:")
    for i, hc in enumerate(hot_coins[:top_n]):
        print(f"  {i+1:2d}. {hc.coin:6s} | Opp:{hc.opportunity_score:5.1f} | "
              f"Heat:{hc.heat_score:4.0f} Chart:{hc.chart_score:4.0f} "
              f"Media:{hc.media_validation_score:4.0f} | "
              f"{hc.direction:>7s}({hc.direction_confidence:.0%})")
    return {
        "hot_coins": hot_coins, "anomalies": anomalies,
        "btc_dominance": data.get("btc_dominance"),
        "fear_greed": data.get("fear_greed"),
        "scanned_at": data.get("scanned_at"),
        "total_scanned": len(ohlcv_1h),
    }
