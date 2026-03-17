"""기술적 분석 엔진 (investor-agent 스킬).

가격 히스토리 기반으로 RSI, SMA, EMA, MACD, 볼린저밴드 등을 계산하고
매수/매도 시그널을 생성합니다. 외부 라이브러리 의존 없음.
"""

from typing import Any
import math


def calculate_sma(prices: list[float], period: int) -> list[float | None]:
    """단순이동평균(SMA)."""
    result = [None] * len(prices)
    for i in range(period - 1, len(prices)):
        result[i] = sum(prices[i - period + 1: i + 1]) / period
    return result


def calculate_ema(prices: list[float], period: int) -> list[float | None]:
    """지수이동평균(EMA)."""
    result = [None] * len(prices)
    if len(prices) < period:
        return result

    k = 2 / (period + 1)
    result[period - 1] = sum(prices[:period]) / period

    for i in range(period, len(prices)):
        result[i] = prices[i] * k + result[i - 1] * (1 - k)

    return result


def calculate_rsi(prices: list[float], period: int = 14) -> list[float | None]:
    """상대강도지수(RSI)."""
    result = [None] * len(prices)
    if len(prices) < period + 1:
        return result

    gains = []
    losses = []
    for i in range(1, len(prices)):
        diff = prices[i] - prices[i - 1]
        gains.append(max(diff, 0))
        losses.append(max(-diff, 0))

    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    if avg_loss == 0:
        result[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        result[period] = 100 - (100 / (1 + rs))

    for i in range(period, len(gains)):
        avg_gain = (avg_gain * (period - 1) + gains[i]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i]) / period

        if avg_loss == 0:
            result[i + 1] = 100.0
        else:
            rs = avg_gain / avg_loss
            result[i + 1] = 100 - (100 / (1 + rs))

    return result


def calculate_macd(
    prices: list[float],
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> dict[str, list[float | None]]:
    """MACD (이동평균수렴확산)."""
    ema_fast = calculate_ema(prices, fast)
    ema_slow = calculate_ema(prices, slow)

    macd_line = [None] * len(prices)
    for i in range(len(prices)):
        if ema_fast[i] is not None and ema_slow[i] is not None:
            macd_line[i] = ema_fast[i] - ema_slow[i]

    # Signal line (MACD의 EMA)
    macd_values = [v for v in macd_line if v is not None]
    if len(macd_values) >= signal:
        signal_ema = calculate_ema(macd_values, signal)
        # 매핑
        signal_line = [None] * len(prices)
        start_idx = next(i for i, v in enumerate(macd_line) if v is not None)
        for i, val in enumerate(signal_ema):
            if val is not None:
                signal_line[start_idx + i] = val
    else:
        signal_line = [None] * len(prices)

    # Histogram
    histogram = [None] * len(prices)
    for i in range(len(prices)):
        if macd_line[i] is not None and signal_line[i] is not None:
            histogram[i] = macd_line[i] - signal_line[i]

    return {"macd": macd_line, "signal": signal_line, "histogram": histogram}


def calculate_bollinger(
    prices: list[float], period: int = 20, num_std: float = 2.0
) -> dict[str, list[float | None]]:
    """볼린저 밴드."""
    sma = calculate_sma(prices, period)
    upper = [None] * len(prices)
    lower = [None] * len(prices)

    for i in range(period - 1, len(prices)):
        window = prices[i - period + 1: i + 1]
        mean = sma[i]
        std = math.sqrt(sum((x - mean) ** 2 for x in window) / period)
        upper[i] = mean + num_std * std
        lower[i] = mean - num_std * std

    return {"upper": upper, "middle": sma, "lower": lower}


def calculate_atr(
    highs: list[float], lows: list[float], closes: list[float], period: int = 14
) -> list[float | None]:
    """평균진폭(ATR) — 고가/저가/종가가 필요하지만 종가만 있을 때 근사치 사용."""
    # 종가만 있는 경우: 일일 변동폭을 TR 대용으로 사용
    if not highs or not lows:
        trs = [abs(closes[i] - closes[i-1]) for i in range(1, len(closes))]
        trs.insert(0, 0)
    else:
        trs = [0]
        for i in range(1, len(closes)):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i-1]),
                abs(lows[i] - closes[i-1]),
            )
            trs.append(tr)

    result = [None] * len(closes)
    if len(trs) < period:
        return result

    result[period - 1] = sum(trs[:period]) / period
    for i in range(period, len(trs)):
        result[i] = (result[i-1] * (period - 1) + trs[i]) / period

    return result


def run_full_technical_analysis(price_history: list[dict]) -> dict[str, Any]:
    """가격 히스토리에 대한 전체 기술적 분석을 수행합니다."""
    if len(price_history) < 5:
        return {"error": "데이터 부족 (최소 5일 필요)"}

    prices = [p["price"] for p in price_history]
    dates = [p["date"] for p in price_history]
    volumes = [p.get("volume", 0) for p in price_history]

    current_price = prices[-1]
    prev_price = prices[-2] if len(prices) > 1 else current_price

    # === 지표 계산 ===
    sma_7 = calculate_sma(prices, 7)
    sma_20 = calculate_sma(prices, 20)
    sma_50 = calculate_sma(prices, min(50, len(prices)))
    ema_12 = calculate_ema(prices, 12)
    ema_26 = calculate_ema(prices, 26)
    rsi_14 = calculate_rsi(prices, 14)
    macd_data = calculate_macd(prices)
    bb = calculate_bollinger(prices, min(20, len(prices)))

    # 최신 값 추출
    latest_rsi = _last_valid(rsi_14)
    latest_macd = _last_valid(macd_data["macd"])
    latest_macd_signal = _last_valid(macd_data["signal"])
    latest_macd_hist = _last_valid(macd_data["histogram"])
    latest_sma_7 = _last_valid(sma_7)
    latest_sma_20 = _last_valid(sma_20)
    latest_bb_upper = _last_valid(bb["upper"])
    latest_bb_lower = _last_valid(bb["lower"])

    # === 시그널 판단 ===
    signals = []
    signal_score = 0  # -10 ~ +10

    # RSI
    if latest_rsi is not None:
        if latest_rsi > 70:
            signals.append({"indicator": "RSI", "value": f"{latest_rsi:.1f}", "signal": "OVERBOUGHT", "direction": "bearish"})
            signal_score -= 2
        elif latest_rsi < 30:
            signals.append({"indicator": "RSI", "value": f"{latest_rsi:.1f}", "signal": "OVERSOLD", "direction": "bullish"})
            signal_score += 2
        elif latest_rsi < 45:
            signals.append({"indicator": "RSI", "value": f"{latest_rsi:.1f}", "signal": "WEAK", "direction": "bearish"})
            signal_score -= 1
        else:
            signals.append({"indicator": "RSI", "value": f"{latest_rsi:.1f}", "signal": "NEUTRAL", "direction": "neutral"})

    # SMA 크로스
    if latest_sma_7 is not None and latest_sma_20 is not None:
        if latest_sma_7 > latest_sma_20:
            signals.append({"indicator": "SMA 7/20", "value": "Golden Cross", "signal": "BULLISH", "direction": "bullish"})
            signal_score += 2
        else:
            signals.append({"indicator": "SMA 7/20", "value": "Death Cross", "signal": "BEARISH", "direction": "bearish"})
            signal_score -= 2

    # 가격 vs SMA
    if latest_sma_20 is not None:
        if current_price > latest_sma_20:
            signals.append({"indicator": "Price vs SMA20", "value": f"Above ({current_price:.0f} > {latest_sma_20:.0f})", "signal": "BULLISH", "direction": "bullish"})
            signal_score += 1
        else:
            signals.append({"indicator": "Price vs SMA20", "value": f"Below ({current_price:.0f} < {latest_sma_20:.0f})", "signal": "BEARISH", "direction": "bearish"})
            signal_score -= 1

    # MACD
    if latest_macd_hist is not None:
        if latest_macd_hist > 0:
            signals.append({"indicator": "MACD", "value": f"Histogram: {latest_macd_hist:.2f}", "signal": "BULLISH", "direction": "bullish"})
            signal_score += 1.5
        else:
            signals.append({"indicator": "MACD", "value": f"Histogram: {latest_macd_hist:.2f}", "signal": "BEARISH", "direction": "bearish"})
            signal_score -= 1.5

    # 볼린저 밴드
    if latest_bb_upper is not None and latest_bb_lower is not None:
        if current_price >= latest_bb_upper:
            signals.append({"indicator": "Bollinger", "value": f"Upper Band Touch", "signal": "OVERBOUGHT", "direction": "bearish"})
            signal_score -= 1.5
        elif current_price <= latest_bb_lower:
            signals.append({"indicator": "Bollinger", "value": f"Lower Band Touch", "signal": "OVERSOLD", "direction": "bullish"})
            signal_score += 1.5
        else:
            band_pos = (current_price - latest_bb_lower) / (latest_bb_upper - latest_bb_lower) if latest_bb_upper != latest_bb_lower else 0.5
            signals.append({"indicator": "Bollinger", "value": f"Band Position: {band_pos:.1%}", "signal": "NEUTRAL", "direction": "neutral"})

    # 거래량 추세
    if len(volumes) >= 7:
        recent_vol = sum(volumes[-7:]) / 7
        prev_vol = sum(volumes[-14:-7]) / 7 if len(volumes) >= 14 else recent_vol
        if prev_vol > 0:
            vol_change = (recent_vol - prev_vol) / prev_vol
            if vol_change > 0.3:
                signals.append({"indicator": "Volume", "value": f"+{vol_change:.0%} vs prev week", "signal": "HIGH_VOLUME", "direction": "neutral"})
            elif vol_change < -0.3:
                signals.append({"indicator": "Volume", "value": f"{vol_change:.0%} vs prev week", "signal": "LOW_VOLUME", "direction": "neutral"})

    # 종합 판단
    if signal_score > 3:
        overall = "STRONG_BUY"
    elif signal_score > 1:
        overall = "BUY"
    elif signal_score > -1:
        overall = "NEUTRAL"
    elif signal_score > -3:
        overall = "SELL"
    else:
        overall = "STRONG_SELL"

    return {
        "current_price": current_price,
        "date_range": f"{dates[0]} ~ {dates[-1]}",
        "data_points": len(prices),
        "indicators": {
            "rsi_14": latest_rsi,
            "sma_7": latest_sma_7,
            "sma_20": latest_sma_20,
            "macd": latest_macd,
            "macd_signal": latest_macd_signal,
            "macd_histogram": latest_macd_hist,
            "bb_upper": latest_bb_upper,
            "bb_lower": latest_bb_lower,
        },
        "signals": signals,
        "signal_score": round(signal_score, 1),
        "overall": overall,
    }


def format_ta_report(ticker: str, ta: dict) -> str:
    """기술적 분석 결과를 리포트로 포맷합니다."""
    if "error" in ta:
        return f"[{ticker}] {ta['error']}"

    lines = [
        f"## {ticker} 기술적 분석",
        f"분석 기간: {ta['date_range']} ({ta['data_points']}일)",
        f"현재가: ${ta['current_price']:,.2f}",
        f"",
        f"### 지표 현황",
    ]

    ind = ta["indicators"]
    if ind["rsi_14"]:
        lines.append(f"- RSI(14): {ind['rsi_14']:.1f}")
    if ind["sma_7"]:
        lines.append(f"- SMA(7): ${ind['sma_7']:,.2f}")
    if ind["sma_20"]:
        lines.append(f"- SMA(20): ${ind['sma_20']:,.2f}")
    if ind["macd"]:
        lines.append(f"- MACD: {ind['macd']:.2f} / Signal: {ind['macd_signal']:.2f}" if ind["macd_signal"] else f"- MACD: {ind['macd']:.2f}")
    if ind["bb_upper"]:
        lines.append(f"- Bollinger: ${ind['bb_lower']:,.2f} ~ ${ind['bb_upper']:,.2f}")

    lines.append(f"\n### 시그널 ({ta['overall']}  점수: {ta['signal_score']:+.1f})")
    for sig in ta["signals"]:
        icon = {"bullish": "UP", "bearish": "DN", "neutral": "--"}[sig["direction"]]
        lines.append(f"  [{icon}] {sig['indicator']}: {sig['value']} → {sig['signal']}")

    return "\n".join(lines)


def _last_valid(lst: list) -> float | None:
    """리스트에서 마지막 유효한 값을 반환합니다."""
    for v in reversed(lst):
        if v is not None:
            return v
    return None
