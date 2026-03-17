"""Feature Store — 학습과 동일한 피처 파이프라인 (라이브용).

핵심 제약:
  - 학습 파이프라인과 100% 동일한 함수 재사용
  - 4h 캔들 종가 확정 후에만 피처 계산
  - ATR(14) 학습과 동일한 방식으로 계산
  - feature_cols 순서 = 학습 시 저장된 순서 그대로
"""

from __future__ import annotations

import logging
import warnings
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# 4h bar boundaries (UTC)
BAR_HOURS = [0, 4, 8, 12, 16, 20]


class FeatureStore:
    """라이브 피처 계산 — 학습 파이프라인 100% 재사용.

    Usage:
        store = FeatureStore(feature_cols=["rsi_14", "sma_7", ...])
        if store.is_bar_complete("BTC", now):
            X, atr, price = store.update_and_compute("BTC")
            # X: (1, n_features) array, feature_cols 순서
    """

    def __init__(
        self,
        feature_cols: list[str] | None = None,
        lookback_days: int = 60,
        macro_stale_hours: int = 24,
    ):
        self.feature_cols = feature_cols or []
        self.lookback_days = lookback_days
        self.macro_stale_hours = macro_stale_hours

        # Cache
        self._ohlcv_cache: dict[str, pd.DataFrame] = {}   # symbol → 4h DataFrame
        self._last_update: dict[str, datetime] = {}
        self._macro_cache: pd.DataFrame | None = None
        self._macro_updated: datetime | None = None

    def is_bar_complete(self, symbol: str, now: datetime | None = None) -> bool:
        """4h 캔들이 완전히 닫힌 후인지 확인.

        4h boundaries: 00:00, 04:00, 08:00, 12:00, 16:00, 20:00 UTC.
        Bar close 후 5분 대기 (데이터 정착).
        """
        now = now or datetime.now(timezone.utc)
        current_hour = now.hour
        current_minute = now.minute

        # Find the most recent bar close
        for h in reversed(BAR_HOURS):
            if current_hour > h or (current_hour == h and current_minute >= 5):
                bar_close = now.replace(hour=h, minute=0, second=0, microsecond=0)
                break
        else:
            # Before 00:05 UTC — previous day's 20:00 bar
            bar_close = (now - timedelta(days=1)).replace(
                hour=20, minute=0, second=0, microsecond=0
            )

        # Check if we already computed for this bar
        last = self._last_update.get(symbol)
        if last and last >= bar_close:
            return False

        return True

    def update_and_compute(
        self,
        symbol: str,
    ) -> tuple[np.ndarray, float, float]:
        """피처 계산: 학습과 동일한 파이프라인.

        Returns: (feature_vector, atr_14, last_close)
            - feature_vector: shape (n_features,)
            - atr_14: ATR(14) for barrier computation
            - last_close: 마지막 종가
        """
        # 1. Fetch OHLCV (1h) via yfinance
        df_1h = self._fetch_ohlcv(symbol)

        # 2. Resample to 4h
        df_4h = self._resample_4h(df_1h)

        # 3. Technical indicators (학습과 동일 함수)
        from src.data.crawlers.crypto_ohlcv import add_technical_indicators
        df_4h = add_technical_indicators(df_4h)

        # 4. Signal features (학습과 동일 함수)
        from src.data.crawlers.signal_features import add_signal_features
        df_4h = add_signal_features(df_4h, verbose=False)

        # 5. Macro merge
        df_4h = self._merge_macro(df_4h)

        # Cache
        self._ohlcv_cache[symbol] = df_4h
        self._last_update[symbol] = datetime.now(timezone.utc)

        # 6. Extract last row features
        last_row = df_4h.iloc[-1]
        atr = last_row.get("atr_14", self._compute_atr_fallback(df_4h))
        last_close = last_row["close"]

        # 7. Feature vector in training column order
        if self.feature_cols:
            available = [c for c in self.feature_cols if c in df_4h.columns]
            missing = [c for c in self.feature_cols if c not in df_4h.columns]
            if missing:
                logger.warning(f"[FeatureStore:{symbol}] Missing {len(missing)} features: {missing[:5]}...")

            X = np.zeros(len(self.feature_cols))
            for i, col in enumerate(self.feature_cols):
                if col in df_4h.columns:
                    val = last_row[col]
                    X[i] = val if np.isfinite(val) else 0.0
                else:
                    X[i] = 0.0
        else:
            # No feature_cols specified — return all numeric columns
            exclude = {"open", "high", "low", "close", "volume", "label"}
            feat_cols = [c for c in df_4h.columns
                        if c not in exclude
                        and df_4h[c].dtype in [np.float64, np.float32, np.int64, np.int32]]
            X = last_row[feat_cols].values.astype(np.float64)
            np.nan_to_num(X, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        return X, float(atr), float(last_close)

    def get_atr(self, symbol: str) -> float:
        """Cached ATR(14) for symbol."""
        if symbol in self._ohlcv_cache:
            df = self._ohlcv_cache[symbol]
            atr = df["atr_14"].iloc[-1] if "atr_14" in df.columns else 0.0
            if np.isfinite(atr) and atr > 0:
                return float(atr)
            return self._compute_atr_fallback(df)
        return 0.0

    def get_current_price(self, symbol: str) -> float:
        """Last close from cached data."""
        if symbol in self._ohlcv_cache:
            return float(self._ohlcv_cache[symbol]["close"].iloc[-1])
        return 0.0

    # ── Internal ────────────────────────────────────────────

    def _fetch_ohlcv(self, symbol: str) -> pd.DataFrame:
        """yfinance로 1h OHLCV 수집 — 학습과 동일."""
        from src.data.crawlers.crypto_ohlcv import YAHOO_MAP

        yahoo_sym = YAHOO_MAP.get(symbol, f"{symbol}-USD")

        try:
            import yfinance as yf
            ticker = yf.Ticker(yahoo_sym)
            df = ticker.history(
                period=f"{self.lookback_days}d",
                interval="1h",
                auto_adjust=True,
            )
            df.columns = [c.lower() for c in df.columns]

            # Ensure required columns
            required = ["open", "high", "low", "close", "volume"]
            for col in required:
                if col not in df.columns:
                    raise ValueError(f"Missing column: {col}")

            # Timezone handling — tz-naive로 통일
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)

            df = df[required].dropna()
            logger.info(f"[FeatureStore] {symbol}: {len(df)} 1h bars fetched")
            return df

        except Exception as e:
            logger.error(f"[FeatureStore] {symbol} fetch failed: {e}")
            raise

    def _resample_4h(self, df_1h: pd.DataFrame) -> pd.DataFrame:
        """1h → 4h resample — 학습과 동일."""
        df_4h = df_1h.resample("4h").agg({
            "open": "first",
            "high": "max",
            "low": "min",
            "close": "last",
            "volume": "sum",
        }).dropna()
        return df_4h

    def _merge_macro(self, df: pd.DataFrame) -> pd.DataFrame:
        """매크로 데이터 병합."""
        try:
            # Refresh macro if stale
            if self._macro_cache is None or self._is_macro_stale():
                self._refresh_macro()

            if self._macro_cache is not None and len(self._macro_cache) > 0:
                macro = self._macro_cache.copy()
                if macro.index.tz is not None:
                    macro.index = macro.index.tz_localize(None)

                # Resample macro to 4h and forward-fill
                macro_4h = macro.resample("4h").ffill()

                # Join
                df = df.join(macro_4h, how="left")
                df = df.ffill().bfill()

        except Exception as e:
            logger.warning(f"[FeatureStore] Macro merge failed: {e}")

        return df

    def _refresh_macro(self) -> None:
        """매크로 데이터 새로고침."""
        try:
            from src.data.crawlers.macro_commodity_crawler import crawl_all_macro_data
            self._macro_cache = crawl_all_macro_data(verbose=False)
            self._macro_updated = datetime.now(timezone.utc)
            logger.info(f"[FeatureStore] Macro data refreshed: {len(self._macro_cache)} rows")
        except Exception as e:
            logger.warning(f"[FeatureStore] Macro refresh failed: {e}")

    def _is_macro_stale(self) -> bool:
        if self._macro_updated is None:
            return True
        elapsed = (datetime.now(timezone.utc) - self._macro_updated).total_seconds()
        return elapsed > self.macro_stale_hours * 3600

    @staticmethod
    def _compute_atr_fallback(df: pd.DataFrame, period: int = 14) -> float:
        """Fallback ATR computation if atr_14 column missing."""
        if len(df) < period + 1:
            return 0.0
        high = df["high"].values
        low = df["low"].values
        close = df["close"].values
        tr = np.maximum(
            high[1:] - low[1:],
            np.maximum(
                np.abs(high[1:] - close[:-1]),
                np.abs(low[1:] - close[:-1]),
            ),
        )
        if len(tr) < period:
            return float(np.mean(tr)) if len(tr) > 0 else 0.0
        return float(np.mean(tr[-period:]))
