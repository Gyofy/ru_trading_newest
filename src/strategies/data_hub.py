"""DataHub — centralized data fetching + caching for multi-strategy bot.

Prevents duplicate ccxt API calls when multiple strategies need the same data.
Caches OHLCV by (coin, timeframe) with auto-invalidation.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import time
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from src.execution.exchange_adapter import ExchangeAdapter

logger = logging.getLogger("data_hub")

# 2 concurrent requests — enough for asyncio.gather overlap without 429 bursts.
# Per-key locks prevent duplicate fetches for the same (coin, timeframe).
MAX_CONCURRENT = 2


class DataHub:
    """Shared data provider for all strategies."""

    def __init__(self, exchange: ExchangeAdapter, executor: concurrent.futures.ProcessPoolExecutor | None = None):
        self.exchange = exchange
        self._ohlcv_cache: dict[str, tuple[float, pd.DataFrame]] = {}
        self._ticker_cache: dict[str, tuple[float, dict]] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._semaphore = asyncio.Semaphore(MAX_CONCURRENT)
        self._executor = executor  # ProcessPoolExecutor for CPU-bound computation

        # Cache TTLs (seconds) — conservative to avoid 429
        self.ohlcv_ttl = 58      # refresh just before next 1m bar
        self.ticker_ttl = 10      # ticker → 10s staleness ok (was 5s)

    def _get_lock(self, key: str) -> asyncio.Lock:
        if key not in self._locks:
            self._locks[key] = asyncio.Lock()
        return self._locks[key]

    # ── OHLCV ─────────────────────────────────────────────

    async def get_ohlcv(
        self,
        coin: str,
        timeframe: str = "1m",
        limit: int = 500,
    ) -> pd.DataFrame | None:
        """Get OHLCV with caching. Returns DataFrame with columns: open, high, low, close, volume."""
        cache_key = f"ohlcv:{coin}:{timeframe}:{limit}"
        now = time.time()

        # Check cache
        if cache_key in self._ohlcv_cache:
            ts, df = self._ohlcv_cache[cache_key]
            if now - ts < self.ohlcv_ttl:
                return df

        # Fetch with lock (prevent duplicate requests)
        lock = self._get_lock(cache_key)
        async with lock:
            # Double-check after acquiring lock
            if cache_key in self._ohlcv_cache:
                ts, df = self._ohlcv_cache[cache_key]
                if now - ts < self.ohlcv_ttl:
                    return df

            try:
                async with self._semaphore:
                    df = await self._fetch_ohlcv(coin, timeframe, limit)
                if df is not None and len(df) > 0:
                    self._ohlcv_cache[cache_key] = (time.time(), df)
                return df
            except Exception as e:
                logger.error(f"[DataHub] OHLCV fetch failed {coin}/{timeframe}: {e}")
                # Return stale cache if available
                if cache_key in self._ohlcv_cache:
                    return self._ohlcv_cache[cache_key][1]
                return None

    async def _fetch_ohlcv(
        self, coin: str, timeframe: str, limit: int
    ) -> pd.DataFrame | None:
        """Raw ccxt OHLCV fetch."""
        from src.execution.exchange_adapter import SYMBOL_MAP
        symbol = SYMBOL_MAP.get(coin, f"{coin}/USDT:USDT")

        ohlcv = await self.exchange.async_exchange.fetch_ohlcv(
            symbol, timeframe=timeframe, limit=limit
        )
        if not ohlcv:
            return None

        df = pd.DataFrame(
            ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df.set_index("timestamp", inplace=True)
        return df

    # ── Ticker ────────────────────────────────────────────

    async def get_ticker(self, coin: str) -> dict:
        """Get ticker with 5s cache."""
        now = time.time()
        if coin in self._ticker_cache:
            ts, data = self._ticker_cache[coin]
            if now - ts < self.ticker_ttl:
                return data

        try:
            async with self._semaphore:
                data = await self.exchange.fetch_ticker(coin)
            self._ticker_cache[coin] = (time.time(), data)
            return data
        except Exception as e:
            logger.error(f"[DataHub] Ticker fetch failed {coin}: {e}")
            if coin in self._ticker_cache:
                return self._ticker_cache[coin][1]
            return {"bid": 0, "ask": 0, "last": 0, "spread_bps": 0}

    # ── Batch refresh ─────────────────────────────────────

    async def refresh_all(self, coins: list[str], timeframe: str = "1m") -> None:
        """Pre-fetch all coins in parallel (respecting semaphore)."""
        tasks = [self.get_ohlcv(coin, timeframe) for coin in coins]
        await asyncio.gather(*tasks, return_exceptions=True)

    async def compute_cvd_async(self, df: pd.DataFrame) -> pd.Series:
        """Compute CVD in process pool (non-blocking)."""
        if self._executor is None:
            return self.compute_cvd(df)
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, self.compute_cvd, df)

    async def compute_ofi_async(self, df: pd.DataFrame) -> pd.Series:
        """Compute OFI in process pool (non-blocking)."""
        if self._executor is None:
            return self.compute_ofi(df)
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, self.compute_ofi, df)

    async def compute_vpin_async(self, df: pd.DataFrame, window: int = 24) -> float:
        """Compute VPIN in process pool (non-blocking)."""
        if self._executor is None:
            return self.compute_vpin(df, window)
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self._executor, self.compute_vpin, df, window)

    # ── Microstructure helpers ────────────────────────────

    @staticmethod
    def compute_cvd(df: pd.DataFrame) -> pd.Series:
        """BVC-based Cumulative Volume Delta from OHLCV."""
        if df is None or len(df) < 2:
            return pd.Series(dtype=float)
        c = df["close"].values
        o = df["open"].values
        h = df["high"].values
        l = df["low"].values
        v = df["volume"].values

        rng = h - l
        rng[rng == 0] = 1e-10
        # BVC: bullish candle → buy ratio from low, bearish → sell pressure from high
        buy_ratio = np.where(c >= o, (c - l) / rng, (h - c) / rng)
        buy_vol = v * buy_ratio
        sell_vol = v * (1 - buy_ratio)
        delta = buy_vol - sell_vol
        return pd.Series(np.cumsum(delta), index=df.index, name="cvd")

    @staticmethod
    def compute_ofi(df: pd.DataFrame) -> pd.Series:
        """Order Flow Imbalance proxy from OHLCV."""
        if df is None or len(df) < 2:
            return pd.Series(dtype=float)
        c = df["close"].values
        o = df["open"].values
        h = df["high"].values
        l = df["low"].values
        v = df["volume"].values

        rng = h - l
        rng[rng == 0] = 1e-10
        ofi = ((c - o) / rng) * v
        return pd.Series(ofi, index=df.index, name="ofi")

    @staticmethod
    def compute_vwap(df: pd.DataFrame, window: int = 20) -> pd.Series:
        """Rolling VWAP."""
        if df is None or len(df) < window:
            return pd.Series(dtype=float)
        typical = (df["high"] + df["low"] + df["close"]) / 3
        vol = df["volume"]
        vwap = (typical * vol).rolling(window).sum() / vol.rolling(window).sum()
        return vwap

    # ── OI via ccxt ───────────────────────────────────────

    async def get_open_interest(self, coin: str) -> float | None:
        """Fetch current open interest via ccxt."""
        from src.execution.exchange_adapter import SYMBOL_MAP
        symbol = SYMBOL_MAP.get(coin, f"{coin}/USDT:USDT")
        try:
            async with self._semaphore:
                oi = await self.exchange.async_exchange.fetch_open_interest(symbol)
            return float(oi.get("openInterestAmount", 0)) if oi else None
        except Exception as e:
            logger.debug(f"[DataHub] OI fetch failed {coin}: {e}")
            return None

    async def get_funding_rate(self, coin: str) -> float | None:
        """Fetch current funding rate."""
        try:
            result = await self.exchange.fetch_funding_rate(coin)
            return result
        except Exception:
            return None

    # ── VPIN ─────────────────────────────────────────────

    @staticmethod
    def compute_vpin(df: pd.DataFrame, window: int = 24) -> float:
        """Compute VPIN (Volume-Synchronized Probability of Informed Trading).

        High VPIN (>0.7) = likely informed trading → signal is meaningful
        Low VPIN (<0.3) = noise → signal is unreliable
        """
        if df is None or len(df) < window:
            return 0.5  # neutral default

        recent = df.iloc[-window:]
        h, l, c, v = recent["high"], recent["low"], recent["close"], recent["volume"]
        o = recent["open"]
        rng = (h - l).replace(0, 1e-10)
        # BVC formula: consistent with compute_cvd fix
        buy_frac = (c - l).where(c >= o, (h - c)) / rng
        buy_vol = buy_frac * v
        sell_vol = (1 - buy_frac) * v

        # VPIN = mean(|buy - sell|) / mean(total_vol)
        abs_imbalance = (buy_vol - sell_vol).abs()
        total_vol = v.replace(0, 1e-10)
        vpin = float(abs_imbalance.mean() / total_vol.mean())
        return min(vpin, 1.0)  # cap at 1.0

    # ── Signal quality ───────────────────────────────────

    async def get_signal_quality(
        self, coin: str, timeframe: str = "1m", window: int = 24
    ) -> dict:
        """Get signal quality metrics for a coin.

        Returns dict with:
          - vpin: float (0-1, higher = more informed trading)
          - funding_rate: float
          - spread_bps: float (from ticker)
        """
        df = await self.get_ohlcv(coin, timeframe, limit=window + 10)
        vpin = self.compute_vpin(df, window)
        funding = await self.get_funding_rate(coin)
        ticker = await self.get_ticker(coin)

        return {
            "vpin": vpin,
            "funding_rate": funding if funding is not None else 0.0,
            "spread_bps": ticker.get("spread_bps", 0),
        }

    # ── Dynamic coin selection ─────────────────────────────

    # Coins to always exclude (stablecoins, wrapped, leveraged tokens)
    EXCLUDE_COINS = {
        "USDC", "FDUSD", "TUSD", "BUSD", "DAI", "USDD",  # stablecoins
        "WBTC", "WBETH", "STETH",                          # wrapped
        "BTC", "ETH",                                       # reference only
    }
    # Leveraged token suffixes to exclude (e.g., BTCUP, ETHDOWN)
    EXCLUDE_SUFFIXES = ("UP", "DOWN", "BULL", "BEAR")

    async def select_tradeable_coins(
        self,
        volatility_pool: int = 30,
        min_volume_usdt: float = 50_000_000,
        max_spread_bps: float = 8.0,
        min_price_usd: float = 0.01,
        held_coins: list[str] | None = None,
    ) -> list[dict]:
        """Select tradeable coins: volatility top 30 → safety filter → all survivors.

        Process:
          1. Fetch all Binance USDT-M tickers (1 API call)
          2. Pre-filter: active, not stablecoin/wrapped, has bid/ask
          3. Rank by 24h volatility (high-low range %) → take top `volatility_pool`
          4. Safety filter on the pool:
             - 24h quoteVolume >= min_volume_usdt
             - spread <= max_spread_bps
             - price >= min_price_usd (penny coin exclusion)
          5. ALL survivors become active monitoring targets (no fixed cap)
          6. Held coins always included

        Returns list of dicts with coin metadata, sorted by volatility desc.
        """
        try:
            ex = self.exchange.async_exchange
            tickers = await ex.fetch_tickers()
        except Exception as e:
            logger.error(f"[CoinSelect] fetch_tickers failed: {e}")
            if held_coins:
                return [{"coin": c, "volatility_pct": 0, "volume_usdt": 0,
                         "spread_bps": 0, "last": 0} for c in held_coins]
            return []

        # ── Step 1: Pre-filter all USDT-M perpetuals ──
        all_coins = []
        for symbol, ticker in tickers.items():
            if not symbol.endswith("/USDT:USDT"):
                continue

            coin = symbol.split("/")[0]

            # Exclude list
            if coin in self.EXCLUDE_COINS:
                continue
            if any(coin.endswith(s) for s in self.EXCLUDE_SUFFIXES):
                continue

            # Active market check
            try:
                market = ex.market(symbol)
                if not market.get("active", False):
                    continue
            except Exception:
                continue

            # Must have valid price data
            last = ticker.get("last", 0) or 0
            high = ticker.get("high", 0) or 0
            low = ticker.get("low", 0) or 0
            bid = ticker.get("bid", 0) or 0
            ask = ticker.get("ask", 0) or 0
            vol_usdt = ticker.get("quoteVolume", 0) or 0

            if last <= 0 or high <= 0 or low <= 0 or bid <= 0 or ask <= 0:
                continue

            # 24h volatility = (high - low) / last × 100
            volatility_pct = (high - low) / last * 100

            # Spread in bps
            mid = (ask + bid) / 2
            spread_bps = (ask - bid) / mid * 10000 if mid > 0 else 999

            all_coins.append({
                "coin": coin,
                "volatility_pct": round(volatility_pct, 3),
                "volume_usdt": vol_usdt,
                "spread_bps": round(spread_bps, 2),
                "last": last,
            })

        # ── Step 2: Rank by volatility, take top N pool ──
        all_coins.sort(key=lambda x: x["volatility_pct"], reverse=True)
        volatility_pool_coins = all_coins[:volatility_pool]

        # ── Step 3: Safety filter on pool ──
        passed = []
        rejected_reasons = {}
        for c in volatility_pool_coins:
            coin = c["coin"]

            if c["volume_usdt"] < min_volume_usdt:
                rejected_reasons[coin] = f"vol ${c['volume_usdt']/1e6:.0f}M < ${min_volume_usdt/1e6:.0f}M"
                continue
            if c["spread_bps"] > max_spread_bps:
                rejected_reasons[coin] = f"spread {c['spread_bps']:.1f}bps > {max_spread_bps}bps"
                continue
            if c["last"] < min_price_usd:
                rejected_reasons[coin] = f"price ${c['last']:.4f} < ${min_price_usd}"
                continue

            passed.append(c)

        # ── Step 4: Always include held coins ──
        passed_coins = {c["coin"] for c in passed}
        if held_coins:
            for coin in held_coins:
                if coin not in passed_coins:
                    # Find coin data from all_coins
                    coin_data = next((c for c in all_coins if c["coin"] == coin), None)
                    if coin_data:
                        passed.append(coin_data)
                    else:
                        passed.append({"coin": coin, "volatility_pct": 0,
                                       "volume_usdt": 0, "spread_bps": 0, "last": 0})

        # ── Logging ──
        logger.info(
            f"[CoinSelect] Volatility top {volatility_pool} → "
            f"{len(passed)} passed safety filter | "
            f"rejected: {len(rejected_reasons)}"
        )
        for c in passed:
            logger.info(
                f"  ✓ {c['coin']:6s} vol%={c['volatility_pct']:5.2f}% "
                f"vol=${c['volume_usdt']/1e6:>7.0f}M "
                f"spread={c['spread_bps']:4.1f}bps "
                f"${c['last']:.4f}"
            )
        for coin, reason in list(rejected_reasons.items())[:5]:
            logger.debug(f"  ✗ {coin:6s} {reason}")

        return passed

    # ── Cache management ──────────────────────────────────

    def invalidate(self, coin: str = None) -> None:
        """Clear cache. If coin=None, clear all."""
        if coin is None:
            self._ohlcv_cache.clear()
            self._ticker_cache.clear()
        else:
            keys_to_remove = [k for k in self._ohlcv_cache if f":{coin}:" in k]
            for k in keys_to_remove:
                del self._ohlcv_cache[k]
            self._ticker_cache.pop(coin, None)
