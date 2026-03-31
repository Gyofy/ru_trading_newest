"""BacktestEngine — 바 단위 역사적 데이터 재생 백테스트.

BacktestDataHub: DataHub 서브클래스, 역사적 OHLCV 슬라이스 제공
BacktestExchangeAdapter: 즉시 체결 시뮬레이션 (슬리피지 포함)
BacktestBot: MultiStrategyBot 서브클래스, 바 단위 실행
"""

from __future__ import annotations

import asyncio
import logging
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np

# Add project root to path
_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.strategies.data_hub import DataHub

logger = logging.getLogger("backtest_engine")


class BacktestDataHub(DataHub):
    """DataHub subclass that serves historical OHLCV slices instead of live data.

    Does NOT call super().__init__() — sets required attributes manually.
    """

    def __init__(self, full_data: dict[str, pd.DataFrame]) -> None:
        # Do NOT call super().__init__() to avoid creating ExchangeAdapter
        self._full_data = full_data
        self._current_bar = 0
        self._ohlcv_cache: dict = {}
        self._ticker_cache: dict = {}
        self._locks: dict = {}
        self._semaphore = asyncio.Semaphore(10)
        self._executor = None
        self._stat_hits = 0
        self._stat_misses = 0
        self._stat_errors = 0
        self.ohlcv_ttl = 0
        self.ticker_ttl = 0

    def set_bar(self, bar_index: int) -> None:
        """Advance simulation time to bar_index."""
        self._current_bar = bar_index

    async def get_ohlcv(
        self,
        coin: str,
        timeframe: str = "1m",
        limit: int = 500,
    ) -> Optional[pd.DataFrame]:
        """Return historical slice up to current bar."""
        df_1m = self._full_data.get(coin)
        if df_1m is None or len(df_1m) == 0:
            return None

        bar = self._current_bar
        start_idx = max(0, bar + 1 - limit)
        end_idx = bar + 1

        if start_idx >= end_idx:
            return None

        slice_1m = df_1m.iloc[start_idx:end_idx].copy()

        if timeframe == "1m":
            return slice_1m
        elif timeframe == "5m":
            return self._resample(slice_1m, "5min")
        elif timeframe == "15m":
            return self._resample(slice_1m, "15min")
        elif timeframe == "1h":
            return self._resample(slice_1m, "1h")
        else:
            # Fallback: return 1m slice
            return slice_1m

    @staticmethod
    def _resample(df: pd.DataFrame, rule: str) -> Optional[pd.DataFrame]:
        """Resample 1m OHLCV to a coarser timeframe."""
        if df is None or len(df) == 0:
            return None
        try:
            agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
            if "taker_buy_base_vol" in df.columns:
                agg["taker_buy_base_vol"] = "sum"
            resampled = df.resample(rule).agg(agg).dropna(subset=["close"])
            return resampled if len(resampled) > 0 else None
        except Exception as e:
            logger.debug(f"[BacktestDataHub] resample({rule}) failed: {e}")
            return None

    async def get_ticker(self, coin: str) -> dict:
        """Return ticker dict from current bar OHLCV."""
        df = self._full_data.get(coin)
        if df is None or self._current_bar >= len(df):
            return {"bid": 0, "ask": 0, "last": 0, "spread_bps": 0, "high": 0, "low": 0}

        bar = df.iloc[self._current_bar]
        last = float(bar["close"])
        high = float(bar["high"])
        low = float(bar["low"])
        # Simulate 1bp spread
        spread_factor = 0.0001
        bid = last * (1 - spread_factor / 2)
        ask = last * (1 + spread_factor / 2)
        spread_bps = (ask - bid) / last * 10000 if last > 0 else 0.0

        return {
            "bid": bid,
            "ask": ask,
            "last": last,
            "spread_bps": spread_bps,
            "high": high,
            "low": low,
        }

    async def get_open_interest(self, coin: str) -> None:
        """OI not available in OHLCV."""
        return None

    async def get_funding_rate(self, coin: str) -> float:
        """Return neutral funding rate."""
        return 0.0001

    async def select_tradeable_coins(self, **kwargs) -> list[dict]:
        """Return all coins in full_data at current bar."""
        result = []
        for coin, df in self._full_data.items():
            if df is None or self._current_bar >= len(df):
                continue
            bar = df.iloc[self._current_bar]
            last = float(bar["close"])
            result.append({
                "coin": coin,
                "volatility_pct": 0.0,
                "volume_usdt": float(bar.get("volume", 0)) * last,
                "spread_bps": 1.0,
                "last": last,
            })
        return result

    def cache_stats(self) -> dict:
        """Return basic cache stats."""
        stats = {
            "hits": self._stat_hits,
            "misses": self._stat_misses,
            "errors": self._stat_errors,
            "cache_size": 0,
        }
        self._stat_hits = 0
        self._stat_misses = 0
        self._stat_errors = 0
        return stats

    def invalidate(self, coin: str = None) -> None:
        """No-op for backtest."""
        pass

    async def get_signal_quality(
        self, coin: str, timeframe: str = "1m", window: int = 24
    ) -> dict:
        """Return empty signal quality dict."""
        return {}

    async def refresh_all(self, coins: list[str], timeframe: str = "1m") -> None:
        """No-op for backtest."""
        pass


class BacktestExchangeAdapter:
    """Simulates exchange order execution with slippage for backtesting."""

    def __init__(
        self,
        bt_hub: BacktestDataHub,
        slippage_bps: float = 5.0,
    ) -> None:
        self._bt_hub = bt_hub
        self._slippage_bps = slippage_bps
        self.mode = "paper"
        self.paper_mode = True
        # precision: 소수점 자릿수 (실거래소 호출 없이 4자리로 통일)
        self._qty_precision = 4
        self._price_precision = 4

    def round_qty(self, symbol: str, qty: float) -> float:
        """백테스트용 수량 반올림 (8자리 소수)."""
        return round(qty, 8)

    def round_price(self, symbol: str, price: float) -> float:
        """백테스트용 가격 반올림 (6자리 소수)."""
        return round(price, 6)

    def round_all(
        self, symbol: str, qty: float, entry: float, sl: float, tp: float
    ) -> tuple:
        return (
            self.round_qty(symbol, qty),
            self.round_price(symbol, entry),
            self.round_price(symbol, sl),
            self.round_price(symbol, tp),
        )

    def _ccxt_symbol(self, symbol: str) -> str:
        return f"{symbol}/USDT:USDT"

    def _client_id_key(self) -> str:
        return "newClientOrderId"

    def _apply_slippage(self, price: float, side: str) -> float:
        """Apply slippage: BUY pays more, SELL receives less."""
        factor = self._slippage_bps / 10000.0
        if side.upper() in ("BUY", "LONG"):
            return price * (1 + factor)
        else:
            return price * (1 - factor)

    async def _get_last_price(self, coin: str) -> float:
        ticker = await self._bt_hub.get_ticker(coin)
        return float(ticker.get("last", 0))

    async def place_post_only_entry(
        self, coin: str, side: str, qty: float, price: float, order_link_id: str, **kwargs
    ) -> dict:
        """Fill at last price + slippage.

        Parameter name matches ExchangeAdapter.place_post_only_entry (order_link_id).
        """
        last = await self._get_last_price(coin)
        fill_price = self._apply_slippage(last if last > 0 else price, side)
        return {
            "success": True,
            "order_id": order_link_id,
            "exchange_order_id": order_link_id,
            "id": order_link_id,
            "status": "closed",
            "filled": qty,
            "price": fill_price,
            "average": fill_price,
        }

    async def place_market_entry(
        self, coin: str, side: str, qty: float, order_id: str, **kwargs
    ) -> dict:
        """Fill at last price + slippage."""
        last = await self._get_last_price(coin)
        fill_price = self._apply_slippage(last, side)
        return {
            "success": True,
            "order_id": order_id,
            "exchange_order_id": order_id,
            "id": order_id,
            "status": "closed",
            "filled": qty,
            "price": fill_price,
            "average": fill_price,
        }

    async def place_protective_stop(
        self,
        symbol: str,
        side: str,
        qty: float,
        stop_price: float,
        order_link_id: str,
        parent_id: str = "",
    ) -> dict:
        """Return fake SL order (barriers handled by BacktestBot bar-by-bar)."""
        return {
            "success": True,
            "order_id": order_link_id,
            "exchange_order_id": order_link_id,
            "status": "conditional",
        }

    async def place_take_profit(
        self,
        symbol: str,
        side: str,
        qty: float,
        tp_price: float,
        order_link_id: str,
        parent_id: str = "",
    ) -> dict:
        """Return fake TP order (barriers handled by BacktestBot bar-by-bar)."""
        return {
            "success": True,
            "order_id": order_link_id,
            "exchange_order_id": order_link_id,
            "status": "conditional",
        }

    async def market_close(
        self,
        symbol: str,
        side: str,
        qty: float,
        order_link_id: str,
    ) -> dict:
        """Simulate immediate market close."""
        last = await self._get_last_price(symbol)
        fill_price = self._apply_slippage(last, side)
        return {
            "success": True,
            "order_id": order_link_id,
            "price": fill_price,
            "status": "closed",
        }

    async def cancel_order(
        self,
        coin: str,
        order_link_id: str | None = None,
        exchange_order_id: str | None = None,
        **kwargs,
    ) -> bool:
        """No-op — matches ExchangeAdapter.cancel_order signature."""
        return True

    async def cancel_all_orders(self, coin: str, **kwargs) -> None:
        """No-op."""
        pass

    async def fetch_ticker(self, coin: str) -> dict:
        """Delegate to bt_hub."""
        return await self._bt_hub.get_ticker(coin)

    async def fetch_balance(self) -> dict:
        """Return simulated balance matching ExchangeAdapter.fetch_balance() structure:
        {"total": float, "free": float, "used": float}
        """
        return {"total": 10000.0, "free": 10000.0, "used": 0.0}

    async def fetch_position(self, coin: str) -> None:
        """Backtest tracks positions internally."""
        return None

    async def set_leverage(self, coin: str, leverage: int) -> None:
        """No-op."""
        pass

    async def set_margin_mode(self, coin: str, mode: str) -> None:
        """No-op."""
        pass

    async def initialize(self) -> None:
        """No-op."""
        pass

    async def close(self) -> None:
        """No-op."""
        pass

    def make_order_id(
        self, symbol: str, side: str, seq: int = 1, prefix: str = "bt"
    ) -> str:
        """Generate a fake order ID matching ExchangeAdapter.make_order_id signature."""
        return f"{prefix}-{symbol.lower()}-{side[0].lower()}-{seq:04d}-{uuid.uuid4().hex[:6]}"


async def fetch_ohlcv_history(
    exchange_adapter,
    coin: str,
    start_date: str,
    end_date: str,
    timeframe: str = "1m",
    cache_dir: Path = Path("data/backtest_cache"),
) -> Optional[pd.DataFrame]:
    """Fetch historical OHLCV from Binance with pagination, cache to parquet."""
    cache_file = cache_dir / f"{coin}_{timeframe}_{start_date}_{end_date}.parquet"
    if cache_file.exists():
        logger.info(f"[DataFetch] {coin}: loading from cache {cache_file}")
        df = pd.read_parquet(cache_file)
        return df

    symbol = f"{coin}/USDT:USDT"  # Binance Futures format
    since_ms = int(pd.Timestamp(start_date, tz="UTC").timestamp() * 1000)
    end_ms = int(pd.Timestamp(end_date, tz="UTC").timestamp() * 1000)

    all_bars: list = []
    while since_ms < end_ms:
        try:
            bars = await exchange_adapter._exchange.fetch_ohlcv(
                symbol, timeframe=timeframe, since=since_ms, limit=1000
            )
        except Exception as e:
            logger.warning(f"[DataFetch] {coin} error: {e}")
            break

        if not bars:
            break
        all_bars.extend(bars)
        since_ms = bars[-1][0] + 1  # next ms after last bar
        await asyncio.sleep(0.1)  # rate limit

    if not all_bars:
        logger.warning(f"[DataFetch] {coin}: no data fetched")
        return None

    df = pd.DataFrame(all_bars, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("timestamp", inplace=True)
    df = df[df.index <= pd.Timestamp(end_date, tz="UTC")]

    cache_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache_file)
    logger.info(f"[DataFetch] {coin}: {len(df)} bars saved to {cache_file}")
    return df


class BacktestBot:
    """MultiStrategyBot subclass that runs bar-by-bar on historical data.

    Injects BacktestDataHub and BacktestExchangeAdapter instead of live components.
    """

    def __init__(
        self,
        config_path: str,
        bt_hub: Optional[BacktestDataHub] = None,
        bt_exchange: Optional[BacktestExchangeAdapter] = None,
    ) -> None:
        # Import here to avoid circular imports at module load time
        from run_multi_strategy import MultiStrategyBot
        self._MultiStrategyBot = MultiStrategyBot

        import yaml
        with open(config_path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

        self.config_path = config_path
        self._bt_hub = bt_hub
        self._bt_exchange = bt_exchange
        self._session_trades: list[dict] = []
        self._paper_equity = float(self.cfg.get("initial_equity", 5000.0))

    async def start(self) -> None:
        """Load history, build backtest components, run bar-by-bar simulation."""
        logger.info("[BacktestBot] Starting backtest...")

        bt_cfg = self.cfg.get("backtest", {})
        start_date = bt_cfg.get("start_date", "2025-01-01")
        end_date = bt_cfg.get("end_date", "2025-03-30")
        coins = bt_cfg.get("coins", ["SOL", "XRP", "DOGE"])
        cache_dir = Path(bt_cfg.get("data_cache_dir", "data/backtest_cache"))
        slippage_bps = float(bt_cfg.get("slippage_bps", 5.0))

        full_data = await self._load_or_fetch_history(
            coins=coins,
            start_date=start_date,
            end_date=end_date,
            cache_dir=cache_dir,
        )

        if not full_data:
            logger.error("[BacktestBot] No data loaded — aborting")
            return

        bt_hub = BacktestDataHub(full_data)
        bt_exchange = BacktestExchangeAdapter(bt_hub, slippage_bps=slippage_bps)

        await self._run_backtest(full_data, bt_hub, bt_exchange, start_date, end_date)

    async def _load_or_fetch_history(
        self,
        coins: list[str],
        start_date: str,
        end_date: str,
        cache_dir: Path,
    ) -> dict[str, pd.DataFrame]:
        """Load OHLCV from parquet cache or fetch from Binance API."""
        from src.execution.exchange_adapter import ExchangeAdapter

        full_data: dict[str, pd.DataFrame] = {}

        # 캐시에서 로드 가능한 코인 먼저 처리
        coins_to_fetch = []
        for coin in coins:
            cache_file = cache_dir / f"{coin}_1m_{start_date}_{end_date}.parquet"
            if cache_file.exists():
                try:
                    df = pd.read_parquet(cache_file)
                    if not isinstance(df.index, pd.DatetimeIndex):
                        df.index = pd.to_datetime(df.index, utc=True)
                    full_data[coin] = df
                    logger.info(f"[DataLoad] {coin}: {len(df)} bars from cache")
                    continue
                except Exception as e:
                    logger.warning(f"[DataLoad] {coin} cache read failed: {e}")
            coins_to_fetch.append(coin)

        if not coins_to_fetch:
            return full_data

        # 하나의 ExchangeAdapter로 모든 코인 fetch → close 1회만 호출
        exchange = ExchangeAdapter(
            mode="live",
            api_key=os.environ.get("BINANCE_API_KEY", ""),
            secret=os.environ.get("BINANCE_API_SECRET", ""),
        )
        try:
            await exchange.initialize()
            for coin in coins_to_fetch:
                logger.info(f"[DataLoad] {coin}: fetching from Binance API...")
                try:
                    df = await fetch_ohlcv_history(
                        exchange_adapter=exchange,
                        coin=coin,
                        start_date=start_date,
                        end_date=end_date,
                        timeframe="1m",
                        cache_dir=cache_dir,
                    )
                    if df is not None:
                        full_data[coin] = df
                        logger.info(f"[DataLoad] {coin}: {len(df)} bars fetched")
                    else:
                        logger.warning(f"[DataLoad] {coin}: fetch returned None")
                except Exception as e:
                    logger.error(f"[DataLoad] {coin}: fetch failed: {e}")
        finally:
            await exchange.close()

        return full_data

    async def _run_backtest(
        self,
        full_data: dict[str, pd.DataFrame],
        bt_hub: BacktestDataHub,
        bt_exchange: BacktestExchangeAdapter,
        start_date: str,
        end_date: str,
    ) -> None:
        """Execute bar-by-bar backtest simulation."""
        from run_multi_strategy import MultiStrategyBot

        # Build a minimal MultiStrategyBot with paper mode to reuse strategy logic.
        # Suppress Discord notifications during backtest: patch the module-level discord_post
        # function so no real webhooks fire.  Use try/finally to guarantee restoration
        # even if the backtest run raises an exception.
        import run_multi_strategy as _rms_mod

        # 백테스트 state_dir → 데모와 동일한 파일 구조 재현
        # TRADES_FILE / EQUITY_STATE_FILE 은 run_multi_strategy 모듈 레벨 상수로
        # import 시점에 기본 STATE_DIR로 고정됨.  패치해서 backtest state_dir을 가리키게 함.
        _bt_state_dir = Path(os.environ.get("TRIPLE_STATE_DIR", "data/reports/multi_strategy"))
        _bt_state_dir.mkdir(parents=True, exist_ok=True)

        _real_discord_post = _rms_mod.discord_post
        _real_trades_file   = _rms_mod.TRADES_FILE
        _real_equity_file   = _rms_mod.EQUITY_STATE_FILE

        _rms_mod.discord_post    = lambda *a, **kw: None          # no-op during backtest
        _rms_mod.TRADES_FILE     = _bt_state_dir / "trades.jsonl"
        _rms_mod.EQUITY_STATE_FILE = _bt_state_dir / "equity_state.json"

        try:
            bot = MultiStrategyBot(self.config_path, mode_override="paper")

            # DATA LEAKAGE GUARD: disable StrategySolver.
            # StrategySolver auto-optimizes params every 3h by reading trade history.
            # If allowed to run during backtest it would tune params on the test period data
            # itself — equivalent to look-ahead bias.  We set it to None so the solver loop
            # (only started inside MultiStrategyBot.run(), which we do NOT call) can't start.
            bot.strategy_solver = None

            # Inject backtest components
            bot.data_hub = bt_hub
            bot.exchange = bt_exchange
            bot.coins = list(full_data.keys())
            bot._paper_equity = self._paper_equity

            # Initialize remaining infrastructure (without real exchange calls)
            await self._init_bot_components(bot, bt_exchange)

            # Determine reference timeline from first coin
            first_coin = next(iter(full_data))
            timeline = full_data[first_coin].index

            start_ts = pd.Timestamp(start_date, tz="UTC")
            end_ts = pd.Timestamp(end_date, tz="UTC")

            bar_range = [
                i for i, ts in enumerate(timeline)
                if start_ts <= ts <= end_ts
            ]

            if not bar_range:
                logger.warning("[BacktestBot] No bars in date range — check start/end dates")
                return

            logger.info(
                f"[Backtest] Running {len(bar_range)} bars "
                f"({start_date} → {end_date}) "
                f"on {len(full_data)} coins"
            )

            _report_interval_sec = 1800  # 30분마다 성적 출력
            _last_report_ts = time.time()
            _initial_equity = bot._paper_equity
            _total_bars = len(bar_range)

            for bar_idx in bar_range:
                bt_hub.set_bar(bar_idx)
                current_ts = timeline[bar_idx]

                # 1. Advance bars_held BEFORE barrier check so TTL is consistent
                if bot.pos_manager:
                    for pos in bot.pos_manager.all_positions():
                        pos.bars_held += 1

                # 2. Check barriers for all open positions
                await self._backtest_check_barriers(bot, bar_idx, full_data)

                # 3. Run strategy ticks (staggered by cycle)
                if bot.strategies:
                    for name, strategy in bot.strategies.items():
                        cycle_bars = max(1, strategy.config.cycle_seconds // 60)
                        if bar_idx % cycle_bars == 0:
                            try:
                                await strategy.tick(list(full_data.keys()))
                            except Exception as e:
                                logger.debug(f"[Backtest] {name} tick error at bar {bar_idx}: {e}")

                # 4. 30분 실시간 주기로 중간 성적 출력
                _now = time.time()
                if _now - _last_report_ts >= _report_interval_sec:
                    _last_report_ts = _now
                    self._print_progress_report(
                        bot=bot,
                        bar_idx=bar_idx,
                        total_bars=_total_bars,
                        current_ts=current_ts,
                        initial_equity=_initial_equity,
                        start_date=start_date,
                        end_date=end_date,
                    )

            # 5. 최종 완료 리포트
            self._print_progress_report(
                bot=bot,
                bar_idx=bar_range[-1],
                total_bars=_total_bars,
                current_ts=timeline[bar_range[-1]],
                initial_equity=_initial_equity,
                start_date=start_date,
                end_date=end_date,
                final=True,
            )
            self._session_trades = bot._session_trades

        finally:
            # 모든 패치 복원 — 예외 발생 여부와 무관하게 항상 실행
            _rms_mod.discord_post      = _real_discord_post
            _rms_mod.TRADES_FILE       = _real_trades_file
            _rms_mod.EQUITY_STATE_FILE = _real_equity_file

    async def _init_bot_components(
        self,
        bot,
        bt_exchange: BacktestExchangeAdapter,
    ) -> None:
        """Initialize bot infrastructure components without real exchange calls."""
        from src.execution.order_ledger import OrderLedger
        from src.strategies.multi_position_manager import MultiPositionManager
        from src.strategies.portfolio_risk import PortfolioRiskConfig, PortfolioRiskManager
        from src.strategies.sl_tp_monitor_v2 import SlTpMonitorV2
        from src.strategies.trade_logger import TradeLogger
        from src.strategies.coin_profile import CoinProfileStore
        from src.strategies.position_sizer import PositionSizer
        from src.strategies.entry_filters import EntryFilters
        from src.strategies.cluster_tracker import ClusterTracker
        from src.meta.drawdown_throttle import DrawdownThrottle
        from src.meta.fee_ev_gate import FeeEVGate
        from src.meta.dsr import DSREvaluator
        from src.meta.param_adjuster import ParamAdjuster, EVGridSearch
        from src.strategies.base import ROUND_TRIP_FEE_RATE
        from run_multi_strategy import STRATEGY_MAP, StrategyConfig

        # Bug fix: do NOT import STATE_DIR from run_multi_strategy — it is a module-level
        # constant resolved at import time, before TRIPLE_STATE_DIR env var is set.
        # Compute inline so each BacktestBot instance gets the correct isolated path.
        state_dir = Path(os.environ.get("TRIPLE_STATE_DIR", "data/reports/multi_strategy"))
        state_dir.mkdir(parents=True, exist_ok=True)

        # DATA LEAKAGE GUARD: if trade_context.jsonl already exists in this state dir
        # (e.g. from a previous backtest run), rename it to a timestamped backup and start
        # fresh.  Adaptive modules (FeeEVGate, ParamAdjuster, CoinProfileStore) calibrate
        # on this file; pre-existing data from a different period constitutes look-ahead bias.
        trade_ctx_file = state_dir / "trade_context.jsonl"
        if trade_ctx_file.exists() and trade_ctx_file.stat().st_size > 0:
            _backup = state_dir / f"trade_context_backup_{int(time.time())}.jsonl"
            trade_ctx_file.rename(_backup)
            logger.warning(
                f"[BacktestBot] DATA LEAKAGE GUARD: renamed pre-existing trade_context.jsonl "
                f"→ {_backup.name} to ensure clean backtest state."
            )

        bot.ledger = OrderLedger(state_dir / "orders.db")
        bot.pos_manager = MultiPositionManager(state_dir / "positions.json")
        # Clear any persisted positions for clean backtest run
        bot.pos_manager.positions = {}

        strategy_allocations = {}
        for name, scfg in bot.cfg.get("strategies", {}).items():
            if scfg.get("enabled", True):
                strategy_allocations[name] = scfg.get("allocation_usdt", 20.0)

        pcfg = bot.cfg.get("portfolio", {})
        bot.portfolio_risk = PortfolioRiskManager(
            config=PortfolioRiskConfig(
                total_exposure_pct=pcfg.get("total_exposure_pct", 0.80),
                same_direction_max=pcfg.get("same_direction_max", 3),
                daily_loss_pct=pcfg.get("daily_loss_pct", 0.20),
                strategy_loss_pct=pcfg.get("strategy_loss_pct", 0.40),
                min_notional_usdt=pcfg.get("min_notional_usdt", 5.0),
                max_funding_rate=pcfg.get("max_funding_rate", 0.001),
            ),
            pos_manager=bot.pos_manager,
            initial_equity=bot._paper_equity,
            strategy_allocations=strategy_allocations,
        )

        bot.trade_logger = TradeLogger(state_dir)

        from src.strategies.ev_guardian import EVGuardian
        ev_cfg = bot.cfg.get("ev_guardian", {})
        bot.ev_guardian = EVGuardian(
            jsonl_path=state_dir / "trade_context.jsonl",
            initial_equity=bot._paper_equity,
            fee_budget_pct=ev_cfg.get("fee_budget_pct", 0.005),
            report_path=state_dir / "ev_report.json",
        )

        bot.coin_profiles = CoinProfileStore(
            persist_path=state_dir / "coin_profiles.json",
            trade_context_path=state_dir / "trade_context.jsonl",
        )

        sizing_cfg = bot.cfg.get("position_sizing", {})
        bot.position_sizer = PositionSizer(sizing_cfg)

        ef_cfg = bot.cfg.get("entry_filters", {})
        bot.entry_filters = EntryFilters(ef_cfg)

        bot.cluster_tracker = ClusterTracker(window_sec=300.0)
        bot.drawdown_throttle = DrawdownThrottle()
        bot.fee_ev_gate = FeeEVGate(
            trade_context_path=state_dir / "trade_context.jsonl",
            round_trip_fee_rate=ROUND_TRIP_FEE_RATE,
        )
        bot.fee_ev_gate.refresh()  # Bug fix: load any existing trades in this state dir
        bot.dsr_evaluator = DSREvaluator(significance_level=0.95)
        bot.param_adjuster = ParamAdjuster(
            trade_context_path=state_dir / "trade_context.jsonl",
        )
        bot.ev_grid_search = EVGridSearch()

        # Monitor (paper mode = no real exchange calls)
        bot.monitor = SlTpMonitorV2(
            exchange=bt_exchange,
            pos_manager=bot.pos_manager,
            close_callback=bot._on_position_close,
            poll_seconds=9999,
            paper_mode=True,
            discord_notify=lambda msg, title="": None,
        )

        # Create strategies
        bot.strategies = {}
        for name, scfg in bot.cfg.get("strategies", {}).items():
            if not scfg.get("enabled", True):
                continue
            if name not in STRATEGY_MAP:
                logger.warning(f"[BacktestBot] Unknown strategy: {name}")
                continue

            config = StrategyConfig(
                name=name,
                enabled=True,
                allocation_usdt=scfg.get("allocation_usdt", 20.0),
                leverage=scfg.get("leverage", 2),
                cycle_seconds=scfg.get("cycle_seconds", 60),
                max_positions=scfg.get("max_positions", 2),
                sl_atr_mult=scfg.get("sl_atr_mult", 3.0),
                extra=scfg.get("extra", {}),
                paper_mode=True,
                bot_version=bot.cfg.get("version", "backtest"),
            )
            strategy = STRATEGY_MAP[name](
                config=config,
                exchange=bt_exchange,
                portfolio_risk=bot.portfolio_risk,
                pos_manager=bot.pos_manager,
                ledger=bot.ledger,
                data_hub=bot.data_hub,
                portfolio_lock=bot._portfolio_lock,
                trade_logger=bot.trade_logger,
                coin_profiles=bot.coin_profiles,
                position_sizer=bot.position_sizer,
                entry_filters=bot.entry_filters,
            )
            strategy.cluster_tracker = bot.cluster_tracker
            strategy.drawdown_throttle = bot.drawdown_throttle
            strategy.fee_ev_gate = bot.fee_ev_gate
            strategy.ev_grid_search = bot.ev_grid_search
            bot.strategies[name] = strategy

        logger.info(f"[BacktestBot] {len(bot.strategies)} strategies initialized")

    def _print_progress_report(
        self,
        bot,
        bar_idx: int,
        total_bars: int,
        current_ts,
        initial_equity: float,
        start_date: str,
        end_date: str,
        final: bool = False,
    ) -> None:
        """30분 주기 — TripleComparator로 paper/backtest/demo 3개 모드 동시 출력."""
        from triple_eval.triple_comparator import TripleComparator
        from triple_eval.triple_runner import STATE_DIRS

        progress_pct = bar_idx / total_bars * 100 if total_bars > 0 else 0.0
        label = "✅ 백테스트 완료" if final else "⏳ 백테스트 진행 중"
        header = (
            f"══ {label} [{current_ts.strftime('%Y-%m-%d %H:%M')} sim] "
            f"진행 {progress_pct:.1f}% ({bar_idx}/{total_bars}bars) ══"
        )
        logger.info(header)
        print("\n" + header, flush=True)

        # Discord webhook 은 전달하지 않음 — 터미널/로그 출력만
        webhook = os.environ.get("DISCORD_WEBHOOK_URL", "") if final else ""
        try:
            comparator = TripleComparator(
                state_dirs=STATE_DIRS,
                discord_webhook=webhook,
                lookback_hours=9999,  # 전체 누적 성적
                initial_equity=initial_equity,
            )
            comparator.run_and_post()
        except Exception as e:
            logger.warning(f"[ProgressReport] TripleComparator 실패: {e}")

    async def _backtest_check_barriers(
        self,
        bot,
        bar_idx: int,
        full_data: dict[str, pd.DataFrame],
    ) -> None:
        """Check SL/TP/TTL barriers for all open positions at current bar."""
        if not bot.pos_manager:
            return

        positions_to_close = []
        for pos in bot.pos_manager.all_positions():
            coin = pos.coin
            df = full_data.get(coin)
            if df is None or bar_idx >= len(df):
                continue

            bar = df.iloc[bar_idx]
            high = float(bar["high"])
            low = float(bar["low"])

            sl_hit = False
            tp_hit = False
            ttl_hit = (
                pos.bars_held >= pos.ttl_bars
                if pos.ttl_bars > 0
                else False
            )

            # Guard: sl_price=0 or tp_price=0 means barrier not set — skip check.
            # Without this, SELL positions would immediately SL (high >= 0 always True).
            if pos.side == "BUY":
                if pos.sl_price > 0:
                    sl_hit = low <= pos.sl_price
                if pos.tp_price > 0:
                    tp_hit = high >= pos.tp_price
            else:
                if pos.sl_price > 0:
                    sl_hit = high >= pos.sl_price
                if pos.tp_price > 0:
                    tp_hit = low <= pos.tp_price

            # When SL and TP are both hit in the same bar (gap open / extreme wick),
            # SL takes priority — conservative assumption: adverse move happened first.
            if sl_hit:
                exit_price = pos.sl_price
                reason = "SL_HIT"
            elif tp_hit:
                exit_price = pos.tp_price
                reason = "TP_HIT"
            elif ttl_hit:
                exit_price = float(df.iloc[bar_idx]["close"])
                reason = "TTL_HIT"
            else:
                continue

            positions_to_close.append((pos, exit_price, reason))

        for pos, exit_price, reason in positions_to_close:
            try:
                await bot._on_position_close(
                    strategy=pos.strategy_tag,
                    coin=pos.coin,
                    reason=reason,
                    price=exit_price,
                )
            except Exception as e:
                logger.debug(
                    f"[Backtest] _on_position_close failed "
                    f"{pos.strategy_tag}:{pos.coin} {reason}: {e}"
                )
