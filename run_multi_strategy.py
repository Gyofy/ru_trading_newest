"""v8.2 Multi-Strategy Trading Bot — 4 strategies, asymmetric-first design.

Strategies:
  A. CVD Spike Reactor   ($12, 3x, 1m cycle)  — counter-trend on extreme orderflow
  B. Liquidation Fade    ($8,  2x, 5m cycle)  — fade forced liquidation cascades
  C. Momentum Breakout   ($8,  2x, 5m cycle)  — volume-confirmed trend following
  D. Asymmetric Sniper   ($32, 5x, 1m cycle)  — Q99+3σ extreme, fixed-dollar risk

Risk: -20% daily kill, -40% per-strategy pause, 80% exposure cap, 3 same-dir max.
Entry: Post-Only Maker | SL/TP: Exchange-side | Funding gate active.

Usage:
  python run_multi_strategy.py                    # paper mode (default)
  python run_multi_strategy.py --mode live        # live mode
  python run_multi_strategy.py --config path.yaml # custom config
"""

from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import json
import logging
import os
import signal
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import yaml

# UTF-8 output for Windows
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── Logging ──────────────────────────────────────────────

STATE_DIR = Path("data/reports/multi_strategy")
STATE_DIR.mkdir(parents=True, exist_ok=True)
EQUITY_STATE_FILE = STATE_DIR / "equity_state.json"
TRADES_FILE = STATE_DIR / "trades.jsonl"

class _DiscordAlertHandler(logging.Handler):
    """ERROR 레벨 로그 중 [SL_FAIL] 태그가 있으면 Discord로 즉시 전송."""
    def emit(self, record: logging.LogRecord) -> None:
        if record.levelno >= logging.ERROR and "[SL_FAIL]" in record.getMessage():
            try:
                discord_post(f"🚨 {record.getMessage()}")
            except Exception:
                pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(STATE_DIR / "bot.log", encoding="utf-8"),
        _DiscordAlertHandler(),
    ],
)
log = logging.getLogger("multi_bot")


# ── Imports ──────────────────────────────────────────────

from src.execution.exchange_adapter import ExchangeAdapter
from src.execution.order_ledger import OrderLedger
from src.strategies.base import StrategyConfig
from src.strategies.cvd_spike import CVDSpikeReactor
from src.strategies.liquidation_fade import LiquidationFade
from src.strategies.momentum_breakout import MomentumBreakout
from src.strategies.asymmetric_sniper import AsymmetricSniper
from src.strategies.multi_position_manager import MultiPositionManager
from src.strategies.portfolio_risk import PortfolioRiskConfig, PortfolioRiskManager
from src.strategies.data_hub import DataHub
from src.strategies.sl_tp_monitor_v2 import SlTpMonitorV2
from src.strategies.trade_logger import TradeLogger
from src.strategies.coin_profile import CoinProfileStore
from src.strategies.position_sizer import PositionSizer
from src.strategies.strategy_analyzer import StrategyAnalyzer

STRATEGY_MAP = {
    "cvd_spike": CVDSpikeReactor,
    "liquidation_fade": LiquidationFade,
    "momentum_breakout": MomentumBreakout,
    "asymmetric_sniper": AsymmetricSniper,
}


# ── Discord ──────────────────────────────────────────────

def _discord_post_sync(message: str, title: str) -> None:
    """동기 Discord 전송 (executor에서 호출용). Rate-limited: 1 per 2s."""
    global _discord_last_ts
    import time as _time
    with _discord_rate_lock:
        elapsed = _time.monotonic() - _discord_last_ts
        if elapsed < _DISCORD_MIN_INTERVAL:
            _time.sleep(_DISCORD_MIN_INTERVAL - elapsed)
        _discord_last_ts = _time.monotonic()
    url = os.environ.get("DISCORD_WEBHOOK_URL", "")
    if not url:
        return
    # Discord embed description 4096자 제한
    if len(message) > 4000:
        message = message[:3997] + "..."
    try:
        payload = json.dumps({
            "embeds": [{
                "title": (title or "Multi-Strategy Bot")[:256],
                "description": message,
                "color": 3447003,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }]
        }).encode("utf-8")
        req = urllib.request.Request(
            url, data=payload,
            headers={
                "Content-Type": "application/json",
                "User-Agent": "DiscordBot (ru_trading_bot, 1.0)",
            },
            method="POST",
        )
        urllib.request.urlopen(req, timeout=5)
    except Exception as e:
        log.error(f"[Discord] Post failed: {e}")


import threading as _threading
_discord_rate_lock = _threading.Lock()
_discord_last_ts: float = 0.0
_DISCORD_MIN_INTERVAL = 2.0  # Discord rate limit: max 30/min → 1 per 2s minimum


def discord_post(message: str, title: str = "") -> None:
    """Discord 전송 — asyncio 이벤트 루프 비블로킹. Rate-limited to 1 per 2s."""
    url = os.environ.get("DISCORD_WEBHOOK_URL", "")
    if not url:
        log.warning("[Discord] DISCORD_WEBHOOK_URL not set")
        return
    try:
        loop = asyncio.get_running_loop()
        loop.run_in_executor(None, _discord_post_sync, message, title)
    except RuntimeError:
        # 이벤트 루프 밖에서 호출된 경우 (startup 등)
        _discord_post_sync(message, title)


class MultiStrategyBot:
    """Orchestrator that runs multiple strategies concurrently."""

    def __init__(self, config_path: str, mode_override: str | None = None):
        with open(config_path, "r", encoding="utf-8") as f:
            self.cfg = yaml.safe_load(f)

        self.mode = mode_override or self.cfg.get("mode", "paper")
        # Load persisted equity — restores balance across restarts
        self.initial_equity = self._load_equity_state()
        self._static_coins = self.cfg.get("coins", ["SOL", "XRP", "ADA", "DOT"])
        self.coins = list(self._static_coins)  # mutable, updated dynamically
        self._coin_metadata: list[dict] = []   # full metadata from selector
        dcfg = self.cfg.get("dynamic_coins", {})
        self._dynamic_coin_selection = dcfg.get("enabled", False)
        self._dc_volatility_pool = dcfg.get("volatility_pool", 30)
        self._dc_selective_pool = dcfg.get("selective_pool_size", 20)
        self._dc_min_vol = dcfg.get("min_volume_usdt", 50_000_000)
        self._dc_max_spread = dcfg.get("max_spread_bps", 8.0)
        self._dc_refresh_sec = dcfg.get("refresh_interval_sec", 900)
        self._dc_base_coins: list[str] = dcfg.get("base_coins", ["XRP", "SOL", "TAO", "DOGE", "ADA"])

        # Shared lock for portfolio-level atomicity
        self._portfolio_lock = asyncio.Lock()
        self._shutdown_event = asyncio.Event()

        # Components (initialized in start())
        self.exchange: ExchangeAdapter | None = None
        self.ledger: OrderLedger | None = None
        self.pos_manager: MultiPositionManager | None = None
        self.portfolio_risk: PortfolioRiskManager | None = None
        self.data_hub: DataHub | None = None
        self.monitor: SlTpMonitorV2 | None = None
        self.trade_logger: TradeLogger | None = None
        self.coin_profiles: CoinProfileStore | None = None
        self.position_sizer: PositionSizer | None = None
        self.strategies: dict[str, object] = {}

        # Paper mode tracking — load historical trades for continuity
        self._paper_equity = self.initial_equity
        self._paper_trades: list[dict] = self._load_trades_history()
        self._first_trade_notified = len(self._paper_trades) > 0

        # Graceful update: --keep-positions prevents closing positions on shutdown
        self._keep_positions: bool = False

        # Process pool for CPU-bound signal computation (50% of cores)
        mp_cfg = self.cfg.get("multiprocessing", {})
        _workers = mp_cfg.get("workers", 8)
        self._executor = concurrent.futures.ProcessPoolExecutor(max_workers=_workers)
        log.info(f"ProcessPoolExecutor initialized: {_workers} workers")

    async def start(self) -> None:
        """Initialize all components and start trading."""
        log.info(f"{'='*60}")
        log.info(f"Multi-Strategy Bot v8.2 | mode={self.mode}")
        log.info(f"equity=${self.initial_equity} | coins={self.coins}")
        log.info(f"{'='*60}")

        # Exchange
        if self.mode == "live":
            self.exchange = ExchangeAdapter(
                mode="live",
                api_key=os.environ.get("BINANCE_API_KEY", ""),
                secret=os.environ.get("BINANCE_API_SECRET", ""),
            )
            await self.exchange.initialize()
            balance = await self.exchange.fetch_balance()
            self.initial_equity = balance["total"]
            log.info(f"Live balance: ${self.initial_equity:.2f}")
        elif self.mode == "demo":
            # Demo mode — Binance Futures Testnet (real orders, virtual funds)
            testnet_key = os.environ.get("BINANCE_TESTNET_API_KEY", "")
            testnet_secret = os.environ.get("BINANCE_TESTNET_API_SECRET", "")
            if not testnet_key:
                log.warning("[Demo] BINANCE_TESTNET_API_KEY not set — falling back to paper mode")
                self.mode = "paper"
                self.exchange = ExchangeAdapter(
                    mode="live",
                    api_key=os.environ.get("BINANCE_API_KEY", ""),
                    secret=os.environ.get("BINANCE_API_SECRET", ""),
                )
            else:
                self.exchange = ExchangeAdapter(
                    mode="sandbox",
                    api_key=testnet_key,
                    secret=testnet_secret,
                )
            await self.exchange.initialize()
            log.info(f"Demo/testnet mode: virtual funds, real orders on testnet | equity=${self.initial_equity:.2f}")
        else:
            # Paper mode — use live exchange for market data only (no orders)
            self.exchange = ExchangeAdapter(
                mode="live",
                api_key=os.environ.get("BINANCE_API_KEY", ""),
                secret=os.environ.get("BINANCE_API_SECRET", ""),
            )
            await self.exchange.initialize()
            log.info("Paper mode: using live market data, orders simulated")

        # Shared infrastructure
        self.ledger = OrderLedger(STATE_DIR / "orders.db")
        self.pos_manager = MultiPositionManager(STATE_DIR / "positions.json")
        self.data_hub = DataHub(self.exchange, executor=self._executor)
        self.trade_logger = TradeLogger(STATE_DIR)
        self.coin_profiles = CoinProfileStore(
            persist_path=STATE_DIR / "coin_profiles.json",
            trade_context_path=STATE_DIR / "trade_context.jsonl",
        )
        sizing_cfg = self.cfg.get("position_sizing", {})
        self.position_sizer = PositionSizer(sizing_cfg)

        # Portfolio risk
        strategy_allocations = {}
        for name, scfg in self.cfg.get("strategies", {}).items():
            if scfg.get("enabled", True):
                strategy_allocations[name] = scfg.get("allocation_usdt", 20.0)

        pcfg = self.cfg.get("portfolio", {})
        self.portfolio_risk = PortfolioRiskManager(
            config=PortfolioRiskConfig(
                total_exposure_pct=pcfg.get("total_exposure_pct", 0.80),
                same_direction_max=pcfg.get("same_direction_max", 3),
                daily_loss_pct=pcfg.get("daily_loss_pct", 0.20),
                strategy_loss_pct=pcfg.get("strategy_loss_pct", 0.40),
                min_notional_usdt=pcfg.get("min_notional_usdt", 5.0),
                max_funding_rate=pcfg.get("max_funding_rate", 0.001),
            ),
            pos_manager=self.pos_manager,
            initial_equity=self.initial_equity,
            strategy_allocations=strategy_allocations,
        )

        # Monitor
        mcfg = self.cfg.get("monitoring", {})
        self.monitor = SlTpMonitorV2(
            exchange=self.exchange,
            pos_manager=self.pos_manager,
            close_callback=self._on_position_close,
            poll_seconds=mcfg.get("sl_tp_poll_seconds", 15),
            paper_mode=(self.mode == "paper"),
        )

        # Create strategies
        for name, scfg in self.cfg.get("strategies", {}).items():
            if not scfg.get("enabled", True):
                continue
            if name not in STRATEGY_MAP:
                log.warning(f"Unknown strategy: {name}")
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
                paper_mode=(self.mode == "paper"),  # demo/live = real testnet/live orders
            )
            strategy = STRATEGY_MAP[name](
                config=config,
                exchange=self.exchange,
                portfolio_risk=self.portfolio_risk,
                pos_manager=self.pos_manager,
                ledger=self.ledger,
                data_hub=self.data_hub,
                portfolio_lock=self._portfolio_lock,
                trade_logger=self.trade_logger,
                coin_profiles=self.coin_profiles,
                position_sizer=self.position_sizer,
            )
            self.strategies[name] = strategy
            log.info(
                f"[{name}] loaded | ${config.allocation_usdt} | "
                f"{config.leverage}x | cycle={config.cycle_seconds}s | "
                f"max_pos={config.max_positions}"
            )

        # Start monitor
        self.monitor.start()

        # Discord: recovered positions notification (if any)
        recovered = self.pos_manager.all_positions()
        if recovered:
            rec_lines = "\n".join(
                f"• `{p.coin}` {p.side} @ `{p.entry_price:.4f}` | SL:`{p.sl_price:.4f}` | [{p.strategy_tag}]"
                for p in recovered
            )
            discord_post(
                f"**{len(recovered)}개 포지션 복구됨 — 모니터링 재개**\n{rec_lines}",
                title="🔄 포지션 복구 — 재시작",
            )
            log.info(f"[Recovery] Resuming monitoring for {len(recovered)} recovered positions")

        # Discord: system ready notification
        strategy_summary = "\n".join(
            f"• {n}: ${scfg.get('allocation_usdt',0):.0f} / {scfg.get('leverage',2)}x / max {scfg.get('max_positions',1)} pos"
            for n, scfg in self.cfg.get("strategies", {}).items()
            if scfg.get("enabled", True)
        )
        discord_post(
            f"**모드:** `{self.mode}`\n"
            f"**자본금:** `${self.initial_equity:,.2f}`\n"
            f"**최대 노셔널:** `{self.cfg.get('portfolio',{}).get('total_exposure_pct',0.7)*100:.0f}%`\n"
            f"**코인:** `{', '.join(self.coins)}`\n\n"
            f"**전략 구성:**\n{strategy_summary}",
            title="✅ v8.2 시스템 준비 완료 — 거래 시작",
        )
        log.info("[Discord] System ready notification sent")

        # Run strategies concurrently
        await self._run()

    async def _run(self) -> None:
        """Run all strategy loops + maintenance concurrently."""
        tasks = []
        for name, strategy in self.strategies.items():
            tasks.append(asyncio.create_task(
                self._strategy_loop(strategy), name=f"loop:{name}"
            ))

        # Bar counter (1m resolution for TTL tracking)
        tasks.append(asyncio.create_task(
            self._bar_counter(), name="bar_counter"
        ))

        # Heartbeat (log)
        tasks.append(asyncio.create_task(
            self._heartbeat(), name="heartbeat"
        ))

        # Discord: 1h position briefing
        tasks.append(asyncio.create_task(
            self._discord_hourly_briefing(), name="discord_hourly"
        ))

        # Discord: 30m signal scan
        tasks.append(asyncio.create_task(
            self._discord_signal_scan(), name="discord_signal_scan"
        ))

        # Dynamic coin refresh (1h)
        if self._dynamic_coin_selection:
            tasks.append(asyncio.create_task(
                self._coin_refresh_loop(), name="coin_refresh"
            ))

        # Strategy analysis (every 6h)
        tasks.append(asyncio.create_task(
            self._strategy_analysis_loop(), name="strategy_analysis"
        ))

        # Wait for shutdown
        tasks.append(asyncio.create_task(
            self._shutdown_event.wait(), name="shutdown_wait"
        ))

        done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

        # Cancel remaining tasks
        for t in pending:
            t.cancel()

        await self.shutdown()

    async def _strategy_loop(self, strategy) -> None:
        """Per-strategy evaluation loop."""
        cycle = strategy.config.cycle_seconds
        log.info(f"[{strategy.name}] loop started (every {cycle}s)")

        # Initial delay to stagger strategies
        delay = {"cvd_spike": 0, "liquidation_fade": 10, "momentum_breakout": 20, "asymmetric_sniper": 5}
        await asyncio.sleep(delay.get(strategy.name, 0))

        while not self._shutdown_event.is_set():
            try:
                # Check kill switch
                if self.portfolio_risk.is_killed:
                    log.warning(f"[{strategy.name}] kill switch active, stopping")
                    break

                # Daily reset
                self.portfolio_risk.maybe_reset_daily()

                # Evaluate
                results = await strategy.tick(self.coins)
                # Per-scan diagnostics: cache stats + result summary
                cache_info = self.data_hub.cache_stats()
                log.info(
                    f"[Scan:{strategy.name}] trades={len(results)} | "
                    f"cache hits={cache_info.get('hits',0)} miss={cache_info.get('misses',0)} "
                    f"err={cache_info.get('errors',0)} | "
                    f"positions={self.pos_manager.count()}"
                )
                for r in results:
                    log.info(f"[{strategy.name}] TRADE: {r}")
                    # Discord: 진입 알림 (base.py는 "side" 키로 반환)
                    _r_side = r.get("side", r.get("action", "")) if isinstance(r, dict) else ""
                    if _r_side in ("BUY", "SELL", "LONG", "SHORT"):
                        side = _r_side
                        coin_r = r.get("coin", "?")
                        price_r = r.get("price", r.get("entry_price", 0))
                        sl_r = r.get("sl_price", r.get("sl", 0))
                        tp_r = r.get("tp_price", r.get("tp", 0))
                        trail_r = r.get("trailing", False)
                        try:
                            portfolio_snap = await self._build_portfolio_snapshot(
                                event_coin=coin_r, event_side=side
                            )
                        except Exception:
                            portfolio_snap = "(포트폴리오 조회 실패)"
                        tp_label = "trailing" if trail_r else f"`{tp_r:.4f}`"
                        discord_post(
                            f"**{'🟢 LONG' if side in ('BUY','LONG') else '🔴 SHORT'} 진입**\n"
                            f"종목: `{coin_r}` | 전략: `[{strategy.name}]`\n"
                            f"진입가: `{price_r:.4f}` | SL: `{sl_r:.4f}` | TP: {tp_label}\n\n"
                            f"{portfolio_snap}",
                            title=f"📥 {'🟢 LONG' if side in ('BUY','LONG') else '🔴 SHORT'} 진입 — {coin_r}",
                        )

            except asyncio.CancelledError:
                break
            except Exception as e:
                log.error(f"[{strategy.name}] loop error: {e}", exc_info=True)

            try:
                await asyncio.sleep(cycle)
            except asyncio.CancelledError:
                break

    async def _coin_refresh_loop(self) -> None:
        """Refresh coin list based on volatility ranking + safety filter."""
        # Initial selection at startup
        await self._refresh_coins()

        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(self._dc_refresh_sec)
                await self._refresh_coins()
            except asyncio.CancelledError:
                break

    async def _refresh_coins(self) -> None:
        """코인 선택: base_coins(무조건) + 변동성 상위 selective_pool_size(선택적).

        base_coins (XRP/SOL/TAO/DOGE/ADA): 필터 무관 항상 포함.
        selective pool: 변동성 상위 30 → safety filter → base_coins 제외 → 상위 20개.
        """
        try:
            held_coins = list({
                self.pos_manager._parse_key(k)[1]
                for k in self.pos_manager.positions
            })

            # ── 선택적 풀: 변동성 상위 30 → safety filter ──
            selective_results = await self.data_hub.select_tradeable_coins(
                volatility_pool=self._dc_volatility_pool,
                min_volume_usdt=self._dc_min_vol,
                max_spread_bps=self._dc_max_spread,
                held_coins=held_coins,
            )

            base_set = set(self._dc_base_coins)

            # base_coins 제외한 선택적 풀 (상위 selective_pool_size개)
            selective_coins = [
                r for r in selective_results
                if r["coin"] not in base_set
            ][:self._dc_selective_pool]

            # ── base_coins: 항상 포함 (거래소 가격 조회 가능 여부 무관) ──
            base_meta = [
                next(
                    (r for r in selective_results if r["coin"] == c),
                    {"coin": c, "volatility_pct": 0, "volume_usdt": 0,
                     "spread_bps": 0, "last": 0},
                )
                for c in self._dc_base_coins
            ]

            # 최종: base_coins 먼저, 선택적 풀 뒤에
            all_results = base_meta + selective_coins
            new_coins = [r["coin"] for r in all_results]
            # 중복 제거 (순서 유지)
            seen = set()
            new_coins = [c for c in new_coins if not (c in seen or seen.add(c))]
            all_results = [r for r in all_results if r["coin"] in set(new_coins)]

            self._coin_metadata = all_results
            old_coins = set(self.coins)
            self.coins = new_coins

            added = set(new_coins) - old_coins
            removed = old_coins - set(new_coins)

            log.info(
                f"[CoinRefresh] {len(new_coins)} coins active "
                f"(base={len(self._dc_base_coins)} + selective={len(selective_coins)}) | "
                f"+{added or '{}'} -{removed or '{}'}"
            )

            coin_lines = [f"📌 고정: `{', '.join(self._dc_base_coins)}`"]
            if selective_coins:
                sel_str = ", ".join(f"`{r['coin']}`({r['volatility_pct']:.1f}%)"
                                    for r in selective_coins[:10])
                coin_lines.append(f"📊 선택: {sel_str}")
            else:
                coin_lines.append("📊 선택: 없음 (필터 통과 없음)")

            if added or removed:
                discord_post(
                    "\n".join(coin_lines) +
                    f"\n\n추가: `{', '.join(added) if added else '-'}`\n"
                    f"제거: `{', '.join(removed) if removed else '-'}`",
                    title=f"🔄 Coin Scan — {len(new_coins)}개 활성",
                )
        except Exception as e:
            log.error(f"[CoinRefresh] Failed: {e}", exc_info=True)
            # 실패 시 base_coins만이라도 보장
            base_set = set(self._dc_base_coins)
            if not base_set.issubset(set(self.coins)):
                self.coins = list(self._dc_base_coins) + [
                    c for c in self.coins if c not in base_set
                ]

    async def _bar_counter(self) -> None:
        """Increment bars_held every 60s (1m resolution)."""
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(60)
                self.monitor.increment_bars()
            except asyncio.CancelledError:
                break

    async def _heartbeat(self) -> None:
        """Periodic status log."""
        interval = self.cfg.get("monitoring", {}).get("heartbeat_seconds", 60)
        while not self._shutdown_event.is_set():
            try:
                await asyncio.sleep(interval)
                status = self.portfolio_risk.status()
                pos_count = self.pos_manager.count()
                log.info(
                    f"[Heartbeat] equity=${status['equity']:.2f} | "
                    f"daily={status['daily_loss_pct']:.2%} | "
                    f"pos={pos_count} | "
                    f"notional=${status['notional']:.2f} | "
                    f"kill={status['kill_switch']}"
                )
            except asyncio.CancelledError:
                break

    async def _discord_hourly_briefing(self) -> None:
        """매 1시간마다 포지션 현황 + 잔고 브리핑."""
        await asyncio.sleep(60)  # 시작 후 1분 뒤 첫 실행
        while not self._shutdown_event.is_set():
            try:
                status = self.portfolio_risk.status()
                positions = self.pos_manager.all_positions()
                now = datetime.now(timezone.utc)

                # 포지션 목록
                if positions:
                    pos_lines = []
                    for p in positions:
                        try:
                            ticker = await self.data_hub.get_ticker(p.coin)
                            cur_price = ticker.get("last", p.entry_price)
                            if p.side == "BUY":
                                unreal_pct = (cur_price - p.entry_price) / p.entry_price
                            else:
                                unreal_pct = (p.entry_price - cur_price) / p.entry_price
                            unreal_usdt = unreal_pct * p.entry_price * p.current_qty
                            sign = "🟢" if unreal_pct >= 0 else "🔴"
                            pos_lines.append(
                                f"{sign} **{p.coin}** `{p.side}` [{p.strategy_tag}]\n"
                                f"  진입: `{p.entry_price:.4f}` | 현재: `{cur_price:.4f}`\n"
                                f"  미실현: `{unreal_usdt:+.2f} USDT` (`{unreal_pct:+.2%}`) | 바: `{p.bars_held}`"
                            )
                        except Exception:
                            pos_lines.append(f"• **{p.coin}** `{p.side}` [{p.strategy_tag}] | 진입: `{p.entry_price:.4f}`")
                    pos_text = "\n".join(pos_lines)
                else:
                    pos_text = "포지션 없음"

                # 전략별 오늘 PnL
                strat_pnl = status.get("strategy_pnl", {})
                strat_lines = "\n".join(
                    f"• {n}: `{v:+.4f} USDT`" for n, v in strat_pnl.items()
                ) or "• 거래 없음"

                # 오늘 세션 통계
                total_trades = len(self._paper_trades)
                session_pnl = sum(t["pnl_usdt"] for t in self._paper_trades)
                wins = sum(1 for t in self._paper_trades if t["pnl_usdt"] > 0)
                wr = f"{wins/total_trades*100:.1f}%" if total_trades > 0 else "N/A"

                discord_post(
                    f"**💰 잔고**\n"
                    f"자본: `${status['equity']:,.2f}` | 초기: `${self.initial_equity:,.2f}`\n"
                    f"일일 손익: `{status['daily_loss_pct']:+.2%}` | 노셔널: `${status['notional']:.2f}`\n\n"
                    f"**📊 오픈 포지션 ({len(positions)}개)**\n{pos_text}\n\n"
                    f"**📈 세션 통계** (거래: {total_trades}건)\n"
                    f"세션 PnL: `{session_pnl:+.4f} USDT` | 승률: `{wr}`\n\n"
                    f"**전략별 PnL:**\n{strat_lines}",
                    title=f"🕐 1시간 브리핑 — {now.strftime('%H:%M UTC')}",
                )
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.error(f"[Discord] hourly briefing error: {e}")

            try:
                await asyncio.sleep(3600)  # 1시간
            except asyncio.CancelledError:
                break

    async def _discord_signal_scan(self) -> None:
        """매 10분마다 변동성 상위 30개 전체 신호 분석 리포트 출력.

        self.coins(safety filter 통과 목록)가 아닌, 변동성 상위 30개 전체를
        필터 없이 스캔하여 진입 신호 강도를 분석한다.
        """
        await asyncio.sleep(60)  # 시작 후 1분 뒤 첫 실행
        while not self._shutdown_event.is_set():
            try:
                now = datetime.now(timezone.utc)

                # ── 변동성 상위 30 전체 (safety filter 제거) ──
                universe = await self.data_hub.select_tradeable_coins(
                    volatility_pool=30,
                    min_volume_usdt=0,       # 필터 없음 — 전체 스캔
                    max_spread_bps=9999.0,   # 필터 없음
                )
                if not universe:
                    # testnet에서 fetch 실패 시 현재 coin 리스트 fallback
                    universe = [{"coin": c, "volatility_pct": 0,
                                 "volume_usdt": 0, "spread_bps": 0, "last": 0}
                                for c in self.coins]

                active_set = set(self.coins)  # safety filter 통과한 코인
                open_coins = {
                    self.pos_manager._parse_key(k)[1]
                    for k in self.pos_manager.positions
                }

                scored = []
                for meta in universe:
                    coin = meta["coin"]
                    try:
                        df = await self.data_hub.get_ohlcv(coin, "1m", limit=500)
                        if df is None or len(df) < 100:
                            continue

                        price = df["close"].iloc[-1]

                        # CVD z-score (단순 rolling std 기반)
                        cvd = DataHub.compute_cvd(df)
                        cvd_pct = float(cvd.rank(pct=True).iloc[-1])
                        cvd_val = float(cvd.iloc[-1])
                        cvd_mean = float(cvd.rolling(480).mean().iloc[-1])
                        cvd_std = float(cvd.rolling(480).std().iloc[-1])
                        cvd_z = (cvd_val - cvd_mean) / cvd_std if cvd_std > 0 else 0

                        # OFI
                        ofi = DataHub.compute_ofi(df)
                        ofi_pct = float(ofi.rank(pct=True).iloc[-1])

                        # ATR%
                        atr = (df["high"] - df["low"]).rolling(14).mean().iloc[-1]
                        atr_pct = atr / price * 100 if price > 0 else 0

                        # 추세
                        ret_20 = (price - df["close"].iloc[-20]) / df["close"].iloc[-20]
                        ret_5 = (price - df["close"].iloc[-5]) / df["close"].iloc[-5]

                        # 신호 판정 (CVD 극단 = 역추세 전략 진입 방향 기준)
                        # CVD 극단 매수(>0.95) → 소진 예상 → 역추세 SHORT 후보
                        # CVD 극단 매도(<0.05) → 소진 예상 → 역추세 LONG 후보
                        direction = "─"
                        score = 0
                        if cvd_pct > 0.95 or cvd_z > 2.0:
                            direction = "역추세SHORT 🔴"
                            score += 2 + (1 if cvd_z > 3.0 else 0)
                        elif cvd_pct < 0.05 or cvd_z < -2.0:
                            direction = "역추세LONG 🟢"
                            score += 2 + (1 if cvd_z < -3.0 else 0)
                        if ofi_pct > 0.90:
                            score += 1
                        elif ofi_pct < 0.10:
                            score += 1
                        if atr_pct > 0.5:
                            score += 1  # 충분한 변동성

                        status = "🔵 OPEN" if coin in open_coins else (
                                 "✅ 활성" if coin in active_set else "⚪ 대기")

                        scored.append({
                            "coin": coin,
                            "direction": direction,
                            "score": score,
                            "cvd_pct": cvd_pct,
                            "cvd_z": cvd_z,
                            "ofi_pct": ofi_pct,
                            "atr_pct": atr_pct,
                            "ret_5": ret_5,
                            "ret_20": ret_20,
                            "price": price,
                            "vol_m": meta["volume_usdt"] / 1e6,
                            "status": status,
                        })
                    except Exception:
                        continue

                scored.sort(key=lambda x: x["score"], reverse=True)

                # ── 리포트 구성 ──
                strong = [s for s in scored if s["score"] >= 2]
                neutral = [s for s in scored if s["score"] < 2]

                lines = [f"**전체 스캔: {len(scored)}개** | 신호: {len(strong)}개 | 중립: {len(neutral)}개"]
                lines.append(f"활성 코인(safety 통과): `{', '.join(active_set) if active_set else '없음'}`")
                lines.append("")

                if strong:
                    lines.append("**━ 진입 후보 ━**")
                    for s in strong[:10]:
                        lines.append(
                            f"{s['status']} **{s['coin']}** {s['direction']} (점{s['score']})\n"
                            f"  `{s['price']:.4f}` | CVD-z: `{s['cvd_z']:+.1f}` | OFI: `{s['ofi_pct']:.2f}` "
                            f"| ATR: `{s['atr_pct']:.2f}%` | 5봉: `{s['ret_5']:+.2%}`"
                        )
                else:
                    lines.append("현재 진입 조건 충족 후보 없음 — 시장 중립 구간")

                if neutral and len(neutral) <= 20:
                    lines.append("")
                    lines.append(f"**━ 중립 ({len(neutral)}개) ━**")
                    neutral_summary = ", ".join(
                        f"`{s['coin']}`({s['atr_pct']:.1f}%)" for s in neutral[:15]
                    )
                    lines.append(neutral_summary)

                # ── 보유 포지션 요약 + 예상 청산 타이밍 ──
                positions = self.pos_manager.positions
                if positions:
                    lines.append("")
                    lines.append("**━ 보유 포지션 ━**")
                    for key, pos in positions.items():
                        try:
                            # 현재 가격
                            ticker = await self.data_hub.get_ticker(pos.coin)
                            cur_price = ticker.get("last", pos.entry_price)

                            # 미실현 PnL
                            if pos.side == "BUY":
                                unreal_pct = (cur_price - pos.entry_price) / pos.entry_price
                            else:
                                unreal_pct = (pos.entry_price - cur_price) / pos.entry_price
                            unreal_usdt = unreal_pct * pos.current_qty * pos.entry_price

                            # 예상 청산 타이밍 (남은 TTL bars → 분 환산)
                            bars_remaining = max(0, pos.ttl_bars - pos.bars_held)
                            eta_min = bars_remaining  # 1 bar = 1분
                            eta_str = f"{eta_min//60}h{eta_min%60}m" if eta_min >= 60 else f"{eta_min}m"

                            # SL 거리
                            sl_dist_pct = abs(cur_price - pos.sl_price) / cur_price * 100

                            side_emoji = "📈" if pos.side == "BUY" else "📉"
                            pnl_emoji = "🟢" if unreal_usdt >= 0 else "🔴"
                            lines.append(
                                f"{side_emoji} **{pos.coin}** {pos.side} `@{pos.entry_price:.4f}` "
                                f"→ 현재 `{cur_price:.4f}`\n"
                                f"  {pnl_emoji} 미실현: `{unreal_usdt:+.2f}$` (`{unreal_pct:+.2%}`) | "
                                f"SL까지: `{sl_dist_pct:.2f}%` | "
                                f"TTL잔여: `{eta_str}` ({bars_remaining}봉)"
                            )
                        except Exception:
                            lines.append(f"• {pos.coin} {pos.side} @{pos.entry_price:.4f}")

                    status = self.portfolio_risk.status()
                    lines.append(
                        f"\n💼 총 {len(positions)}포지션 | 노셔널 `${status['notional']:.0f}` | "
                        f"잔고 `${status['equity']:.2f}` | 일일 `{status['daily_loss_pct']:+.2%}`"
                    )

                discord_post(
                    "\n".join(lines),
                    title=f"📡 10분 전체 신호 스캔 — {now.strftime('%H:%M UTC')} | 상위30 분석",
                )
                log.info(f"[SignalScan] {len(scored)} coins | {len(strong)} candidates")

            except asyncio.CancelledError:
                break
            except Exception as e:
                log.error(f"[Discord] signal scan error: {e}", exc_info=True)

            try:
                await asyncio.sleep(600)  # 10분
            except asyncio.CancelledError:
                break

    async def _on_position_close(
        self, strategy: str, coin: str, reason: str, price: float
    ) -> None:
        """Callback when monitor detects SL/TP/TTL hit."""
        pos = self.pos_manager.get_position(strategy, coin)
        if pos is None:
            return

        # Calculate PnL
        if pos.side == "BUY":
            pnl_pct = (price - pos.entry_price) / pos.entry_price
        else:
            pnl_pct = (pos.entry_price - price) / pos.entry_price

        pnl_usdt = pnl_pct * pos.entry_price * pos.current_qty

        # Close on exchange (live + demo mode: real orders on exchange/testnet)
        _exchange_close_ok = True
        if self.mode in ("live", "demo"):
            try:
                close_side = "SELL" if pos.side == "BUY" else "BUY"
                await self.exchange.market_close(
                    coin, close_side, pos.current_qty,
                    order_link_id=self.exchange.make_order_id(coin, close_side, prefix="v8cl"),
                )
            except Exception as e:
                err_str = str(e)
                # Exchange SL/TP already closed the position — not an error, just a race condition
                _already_closed = any(code in err_str for code in (
                    "-2022", "-4061", "ReduceOnly", "reduceOnly",
                    "position side does not match", "No position",
                ))
                if _already_closed:
                    log.info(f"[Close] {coin} already closed by exchange SL/TP (race condition — OK)")
                else:
                    _exchange_close_ok = False
                    log.error(f"[Close] {coin} exchange close failed: {e}")
                    # Notify Discord about manual intervention needed (orphan position risk)
                    discord_post(
                        f"⚠️ **거래소 청산 실패** — 수동 확인 필요\n"
                        f"종목: `{coin}` | 수량: `{pos.current_qty}` | 사유: `{e}`\n"
                        f"봇 내부 포지션은 제거됩니다. 거래소에서 직접 확인하세요.",
                        title="⚠️ 거래소 청산 실패",
                    )

        # Record PnL
        self.portfolio_risk.record_trade_pnl(strategy, pnl_usdt)

        # Record full trade context for per-coin optimization
        if self.trade_logger:
            try:
                self.trade_logger.record_close(
                    strategy=strategy,
                    coin=coin,
                    exit_price=price,
                    exit_reason=reason,
                    pos=pos,
                )
            except Exception as e:
                log.warning(f"[TradeLog] record_close failed: {e}")

        # Update per-coin adaptive profiles
        if self.coin_profiles:
            try:
                self.coin_profiles.update_from_trade({
                    "coin": coin,
                    "pnl_pct": pnl_pct,
                    "pnl_usdt": pnl_usdt,
                    "mfe_pct": pos.mfe_pct,
                    "mae_pct": pos.mae_pct,
                    "bars_held": pos.bars_held,
                    "exit_reason": reason,
                })
            except Exception as e:
                log.warning(f"[CoinProfile] update failed: {e}")

        # Remove position
        self.pos_manager.remove_position(strategy, coin)

        # Log
        emoji = "✅" if pnl_usdt > 0 else "❌"
        log.info(
            f"[Close] {emoji} {strategy}:{coin} {reason} | "
            f"PnL={pnl_usdt:+.4f} ({pnl_pct:+.2%}) | "
            f"entry={pos.entry_price:.4f} exit={price:.4f} | "
            f"bars={pos.bars_held} | "
            f"mfe={pos.mfe_pct:.2%} mae={pos.mae_pct:.2%}"
        )

        # Discord: 거래 완료 + 잔고 + 거래내역 요약
        status = self.portfolio_risk.status()
        # [Fix] trade를 먼저 append해야 최근 거래내역에 현재 거래가 포함됨
        trade = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "strategy": strategy,
            "coin": coin,
            "side": pos.side,
            "entry": pos.entry_price,
            "exit": price,
            "pnl_usdt": round(pnl_usdt, 6),
            "pnl_pct": round(pnl_pct, 6),
            "reason": reason,
            "bars": pos.bars_held,
            "mfe": round(pos.mfe_pct, 6),
            "mae": round(pos.mae_pct, 6),
        }
        self._paper_trades.append(trade)
        # Cap in-memory list to prevent unbounded growth over long sessions
        if len(self._paper_trades) > 1000:
            self._paper_trades = self._paper_trades[-1000:]
        self._save_trade(trade)

        # Discord: 거래 완료 + 잔고 + 거래내역 요약
        total_trades = len(self._paper_trades)
        session_pnl = sum(t["pnl_usdt"] for t in self._paper_trades)
        wins = sum(1 for t in self._paper_trades if t["pnl_usdt"] > 0)
        wr = f"{wins/total_trades*100:.1f}%" if total_trades > 0 else "N/A"

        # 최근 5건 (현재 거래 포함)
        recent = self._paper_trades[-5:]
        recent_lines = []
        for t in reversed(recent):
            e = "✅" if t["pnl_usdt"] > 0 else "❌"
            recent_lines.append(f"{e} **{t['coin']}** `{t['side']}` `{t['pnl_usdt']:+.4f}` ({t['pnl_pct']:+.2%}) `{t['reason']}`")
        recent_text = "\n".join(recent_lines)

        emoji_trade = "✅" if pnl_usdt > 0 else "❌"
        reason_label = {"SL_HIT": "🛑 손절", "TP_HIT": "🎯 익절", "TTL_HIT": "⏰ 시간초과", "TIME_STOP": "⏰ 시간초과", "KILL_SWITCH": "🚨 킬스위치"}.get(reason, f"📌 {reason}")

        try:
            portfolio_snap = await self._build_portfolio_snapshot()
        except Exception:
            portfolio_snap = f"자본: `${status['equity']:,.2f}` | 노셔널: `${status['notional']:.2f}`"

        discord_post(
            f"**{emoji_trade} 청산 결과**\n"
            f"종목: `{coin}` `{pos.side}` | [{strategy}]\n"
            f"진입: `{pos.entry_price:.4f}` → 청산: `{price:.4f}`\n"
            f"**PnL: `{pnl_usdt:+.4f} USDT` (`{pnl_pct:+.2%}`)**\n"
            f"사유: {reason_label} | 보유: `{pos.bars_held}봉`\n"
            f"MFE: `{pos.mfe_pct:.2%}` | MAE: `{pos.mae_pct:.2%}`\n\n"
            f"{portfolio_snap}\n\n"
            f"**📋 세션** ({total_trades}건 | 승률 {wr} | PnL `{session_pnl:+.4f}`)\n"
            f"{recent_text}",
            title=f"{'✅ 익절' if pnl_usdt > 0 else '❌ 손절/청산'} — {coin} {reason_label}",
        )
        if not self._first_trade_notified:
            self._first_trade_notified = True
            discord_post(
                f"🚀 **시스템 첫 거래 완료 — 정상 작동 확인**\n"
                f"현재 자본: `${status['equity']:.2f}` | 일일: `{status['daily_loss_pct']:+.2%}`\n"
                f"오픈 포지션: `{self.pos_manager.count()}개`",
                title="🎯 거래 시작 확인",
            )

    async def _build_portfolio_snapshot(self, event_coin: str = "", event_side: str = "") -> str:
        """모든 오픈 포지션의 현재 손익을 실시간으로 조회해 Discord 임베드 텍스트로 반환."""
        positions = self.pos_manager.all_positions()
        status = self.portfolio_risk.status()
        total_unrealized = 0.0
        lines = []

        for p in positions:
            try:
                ticker = await self.data_hub.get_ticker(p.coin)
                cur = ticker.get("last", p.entry_price)
            except Exception:
                cur = p.entry_price

            if p.side == "BUY":
                unreal_pct = (cur - p.entry_price) / p.entry_price if p.entry_price > 0 else 0
            else:
                unreal_pct = (p.entry_price - cur) / p.entry_price if p.entry_price > 0 else 0

            notional = p.entry_price * p.current_qty
            unreal_usdt = unreal_pct * notional
            total_unrealized += unreal_usdt

            mark = "👉 " if p.coin == event_coin else "  "
            sign = "🟢" if unreal_pct >= 0 else "🔴"
            arrow = "↑" if p.side == "BUY" else "↓"
            lines.append(
                f"{mark}{sign} `{p.coin}` {arrow}`{p.side}` [{p.strategy_tag}]\n"
                f"    진입`{p.entry_price:.4f}` → 현재`{cur:.4f}` | "
                f"`{unreal_usdt:+.2f}` (`{unreal_pct:+.1%}`) | "
                f"SL`{p.sl_price:.4f}`"
            )

        pos_section = "\n".join(lines) if lines else "포지션 없음"
        total_sign = "🟢" if total_unrealized >= 0 else "🔴"
        session_pnl = sum(t["pnl_usdt"] for t in self._paper_trades)
        total_trades = len(self._paper_trades)
        wins = sum(1 for t in self._paper_trades if t["pnl_usdt"] > 0)
        wr = f"{wins/total_trades*100:.0f}%" if total_trades > 0 else "-"

        return (
            f"**📊 포지션 현황 ({len(positions)}개)**\n{pos_section}\n\n"
            f"**💰 계좌**\n"
            f"자본: `${status['equity']:,.2f}` | 노셔널: `${status['notional']:.2f}`\n"
            f"{total_sign} 미실현손익: `{total_unrealized:+.2f} USDT` | "
            f"일일: `{status['daily_loss_pct']:+.1%}`\n"
            f"세션: `{total_trades}건` WR`{wr}` 실현PnL`{session_pnl:+.2f}`"
        )

    @staticmethod
    def _load_equity_state() -> float:
        """Load persisted equity from disk. Falls back to YAML initial_equity."""
        try:
            if EQUITY_STATE_FILE.exists():
                with open(EQUITY_STATE_FILE, "r") as f:
                    data = json.load(f)
                equity = float(data.get("current_equity", 0))
                if equity > 0:
                    log.info(f"[EquityState] Restored equity=${equity:.2f} (restarts={data.get('restarts', 0)+1})")
                    return equity
        except Exception as e:
            log.warning(f"[EquityState] Load failed: {e}")
        # Read from YAML as fallback
        try:
            with open("config/multi_strategy.yaml", "r") as f:
                cfg = yaml.safe_load(f)
            return float(cfg.get("initial_equity", 5000.0))
        except Exception:
            return 5000.0

    def _save_equity_state(self) -> None:
        """Persist current equity to disk after every trade close."""
        try:
            current = self.portfolio_risk.current_equity if self.portfolio_risk else self.initial_equity
            existing = {}
            if EQUITY_STATE_FILE.exists():
                with open(EQUITY_STATE_FILE, "r") as f:
                    existing = json.load(f)
            existing.update({
                "current_equity": round(current, 6),
                "initial_equity": round(self.initial_equity, 6),
                "restarts": existing.get("restarts", 0) + (1 if not existing else 0),
                "total_trades": len(self._paper_trades),
                "total_pnl": round(sum(t.get("pnl_usdt", 0) for t in self._paper_trades), 6),
                "last_updated": datetime.now(timezone.utc).isoformat(),
            })
            with open(EQUITY_STATE_FILE, "w") as f:
                json.dump(existing, f, indent=2)
        except Exception as e:
            log.warning(f"[EquityState] Save failed: {e}")

    @staticmethod
    def _load_trades_history() -> list:
        """Load historical trades from JSONL. Keeps last 1000 in memory, counts all for stats."""
        if not TRADES_FILE.exists():
            return []
        try:
            trades = []
            with open(TRADES_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        trades.append(json.loads(line))
            log.info(f"[TradeHistory] Loaded {len(trades)} total trades from history")
            return trades[-1000:]  # keep last 1000 in memory
        except Exception as e:
            log.warning(f"[TradeHistory] Load failed: {e}")
            return []

    def _save_trade(self, trade: dict) -> None:
        """Append trade to JSONL file."""
        with open(TRADES_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(trade, default=str) + "\n")
        self._save_equity_state()

    async def _strategy_analysis_loop(self) -> None:
        """6시간마다 전략 성과 분석 + Discord 보고."""
        await asyncio.sleep(3600 * 2)  # 첫 분석: 2시간 후
        while not self._shutdown_event.is_set():
            try:
                analyzer = StrategyAnalyzer(STATE_DIR)
                loop = asyncio.get_running_loop()
                report = await loop.run_in_executor(None, analyzer.run)
                if report.total_trades > 0:
                    summary = analyzer.discord_summary(report)
                    discord_post(summary, title="📊 전략 분석 리포트 (6h)")
                    log.info(f"[Analysis] Report generated: {report.total_trades} trades")
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.error(f"[Analysis] Error: {e}", exc_info=True)

            try:
                await asyncio.sleep(3600 * 6)  # 이후 6시간마다
            except asyncio.CancelledError:
                break

    async def shutdown(self) -> None:
        """Graceful shutdown."""
        log.info("Shutting down...")
        # Persist equity state before exiting
        self._save_equity_state()

        if self.monitor:
            await self.monitor.stop()

        # Close all positions (live / demo mode) — unless --keep-positions specified
        # Paper mode: just clear positions.json to avoid stale ghost positions on next restart
        if self.mode == "paper" and not self._keep_positions and self.pos_manager:
            self.pos_manager.positions.clear()
            self.pos_manager._save()

        if self.mode in ("live", "demo") and not self._keep_positions:
            for pos in self.pos_manager.all_positions():
                try:
                    close_side = "SELL" if pos.side == "BUY" else "BUY"
                    await self.exchange.market_close(
                        pos.coin, close_side, pos.current_qty,
                        order_link_id=self.exchange.make_order_id(pos.coin, close_side, prefix="v8sd"),
                    )
                except Exception as e:
                    log.error(f"Shutdown close {pos.coin} failed: {e}")
        elif self._keep_positions and self.pos_manager.count() > 0:
            # Save positions for next restart (positions.json already auto-saved)
            saved_count = self.pos_manager.count()
            pos_summary = ", ".join(
                f"{p.coin}({p.side}@{p.entry_price:.3f})"
                for p in self.pos_manager.all_positions()
            )
            log.info(f"[Shutdown] --keep-positions: {saved_count} positions saved for restart: {pos_summary}")
            discord_post(
                f"**포지션 {saved_count}개 저장됨 (재시작 후 계속 모니터링)**\n{pos_summary}\n\n"
                f"`--keep-positions` 플래그로 종료. 다음 재시작 시 자동 복구됩니다.",
                title="⏸ 봇 일시정지 — 포지션 유지",
            )

        # Summary
        total_trades = len(self._paper_trades)
        total_pnl = sum(t["pnl_usdt"] for t in self._paper_trades)
        wins = sum(1 for t in self._paper_trades if t["pnl_usdt"] > 0)
        wr = wins / total_trades * 100 if total_trades > 0 else 0

        log.info(f"{'='*60}")
        log.info(f"Session summary:")
        log.info(f"  Trades: {total_trades}")
        log.info(f"  PnL: ${total_pnl:+.4f}")
        log.info(f"  Win rate: {wr:.1f}%")
        log.info(f"  Final equity: ${self.portfolio_risk.current_equity:.2f}")

        for name in self.strategies:
            strat_trades = [t for t in self._paper_trades if t["strategy"] == name]
            strat_pnl = sum(t["pnl_usdt"] for t in strat_trades)
            log.info(f"  [{name}] trades={len(strat_trades)} pnl=${strat_pnl:+.4f}")
        log.info(f"{'='*60}")

        # Shutdown executor
        self._executor.shutdown(wait=False)

        # Close exchange
        if self.exchange and hasattr(self.exchange, "async_exchange"):
            try:
                await self.exchange.async_exchange.close()
            except Exception:
                pass


def main():
    parser = argparse.ArgumentParser(description="Multi-Strategy Trading Bot v8.0")
    parser.add_argument(
        "--config",
        default="config/multi_strategy.yaml",
        help="Config file path",
    )
    parser.add_argument(
        "--mode",
        choices=["paper", "demo", "live"],
        default=None,
        help="Override mode (paper/demo/live)",
    )
    parser.add_argument(
        "--keep-positions",
        action="store_true",
        default=False,
        help="On shutdown, do NOT close exchange positions — save to JSON for next restart",
    )
    args = parser.parse_args()

    bot = MultiStrategyBot(args.config, args.mode)
    bot._keep_positions = args.keep_positions

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)

    # Handle SIGINT/SIGTERM — thread-safe asyncio event set
    def handle_signal(sig, frame):
        log.info(f"Signal {sig} received, shutting down...")
        loop.call_soon_threadsafe(bot._shutdown_event.set)

    # Handle SIGHUP — hot-reload config without restart
    def handle_sighup(sig, frame):
        log.info("[SIGHUP] Hot-reload config requested...")
        loop.call_soon_threadsafe(_reload_config)

    def _reload_config():
        try:
            with open(args.config, "r", encoding="utf-8") as f:
                new_cfg = yaml.safe_load(f)
            bot.cfg = new_cfg
            # Update portfolio risk parameters
            if bot.portfolio_risk:
                port_cfg = new_cfg.get("portfolio", {})
                bot.portfolio_risk.config.total_exposure_pct = port_cfg.get("total_exposure_pct", 0.85)
                bot.portfolio_risk.config.same_direction_max = port_cfg.get("same_direction_max", 5)
                bot.portfolio_risk.config.daily_loss_pct = port_cfg.get("daily_loss_pct", 0.10)
                bot.portfolio_risk.config.strategy_loss_pct = port_cfg.get("strategy_loss_pct", 0.20)
            # Update strategy extra configs
            strat_cfgs = new_cfg.get("strategies", {})
            for name, strat in bot.strategies.items():
                if name in strat_cfgs and strat_cfgs[name].get("extra"):
                    strat.config.extra = strat_cfgs[name]["extra"]
            log.info("[SIGHUP] Config reloaded successfully — no restart needed")
            discord_post("⚙️ Config hot-reloaded (SIGHUP) — 재시작 없음")
        except Exception as e:
            log.error(f"[SIGHUP] Config reload failed: {e}")

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGHUP, handle_sighup)

    try:
        loop.run_until_complete(bot.start())
    except KeyboardInterrupt:
        log.info("KeyboardInterrupt, shutting down...")
        bot._shutdown_event.set()
        loop.run_until_complete(bot.shutdown())
    finally:
        loop.close()


if __name__ == "__main__":
    main()
