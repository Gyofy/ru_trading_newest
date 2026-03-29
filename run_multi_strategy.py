"""v8.5 Multi-Strategy Trading Bot — 4 strategies, asymmetric-first design.

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
import logging.handlers
import os
import signal
import sys
import threading
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import yaml
from dotenv import load_dotenv
load_dotenv()

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
        logging.handlers.RotatingFileHandler(
            STATE_DIR / "bot.log",
            maxBytes=10 * 1024 * 1024,  # 10MB per file
            backupCount=5,              # bot.log + bot.log.1~5 = 최대 60MB
            encoding="utf-8",
        ),
        _DiscordAlertHandler(),
    ],
)
log = logging.getLogger("multi_bot")


# ── Imports ──────────────────────────────────────────────

from src.execution.exchange_adapter import ExchangeAdapter
from src.execution.order_ledger import OrderLedger
from src.strategies.base import StrategyConfig, StrategyBase, ROUND_TRIP_FEE_RATE
from src.strategies.cvd_spike import CVDSpikeReactor
from src.strategies.liquidation_fade import LiquidationFade
from src.strategies.momentum_breakout import MomentumBreakout
from src.strategies.asymmetric_sniper import AsymmetricSniper
from src.strategies.cvd_extreme import CVDExtreme
from src.strategies.vwap_reversion import VWAPReversion
from src.strategies.funding_arb import FundingArb
from src.strategies.volume_impulse import VolumeImpulse
from src.strategies.oi_divergence import OIDivergence
from src.strategies.multi_position_manager import MultiPositionManager
from src.strategies.portfolio_risk import PortfolioRiskConfig, PortfolioRiskManager
from src.strategies.data_hub import DataHub
from src.strategies.sl_tp_monitor_v2 import SlTpMonitorV2
from src.strategies.trade_logger import TradeLogger
from src.strategies.coin_profile import CoinProfileStore
from src.strategies.position_sizer import PositionSizer
from src.strategies.strategy_analyzer import StrategyAnalyzer
from src.strategies.ev_guardian import EVGuardian
from src.strategies.entry_filters import EntryFilters
from src.utils.bnb_keeper import BnbKeeper
from src.utils.fee_scanner import FeeScanner

STRATEGY_MAP = {
    "cvd_spike": CVDSpikeReactor,
    "liquidation_fade": LiquidationFade,
    "momentum_breakout": MomentumBreakout,
    "asymmetric_sniper": AsymmetricSniper,
    "cvd_extreme": CVDExtreme,
    "vwap_reversion": VWAPReversion,
    "funding_arb": FundingArb,
    "volume_impulse": VolumeImpulse,
    "oi_divergence": OIDivergence,
}


# ── Discord ──────────────────────────────────────────────

_discord_rate_lock = threading.Lock()
_discord_last_ts: float = 0.0
_DISCORD_MIN_INTERVAL = 2.0  # Discord rate limit: max 30/min → 1 per 2s minimum


def _discord_post_sync(message: str, title: str) -> None:
    """동기 Discord 전송 (executor에서 호출용). Rate-limited: 1 per 2s."""
    global _discord_last_ts
    with _discord_rate_lock:
        elapsed = time.monotonic() - _discord_last_ts
        if elapsed < _DISCORD_MIN_INTERVAL:
            time.sleep(_DISCORD_MIN_INTERVAL - elapsed)
        _discord_last_ts = time.monotonic()
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


def discord_post(message: str, title: str = "") -> None:
    """Discord 전송 — asyncio 이벤트 루프 비블로킹. Rate-limited to 1 per 2s."""
    url = os.environ.get("DISCORD_WEBHOOK_URL", "")
    if not url:
        log.warning("[Discord] DISCORD_WEBHOOK_URL not set")
        return
    try:
        loop = asyncio.get_running_loop()
        # ensure_future accepts Future (run_in_executor returns Future, not coroutine)
        asyncio.ensure_future(
            loop.run_in_executor(None, _discord_post_sync, message, title)
        )
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
        # Load baseline cumulative counters (for _save_equity_state accumulation)
        self._baseline_total_trades, self._baseline_total_pnl = self._load_baseline_counters()
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
        self._dc_exclude_coins: set[str] = set(dcfg.get("exclude_coins", []))

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
        self.ev_guardian: EVGuardian | None = None
        self.entry_filters: EntryFilters | None = None
        self.bnb_keeper: BnbKeeper | None = None
        self.fee_scanner: FeeScanner | None = None
        self.strategies: dict[str, object] = {}

        # Per-coin cooldown after exit (prevent immediate re-entry)
        self._coin_cooldowns: dict[str, float] = {}  # coin -> unix timestamp of last exit
        self._coin_cooldown_sec = 900  # 15 minutes

        # Bot-wide daily trade cap (prevents overtrading across all strategies)
        self._bot_daily_trade_count: int = 0
        self._bot_daily_trade_date: str = ""
        self._bot_max_daily_trades: int = 100  # demo: 6전략 × ~15건 = ~90건/일 여유

        # Paper mode tracking — session trades start empty (current run only)
        self._paper_equity = self.initial_equity
        self._session_trades: list[dict] = []  # current session only — NOT loaded from history
        self._session_start_time: float = time.time()  # for filtering in briefings
        self._first_trade_notified = False

        # Graceful update: --keep-positions prevents closing positions on shutdown
        self._keep_positions: bool = False

        # Process pool for CPU-bound signal computation (50% of cores)
        mp_cfg = self.cfg.get("multiprocessing", {})
        import os as _os
        _cpu_count = _os.cpu_count() or 8
        _default_workers = max(1, int(_cpu_count * 0.8))  # 80% of CPU cores
        _workers = mp_cfg.get("workers", _default_workers)
        self._executor = concurrent.futures.ProcessPoolExecutor(max_workers=_workers)
        log.info(f"ProcessPoolExecutor initialized: {_workers} workers ({_cpu_count} cores, 80% target)")

    async def start(self) -> None:
        """Initialize all components and start trading."""
        log.info(f"{'='*60}")
        log.info(f"Multi-Strategy Bot v8.5 | mode={self.mode}")
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
            # Demo 모드: 실제 거래소 잔고를 가져와 내부 equity와 동기화
            # exchange_adapter.fetch_balance()는 {"total": X, "free": X, "used": X} 반환
            try:
                _real_bal = await self.exchange.fetch_balance()
                _real_usdt = float(_real_bal.get("total", 0) or 0)
                if _real_usdt > 0:
                    _delta = abs(_real_usdt - self.initial_equity)
                    if _delta > 10:  # $10 이상 차이 시 동기화
                        log.warning(
                            f"[BalanceSync] 내부equity=${self.initial_equity:.2f} vs 거래소USDT=${_real_usdt:.2f} "
                            f"(차이=${_delta:.2f}) → 거래소 잔고로 동기화"
                        )
                        self.initial_equity = _real_usdt
                    else:
                        log.info(f"[BalanceSync] 잔고 일치 — equity=${_real_usdt:.2f}")
            except Exception as _e:
                log.warning(f"[BalanceSync] 잔고 동기화 실패: {_e}")
            # Ghost 포지션은 pos_manager 초기화 후에 정리 (아래 _ghost_positions_raw에 저장)
            _ghost_positions_raw = []
            try:
                _exch_positions = await self.exchange._exchange.fetch_positions()
                _ghost_positions_raw = [
                    p for p in _exch_positions
                    if abs(float(p.get("contracts", 0) or 0)) > 0
                ]
            except Exception as _e:
                log.warning(f"[GhostClean] ghost 포지션 사전 수집 실패: {_e}")
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

        # Ghost 포지션 정리: pos_manager 로드 후 교차 검증
        if hasattr(self, "_ghost_positions_raw") and self._ghost_positions_raw:
            _ghost = [
                p for p in self._ghost_positions_raw
                if not self.pos_manager.has_coin_any_strategy(p.get("symbol", "").split("/")[0])
            ]
            if _ghost:
                log.warning(f"[GhostClean] ghost 포지션 {len(_ghost)}개 — 자동 청산")
                _ghost_lines = []
                for _gp in _ghost:
                    _gsym = _gp["symbol"]
                    _gside = _gp["side"]
                    _gqty = abs(float(_gp.get("contracts", 0)))
                    _gmargin = round(float(_gp.get("initialMargin", 0) or 0), 2)
                    _gclose = "sell" if _gside == "long" else "buy"
                    try:
                        await self.exchange._exchange.create_order(
                            _gsym, "market", _gclose, _gqty, params={"reduceOnly": True}
                        )
                        log.info(f"[GhostClean] {_gsym} {_gside} qty={_gqty} margin=${_gmargin} ✅")
                        _ghost_lines.append(f"• `{_gsym}` {_gside} margin=`${_gmargin}` ✅")
                    except Exception as _ge:
                        log.error(f"[GhostClean] {_gsym} 청산 실패: {_ge}")
                        _ghost_lines.append(f"• `{_gsym}` {_gside} ❌")
                discord_post(
                    f"**{len(_ghost)}개 ghost 포지션 자동 청산**\n" + "\n".join(_ghost_lines),
                    title="🧹 Ghost 포지션 정리",
                )
            else:
                log.info("[GhostClean] ghost 포지션 없음 — 거래소/로컬 일치")

        self.data_hub = DataHub(self.exchange, executor=self._executor)
        self.trade_logger = TradeLogger(STATE_DIR)
        ev_cfg = self.cfg.get("ev_guardian", {})
        self.ev_guardian = EVGuardian(
            jsonl_path=STATE_DIR / "trade_context.jsonl",
            initial_equity=self.initial_equity,
            fee_budget_pct=ev_cfg.get("fee_budget_pct", 0.005),
            report_path=STATE_DIR / "ev_report.json",
            ev_threshold=ev_cfg.get("ev_threshold", None),
            ev_min_sample=ev_cfg.get("ev_min_sample", None),
        )
        # 시작 시 ev_reset.flag 파일 존재하면 EV 리셋 후 파일 삭제
        _ev_reset_flag = STATE_DIR / "ev_reset.flag"
        if _ev_reset_flag.exists():
            self.ev_guardian.reset()
            _ev_reset_flag.unlink()
            log.info("[EVGuardian] ev_reset.flag detected — EV stats reset, fresh start")
        # 시작 시 기존 데이터로 즉시 1회 평가
        _ev_init = self.ev_guardian.evaluate()
        if _ev_init:
            log.info(f"[EVGuardian] Initial evaluation: {list(_ev_init.keys())}")
            self.ev_guardian.save_report()
        self.coin_profiles = CoinProfileStore(
            persist_path=STATE_DIR / "coin_profiles.json",
            trade_context_path=STATE_DIR / "trade_context.jsonl",
        )
        sizing_cfg = self.cfg.get("position_sizing", {})
        self.position_sizer = PositionSizer(sizing_cfg)

        # Entry filters (rule-based reject gates)
        ef_cfg = self.cfg.get("entry_filters", {})
        self.entry_filters = EntryFilters(ef_cfg)
        log.info(
            f"[EntryFilters] vpin={ef_cfg.get('vpin_filter_enabled', True)} "
            f"blacklist={ef_cfg.get('blacklist_enabled', True)} "
            f"atr_regime={ef_cfg.get('atr_regime_enabled', True)}"
        )

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
        # Restore daily state (strategy_pnl, day_start_equity) if same UTC day
        try:
            if EQUITY_STATE_FILE.exists():
                with open(EQUITY_STATE_FILE, "r") as _f:
                    _es = json.load(_f)
                _ds = _es.get("daily_state", {})
                _today = datetime.now(timezone.utc).date().isoformat()
                if _ds.get("date") == _today:
                    self.portfolio_risk._day_start_equity = float(_ds.get("day_start_equity", self.initial_equity))
                    self.portfolio_risk._strategy_pnl = {k: float(v) for k, v in _ds.get("strategy_pnl", {}).items()}
                    log.info(
                        f"[PortRisk] Daily state restored: "
                        f"day_start=${self.portfolio_risk._day_start_equity:.2f} "
                        f"strategy_pnl={self.portfolio_risk._strategy_pnl}"
                    )
        except Exception as _e:
            log.warning(f"[PortRisk] Daily state restore failed: {_e}")

        # Monitor
        mcfg = self.cfg.get("monitoring", {})
        self.monitor = SlTpMonitorV2(
            exchange=self.exchange,
            pos_manager=self.pos_manager,
            close_callback=self._on_position_close,
            poll_seconds=mcfg.get("sl_tp_poll_seconds", 15),
            paper_mode=(self.mode == "paper"),
            discord_notify=discord_post,
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
                bot_version=self.cfg.get("version", "unknown"),
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
                entry_filters=self.entry_filters,
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

        # ── Fee Optimization: commission check + BnbKeeper + FeeScanner ──────────
        fee_opt_cfg = self.cfg.get("fee_optimization", {})

        # Commission rate check — 현재 계정의 실제 수수료율 조회 및 Discord 알림
        if fee_opt_cfg.get("commission_check", {}).get("enabled", True):
            try:
                _comm = await self.exchange._exchange.fapiPrivateGetCommissionRate(
                    {"symbol": "BTCUSDT"}
                )
                _maker_rate = float(_comm.get("makerCommissionRate", 0.0002))
                _taker_rate = float(_comm.get("takerCommissionRate", 0.0005))
                _discount = (_taker_rate < 0.0005)
                log.info(
                    f"[FeeCheck] maker={_maker_rate:.4%} taker={_taker_rate:.4%} "
                    f"discount={'YES' if _discount else 'NO'}"
                )
                discord_post(
                    f"**Maker:** `{_maker_rate:.4%}` | **Taker:** `{_taker_rate:.4%}`\n"
                    f"**BNB 할인:** `{'적용 중' if _discount else '미적용 — BNB 잔고 확인 필요'}`",
                    title="수수료 확인",
                )
            except Exception as _ce:
                log.warning(f"[FeeCheck] commission rate check failed: {_ce}")

        # BnbKeeper 초기화
        bnb_cfg = fee_opt_cfg.get("bnb_keeper", {})
        if bnb_cfg.get("enabled", True):
            self.bnb_keeper = BnbKeeper(
                exchange=self.exchange,
                min_bnb_usdt=float(bnb_cfg.get("min_bnb_usdt", 10.0)),
                check_interval=int(bnb_cfg.get("check_interval_sec", 3600)),
            )
            log.info(
                f"[BnbKeeper] 초기화 완료 "
                f"(min=${bnb_cfg.get('min_bnb_usdt', 10.0):.2f} USDT)"
            )
            # 즉시 1회 체크 (백그라운드 태스크 시작 전)
            try:
                _bnb_result = await self.bnb_keeper.check_and_buy()
                log.info(
                    f"[BnbKeeper] 초기 체크 완료: "
                    f"{_bnb_result['bnb_qty']:.4f} BNB "
                    f"(${_bnb_result['bnb_usdt_value']:.2f})"
                )
            except Exception as _be:
                log.warning(f"[BnbKeeper] 초기 체크 실패: {_be}")

        # FeeScanner 초기화
        scanner_cfg = fee_opt_cfg.get("fee_scanner", {})
        if scanner_cfg.get("enabled", True):
            self.fee_scanner = FeeScanner(
                exchange=self.exchange,
                scan_interval=int(scanner_cfg.get("scan_interval_sec", 86400)),
            )
            log.info("[FeeScanner] 초기화 완료")
            # 즉시 1회 스캔
            try:
                _zero_fee = await self.fee_scanner.scan()
                if _zero_fee:
                    log.info(f"[FeeScanner] ZERO-FEE 페어: {_zero_fee}")
            except Exception as _se:
                log.warning(f"[FeeScanner] 초기 스캔 실패: {_se}")

        # Discord: system ready notification
        strategy_summary = "\n".join(
            f"• {n}: ${scfg.get('allocation_usdt',0):.0f} / {scfg.get('leverage',2)}x / max {scfg.get('max_positions',1)} pos"
            for n, scfg in self.cfg.get("strategies", {}).items()
            if scfg.get("enabled", True)
        )
        _bot_ver = self.cfg.get("version", "v8.5")
        discord_post(
            f"**모드:** `{self.mode}`\n"
            f"**자본금:** `${self.initial_equity:,.2f}`\n"
            f"**최대 노셔널:** `{self.cfg.get('portfolio',{}).get('total_exposure_pct',0.7)*100:.0f}%`\n"
            f"**코인:** `{', '.join(self.coins)}`\n\n"
            f"**전략 구성:**\n{strategy_summary}",
            title=f"✅ {_bot_ver} 시스템 준비 완료 — 거래 시작",
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

        # EV Guardian + Fee Budget evaluation (every 1h)
        tasks.append(asyncio.create_task(
            self._ev_guardian_loop(), name="ev_guardian"
        ))

        # BNB Keeper — BNB 잔고 자동 유지 (수수료 할인 10% 확보)
        if self.bnb_keeper is not None:
            tasks.append(asyncio.create_task(
                self.bnb_keeper.start_background(), name="bnb_keeper"
            ))

        # Fee Scanner — 수수료 0원 페어 탐지 (24h 주기)
        if self.fee_scanner is not None:
            tasks.append(asyncio.create_task(
                self.fee_scanner.start_background(), name="fee_scanner"
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

        # Initial delay to stagger strategies (prevent simultaneous API bursts)
        delay = {
            "cvd_extreme": 0, "liquidation_fade": 5, "vwap_reversion": 10,
            "volume_impulse": 15, "oi_divergence": 20, "funding_arb": 25,
            "cvd_spike": 0, "momentum_breakout": 0, "asymmetric_sniper": 0,
        }
        await asyncio.sleep(delay.get(strategy.name, 0))

        while not self._shutdown_event.is_set():
            try:
                # Check kill switch
                if self.portfolio_risk.is_killed:
                    log.warning(f"[{strategy.name}] kill switch active, stopping")
                    break

                # Daily reset
                self.portfolio_risk.maybe_reset_daily()

                # ── F1 + F5: EV/Fee budget gate ──────────────
                if self.ev_guardian:
                    allowed, ev_reason = self.ev_guardian.check_entry_allowed(strategy.name)
                    if not allowed:
                        log.info(f"[{strategy.name}] blocked by EVGuardian: {ev_reason}")
                        try:
                            await asyncio.sleep(cycle)
                        except asyncio.CancelledError:
                            break
                        continue

                # Filter out coins in cooldown (15min after exit)
                active_coins = [
                    c for c in self.coins
                    if time.time() - self._coin_cooldowns.get(c, 0) >= self._coin_cooldown_sec
                ]

                # FeeScanner: zero-fee 페어 우선 로깅 (진입 거부는 하지 않음 — 정보성)
                if self.fee_scanner is not None and self.fee_scanner.zero_fee_pairs:
                    _zero_in_active = [
                        c for c in active_coins if self.fee_scanner.is_zero_fee(c)
                    ]
                    if _zero_in_active:
                        log.info(
                            f"[{strategy.name}] ZERO-FEE 우선 대상: {_zero_in_active}"
                        )

                # Bot-wide daily trade cap
                _today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                if _today != self._bot_daily_trade_date:
                    self._bot_daily_trade_count = 0
                    self._bot_daily_trade_date = _today
                if self._bot_daily_trade_count >= self._bot_max_daily_trades:
                    log.info(
                        f"[{strategy.name}] bot daily cap reached "
                        f"({self._bot_daily_trade_count}/{self._bot_max_daily_trades})"
                    )
                    try:
                        await asyncio.sleep(cycle)
                    except asyncio.CancelledError:
                        break
                    continue

                # Evaluate
                results = await strategy.tick(active_coins)
                # Per-scan diagnostics: cache stats + result summary
                cache_info = self.data_hub.cache_stats()
                log.info(
                    f"[Scan:{strategy.name}] trades={len(results)} | "
                    f"cache hits={cache_info.get('hits',0)} miss={cache_info.get('misses',0)} "
                    f"err={cache_info.get('errors',0)} | "
                    f"positions={self.pos_manager.count()}"
                )
                for r in results:
                    self._bot_daily_trade_count += 1
                    log.info(f"[{strategy.name}] TRADE: {r} (bot daily #{self._bot_daily_trade_count})")
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
                        except Exception as _snap_err:
                            log.warning(f"[Discord] portfolio snapshot 실패: {_snap_err}", exc_info=True)
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

            # base_coins 제외한 선택적 풀 (상위 selective_pool_size개), exclude_coins 필터 적용
            selective_coins = [
                r for r in selective_results
                if r["coin"] not in base_set and r["coin"] not in self._dc_exclude_coins
            ][:self._dc_selective_pool]

            # ── base_coins: 항상 포함 (exclude_coins는 base_coins에서도 제외) ──
            base_meta = [
                next(
                    (r for r in selective_results if r["coin"] == c),
                    {"coin": c, "volatility_pct": 0, "volume_usdt": 0,
                     "spread_bps": 0, "last": 0},
                )
                for c in self._dc_base_coins
                if c not in self._dc_exclude_coins
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
        await asyncio.sleep(3600)  # 시작 후 1시간 뒤 첫 실행
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

                # 오늘 세션 통계 (수수료 포함 순수익 기준)
                total_trades = len(self._session_trades)
                session_pnl = sum(t.get("pnl_net_usdt", t["pnl_usdt"]) for t in self._session_trades)
                wins = sum(1 for t in self._session_trades if t.get("pnl_net_usdt", t["pnl_usdt"]) > 0)
                wr = f"{wins/total_trades*100:.1f}%" if total_trades > 0 else "N/A"

                # 수수료 통계
                session_fee = sum(t.get("fee_usdt", 0.0) for t in self._session_trades)
                session_net = session_pnl  # already net
                ev_text = self.ev_guardian.format_discord_summary() if self.ev_guardian else ""

                # 세션 통계 섹션 — 거래 있을 때만 상세 표시
                if total_trades > 0:
                    session_stat_text = (
                        f"**📈 세션 통계** (거래: {total_trades}건)\n"
                        f"세션 PnL: `{session_pnl:+.4f} USDT` | 수수료: `{session_fee:.4f} USDT` | 순수익: `{session_net:+.4f} USDT`\n"
                        f"승률: `{wr}`\n\n"
                        f"**전략별 PnL:**\n{strat_lines}"
                    )
                else:
                    session_stat_text = "**📈 세션 통계** — 아직 거래 없음"

                discord_post(
                    f"**💰 잔고**\n"
                    f"자본: `${status['equity']:,.2f}` | 초기: `${self.initial_equity:,.2f}`\n"
                    f"일일 손익: `{status['daily_loss_pct']:+.2%}` | 노셔널: `${status['notional']:.2f}`\n\n"
                    f"**📊 오픈 포지션 ({len(positions)}개)**\n{pos_text}\n\n"
                    + session_stat_text
                    + (f"\n\n**🔍 EV Guardian:**\n{ev_text}" if ev_text else ""),
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
        """매 15분마다 변동성 상위 30개 전체 신호 분석 리포트 출력."""
        await asyncio.sleep(60)  # 시작 후 1분 뒤 첫 실행
        while not self._shutdown_event.is_set():
            try:
                now = datetime.now(timezone.utc)
                scored = await self._score_signal_universe()
                pos_lines = await self._format_open_positions_for_scan()
                msg = self._format_signal_scan_message(scored, pos_lines)
                discord_post(
                    msg,
                    title=f"📡 15분 전체 신호 스캔 — {now.strftime('%H:%M UTC')} | 상위30 분석",
                )
                strong = [s for s in scored if s["score"] >= 2]
                log.info(f"[SignalScan] {len(scored)} coins | {len(strong)} candidates")
            except asyncio.CancelledError:
                break
            except Exception as e:
                log.error(f"[Discord] signal scan error: {e}", exc_info=True)

            try:
                await asyncio.sleep(900)  # 15분
            except asyncio.CancelledError:
                break

    async def _score_signal_universe(self) -> list[dict]:
        """변동성 상위 30개 코인에 대해 CVD/OFI 기반 신호 점수를 계산한다."""
        universe = await self.data_hub.select_tradeable_coins(
            volatility_pool=30,
            min_volume_usdt=0,
            max_spread_bps=9999.0,
        )
        if not universe:
            universe = [{"coin": c, "volatility_pct": 0,
                         "volume_usdt": 0, "spread_bps": 0, "last": 0}
                        for c in self.coins]

        active_set = set(self.coins)
        open_coins = {
            self.pos_manager._parse_key(k)[1]
            for k in list(self.pos_manager.positions)  # copy: avoid RuntimeError on concurrent modification
        }

        scored = []
        for meta in universe:
            coin = meta["coin"]
            try:
                df = await self.data_hub.get_ohlcv(coin, "1m", limit=500)
                if df is None or len(df) < 100:
                    continue

                price = float(df["close"].iloc[-1])
                cvd = DataHub.compute_cvd(df)
                cvd_val = float(cvd.iloc[-1])
                cvd_mean = float(cvd.rolling(480).mean().iloc[-1])
                cvd_std = float(cvd.rolling(480).std().iloc[-1])
                cvd_z = (cvd_val - cvd_mean) / cvd_std if cvd_std > 0 else 0.0
                cvd_pct = float(cvd.rank(pct=True).iloc[-1])

                ofi = DataHub.compute_ofi(df)
                ofi_pct = float(ofi.rank(pct=True).iloc[-1])

                atr = StrategyBase._compute_atr(df, period=14)
                atr_pct = atr / price * 100 if price > 0 else 0.0

                ret_5 = (price - float(df["close"].iloc[-5])) / float(df["close"].iloc[-5])
                ret_20 = (price - float(df["close"].iloc[-20])) / float(df["close"].iloc[-20])

                direction, score = "─", 0
                if cvd_pct > 0.95 or cvd_z > 2.0:
                    direction = "역추세SHORT 🔴"
                    score += 2 + (1 if cvd_z > 3.0 else 0)
                elif cvd_pct < 0.05 or cvd_z < -2.0:
                    direction = "역추세LONG 🟢"
                    score += 2 + (1 if cvd_z < -3.0 else 0)
                if ofi_pct > 0.90 or ofi_pct < 0.10:
                    score += 1
                if atr_pct > 0.5:
                    score += 1

                status = "🔵 OPEN" if coin in open_coins else (
                         "✅ 활성" if coin in active_set else "⚪ 대기")
                scored.append({
                    "coin": coin, "direction": direction, "score": score,
                    "cvd_pct": cvd_pct, "cvd_z": cvd_z, "ofi_pct": ofi_pct,
                    "atr_pct": atr_pct, "ret_5": ret_5, "ret_20": ret_20,
                    "price": price, "vol_m": meta["volume_usdt"] / 1e6, "status": status,
                })
            except Exception:
                continue

        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored

    async def _format_open_positions_for_scan(self) -> list[str]:
        """보유 포지션 현황 라인 목록 반환 (신호 스캔 Discord 메시지용)."""
        positions = self.pos_manager.positions
        if not positions:
            return []
        lines = ["", "**━ 보유 포지션 ━**"]
        for pos in positions.values():
            try:
                ticker = await self.data_hub.get_ticker(pos.coin)
                cur_price = ticker.get("last", pos.entry_price)
                if pos.side == "BUY":
                    unreal_pct = (cur_price - pos.entry_price) / pos.entry_price
                else:
                    unreal_pct = (pos.entry_price - cur_price) / pos.entry_price
                unreal_usdt = unreal_pct * pos.current_qty * pos.entry_price
                bars_remaining = max(0, pos.ttl_bars - pos.bars_held)
                eta_min = bars_remaining
                eta_str = f"{eta_min//60}h{eta_min%60}m" if eta_min >= 60 else f"{eta_min}m"
                sl_dist_pct = abs(cur_price - pos.sl_price) / cur_price * 100
                side_emoji = "📈" if pos.side == "BUY" else "📉"
                pnl_emoji = "🟢" if unreal_usdt >= 0 else "🔴"
                lines.append(
                    f"{side_emoji} **{pos.coin}** {pos.side} `@{pos.entry_price:.4f}` "
                    f"→ 현재 `{cur_price:.4f}`\n"
                    f"  {pnl_emoji} 미실현: `{unreal_usdt:+.2f}$` (`{unreal_pct:+.2%}`) | "
                    f"SL까지: `{sl_dist_pct:.2f}%` | TTL잔여: `{eta_str}` ({bars_remaining}봉)"
                )
            except Exception:
                lines.append(f"• {pos.coin} {pos.side} @{pos.entry_price:.4f}")
        status = self.portfolio_risk.status()
        lines.append(
            f"\n💼 총 {len(positions)}포지션 | 노셔널 `${status['notional']:.0f}` | "
            f"잔고 `${status['equity']:.2f}` | 일일 `{status['daily_loss_pct']:+.2%}`"
        )
        return lines

    def _format_signal_scan_message(
        self, scored: list[dict], pos_lines: list[str]
    ) -> str:
        """신호 스캔 결과를 Discord 메시지 문자열로 포맷."""
        active_set = set(self.coins)
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
            lines.append(", ".join(f"`{s['coin']}`({s['atr_pct']:.1f}%)" for s in neutral[:15]))

        lines.extend(pos_lines)
        return "\n".join(lines)

    async def _on_position_close(
        self, strategy: str, coin: str, reason: str, price: float
    ) -> None:
        """Callback when monitor detects SL/TP/TTL hit."""
        pos = self.pos_manager.get_position(strategy, coin)
        if pos is None:
            return

        # Ghost positions never existed on exchange — just remove from tracker, no PnL/fee
        if reason == "GHOST_CLEANUP":
            self.pos_manager.remove_position(strategy, coin)
            log.info(
                f"[Close] 👻 {strategy}:{coin} GHOST_CLEANUP "
                f"— removed stale tracker entry (was never on exchange)"
            )
            return

        # Calculate PnL — use barrier price for SL/TP (more accurate than ticker snapshot)
        # When exchange SL fires, actual fill ≈ sl_price, not ticker["last"] at detection time.
        _fill_price = price
        if reason == "SL_HIT" and pos.sl_price > 0:
            _fill_price = pos.sl_price
        elif reason == "TP_HIT" and pos.tp_price > 0:
            _fill_price = pos.tp_price

        if pos.side == "BUY":
            pnl_pct = (_fill_price - pos.entry_price) / pos.entry_price
        else:
            pnl_pct = (pos.entry_price - _fill_price) / pos.entry_price

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
            except asyncio.CancelledError:
                # Task was cancelled mid-close — position state on exchange is unknown.
                # Keep position in tracker so monitor can detect and retry cleanup.
                log.warning(
                    f"[Close] {coin} CancelledError during market_close — "
                    f"position preserved in tracker, monitor will handle cleanup"
                )
                raise
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
                    # 청산 실패 시 pos_manager에 포지션 유지 — 모니터가 계속 추적
                    # 소프트웨어에서 삭제하면 고아 포지션(orphan) 발생 위험
                    discord_post(
                        f"⚠️ **거래소 청산 실패** — 수동 확인 필요\n"
                        f"종목: `{coin}` | 수량: `{pos.current_qty}` | 사유: `{e}`\n"
                        f"포지션 추적 유지 중 — 모니터가 재시도합니다.",
                        title="⚠️ 거래소 청산 실패",
                    )
                    return  # 청산 미확인 시 PnL 기록도 하지 않음

        # 수수료 계산 — 단 1회, EVGuardian + 로그 모두 이 값 사용
        # Entry: Post-Only maker + slip_in  /  Exit: taker + slip_out
        # FIXED: was `(entry+exit) * (ROUND_TRIP_FEE_RATE / 2)` which divides twice.
        _fee_maker   = 0.0002   # 0.0200%
        _fee_taker   = 0.0005   # 0.0500%
        _fee_slip_in = 0.0003   # 0.030%
        _fee_slip_out = 0.0005  # 0.050%
        _entry_notional = pos.entry_price * pos.current_qty
        _exit_notional = price * pos.current_qty
        _fee_usdt = round(
            _entry_notional * (_fee_maker + _fee_slip_in)
            + _exit_notional * (_fee_taker + _fee_slip_out), 4
        )

        # Record net PnL (gross - fee) so current_equity reflects actual balance
        self.portfolio_risk.record_trade_pnl(strategy, pnl_usdt - _fee_usdt)

        # F5: 실수수료 기록 (EVGuardian 일일 예산 누적)
        if self.ev_guardian:
            self.ev_guardian.record_fee(strategy, _fee_usdt)

        # exit ATR 계산 — True ATR (StrategyBase._compute_atr 재사용)
        _exit_atr = 0.0
        try:
            _df_exit = await self.data_hub.get_ohlcv(coin, "1m", limit=20)
            if _df_exit is not None and len(_df_exit) >= 15:
                _exit_atr = StrategyBase._compute_atr(_df_exit, period=14)
        except Exception:
            pass

        # Record full trade context for per-coin optimization
        _tc_record = None
        if self.trade_logger:
            try:
                _tc_record = self.trade_logger.record_close(
                    strategy=strategy,
                    coin=coin,
                    exit_price=price,
                    exit_reason=reason,
                    pos=pos,
                    exit_atr=_exit_atr,
                )
            except Exception as e:
                # ERROR (not WARNING) — trade_context 누락은 백테스트 데이터 손실
                log.error(f"[TradeLog] record_close FAILED — trade_context missing for {strategy}:{coin} reason={reason}: {e}")

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

        # Record SL hit for entry filter blacklist (only losing SL, not trailing wins)
        if reason == "SL_HIT" and pnl_usdt < 0 and self.entry_filters:
            self.entry_filters.record_sl_hit(coin)

        # Record SL hit for CVD Extreme per-coin cooldown
        if reason == "SL_HIT" and strategy in self.strategies:
            strat_obj = self.strategies[strategy]
            if hasattr(strat_obj, "record_sl_hit"):
                strat_obj.record_sl_hit(coin)

        # Record cooldown timestamp for this coin
        self._coin_cooldowns[coin] = time.time()

        # 순수익 계산 (_fee_usdt는 위에서 이미 계산됨)
        _pnl_net_usdt = round(pnl_usdt - _fee_usdt, 4)
        _pnl_net_pct = round(_pnl_net_usdt / _entry_notional, 6) if _entry_notional > 0 else 0.0

        # Log
        emoji = "✅" if _pnl_net_usdt > 0 else "❌"
        log.info(
            f"[Close] {emoji} {strategy}:{coin} {reason} | "
            f"PnL={pnl_usdt:+.4f} net={_pnl_net_usdt:+.4f} fee={_fee_usdt:.4f} | "
            f"entry={pos.entry_price:.4f} exit={price:.4f} | "
            f"bars={pos.bars_held} | "
            f"mfe={pos.mfe_pct:.2%} mae={pos.mae_pct:.2%}"
        )

        # Discord: 거래 완료 + 잔고 + 거래내역 요약
        status = self.portfolio_risk.status()
        # [Fix] trade를 먼저 append해야 최근 거래내역에 현재 거래가 포함됨
        # Compute SL/TP distances for analysis
        _ep = pos.entry_price
        _sl_pct = abs(pos.sl_price - _ep) / _ep if _ep > 0 else 0.0
        _tp_pct = abs(pos.tp_price - _ep) / _ep if (_ep > 0 and pos.tp_price > 0) else 0.0

        # Compute additional fields for optimization
        _sl_dist_pct = abs(_ep - pos.sl_price) / _ep * 100 if _ep > 0 else 0
        _tp_dist_pct = abs(pos.tp_price - _ep) / _ep * 100 if (_ep > 0 and pos.tp_price > 0) else 0
        _trail_dist_pct = getattr(pos, 'trail_distance', 0) / _ep * 100 if _ep > 0 else 0
        # _fee_usdt and _pnl_net_usdt already computed above (lines ~949, ~1004)
        _last_exit_ts = self._coin_cooldowns.get(coin, None) if hasattr(self, '_coin_cooldowns') else None
        _time_since_last = round(time.time() - _last_exit_ts, 1) if _last_exit_ts else 0.0
        _concurrent = self.pos_manager.count()
        _max_dd = pos.mae_pct  # MAE is the max drawdown during trade

        # trade_context 레코드에서 최적화용 필드 추출
        _tc = _tc_record or {}
        trade = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "strategy": strategy,
            "coin": coin,
            "side": pos.side,
            "entry": pos.entry_price,
            "exit": price,
            "sl_price": pos.sl_price,
            "tp_price": pos.tp_price,
            "sl_pct": round(_sl_pct * 100, 4),
            "tp_pct": round(_tp_pct * 100, 4),
            "pnl_usdt": round(pnl_usdt, 6),
            "pnl_pct": round(pnl_pct, 6),
            "fee_usdt": _fee_usdt,
            "pnl_net_usdt": _pnl_net_usdt,
            "pnl_net_pct": _pnl_net_pct,
            "reason": reason,
            "bars": pos.bars_held,
            "mfe": round(pos.mfe_pct, 6),
            "mae": round(pos.mae_pct, 6),
            "trailing_sl": pos.trailing_sl,
            "sl_tighten_count": getattr(pos, "sl_tighten_count", 0),
            "trail_distance": getattr(pos, "trail_distance", 0.0),
            "notional": round(pos.entry_price * pos.qty, 2),
            "leverage": getattr(pos, "leverage", 0),
            # ── 최적화용 컨텍스트 필드 (trade_context에서 추출) ──
            "hour_utc": _tc.get("hour_utc", -1),
            "session_utc": _tc.get("session_utc", ""),
            "volatility_regime": _tc.get("volatility_regime", ""),
            "btc_regime": _tc.get("btc_regime_1h", ""),
            "trade_vs_btc": _tc.get("trade_vs_btc_regime", ""),
            "signal_strength": _tc.get("signal_strength", 0.0),
            "cvd_z_score": _tc.get("cvd_z_score", 0.0),
            "cvd_quantile": _tc.get("cvd_quantile_breach", 0.0),
            "rr_estimate": _tc.get("rr_estimate", 0.0),
            "entry_atr_pct": _tc.get("entry_atr_pct", 0.0),
            "atr_ratio": _tc.get("atr_ratio_exit_entry", 0.0),
            "concurrent_pos": _tc.get("concurrent_positions", 0),
            "fee_drag_pct": _tc.get("fee_drag_pct", 0.0),
            # ── v8.5 추가 필드 (ML 최적화용) ──
            "sl_distance_pct": round(_sl_dist_pct, 4),
            "tp_distance_pct": round(_tp_dist_pct, 4),
            "trail_distance_pct": round(_trail_dist_pct, 4),
            "concurrent_positions": _concurrent,
            "time_since_last_trade_sec": round(_time_since_last, 1),
            "max_drawdown_pct": round(_max_dd, 6),
            # ── 버전 & 파라미터 (재현성 추적) ──
            "bot_version": _tc.get("bot_version", self.cfg.get("version", "unknown")),
            "strategy_params": _tc.get("strategy_params", {}),
        }
        self._session_trades.append(trade)
        # Cap in-memory list to prevent unbounded growth over long sessions
        if len(self._session_trades) > 1000:
            self._session_trades = self._session_trades[-1000:]
        self._save_trade(trade)

        # Record to OrderLedger (SQLite) — net PnL (fee-included)
        if self.ledger:
            try:
                self.ledger.record_pnl(
                    symbol=coin,
                    realized_pnl=_pnl_net_usdt,
                    fees=_fee_usdt,
                )
            except Exception as e:
                log.warning(f"[Ledger] record_pnl failed: {e}")

        # Discord: 거래 완료 + 잔고 + 거래내역 요약 (수수료 포함 순수익 기준)
        total_trades = len(self._session_trades)
        session_pnl = sum(t.get("pnl_net_usdt", t["pnl_usdt"]) for t in self._session_trades)
        wins = sum(1 for t in self._session_trades if t.get("pnl_net_usdt", t["pnl_usdt"]) > 0)
        wr = f"{wins/total_trades*100:.1f}%" if total_trades > 0 else "N/A"

        # 최근 5건 (현재 거래 포함)
        recent = self._session_trades[-5:]
        recent_lines = []
        for t in reversed(recent):
            _net = t.get("pnl_net_usdt", t["pnl_usdt"])
            e = "✅" if _net > 0 else "❌"
            recent_lines.append(f"{e} **{t['coin']}** `{t['side']}` `{_net:+.4f}` ({t.get('pnl_net_pct', t['pnl_pct']):+.2%}) `{t['reason']}`")
        recent_text = "\n".join(recent_lines)

        emoji_trade = "✅" if pnl_usdt > 0 else "❌"
        reason_label = {"SL_HIT": "🛑 손절", "TP_HIT": "🎯 익절", "TTL_HIT": "⏰ 시간초과", "TIME_STOP": "⏰ 시간초과", "KILL_SWITCH": "🚨 킬스위치"}.get(reason, f"📌 {reason}")

        try:
            portfolio_snap = await self._build_portfolio_snapshot()
        except Exception:
            portfolio_snap = f"자본: `${status['equity']:,.2f}` | 노셔널: `${status['notional']:.2f}`"

        _ev_summary = self.ev_guardian.format_discord_summary() if self.ev_guardian else ""
        discord_post(
            f"**{emoji_trade} 청산 결과**\n"
            f"종목: `{coin}` `{pos.side}` | [{strategy}]\n"
            f"진입: `{pos.entry_price:.4f}` → 청산: `{price:.4f}`\n"
            f"**PnL: `{pnl_usdt:+.4f} USDT` (`{pnl_pct:+.2%}`)**\n"
            f"수수료: `{_fee_usdt:.4f} USDT` | **순수익: `{_pnl_net_usdt:+.4f} USDT` (`{_pnl_net_pct:+.4%}`)**\n"
            f"사유: {reason_label} | 보유: `{pos.bars_held}봉`\n"
            f"MFE: `{pos.mfe_pct:.2%}` | MAE: `{pos.mae_pct:.2%}`\n\n"
            f"{portfolio_snap}\n\n"
            f"**📋 세션** ({total_trades}건 | 승률 {wr} | 순PnL `{session_pnl:+.4f}`)\n"
            f"{recent_text}"
            + (f"\n\n**🔍 EVGuardian:**\n{_ev_summary}" if _ev_summary else ""),
            title=f"{'✅ 익절' if _pnl_net_usdt > 0 else '❌ 손절/청산'} — {coin} {reason_label}",
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
            stale = False
            try:
                ticker = await self.data_hub.get_ticker(p.coin)
                cur = ticker.get("last", p.entry_price)
                if not cur or cur <= 0:
                    cur = p.entry_price
                    stale = True
            except Exception as _te:
                log.warning(f"[Discord] {p.coin} ticker 조회 실패, 진입가로 대체: {_te}")
                cur = p.entry_price
                stale = True

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
            stale_mark = " ⚠️stale" if stale else ""
            tp_str = f"TP`{p.tp_price:.4f}`" if getattr(p, "tp_price", 0) else "TP`trailing`"
            lines.append(
                f"{mark}{sign} `{p.coin}` {arrow}`{p.side}` [{p.strategy_tag}]{stale_mark}\n"
                f"    진입`{p.entry_price:.4f}` → 현재`{cur:.4f}` | "
                f"`{unreal_usdt:+.2f}` (`{unreal_pct:+.1%}`) | "
                f"SL`{p.sl_price:.4f}` {tp_str}"
            )

        pos_section = "\n".join(lines) if lines else "포지션 없음"
        total_sign = "🟢" if total_unrealized >= 0 else "🔴"
        session_pnl = sum(t.get("pnl_net_usdt", t["pnl_usdt"]) for t in self._session_trades)
        total_trades = len(self._session_trades)
        wins = sum(1 for t in self._session_trades if t.get("pnl_net_usdt", t["pnl_usdt"]) > 0)
        wr = f"{wins/total_trades*100:.0f}%" if total_trades > 0 else "-"

        return (
            f"**📊 포지션 현황 ({len(positions)}개)**\n{pos_section}\n\n"
            f"**💰 계좌**\n"
            f"자본: `${status['equity']:,.2f}` | 노셔널: `${status['notional']:.2f}`\n"
            f"{total_sign} 미실현손익: `{total_unrealized:+.2f} USDT` | "
            f"일일: `{status['daily_loss_pct']:+.1%}`\n"
            f"세션: `{total_trades}건` WR`{wr}` 순PnL`{session_pnl:+.2f}`"
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
            session_trades = len(self._session_trades)
            session_pnl = round(sum(t.get("pnl_net_usdt", t.get("pnl_usdt", 0)) for t in self._session_trades), 6)
            today = datetime.now(timezone.utc).date().isoformat()
            # daily state for restart recovery (strategy_pnl, day_start_equity)
            pr = self.portfolio_risk
            daily_state = {
                "date": today,
                "day_start_equity": round(pr._day_start_equity, 6) if pr else round(current, 6),
                "strategy_pnl": {k: round(v, 6) for k, v in pr._strategy_pnl.items()} if pr else {},
            }
            existing.update({
                "current_equity": round(current, 6),
                "initial_equity": round(self.initial_equity, 6),
                "restarts": existing.get("restarts", 0) + (1 if not existing else 0),
                # cumulative all-time counters (baseline + current session)
                "total_trades": self._baseline_total_trades + session_trades,
                "total_pnl": round(self._baseline_total_pnl + session_pnl, 6),
                # current session counters for reference
                "session_trades": session_trades,
                "session_pnl": session_pnl,
                # daily state for restart recovery
                "daily_state": daily_state,
                "last_updated": datetime.now(timezone.utc).isoformat(),
            })
            with open(EQUITY_STATE_FILE, "w") as f:
                json.dump(existing, f, indent=2)
        except Exception as e:
            log.warning(f"[EquityState] Save failed: {e}")

    @staticmethod
    def _load_baseline_counters() -> tuple[int, float]:
        """Load cumulative total_trades / total_pnl from disk (before this session)."""
        try:
            if EQUITY_STATE_FILE.exists():
                with open(EQUITY_STATE_FILE, "r") as f:
                    data = json.load(f)
                return int(data.get("total_trades", 0)), float(data.get("total_pnl", 0.0))
        except Exception:
            pass
        return 0, 0.0

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
        """Append trade to JSONL file with fsync for crash safety."""
        try:
            with open(TRADES_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(trade, default=str) + "\n")
                f.flush()
                os.fsync(f.fileno())  # 디스크 강제 동기화 — 크래시 시 손상 방지
        except Exception as e:
            log.error(f"[SaveTrade] Write failed: {e}")
        self._save_equity_state()

        # ── Auto checkpoint: 50/100건 도달 시 Discord 알림 + 분석 ──
        total = self._baseline_total_trades + len(self._session_trades)
        if total in (50, 100, 150, 200):
            self._auto_checkpoint(total)

    def _auto_checkpoint(self, total_trades: int) -> None:
        """Triggered at 50/100/150/200 trades — run analysis and post to Discord."""
        try:
            analyzer = StrategyAnalyzer(STATE_DIR)
            report = analyzer.run()
            summary = analyzer.discord_summary(report) if report.total_trades > 0 else "(분석 불가)"
            checkpoint_msg = (
                f"**{total_trades}건 체크포인트 도달**\n\n{summary}\n\n"
            )
            if total_trades >= 100:
                checkpoint_msg += (
                    "⚠️ **Live 승급 검토 가능** — Net PnL > 0 + MDD < 30% 확인 필요"
                )
            discord_post(checkpoint_msg, title=f"🏁 {total_trades}건 체크포인트")
            log.info(f"[Checkpoint] {total_trades} trades milestone — analysis posted")
        except Exception as e:
            log.error(f"[Checkpoint] Failed: {e}")

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

    async def _ev_guardian_loop(self) -> None:
        """1시간마다 EV 재평가 + 수수료 예산 현황 Discord 보고.

        F1: 전략별 기대수익 계산 → 음수면 차단, 회복되면 자동 재개.
        F5: 일일 수수료 예산 현황 보고.
        """
        await asyncio.sleep(1800)  # 시작 30분 후 첫 평가 (데이터 축적 대기)
        while not self._shutdown_event.is_set():
            try:
                if not self.ev_guardian:
                    break

                # EV 재평가 (blocking I/O → executor)
                loop = asyncio.get_running_loop()
                ev_stats = await loop.run_in_executor(None, self.ev_guardian.evaluate)
                self.ev_guardian.save_report()

                # 상태 변화 감지 + Discord 알림 + allocation fallback
                suspended_now = [n for n, s in ev_stats.items() if s.get("suspended")]
                active_strats = [
                    n for n in self.strategies
                    if n not in suspended_now and self.strategies[n].config.enabled
                ]
                if suspended_now:
                    lines = []
                    for name in suspended_now:
                        stat = ev_stats[name]
                        lines.append(
                            f"🚫 **{name}** 차단 — EV=`{stat['ev']:.4%}` "
                            f"TP율=`{stat['tp_hit_rate']:.1%}` n=`{stat['n_trades']}`\n"
                            f"  avg_tp=`{stat['avg_tp_pct']:.4%}` "
                            f"avg_sl=`{stat['avg_sl_pct']:.4%}` "
                            f"fee=`{stat['avg_fee_drag_pct']:.4%}`"
                        )

                    # ── Allocation fallback: 차단된 전략의 자본을 남은 전략에 몰아줌 ──
                    if active_strats:
                        total_budget = sum(
                            self.strategies[n].config.allocation_usdt
                            for n in self.strategies
                            if self.strategies[n].config.enabled
                        )
                        per_active = total_budget / len(active_strats)
                        for name in active_strats:
                            old_alloc = self.strategies[name].config.allocation_usdt
                            self.strategies[name].config.allocation_usdt = per_active
                            if abs(old_alloc - per_active) > 1:
                                log.info(
                                    f"[EVGuardian] {name} allocation: "
                                    f"${old_alloc:.0f} → ${per_active:.0f} "
                                    f"(fallback from suspended strategies)"
                                )
                        lines.append(
                            f"\n💰 **Fallback**: 남은 전략에 자본 재배분 "
                            f"→ {', '.join(f'`{n}` ${per_active:.0f}' for n in active_strats)}"
                        )

                    discord_post(
                        "\n".join(lines),
                        title="⚠️ EVGuardian — 음수 EV 전략 차단",
                    )
                    log.warning(f"[EVGuardian] Suspended strategies: {suspended_now}")

                # 매 평가마다 간략 로그
                for name, stat in ev_stats.items():
                    log.info(
                        f"[EVGuardian] {name}: EV={stat['ev']:.4%} "
                        f"tp_rate={stat['tp_hit_rate']:.1%} n={stat['n_trades']} "
                        f"{'SUSPENDED' if stat['suspended'] else 'ok'}"
                    )

                # 수수료 예산 현황
                fee_exceeded, fee_detail = self.ev_guardian.is_fee_budget_exceeded()
                if fee_exceeded:
                    discord_post(
                        f"🔴 **일일 수수료 예산 초과 — 신규 진입 차단**\n{fee_detail}",
                        title="⛔ F5 Fee Budget Exceeded",
                    )
                    log.warning(f"[EVGuardian] Fee budget exceeded: {fee_detail}")

            except asyncio.CancelledError:
                break
            except Exception as e:
                log.error(f"[EVGuardian] loop error: {e}", exc_info=True)

            try:
                await asyncio.sleep(3600)  # 1시간마다
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
                # Cancel exchange SL/TP before market close — prevents residual
                # conditional orders from firing after position is already gone
                for attr_id, attr_link, label in [
                    ("sl_exchange_id", "sl_order_id", "SL"),
                    ("tp_exchange_id", "tp_order_id", "TP"),
                ]:
                    exch_id = getattr(pos, attr_id, "")
                    link_id = getattr(pos, attr_link, "")
                    if exch_id or link_id:
                        try:
                            await self.exchange.cancel_order(
                                pos.coin,
                                exchange_order_id=exch_id or None,
                                order_link_id=link_id or None,
                            )
                            log.info(f"[Shutdown] {pos.coin} exchange {label} cancelled")
                        except Exception as e:
                            log.warning(f"[Shutdown] {pos.coin} {label} cancel: {e}")
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

        # Summary (수수료 포함 순수익 기준)
        total_trades = len(self._session_trades)
        total_pnl = sum(t.get("pnl_net_usdt", t["pnl_usdt"]) for t in self._session_trades)
        total_fee = sum(t.get("fee_usdt", 0.0) for t in self._session_trades)
        wins = sum(1 for t in self._session_trades if t.get("pnl_net_usdt", t["pnl_usdt"]) > 0)
        wr = wins / total_trades * 100 if total_trades > 0 else 0

        log.info(f"{'='*60}")
        log.info(f"Session summary (net of fees):")
        log.info(f"  Trades: {total_trades}")
        log.info(f"  Net PnL: ${total_pnl:+.4f} (fee: ${total_fee:.4f})")
        log.info(f"  Win rate: {wr:.1f}%")
        log.info(f"  Final equity: ${self.portfolio_risk.current_equity:.2f}")

        for name in self.strategies:
            strat_trades = [t for t in self._session_trades if t["strategy"] == name]
            strat_pnl = sum(t.get("pnl_net_usdt", t["pnl_usdt"]) for t in strat_trades)
            log.info(f"  [{name}] trades={len(strat_trades)} net_pnl=${strat_pnl:+.4f}")
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

    # ── 중복 실행 방지 (PID 파일 락) ──────────────────────────
    PID_FILE = STATE_DIR / "bot.pid"
    import fcntl as _fcntl
    _pid_fh = open(PID_FILE, "w")
    try:
        _fcntl.flock(_pid_fh, _fcntl.LOCK_EX | _fcntl.LOCK_NB)
    except OSError:
        try:
            _existing_pid = PID_FILE.read_text().strip()
        except Exception:
            _existing_pid = "unknown"
        print(
            f"[FATAL] 봇이 이미 실행 중입니다 (PID {_existing_pid}). "
            f"중복 실행을 차단합니다. 기존 프로세스를 먼저 종료하세요.",
            file=sys.stderr,
        )
        sys.exit(1)
    _pid_fh.write(str(os.getpid()))
    _pid_fh.flush()
    import atexit as _atexit
    _atexit.register(lambda: PID_FILE.unlink(missing_ok=True))

    # ── Discord 연동 확인 (필수 전제조건) ──────────────────────
    _discord_url = os.environ.get("DISCORD_WEBHOOK_URL", "")
    if not _discord_url:
        print("[FATAL] DISCORD_WEBHOOK_URL이 .env에 설정되지 않았습니다. 봇을 시작할 수 없습니다.", file=sys.stderr)
        sys.exit(1)
    try:
        _dc_check_payload = json.dumps({"content": "🟢 봇 시작 — Discord 연동 확인 완료"}).encode("utf-8")
        _dc_req = urllib.request.Request(
            _discord_url,
            data=_dc_check_payload,
            headers={"Content-Type": "application/json", "User-Agent": "DiscordBot (ru_trading_bot, 1.0)"},
            method="POST",
        )
        urllib.request.urlopen(_dc_req, timeout=8)
        print("[OK] Discord 연동 확인 완료")
    except Exception as _e:
        print(f"[FATAL] Discord webhook 연결 실패: {_e}\n.env의 DISCORD_WEBHOOK_URL을 확인하세요.", file=sys.stderr)
        sys.exit(1)

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
