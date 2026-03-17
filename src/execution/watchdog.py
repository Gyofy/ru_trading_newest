"""Watchdog — 독립 감시 프로세스.

LiveEngine과 별도로 실행. heartbeat, drawdown, 네트워크 감시.
문제 발생 시 kill switch 파일 작성 + 비상 청산 시도.

Usage:
    python -m src.execution.watchdog
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path("C:/Users/RJ/Desktop/CLAUDE_CRYPTO_AGENT")
HEARTBEAT_PATH = PROJECT_ROOT / "data" / "heartbeat.json"
KILL_SWITCH_PATH = PROJECT_ROOT / "data" / "kill_switch.json"
ALERTS_DIR = PROJECT_ROOT / "data" / "alerts"
LOG_PATH = PROJECT_ROOT / "logs" / "watchdog.log"


class WatchdogConfig:
    check_interval_sec: float = 60
    heartbeat_stale_sec: float = 300     # 5분
    max_daily_drawdown_pct: float = 0.03  # 3% (watchdog은 더 보수적)
    max_consecutive_failures: int = 3
    bybit_ping_url: str = "https://api.bybit.com/v5/market/time"


class Watchdog:
    """독립 감시 프로세스.

    Checks:
      1. Heartbeat freshness (LiveEngine alive?)
      2. Network connectivity (Bybit reachable?)
      3. Drawdown monitoring (SQLite 직접 읽기)
      4. Kill switch management
    """

    def __init__(self, config: WatchdogConfig | None = None):
        self.config = config or WatchdogConfig()
        self._failure_count: dict[str, int] = {}  # check_name → consecutive failures
        self._running = True

    def run(self) -> None:
        """Main watchdog loop."""
        _setup_logging()
        logger.info("Watchdog started")
        ALERTS_DIR.mkdir(parents=True, exist_ok=True)

        while self._running:
            try:
                results = {
                    "heartbeat": self._check_heartbeat(),
                    "network": self._check_network(),
                    "drawdown": self._check_drawdown(),
                }

                # Count failures
                for name, ok in results.items():
                    if not ok:
                        self._failure_count[name] = self._failure_count.get(name, 0) + 1
                        logger.warning(
                            f"[Watchdog] {name} FAIL "
                            f"({self._failure_count[name]}/{self.config.max_consecutive_failures})"
                        )
                    else:
                        self._failure_count[name] = 0

                # Trigger emergency if any check fails N times
                for name, count in self._failure_count.items():
                    if count >= self.config.max_consecutive_failures:
                        self._trigger_emergency(f"{name} failed {count}x consecutively")
                        self._failure_count[name] = 0  # Reset after trigger

            except KeyboardInterrupt:
                logger.info("Watchdog stopped by user")
                break
            except Exception as e:
                logger.error(f"Watchdog error: {e}")

            time.sleep(self.config.check_interval_sec)

    # ── Checks ──────────────────────────────────────────────

    def _check_heartbeat(self) -> bool:
        """LiveEngine heartbeat freshness."""
        if not HEARTBEAT_PATH.exists():
            logger.warning("No heartbeat file")
            return False

        try:
            with open(HEARTBEAT_PATH) as f:
                data = json.load(f)

            ts = datetime.fromisoformat(data["ts"])
            # Make timezone-aware if needed
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)

            age = (datetime.now(timezone.utc) - ts).total_seconds()
            if age > self.config.heartbeat_stale_sec:
                logger.warning(f"Heartbeat stale: {age:.0f}s old")
                return False

            # Check for kill switch in heartbeat
            if data.get("kill_switch", False):
                logger.warning("Kill switch active in heartbeat")

            return True

        except Exception as e:
            logger.error(f"Heartbeat read error: {e}")
            return False

    def _check_network(self) -> bool:
        """Bybit API reachability."""
        try:
            import urllib.request
            req = urllib.request.Request(
                self.config.bybit_ping_url,
                method="GET",
            )
            req.add_header("User-Agent", "watchdog/1.0")
            with urllib.request.urlopen(req, timeout=10) as resp:
                return resp.status == 200
        except Exception as e:
            logger.warning(f"Network check failed: {e}")
            return False

    def _check_drawdown(self) -> bool:
        """SQLite에서 일일 P&L 직접 확인."""
        try:
            from src.execution.order_ledger import OrderLedger, DEFAULT_DB_PATH
            if not DEFAULT_DB_PATH.exists():
                return True  # No ledger = no trades = ok

            ledger = OrderLedger(DEFAULT_DB_PATH)
            total_pnl = ledger.get_daily_total_pnl()
            ledger.close()

            # Read initial equity from heartbeat if available
            initial_equity = 10000  # default fallback
            if HEARTBEAT_PATH.exists():
                with open(HEARTBEAT_PATH) as f:
                    hb = json.load(f)
                initial_equity = hb.get("initial_equity", initial_equity)

            if initial_equity > 0:
                dd_pct = abs(min(0, total_pnl)) / initial_equity
                if dd_pct > self.config.max_daily_drawdown_pct:
                    logger.warning(f"Drawdown alert: {dd_pct:.2%}")
                    return False

            return True

        except Exception as e:
            logger.error(f"Drawdown check failed: {e}")
            return True  # Fail open (don't trigger emergency for DB errors)

    # ── Emergency ───────────────────────────────────────────

    def _trigger_emergency(self, reason: str) -> None:
        """Write kill switch file and create alert."""
        logger.error(f"EMERGENCY TRIGGER: {reason}")

        # Write kill switch file
        KILL_SWITCH_PATH.parent.mkdir(parents=True, exist_ok=True)
        kill_data = {
            "triggered_at": datetime.now(timezone.utc).isoformat(),
            "reason": reason,
            "source": "watchdog",
        }
        with open(KILL_SWITCH_PATH, "w") as f:
            json.dump(kill_data, f, indent=2)

        # Write alert
        alert_path = ALERTS_DIR / f"alert_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(alert_path, "w") as f:
            json.dump(kill_data, f, indent=2)

        # Attempt emergency close via separate exchange instance
        self._attempt_emergency_close(reason)

    def _attempt_emergency_close(self, reason: str) -> None:
        """비상 청산 시도 (동기 ccxt 사용)."""
        try:
            import ccxt as ccxt_sync

            # Read config (in production, from env vars or config file)
            # For now, log the intent
            logger.warning(
                f"[Watchdog] Emergency close requested: {reason}. "
                f"Kill switch file written to {KILL_SWITCH_PATH}"
            )

            # The LiveEngine should read the kill switch file on next iteration
            # and perform the actual close.
            # Direct close from watchdog requires API credentials which
            # should be loaded from environment.

        except Exception as e:
            logger.error(f"Emergency close attempt failed: {e}")


def _setup_logging() -> None:
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(LOG_PATH, encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def main():
    watchdog = Watchdog()
    watchdog.run()


if __name__ == "__main__":
    main()
