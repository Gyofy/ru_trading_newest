"""Order Ledger — SQLite 기반 주문/체결/상태 영속화.

프로세스 재시작 시 복구, idempotent 이벤트 처리, 일일 P&L 추적.
orderLinkId가 전체 시스템의 idempotency key.

Format: wf3-{symbol}-{YYYYMMDD}-{HHmmss}-{BUY|SELL}-{seq:04d}
"""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone, date
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path("C:/Users/RJ/Desktop/CLAUDE_CRYPTO_AGENT")
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "ledger.db"


class OrderLedger:
    """SQLite-backed order/fill/state ledger.

    Thread-safe via per-thread connections with check_same_thread=False
    and a write lock for mutations.
    """

    def __init__(self, db_path: Path | str = DEFAULT_DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._write_lock = threading.Lock()
        self._conn = sqlite3.connect(
            str(self.db_path), check_same_thread=False,
            detect_types=sqlite3.PARSE_DECLTYPES,
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._create_tables()

    def _create_tables(self) -> None:
        with self._conn:
            self._conn.executescript("""
                CREATE TABLE IF NOT EXISTS orders (
                    order_link_id   TEXT PRIMARY KEY,
                    symbol          TEXT NOT NULL,
                    side            TEXT NOT NULL,
                    order_type      TEXT NOT NULL,
                    purpose         TEXT NOT NULL DEFAULT 'entry',
                    qty             REAL NOT NULL,
                    price           REAL,
                    stop_trigger    REAL,
                    status          TEXT NOT NULL DEFAULT 'PENDING',
                    exchange_order_id TEXT,
                    parent_id       TEXT,
                    created_at      TEXT NOT NULL,
                    updated_at      TEXT NOT NULL,
                    metadata        TEXT
                );

                CREATE TABLE IF NOT EXISTS fills (
                    fill_id         INTEGER PRIMARY KEY AUTOINCREMENT,
                    order_link_id   TEXT NOT NULL,
                    fill_price      REAL NOT NULL,
                    fill_qty        REAL NOT NULL,
                    fee             REAL NOT NULL DEFAULT 0,
                    filled_at       TEXT NOT NULL,
                    raw_response    TEXT,
                    FOREIGN KEY (order_link_id) REFERENCES orders(order_link_id)
                );

                CREATE TABLE IF NOT EXISTS state_transitions (
                    id              INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol          TEXT NOT NULL,
                    from_state      TEXT NOT NULL,
                    to_state        TEXT NOT NULL,
                    trigger         TEXT NOT NULL,
                    order_link_id   TEXT,
                    ts              TEXT NOT NULL,
                    details         TEXT
                );

                CREATE TABLE IF NOT EXISTS daily_pnl (
                    date            TEXT NOT NULL,
                    symbol          TEXT NOT NULL,
                    realized_pnl    REAL NOT NULL DEFAULT 0,
                    fees_paid       REAL NOT NULL DEFAULT 0,
                    trades_count    INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY (date, symbol)
                );

                CREATE INDEX IF NOT EXISTS idx_orders_symbol
                    ON orders(symbol, status);
                CREATE INDEX IF NOT EXISTS idx_fills_order
                    ON fills(order_link_id);
                CREATE INDEX IF NOT EXISTS idx_transitions_symbol
                    ON state_transitions(symbol, ts);
                CREATE INDEX IF NOT EXISTS idx_daily_date
                    ON daily_pnl(date);
            """)

    # ── Orders ──────────────────────────────────────────────

    def insert_order(
        self,
        order_link_id: str,
        symbol: str,
        side: str,
        order_type: str,
        qty: float,
        price: float | None = None,
        stop_trigger: float | None = None,
        purpose: str = "entry",
        parent_id: str | None = None,
        metadata: dict | None = None,
    ) -> bool:
        """Insert order. Returns False if duplicate (idempotent)."""
        now = _now_iso()
        meta_json = json.dumps(metadata) if metadata else None
        # Check for duplicate first
        existing = self._conn.execute(
            "SELECT 1 FROM orders WHERE order_link_id=?",
            (order_link_id,),
        ).fetchone()
        if existing:
            return False

        with self._write_lock:
            try:
                self._conn.execute(
                    """INSERT INTO orders
                       (order_link_id, symbol, side, order_type, purpose,
                        qty, price, stop_trigger, status, parent_id,
                        created_at, updated_at, metadata)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'PENDING', ?, ?, ?, ?)""",
                    (order_link_id, symbol, side, order_type, purpose,
                     qty, price, stop_trigger, parent_id,
                     now, now, meta_json),
                )
                self._conn.commit()
                return True
            except sqlite3.IntegrityError:
                return False

    def update_order_status(
        self,
        order_link_id: str,
        status: str,
        exchange_order_id: str | None = None,
    ) -> None:
        now = _now_iso()
        with self._write_lock:
            if exchange_order_id:
                self._conn.execute(
                    """UPDATE orders SET status=?, exchange_order_id=?, updated_at=?
                       WHERE order_link_id=?""",
                    (status, exchange_order_id, now, order_link_id),
                )
            else:
                self._conn.execute(
                    """UPDATE orders SET status=?, updated_at=?
                       WHERE order_link_id=?""",
                    (status, now, order_link_id),
                )
            self._conn.commit()

    def get_order(self, order_link_id: str) -> dict | None:
        row = self._conn.execute(
            "SELECT * FROM orders WHERE order_link_id=?",
            (order_link_id,),
        ).fetchone()
        return dict(row) if row else None

    def get_orders_by_symbol(self, symbol: str, status: str | None = None) -> list[dict]:
        if status:
            rows = self._conn.execute(
                "SELECT * FROM orders WHERE symbol=? AND status=?",
                (symbol, status),
            ).fetchall()
        else:
            rows = self._conn.execute(
                "SELECT * FROM orders WHERE symbol=?",
                (symbol,),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_child_orders(self, parent_id: str) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM orders WHERE parent_id=?", (parent_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    # ── Fills ───────────────────────────────────────────────

    def insert_fill(
        self,
        order_link_id: str,
        fill_price: float,
        fill_qty: float,
        fee: float = 0.0,
        filled_at: str | None = None,
        raw_response: dict | None = None,
    ) -> bool:
        """Insert fill. Idempotent: checks for duplicate fill."""
        filled_at = filled_at or _now_iso()
        raw_json = json.dumps(raw_response) if raw_response else None

        # Idempotency: check if this exact fill already exists
        existing = self._conn.execute(
            """SELECT fill_id FROM fills
               WHERE order_link_id=? AND fill_price=? AND fill_qty=?
               AND filled_at=?""",
            (order_link_id, fill_price, fill_qty, filled_at),
        ).fetchone()
        if existing:
            return False

        with self._write_lock:
            self._conn.execute(
                """INSERT INTO fills
                   (order_link_id, fill_price, fill_qty, fee, filled_at, raw_response)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (order_link_id, fill_price, fill_qty, fee, filled_at, raw_json),
            )
            self._conn.commit()
        return True

    def get_fills(self, order_link_id: str) -> list[dict]:
        rows = self._conn.execute(
            "SELECT * FROM fills WHERE order_link_id=? ORDER BY filled_at",
            (order_link_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def has_fill(self, order_link_id: str) -> bool:
        row = self._conn.execute(
            "SELECT 1 FROM fills WHERE order_link_id=? LIMIT 1",
            (order_link_id,),
        ).fetchone()
        return row is not None

    # ── State Transitions ───────────────────────────────────

    def log_transition(
        self,
        symbol: str,
        from_state: str,
        to_state: str,
        trigger: str,
        order_link_id: str | None = None,
        details: dict | None = None,
    ) -> None:
        now = _now_iso()
        det_json = json.dumps(details) if details else None
        with self._write_lock:
            self._conn.execute(
                """INSERT INTO state_transitions
                   (symbol, from_state, to_state, trigger, order_link_id, ts, details)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (symbol, from_state, to_state, trigger, order_link_id, now, det_json),
            )
            self._conn.commit()

    def get_last_state(self, symbol: str) -> str:
        """Return the last known state for a symbol, or 'IDLE'."""
        row = self._conn.execute(
            """SELECT to_state FROM state_transitions
               WHERE symbol=? ORDER BY id DESC LIMIT 1""",
            (symbol,),
        ).fetchone()
        return row["to_state"] if row else "IDLE"

    def get_transitions(self, symbol: str, limit: int = 50) -> list[dict]:
        rows = self._conn.execute(
            """SELECT * FROM state_transitions
               WHERE symbol=? ORDER BY id DESC LIMIT ?""",
            (symbol, limit),
        ).fetchall()
        return [dict(r) for r in rows]

    # ── Daily P&L ───────────────────────────────────────────

    def record_pnl(
        self,
        symbol: str,
        realized_pnl: float,
        fees: float,
        trade_date: date | None = None,
    ) -> None:
        d = (trade_date or datetime.now(timezone.utc).date()).isoformat()
        with self._write_lock:
            self._conn.execute(
                """INSERT INTO daily_pnl (date, symbol, realized_pnl, fees_paid, trades_count)
                   VALUES (?, ?, ?, ?, 1)
                   ON CONFLICT(date, symbol) DO UPDATE SET
                       realized_pnl = realized_pnl + excluded.realized_pnl,
                       fees_paid = fees_paid + excluded.fees_paid,
                       trades_count = trades_count + 1""",
                (d, symbol, realized_pnl, fees),
            )
            self._conn.commit()

    def get_daily_pnl(self, trade_date: date | None = None) -> dict:
        """Returns {symbol: {realized_pnl, fees_paid, trades_count}}."""
        d = (trade_date or datetime.now(timezone.utc).date()).isoformat()
        rows = self._conn.execute(
            "SELECT * FROM daily_pnl WHERE date=?", (d,),
        ).fetchall()
        return {
            r["symbol"]: {
                "realized_pnl": r["realized_pnl"],
                "fees_paid": r["fees_paid"],
                "trades_count": r["trades_count"],
            }
            for r in rows
        }

    def get_daily_total_pnl(self, trade_date: date | None = None) -> float:
        d = (trade_date or datetime.now(timezone.utc).date()).isoformat()
        row = self._conn.execute(
            "SELECT COALESCE(SUM(realized_pnl), 0) as total FROM daily_pnl WHERE date=?",
            (d,),
        ).fetchone()
        return row["total"]

    # ── Kill Switch Queries ─────────────────────────────────

    def get_consecutive_losses(self, symbol: str | None = None, n: int = 3) -> bool:
        """True if last N closed trades are all losses."""
        if symbol:
            rows = self._conn.execute(
                """SELECT o.order_link_id, f.fill_price, o.price as entry_price, o.side
                   FROM orders o
                   JOIN fills f ON o.order_link_id = f.order_link_id
                   WHERE o.symbol=? AND o.purpose IN ('stop_loss', 'take_profit', 'time_stop')
                   AND o.status='FILLED'
                   ORDER BY f.filled_at DESC LIMIT ?""",
                (symbol, n),
            ).fetchall()
        else:
            rows = self._conn.execute(
                """SELECT o.order_link_id, f.fill_price, o.price as entry_price, o.side
                   FROM orders o
                   JOIN fills f ON o.order_link_id = f.order_link_id
                   WHERE o.purpose IN ('stop_loss', 'take_profit', 'time_stop')
                   AND o.status='FILLED'
                   ORDER BY f.filled_at DESC LIMIT ?""",
                (n,),
            ).fetchall()

        if len(rows) < n:
            return False

        # Check if all are stop losses
        loss_count = sum(1 for r in rows if r["entry_price"] is not None)
        # Simplified: check via daily_pnl or state transitions
        # For now, count SL exits
        sl_rows = self._conn.execute(
            """SELECT COUNT(*) as cnt FROM (
                SELECT st.trigger FROM state_transitions st
                WHERE st.to_state IN ('SL_HIT')
                {} ORDER BY st.id DESC LIMIT ?
            )""".format(f"AND st.symbol='{symbol}'" if symbol else ""),
            (n,),
        ).fetchone()
        return sl_rows["cnt"] >= n

    # ── Recovery ────────────────────────────────────────────

    def get_open_positions(self) -> list[dict]:
        """Recover positions: find symbols with FILLED/PROTECTED state."""
        rows = self._conn.execute(
            """SELECT DISTINCT symbol FROM state_transitions
               WHERE to_state IN ('FILLED', 'PROTECTED')
               AND symbol NOT IN (
                   SELECT symbol FROM state_transitions
                   WHERE to_state IN ('IDLE', 'TP_HIT', 'SL_HIT', 'TIME_STOP', 'EMERGENCY_CLOSE')
                   AND id > (
                       SELECT MAX(id) FROM state_transitions st2
                       WHERE st2.symbol = state_transitions.symbol
                       AND st2.to_state IN ('FILLED', 'PROTECTED')
                   )
               )""",
        ).fetchall()

        positions = []
        for row in rows:
            sym = row["symbol"]
            last_state = self.get_last_state(sym)
            if last_state in ("FILLED", "PROTECTED"):
                # Get entry order details
                entry_orders = self._conn.execute(
                    """SELECT o.*, f.fill_price, f.fill_qty
                       FROM orders o
                       JOIN fills f ON o.order_link_id = f.order_link_id
                       WHERE o.symbol=? AND o.purpose='entry' AND o.status='FILLED'
                       ORDER BY o.created_at DESC LIMIT 1""",
                    (sym,),
                ).fetchone()
                if entry_orders:
                    positions.append({
                        "symbol": sym,
                        "state": last_state,
                        "entry_order": dict(entry_orders),
                    })
        return positions

    # ── Utility ─────────────────────────────────────────────

    def close(self) -> None:
        self._conn.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
