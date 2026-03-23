"""Paper Bot Monitor — 상태 조회, 포지션, PnL, 헬스체크.

Usage:
  python run_monitor.py              # 전체 대시보드
  python run_monitor.py --check      # 헬스체크만 (프로세스 살아있는지)
  python run_monitor.py --restart    # 죽어있으면 재시작
"""

import sys, os, json, subprocess, argparse
from pathlib import Path
from datetime import datetime, timezone

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

STATE_DIR = Path("data/reports/tsmom_paper")
STATE_FILE = STATE_DIR / "state.json"
TRADES_FILE = STATE_DIR / "trades.jsonl"
LOG_FILE = STATE_DIR / "bot.log"
SIGNAL_LOG = STATE_DIR / "signal_log.jsonl"


def load_state() -> dict:
    if STATE_FILE.exists():
        with open(STATE_FILE) as f:
            return json.load(f)
    return {}


def load_trades() -> list[dict]:
    trades = []
    if TRADES_FILE.exists():
        with open(TRADES_FILE) as f:
            for line in f:
                line = line.strip()
                if line:
                    trades.append(json.loads(line))
    return trades


def count_signals() -> int:
    if SIGNAL_LOG.exists():
        with open(SIGNAL_LOG) as f:
            return sum(1 for _ in f)
    return 0


def is_bot_running() -> bool:
    try:
        result = subprocess.run(
            ["pgrep", "-f", "run_tsmom_paper"],
            capture_output=True, text=True, timeout=5
        )
        return result.returncode == 0
    except Exception:
        # Windows fallback
        try:
            result = subprocess.run(
                ["tasklist", "/FI", "IMAGENAME eq python.exe"],
                capture_output=True, text=True, timeout=5
            )
            return "run_tsmom_paper" in result.stdout
        except Exception:
            return False


def print_dashboard():
    state = load_state()
    trades = load_trades()

    print("=" * 70)
    print("  TSMOM v5.1 Paper Bot Monitor")
    print("=" * 70)

    # Bot status
    running = is_bot_running()
    status = "RUNNING" if running else "STOPPED"
    updated = state.get("updated", "unknown")
    print(f"\n  Status:     {status}")
    print(f"  Last update: {updated}")

    # Equity
    equity = state.get("equity", 0)
    initial = 1000.0
    pnl_pct = (equity - initial) / initial * 100
    print(f"  Equity:     ${equity:.2f} ({pnl_pct:+.1f}%)")

    # Positions
    positions = state.get("positions", {})
    print(f"\n  Open Positions: {len(positions)}")
    for coin, pos in positions.items():
        side = "LONG" if pos["side"] == 1 else "SHORT"
        entry = pos["entry_price"]
        tp = pos["tp_price"]
        sl = pos["sl_price"]
        bars = pos["bars_held"]
        print(f"    {coin:5s} {side:5s} @ ${entry:.4f} | TP=${tp:.4f} SL=${sl:.4f} | bars={bars}")

    # Trade history
    print(f"\n  Completed Trades: {len(trades)}")
    if trades:
        pnls = [t["pnl_net"] for t in trades]
        wins = sum(1 for p in pnls if p > 0)
        wr = wins / len(pnls) * 100
        avg = sum(pnls) / len(pnls) * 100
        total = sum(pnls) * 100

        print(f"    WR: {wr:.1f}% ({wins}/{len(pnls)})")
        print(f"    Avg PnL: {avg:+.2f}%")
        print(f"    Total PnL: {total:+.2f}%")

        # Last 5 trades
        print(f"\n    Recent trades:")
        for t in trades[-5:]:
            side = t["side"]
            coin = t["coin"]
            pnl = t["pnl_net"] * 100
            exit_t = t["exit_type"]
            bars = t["bars_held"]
            print(f"      {coin:5s} {side:5s} {pnl:+6.2f}% ({exit_t}, {bars} bars)")

    # Signal log
    n_signals = count_signals()
    print(f"\n  RL Signal Log: {n_signals} entries")
    print(f"  RL Training:   {'READY' if n_signals >= 200 else f'Need {200 - n_signals} more'}")

    # Log tail
    if LOG_FILE.exists():
        print(f"\n  Last 5 log lines:")
        with open(LOG_FILE, encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
            for line in lines[-5:]:
                print(f"    {line.rstrip()}")

    print("\n" + "=" * 70)


def healthcheck():
    running = is_bot_running()
    state = load_state()
    updated = state.get("updated", "")

    if not running:
        print("[ALERT] Bot is NOT running!")
        return False

    # Check staleness
    if updated:
        try:
            last = datetime.fromisoformat(updated)
            now = datetime.now(timezone.utc)
            age_min = (now - last).total_seconds() / 60
            if age_min > 300:  # 5 hours stale
                print(f"[WARN] State is {age_min:.0f}min old (>300min)")
                return False
        except Exception:
            pass

    print("[OK] Bot running, state fresh")
    return True


def restart_bot():
    print("Stopping old instance...")
    os.system('pkill -f "run_tsmom_paper" 2>/dev/null')
    import time
    time.sleep(2)
    print("Starting new instance...")
    os.system('nohup python -X utf8 run_tsmom_paper.py > data/reports/tsmom_paper/nohup.log 2>&1 &')
    time.sleep(3)
    if is_bot_running():
        print("[OK] Bot restarted")
    else:
        print("[FAIL] Bot failed to start")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TSMOM Paper Bot Monitor")
    parser.add_argument("--check", action="store_true", help="Healthcheck only")
    parser.add_argument("--restart", action="store_true", help="Restart if stopped")
    args = parser.parse_args()

    if args.check:
        ok = healthcheck()
        sys.exit(0 if ok else 1)
    elif args.restart:
        if not healthcheck():
            restart_bot()
    else:
        print_dashboard()
