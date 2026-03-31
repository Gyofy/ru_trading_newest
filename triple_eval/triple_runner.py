"""TripleRunner — 3가지 모드를 서브프로세스로 동시 실행 + 주기적 비교."""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

logger = logging.getLogger("triple_runner")

STATE_DIRS = {
    "paper": "data/reports/triple_paper",
    "backtest": "data/reports/triple_backtest",
    "demo": "data/reports/triple_demo",
}


class TripleRunner:
    """Launch paper, demo, and backtest bots as subprocesses and periodically compare."""

    def __init__(
        self,
        paper_config: str = "config/triple_paper.yaml",
        demo_config: str = "config/multi_strategy.yaml",
        backtest_config: str = "config/triple_backtest.yaml",
        discord_webhook: str = "",
        comparison_interval_sec: int = 1800,  # 30분
        initial_equity: float = 5000.0,
    ) -> None:
        self.paper_config = paper_config
        self.demo_config = demo_config
        self.backtest_config = backtest_config
        self.discord_webhook = discord_webhook
        self.comparison_interval_sec = comparison_interval_sec
        self.initial_equity = initial_equity

        self._procs: dict[str, subprocess.Popen] = {}
        self._python = sys.executable
        self._project_root = str(Path(__file__).parent.parent.resolve())

    def launch_all(self) -> None:
        """Launch all three subprocesses."""
        logger.info("[TripleRunner] Launching all 3 modes...")

        self._procs["paper"] = self._launch_paper()
        logger.info(f"[TripleRunner] Paper PID={self._procs['paper'].pid}")

        self._procs["demo"] = self._launch_demo()
        logger.info(f"[TripleRunner] Demo PID={self._procs['demo'].pid}")

        self._procs["backtest"] = self._launch_backtest()
        logger.info(f"[TripleRunner] Backtest PID={self._procs['backtest'].pid}")

    def _open_log(self, mode: str):
        """Open a log file for subprocess stdout/stderr (append mode)."""
        log_dir = Path(self._project_root) / "data" / "logs"
        log_dir.mkdir(parents=True, exist_ok=True)
        log_path = log_dir / f"triple_{mode}.log"
        return open(log_path, "a", encoding="utf-8")  # noqa: SIM115

    def _launch_paper(self) -> subprocess.Popen:
        """Launch paper mode bot."""
        env = os.environ.copy()
        env["TRIPLE_STATE_DIR"] = STATE_DIRS["paper"]
        log_fh = self._open_log("paper")
        return subprocess.Popen(
            [
                self._python,
                os.path.join(self._project_root, "run_multi_strategy.py"),
                "--mode", "paper",
                "--config", self.paper_config,
            ],
            cwd=self._project_root,
            env=env,
            stdout=log_fh,
            stderr=log_fh,
        )

    def _launch_demo(self) -> subprocess.Popen:
        """Launch demo mode bot."""
        env = os.environ.copy()
        env["TRIPLE_STATE_DIR"] = STATE_DIRS["demo"]
        log_fh = self._open_log("demo")
        return subprocess.Popen(
            [
                self._python,
                os.path.join(self._project_root, "run_multi_strategy.py"),
                "--mode", "demo",
                "--config", self.demo_config,
            ],
            cwd=self._project_root,
            env=env,
            stdout=log_fh,
            stderr=log_fh,
        )

    def _launch_backtest(self) -> subprocess.Popen:
        """Launch backtest mode."""
        env = os.environ.copy()
        env["TRIPLE_STATE_DIR"] = STATE_DIRS["backtest"]
        log_fh = self._open_log("backtest")
        return subprocess.Popen(
            [
                self._python,
                os.path.join(self._project_root, "run_triple_eval.py"),
                "--mode", "backtest",
                "--backtest-config", self.backtest_config,
            ],
            cwd=self._project_root,
            env=env,
            stdout=log_fh,
            stderr=log_fh,
        )

    def monitor_and_compare(self) -> None:
        """Main loop: check process health and run periodic comparison."""
        last_compare_ts = time.time()
        logger.info(
            f"[TripleRunner] Monitoring processes. "
            f"Comparison every {self.comparison_interval_sec}s."
        )

        try:
            while True:
                # Check health of each process
                for mode, proc in list(self._procs.items()):
                    ret = proc.poll()
                    if ret is not None:
                        logger.warning(
                            f"[TripleRunner] {mode} process exited with code {ret}"
                        )

                # Periodic comparison
                now = time.time()
                if now - last_compare_ts >= self.comparison_interval_sec:
                    try:
                        self._run_comparison()
                    except Exception as e:
                        logger.error(f"[TripleRunner] Comparison failed: {e}")
                    last_compare_ts = now

                time.sleep(30)

        except KeyboardInterrupt:
            logger.info("[TripleRunner] Interrupted — shutting down")
            self.shutdown()

    def _run_comparison(self) -> None:
        """Run TripleComparator and log results."""
        from triple_eval.triple_comparator import TripleComparator
        comparator = TripleComparator(
            state_dirs=STATE_DIRS,
            discord_webhook=self.discord_webhook,
            lookback_hours=24,
            initial_equity=self.initial_equity,
        )
        results = comparator.run_and_post()
        logger.info(
            f"[TripleRunner] Comparison done: "
            + " | ".join(
                f"{mode}={metrics.get('total_trades',0)} trades "
                f"WR={metrics.get('win_rate',0):.1%}"
                for mode, metrics in results.items()
            )
        )

    def shutdown(self) -> None:
        """Send SIGTERM to all subprocesses."""
        for mode, proc in self._procs.items():
            if proc.poll() is None:
                logger.info(f"[TripleRunner] Terminating {mode} (PID={proc.pid})")
                try:
                    proc.send_signal(signal.SIGTERM)
                except Exception as e:
                    logger.warning(f"[TripleRunner] Could not terminate {mode}: {e}")

        # Wait for clean shutdown
        for mode, proc in self._procs.items():
            try:
                proc.wait(timeout=10)
                logger.info(f"[TripleRunner] {mode} stopped cleanly")
            except subprocess.TimeoutExpired:
                logger.warning(f"[TripleRunner] {mode} SIGKILL (timeout)")
                proc.kill()
