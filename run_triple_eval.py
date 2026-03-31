#!/usr/bin/env python3
"""Triple Evaluation System — paper + backtest + demo 동시 실행 및 비교."""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

# Ensure project root is in path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main() -> None:
    parser = argparse.ArgumentParser(description="Triple Evaluation System")
    parser.add_argument(
        "--mode",
        choices=["all", "paper", "demo", "backtest", "compare"],
        default="all",
        help="Which mode to run",
    )
    parser.add_argument(
        "--paper-config",
        default="config/triple_paper.yaml",
        help="Config file for paper mode",
    )
    parser.add_argument(
        "--demo-config",
        default="config/multi_strategy.yaml",
        help="Config file for demo mode",
    )
    parser.add_argument(
        "--backtest-config",
        default="config/triple_backtest.yaml",
        help="Config file for backtest mode",
    )
    parser.add_argument(
        "--compare-only",
        action="store_true",
        help="Only run comparator (skip launching bots)",
    )
    parser.add_argument(
        "--lookback-hours",
        type=int,
        default=24,
        help="Lookback window in hours for comparison",
    )
    args = parser.parse_args()

    if args.mode == "compare" or args.compare_only:
        # Just run the comparator
        from triple_eval.triple_comparator import TripleComparator
        webhook = os.environ.get("DISCORD_WEBHOOK_URL", "")
        state_dirs = {
            "paper": "data/reports/triple_paper",
            "backtest": "data/reports/triple_backtest",
            "demo": "data/reports/triple_demo",
        }
        c = TripleComparator(state_dirs, webhook, args.lookback_hours)
        result = c.run_and_post()
        import json
        print(json.dumps(result, indent=2, ensure_ascii=False))

    elif args.mode == "backtest":
        # Run only backtest
        os.environ["TRIPLE_STATE_DIR"] = "data/reports/triple_backtest"
        from triple_eval.backtest_engine import BacktestBot
        bot = BacktestBot(args.backtest_config)
        asyncio.run(bot.start())

    elif args.mode == "paper":
        os.environ["TRIPLE_STATE_DIR"] = "data/reports/triple_paper"
        # Launch paper mode via run_multi_strategy
        import subprocess
        env = os.environ.copy()
        subprocess.run(
            [sys.executable, "run_multi_strategy.py", "--mode", "paper", "--config", args.paper_config],
            env=env,
        )

    elif args.mode == "demo":
        os.environ["TRIPLE_STATE_DIR"] = "data/reports/triple_demo"
        import subprocess
        env = os.environ.copy()
        subprocess.run(
            [sys.executable, "run_multi_strategy.py", "--mode", "demo", "--config", args.demo_config],
            env=env,
        )

    elif args.mode == "all":
        # Launch all three simultaneously
        from triple_eval.triple_runner import TripleRunner
        webhook = os.environ.get("DISCORD_WEBHOOK_URL", "")
        runner = TripleRunner(
            paper_config=args.paper_config,
            demo_config=args.demo_config,
            backtest_config=args.backtest_config,
            discord_webhook=webhook,
        )
        runner.launch_all()
        runner.monitor_and_compare()


if __name__ == "__main__":
    main()
