"""Live Engine Launcher — 실전 매매 엔진 실행.

Usage:
    # Sandbox mode (기본, Binance Testnet)
    python run_live_engine.py

    # With API keys (환경변수)
    BINANCE_API_KEY=xxx BINANCE_API_SECRET=yyy python run_live_engine.py

    # Live mode (주의!)
    python run_live_engine.py --mode live
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import argparse
import asyncio
import logging
import os

from src.execution.live_engine import LiveEngine, LiveEngineConfig

PROJECT_ROOT = Path(__file__).parent


def setup_logging():
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_dir / "live_engine.log", encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )


def load_config(mode: str) -> LiveEngineConfig:
    """Load config from settings.yaml + environment."""
    import yaml

    config_path = PROJECT_ROOT / "config" / "settings.yaml"
    with open(config_path) as f:
        settings = yaml.safe_load(f)

    exec_settings = settings.get("execution_live", {})
    exec_settings["mode"] = mode
    exec_settings["exchange"] = "binance"
    exec_settings["api_key"] = os.environ.get("BINANCE_API_KEY", "")
    exec_settings["api_secret"] = os.environ.get("BINANCE_API_SECRET", "")

    return LiveEngineConfig(exec_settings)


async def main(mode: str):
    setup_logging()
    logger = logging.getLogger(__name__)

    logger.info("=" * 60)
    logger.info(f"  Live Engine Starting (mode={mode}, exchange=binance)")
    logger.info("=" * 60)

    if mode == "live":
        logger.warning("=" * 60)
        logger.warning("  *** LIVE MODE — REAL MONEY AT RISK ***")
        logger.warning("=" * 60)
        confirm = input("Type 'YES' to confirm live trading: ")
        if confirm != "YES":
            logger.info("Aborted.")
            return

    config = load_config(mode)
    engine = LiveEngine(config)

    try:
        await engine.start()
        await engine.run_forever()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        await engine.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Live Trading Engine")
    parser.add_argument("--mode", default="sandbox", choices=["sandbox", "live"])
    args = parser.parse_args()

    asyncio.run(main(args.mode))
