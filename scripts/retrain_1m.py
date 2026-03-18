#!/usr/bin/env python3
"""1분봉 병렬 재학습 스크립트.

사용법:
    python scripts/retrain_1m.py [--coins BTC ETH SOL ...] [--workers 4] [--dry-run]

동작:
    1. Binance USDT-M Futures에서 1m OHLCV 데이터를 병렬로 수집 (pagination 포함)
    2. 기술 피처 + 시그널 피처 + 마이크로스트럭처 피처 계산
    3. 코인별 2-Stage Binary 모델을 multiprocessing.Pool로 병렬 학습
    4. data/models/live_v4_3_1m/{coin}_combo.pkl 에 저장

설정:
    - config/settings_1m.yaml   (bar_minutes=1, horizons=[30,60,120,240])
    - config/frozen_params_v4_3_1m.yaml  (k_upper=3.0, max_horizon=60, ...)

주의:
    - BINANCE_API_KEY / BINANCE_API_SECRET 환경변수 필요
    - SETTINGS_YAML_PATH 가 이미 설정된 경우 해당 경로 사용
"""

from __future__ import annotations

# ── CRITICAL: set SETTINGS_YAML_PATH BEFORE any src imports ──
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

# workers 도 이 환경변수를 읽어서 settings_1m.yaml 을 사용
_SETTINGS_1M = str(PROJECT_ROOT / "config" / "settings_1m.yaml")
if not os.environ.get("SETTINGS_YAML_PATH"):
    os.environ["SETTINGS_YAML_PATH"] = _SETTINGS_1M

# ── Standard Imports ─────────────────────────────────────────
import argparse
import asyncio
import logging
import multiprocessing as mp
import time
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd
import yaml
from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

# ── Project Imports (AFTER env var is set) ────────────────────
from src.data.crawlers.crypto_ohlcv import (
    add_technical_indicators, _add_decomposition, _add_cross_asset_correlation,
)
from src.data.crawlers.signal_features import add_signal_features
from src.data.crawlers.microstructure_rollup import add_microstructure_rollup
from src.execution.live_predictor import train_combo, save_combo
from src.utils.feature_policy import is_excluded_feature

# ── Logging ──────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("retrain_1m")

# ── Config ───────────────────────────────────────────────────
FROZEN_PATH = PROJECT_ROOT / "config" / "frozen_params_v4_3_1m.yaml"

SYMBOL_MAP = {
    "BTC": "BTC/USDT:USDT",
    "ETH": "ETH/USDT:USDT",
    "SOL": "SOL/USDT:USDT",
    "XRP": "XRP/USDT:USDT",
    "ADA": "ADA/USDT:USDT",
    "DOT": "DOT/USDT:USDT",
    "LINK": "LINK/USDT:USDT",
    "DOGE": "DOGE/USDT:USDT",
    "BNB": "BNB/USDT:USDT",
    "AVAX": "AVAX/USDT:USDT",
    "TAO": "TAO/USDT:USDT",
}

DEFAULT_COINS = ["BTC", "ETH", "SOL", "XRP", "ADA", "DOT", "LINK", "DOGE"]


def load_frozen() -> dict:
    with open(FROZEN_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


# ══════════════════════════════════════════════════════════
# PHASE 1: 1m OHLCV 수집 (async, pagination)
# ══════════════════════════════════════════════════════════

async def _fetch_1m_ohlcv_paged(
    exchange,
    symbol: str,
    ccxt_sym: str,
    n_bars: int,
    limit_per_req: int = 1500,
) -> Optional[pd.DataFrame]:
    """Binance max 1500 bars/request → 여러 번 요청해 n_bars 수집."""
    all_ohlcv = []
    end_ts = None  # None = latest, then work backwards

    remaining = n_bars
    for _ in range((n_bars // limit_per_req) + 2):  # max pages
        if remaining <= 0:
            break
        batch_limit = min(remaining, limit_per_req)
        try:
            kwargs = {"timeframe": "1m", "limit": batch_limit}
            if end_ts is not None:
                kwargs["params"] = {"endTime": end_ts}
            batch = await exchange.fetch_ohlcv(ccxt_sym, **kwargs)
            if not batch:
                break
            all_ohlcv = batch + all_ohlcv  # prepend older data
            end_ts = batch[0][0] - 1       # next page: fetch before this timestamp
            remaining -= len(batch)
            await asyncio.sleep(0.2)       # Binance rate limit margin
        except Exception as e:
            logger.warning(f"[Fetch] {symbol}: {e}")
            break

    if not all_ohlcv:
        return None

    df = pd.DataFrame(all_ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.drop_duplicates("timestamp", inplace=True)
    df.sort_values("timestamp", inplace=True)
    df.set_index("timestamp", inplace=True)
    df = df.astype(float)
    # Keep most recent n_bars
    if len(df) > n_bars:
        df = df.iloc[-n_bars:]
    logger.info(f"[Fetch] {symbol}: {len(df)} 1m bars (${df['close'].iloc[-1]:,.4f})")
    return df


_FETCH_SEM = None


async def _limited_fetch(exchange, symbol: str, n_bars: int, limit_per_req: int):
    global _FETCH_SEM
    if _FETCH_SEM is None:
        _FETCH_SEM = asyncio.Semaphore(3)
    ccxt_sym = SYMBOL_MAP.get(symbol, f"{symbol}/USDT:USDT")
    async with _FETCH_SEM:
        for attempt in range(3):
            df = await _fetch_1m_ohlcv_paged(exchange, symbol, ccxt_sym, n_bars, limit_per_req)
            if df is not None and len(df) >= 500:
                return symbol, df
            logger.warning(f"[Fetch] {symbol} attempt {attempt+1} got only {len(df) if df is not None else 0} bars")
            await asyncio.sleep(2 ** attempt)
    return symbol, None


async def fetch_all_1m(
    coins: list[str],
    api_key: str,
    api_secret: str,
    mode: str,
    n_bars: int,
    limit_per_req: int,
) -> dict[str, pd.DataFrame]:
    """모든 코인 1m OHLCV 병렬 수집."""
    try:
        import ccxt.async_support as ccxt_async
    except ImportError:
        raise ImportError("ccxt 필요: pip install ccxt")

    fetch_coins = list(set(coins + ["BTC", "ETH"]))

    exchange_cfg = {
        "apiKey": api_key,
        "secret": api_secret,
        "enableRateLimit": True,
        "options": {
            "defaultType": "future",
            "adjustForTimeDifference": True,   # WSL2 clock drift fix
            "recvWindow": 10000,
        },
    }
    if mode == "sandbox":
        exchange_cfg["urls"] = {"api": {"public": "https://testnet.binancefuture.com"}}

    exchange = ccxt_async.binanceusdm(exchange_cfg)

    try:
        tasks = [_limited_fetch(exchange, coin, n_bars, limit_per_req) for coin in fetch_coins]
        results = await asyncio.gather(*tasks)
    finally:
        await exchange.close()

    raw = {}
    for coin, df in results:
        if df is not None and not df.empty:
            raw[coin] = df
        else:
            logger.error(f"[Fetch] {coin}: FAILED — skipped")
    return raw


# ══════════════════════════════════════════════════════════
# PHASE 2: 피처 엔지니어링
# ══════════════════════════════════════════════════════════

def compute_features_1m(raw_data: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """1m 데이터에 피처 계산. decomposition period를 120으로 조정."""
    featured = {}
    all_coins = list(raw_data.keys())

    for coin, df in raw_data.items():
        if len(df) < 200:
            logger.warning(f"[Features] {coin}: too few bars ({len(df)}) — skip")
            continue
        try:
            df = df.copy()
            df = add_technical_indicators(df)
            df = _add_decomposition(df, period=120)   # 2h cycle for 1m bars
            df = add_signal_features(df, verbose=False)
            df = add_microstructure_rollup(df, verbose=False)
            df.ffill(inplace=True)
            df.replace([np.inf, -np.inf], np.nan, inplace=True)
            df.fillna(0, inplace=True)
            cols_drop = [c for c in df.columns if is_excluded_feature(c)]
            if cols_drop:
                df.drop(columns=cols_drop, inplace=True, errors="ignore")
            featured[coin] = df
            logger.info(f"[Features] {coin}: {len(df)} bars, {len(df.columns)} features")
        except Exception as e:
            logger.error(f"[Features] {coin}: {e}", exc_info=True)

    # Cross-asset correlation (BTC/ETH as reference)
    if len(featured) >= 2:
        for coin in all_coins:
            if coin in featured:
                try:
                    featured[coin] = _add_cross_asset_correlation(featured, coin, window=20)
                except Exception:
                    pass

    return featured


# ══════════════════════════════════════════════════════════
# PHASE 3: 병렬 학습 (multiprocessing worker)
# ══════════════════════════════════════════════════════════

def _train_worker(args: tuple) -> dict:
    """Multiprocessing worker: 단일 코인 학습 + 저장.

    각 worker는 별도 프로세스 → SETTINGS_YAML_PATH 를 재설정해야 함.
    """
    coin, df_bytes, common_cfg, coin_cfg, artifact_dir, settings_path, dry_run = args

    # Restore env var in worker process (fork inherits, spawn does not)
    os.environ["SETTINGS_YAML_PATH"] = settings_path

    # Re-import inside worker to pick up fresh settings cache
    import importlib
    import src.utils.config as _cfg_mod
    _cfg_mod._settings_cache = None  # invalidate cache in this process

    from src.execution.live_predictor import train_combo as _train, save_combo as _save

    import pickle
    df = pickle.loads(df_bytes)

    t0 = time.time()
    result = {"coin": coin, "success": False, "cv_score": 0.0, "samples": 0, "elapsed": 0.0}

    try:
        logger.info(f"[Worker] {coin}: starting training ({len(df)} bars)")
        combo = _train(df, coin, common_cfg, coin_cfg)
        if combo is None:
            result["error"] = "train_combo returned None"
            return result

        result["cv_score"] = combo.cv_score
        result["samples"] = combo.train_samples

        if not dry_run:
            out_dir = Path(artifact_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            _save(combo, out_dir)
            logger.info(
                f"[Worker] {coin}: saved to {out_dir / coin}_combo.pkl "
                f"(CV={combo.cv_score:.4f}, n={combo.train_samples})"
            )

        result["success"] = True
    except Exception as e:
        result["error"] = str(e)
        logger.error(f"[Worker] {coin}: {e}", exc_info=True)
    finally:
        result["elapsed"] = round(time.time() - t0, 1)

    return result


def train_all_parallel(
    featured: dict[str, pd.DataFrame],
    coins: list[str],
    frozen: dict,
    n_workers: int,
    dry_run: bool,
) -> list[dict]:
    """모든 코인을 multiprocessing.Pool로 병렬 학습."""
    import pickle

    common_cfg = frozen["common"]
    artifact_dir = PROJECT_ROOT / frozen["data_fetch"]["artifact_dir"]
    settings_path = os.environ.get("SETTINGS_YAML_PATH", _SETTINGS_1M)

    tasks = []
    for coin in coins:
        df = featured.get(coin)
        if df is None:
            logger.warning(f"[Train] {coin}: no featured data — skipped")
            continue
        coin_cfg = frozen.get("coins", {}).get(coin, {})
        # Default combo fallback
        if "model_combo" not in coin_cfg:
            coin_cfg = dict(coin_cfg, model_combo="et+cb", model_weights={"et": 0.5, "cb": 0.5})

        df_bytes = pickle.dumps(df)
        tasks.append((coin, df_bytes, common_cfg, coin_cfg, str(artifact_dir), settings_path, dry_run))

    if not tasks:
        logger.error("[Train] No coins to train!")
        return []

    actual_workers = min(n_workers, len(tasks), mp.cpu_count())
    logger.info(f"[Train] Launching {len(tasks)} coin jobs across {actual_workers} workers")

    ctx = mp.get_context("spawn")   # spawn is safer than fork with GPU/BLAS
    with ctx.Pool(processes=actual_workers) as pool:
        results = pool.map(_train_worker, tasks)

    return results


# ══════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="1분봉 병렬 재학습")
    p.add_argument("--coins", nargs="+", default=DEFAULT_COINS,
                   help=f"학습 코인 리스트 (기본값: {DEFAULT_COINS})")
    p.add_argument("--workers", type=int, default=4,
                   help="병렬 학습 프로세스 수 (기본값: 4)")
    p.add_argument("--n-bars", type=int, default=None,
                   help="수집할 1m 봉 수 (기본값: frozen_params 값 사용)")
    p.add_argument("--mode", choices=["sandbox", "live"], default="sandbox",
                   help="Binance 접속 모드 (기본값: sandbox)")
    p.add_argument("--dry-run", action="store_true",
                   help="학습만 하고 저장은 하지 않음")
    return p.parse_args()


def main():
    args = parse_args()
    frozen = load_frozen()

    n_bars = args.n_bars or frozen["data_fetch"]["n_bars"]
    limit_per_req = frozen["data_fetch"]["binance_limit_per_request"]
    artifact_dir = PROJECT_ROOT / frozen["data_fetch"]["artifact_dir"]

    api_key = os.environ.get("BINANCE_API_KEY", "")
    api_secret = os.environ.get("BINANCE_API_SECRET", "")

    if not api_key or not api_secret:
        logger.error("BINANCE_API_KEY / BINANCE_API_SECRET 환경변수 없음 — .env 확인")
        sys.exit(1)

    logger.info(
        f"===== 1m 재학습 시작 =====\n"
        f"  코인: {args.coins}\n"
        f"  bars: {n_bars} (limit/req: {limit_per_req})\n"
        f"  workers: {args.workers}\n"
        f"  mode: {args.mode}\n"
        f"  dry_run: {args.dry_run}\n"
        f"  artifact_dir: {artifact_dir}\n"
        f"  settings: {os.environ.get('SETTINGS_YAML_PATH')}"
    )

    # ── Phase 1: Fetch ───────────────────────────────────────
    t_fetch = time.time()
    logger.info(f"[Phase 1] 1m OHLCV 수집 중 ({n_bars} bars per coin)...")
    raw_data = asyncio.run(fetch_all_1m(
        coins=args.coins,
        api_key=api_key,
        api_secret=api_secret,
        mode=args.mode,
        n_bars=n_bars,
        limit_per_req=limit_per_req,
    ))
    logger.info(f"[Phase 1] 완료: {len(raw_data)}/{len(args.coins)} 코인 ({time.time()-t_fetch:.0f}s)")

    if not raw_data:
        logger.error("데이터 수집 실패 — 종료")
        sys.exit(1)

    # ── Phase 2: Features ────────────────────────────────────
    t_feat = time.time()
    logger.info("[Phase 2] 피처 계산 중...")
    featured = compute_features_1m(raw_data)
    logger.info(f"[Phase 2] 완료: {len(featured)} 코인 ({time.time()-t_feat:.0f}s)")

    # ── Phase 3: Train ───────────────────────────────────────
    t_train = time.time()
    train_coins = [c for c in args.coins if c in featured]
    logger.info(f"[Phase 3] 병렬 학습 시작: {train_coins}")

    results = train_all_parallel(
        featured=featured,
        coins=train_coins,
        frozen=frozen,
        n_workers=args.workers,
        dry_run=args.dry_run,
    )
    logger.info(f"[Phase 3] 완료 ({time.time()-t_train:.0f}s)")

    # ── Summary ──────────────────────────────────────────────
    total_elapsed = time.time() - t_fetch
    success = [r for r in results if r.get("success")]
    failed = [r for r in results if not r.get("success")]

    print("\n" + "="*55)
    print(f"  1m 재학습 완료 ({total_elapsed:.0f}s)")
    print("="*55)
    print(f"  {'코인':<6} {'CV Score':>10} {'Samples':>9} {'Time':>7} {'Status'}")
    print("-"*55)
    for r in sorted(results, key=lambda x: x["coin"]):
        status = "OK" if r.get("success") else f"FAIL: {r.get('error','?')[:25]}"
        print(f"  {r['coin']:<6} {r.get('cv_score',0):>10.4f} {r.get('samples',0):>9} {r.get('elapsed',0):>6.0f}s  {status}")
    print("="*55)
    print(f"  성공: {len(success)}/{len(results)} 코인")
    if failed:
        print(f"  실패: {[r['coin'] for r in failed]}")
    if not args.dry_run:
        print(f"  저장 위치: {artifact_dir}")
    print("="*55 + "\n")

    # Exit code: 0 if all succeeded, 1 if any failed
    sys.exit(0 if not failed else 1)


if __name__ == "__main__":
    main()
