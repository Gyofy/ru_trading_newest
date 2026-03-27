"""
Binance Futures 최대 데이터 샘플링 스크립트
=============================================
실제 트레이드가 발생한 코인들에 대해 Binance 공개 REST API로
가능한 최대 데이터를 수집해서 저장한다.

수집 데이터:
  1. OHLCV klines       — 1m, 5m, 15m, 1h, 4h
  2. Mark price klines  — 15m, 1h
  3. Funding rate       — 8시간 간격
  4. Open interest      — 5m, 1h
  5. L/S Account ratio  — 5m, 1h
  6. Taker buy/sell vol — 5m, 1h

대상 심볼: trades.jsonl에 기록된 모든 코인
기간: 2026-03-15 ~ 오늘

저장 경로: data/raw/binance/{symbol}/{type}_{interval}.csv
"""

import argparse
import time
import json
import logging
import pandas as pd
import numpy as np
import requests
from pathlib import Path
from datetime import datetime, timezone, timedelta

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger(__name__)

FAPI = "https://fapi.binance.com"
SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "ru_trading_data_sampler/1.0"})


def _get(endpoint: str, params: dict, retries: int = 5) -> list | dict:
    """Binance REST GET with retry."""
    url = FAPI + endpoint
    for attempt in range(retries):
        try:
            r = SESSION.get(url, params=params, timeout=20)
            if r.status_code == 429:
                wait = int(r.headers.get("Retry-After", 10))
                log.warning(f"Rate limit — sleeping {wait}s")
                time.sleep(wait)
                continue
            r.raise_for_status()
            return r.json()
        except Exception as e:
            if attempt == retries - 1:
                raise
            log.warning(f"Retry {attempt+1}/{retries}: {e}")
            time.sleep(2 ** attempt)
    return []


def _to_ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def fetch_klines(symbol: str, interval: str, start: datetime, end: datetime) -> pd.DataFrame:
    """OHLCV klines — 1500건씩 페이징."""
    rows = []
    cursor = _to_ms(start)
    end_ms = _to_ms(end)

    while cursor < end_ms:
        data = _get("/fapi/v1/klines", {
            "symbol": symbol,
            "interval": interval,
            "startTime": cursor,
            "endTime": end_ms,
            "limit": 1500,
        })
        if not data:
            break
        rows.extend(data)
        cursor = data[-1][0] + 1
        if len(data) < 1500:
            break
        time.sleep(0.1)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_volume", "trade_count",
        "taker_buy_base_vol", "taker_buy_quote_vol", "ignore",
    ])
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    for c in ["open", "high", "low", "close", "volume", "quote_volume",
              "taker_buy_base_vol", "taker_buy_quote_vol"]:
        df[c] = df[c].astype(float)
    df["trade_count"] = df["trade_count"].astype(int)
    df = df.drop(columns=["ignore"]).drop_duplicates("open_time").sort_values("open_time")
    return df


def fetch_mark_klines(symbol: str, interval: str, start: datetime, end: datetime) -> pd.DataFrame:
    """Mark price klines."""
    rows = []
    cursor = _to_ms(start)
    end_ms = _to_ms(end)

    while cursor < end_ms:
        data = _get("/fapi/v1/markPriceKlines", {
            "symbol": symbol,
            "interval": interval,
            "startTime": cursor,
            "endTime": end_ms,
            "limit": 1500,
        })
        if not data:
            break
        rows.extend(data)
        cursor = data[-1][0] + 1
        if len(data) < 1500:
            break
        time.sleep(0.1)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows, columns=[
        "open_time", "open", "high", "low", "close", "ignore1",
        "close_time", "ignore2", "trade_count", "ignore3", "ignore4", "ignore5",
    ])
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)
    for c in ["open", "high", "low", "close"]:
        df[c] = df[c].astype(float)
    df = df[["open_time", "close_time", "open", "high", "low", "close", "trade_count"]]
    df = df.drop_duplicates("open_time").sort_values("open_time")
    return df


def fetch_funding_rate(symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
    """펀딩비 — 8h 간격, 최대 1000건."""
    rows = []
    cursor = _to_ms(start)
    end_ms = _to_ms(end)

    while cursor < end_ms:
        data = _get("/fapi/v1/fundingRate", {
            "symbol": symbol,
            "startTime": cursor,
            "endTime": end_ms,
            "limit": 1000,
        })
        if not data:
            break
        rows.extend(data)
        cursor = data[-1]["fundingTime"] + 1
        if len(data) < 1000:
            break
        time.sleep(0.1)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["fundingTime"] = pd.to_datetime(df["fundingTime"], unit="ms", utc=True)
    df["fundingRate"] = df["fundingRate"].astype(float)
    if "markPrice" in df.columns:
        df["markPrice"] = df["markPrice"].astype(float)
    df = df.drop_duplicates("fundingTime").sort_values("fundingTime")
    return df


def fetch_open_interest(symbol: str, interval: str, start: datetime, end: datetime) -> pd.DataFrame:
    """오픈 인터레스트 히스토리 — 5m / 1h."""
    rows = []
    cursor = _to_ms(start)
    end_ms = _to_ms(end)

    while cursor < end_ms:
        data = _get("/futures/data/openInterestHist", {
            "symbol": symbol,
            "period": interval,
            "startTime": cursor,
            "endTime": end_ms,
            "limit": 500,
        })
        if not data:
            break
        rows.extend(data)
        cursor = data[-1]["timestamp"] + 1
        if len(data) < 500:
            break
        time.sleep(0.15)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    for c in ["sumOpenInterest", "sumOpenInterestValue"]:
        if c in df.columns:
            df[c] = df[c].astype(float)
    df = df.drop_duplicates("timestamp").sort_values("timestamp")
    return df


def fetch_long_short_ratio(symbol: str, interval: str, start: datetime, end: datetime) -> pd.DataFrame:
    """글로벌 L/S 비율 (전체 계정)."""
    rows = []
    cursor = _to_ms(start)
    end_ms = _to_ms(end)

    while cursor < end_ms:
        data = _get("/futures/data/globalLongShortAccountRatio", {
            "symbol": symbol,
            "period": interval,
            "startTime": cursor,
            "endTime": end_ms,
            "limit": 500,
        })
        if not data:
            break
        rows.extend(data)
        cursor = data[-1]["timestamp"] + 1
        if len(data) < 500:
            break
        time.sleep(0.15)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    for c in ["longAccount", "shortAccount", "longShortRatio"]:
        if c in df.columns:
            df[c] = df[c].astype(float)
    df = df.drop_duplicates("timestamp").sort_values("timestamp")
    return df


def fetch_top_trader_ls(symbol: str, interval: str, start: datetime, end: datetime) -> pd.DataFrame:
    """Top Trader L/S 포지션 비율."""
    rows = []
    cursor = _to_ms(start)
    end_ms = _to_ms(end)

    while cursor < end_ms:
        data = _get("/futures/data/topLongShortPositionRatio", {
            "symbol": symbol,
            "period": interval,
            "startTime": cursor,
            "endTime": end_ms,
            "limit": 500,
        })
        if not data:
            break
        rows.extend(data)
        cursor = data[-1]["timestamp"] + 1
        if len(data) < 500:
            break
        time.sleep(0.15)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    for c in ["longAccount", "shortAccount", "longShortRatio"]:
        if c in df.columns:
            df[c] = df[c].astype(float)
    df = df.drop_duplicates("timestamp").sort_values("timestamp")
    return df


def fetch_taker_vol(symbol: str, interval: str, start: datetime, end: datetime) -> pd.DataFrame:
    """Taker Buy/Sell Volume."""
    rows = []
    cursor = _to_ms(start)
    end_ms = _to_ms(end)

    while cursor < end_ms:
        data = _get("/futures/data/takerlongshortRatio", {
            "symbol": symbol,
            "period": interval,
            "startTime": cursor,
            "endTime": end_ms,
            "limit": 500,
        })
        if not data:
            break
        rows.extend(data)
        cursor = data[-1]["timestamp"] + 1
        if len(data) < 500:
            break
        time.sleep(0.15)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    for c in ["buySellRatio", "buyVol", "sellVol"]:
        if c in df.columns:
            df[c] = df[c].astype(float)
    df = df.drop_duplicates("timestamp").sort_values("timestamp")
    return df


def save_csv(df: pd.DataFrame, path: Path) -> None:
    if df.empty:
        log.warning(f"  EMPTY — skipped: {path.name}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    log.info(f"  Saved {len(df):,} rows → {path.relative_to(path.parents[3])}")


def get_traded_symbols() -> list[str]:
    """trades.jsonl에서 실제 트레이드된 심볼 추출."""
    base = Path(__file__).parent.parent
    trades_path = base / "data/reports/multi_strategy/trades.jsonl"
    fills_path = base / "trading_result/fills.csv"

    symbols = set()

    if trades_path.exists():
        with open(trades_path) as f:
            for line in f:
                try:
                    t = json.loads(line)
                    coin = t.get("coin", "")
                    if coin:
                        symbols.add(coin + "USDT")
                except:
                    pass

    if fills_path.exists():
        df = pd.read_csv(fills_path)
        if "order_link_id" in df.columns:
            # 심볼은 order_link_id의 두 번째 필드 (v43-sol-...)
            for oid in df["order_link_id"]:
                parts = str(oid).split("-")
                if len(parts) >= 2:
                    symbols.add(parts[1].upper() + "USDT")

    # 항상 BTC/ETH 포함 (레퍼런스)
    symbols.update({"BTCUSDT", "ETHUSDT"})

    return sorted(symbols)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=14, help="수집 기간 (일)")
    parser.add_argument("--output", default="data/raw/binance", help="저장 경로")
    parser.add_argument("--symbols", nargs="*", help="심볼 지정 (미지정 시 자동 감지)")
    args = parser.parse_args()

    base = Path(__file__).parent.parent
    out_dir = base / args.output

    end_dt = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=args.days)

    symbols = args.symbols if args.symbols else get_traded_symbols()
    log.info(f"수집 심볼 ({len(symbols)}개): {symbols}")
    log.info(f"기간: {start_dt.date()} ~ {end_dt.date()}")
    log.info(f"저장 경로: {out_dir}")

    # 데이터 품질 리포트
    quality_report = {}

    for sym in symbols:
        log.info(f"\n{'='*50}")
        log.info(f"[{sym}] 데이터 수집 시작")
        sym_dir = out_dir / sym
        sym_report = {}

        # 1. OHLCV klines
        for interval in ["1m", "5m", "15m", "1h", "4h"]:
            try:
                df = fetch_klines(sym, interval, start_dt, end_dt)
                save_csv(df, sym_dir / f"klines_{interval}.csv")
                sym_report[f"klines_{interval}"] = len(df)
            except Exception as e:
                log.error(f"  klines_{interval} 실패: {e}")
                sym_report[f"klines_{interval}"] = f"ERROR: {e}"

        # 2. Mark price klines
        for interval in ["5m", "15m", "1h"]:
            try:
                df = fetch_mark_klines(sym, interval, start_dt, end_dt)
                save_csv(df, sym_dir / f"mark_klines_{interval}.csv")
                sym_report[f"mark_{interval}"] = len(df)
            except Exception as e:
                log.error(f"  mark_klines_{interval} 실패: {e}")
                sym_report[f"mark_{interval}"] = f"ERROR: {e}"

        # 3. Funding rate
        try:
            df = fetch_funding_rate(sym, start_dt, end_dt)
            save_csv(df, sym_dir / "funding_rate.csv")
            sym_report["funding_rate"] = len(df)
        except Exception as e:
            log.error(f"  funding_rate 실패: {e}")
            sym_report["funding_rate"] = f"ERROR: {e}"

        # 4. Open interest
        for interval in ["5m", "1h"]:
            try:
                df = fetch_open_interest(sym, interval, start_dt, end_dt)
                save_csv(df, sym_dir / f"open_interest_{interval}.csv")
                sym_report[f"oi_{interval}"] = len(df)
            except Exception as e:
                log.error(f"  open_interest_{interval} 실패: {e}")
                sym_report[f"oi_{interval}"] = f"ERROR: {e}"

        # 5. Global L/S ratio
        for interval in ["5m", "1h"]:
            try:
                df = fetch_long_short_ratio(sym, interval, start_dt, end_dt)
                save_csv(df, sym_dir / f"ls_ratio_{interval}.csv")
                sym_report[f"ls_{interval}"] = len(df)
            except Exception as e:
                log.error(f"  ls_ratio_{interval} 실패: {e}")
                sym_report[f"ls_{interval}"] = f"ERROR: {e}"

        # 6. Top trader L/S position ratio
        for interval in ["5m", "1h"]:
            try:
                df = fetch_top_trader_ls(sym, interval, start_dt, end_dt)
                save_csv(df, sym_dir / f"top_trader_ls_{interval}.csv")
                sym_report[f"top_ls_{interval}"] = len(df)
            except Exception as e:
                log.error(f"  top_trader_ls_{interval} 실패: {e}")
                sym_report[f"top_ls_{interval}"] = f"ERROR: {e}"

        # 7. Taker buy/sell volume
        for interval in ["5m", "1h"]:
            try:
                df = fetch_taker_vol(sym, interval, start_dt, end_dt)
                save_csv(df, sym_dir / f"taker_vol_{interval}.csv")
                sym_report[f"taker_{interval}"] = len(df)
            except Exception as e:
                log.error(f"  taker_vol_{interval} 실패: {e}")
                sym_report[f"taker_{interval}"] = f"ERROR: {e}"

        quality_report[sym] = sym_report
        time.sleep(0.5)

    # 품질 리포트 저장
    report_path = out_dir / "collection_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump({
            "collected_at": end_dt.isoformat(),
            "start_date": start_dt.date().isoformat(),
            "end_date": end_dt.date().isoformat(),
            "symbols": symbols,
            "results": quality_report,
        }, f, indent=2, ensure_ascii=False)

    log.info(f"\n{'='*50}")
    log.info(f"수집 완료. 리포트: {report_path}")

    # 요약 출력
    print("\n=== 수집 결과 요약 ===")
    for sym, rep in quality_report.items():
        errors = [k for k, v in rep.items() if isinstance(v, str) and "ERROR" in v]
        total_rows = sum(v for v in rep.values() if isinstance(v, int))
        print(f"  {sym:12s}: {total_rows:,} rows, {len(rep) - len(errors)}/{len(rep)} 성공")
        if errors:
            print(f"    실패: {errors}")


if __name__ == "__main__":
    main()
