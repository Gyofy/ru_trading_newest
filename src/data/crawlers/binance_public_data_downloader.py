"""
Binance Public Data Downloader
-------------------------------
Binance public data repo에서 Futures UM 데이터를 다운로드합니다.
https://github.com/binance/binance-public-data

지원 타입:
  - funding_rate       (8h, ~10MB / 7coins / 365d)
  - open_interest      (5m,  ~500MB / 7coins / 365d)
  - long_short_ratio   (5m,  ~200MB / 7coins / 365d)
  - taker_volume       (5m,  ~200MB / 7coins / 365d)
  - liquidation        (tick, ~100MB / 7coins / 180d)
  - agg_trades         (tick, ~15GB / BTC / 180d)
  - book_depth         (snap, ~30-50GB / 3coins / 90d)
  - klines_1m          (1m,  ~2GB / 3coins / 90d)

사용법:
  python src/data/crawlers/binance_public_data_downloader.py \\
      --types funding_rate open_interest long_short_ratio \\
      --symbols BTCUSDT ETHUSDT SOLUSDT XRPUSDT ADAUSDT DOTUSDT LINKUSDT \\
      --days 365 \\
      --output data/raw/binance_public
"""

import argparse
import os
import time
import zipfile
import logging
from datetime import date, timedelta
from pathlib import Path

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

BASE_URL = "https://data.binance.vision/data/futures/um/daily"

# 재시도 정책: 연결 오류 / 5xx 에 대해 최대 5회, 지수 백오프
_RETRY = Retry(
    total=5,
    backoff_factor=1.5,        # 1.5, 3, 4.5, ... 초
    status_forcelist=[500, 502, 503, 504],
    allowed_methods=["GET"],
    raise_on_status=False,
)

# data type → (url_path_segment, file_suffix)
TYPE_MAP = {
    "funding_rate":     ("fundingRate",                    "fundingRate"),
    "open_interest":    ("openInterestHist",               "openInterestHist"),
    "long_short_ratio": ("globalLongShortAccountRatio",    "globalLongShortAccountRatio"),
    "taker_volume":     ("takerlongshortRatio",            "takerlongshortRatio"),
    "liquidation":      ("forceOrders",                    "forceOrders"),
    "agg_trades":       ("aggTrades",                      "aggTrades"),
    "book_depth":       ("bookDepth",                      "bookDepth_T5"),  # top-5 레벨
    "klines_1m":        ("klines/{symbol}/1m",             "1m"),
}


def build_url(data_type: str, symbol: str, day: date) -> str:
    path_tmpl, suffix = TYPE_MAP[data_type]
    path = path_tmpl.replace("{symbol}", symbol)
    date_str = day.strftime("%Y-%m-%d")
    return f"{BASE_URL}/{path}/{symbol}/{symbol}-{suffix}-{date_str}.zip"


def _make_session() -> requests.Session:
    session = requests.Session()
    adapter = HTTPAdapter(max_retries=_RETRY)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def download_file(url: str, dest: Path, session: requests.Session, max_attempts: int = 5) -> bool:
    """Download a single zip file. Returns True on success, False if 404."""
    if dest.exists():
        return True  # 이미 있으면 스킵

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(".tmp")

    for attempt in range(1, max_attempts + 1):
        try:
            resp = session.get(url, stream=True, timeout=60)
            if resp.status_code == 404:
                return False
            resp.raise_for_status()

            with open(tmp, "wb") as f:
                for chunk in resp.iter_content(chunk_size=1 << 20):
                    f.write(chunk)
            tmp.rename(dest)
            return True

        except Exception as exc:
            if tmp.exists():
                tmp.unlink()
            if attempt == max_attempts:
                log.warning("다운로드 실패 (최종): %s — %s", url, exc)
                return False
            wait = 2 ** attempt
            log.debug("재시도 %d/%d (%.0fs 후): %s", attempt, max_attempts, wait, exc)
            time.sleep(wait)

    return False


def extract_zip(zip_path: Path, out_dir: Path) -> None:
    """압축 해제 후 원본 zip 삭제."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)
    zip_path.unlink()


def date_range(days: int) -> list[date]:
    today = date.today()
    return [today - timedelta(days=i) for i in range(1, days + 1)]


def download(
    data_types: list[str],
    symbols: list[str],
    days: int,
    output_dir: str,
    extract: bool = True,
) -> None:
    out = Path(output_dir)
    session = _make_session()
    dates = date_range(days)

    for dtype in data_types:
        if dtype not in TYPE_MAP:
            log.warning("알 수 없는 타입 무시: %s", dtype)
            continue

        for symbol in symbols:
            log.info("[%s] %s 다운로드 시작 (%d일)", dtype, symbol, days)
            sym_dir = out / dtype / symbol
            sym_dir.mkdir(parents=True, exist_ok=True)

            missing = 0
            for day in tqdm(dates, desc=f"{dtype}/{symbol}", leave=False):
                url = build_url(dtype, symbol, day)
                zip_path = sym_dir / Path(url).name
                ok = download_file(url, zip_path, session)
                if not ok:
                    missing += 1
                    continue
                if extract and zip_path.exists():
                    extract_zip(zip_path, sym_dir)

            log.info("[%s] %s 완료 (누락 %d일)", dtype, symbol, missing)


def main() -> None:
    parser = argparse.ArgumentParser(description="Binance Public Data Downloader")
    parser.add_argument(
        "--types",
        nargs="+",
        default=["funding_rate", "open_interest", "long_short_ratio"],
        help="다운로드할 데이터 타입 (공백 구분)",
    )
    parser.add_argument(
        "--symbols",
        nargs="+",
        default=["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "ADAUSDT", "DOTUSDT", "LINKUSDT"],
        help="심볼 목록",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=365,
        help="수집 기간 (일 수, 오늘 기준 과거)",
    )
    parser.add_argument(
        "--output",
        default="data/raw/binance_public",
        help="저장 디렉토리",
    )
    parser.add_argument(
        "--no-extract",
        action="store_true",
        help="zip 압축 해제 생략",
    )
    args = parser.parse_args()

    log.info("수집 시작: types=%s symbols=%s days=%d", args.types, args.symbols, args.days)
    download(
        data_types=args.types,
        symbols=args.symbols,
        days=args.days,
        output_dir=args.output,
        extract=not args.no_extract,
    )
    log.info("완료")


if __name__ == "__main__":
    main()
