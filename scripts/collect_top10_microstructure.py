"""Top 10 코인 마이크로스트럭처 데이터 수집 스크립트.

전체 피처 파이프라인(기술지표 + 신호분석 + CVD/OFI/VPIN/Roll/Amihud)을
상위 10개 코인에 적용하고 CSV + Parquet으로 저장.

Usage:
    python scripts/collect_top10_microstructure.py
"""

from __future__ import annotations

import sys
from pathlib import Path
import warnings
import json
from datetime import datetime, timezone

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.data.crawlers.crypto_ohlcv import (
    fetch_ohlcv, resample_to_4h, add_technical_indicators,
    _add_decomposition, TOP10_YAHOO,
)
from src.data.crawlers.signal_features import add_signal_features
from src.data.crawlers.microstructure_rollup import (
    add_microstructure_rollup, ALL_MS_FEATURES,
    CVD_FEATURES, OFI_FEATURES, VPIN_FEATURES,
    ROLL_FEATURES, AMIHUD_FEATURES, COMPOSITE_FEATURES,
)

OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data" / "microstructure"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def collect_coin(ticker: str, yahoo_sym: str,
                 period: str = "365d") -> pd.DataFrame | None:
    """단일 코인 전체 피처 파이프라인 실행."""
    print(f"\n{'='*60}")
    print(f"  {ticker} ({yahoo_sym})")
    print(f"{'='*60}")

    try:
        # 1. 1h OHLCV
        df = fetch_ohlcv(ticker, yahoo_sym, period, "1h")
        if df.empty:
            print(f"  [SKIP] {ticker}: empty data")
            return None
        print(f"  [OK] 1h OHLCV: {len(df)} bars")

        # 2. 4h resample
        df = resample_to_4h(df)
        print(f"  [OK] 4h resample: {len(df)} bars")
        print(f"       기간: {df.index[0].date()} ~ {df.index[-1].date()}")
        print(f"       현재가: ${df['close'].iloc[-1]:,.4f}")

        # 3. 기술지표
        df = add_technical_indicators(df)
        df = _add_decomposition(df, period=42)
        print(f"  [OK] 기술지표 완료: {len(df.columns)} cols")

        # 4. 신호분석 피처 (wavelet, FFT, entropy, 기존 microstructure 포함)
        n_before = len(df.columns)
        df = add_signal_features(df, verbose=False)
        print(f"  [OK] 신호분석 피처: +{len(df.columns) - n_before} cols → 총 {len(df.columns)}")

        # 5. 마이크로스트럭처 롤업 (CVD/OFI/VPIN/Roll/Amihud)
        n_before = len(df.columns)
        df = add_microstructure_rollup(df, verbose=True)
        print(f"  [OK] 마이크로스트럭처: +{len(df.columns) - n_before} cols → 총 {len(df.columns)}")

        # NaN 정리
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.ffill().bfill().fillna(0)

        # 마이크로스트럭처 피처 현황 출력
        available_ms = [c for c in ALL_MS_FEATURES if c in df.columns]
        print(f"\n  마이크로스트럭처 피처: {len(available_ms)}/{len(ALL_MS_FEATURES)}개 생성")
        print(f"    CVD    : {len([c for c in CVD_FEATURES if c in df.columns])}")
        print(f"    OFI    : {len([c for c in OFI_FEATURES if c in df.columns])}")
        print(f"    VPIN   : {len([c for c in VPIN_FEATURES if c in df.columns])}")
        print(f"    Roll   : {len([c for c in ROLL_FEATURES if c in df.columns])}")
        print(f"    Amihud : {len([c for c in AMIHUD_FEATURES if c in df.columns])}")
        print(f"    복합   : {len([c for c in COMPOSITE_FEATURES if c in df.columns])}")

        # 최신 주요 지표값
        last = df.iloc[-1]
        print(f"\n  최신 마이크로스트럭처 스냅샷 ({df.index[-1]}):")
        snap_cols = ["vd", "cvd", "buy_frac", "ofi_raw",
                     "vpin_12", "vpin_24", "roll_spread_12",
                     "illiq_6", "ms_composite"]
        for col in snap_cols:
            if col in df.columns:
                print(f"    {col:25s}: {last[col]:.6f}")

        return df

    except Exception as e:
        print(f"  [ERROR] {ticker}: {e}")
        import traceback; traceback.print_exc()
        return None


def save_coin_data(ticker: str, df: pd.DataFrame) -> dict:
    """CSV + Parquet 저장, 반환값: 메타데이터."""
    # 전체 피처 CSV
    csv_path = OUTPUT_DIR / f"{ticker}_features_4h.csv"
    df.to_csv(csv_path)

    # 마이크로스트럭처 피처만 별도 저장
    ms_cols = ["open", "high", "low", "close", "volume"] + \
              [c for c in ALL_MS_FEATURES if c in df.columns]
    df_ms = df[ms_cols]
    ms_csv_path = OUTPUT_DIR / f"{ticker}_microstructure_4h.csv"
    df_ms.to_csv(ms_csv_path)

    # Parquet (더 효율적)
    try:
        parquet_path = OUTPUT_DIR / f"{ticker}_features_4h.parquet"
        df.to_parquet(parquet_path, compression="gzip")
    except Exception:
        parquet_path = None

    meta = {
        "ticker": ticker,
        "bars": len(df),
        "cols_total": len(df.columns),
        "cols_ms": len([c for c in ALL_MS_FEATURES if c in df.columns]),
        "date_start": str(df.index[0].date()),
        "date_end": str(df.index[-1].date()),
        "latest_close": round(float(df["close"].iloc[-1]), 6),
        "csv_path": str(csv_path.relative_to(Path(__file__).parents[1])),
        "ms_csv_path": str(ms_csv_path.relative_to(Path(__file__).parents[1])),
    }

    print(f"  [SAVED] {csv_path.name} ({len(df)} rows × {len(df.columns)} cols)")
    return meta


def main():
    print("=" * 60)
    print("  TOP 10 코인 마이크로스트럭처 데이터 수집")
    print(f"  시작: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 60)

    manifest = {
        "collected_at": datetime.now(timezone.utc).isoformat(),
        "coins": {},
        "total_features": 0,
        "ms_features": ALL_MS_FEATURES,
    }

    success, failed = [], []

    for ticker, yahoo_sym in TOP10_YAHOO.items():
        df = collect_coin(ticker, yahoo_sym, period="365d")
        if df is not None:
            meta = save_coin_data(ticker, df)
            manifest["coins"][ticker] = meta
            manifest["total_features"] = meta["cols_total"]
            success.append(ticker)
        else:
            failed.append(ticker)

    # 매니페스트 저장
    manifest_path = OUTPUT_DIR / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print("  수집 완료 요약")
    print("=" * 60)
    print(f"  성공: {len(success)}개 → {', '.join(success)}")
    if failed:
        print(f"  실패: {len(failed)}개 → {', '.join(failed)}")
    print(f"  저장 위치: {OUTPUT_DIR}")
    print(f"  피처 수: {manifest['total_features']}개")
    print(f"  마이크로스트럭처 피처: {len(ALL_MS_FEATURES)}개 정의")
    print(f"  매니페스트: {manifest_path.name}")


if __name__ == "__main__":
    main()
