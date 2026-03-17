"""Top 10 코인 완전 무결 Raw 데이터 수집.

저장 파일:
  {COIN}_1h_raw.csv          — yfinance 원본 1h OHLCV (갭 표시 포함)
  {COIN}_4h_ohlcv.csv        — 4h 리샘플, 갭 없는 완전 인덱스 (ffill)
  {COIN}_4h_features.csv     — 4h 전체 피처 (204+개, NaN 0%)
  {COIN}_4h_microstructure.csv — 4h 마이크로스트럭처만 (OHLCV + 89 MS 피처)
  data_quality_report.json   — 코인별 데이터 품질 리포트
  manifest_v2.json           — 전체 매니페스트
"""

from __future__ import annotations
import sys, json, warnings
from pathlib import Path
from datetime import datetime, timezone

warnings.filterwarnings("ignore")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import yfinance as yf

from src.data.crawlers.crypto_ohlcv import (
    resample_to_4h, add_technical_indicators, _add_decomposition, TOP10_YAHOO,
)
from src.data.crawlers.signal_features import add_signal_features
from src.data.crawlers.microstructure_rollup import add_microstructure_rollup, ALL_MS_FEATURES

OUT = Path(__file__).resolve().parents[1] / "data" / "microstructure"
OUT.mkdir(parents=True, exist_ok=True)

PERIOD   = "730d"      # 2년 — yfinance 1h 최대 제공 기간
INTERVAL = "1h"


# ── 갭 없는 완전 4h 인덱스 생성 ──────────────────────────────────

def make_complete_4h_index(start: pd.Timestamp, end: pd.Timestamp) -> pd.DatetimeIndex:
    """UTC 기준 4h 완전 인덱스 (00:00, 04:00, ..., 20:00)."""
    return pd.date_range(start=start.floor("4h"), end=end.floor("4h"),
                         freq="4h", tz="UTC")


def fill_gaps(df_4h: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """4h DataFrame을 완전 인덱스로 재인덱싱 + 갭 보고."""
    # tz 통일
    if df_4h.index.tz is None:
        df_4h.index = df_4h.index.tz_localize("UTC")

    full_idx = make_complete_4h_index(df_4h.index[0], df_4h.index[-1])
    n_before = len(df_4h)

    df_full = df_4h.reindex(full_idx)

    # 갭 위치 기록
    gap_rows = df_full[df_full["close"].isna()]
    gaps = [str(t) for t in gap_rows.index]

    # OHLCV: ffill (갭 = 이전 종가 유지)
    for col in ["open", "high", "low", "close"]:
        df_full[col] = df_full[col].ffill()
    # volume: 0으로 (거래 없음 표시)
    df_full["volume"] = df_full["volume"].fillna(0)

    quality = {
        "original_bars" : n_before,
        "complete_bars" : len(df_full),
        "filled_bars"   : len(gaps),
        "fill_rate_pct" : round(len(gaps) / len(df_full) * 100, 4),
        "gap_timestamps": gaps,
    }
    return df_full, quality


# ── 단일 코인 수집 ────────────────────────────────────────────────

def collect(ticker: str, yahoo_sym: str) -> dict | None:
    print(f"\n{'='*64}")
    print(f"  {ticker}  ({yahoo_sym})")
    print(f"{'='*64}")

    # 1. 1h Raw OHLCV
    try:
        tk = yf.Ticker(yahoo_sym)
        df_1h = tk.history(period=PERIOD, interval=INTERVAL, auto_adjust=True)
        df_1h.columns = [c.lower() for c in df_1h.columns]
        df_1h = df_1h[["open", "high", "low", "close", "volume"]].dropna()
        if df_1h.index.tz is None:
            df_1h.index = df_1h.index.tz_localize("UTC")
        print(f"  [1h raw] {len(df_1h)} bars  "
              f"{df_1h.index[0].date()} ~ {df_1h.index[-1].date()}")
    except Exception as e:
        print(f"  [ERROR] 1h fetch: {e}"); return None

    # 2. 4h 리샘플
    df_4h = resample_to_4h(df_1h)
    print(f"  [4h]    {len(df_4h)} bars  (before gap-fill)")

    # 3. 갭 완전 보정
    df_4h, gap_info = fill_gaps(df_4h)
    print(f"  [fill]  {gap_info['filled_bars']}개 갭 보정 "
          f"({gap_info['fill_rate_pct']}%)")

    # 4. 기술지표 + 신호피처 + 마이크로스트럭처
    n0 = len(df_4h.columns)
    df_feat = add_technical_indicators(df_4h.copy())
    df_feat = _add_decomposition(df_feat, period=42)
    df_feat = add_signal_features(df_feat, verbose=False)
    df_feat = add_microstructure_rollup(df_feat, verbose=False)
    print(f"  [feat]  {n0} → {len(df_feat.columns)} cols")

    # 5. 최종 클린업 — NaN/Inf 완전 제거
    df_feat = df_feat.replace([np.inf, -np.inf], np.nan)
    df_feat = df_feat.ffill().bfill().fillna(0)

    # 6. 마이크로스트럭처 전용 DataFrame
    ms_cols = ["open", "high", "low", "close", "volume"] + \
              [c for c in ALL_MS_FEATURES if c in df_feat.columns]
    df_ms = df_feat[ms_cols]

    # 7. 품질 검증
    nan_total   = int(df_feat.isna().sum().sum())
    inf_total   = int(np.isinf(df_feat.select_dtypes(include=np.number).values).sum())
    ohlcv_gaps  = int((df_feat["volume"] == 0).sum())  # volume=0 = 보정된 갭

    quality = {
        **gap_info,
        "final_bars"       : len(df_feat),
        "total_features"   : len(df_feat.columns),
        "ms_features"      : len([c for c in ALL_MS_FEATURES if c in df_feat.columns]),
        "nan_cells"        : nan_total,
        "inf_cells"        : inf_total,
        "zero_vol_bars"    : ohlcv_gaps,   # 보정된 갭 바
        "latest_close"     : round(float(df_feat["close"].iloc[-1]), 6),
        "date_start"       : str(df_feat.index[0]),
        "date_end"         : str(df_feat.index[-1]),
    }

    assert nan_total == 0, f"NaN 잔존: {nan_total}"
    assert inf_total == 0, f"Inf 잔존: {inf_total}"

    # 8. 저장
    df_1h.to_csv(OUT / f"{ticker}_1h_raw.csv")
    df_4h[["open","high","low","close","volume"]].to_csv(OUT / f"{ticker}_4h_ohlcv.csv")
    df_feat.to_csv(OUT / f"{ticker}_4h_features.csv")
    df_ms.to_csv(OUT / f"{ticker}_4h_microstructure.csv")

    print(f"  [OK] 저장 완료  NaN={nan_total}  Inf={inf_total}  "
          f"vol0_bars={ohlcv_gaps}  total_feat={len(df_feat.columns)}")

    return quality


# ── 메인 ──────────────────────────────────────────────────────────

def main():
    print("=" * 64)
    print("  TOP 10 코인 완전 무결 Raw 데이터 수집 v2")
    print(f"  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    print("=" * 64)

    manifest = {
        "version"       : "v2",
        "collected_at"  : datetime.now(timezone.utc).isoformat(),
        "period"        : PERIOD,
        "interval_raw"  : INTERVAL,
        "interval_feat" : "4h",
        "files": {
            "{COIN}_1h_raw.csv"          : "yfinance 원본 1h OHLCV",
            "{COIN}_4h_ohlcv.csv"        : "4h OHLCV (갭 ffill 완전 인덱스)",
            "{COIN}_4h_features.csv"     : "4h 전체 피처 (기술지표+신호분석+MS)",
            "{COIN}_4h_microstructure.csv": "4h OHLCV + 마이크로스트럭처 피처만",
        },
        "gap_fill_method": "OHLCV ffill (close=prev), volume=0",
        "coins"         : {},
    }

    quality_report = {}
    ok, fail = [], []

    for ticker, yahoo_sym in TOP10_YAHOO.items():
        q = collect(ticker, yahoo_sym)
        if q:
            manifest["coins"][ticker] = q
            quality_report[ticker] = q
            ok.append(ticker)
        else:
            fail.append(ticker)

    # 저장
    with open(OUT / "manifest_v2.json", "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    with open(OUT / "data_quality_report.json", "w") as f:
        json.dump(quality_report, f, indent=2, ensure_ascii=False)

    # 요약 출력
    print("\n" + "=" * 64)
    print("  최종 요약")
    print("=" * 64)
    print(f"  성공: {len(ok)}개 | 실패: {len(fail)}개")
    print(f"\n  {'코인':<6} {'bars':>5} {'feat':>5} {'MS':>4} {'NaN':>4} {'갭수':>5} {'fill%':>6}")
    print("  " + "-" * 45)
    for t, q in quality_report.items():
        print(f"  {t:<6} {q['final_bars']:>5} {q['total_features']:>5} "
              f"{q['ms_features']:>4} {q['nan_cells']:>4} "
              f"{q['filled_bars']:>5} {q['fill_rate_pct']:>5.2f}%")


if __name__ == "__main__":
    main()
