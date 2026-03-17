"""9시까지 파이프라인 반복 실행 스크립트.

매 실행마다:
1. 데이터 재수집 (최신 데이터 반영)
2. MOMENT pretrain + classification
3. 5-model ensemble + GridSearch
4. 리포트 저장 (run_N 형태)

9시 이후 자동 종료.
"""

import os
import sys
import time
import json
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ["PYTHONUNBUFFERED"] = "1"

DEADLINE = datetime(2026, 3, 12, 9, 0, 0)  # 3월 12일 오전 9시


def run_single_iteration(run_id: int):
    """파이프라인 1회 실행."""
    from src.models.run_masking_pipeline import collect_all_data, run_loop, format_report

    print(f"\n{'#' * 70}")
    print(f"  OVERNIGHT RUN #{run_id}")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Deadline: {DEADLINE.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Time remaining: {(DEADLINE - datetime.now()).total_seconds()/3600:.1f}h")
    print(f"{'#' * 70}\n")

    t0 = time.time()

    # 데이터 수집
    ohlcv, media, macro_aligned = collect_all_data()
    if not ohlcv:
        print("[FATAL] No OHLCV data. Skipping this run.")
        return None

    # 학습 + 평가
    report = run_loop(ohlcv, media, num_iterations=7, macro_aligned=macro_aligned)

    # 리포트 저장
    md = format_report(report)
    date_str = datetime.now().strftime("%Y%m%d")
    rdir = Path("data/reports") / date_str
    rdir.mkdir(parents=True, exist_ok=True)

    # run별 리포트 저장
    (rdir / f"masking_loop_report_run{run_id}.md").write_text(md, encoding="utf-8")
    with open(rdir / f"masking_loop_report_run{run_id}.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str, ensure_ascii=False)

    # 최신 리포트도 덮어쓰기
    (rdir / "masking_loop_report.md").write_text(md, encoding="utf-8")
    with open(rdir / "masking_loop_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str, ensure_ascii=False)

    elapsed = time.time() - t0
    print(f"\n  Run #{run_id} complete: {elapsed:.0f}s ({elapsed/60:.1f}m)")

    # 결과 요약 추출
    perf = report.get("performance_summary", {})
    dl = report.get("deep_learning_results", {})
    avg_dl_acc = sum(r["accuracy"] for r in dl.values()) / len(dl) if dl else 0

    summary = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "elapsed_seconds": elapsed,
        "ml_accuracy": perf.get("best_accuracy", 0),
        "ml_f1": perf.get("best_f1", 0),
        "ml_sharpe": perf.get("final_sharpe", 0),
        "dl_avg_accuracy": avg_dl_acc,
    }

    print(f"  ML Best Acc: {summary['ml_accuracy']:.1%} | DL Avg Acc: {avg_dl_acc:.1%}")
    return summary


def main():
    print("=" * 70)
    print("  OVERNIGHT OPTIMIZATION LOOP")
    print(f"  Current: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Deadline: {DEADLINE.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Max duration: {(DEADLINE - datetime.now()).total_seconds()/3600:.1f}h")
    print("=" * 70)

    all_summaries = []
    run_id = 1

    while datetime.now() < DEADLINE:
        remaining = (DEADLINE - datetime.now()).total_seconds()
        if remaining < 1800:  # 30분 미만이면 종료
            print(f"\n[STOP] 남은 시간 {remaining/60:.0f}분 — 새 실행 불가. 종료.")
            break

        try:
            summary = run_single_iteration(run_id)
            if summary:
                all_summaries.append(summary)
        except Exception as e:
            print(f"\n[ERROR] Run #{run_id} failed: {e}")
            import traceback
            traceback.print_exc()

        run_id += 1

        # 결과 기록
        date_str = datetime.now().strftime("%Y%m%d")
        rdir = Path("data/reports") / date_str
        rdir.mkdir(parents=True, exist_ok=True)
        with open(rdir / "overnight_summary.json", "w", encoding="utf-8") as f:
            json.dump(all_summaries, f, indent=2, default=str)

    # 최종 요약
    print(f"\n{'=' * 70}")
    print(f"  OVERNIGHT LOOP COMPLETE")
    print(f"  Total runs: {len(all_summaries)}")
    if all_summaries:
        best = max(all_summaries, key=lambda x: x.get("ml_accuracy", 0))
        print(f"  Best ML Acc: {best['ml_accuracy']:.1%} (Run #{best['run_id']})")
        best_dl = max(all_summaries, key=lambda x: x.get("dl_avg_accuracy", 0))
        print(f"  Best DL Acc: {best_dl['dl_avg_accuracy']:.1%} (Run #{best_dl['run_id']})")
    print(f"  Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
