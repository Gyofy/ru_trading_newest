"""Iterative Masking Loop Pipeline v2.

Phase 1: Data Collection (yfinance + 15 media sources)
Phase 2: Feature Engineering + Label Creation (UP/DOWN/HOLD)
Phase 3a: Deep Learning (OLinear + RevIN + Multi-coin Pooling)
Phase 3b: 5-Model Ensemble + GridSearch (LGB/XGB/CB/RF/ET)
Phase 4: Final Report
"""

import os
import sys
import json
import time
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*DataFrame is highly fragmented.*")

os.environ["PYTHONIOENCODING"] = "utf-8"
sys.stdout.reconfigure(encoding="utf-8") if hasattr(sys.stdout, "reconfigure") else None

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.data.crawlers.crypto_ohlcv import fetch_all_top10, save_ohlcv_data
from src.data.crawlers.google_news_crawler import crawl_all_google_news
from src.data.crawlers.reddit_crawler import crawl_all_reddit
from src.data.crawlers.x_crawler import crawl_all_x
from src.data.crawlers.youtube_crawler import crawl_all_youtube, crawl_tiktok_proxy, crawl_instagram_proxy
from src.data.crawlers.sentiment_analyzer import aggregate_sentiment
from src.data.crawlers.macro_commodity_crawler import crawl_all_macro_data
from src.models.masking_loop import IterativeMaskingLoop, build_feature_matrix, HORIZONS, HORIZON_LABELS
from src.utils.config import seq_len as _cfg_seq_len


def collect_all_data():
    print("\n" + "=" * 70)
    print("  PHASE 1: DATA COLLECTION (4h OHLCV + 5 Macro)")
    print("=" * 70)

    # 1. OHLCV (1h fetch → 4h resample)
    print("\n[1/9] OHLCV (yfinance, 365d 1h → 4h resample)...")
    ohlcv_data = fetch_all_top10("365d", "1h")
    if ohlcv_data:
        save_ohlcv_data(ohlcv_data)
    print(f"  => {len(ohlcv_data)} coins collected")

    # v4: 미디어 크롤링 스킵 (합성 노이즈 피처 제거됨)
    print("\n  [SKIP] Media crawling disabled (v4: synthetic features removed)")
    media_data = {}

    # 2. Macro / Commodity / Liquidity (5 tickers, ~20 features)
    print("\n[2/2] Macro / Commodity / Liquidity (5 tickers)...")
    # crypto_index 생성 (첫 코인의 인덱스 사용)
    first_coin_df = next(iter(ohlcv_data.values())) if ohlcv_data else None
    crypto_index = first_coin_df.index if first_coin_df is not None else None
    macro_result = crawl_all_macro_data(crypto_index)
    macro_aligned = macro_result.get("aligned", None)
    macro_feature_count = macro_result.get("feature_count", 0)
    print(f"  => {macro_feature_count} macro features aligned to crypto bars")

    return ohlcv_data, media_data, macro_aligned


def run_loop(ohlcv_data, media_data, num_iterations=7, macro_aligned=None):
    print("\n" + "=" * 70)
    print("  PHASE 2: FEATURE ENGINEERING")
    print("=" * 70)

    from src.utils.config import max_horizon as _cfg_max_h
    feature_matrices = build_feature_matrix(ohlcv_data, media_data, horizon=_cfg_max_h())

    # Macro/Commodity features 병합 (시간 정렬된 데이터)
    if macro_aligned is not None and len(macro_aligned) > 0:
        macro_cols = len(macro_aligned.columns)
        for ticker, df in feature_matrices.items():
            # timezone 통일: tz-aware → naive 변환
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            # 인덱스 기준으로 merge (시간 정렬)
            merged = df.join(macro_aligned, how="left")
            merged = merged.ffill().fillna(0)  # no bfill: prevents future data leakage
            feature_matrices[ticker] = merged
        print(f"  => Macro features merged: +{macro_cols} columns per coin")

    from src.utils.config import bar_minutes as _cfg_bm, horizon_labels as _cfg_hlabels
    bm = _cfg_bm()
    h_labels = _cfg_hlabels()
    h_str = ", ".join(h_labels.values())
    print(f"  => {len(feature_matrices)} coins with features ({bm}min bars, horizons: {h_str})")
    for t, df in feature_matrices.items():
        print(f"    {t}: {len(df)} rows x {len(df.columns)} cols")

    if not feature_matrices:
        print("  [ERROR] No feature matrices. Aborting.")
        return {"error": "no_data"}

    # Deep Learning phase
    print("\n" + "=" * 70)
    print("  PHASE 3a: DEEP LEARNING (MOMENT Pretrain + OLinear + RevIN)")
    print("=" * 70)

    deep_results = {}
    try:
        from src.models.multimodal_classifier import (
            MultiModalTrainer, MultiModalDataset,
            MEDIA_SOURCES, device,
        )
        import numpy as np
        print(f"  Device: {device}")
        print(f"  Architecture: Per-Coin Pretrain (MAE) + Fine-tune (Classification)")
        print(f"  Pretrain: MOMENT MAE + RevIN + SelectiveMasking (30ep)")
        print(f"  Fine-tune: NormLin + MediaAttention + CrossFusion (80ep)")

        SEQ_LEN = _cfg_seq_len()  # config: timeframes.tactical.seq_len

        dl_total = len(feature_matrices)
        dl_done = 0
        dl_start_time = time.time()

        # 2-Stage 여부
        from src.utils.config import load_settings as _load_dl_settings
        settings = _load_dl_settings()
        labeling_cfg = settings.get("labeling", {})
        two_stage = labeling_cfg.get("two_stage", False)
        s1_threshold = labeling_cfg.get("stage1_threshold", 0.50)

        for ticker, df in feature_matrices.items():
            price_cols = [c for c in df.columns
                          if not c.startswith("media_") and c not in ["label", "future_return", "open", "high", "low", "close", "volume"]
                          and df[c].dtype in [float, np.float64, np.float32, int, np.int64]]
            media_cols = []

            valid = df.replace([np.inf, -np.inf], np.nan)
            valid.ffill(inplace=True)
            valid.fillna(0, inplace=True)  # no bfill
            valid = valid.dropna(subset=["label"])

            if len(valid) < 100:
                print(f"  [{ticker}] SKIP — insufficient data ({len(valid)} rows)")
                continue

            X_price = valid[price_cols].values
            X_media = np.zeros((len(valid), 1))
            y_3class = valid["label"].values

            n_price = X_price.shape[1]
            n_media = X_media.shape[1]

            split = int(len(X_price) * 0.8)
            price_train, price_test = X_price[:split], X_price[split:]
            media_train, media_test = X_media[:split], X_media[split:]
            y_train_3c, y_test_3c = y_3class[:split], y_3class[split:]

            if two_stage:
                # === 2-Stage DL ===
                print(f"\n  [{ticker}] 2-Stage DL (train={len(y_train_3c)}, test={len(y_test_3c)})")

                # Stage 1: Trade/NoTrade
                y_s1_train = (y_train_3c != 1).astype(int)  # HOLD=0, Trade=1
                y_s1_test = (y_test_3c != 1).astype(int)

                trainer_s1 = MultiModalTrainer(
                    num_price_features=n_price, num_media_features=max(n_media, 1),
                    seq_len=SEQ_LEN, num_classes=2,
                )
                info_s1 = trainer_s1.train(
                    price_train, media_train, y_s1_train,
                    epochs=80, batch_size=512, lr=3e-4, num_workers=6,
                    pretrain_epochs=30,
                )
                if info_s1.get("status") == "insufficient_data":
                    continue

                eval_s1 = trainer_s1.evaluate(price_test, media_test, y_s1_test)
                _, proba_s1, _ = trainer_s1.predict(X_price[-SEQ_LEN:], X_media[-SEQ_LEN:])
                is_trade = float(proba_s1[0][1]) >= s1_threshold

                # Stage 2: Long/Short (Trade samples only)
                trade_mask = y_train_3c != 1
                if trade_mask.sum() >= 50:
                    y_s2_train = (y_train_3c[trade_mask] == 2).astype(int)  # UP=Long=1
                    trade_test_mask = y_test_3c != 1
                    y_s2_test = (y_test_3c[trade_test_mask] == 2).astype(int)

                    trainer_s2 = MultiModalTrainer(
                        num_price_features=n_price, num_media_features=max(n_media, 1),
                        seq_len=SEQ_LEN, num_classes=2,
                    )
                    info_s2 = trainer_s2.train(
                        price_train[trade_mask], media_train[trade_mask], y_s2_train,
                        epochs=60, batch_size=512, lr=3e-4, num_workers=6,
                        pretrain_epochs=20,
                    )
                    eval_s2 = trainer_s2.evaluate(
                        price_test[trade_test_mask], media_test[trade_test_mask], y_s2_test
                    ) if trade_test_mask.sum() > 0 else {"accuracy": 0, "f1": 0}

                    _, proba_s2, _ = trainer_s2.predict(X_price[-SEQ_LEN:], X_media[-SEQ_LEN:])

                    if is_trade:
                        long_prob = float(proba_s2[0][1])
                        direction = "UP" if long_prob >= 0.5 else "DOWN"
                    else:
                        direction = "HOLD"
                else:
                    eval_s2 = {"accuracy": 0, "f1": 0}
                    direction = "HOLD"

                deep_results[ticker] = {
                    "accuracy": eval_s1["accuracy"],
                    "f1": eval_s1["f1"],
                    "prediction": direction,
                    "stage1_acc": eval_s1["accuracy"],
                    "stage2_acc": eval_s2["accuracy"],
                    "trade_prob": round(float(proba_s1[0][1]), 3),
                    "source_weights": eval_s1.get("source_weights", {}),
                    "data_size": info_s1.get("data_size", 0),
                }
                print(f"    S1 Acc: {eval_s1['accuracy']:.1%} | S2 Acc: {eval_s2['accuracy']:.1%} | Pred: {direction}")

            else:
                # === 기존 3-class DL ===
                print(f"\n  [{ticker}] Pretrain+Fine-tune (train={len(y_train_3c)}, test={len(y_test_3c)})")
                print(f"    Price: {n_price} | Media: {n_media} (dummy)")

                trainer = MultiModalTrainer(
                    num_price_features=n_price, num_media_features=max(n_media, 1),
                    seq_len=SEQ_LEN, num_classes=3,
                )
                train_info = trainer.train(
                    price_train, media_train, y_train_3c,
                    epochs=80, batch_size=512, lr=3e-4, num_workers=6,
                    pretrain_epochs=30,
                )
                if train_info.get("status") == "insufficient_data":
                    continue

                eval_result = trainer.evaluate(price_test, media_test, y_test_3c)
                pred_class, pred_proba, src_weights = trainer.predict(
                    X_price[-SEQ_LEN:], X_media[-SEQ_LEN:]
                )

                label_names = {0: "DOWN", 1: "HOLD", 2: "UP"}
                deep_results[ticker] = {
                    "accuracy": eval_result["accuracy"],
                    "f1": eval_result["f1"],
                    "prediction": label_names[int(pred_class[0])],
                    "probabilities": {
                        "DOWN": round(float(pred_proba[0][0]), 3),
                        "HOLD": round(float(pred_proba[0][1]), 3),
                        "UP": round(float(pred_proba[0][2]), 3),
                    },
                    "source_weights": eval_result.get("source_weights", {}),
                    "data_size": train_info.get("data_size", 0),
                }
                print(f"    Acc: {eval_result['accuracy']:.1%} | F1: {eval_result['f1']:.3f} | Pred: {deep_results[ticker]['prediction']}")

            dl_done += 1
            dl_elapsed = time.time() - dl_start_time
            dl_per_coin = dl_elapsed / dl_done
            dl_remaining = dl_per_coin * (dl_total - dl_done)
            dl_eta = datetime.now() + __import__('datetime').timedelta(seconds=dl_remaining)
            print(f"    [{dl_done}/{dl_total}] {dl_elapsed:.0f}s elapsed | ETA: {dl_eta.strftime('%H:%M:%S')} ({dl_remaining:.0f}s left)")
            print(f"    Source weights: {json.dumps(deep_results[ticker].get('source_weights', {}), indent=0)}")

    except Exception as e:
        print(f"  [WARN] Deep learning failed: {e}")
        import traceback
        traceback.print_exc()

    # 5-Model Ensemble + GridSearch
    print("\n" + "=" * 70)
    print("  PHASE 3b: 5-MODEL ENSEMBLE + GRIDSEARCH (LGB/XGB/CB/RF/ET)")
    print("=" * 70)

    max_h = _cfg_max_h()
    lookback_bars = 7 * 24 // (bm // 60)  # ~7 days in bars (4h = 42 bars)
    loop = IterativeMaskingLoop(horizon=max_h, lookback=lookback_bars)

    ml_start_time = time.time()
    for i in range(1, num_iterations + 1):
        t0 = time.time()
        result = loop.run_iteration(feature_matrices, i)
        loop.save_iteration_result(result)
        loop.save_state()
        iter_time = time.time() - t0
        ml_elapsed = time.time() - ml_start_time
        avg_per_iter = ml_elapsed / i
        ml_remaining = avg_per_iter * (num_iterations - i)
        ml_eta = datetime.now() + __import__('datetime').timedelta(seconds=ml_remaining)
        print(f"  [Iter {i}/{num_iterations}] {iter_time:.0f}s | Total: {ml_elapsed:.0f}s | ETA: {ml_eta.strftime('%H:%M:%S')} ({ml_remaining:.0f}s left)")

    # Final report
    print("\n" + "=" * 70)
    print("  PHASE 4: FINAL REPORT")
    print("=" * 70)

    report = loop.generate_final_report()
    report["deep_learning_results"] = deep_results

    if deep_results:
        for ticker, dr in deep_results.items():
            if ticker in report.get("investment_grades", {}):
                report["investment_grades"][ticker]["deep_prediction"] = dr["prediction"]
                report["investment_grades"][ticker]["deep_accuracy"] = dr["accuracy"]
                report["investment_grades"][ticker]["deep_probabilities"] = dr.get("probabilities", {})

    return report


def format_report(report):
    lines = []
    lines.append("# Iterative Masking Loop v3 - Final Report")
    lines.append(f"**Generated**: {report.get('generated_at', datetime.now().isoformat())}")
    lines.append(f"**Iterations**: {report.get('total_iterations', 0)}")
    lines.append(f"**Ensemble**: LightGBM + XGBoost + CatBoost + RandomForest + ExtraTrees")
    lines.append(f"**DL**: MOMENT(MaskedPatchRecon) + NormLin + RevIN + MediaAttention + SelectiveLearning")
    lines.append(f"**Pretraining**: MOMENT — Selective Masking (random → hard-mining → curriculum)")
    lines.append(f"**Prediction**: multi-horizon (config-driven)")
    lines.append(f"**Label Scheme**: {json.dumps(report.get('label_scheme', {}))}")
    lines.append("")

    # Performance
    perf = report.get("performance_summary", {})
    lines.append("## 1. Algorithm Performance")
    lines.append("")
    lines.append("| Metric | Initial | Final | Best | Improvement |")
    lines.append("|--------|---------|-------|------|-------------|")
    ia, fa, ba = perf.get("initial_accuracy", 0), perf.get("final_accuracy", 0), perf.get("best_accuracy", 0)
    lines.append(f"| Accuracy | {ia:.1%} | {fa:.1%} | {ba:.1%} | {perf.get('accuracy_improvement_pct',0):+.1f}% |")
    if1, ff1, bf1 = perf.get("initial_f1", 0), perf.get("final_f1", 0), perf.get("best_f1", 0)
    lines.append(f"| F1 (macro) | {if1:.4f} | {ff1:.4f} | {bf1:.4f} | - |")
    lines.append(f"| Sharpe | - | {perf.get('final_sharpe',0):.4f} | - | - |")
    lines.append("")

    # Model Scores
    ms = report.get("model_scores", {})
    if ms:
        lines.append("### Individual Model Accuracy")
        lines.append("")
        lines.append("| Model | Accuracy |")
        lines.append("|-------|----------|")
        for m, s in sorted(ms.items(), key=lambda x: -x[1]):
            bar = chr(9608) * int(s * 30)
            lines.append(f"| {m} | {s:.1%} {bar} |")
        lines.append("")

    # GridSearch
    gsp = report.get("grid_search_params", {})
    if gsp:
        lines.append(f"**GridSearch Best Params**: {json.dumps(gsp)}")
        lines.append("")

    # Per-Horizon Accuracy (ML)
    # Extract from latest iteration predictions
    latest_preds = report.get("investment_grades", {})
    sample_horizons = None
    for t, g in latest_preds.items():
        hp = g.get("horizon_predictions", {})
        if hp:
            sample_horizons = hp
            break
    if sample_horizons:
        lines.append("### Per-Horizon Accuracy (ML Ensemble)")
        lines.append("")
        lines.append("| Horizon | Accuracy | F1 | Direction | Confidence |")
        lines.append("|---------|----------|----|-----------:|-----------:|")
        for h_name in sorted(sample_horizons.keys(), key=lambda x: int(x.replace("min",""))):
            hr = sample_horizons[h_name]
            lines.append(f"| {h_name} | {hr.get('accuracy',0):.1%} | {hr.get('f1',0):.3f} | {hr.get('direction','?')} | {hr.get('confidence',0):.1%} |")
        lines.append("")

    # Deep Learning
    dl = report.get("deep_learning_results", {})
    if dl:
        lines.append("## 2. Deep Learning Results (OLinear + RevIN)")
        lines.append("")
        lines.append("| Coin | Accuracy | F1 | Prediction | DOWN% | HOLD% | UP% | Data |")
        lines.append("|------|----------|----|-----------:|------:|------:|----:|-----:|")
        for t, r in sorted(dl.items()):
            p = r.get("probabilities", {})
            ds = r.get("data_size", "?")
            lines.append(f"| {t} | {r['accuracy']:.1%} | {r['f1']:.3f} | **{r['prediction']}** | {p.get('DOWN',0):.0%} | {p.get('HOLD',0):.0%} | {p.get('UP',0):.0%} | {ds} |")
        lines.append("")
        lines.append("*Architecture: RevIN + OLinear(NormLin) + Media Source Attention + Cross-Modal Fusion*")
        lines.append("")

        last_sw = {}
        for t, r in dl.items():
            if "source_weights" in r and r["source_weights"]:
                last_sw = r["source_weights"]
        if last_sw:
            lines.append("### Learned Media Source Weights (DL)")
            lines.append("")
            lines.append("| Source | DL Weight |")
            lines.append("|--------|-----------|")
            for s, w in sorted(last_sw.items(), key=lambda x: -x[1]):
                bar = chr(9608) * int(w * 60)
                lines.append(f"| {s} | {w:.1%} {bar} |")
            lines.append("")

    # Media Weights
    w = report.get("media_reliability_weights", {})
    lines.append("## 3. Media Reliability Weights (ML)")
    lines.append("")
    lines.append("| Source | Weight | Rank |")
    lines.append("|--------|--------|------|")
    for rank, (k, v) in enumerate(sorted(w.items(), key=lambda x: -x[1]), 1):
        bar = chr(9608) * int(v * 40)
        lines.append(f"| {k} | {v:.1%} {bar} | #{rank} |")
    lines.append("")

    noise = report.get("noise_sources_identified", [])
    if noise:
        lines.append(f"**Noise sources**: {', '.join(noise[:10])}")
        lines.append("")

    # Investment Grades
    grades = report.get("investment_grades", {})
    lines.append("## 4. Top 10 Crypto - Multi-Horizon Forecast & Investment Grade")
    lines.append("")
    lines.append("| Coin | ML Direction | ML Confidence | DL Direction | Grade |")
    lines.append("|------|-------------|---------------|--------------|-------|")
    grade_icon = {"Strong Buy": "[++]", "Buy": "[+]", "Hold": "[=]", "Sell": "[-]", "Strong Sell": "[--]"}
    for t, g in sorted(grades.items(), key=lambda x: -x[1].get("confidence", 0)):
        icon = grade_icon.get(g.get("grade", "Hold"), "[?]")
        dl_pred = g.get("deep_prediction", "-")
        lines.append(f"| {t} | {g.get('direction','?')} | {g.get('confidence',0):.0%} | {dl_pred} | {icon} {g.get('grade','Hold')} |")
    lines.append("")

    # Iteration Trends
    trends = report.get("iteration_trends", {})
    acc_t, f1_t, sh_t = trends.get("accuracy", []), trends.get("f1_macro", []), trends.get("sharpe", [])
    lines.append("## 5. Iteration Trends")
    lines.append("")
    lines.append("| Iter | Accuracy | F1 | Sharpe |")
    lines.append("|------|----------|------|--------|")
    for i in range(len(acc_t)):
        lines.append(f"| {i+1} | {acc_t[i]:.1%} | {f1_t[i]:.4f} | {sh_t[i]:.4f} |")
    lines.append("")

    # Top Features
    feats = report.get("top_features", {})
    if feats:
        lines.append("## 6. Top Predictive Features")
        lines.append("")
        lines.append("| Feature | Importance |")
        lines.append("|---------|------------|")
        for f, imp in list(feats.items())[:15]:
            bar = chr(9608) * int(imp * 80)
            lines.append(f"| {f} | {imp:.4f} {bar} |")
        lines.append("")

    # Backtest Summary
    lines.append("## 7. Backtest Summary")
    lines.append("")
    lines.append(f"- **Final Classification Accuracy**: {perf.get('final_accuracy', 0):.1%}")
    lines.append(f"- **Best Classification Accuracy**: {perf.get('best_accuracy', 0):.1%}")
    lines.append(f"- **Final F1 Score**: {perf.get('final_f1', 0):.4f}")
    lines.append(f"- **Sharpe Ratio**: {perf.get('final_sharpe', 0):.4f}")
    if dl:
        avg_dl_acc = sum(r["accuracy"] for r in dl.values()) / len(dl) if dl else 0
        lines.append(f"- **Deep Learning Avg Accuracy**: {avg_dl_acc:.1%}")
    lines.append(f"- **Models Used**: LightGBM, XGBoost, CatBoost, RandomForest, ExtraTrees")
    lines.append(f"- **Media Sources**: {len(w)} sources")
    lines.append("")

    return "\n".join(lines)


def main():
    print("=" * 70)
    print("  ITERATIVE MASKING LOOP PIPELINE v2")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  ML: 5-Model Ensemble + GridSearch")
    print(f"  DL: OLinear + RevIN + MediaSourceAttention")
    print(f"  Classification: UP / DOWN / HOLD")
    print("=" * 70)

    t0 = time.time()

    ohlcv, media, macro_aligned = collect_all_data()
    if not ohlcv:
        print("[FATAL] No OHLCV data. Aborting.")
        return

    report = run_loop(ohlcv, media, num_iterations=3, macro_aligned=macro_aligned)

    md = format_report(report)
    date_str = datetime.now().strftime("%Y%m%d")
    rdir = Path("data/reports") / date_str
    rdir.mkdir(parents=True, exist_ok=True)

    (rdir / "masking_loop_report.md").write_text(md, encoding="utf-8")
    with open(rdir / "masking_loop_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str, ensure_ascii=False)

    elapsed = time.time() - t0
    print(f"\n{'='*70}")
    print(f"  DONE! {elapsed:.0f}s ({elapsed/60:.1f}m)")
    print(f"  Report: {rdir / 'masking_loop_report.md'}")
    print(f"{'='*70}")
    print()
    print(md)


if __name__ == "__main__":
    main()
