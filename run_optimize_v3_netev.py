"""Walk-Forward v3.1 -- net EV 기반 재최적화.

v3 대비 변경:
1. 목적함수: stability(accuracy) → post-cost net EV (equity%)
2. CostModel 통합: maker/taker/funding/slippage/miss-fill
3. RegimeFilter 통합: 4상태 레짐별 EV 분리
4. exit 비대칭 R:R 확대 탐색 (k_upper/k_lower 비대칭 중심)
5. TradeAuditor로 최종 감사표 출력
6. 대상: XRP, DOT, ADA (3코인 압축)
"""

import sys
sys.path.insert(0, "C:/Users/RJ/Desktop/CLAUDE_CRYPTO_AGENT")
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None

import json
import time
import warnings
import traceback
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import ParameterSampler

warnings.filterwarnings("ignore")

from src.data.crawlers.crypto_ohlcv import fetch_all_top10
from src.data.crawlers.macro_commodity_crawler import crawl_all_macro_data
from src.data.crawlers.signal_features import add_signal_features
from src.models.masking_loop import (
    create_labels_triple_barrier, compute_extended_metrics,
    LABEL_MAP, LABEL_NAMES, HORIZONS, HORIZON_LABELS,
    STAGE1_NAMES, STAGE2_NAMES,
)
from src.models.enhanced_ensemble import EnhancedEnsemble
from src.models.regime_filter import RegimeFilter, Regime4
from src.execution.cost_model import CostModel, FeeSchedule, FundingConfig, MissFillConfig
from src.evaluation.trade_audit import TradeAuditor
from src.evaluation.trade_level_ev import compute_trade_level_ev
from src.utils.config import load_settings, bar_minutes as cfg_bar_minutes
from src.utils.feature_policy import is_excluded_feature

# ==================== 설정 ====================
DEADLINE = datetime(2026, 3, 17, 10, 30, 0)  # 내일 오전 10시 30분
REPORT_DIR = Path("data/reports/walkforward_v3_1")
REPORT_DIR.mkdir(parents=True, exist_ok=True)

SETTINGS = load_settings()
BM = cfg_bar_minutes()
MAX_HORIZON = max(HORIZONS)
PURGE_BARS = MAX_HORIZON * 2
EMBARGO_BARS = 6

# 3코인 압축 (BTC 제외, DOGE 관찰만)
ACTIVE_COINS = ["XRP", "DOT", "ADA"]

# 비용 모델 (Bybit VIP0 Perpetual)
COST_MODEL = CostModel(
    fee_schedule=FeeSchedule(
        maker_fee=0.0002,           # 0.02%
        taker_fee=0.00055,          # 0.055%
        slippage_entry=0.0003,      # 0.03%
        slippage_exit_limit=0.0001, # TP limit
        slippage_exit_market=0.0005, # SL market
    ),
    funding_config=FundingConfig(
        interval_hours=8.0,
        default_rate=0.0001,        # 0.01% per 8h
    ),
    miss_fill_config=MissFillConfig(
        reject_prob=0.15,           # Post-Only 15% reject
        missed_ev_pct=0.0015,       # 놓친 시그널 EV 0.15% (equity 소수)
    ),
)

REGIME_FILTER = RegimeFilter()
RISK_FRAC = 0.005  # 0.5% equity per trade

# 코인별 실측 ATR% (2026-03-16 yfinance 기준)
COIN_ATR_PCT = {
    "XRP": 0.0098,
    "DOT": 0.0133,
    "ADA": 0.0125,
    "BTC": 0.0089,
    "DOGE": 0.0131,
}

# 코인별 대략적 현재가격 (비용 계산용)
COIN_PRICES = {
    "XRP": 2.30,
    "DOT": 4.20,
    "ADA": 0.74,
    "BTC": 84000,
    "DOGE": 0.17,
}

# 확장된 파라미터 공간 -- 비대칭 R:R 중심 탐색
PARAM_SPACE = {
    "num_leaves": [15, 31, 47, 63],
    "learning_rate": [0.01, 0.02, 0.05, 0.08, 0.1],
    "min_child_samples": [3, 5, 10, 20, 30],
    "n_estimators": [100, 200, 300, 400],
    # 비대칭 R:R 탐색 확대 (k_upper > k_lower 선호)
    "k_upper": [1.0, 1.2, 1.5, 2.0, 2.5, 3.0],
    "k_lower": [0.6, 0.8, 1.0, 1.2, 1.5],
    "stage1_threshold": [0.40, 0.45, 0.50, 0.55, 0.60],
    "max_features": [80, 120, 150],
    "max_depth_tree": [6, 8, 10],
    "subsample": [0.7, 0.8, 0.9],
    "colsample": [0.6, 0.7, 0.8, 0.9],
}

# 데이터 윈도우 설정
TRAIN_DAYS_OPTIONS = [90, 120, 150, 180, 240, 300]
TEST_DAYS_OPTIONS = [21, 30, 45]
WINDOW_STRIDE_DAYS = 7


# ==================== 데이터 관리 (v3에서 재사용) ====================

class DataManager:
    def __init__(self):
        self.ohlcv = None
        self.macro = None
        self.feature_data = {}
        self.ohlcv_4h = {}  # 레짐 분석용

    def fetch_and_prepare(self):
        print(f"\n{'='*70}")
        print(f"  DATA COLLECTION + SIGNAL FEATURES")
        print(f"{'='*70}")

        self.ohlcv = fetch_all_top10("365d", "1h")
        print(f"  => {len(self.ohlcv)} coins loaded")

        first_coin = list(self.ohlcv.values())[0]
        self.macro = crawl_all_macro_data(first_coin.index)
        macro_aligned = self.macro.get("aligned", pd.DataFrame())

        for coin in ACTIVE_COINS:
            if coin not in self.ohlcv:
                print(f"  [SKIP] {coin}: no data")
                continue
            df = self.ohlcv[coin].copy()
            if len(macro_aligned) > 0:
                common_idx = df.index.intersection(macro_aligned.index)
                if len(common_idx) > 0:
                    for col in macro_aligned.columns:
                        df[col] = macro_aligned[col].reindex(df.index).ffill().bfill().fillna(0)
            self.feature_data[coin] = df

            # 4h resample for regime (레짐은 4h OHLCV에서 분석)
            # 칼럼명: lowercase (crypto_ohlcv.py 출력 기준)
            ohlcv_cols = {}
            for c in ["open", "high", "low", "close", "volume"]:
                if c in df.columns:
                    ohlcv_cols[c] = c
                elif c.capitalize() in df.columns:
                    ohlcv_cols[c] = c.capitalize()
            if len(ohlcv_cols) < 5:
                print(f"  [WARN] {coin}: OHLCV columns not found for regime, skipping 4h resample")
                self.ohlcv_4h[coin] = pd.DataFrame()
                continue
            raw_ohlcv = df[[ohlcv_cols["open"], ohlcv_cols["high"],
                            ohlcv_cols["low"], ohlcv_cols["close"],
                            ohlcv_cols["volume"]]].copy()
            raw_ohlcv.columns = ["Open", "High", "Low", "Close", "Volume"]
            ohlcv_4h = raw_ohlcv.resample("4h").agg({
                "Open": "first", "High": "max", "Low": "min",
                "Close": "last", "Volume": "sum",
            }).dropna()
            self.ohlcv_4h[coin] = ohlcv_4h
            print(f"  {coin}: {len(df)} bars (1h), {len(ohlcv_4h)} bars (4h), {len(df.columns)} features")

        return self.feature_data

    def generate_augmented_windows(self):
        windows = []
        sample_df = list(self.feature_data.values())[0]
        idx = sample_df.index
        data_start, data_end = idx[0], idx[-1]
        total_days = (data_end - data_start).days
        purge_td = timedelta(hours=(PURGE_BARS + EMBARGO_BARS) * (BM // 60))

        wid = 0
        for train_days in TRAIN_DAYS_OPTIONS:
            for test_days in TEST_DAYS_OPTIONS:
                if train_days + test_days > total_days:
                    continue
                test_end = data_end
                while True:
                    test_start = test_end - timedelta(days=test_days)
                    train_end = test_start - purge_td
                    train_start = train_end - timedelta(days=train_days)
                    if train_start < data_start:
                        break
                    train_mask = (idx >= train_start) & (idx <= train_end)
                    test_mask = (idx >= test_start) & (idx <= test_end)
                    if train_mask.sum() < 80 or test_mask.sum() < 10:
                        test_end -= timedelta(days=WINDOW_STRIDE_DAYS)
                        continue
                    wid += 1
                    windows.append({
                        "name": f"w{wid}_tr{train_days}d_te{test_days}d",
                        "train_start": train_start, "train_end": train_end,
                        "test_start": test_start, "test_end": test_end,
                        "train_days": train_days, "test_days": test_days,
                    })
                    test_end -= timedelta(days=WINDOW_STRIDE_DAYS)

        unique = []
        seen = set()
        for w in windows:
            key = (w["test_start"].strftime("%Y-%m-%d"), w["train_days"])
            if key not in seen:
                seen.add(key)
                unique.append(w)
        unique.sort(key=lambda w: w["test_end"], reverse=True)

        if len(unique) > 15:
            sampled = []
            seen_configs = set()
            for w in unique:
                cfg_key = ((w["train_days"], w["test_days"]),
                           w["test_end"].strftime("%Y-%m"))
                if cfg_key not in seen_configs:
                    seen_configs.add(cfg_key)
                    sampled.append(w)
                if len(sampled) >= 15:
                    break
            unique = sampled

        print(f"\n  {len(unique)} augmented windows generated")
        return unique

    def split_data(self, coin_df, window):
        train_mask = (coin_df.index >= window["train_start"]) & (coin_df.index <= window["train_end"])
        test_mask = (coin_df.index >= window["test_start"]) & (coin_df.index <= window["test_end"])
        train_df, test_df = coin_df[train_mask], coin_df[test_mask]

        if len(train_df) > 0 and len(test_df) > 0:
            train_last, test_first = train_df.index[-1], test_df.index[0]
            gap = len(coin_df[(coin_df.index > train_last) & (coin_df.index < test_first)])
            if gap < PURGE_BARS:
                cut = PURGE_BARS - gap
                if len(train_df) > cut + 60:
                    train_df = train_df.iloc[:-cut]
                else:
                    return pd.DataFrame(), pd.DataFrame()
            overlap = train_df.index.intersection(test_df.index)
            if len(overlap) > 0:
                train_df = train_df[~train_df.index.isin(overlap)]

        return train_df, test_df


# ==================== 모델 학습 (v3 호환) ====================

class ModelTrainerV3:
    def __init__(self, lightweight=True):
        self.lightweight = lightweight

    def train_and_evaluate(self, train_df, test_df, params, coin="", horizon=None):
        h = horizon or HORIZONS[-1]
        bm = BM
        h_label = f"label_{h*bm}min"

        if len(train_df) > 0 and len(test_df) > 0:
            assert train_df.index[-1] < test_df.index[0], \
                f"[LEAK] train({train_df.index[-1]}) >= test({test_df.index[0]})"

        k_upper = params.get("k_upper", 1.5)
        k_lower = params.get("k_lower", 1.5)

        train_labeled = create_labels_triple_barrier(
            train_df.copy(), h, k_upper_override=k_upper, k_lower_override=k_lower, verbose=False)
        test_labeled = create_labels_triple_barrier(
            test_df.copy(), h, k_upper_override=k_upper, k_lower_override=k_lower, verbose=False)

        if h_label not in train_labeled.columns or h_label not in test_labeled.columns:
            return None

        exclude = {"label", "future_return", "open", "high", "low", "close", "volume"}
        for hh in HORIZONS:
            exclude.add(f"label_{hh*bm}min")
            exclude.add(f"return_{hh*bm}min")
        leak_keywords = ["future", "target", "label_", "return_", "fwd_", "forward_"]

        feature_cols = []
        for c in train_labeled.columns:
            if c in exclude:
                continue
            if any(kw in c.lower() for kw in leak_keywords):
                continue
            if is_excluded_feature(c):
                continue
            if train_labeled[c].dtype not in [np.float64, np.float32, np.int64, np.int32, float, int]:
                continue
            feature_cols.append(c)

        max_feat = params.get("max_features", 150)
        if len(feature_cols) > max_feat:
            from sklearn.feature_selection import mutual_info_classif
            X_mi = train_labeled[feature_cols].replace([np.inf, -np.inf], 0).fillna(0).values
            y_mi = train_labeled[h_label].values
            n_mi = min(2000, len(X_mi))
            mi = mutual_info_classif(X_mi[:n_mi], y_mi[:n_mi],
                                     discrete_features=False, random_state=42, n_neighbors=5)
            top_idx = np.argsort(mi)[-max_feat:]
            feature_cols = [feature_cols[i] for i in sorted(top_idx)]

        train_clean = train_labeled.replace([np.inf, -np.inf], np.nan).ffill().bfill()
        test_clean = test_labeled.replace([np.inf, -np.inf], np.nan).ffill().bfill()

        X_train = train_clean[feature_cols].fillna(0).values
        X_test = test_clean[feature_cols].fillna(0).values
        y_train_3c = train_clean[h_label].fillna(1).values.astype(int)
        y_test_3c = test_clean[h_label].fillna(1).values.astype(int)

        if len(X_train) < 60 or len(X_test) < 5:
            return None

        # === Stage 1: Trade/NoTrade ===
        y_s1_train = (y_train_3c != LABEL_MAP["HOLD"]).astype(int)
        y_s1_test = (y_test_3c != LABEL_MAP["HOLD"]).astype(int)

        if len(np.unique(y_s1_train)) < 2:
            return None

        s1_counts = np.bincount(y_s1_train, minlength=2)
        s1_sw = np.where(s1_counts > 0, len(y_s1_train) / (2 * s1_counts + 1e-10), 1.0)[y_s1_train]

        if self.lightweight:
            # 최적화용 경량: LightGBM만 사용 (GPU, 빠름)
            from lightgbm import LGBMClassifier
            ens_s1 = LGBMClassifier(
                n_estimators=params.get("n_estimators", 200),
                num_leaves=params.get("num_leaves", 31),
                learning_rate=params.get("learning_rate", 0.05),
                min_child_samples=params.get("min_child_samples", 10),
                subsample=params.get("subsample", 0.8),
                colsample_bytree=params.get("colsample", 0.8),
                device="gpu", gpu_use_dp=False, verbose=-1,
                is_unbalance=True, random_state=42,
            )
        else:
            ens_s1 = EnhancedEnsemble(n_classes=2, use_stacking=True, verbose=False)
        try:
            ens_s1.fit(X_train, y_s1_train, sample_weight=s1_sw)
        except Exception:
            return None

        s1_probs_raw = ens_s1.predict_proba(X_test)
        # LGBMClassifier returns ndarray, EnhancedEnsemble too
        s1_probs = s1_probs_raw if s1_probs_raw.ndim == 2 else np.column_stack([1-s1_probs_raw, s1_probs_raw])
        threshold = params.get("stage1_threshold", 0.5)
        s1_pred = (s1_probs[:, 1] >= threshold).astype(int)
        s1_metrics = compute_extended_metrics(y_s1_test, s1_pred, s1_probs,
                                              num_classes=2, label_names=STAGE1_NAMES)

        # === Stage 2: Long/Short ===
        trade_mask_train = y_train_3c != LABEL_MAP["HOLD"]
        trade_mask_test = s1_pred == 1

        s2_metrics = None
        final_pred = np.full(len(X_test), LABEL_MAP["HOLD"])

        if trade_mask_train.sum() >= 30 and len(np.unique((y_train_3c[trade_mask_train] == LABEL_MAP["UP"]).astype(int))) >= 2:
            X_s2_train = X_train[trade_mask_train]
            y_s2_train = (y_train_3c[trade_mask_train] == LABEL_MAP["UP"]).astype(int)
            s2_counts = np.bincount(y_s2_train, minlength=2)
            s2_sw = np.where(s2_counts > 0, len(y_s2_train) / (2 * s2_counts + 1e-10), 1.0)[y_s2_train]

            if self.lightweight:
                from lightgbm import LGBMClassifier
                ens_s2 = LGBMClassifier(
                    n_estimators=params.get("n_estimators", 200),
                    num_leaves=params.get("num_leaves", 31),
                    learning_rate=params.get("learning_rate", 0.05),
                    min_child_samples=params.get("min_child_samples", 10),
                    subsample=params.get("subsample", 0.8),
                    colsample_bytree=params.get("colsample", 0.8),
                    device="gpu", gpu_use_dp=False, verbose=-1,
                    is_unbalance=True, random_state=42,
                )
            else:
                ens_s2 = EnhancedEnsemble(n_classes=2, use_stacking=True, verbose=False)
            try:
                ens_s2.fit(X_s2_train, y_s2_train, sample_weight=s2_sw)
                s2_probs_raw = ens_s2.predict_proba(X_test)
                s2_probs = s2_probs_raw if s2_probs_raw.ndim == 2 else np.column_stack([1-s2_probs_raw, s2_probs_raw])

                if trade_mask_test.sum() > 0:
                    s2_pred_trade = np.argmax(s2_probs[trade_mask_test], axis=1)
                    trade_indices = np.where(trade_mask_test)[0]
                    for idx, s2 in zip(trade_indices, s2_pred_trade):
                        final_pred[idx] = LABEL_MAP["UP"] if s2 == 1 else LABEL_MAP["DOWN"]

                    y_s2_test_real = (y_test_3c[trade_mask_test] == LABEL_MAP["UP"]).astype(int)
                    if len(np.unique(y_s2_test_real)) >= 2:
                        s2_metrics = compute_extended_metrics(
                            y_s2_test_real, s2_pred_trade,
                            s2_probs[trade_mask_test],
                            num_classes=2, label_names=STAGE2_NAMES)
            except Exception:
                pass

        combined = compute_extended_metrics(y_test_3c, final_pred, num_classes=3)

        # Trade-level EV (replaces summary formula)
        s2_pred_all = np.zeros(len(X_test), dtype=int)
        s2_prob_all = np.full(len(X_test), 0.5)
        if 's2_probs' in dir():
            s2_pred_all = np.argmax(s2_probs, axis=1) if s2_probs is not None else s2_pred_all
            s2_prob_all = s2_probs[:, 1] if s2_probs is not None else s2_prob_all

        trade_ev = compute_trade_level_ev(
            test_df=test_clean,
            s1_pred=s1_pred,
            s1_prob=s1_probs[:, 1],
            s2_pred=s2_pred_all,
            s2_prob=s2_prob_all,
            k_upper=k_upper, k_lower=k_lower,
            max_hold=HORIZONS[-1],
            risk_frac=RISK_FRAC,
            cost_model=COST_MODEL,
        )

        return {
            "stage1_balanced_accuracy": s1_metrics["balanced_accuracy"],
            "stage2_balanced_accuracy": s2_metrics["balanced_accuracy"] if s2_metrics else None,
            "combined_balanced_accuracy": combined["balanced_accuracy"],
            "combined_mcc": combined.get("mcc", 0),
            "combined_f1": combined.get("f1_macro", 0),
            "trade_ratio": float(trade_mask_test.mean()),
            "n_train": len(X_train),
            "n_test": len(X_test),
            "n_features": len(feature_cols),
            # Trade-level metrics (NEW)
            "tl_trade_count": trade_ev["trade_count"],
            "tl_avg_net_pnl": trade_ev["avg_net_pnl"],
            "tl_total_net_pnl": trade_ev["total_net_pnl"],
            "tl_win_rate": trade_ev["win_rate"],
            "tl_max_dd": trade_ev["max_dd"],
            "tl_cost_share": trade_ev["cost_share"],
            "tl_score": trade_ev["score"],
        }


# ==================== net EV 기반 최적화 ====================

def compute_net_ev_score(
    s2_accuracy: float,
    k_upper: float,
    k_lower: float,
    coin: str,
    std_combined: float = 0.0,
    mcc: float = 0.0,
) -> dict:
    """net EV + 안정성 복합 스코어."""
    atr_pct = COIN_ATR_PCT.get(coin, 0.01)
    price = COIN_PRICES.get(coin, 1.0)

    ev_data = COST_MODEL.compute_net_ev(
        s2_accuracy=s2_accuracy,
        k_upper=k_upper,
        k_lower=k_lower,
        atr_pct=atr_pct,
        entry_price=price,
        risk_frac=RISK_FRAC,
    )

    ev_net = ev_data["ev_net_eq"]
    bep = ev_data["bep"]
    margin = ev_data["margin_pct"]

    # 복합 스코어: net EV 중심 + 안정성 보정
    # net EV > 0인 파라미터만 의미있음
    # std가 작을수록 안정적 → 페널티 적용
    if ev_net > 0:
        score = ev_net * 10000 * (1 - 0.5 * std_combined) + mcc * 10
    else:
        score = ev_net * 10000 - 100  # 강한 페널티

    return {
        "ev_net": ev_net,
        "ev_gross": ev_data["ev_gross_eq"],
        "cost_total": ev_data["cost"].total_eq,
        "bep": bep,
        "margin": margin,
        "rr": ev_data["rr"],
        "score": score,
    }


class CoinOptimizer:
    def __init__(self, data_mgr, trainer):
        self.data_mgr = data_mgr
        self.trainer = trainer
        self.coin_results = {}
        self.coin_best = {}
        self.global_results = []
        self.round_num = 0
        self._load_previous_checkpoint()

    def _load_previous_checkpoint(self):
        """이전 체크포인트에서 best 결과와 라운드 번호 로드."""
        latest_r = 0
        latest_path = None
        for p in REPORT_DIR.glob("wf3_1_checkpoint_r*.json"):
            try:
                r = int(p.stem.split("_r")[-1])
                if r > latest_r:
                    latest_r = r
                    latest_path = p
            except ValueError:
                continue
        if latest_path and latest_path.exists():
            with open(latest_path) as f:
                prev = json.load(f)
            for coin, best in prev.get("coin_best", {}).items():
                if coin in ACTIVE_COINS:
                    self.coin_best[coin] = best
            prev_evals = prev.get("total_evaluations", 0)
            self.round_num = latest_r
            summary = ', '.join(f'{c}(score={b.get("score","?")})' for c, b in self.coin_best.items())
            print(f"  [RESUME] R{latest_r} loaded, {prev_evals} prev evals")
            print(f"  [RESUME] Best: {summary}")
            print(f"  [RESUME] Continuing from R{latest_r + 1}")

    def optimize_coin(self, coin, windows, n_iter=10, seed=None):
        seed = seed or (42 + hash(coin) % 1000)
        param_list = list(ParameterSampler(PARAM_SPACE, n_iter=n_iter, random_state=seed))
        coin_df = self.data_mgr.feature_data[coin]
        results = []

        for pi, params in enumerate(param_list):
            params_dict = dict(params)

            # R:R 필터: k_upper/k_lower < 0.5면 skip (구조적 불리)
            rr = params_dict["k_upper"] / params_dict["k_lower"]
            if rr < 0.6:
                print(f"    [{coin}] P{pi+1}/{n_iter} | SKIP R:R={rr:.2f} < 0.6")
                continue

            window_scores = []
            t_param_start = time.time()

            for wi, window in enumerate(windows):
                if datetime.now() >= DEADLINE:
                    break
                train_df, test_df = self.data_mgr.split_data(coin_df, window)
                if len(train_df) < 80 or len(test_df) < 8:
                    continue
                try:
                    metrics = self.trainer.train_and_evaluate(
                        train_df, test_df, params_dict, coin=coin, horizon=HORIZONS[-1])
                except Exception as e:
                    print(f"      [{coin}] P{pi+1} W{wi+1}/{len(windows)} ERROR: {e}")
                    continue

                if metrics:
                    results.append({"params": params_dict, "window": window["name"],
                                    "coin": coin, **metrics})
                    window_scores.append(metrics)

            if window_scores:
                s1_avg = np.mean([m["stage1_balanced_accuracy"] for m in window_scores])
                s2_scores = [m["stage2_balanced_accuracy"] for m in window_scores
                             if m["stage2_balanced_accuracy"] is not None]
                s2_avg = np.mean(s2_scores) if s2_scores else 0
                comb_avg = np.mean([m["combined_balanced_accuracy"] for m in window_scores])
                comb_std = np.std([m["combined_balanced_accuracy"] for m in window_scores])
                mcc_avg = np.mean([m["combined_mcc"] for m in window_scores])

                # ★ net EV 기반 스코어
                ev_info = compute_net_ev_score(
                    s2_accuracy=s2_avg,
                    k_upper=params_dict["k_upper"],
                    k_lower=params_dict["k_lower"],
                    coin=coin,
                    std_combined=comb_std,
                    mcc=mcc_avg,
                )

                elapsed_p = time.time() - t_param_start
                print(f"    [{coin}] P{pi+1}/{n_iter} | S1:{s1_avg:.1%} S2:{s2_avg:.1%} "
                      f"netEV:{ev_info['ev_net']:+.4%} cost:{ev_info['cost_total']:.4%} "
                      f"R:R={ev_info['rr']:.2f} BEP:{ev_info['bep']:.1%} margin:{ev_info['margin']:+.1%}p "
                      f"MCC:{mcc_avg:.3f} ({len(window_scores)} wins) | "
                      f"k={params_dict['k_upper']}/{params_dict['k_lower']} "
                      f"th={params_dict['stage1_threshold']} "
                      f"[{elapsed_p:.0f}s]")

        return results

    def run_full_optimization(self):
        coins = [c for c in ACTIVE_COINS if c in self.data_mgr.feature_data]
        windows = self.data_mgr.generate_augmented_windows()

        self.round_num = 0
        while datetime.now() < DEADLINE:
            self.round_num += 1
            remaining = (DEADLINE - datetime.now()).total_seconds() / 3600
            if remaining < 0.5:
                break

            n_iter = max(5, min(12, int(remaining * 0.8)))
            seed_offset = self.round_num * 1000

            print(f"\n{'='*70}")
            print(f"  === Round {self.round_num} === (n_iter={n_iter}/coin, "
                  f"{len(windows)} windows, {remaining:.1f}h remaining)")
            print(f"  Objective: net EV (post-cost equity%)")
            print(f"{'='*70}")

            for ci, coin in enumerate(coins):
                if datetime.now() >= DEADLINE - timedelta(minutes=30):
                    break
                print(f"\n  >> {coin} ({ci+1}/{len(coins)})")
                results = self.optimize_coin(coin, windows, n_iter=n_iter,
                                             seed=42 + seed_offset + ci)
                if results:
                    self.coin_results.setdefault(coin, []).extend(results)
                    self.global_results.extend(results)

            self._update_coin_bests()
            self._checkpoint()

        return self.coin_best

    def _update_coin_bests(self):
        print(f"\n  {'='*60}")
        print(f"  COIN-LEVEL BEST (Round {self.round_num}) -- net EV ranking")
        print(f"  {'='*60}")

        for coin, results in self.coin_results.items():
            param_groups = {}
            for r in results:
                key = json.dumps(r["params"], sort_keys=True)
                param_groups.setdefault(key, []).append(r)

            rankings = []
            for key, group in param_groups.items():
                combined = [r["combined_balanced_accuracy"] for r in group]
                s1 = [r["stage1_balanced_accuracy"] for r in group]
                s2 = [r["stage2_balanced_accuracy"] for r in group
                      if r["stage2_balanced_accuracy"] is not None]
                mcc = [r["combined_mcc"] for r in group]

                mean_c = np.mean(combined)
                std_c = np.std(combined)
                mean_s2 = np.mean(s2) if s2 else 0
                mean_mcc = np.mean(mcc)

                params = json.loads(key)

                # ★ Trade-level EV (v2: replaces summary formula)
                tl_scores = [r.get("tl_score", 0) for r in group]
                tl_avg_pnls = [r.get("tl_avg_net_pnl", 0) for r in group]
                tl_trades = [r.get("tl_trade_count", 0) for r in group]
                tl_win_rates = [r.get("tl_win_rate", 0) for r in group]
                tl_max_dds = [r.get("tl_max_dd", 0) for r in group]

                mean_tl_score = np.mean(tl_scores) if tl_scores else 0
                mean_tl_avg_pnl = np.mean(tl_avg_pnls) if tl_avg_pnls else 0
                mean_tl_trades = np.mean(tl_trades) if tl_trades else 0

                # Fallback to summary if trade-level unavailable
                if mean_tl_trades > 0:
                    score = mean_tl_score
                else:
                    ev_info = compute_net_ev_score(
                        s2_accuracy=mean_s2,
                        k_upper=params["k_upper"],
                        k_lower=params["k_lower"],
                        coin=coin,
                        std_combined=std_c,
                        mcc=mean_mcc,
                    )
                    score = ev_info.get("score", 0)

                rankings.append({
                    "params": params,
                    "mean_s1": round(np.mean(s1), 4),
                    "mean_s2": round(mean_s2, 4),
                    "mean_combined": round(mean_c, 4),
                    "std_combined": round(std_c, 4),
                    "mean_mcc": round(mean_mcc, 4),
                    "tl_avg_net_pnl": round(mean_tl_avg_pnl, 6),
                    "tl_avg_trades": round(mean_tl_trades, 1),
                    "tl_win_rate": round(np.mean(tl_win_rates), 4) if tl_win_rates else 0,
                    "tl_max_dd": round(np.mean(tl_max_dds), 6) if tl_max_dds else 0,
                    "ev_gross": round(ev_info["ev_gross"], 6),
                    "cost_total": round(ev_info["cost_total"], 6),
                    "bep": round(ev_info["bep"], 4),
                    "score": round(score, 4),
                    "n_evals": len(group),
                })

            # ★ Trade-level score로 정렬
            rankings.sort(key=lambda x: -x["score"])

            if rankings:
                best = rankings[0]
                prev_score = self.coin_best.get(coin, {}).get("score", -999)
                if best["score"] >= prev_score:
                    tag = " * NEW BEST" if best["score"] > prev_score else ""
                    self.coin_best[coin] = best
                else:
                    tag = f" (prev {prev_score:.1f} better, kept)"
                print(f"  {coin:5s} | tl_avgPnL:{best['tl_avg_net_pnl']:+.4%} "
                      f"trades:{best['tl_avg_trades']:.0f} win:{best['tl_win_rate']:.1%} "
                      f"dd:{best['tl_max_dd']:.4%} score:{best['score']:.1f} "
                      f"| S2:{best['mean_s2']:.1%} MCC:{best['mean_mcc']:.3f} "
                      f"({best['n_evals']} evals){tag}")

    def _checkpoint(self):
        report = {
            "version": "v3.2_trade_level",
            "round": self.round_num,
            "timestamp": datetime.now().isoformat(),
            "total_evaluations": len(self.global_results),
            "objective": "post-cost net EV (equity%)",
            "cost_model": {
                "maker_fee": COST_MODEL.fees.maker_fee,
                "taker_fee": COST_MODEL.fees.taker_fee,
                "slippage_entry": COST_MODEL.fees.slippage_entry,
                "slippage_exit_market": COST_MODEL.fees.slippage_exit_market,
                "funding_rate": COST_MODEL.funding.default_rate,
                "miss_fill_prob": COST_MODEL.miss_fill.reject_prob,
            },
            "coin_best": {},
        }
        for coin, best in self.coin_best.items():
            report["coin_best"][coin] = best

        path = REPORT_DIR / f"wf3_1_checkpoint_r{self.round_num}.json"
        with open(path, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"  [CHECKPOINT] {path}")

    def run_final_audit(self):
        """최종 감사표 출력."""
        print(f"\n{'='*70}")
        print(f"  FINAL TRADE AUDIT")
        print(f"{'='*70}")

        auditor = TradeAuditor(
            cost_model=COST_MODEL,
            regime_filter=REGIME_FILTER,
            risk_frac=RISK_FRAC,
            bar_minutes=BM,
        )

        coins_data = {}
        for coin, best in self.coin_best.items():
            coins_data[coin] = {
                "s2": best["mean_s2"],
                "mcc": best["mean_mcc"],
                "k_upper": best["params"]["k_upper"],
                "k_lower": best["params"]["k_lower"],
                "s1": best["mean_s1"],
            }

        results = auditor.audit_portfolio(
            coins=coins_data,
            atr_data=COIN_ATR_PCT,
            price_data=COIN_PRICES,
            ohlcv_data=self.data_mgr.ohlcv_4h,
        )

        report = auditor.print_audit_report(results)
        print(report)

        # Save
        path = REPORT_DIR / "final_audit_report.txt"
        with open(path, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"  [SAVED] {path}")

        return results


# ==================== MAIN ====================

def main():
    print(f"\n{'='*70}")
    print(f"  WALK-FORWARD v3.1 -- net EV Optimization")
    print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Deadline: {DEADLINE.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Remaining: {(DEADLINE - datetime.now()).total_seconds()/3600:.1f} hours")
    print(f"  Coins: {ACTIVE_COINS}")
    print(f"  Objective: post-cost net EV (equity%)")
    print(f"  Cost model: Bybit VIP0 (maker={COST_MODEL.fees.maker_fee:.4%} "
          f"taker={COST_MODEL.fees.taker_fee:.4%})")
    print(f"  Risk frac: {RISK_FRAC:.1%} per trade")
    print(f"{'='*70}")

    start_time = time.time()

    # 1. 데이터 수집
    data_mgr = DataManager()
    data_mgr.fetch_and_prepare()

    # 2. 최적화 (lightweight=True: LightGBM only, 7x faster)
    trainer = ModelTrainerV3(lightweight=True)
    optimizer = CoinOptimizer(data_mgr, trainer)
    coin_best = optimizer.run_full_optimization()

    # 3. 최종 감사표
    audit_results = optimizer.run_final_audit()

    elapsed = time.time() - start_time
    print(f"\n{'='*70}")
    print(f"  COMPLETED in {elapsed/3600:.1f}h")
    print(f"  Total evaluations: {len(optimizer.global_results)}")
    print(f"  Results: {REPORT_DIR}")
    print(f"{'='*70}")


if __name__ == "__main__":
    import asyncio
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    # Force unbuffered output for log file redirect
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, line_buffering=True)
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, line_buffering=True)
    try:
        main()
    except Exception as e:
        import traceback
        print(f"\n[FATAL ERROR] {e}")
        traceback.print_exc()
        sys.exit(1)
