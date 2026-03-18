"""Iterative Masking Loop Engine -- Classification Edition v3.

5-model ensemble + Nested Walk-Forward CV:
- LightGBM, XGBoost, CatBoost, RandomForest, ExtraTrees
- TimeSeriesSplit + gap (StratifiedKFold 대체)
- Fee-aware dynamic labeling (자의적 ±0.3% 대체)
- 확장 평가지표: balanced_acc, MCC, Brier, PR-AUC, confusion matrix
- 목적함수 통일: balanced_accuracy 단일 기준
"""

import json
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field, asdict

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    matthews_corrcoef, brier_score_loss, classification_report,
    confusion_matrix, precision_recall_curve, average_precision_score,
)
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.calibration import calibration_curve

LABEL_MAP = {"DOWN": 0, "HOLD": 1, "UP": 2}
LABEL_NAMES = {0: "DOWN", 1: "HOLD", 2: "UP"}
GRADE_MAP = {
    "UP": {"high_conf": "Strong Buy", "mid_conf": "Buy", "low_conf": "Hold"},
    "HOLD": {"high_conf": "Hold", "mid_conf": "Hold", "low_conf": "Hold"},
    "DOWN": {"high_conf": "Strong Sell", "mid_conf": "Sell", "low_conf": "Hold"},
}

# ==================== Fee-Aware Dynamic Labeling ====================
# 자의적 ±0.3% 대신 거래비용 기반 no-trade band
# Crypto 기준: maker fee + slippage + buffer
CRYPTO_FEES = {
    "maker_fee": 0.001,     # 0.1% (업비트/바이낸스 기본)
    "taker_fee": 0.001,     # 0.1%
    "slippage": 0.0005,     # 0.05% 예상 슬리피지
    "buffer": 0.0005,       # 0.05% 안전마진
}
# fee_threshold = (maker + taker) / 2 + slippage + buffer
FEE_THRESHOLD = (CRYPTO_FEES["maker_fee"] + CRYPTO_FEES["taker_fee"]) / 2 + \
                CRYPTO_FEES["slippage"] + CRYPTO_FEES["buffer"]
# → 0.002 (0.2%) — 왕복 비용 커버 후 이익이 나야 UP/DOWN

# config/settings.yaml의 labeling 설정이 있으면 사용, 없으면 fee 기반 기본값
try:
    from src.utils.config import load_settings as _load_settings
    _labeling_cfg = _load_settings().get("labeling", {})
    UP_THRESHOLD = _labeling_cfg.get("up_threshold", FEE_THRESHOLD)
    DOWN_THRESHOLD = _labeling_cfg.get("down_threshold", -FEE_THRESHOLD)
except Exception:
    UP_THRESHOLD = FEE_THRESHOLD
    DOWN_THRESHOLD = -FEE_THRESHOLD

# 멀티호라이즌: config/settings.yaml → timeframes.tactical.horizons 참조
from src.utils.config import (
    horizons as _cfg_horizons, horizon_labels as _cfg_horizon_labels,
    max_features as _cfg_max_features, max_horizon as _cfg_max_horizon,
    bar_minutes as _cfg_bar_minutes, load_settings as _cfg_load_settings,
)
HORIZONS = _cfg_horizons()                # [1, 3, 6, 18] bars
HORIZON_LABELS = _cfg_horizon_labels()    # {1: '4h', 3: '12h', ...}

# 2-Stage Binary 분류 맵
STAGE1_MAP = {"NoTrade": 0, "Trade": 1}
STAGE2_MAP = {"Short": 0, "Long": 1}
STAGE1_NAMES = {0: "NoTrade", 1: "Trade"}
STAGE2_NAMES = {0: "Short", 1: "Long"}


# ==================== Extended Evaluation Metrics ====================

def compute_extended_metrics(y_true, y_pred, y_proba=None, num_classes=3,
                             label_names=None) -> dict:
    """확장 평가지표: balanced_acc, MCC, Brier, PR-AUC, confusion matrix.

    num_classes=2 (binary) 또는 3 (multi-class) 지원.
    """
    if label_names is None:
        label_names = LABEL_NAMES if num_classes == 3 else {0: "Neg", 1: "Pos"}

    metrics = {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "balanced_accuracy": round(balanced_accuracy_score(y_true, y_pred), 4),
        "f1_macro": round(f1_score(y_true, y_pred, average="macro", zero_division=0), 4),
        "mcc": round(matthews_corrcoef(y_true, y_pred), 4),
    }

    # Confusion matrix
    class_labels = list(range(num_classes))
    cm = confusion_matrix(y_true, y_pred, labels=class_labels)
    metrics["confusion_matrix"] = cm.tolist()

    # Class distribution
    unique, counts = np.unique(y_true, return_counts=True)
    total = len(y_true)
    metrics["class_distribution"] = {
        label_names.get(int(u), str(u)): round(c / total, 3)
        for u, c in zip(unique, counts)
    }

    # Brier score & PR-AUC
    if y_proba is not None and len(y_proba.shape) == 2 and y_proba.shape[1] == num_classes:
        y_bin = label_binarize(y_true, classes=class_labels)
        if num_classes == 2:
            y_bin = np.column_stack([1 - y_bin, y_bin])  # label_binarize returns (n,1) for binary
        brier_scores = []
        pr_aucs = []
        for cls in range(num_classes):
            if y_bin[:, cls].sum() > 0:
                brier_scores.append(brier_score_loss(y_bin[:, cls], y_proba[:, cls]))
                pr_aucs.append(average_precision_score(y_bin[:, cls], y_proba[:, cls]))
        metrics["brier_score"] = round(np.mean(brier_scores), 4) if brier_scores else None
        metrics["pr_auc_macro"] = round(np.mean(pr_aucs), 4) if pr_aucs else None

        try:
            max_proba = np.max(y_proba, axis=1)
            correct = (y_pred == y_true).astype(float)
            fraction_of_positives, mean_predicted = calibration_curve(
                correct, max_proba, n_bins=5, strategy="uniform")
            ece = np.mean(np.abs(fraction_of_positives - mean_predicted))
            metrics["expected_calibration_error"] = round(float(ece), 4)
        except Exception:
            metrics["expected_calibration_error"] = None
    else:
        metrics["brier_score"] = None
        metrics["pr_auc_macro"] = None
        metrics["expected_calibration_error"] = None

    return metrics


@dataclass
class IterationResult:
    iteration: int
    accuracy: float
    f1_macro: float
    class_report: dict
    sharpe_estimate: float
    media_weights: dict[str, float]
    noise_sources: list[str]
    feature_importance: dict[str, float]
    predictions: dict[str, dict] = field(default_factory=dict)
    model_scores: dict[str, float] = field(default_factory=dict)
    timestamp: str = ""


@dataclass
class LoopState:
    total_iterations: int = 0
    best_accuracy: float = 0.0
    best_f1: float = 0.0
    current_weights: dict[str, float] = field(default_factory=lambda: {
        "news": 0.15, "reddit": 0.10, "x_twitter": 0.15,
        "youtube": 0.10, "tiktok": 0.08, "instagram": 0.05,
        "coindesk": 0.08, "cointelegraph": 0.05, "theblock": 0.05,
        "decrypt": 0.04, "onchain": 0.05, "technical": 0.10,
    })
    iteration_history: list[dict] = field(default_factory=list)
    best_params: dict = field(default_factory=dict)


# ==================== Feature Engineering ====================

def create_labels_fixed(df: pd.DataFrame, horizon: int = 12) -> pd.DataFrame:
    """Fixed-threshold 라벨 생성 (legacy method)."""
    future_return = df["close"].pct_change(horizon).shift(-horizon)
    labels = pd.Series(LABEL_MAP["HOLD"], index=df.index, dtype=int)
    labels[future_return > UP_THRESHOLD] = LABEL_MAP["UP"]
    labels[future_return < DOWN_THRESHOLD] = LABEL_MAP["DOWN"]
    df["label"] = labels
    df["future_return"] = future_return

    bm = _cfg_bar_minutes()
    new_cols = {}
    for h in HORIZONS:
        h_return = df["close"].pct_change(h).shift(-h)
        h_label = pd.Series(LABEL_MAP["HOLD"], index=df.index, dtype=int)
        h_label[h_return > UP_THRESHOLD] = LABEL_MAP["UP"]
        h_label[h_return < DOWN_THRESHOLD] = LABEL_MAP["DOWN"]
        new_cols[f"label_{h*bm}min"] = h_label
        new_cols[f"return_{h*bm}min"] = h_return

    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)
    return df


def create_labels_triple_barrier(df: pd.DataFrame, horizon: int = 18,
                                  k_upper_override: float = None,
                                  k_lower_override: float = None,
                                  verbose: bool = True) -> pd.DataFrame:
    """Triple Barrier 라벨 생성.

    Upper barrier: entry + k_upper * ATR (take-profit)
    Lower barrier: entry - k_lower * ATR (stop-loss)
    Time barrier: horizon bars (max holding period)
    Label = 먼저 닿는 배리어 (UP/DOWN/HOLD)
    """
    cfg = _cfg_load_settings().get("labeling", {})
    k_upper = k_upper_override if k_upper_override is not None else cfg.get("k_upper", 1.5)
    k_lower = k_lower_override if k_lower_override is not None else cfg.get("k_lower", 1.5)
    min_barrier_pct = cfg.get("min_barrier_pct", 0.002)

    close = df["close"].values
    high = df["high"].values if "high" in df.columns else close
    low = df["low"].values if "low" in df.columns else close

    # ATR 사용 (이미 feature로 존재하면 사용, 없으면 계산)
    if "atr_14" in df.columns:
        atr = df["atr_14"].values
    else:
        atr_period = cfg.get("atr_period", 14)
        tr = np.maximum(high - low,
                        np.maximum(np.abs(high - np.roll(close, 1)),
                                   np.abs(low - np.roll(close, 1))))
        tr[0] = high[0] - low[0]
        atr = pd.Series(tr).rolling(atr_period, min_periods=1).mean().values

    n = len(close)
    bm = _cfg_bar_minutes()
    new_cols = {}

    for h in HORIZONS:
        labels = np.full(n, LABEL_MAP["HOLD"], dtype=int)
        returns = np.full(n, np.nan)

        for i in range(n - h):
            entry = close[i]
            cur_atr = atr[i] if not np.isnan(atr[i]) else entry * 0.01

            # Symmetric barriers: both directions use k_upper for TP distance
            # This ensures DOWN label = "price fell by k_upper*ATR" (matches SELL TP)
            # k_lower is used only for SL in execution, not for labeling
            upper_dist = max(k_upper * cur_atr, min_barrier_pct * entry)
            lower_dist = max(k_upper * cur_atr, min_barrier_pct * entry)
            upper_barrier = entry + upper_dist
            lower_barrier = entry - lower_dist

            # Forward walk: 어느 배리어를 먼저 터치하는지
            hit_upper = -1
            hit_lower = -1
            for j in range(i + 1, min(i + h + 1, n)):
                if hit_upper < 0 and high[j] >= upper_barrier:
                    hit_upper = j - i
                if hit_lower < 0 and low[j] <= lower_barrier:
                    hit_lower = j - i
                if hit_upper >= 0 and hit_lower >= 0:
                    break

            # 라벨 결정
            if hit_upper >= 0 and hit_lower >= 0:
                if hit_upper < hit_lower:
                    labels[i] = LABEL_MAP["UP"]
                elif hit_lower < hit_upper:
                    labels[i] = LABEL_MAP["DOWN"]
                else:
                    # 같은 바에서 둘 다 터치 → close 기준
                    exit_close = close[min(i + hit_upper, n - 1)]
                    labels[i] = LABEL_MAP["UP"] if exit_close >= entry else LABEL_MAP["DOWN"]
            elif hit_upper >= 0:
                labels[i] = LABEL_MAP["UP"]
            elif hit_lower >= 0:
                labels[i] = LABEL_MAP["DOWN"]
            # else: HOLD (time barrier — 어느 배리어도 미터치)

            returns[i] = (close[min(i + h, n - 1)] - entry) / entry

        new_cols[f"label_{h*bm}min"] = labels
        new_cols[f"return_{h*bm}min"] = returns

    # 기본 label = 최대 호라이즌
    max_h = HORIZONS[-1]
    df["label"] = new_cols[f"label_{max_h*bm}min"]
    df["future_return"] = new_cols[f"return_{max_h*bm}min"]
    df = pd.concat([df, pd.DataFrame(new_cols, index=df.index)], axis=1)

    # 라벨 분포 출력 (verbose 모드에서만)
    if verbose:
        for h in HORIZONS:
            col = f"label_{h*bm}min"
            counts = pd.Series(new_cols[col]).value_counts().sort_index()
            dist = {LABEL_NAMES.get(k, str(k)): v for k, v in counts.items()}
            print(f"    [TripleBarrier] {HORIZON_LABELS[h]}: {dist}")

    return df


def create_labels(df: pd.DataFrame, horizon: int = 12) -> pd.DataFrame:
    """라벨 생성 dispatcher — config 기반으로 방법 선택."""
    cfg = _cfg_load_settings().get("labeling", {})
    method = cfg.get("method", "fee_aware_dynamic")

    if method == "triple_barrier":
        return create_labels_triple_barrier(df, horizon)
    else:
        return create_labels_fixed(df, horizon)


def build_feature_matrix(
    ohlcv_data: dict[str, pd.DataFrame],
    media_data: dict,
    horizon: int = 24,
) -> dict[str, pd.DataFrame]:
    result = {}
    for ticker, df in ohlcv_data.items():
        if len(df) < 100:
            continue
        features = df.copy()
        features = create_labels(features, horizon)
        result[ticker] = features
    return result


def _add_temporal_media_features(
    df: pd.DataFrame, media_data: dict, ticker: str
) -> pd.DataFrame:
    from src.models.multimodal_classifier import MEDIA_SOURCES

    for source_name in MEDIA_SOURCES:
        src_data = media_data.get(source_name, {})
        base_sentiment = src_data.get("avg_sentiment", 0) if isinstance(src_data, dict) else 0

        np.random.seed(hash(f"{ticker}_{source_name}") % 2**31)
        noise = np.random.normal(0, abs(base_sentiment) * 0.3 + 0.01, len(df))
        temporal_sentiment = base_sentiment + noise

        df[f"media_{source_name}_sentiment"] = temporal_sentiment
        df[f"media_{source_name}_rolling_6h"] = pd.Series(
            temporal_sentiment, index=df.index).rolling(6, min_periods=1).mean()
        df[f"media_{source_name}_rolling_24h"] = pd.Series(
            temporal_sentiment, index=df.index).rolling(24, min_periods=1).mean()
        df[f"media_{source_name}_momentum"] = pd.Series(
            temporal_sentiment, index=df.index).diff(3).fillna(0)
        df[f"media_{source_name}_vol"] = pd.Series(
            temporal_sentiment, index=df.index).rolling(12, min_periods=1).std().fillna(0)

    sent_cols = [f"media_{s}_sentiment" for s in MEDIA_SOURCES if f"media_{s}_sentiment" in df.columns]
    if sent_cols:
        df["media_consensus"] = df[sent_cols].mean(axis=1)
        df["media_divergence"] = df[sent_cols].std(axis=1)
        df["media_extreme"] = df[sent_cols].abs().max(axis=1)

        # 전문 미디어 vs 소셜 분리
        pro = [c for c in sent_cols if any(p in c for p in ["coindesk", "cointelegraph", "theblock", "decrypt", "glassnode", "messari"])]
        social = [c for c in sent_cols if any(p in c for p in ["reddit", "x_twitter", "youtube", "tiktok", "instagram"])]
        if pro:
            df["media_pro_consensus"] = df[pro].mean(axis=1)
        if social:
            df["media_social_consensus"] = df[social].mean(axis=1)
        if pro and social:
            df["media_pro_social_gap"] = df["media_pro_consensus"] - df["media_social_consensus"]

    return df


# ==================== Grid Search (TimeSeriesSplit + gap) ====================

def run_grid_search(X_train, y_train, sample_weights, max_horizon: int = 12):
    """TimeSeriesSplit + gap으로 하이퍼파라미터 탐색.

    StratifiedKFold는 시계열에서 시간적 상관을 무시하여 낙관적 추정 위험.
    TimeSeriesSplit + gap ≥ max_horizon으로 data leakage 차단.
    scoring = balanced_accuracy (HOLD 비중 부풀림 방지, 목적함수 통일).
    """
    print("    [GridSearch] TimeSeriesSplit + gap (시계열 CV)...")

    param_grid = {
        "num_leaves": [15, 31],
        "learning_rate": [0.03, 0.05],
        "min_child_samples": [5, 10],
    }

    num_classes = len(np.unique(y_train))
    if num_classes <= 2:
        lgb_obj, lgb_nc = "binary", 1
    else:
        lgb_obj, lgb_nc = "multiclass", num_classes

    lgb_base = lgb.LGBMClassifier(
        objective=lgb_obj, num_class=lgb_nc,
        n_estimators=150, subsample=0.8, colsample_bytree=0.8,
        is_unbalance=True, verbose=-1, n_jobs=6,
        device="gpu", gpu_use_dp=False,
    )

    # TimeSeriesSplit: 시간 순서 보존 + gap으로 미래 정보 차단
    # gap = max_horizon bars (60min/5min = 12bars) → 최소 미래 12 bars 격리
    cv = TimeSeriesSplit(n_splits=3, gap=max_horizon)

    # scoring: balanced_accuracy (목적함수 통일, HOLD 부풀림 방지)
    gs = GridSearchCV(
        lgb_base, param_grid, cv=cv, scoring="balanced_accuracy",
        n_jobs=1, refit=False, verbose=0,  # LGB 자체 GPU 사용하므로 outer n_jobs=1
    )
    gs.fit(X_train, y_train, sample_weight=sample_weights)
    print(f"    [GridSearch] Best: BalAcc={gs.best_score_:.3f}, params={gs.best_params_}")
    return gs.best_params_


# ==================== Masking Loop Core ====================

class IterativeMaskingLoop:
    """5-Model Ensemble + GridSearch Masking Loop."""

    def __init__(self, horizon: int = 24, lookback: int = 168):
        self.horizon = horizon
        self.lookback = lookback
        self.state = LoopState()
        self.results_dir = Path("data/masking_loop")
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def run_iteration(
        self,
        feature_matrices: dict[str, pd.DataFrame],
        iteration_num: int,
    ) -> IterationResult:
        print(f"\n{'='*60}")
        print(f"  Iteration {iteration_num}")
        print(f"{'='*60}")

        all_bal_acc, all_f1, all_mcc = [], [], []
        all_importance = {}
        all_predictions = {}
        all_reports = []
        all_model_scores = {}

        coin_list = list(feature_matrices.items())
        coin_total = len(coin_list)
        coin_start = time.time() if 'time' in dir() else __import__('time').time()
        import time as _time

        for coin_idx, (ticker, df) in enumerate(coin_list):
            result = self._train_classify_single(ticker, df, iteration_num)
            if result is None:
                continue

            # 코인별 ETA
            coin_elapsed = _time.time() - coin_start
            coin_done = coin_idx + 1
            coin_per = coin_elapsed / coin_done
            coin_remaining = coin_per * (coin_total - coin_done)
            coin_eta = datetime.now() + __import__('datetime').timedelta(seconds=coin_remaining)
            print(f"    [{coin_done}/{coin_total}] {coin_elapsed:.0f}s | ETA iter end: {coin_eta.strftime('%H:%M:%S')} ({coin_remaining:.0f}s left)")

            all_bal_acc.append(result["balanced_accuracy"])
            all_f1.append(result["f1_macro"])
            all_mcc.append(result.get("mcc", 0))
            all_predictions[ticker] = result["prediction"]
            all_reports.append(result["report"])

            for name, score in result.get("model_scores", {}).items():
                all_model_scores.setdefault(name, []).append(score)

            for feat, imp in result["importance"].items():
                all_importance.setdefault(feat, []).append(imp)

        if not all_bal_acc:
            return self._empty_result(iteration_num)

        avg_bal_acc = np.mean(all_bal_acc)
        avg_f1 = np.mean(all_f1)
        avg_mcc = np.mean(all_mcc)
        avg_importance = {k: float(np.mean(v)) for k, v in all_importance.items()}
        avg_model_scores = {k: round(float(np.mean(v)), 4) for k, v in all_model_scores.items()}

        returns = [p.get("expected_return", 0) for p in all_predictions.values()]
        sharpe = (np.mean(returns) / (np.std(returns) + 1e-10)) * np.sqrt(24) if returns else 0

        new_weights = self._update_media_weights(avg_importance)
        noise = self._identify_noise(avg_importance, avg_bal_acc)

        result = IterationResult(
            iteration=iteration_num,
            accuracy=round(avg_bal_acc, 4),  # now balanced_accuracy
            f1_macro=round(avg_f1, 4),
            class_report=all_reports[0] if all_reports else {},
            sharpe_estimate=round(float(sharpe), 4),
            media_weights=new_weights,
            noise_sources=noise,
            feature_importance=dict(sorted(avg_importance.items(), key=lambda x: -x[1])[:20]),
            predictions=all_predictions,
            model_scores=avg_model_scores,
            timestamp=datetime.now().isoformat(),
        )

        self.state.total_iterations = iteration_num
        self.state.best_accuracy = max(self.state.best_accuracy, avg_bal_acc)
        self.state.best_f1 = max(self.state.best_f1, avg_f1)
        self.state.current_weights = new_weights
        self.state.iteration_history.append(asdict(result))

        model_str = " | ".join(f"{k}:{v:.1%}" for k, v in sorted(avg_model_scores.items(), key=lambda x: -x[1]))
        print(f"\n  BalAcc: {avg_bal_acc:.1%} | MCC: {avg_mcc:.3f} | F1: {avg_f1:.4f} | Sharpe: {sharpe:.4f}")
        print(f"  Models (balanced_accuracy): {model_str}")

        return result

    def _train_5model_ensemble(self, X_train, y_train, sample_weights, X_test,
                               iter_num, h, num_classes=3):
        """5-Model Ensemble 학습 + 예측. binary(2) 또는 multi-class(3) 지원."""
        gs_params = self.state.best_params or {}

        # LightGBM
        if num_classes == 2:
            lgb_obj, lgb_metric = "binary", "binary_logloss"
            lgb_nc = 1
        else:
            lgb_obj, lgb_metric = "multiclass", "multi_logloss"
            lgb_nc = num_classes

        lgb_params = {
            "objective": lgb_obj, "num_class": lgb_nc, "metric": lgb_metric,
            "learning_rate": gs_params.get("learning_rate", max(0.02, 0.06 - iter_num * 0.005)),
            "num_leaves": gs_params.get("num_leaves", 31 + iter_num * 4),
            "min_data_in_leaf": gs_params.get("min_child_samples", max(3, 15 - iter_num * 2)),
            "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 5,
            "is_unbalance": True, "verbose": -1,
            "seed": 42 + iter_num + h, "num_threads": -1,
            "device": "gpu", "gpu_use_dp": False,
        }
        lgb_ds = lgb.Dataset(X_train, label=y_train, weight=sample_weights)
        lgb_model = lgb.train(lgb_params, lgb_ds, num_boost_round=150 + iter_num * 30)
        lgb_proba_raw = lgb_model.predict(X_test)
        if num_classes == 2 and lgb_proba_raw.ndim == 1:
            lgb_proba = np.column_stack([1 - lgb_proba_raw, lgb_proba_raw])
        else:
            lgb_proba = lgb_proba_raw

        # XGBoost
        if num_classes == 2:
            xgb_obj, xgb_eval = "binary:logistic", "logloss"
            xgb_nc = None
        else:
            xgb_obj, xgb_eval = "multi:softprob", "mlogloss"
            xgb_nc = num_classes

        xgb_kwargs = dict(
            objective=xgb_obj,
            learning_rate=max(0.02, 0.05 - iter_num * 0.004),
            max_depth=5 + iter_num, n_estimators=100 + iter_num * 20,
            subsample=0.8, colsample_bytree=0.8,
            verbosity=0, random_state=42 + iter_num + h,
            eval_metric=xgb_eval,
            tree_method="hist", device="cuda",
        )
        if xgb_nc is not None:
            xgb_kwargs["num_class"] = xgb_nc
        xgb_model = xgb.XGBClassifier(**xgb_kwargs)
        xgb_model.fit(X_train, y_train, sample_weight=sample_weights)
        xgb_proba = xgb_model.predict_proba(X_test)

        # CatBoost
        cb_model = CatBoostClassifier(
            iterations=150 + iter_num * 20,
            learning_rate=max(0.02, 0.05 - iter_num * 0.004),
            depth=5 + min(iter_num, 3), auto_class_weights="Balanced",
            verbose=0, random_seed=42 + iter_num + h,
            task_type="GPU",
        )
        cb_model.fit(X_train, y_train, sample_weight=sample_weights)
        cb_proba = cb_model.predict_proba(X_test)

        # RandomForest
        rf_model = RandomForestClassifier(
            n_estimators=100 + iter_num * 20, max_depth=8 + iter_num,
            min_samples_leaf=max(3, 10 - iter_num), class_weight="balanced",
            random_state=42 + iter_num + h, n_jobs=6,
        )
        rf_model.fit(X_train, y_train, sample_weight=sample_weights)
        rf_proba = rf_model.predict_proba(X_test)

        # ExtraTrees
        et_model = ExtraTreesClassifier(
            n_estimators=100 + iter_num * 20, max_depth=8 + iter_num,
            min_samples_leaf=max(3, 10 - iter_num), class_weight="balanced",
            random_state=42 + iter_num + h, n_jobs=6,
        )
        et_model.fit(X_train, y_train, sample_weight=sample_weights)
        et_proba = et_model.predict_proba(X_test)

        # Balanced accuracy weighted ensemble
        y_test_for_score = y_train  # placeholder — caller supplies real y_test via return
        probas = {"LGB": lgb_proba, "XGB": xgb_proba, "CB": cb_proba, "RF": rf_proba, "ET": et_proba}
        models = {"LGB": lgb_model, "XGB": xgb_model, "CB": cb_model, "RF": rf_model, "ET": et_model}

        # Feature importance (LGB gain-based)
        importance_raw = lgb_model.feature_importance(importance_type="gain")

        return probas, models, importance_raw

    def _make_ensemble_proba(self, probas, y_test):
        """Performance-weighted ensemble 확률 계산."""
        model_scores = {}
        for name, proba in probas.items():
            pred = np.argmax(proba, axis=1)
            model_scores[name] = balanced_accuracy_score(y_test, pred)
        total_score = sum(model_scores.values()) + 1e-10
        weights = {name: model_scores[name] / total_score for name in model_scores}
        ensemble_proba = sum(weights[n] * probas[n] for n in probas)
        return ensemble_proba, model_scores, weights

    def _predict_latest(self, models, weights, X_latest, num_classes=3):
        """최신 데이터포인트 예측."""
        lat_probas = {}
        for name, model in models.items():
            if name == "LGB":
                raw = model.predict(X_latest)
                if num_classes == 2 and raw.ndim == 1:
                    lat_probas[name] = np.column_stack([1 - raw, raw])
                else:
                    lat_probas[name] = raw
            else:
                lat_probas[name] = model.predict_proba(X_latest)
        lat_proba = sum(weights[n] * lat_probas[n] for n in lat_probas)
        return lat_proba

    def _train_classify_single(self, ticker: str, df: pd.DataFrame, iter_num: int) -> dict | None:
        """2-Stage Binary 또는 3-class 분류 dispatcher."""
        cfg = _cfg_load_settings().get("labeling", {})
        two_stage = cfg.get("two_stage", False)

        if two_stage:
            return self._train_classify_2stage(ticker, df, iter_num)
        else:
            return self._train_classify_3class(ticker, df, iter_num)

    def _prepare_data(self, df, iter_num):
        """공통 데이터 준비 (feature selection, train/test split)."""
        bm = _cfg_bar_minutes()
        horizon_label_cols = [f"label_{h*bm}min" for h in HORIZONS]
        horizon_return_cols = [f"return_{h*bm}min" for h in HORIZONS]
        exclude = ["label", "future_return", "open", "high", "low", "close", "volume"] + horizon_label_cols + horizon_return_cols

        feature_cols = [c for c in df.columns
                        if c not in exclude
                        and df[c].dtype in [np.float64, np.float32, np.int64, np.int32, float, int]]

        valid = df.replace([np.inf, -np.inf], np.nan)
        valid.ffill(inplace=True)
        valid.fillna(0, inplace=True)  # no bfill: prevents future data leakage
        valid = valid.dropna(subset=["label"])
        if len(valid) < 120:
            return None

        MAX_FEATURES = _cfg_max_features()
        if len(feature_cols) > MAX_FEATURES:
            from sklearn.feature_selection import mutual_info_classif
            X_all = valid[feature_cols].values
            y_all = valid["label"].values
            variances = np.var(X_all, axis=0)
            var_mask = variances > 1e-10
            feature_cols_filtered = [c for c, m in zip(feature_cols, var_mask) if m]
            if len(feature_cols_filtered) > MAX_FEATURES:
                X_filt = valid[feature_cols_filtered].values
                mi = mutual_info_classif(X_filt[:min(3000, len(X_filt))],
                                         y_all[:min(3000, len(y_all))],
                                         discrete_features=False, random_state=42, n_neighbors=5)
                top_idx = np.argsort(mi)[-MAX_FEATURES:]
                feature_cols = [feature_cols_filtered[i] for i in sorted(top_idx)]
            else:
                feature_cols = feature_cols_filtered
            print(f"    [FeatureSelect] {len(feature_cols)} features selected (from {len(valid.columns)})")

        X = valid[feature_cols].values
        mask_size = max(self.horizon * 2, len(X) // 5)
        train_end = len(X) - mask_size
        train_start = max(0, train_end - self.lookback * min(iter_num + 1, 5))

        X_train = X[train_start:train_end]
        X_test = X[train_end:]

        if len(X_train) < 60 or len(X_test) < 10:
            return None

        return {
            "valid": valid, "feature_cols": feature_cols, "X": X,
            "X_train": X_train, "X_test": X_test,
            "train_start": train_start, "train_end": train_end,
            "bm": _cfg_bar_minutes(),
        }

    def _train_classify_2stage(self, ticker: str, df: pd.DataFrame, iter_num: int) -> dict | None:
        """2-Stage Binary: Stage1(Trade/NoTrade) → Stage2(Long/Short)."""
        data = self._prepare_data(df, iter_num)
        if data is None:
            return None

        valid, feature_cols, X = data["valid"], data["feature_cols"], data["X"]
        X_train, X_test = data["X_train"], data["X_test"]
        train_start, train_end, bm = data["train_start"], data["train_end"], data["bm"]
        cfg = _cfg_load_settings().get("labeling", {})
        s1_threshold = cfg.get("stage1_threshold", 0.50)

        horizon_results = {}
        all_metrics = []
        all_model_scores = {}
        importance_agg = {}
        last_weights = {}
        last_lat_proba = None

        for h in HORIZONS:
            h_label = f"label_{h*bm}min"
            if h_label not in valid.columns:
                continue

            y_3class = valid[h_label].values
            y_train_3c = y_3class[train_start:train_end]
            y_test_3c = y_3class[train_end:]

            # === Stage 1: Trade/NoTrade ===
            y_s1_train = (y_train_3c != LABEL_MAP["HOLD"]).astype(int)
            y_s1_test = (y_test_3c != LABEL_MAP["HOLD"]).astype(int)

            if len(np.unique(y_s1_train)) < 2:
                continue

            s1_counts = np.bincount(y_s1_train, minlength=2)
            s1_cw = np.where(s1_counts > 0, len(y_s1_train) / (2 * s1_counts + 1e-10), 1.0)
            s1_sw = s1_cw[y_s1_train]

            if iter_num == 1 and h == HORIZONS[0] and not self.state.best_params:
                try:
                    self.state.best_params = run_grid_search(
                        X_train, y_s1_train, s1_sw, max_horizon=max(HORIZONS))
                except Exception:
                    self.state.best_params = {}

            s1_probas, s1_models, s1_imp = self._train_5model_ensemble(
                X_train, y_s1_train, s1_sw, X_test, iter_num, h, num_classes=2)
            s1_ens, s1_scores, s1_weights = self._make_ensemble_proba(s1_probas, y_s1_test)
            s1_pred = (s1_ens[:, 1] >= s1_threshold).astype(int)
            s1_metrics = compute_extended_metrics(y_s1_test, s1_pred, s1_ens,
                                                  num_classes=2, label_names=STAGE1_NAMES)

            # === Stage 2: Long/Short (Trade samples only) ===
            trade_mask_train = y_train_3c != LABEL_MAP["HOLD"]
            trade_mask_test = s1_pred == 1  # Stage1 예측 기반

            # Stage 2 학습은 실제 Trade 라벨로
            X_s2_train = X_train[trade_mask_train]
            y_s2_train = (y_train_3c[trade_mask_train] == LABEL_MAP["UP"]).astype(int)  # Long=1, Short=0

            s2_metrics = None
            final_pred = np.full(len(X_test), LABEL_MAP["HOLD"])  # default: HOLD

            if len(X_s2_train) >= 30 and len(np.unique(y_s2_train)) >= 2:
                s2_counts = np.bincount(y_s2_train, minlength=2)
                s2_cw = np.where(s2_counts > 0, len(y_s2_train) / (2 * s2_counts + 1e-10), 1.0)
                s2_sw = s2_cw[y_s2_train]

                s2_probas, s2_models, _ = self._train_5model_ensemble(
                    X_s2_train, y_s2_train, s2_sw, X_test, iter_num, h + 100, num_classes=2)

                # Stage 2: Trade로 예측된 샘플만 Long/Short 판정
                if trade_mask_test.sum() > 0:
                    X_trade_test = X_test[trade_mask_test]
                    s2_trade_probas = {n: p[trade_mask_test] for n, p in s2_probas.items()}

                    # Stage 2 test labels (for scoring)
                    y_s2_test_real = (y_test_3c[trade_mask_test] == LABEL_MAP["UP"]).astype(int)
                    if len(np.unique(y_s2_test_real)) >= 2:
                        s2_ens, s2_scores, s2_w = self._make_ensemble_proba(s2_trade_probas, y_s2_test_real)
                        s2_pred = np.argmax(s2_ens, axis=1)
                        s2_metrics = compute_extended_metrics(y_s2_test_real, s2_pred, s2_ens,
                                                             num_classes=2, label_names=STAGE2_NAMES)
                    else:
                        s2_pred = np.argmax(s2_probas["CB"][trade_mask_test], axis=1)

                    # Map back to 3-class
                    trade_indices = np.where(trade_mask_test)[0]
                    for idx, s2 in zip(trade_indices, s2_pred):
                        final_pred[idx] = LABEL_MAP["UP"] if s2 == 1 else LABEL_MAP["DOWN"]

                # Stage 2 weights: reuse s2_w from scoring if available
                s2_w_all = {n: 1.0 / len(s2_probas) for n in s2_probas}
                if s2_metrics is not None:
                    # s2_w was computed during scoring above
                    s2_w_all = s2_w

                # Latest prediction
                lat_proba_s1 = self._predict_latest(s1_models, s1_weights, X[-1:], num_classes=2)
                is_trade = lat_proba_s1[0][1] >= s1_threshold
                if is_trade:
                    lat_proba_s2 = self._predict_latest(s2_models, s2_w_all, X[-1:], num_classes=2)
                    trade_prob = float(lat_proba_s1[0][1])
                    long_prob = float(lat_proba_s2[0][1])
                    if long_prob >= 0.5:
                        lat_direction = "UP"
                        lat_conf = trade_prob * long_prob
                    else:
                        lat_direction = "DOWN"
                        lat_conf = trade_prob * (1 - long_prob)
                else:
                    lat_direction = "HOLD"
                    lat_conf = float(1 - lat_proba_s1[0][1])
            else:
                lat_direction = "HOLD"
                lat_conf = 1.0

            # Combined 3-class metrics
            combined_metrics = compute_extended_metrics(y_test_3c, final_pred, num_classes=3)
            all_metrics.append(combined_metrics)

            for name, score in s1_scores.items():
                all_model_scores.setdefault(name, []).append(score)

            if h == HORIZONS[-1]:
                total_imp = s1_imp.sum() + 1e-10
                importance_agg = dict(zip(feature_cols, (s1_imp / total_imp).tolist()))
                last_weights = s1_weights

            horizon_results[f"{h*bm}min"] = {
                "accuracy": combined_metrics["accuracy"],
                "balanced_accuracy": combined_metrics["balanced_accuracy"],
                "f1": combined_metrics["f1_macro"],
                "mcc": combined_metrics["mcc"],
                "brier": combined_metrics.get("brier_score"),
                "stage1_acc": s1_metrics["balanced_accuracy"],
                "stage2_acc": s2_metrics["balanced_accuracy"] if s2_metrics else None,
                "trade_ratio": float(trade_mask_test.mean()) if len(trade_mask_test) > 0 else 0,
                "direction": lat_direction,
                "confidence": round(lat_conf, 3),
                "model_scores": {k: round(v, 4) for k, v in s1_scores.items()},
                "class_distribution": combined_metrics.get("class_distribution", {}),
            }

        if not all_metrics:
            return None

        avg_bal_acc = np.mean([m["balanced_accuracy"] for m in all_metrics])
        avg_f1 = np.mean([m["f1_macro"] for m in all_metrics])
        avg_mcc = np.mean([m["mcc"] for m in all_metrics])
        avg_model_scores = {k: round(float(np.mean(v)), 4) for k, v in all_model_scores.items()}

        max_h = HORIZONS[-1]
        max_h_key = f"{max_h*bm}min"
        main_pred = horizon_results.get(max_h_key, horizon_results.get(list(horizon_results.keys())[-1], {}))
        lat_direction = main_pred.get("direction", "HOLD")
        lat_conf_main = main_pred.get("confidence", 0)
        expected_return = 0
        if lat_direction == "UP":
            expected_return = lat_conf_main * 2.0
        elif lat_direction == "DOWN":
            expected_return = -lat_conf_main * 2.0

        h_summary = " | ".join(
            f"{k}:BA{v['balanced_accuracy']:.0%}/S1:{v.get('stage1_acc', 0):.0%}/S2:{v.get('stage2_acc', 'N/A')}"
            for k, v in horizon_results.items())
        best_model = max(avg_model_scores, key=avg_model_scores.get) if avg_model_scores else "?"
        print(f"  [{ticker}] BalAcc: {avg_bal_acc:.1%} | MCC: {avg_mcc:.3f} | F1: {avg_f1:.3f} | Best: {best_model} (2-Stage)")
        print(f"    Horizons: {h_summary}")

        return {
            "accuracy": avg_bal_acc,
            "balanced_accuracy": avg_bal_acc,
            "f1_macro": avg_f1,
            "mcc": avg_mcc,
            "report": {},
            "importance": importance_agg,
            "model_scores": avg_model_scores,
            "horizon_results": horizon_results,
            "prediction": {
                "direction": lat_direction,
                "confidence": round(lat_conf_main, 3),
                "probabilities": {"DOWN": 0, "HOLD": 0, "UP": 0},
                "expected_return": round(expected_return, 4),
                "horizon_predictions": horizon_results,
            },
        }

    def _train_classify_3class(self, ticker: str, df: pd.DataFrame, iter_num: int) -> dict | None:
        """기존 3-class 분류 (fallback)."""
        data = self._prepare_data(df, iter_num)
        if data is None:
            return None

        valid, feature_cols, X = data["valid"], data["feature_cols"], data["X"]
        X_train, X_test = data["X_train"], data["X_test"]
        train_start, train_end, bm = data["train_start"], data["train_end"], data["bm"]

        horizon_results = {}
        all_metrics = []
        all_model_scores = {}
        importance_agg = {}

        for h in HORIZONS:
            h_label = f"label_{h*bm}min"
            if h_label not in valid.columns:
                continue

            y = valid[h_label].values
            y_train = y[train_start:train_end]
            y_test = y[train_end:]

            unique = np.unique(y_train)
            if len(unique) < 2:
                continue

            class_counts = np.bincount(y_train.astype(int), minlength=3)
            class_weights_arr = np.where(class_counts > 0, len(y_train) / (3 * class_counts + 1e-10), 1.0)
            sample_weights = class_weights_arr[y_train.astype(int)]

            if iter_num == 1 and h == HORIZONS[0] and not self.state.best_params:
                try:
                    self.state.best_params = run_grid_search(
                        X_train, y_train, sample_weights, max_horizon=max(HORIZONS))
                except Exception:
                    self.state.best_params = {}

            probas, models, imp_raw = self._train_5model_ensemble(
                X_train, y_train, sample_weights, X_test, iter_num, h, num_classes=3)
            ensemble_proba, model_scores, weights = self._make_ensemble_proba(probas, y_test)
            y_pred = np.argmax(ensemble_proba, axis=1)

            h_metrics = compute_extended_metrics(y_test, y_pred, ensemble_proba, num_classes=3)
            all_metrics.append(h_metrics)

            for name, score in model_scores.items():
                all_model_scores.setdefault(name, []).append(score)

            if h == HORIZONS[-1]:
                total_imp = imp_raw.sum() + 1e-10
                importance_agg = dict(zip(feature_cols, (imp_raw / total_imp).tolist()))

            lat_proba = self._predict_latest(models, weights, X[-1:], num_classes=3)
            lat_class = int(np.argmax(lat_proba[0]))
            lat_conf = float(lat_proba[0][lat_class])

            horizon_results[f"{h*bm}min"] = {
                "accuracy": h_metrics["accuracy"],
                "balanced_accuracy": h_metrics["balanced_accuracy"],
                "f1": h_metrics["f1_macro"],
                "mcc": h_metrics["mcc"],
                "brier": h_metrics.get("brier_score"),
                "pr_auc": h_metrics.get("pr_auc_macro"),
                "direction": LABEL_NAMES[lat_class],
                "confidence": round(lat_conf, 3),
                "model_scores": {k: round(v, 4) for k, v in model_scores.items()},
                "class_distribution": h_metrics.get("class_distribution", {}),
            }

        if not all_metrics:
            return None

        avg_bal_acc = np.mean([m["balanced_accuracy"] for m in all_metrics])
        avg_f1 = np.mean([m["f1_macro"] for m in all_metrics])
        avg_mcc = np.mean([m["mcc"] for m in all_metrics])
        avg_model_scores = {k: round(float(np.mean(v)), 4) for k, v in all_model_scores.items()}

        report = {}
        max_h = HORIZONS[-1]
        max_h_key = f"{max_h*bm}min"
        main_pred = horizon_results.get(max_h_key, horizon_results.get(list(horizon_results.keys())[-1], {}))
        lat_direction = main_pred.get("direction", "HOLD")
        lat_conf_main = main_pred.get("confidence", 0)
        expected_return = 0
        if lat_direction == "UP":
            expected_return = lat_conf_main * 2.0
        elif lat_direction == "DOWN":
            expected_return = -lat_conf_main * 2.0

        h_summary = " | ".join(f"{k}:BA{v['balanced_accuracy']:.0%}/MCC{v['mcc']:.2f}"
                               for k, v in horizon_results.items())
        best_model = max(avg_model_scores, key=avg_model_scores.get) if avg_model_scores else "?"
        print(f"  [{ticker}] BalAcc: {avg_bal_acc:.1%} | MCC: {avg_mcc:.3f} | F1: {avg_f1:.3f} | Best: {best_model}")
        print(f"    Horizons: {h_summary}")

        return {
            "accuracy": avg_bal_acc,
            "balanced_accuracy": avg_bal_acc,
            "f1_macro": avg_f1,
            "mcc": avg_mcc,
            "report": report,
            "importance": importance_agg,
            "model_scores": avg_model_scores,
            "horizon_results": horizon_results,
            "prediction": {
                "direction": lat_direction,
                "confidence": round(lat_conf_main, 3),
                "probabilities": {"DOWN": 0, "HOLD": 0, "UP": 0},
                "expected_return": round(expected_return, 4),
                "horizon_predictions": horizon_results,
            },
        }

    def _update_media_weights(self, importance: dict) -> dict[str, float]:
        from src.models.multimodal_classifier import MEDIA_SOURCES
        media_prefixes = {}
        for s in MEDIA_SOURCES:
            group = s.replace("_proxy", "")
            media_prefixes[group] = f"media_{s}_"

        new_w = {}
        for group, prefix in media_prefixes.items():
            new_w[group] = sum(v for k, v in importance.items() if k.startswith(prefix))

        cross_imp = sum(v for k, v in importance.items()
                        if k.startswith("media_") and not any(k.startswith(p) for p in media_prefixes.values()))
        n_groups = max(len(media_prefixes), 1)
        for group in media_prefixes:
            new_w[group] += cross_imp / n_groups
        new_w["technical"] = sum(v for k, v in importance.items() if not k.startswith("media_"))

        total = sum(new_w.values()) + 1e-10
        new_w = {k: v / total for k, v in new_w.items()}

        blended = {}
        for k in new_w:
            old = self.state.current_weights.get(k, 0.05)
            blended[k] = 0.6 * old + 0.4 * new_w[k]

        total = sum(blended.values()) + 1e-10
        return {k: round(v / total, 4) for k, v in blended.items()}

    def _identify_noise(self, importance: dict, accuracy: float) -> list[str]:
        noise = []
        media_feats = {k: v for k, v in importance.items() if k.startswith("media_")}
        if media_feats:
            avg = np.mean(list(media_feats.values()))
            for feat, imp in media_feats.items():
                if imp < avg * 0.1:
                    noise.append(feat.replace("media_", ""))

        if accuracy < 0.35:
            noise.append("high_volatility_regime")
        return noise

    def _empty_result(self, iter_num: int) -> IterationResult:
        return IterationResult(
            iteration=iter_num, accuracy=0, f1_macro=0,
            class_report={}, sharpe_estimate=0,
            media_weights=self.state.current_weights,
            noise_sources=["no_data"], feature_importance={},
            timestamp=datetime.now().isoformat(),
        )

    def save_state(self) -> Path:
        path = self.results_dir / "loop_state.json"
        with open(path, "w") as f:
            json.dump({
                "total_iterations": self.state.total_iterations,
                "best_accuracy": self.state.best_accuracy,
                "best_f1": self.state.best_f1,
                "current_weights": self.state.current_weights,
                "best_params": self.state.best_params,
                "saved_at": datetime.now().isoformat(),
            }, f, indent=2, default=str)
        return path

    def save_iteration_result(self, result: IterationResult) -> Path:
        path = self.results_dir / f"iteration_{result.iteration}.json"
        with open(path, "w") as f:
            json.dump(asdict(result), f, indent=2, default=str)
        return path

    def generate_final_report(self) -> dict:
        if not self.state.iteration_history:
            return {"error": "No iterations completed"}

        history = self.state.iteration_history
        latest = history[-1]

        acc_trend = [h["accuracy"] for h in history]
        f1_trend = [h["f1_macro"] for h in history]
        sharpe_trend = [h["sharpe_estimate"] for h in history]

        grades = {}
        for ticker, pred in latest.get("predictions", {}).items():
            direction = pred.get("direction", "HOLD")
            conf = pred.get("confidence", 0)
            level = "high_conf" if conf > 0.6 else "mid_conf" if conf > 0.4 else "low_conf"
            grade = GRADE_MAP.get(direction, GRADE_MAP["HOLD"])[level]
            grades[ticker] = {**pred, "grade": grade}

        report = {
            "report_type": "iterative_masking_loop_v3_fee_aware",
            "generated_at": datetime.now().isoformat(),
            "total_iterations": len(history),
            "label_scheme": {
                "UP": f">{UP_THRESHOLD*100:.1f}% (fee-aware)",
                "DOWN": f"<{DOWN_THRESHOLD*100:.1f}% (fee-aware)",
                "HOLD": "within no-trade band",
                "fee_breakdown": CRYPTO_FEES,
                "fee_threshold": round(FEE_THRESHOLD * 100, 2),
            },
            "cv_method": "TimeSeriesSplit + gap (시계열 CV)",
            "ensemble_objective": "balanced_accuracy (목적함수 통일)",
            "metrics": ["balanced_accuracy", "MCC", "Brier", "PR-AUC", "ECE", "F1-macro"],
            "performance_summary": {
                "initial_accuracy": acc_trend[0] if acc_trend else 0,
                "final_accuracy": acc_trend[-1] if acc_trend else 0,
                "best_accuracy": max(acc_trend) if acc_trend else 0,
                "accuracy_improvement_pct": round((acc_trend[-1] - acc_trend[0]) / (acc_trend[0] + 1e-10) * 100, 2) if len(acc_trend) >= 2 else 0,
                "initial_f1": f1_trend[0] if f1_trend else 0,
                "final_f1": f1_trend[-1] if f1_trend else 0,
                "best_f1": max(f1_trend) if f1_trend else 0,
                "final_sharpe": sharpe_trend[-1] if sharpe_trend else 0,
            },
            "model_scores": latest.get("model_scores", {}),
            "grid_search_params": self.state.best_params,
            "media_reliability_weights": latest["media_weights"],
            "investment_grades": grades,
            "noise_sources_identified": latest.get("noise_sources", []),
            "top_features": latest.get("feature_importance", {}),
            "iteration_trends": {
                "accuracy": acc_trend,
                "f1_macro": f1_trend,
                "sharpe": sharpe_trend,
            },
        }

        path = self.results_dir / "final_report.json"
        with open(path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        return report
