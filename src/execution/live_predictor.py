"""2-Stage Binary Prediction with Multi-Model Combos.

Aligns live prediction with backtesting pipeline:
  Stage 1: Trade / NoTrade (binary)
  Stage 2: Long / Short   (binary, Trade samples only)

Supports model combos from Mega Search v2:
  et, cb, xgb, tabm, tabpfn and weighted blends.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import balanced_accuracy_score

from src.models.masking_loop import create_labels_triple_barrier, LABEL_MAP, HORIZONS
from src.utils.feature_policy import is_excluded_feature

logger = logging.getLogger("live_bot.predictor")

# ── Model builders ──────────────────────────────────────────

def _build_et(n_est: int = 300, max_depth: int = 8, min_child: int = 10):
    return ExtraTreesClassifier(
        n_estimators=n_est, max_depth=max_depth,
        min_samples_leaf=min_child, class_weight="balanced",
        n_jobs=6, random_state=42,
    )


def _build_cb(n_est: int = 300, max_depth: int = 6):
    from catboost import CatBoostClassifier
    return CatBoostClassifier(
        iterations=n_est, depth=max_depth, learning_rate=0.05,
        auto_class_weights="Balanced", task_type="CPU",
        thread_count=6, verbose=0, random_seed=42,
    )


def _build_xgb(n_est: int = 200, max_depth: int = 6):
    import xgboost as xgb
    return xgb.XGBClassifier(
        n_estimators=n_est, max_depth=max_depth, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, tree_method="hist",
        device="cuda", verbosity=0, random_state=42, eval_metric="logloss",
    )


class TabMWrap:
    """TabM sklearn-compatible wrapper. Defined at module level for pickling."""

    def __init__(self, nf, k=32, d=128, nb=3, ep=150):
        self.nf = nf; self.k = k; self.d = d
        self.nb = nb; self.ep = ep
        self._scaler = None
        self._model = None
        self._device = None

    def _ensure_imports(self):
        import torch
        if self._device is None:
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
        if self._scaler is None:
            from sklearn.preprocessing import StandardScaler
            self._scaler = StandardScaler()

    def fit(self, X, y, sample_weight=None):
        import torch
        import torch.nn as nn
        from tabm import TabM

        self._ensure_imports()
        Xs = self._scaler.fit_transform(X).astype(np.float32)
        self._model = TabM.make(
            n_num_features=self.nf, cat_cardinalities=[], d_out=2,
            arch_type="tabm", k=self.k, d_block=self.d,
            n_blocks=self.nb, dropout=0.1,
        ).to(self._device)
        opt = torch.optim.AdamW(self._model.parameters(), lr=1e-3, weight_decay=1e-4)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.ep)
        Xt = torch.tensor(Xs, device=self._device)
        yt = torch.tensor(y, dtype=torch.long, device=self._device)
        wt = (torch.tensor(sample_weight, dtype=torch.float32, device=self._device)
              if sample_weight is not None else None)
        self._model.train()
        best_loss = float("inf"); best_state = None; no_improve = 0
        for ep in range(self.ep):
            perm = torch.randperm(len(Xt), device=self._device)
            epoch_loss = 0; n_batch = 0
            for i in range(0, len(Xt), 128):
                ix = perm[i:i+128]
                out = self._model(x_num=Xt[ix], x_cat=None).mean(dim=1)
                if wt is not None:
                    loss = (nn.functional.cross_entropy(out, yt[ix], reduction="none") * wt[ix]).mean()
                else:
                    loss = nn.functional.cross_entropy(out, yt[ix])
                opt.zero_grad(); loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
                opt.step(); epoch_loss += loss.item(); n_batch += 1
            sch.step()
            avg_loss = epoch_loss / max(n_batch, 1)
            if avg_loss < best_loss:
                best_loss = avg_loss; no_improve = 0
                best_state = {k: v.clone() for k, v in self._model.state_dict().items()}
            else:
                no_improve += 1
            if no_improve >= 15:
                break
        if best_state:
            self._model.load_state_dict(best_state)
        return self

    def predict_proba(self, X):
        import torch
        self._ensure_imports()
        Xs = self._scaler.transform(X).astype(np.float32)
        self._model.eval()
        with torch.no_grad():
            out = self._model(
                x_num=torch.tensor(Xs, device=self._device), x_cat=None
            ).mean(dim=1)
            return torch.softmax(out, dim=1).cpu().numpy()

    def get_params(self, deep=True):
        return {"nf": self.nf}

    def set_params(self, **params):
        return self


def _build_tabm(n_features: int):
    """Build TabM wrapper (checks availability)."""
    try:
        from tabm import TabM  # noqa: F401
    except ImportError:
        logger.warning("TabM not available, falling back to ET")
        return None
    return TabMWrap(n_features)


# ── Builder dispatcher ──────────────────────────────────────

def build_model(name: str, n_features: int = 80, coin_cfg: dict = None):
    """Build a model by short name."""
    cfg = coin_cfg or {}
    n_est = cfg.get("n_estimators", 300)
    max_d = cfg.get("max_depth_tree", 8)
    min_c = cfg.get("min_child_samples", 10)

    if name == "et":
        return _build_et(n_est, max_d, min_c)
    elif name == "cb":
        return _build_cb(n_est, min(max_d, 6))
    elif name == "xgb":
        return _build_xgb(min(n_est, 200), min(max_d, 6))
    elif name == "tabm":
        m = _build_tabm(n_features)
        if m is None:
            return _build_et(n_est, max_d, min_c)
        return m
    else:
        logger.warning(f"Unknown model '{name}', falling back to ET")
        return _build_et(n_est, max_d, min_c)


# ── Prediction result ──────────────────────────────────────

@dataclass
class PredictionResult:
    """Output of 2-stage prediction."""
    side: str           # "BUY", "SELL", "HOLD"
    p_trade: float      # S1 probability of Trade
    p_direction: float  # S2 probability of winning direction
    confidence: float   # p_trade * p_direction
    regime: str = ""


# ── Trained Model Artifact ──────────────────────────────────

@dataclass
class TrainedCombo:
    """Holds 2-stage multi-model combo for one coin."""
    coin: str
    s1_models: list       # list of fitted model objects (Stage 1)
    s2_models: list       # list of fitted model objects (Stage 2)
    model_names: list     # e.g. ["et", "cb"]
    weights: list         # e.g. [0.5, 0.5]
    feature_columns: list
    train_date: str
    train_samples: int
    cv_score: float


# ── Feature Selection ──────────────────────────────────────

_EXCLUDE_COLS = frozenset({"open", "high", "low", "close", "volume", "label"})
_LEAKY_KEYS = ("future", "target", "label_", "return_", "fwd_", "forward_")


def select_features(df: pd.DataFrame) -> list[str]:
    """Select numeric feature columns, excluding OHLCV/labels/leaky cols."""
    cols = []
    for c in df.columns:
        cl = c.lower()
        if cl in _EXCLUDE_COLS:
            continue
        if any(k in cl for k in _LEAKY_KEYS):
            continue
        if is_excluded_feature(c):
            continue
        if df[c].dtype in (np.float64, np.float32, np.int64, np.int32, float, int):
            cols.append(c)
    return cols


def mi_select(X: np.ndarray, y: np.ndarray, cols: list, max_feat: int) -> list:
    """Mutual information based feature selection."""
    if len(cols) <= max_feat:
        return cols
    try:
        mi = mutual_info_classif(
            X[:min(2000, len(X))], y[:min(2000, len(X))],
            discrete_features=False, random_state=42, n_neighbors=5,
        )
        top_idx = np.argsort(mi)[-max_feat:]
        return [cols[i] for i in sorted(top_idx)]
    except Exception:
        return cols[:max_feat]


# ── Core: Train 2-Stage Multi-Model ────────────────────────

def train_combo(
    df: pd.DataFrame,
    coin: str,
    common_cfg: dict,
    coin_cfg: dict,
) -> Optional[TrainedCombo]:
    """Train 2-stage binary models with the coin's optimal combo.

    Stage 1: Trade(1) / NoTrade(0) — HOLD vs (UP|DOWN)
    Stage 2: Long(1) / Short(0)   — UP vs DOWN (Trade samples only)
    """
    t0 = time.time()

    # ── Labeling (same as backtesting pipeline) ────────────
    labeled = create_labels_triple_barrier(
        df,
        horizon=HORIZONS[-1],
        k_upper_override=common_cfg["k_upper"],
        k_lower_override=common_cfg.get("k_lower", 0.6),
        verbose=False,
    )

    bm = common_cfg.get("bar_minutes", 240)
    label_col = f"label_{HORIZONS[-1] * bm}min"
    if label_col not in labeled.columns:
        label_col = "label"
    if label_col not in labeled.columns:
        logger.error(f"[Train] {coin}: no label column found")
        return None

    # ── Feature selection ──────────────────────────────────
    all_cols = select_features(labeled)
    max_feat = coin_cfg.get("max_features", 120)
    clean = labeled.replace([np.inf, -np.inf], np.nan).ffill().bfill()

    y_3c = clean[label_col].fillna(LABEL_MAP["HOLD"]).values.astype(int)
    X_all = clean[all_cols].fillna(0).values
    feature_cols = mi_select(X_all, y_3c, all_cols, max_feat)
    X = clean[feature_cols].fillna(0).values

    if len(X) < 200:
        logger.warning(f"[Train] {coin}: too few samples ({len(X)})")
        return None

    # ── 2-Stage labels ─────────────────────────────────────
    # Stage 1: Trade(1) = UP or DOWN, NoTrade(0) = HOLD
    y_s1 = (y_3c != LABEL_MAP["HOLD"]).astype(int)
    s1_counts = np.bincount(y_s1, minlength=2)
    s1_weights = np.where(s1_counts > 0, len(y_s1) / (2 * s1_counts + 1e-10), 1.0)[y_s1]

    # Stage 2: Long(1) = UP, Short(0) = DOWN (Trade samples only)
    trade_mask = y_3c != LABEL_MAP["HOLD"]
    y_s2 = (y_3c[trade_mask] == LABEL_MAP["UP"]).astype(int)
    X_s2 = X[trade_mask]
    s2_counts = np.bincount(y_s2, minlength=2)
    s2_weights = np.where(s2_counts > 0, len(y_s2) / (2 * s2_counts + 1e-10), 1.0)[y_s2]

    # ── Parse combo config ─────────────────────────────────
    combo = coin_cfg.get("model_combo", "et")
    weights_dict = coin_cfg.get("model_weights", {"et": 1.0})
    model_names = list(weights_dict.keys())
    model_wts = [weights_dict[n] for n in model_names]

    # ── Cross-validation (Stage 1 only for speed) ──────────
    tscv = TimeSeriesSplit(n_splits=3, gap=12)
    cv_scores = []
    for train_idx, val_idx in tscv.split(X):
        et_cv = _build_et(
            coin_cfg.get("n_estimators", 300),
            coin_cfg.get("max_depth_tree", 8),
            coin_cfg.get("min_child_samples", 10),
        )
        et_cv.fit(X[train_idx], y_s1[train_idx], sample_weight=s1_weights[train_idx])
        preds = et_cv.predict(X[val_idx])
        cv_scores.append(balanced_accuracy_score(y_s1[val_idx], preds))
    avg_cv = np.mean(cv_scores)

    # ── Train all models ───────────────────────────────────
    s1_models = []
    s2_models = []

    for name in model_names:
        # Stage 1
        m1 = build_model(name, n_features=len(feature_cols), coin_cfg=coin_cfg)
        try:
            m1.fit(X, y_s1, sample_weight=s1_weights)
        except TypeError:
            m1.fit(X, y_s1)
        s1_models.append(m1)

        # Stage 2
        m2 = build_model(name, n_features=len(feature_cols), coin_cfg=coin_cfg)
        try:
            m2.fit(X_s2, y_s2, sample_weight=s2_weights)
        except TypeError:
            m2.fit(X_s2, y_s2)
        s2_models.append(m2)

    elapsed = time.time() - t0
    logger.info(
        f"[Train] {coin}: combo={combo} | CV={avg_cv:.4f} | "
        f"models={model_names} | {len(X)} samples | {elapsed:.1f}s"
    )

    return TrainedCombo(
        coin=coin,
        s1_models=s1_models,
        s2_models=s2_models,
        model_names=model_names,
        weights=model_wts,
        feature_columns=feature_cols,
        train_date=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"),
        train_samples=len(X),
        cv_score=round(avg_cv, 4),
    )


# ── Core: 2-Stage Prediction ──────────────────────────────

def _weighted_ensemble_proba(
    models: list, weights: list, X: np.ndarray, stage_name: str,
) -> Optional[np.ndarray]:
    """Run weighted ensemble prediction. Returns averaged proba or None."""
    proba_list = []
    for model in models:
        try:
            proba_list.append(model.predict_proba(X)[0])
        except Exception as e:
            logger.warning(f"[Predict] {stage_name} model error: {e}")
    if not proba_list:
        return None
    wts = weights[:len(proba_list)]
    w_sum = sum(wts)
    return sum(w / w_sum * p for w, p in zip(wts, proba_list))


def predict_2stage(
    combo: TrainedCombo,
    df: pd.DataFrame,
    s1_threshold: float = 0.50,
) -> PredictionResult:
    """Run 2-stage prediction on the latest bar.

    Stage 1: p(Trade) >= threshold -> proceed to Stage 2
    Stage 2: p(Long) > 0.5 -> BUY, else SELL
    """
    fcols = combo.feature_columns
    missing = [c for c in fcols if c not in df.columns]
    if missing:
        df = df.copy()
        for c in missing:
            df[c] = 0.0

    X = np.nan_to_num(df[fcols].iloc[[-1]].values, nan=0.0, posinf=0.0, neginf=0.0)

    # Stage 1
    s1 = _weighted_ensemble_proba(combo.s1_models, combo.weights, X, "S1")
    if s1 is None:
        return PredictionResult("HOLD", 0.0, 0.5, 0.0)
    p_trade = float(s1[1]) if len(s1) > 1 else float(s1[0])
    if p_trade < s1_threshold:
        return PredictionResult("HOLD", p_trade, 0.5, 0.0)

    # Stage 2
    s2 = _weighted_ensemble_proba(combo.s2_models, combo.weights, X, "S2")
    if s2 is None:
        return PredictionResult("HOLD", p_trade, 0.5, 0.0)
    p_long = float(s2[1]) if len(s2) > 1 else float(s2[0])

    if p_long > 0.5:
        return PredictionResult("BUY", p_trade, p_long, p_trade * p_long)
    return PredictionResult("SELL", p_trade, 1 - p_long, p_trade * (1 - p_long))


# ── Artifact Save/Load ─────────────────────────────────────

ARTIFACT_DIR = Path("data/models/live_v4_2")


def save_combo(combo: TrainedCombo, artifact_dir: Path = ARTIFACT_DIR) -> Path:
    """Save trained combo to disk."""
    artifact_dir.mkdir(parents=True, exist_ok=True)
    path = artifact_dir / f"{combo.coin}_combo.joblib"
    joblib.dump({
        "coin": combo.coin,
        "s1_models": combo.s1_models,
        "s2_models": combo.s2_models,
        "model_names": combo.model_names,
        "weights": combo.weights,
        "feature_columns": combo.feature_columns,
        "train_date": combo.train_date,
        "train_samples": combo.train_samples,
        "cv_score": combo.cv_score,
    }, path, compress=3)
    logger.info(f"[Save] {combo.coin} → {path}")
    return path


def load_combo(coin: str, artifact_dir: Path = ARTIFACT_DIR) -> Optional[TrainedCombo]:
    """Load trained combo from disk."""
    path = artifact_dir / f"{coin}_combo.joblib"
    if not path.exists():
        return None
    try:
        d = joblib.load(path)
        return TrainedCombo(**d)
    except Exception as e:
        logger.error(f"[Load] {coin}: {e}")
        return None
