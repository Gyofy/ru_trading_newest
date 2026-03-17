"""Enhanced ML Ensemble — 7모델 + Stacking Meta-Learner.

기존 5모델에 BalancedRandomForest, HistGradientBoosting 추가.
단순 가중 평균 → StackingClassifier 메타러너로 교체.

Models:
  1. LightGBM (GPU)
  2. XGBoost (GPU)
  3. CatBoost (GPU)
  4. BalancedRandomForest (CPU) — RandomForest 대체
  5. ExtraTrees (CPU)
  6. HistGradientBoostingClassifier (CPU) — 신규
  7. Stacking Meta-Learner (LogisticRegression)
"""

from __future__ import annotations

import warnings
import numpy as np
import pandas as pd
from typing import Optional
from concurrent.futures import ThreadPoolExecutor

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    StackingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import balanced_accuracy_score

try:
    from imblearn.ensemble import BalancedRandomForestClassifier
    HAS_IMBLEARN = True
except ImportError:
    from sklearn.ensemble import RandomForestClassifier
    HAS_IMBLEARN = False
    warnings.warn("imbalanced-learn not installed, using RandomForest instead of BalancedRF")

warnings.filterwarnings("ignore", category=UserWarning)


class EnhancedEnsemble:
    """7-모델 앙상블 + Stacking 메타러너.

    Usage:
        ens = EnhancedEnsemble(n_classes=2, use_stacking=True)
        ens.fit(X_train, y_train)
        probs = ens.predict_proba(X_test)
        preds = ens.predict(X_test)
    """

    def __init__(
        self,
        n_classes: int = 2,
        use_stacking: bool = True,
        n_jobs: int = -1,
        random_state: int = 42,
        verbose: bool = False,
    ):
        self.n_classes = n_classes
        self.use_stacking = use_stacking
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose

        self.models_ = {}
        self.weights_ = {}
        self.stacker_ = None
        self.feature_importance_ = None

    def _build_models(self, iter_num: int = 0) -> dict:
        """모델 딕셔너리 생성."""
        is_binary = self.n_classes == 2
        rs = self.random_state + iter_num

        models = {}

        # 1. LightGBM (GPU)
        lgb_params = {
            "objective": "binary" if is_binary else "multiclass",
            "metric": "binary_logloss" if is_binary else "multi_logloss",
            "num_leaves": 31 + iter_num * 4,
            "learning_rate": max(0.02, 0.06 - iter_num * 0.005),
            "min_data_in_leaf": max(3, 15 - iter_num * 2),
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "is_unbalance": True,
            "num_threads": 16,
            "device": "cpu",
            "verbose": -1,
            "seed": rs,
        }
        if not is_binary:
            lgb_params["num_class"] = self.n_classes
        models["lgb"] = {"type": "lgb", "params": lgb_params,
                         "n_rounds": 150 + iter_num * 30}

        # 2. XGBoost (GPU)
        xgb_params = {
            "objective": "binary:logistic" if is_binary else "multi:softprob",
            "learning_rate": max(0.02, 0.05 - iter_num * 0.004),
            "max_depth": 5 + iter_num,
            "n_estimators": 100 + iter_num * 20,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "tree_method": "hist",
            "device": "cuda",
            "verbosity": 0,
            "random_state": rs,
            "eval_metric": "logloss" if is_binary else "mlogloss",
        }
        if not is_binary:
            xgb_params["num_class"] = self.n_classes
        models["xgb"] = {"type": "xgb", "params": xgb_params}

        # 3. CatBoost (GPU)
        cb_params = {
            "iterations": 150 + iter_num * 20,
            "learning_rate": max(0.02, 0.05 - iter_num * 0.004),
            "depth": 5 + min(iter_num, 3),
            "auto_class_weights": "Balanced",
            "task_type": "GPU",
            "verbose": 0,
            "random_seed": rs,
        }
        models["catboost"] = {"type": "catboost", "params": cb_params}

        # 4. BalancedRandomForest (CPU) — class imbalance 자동 처리
        if HAS_IMBLEARN:
            brf = BalancedRandomForestClassifier(
                n_estimators=200 + iter_num * 20,
                max_depth=8 + iter_num,
                min_samples_leaf=max(3, 10 - iter_num),
                n_jobs=self.n_jobs,
                random_state=rs,
                sampling_strategy="all",
            )
        else:
            brf = RandomForestClassifier(
                n_estimators=200 + iter_num * 20,
                max_depth=8 + iter_num,
                min_samples_leaf=max(3, 10 - iter_num),
                class_weight="balanced",
                n_jobs=self.n_jobs,
                random_state=rs,
            )
        models["balanced_rf"] = {"type": "sklearn", "model": brf}

        # 5. ExtraTrees (CPU)
        et = ExtraTreesClassifier(
            n_estimators=200 + iter_num * 20,
            max_depth=8 + iter_num,
            min_samples_leaf=max(3, 10 - iter_num),
            class_weight="balanced",
            n_jobs=self.n_jobs,
            random_state=rs,
        )
        models["extra_trees"] = {"type": "sklearn", "model": et}

        # 6. HistGradientBoosting (CPU, sklearn native)
        hgb = HistGradientBoostingClassifier(
            max_iter=200 + iter_num * 30,
            learning_rate=max(0.02, 0.05 - iter_num * 0.004),
            max_depth=6 + iter_num,
            min_samples_leaf=max(5, 15 - iter_num * 2),
            max_leaf_nodes=31 + iter_num * 4,
            class_weight="balanced",
            random_state=rs,
            verbose=0,
        )
        models["hist_gb"] = {"type": "sklearn", "model": hgb}

        return models

    def fit(self, X: np.ndarray, y: np.ndarray,
            sample_weight: np.ndarray | None = None,
            iter_num: int = 0) -> "EnhancedEnsemble":
        """전체 앙상블 학습."""
        model_specs = self._build_models(iter_num)

        # GPU 모델(lgb, xgb, catboost)은 순차 실행, CPU 모델은 병렬 실행
        gpu_names  = {"lgb", "xgb", "catboost"}
        cpu_specs  = {n: s for n, s in model_specs.items() if n not in gpu_names}
        gpu_specs  = {n: s for n, s in model_specs.items() if n in gpu_names}

        # GPU 모델 순차 학습
        for name, spec in gpu_specs.items():
            try:
                model = self._fit_single(name, spec, X, y, sample_weight)
                if model is not None:
                    self.models_[name] = model
                    if self.verbose:
                        print(f"  [{name}] trained OK")
            except Exception as e:
                if self.verbose:
                    print(f"  [{name}] FAILED: {e}")

        # CPU 모델 병렬 학습 (ThreadPoolExecutor, GIL-free zone in sklearn/lgb)
        def _train(item):
            name, spec = item
            try:
                return name, self._fit_single(name, spec, X, y, sample_weight)
            except Exception as e:
                return name, None

        n_cpu_workers = min(len(cpu_specs), 4)
        with ThreadPoolExecutor(max_workers=n_cpu_workers) as ex:
            for name, model in ex.map(_train, cpu_specs.items()):
                if model is not None:
                    self.models_[name] = model
                    if self.verbose:
                        print(f"  [{name}] trained OK")

        if not self.models_:
            raise ValueError("No models trained successfully")

        # Stacking Meta-Learner
        if self.use_stacking and len(self.models_) >= 3:
            self._fit_stacker(X, y)
        else:
            # Fallback: balanced_accuracy weighted average
            self._compute_weights(X, y)

        # Feature importance (LGB 기반)
        if "lgb" in self.models_:
            self.feature_importance_ = self.models_["lgb"].feature_importance(
                importance_type="gain"
            )

        return self

    def _fit_single(self, name: str, spec: dict,
                    X: np.ndarray, y: np.ndarray,
                    sample_weight: np.ndarray | None) -> object:
        """개별 모델 학습."""
        mtype = spec["type"]

        if mtype == "lgb":
            dtrain = lgb.Dataset(X, label=y, weight=sample_weight, free_raw_data=False)
            model = lgb.train(
                spec["params"], dtrain,
                num_boost_round=spec["n_rounds"],
                valid_sets=[dtrain],
                callbacks=[lgb.log_evaluation(period=0)],
            )
            return model

        elif mtype == "xgb":
            params = spec["params"]
            n_est = params.pop("n_estimators", 100)
            clf = xgb.XGBClassifier(**params, n_estimators=n_est)
            clf.fit(X, y, sample_weight=sample_weight)
            return clf

        elif mtype == "catboost":
            clf = CatBoostClassifier(**spec["params"])
            clf.fit(X, y, sample_weight=sample_weight)
            return clf

        elif mtype == "sklearn":
            model = spec["model"]
            model.fit(X, y, sample_weight=sample_weight)
            return model

        return None

    def _predict_proba_single(self, name: str, model: object,
                              X: np.ndarray) -> np.ndarray:
        """개별 모델 확률 예측."""
        try:
            if isinstance(model, lgb.Booster):
                raw = model.predict(X)
                if self.n_classes == 2:
                    return np.column_stack([1 - raw, raw])
                return raw
            elif hasattr(model, "predict_proba"):
                return model.predict_proba(X)
        except Exception as e:
            if self.verbose:
                print(f"  [{name}] predict failed: {e}")
        return None

    def _fit_stacker(self, X: np.ndarray, y: np.ndarray) -> None:
        """Stacking meta-learner 학습 (OOF predictions 사용)."""
        n = len(X)
        n_models = len(self.models_)
        oof_probs = np.zeros((n, n_models * self.n_classes))

        # TimeSeriesSplit으로 OOF predictions 생성
        tscv = TimeSeriesSplit(n_splits=3, gap=12)

        for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_tr, y_tr = X[train_idx], y[train_idx]
            X_val = X[val_idx]

            fold_models = self._build_models(iter_num=0)
            for i, (name, spec) in enumerate(fold_models.items()):
                if name not in self.models_:
                    continue
                try:
                    fold_model = self._fit_single(name, spec, X_tr, y_tr, None)
                    if fold_model is not None:
                        probs = self._predict_proba_single(name, fold_model, X_val)
                        if probs is not None:
                            col_start = list(self.models_.keys()).index(name) * self.n_classes
                            oof_probs[val_idx, col_start:col_start + self.n_classes] = probs
                except Exception:
                    pass

        # 유효한 행만 사용
        valid_mask = np.any(oof_probs != 0, axis=1)
        if valid_mask.sum() < 30:
            if self.verbose:
                print("  [Stacker] Not enough OOF data, falling back to weights")
            self._compute_weights(X, y)
            return

        X_meta = oof_probs[valid_mask]
        y_meta = y[valid_mask]

        self.stacker_ = LogisticRegression(
            C=1.0, max_iter=500, class_weight="balanced",
            random_state=self.random_state, solver="lbfgs",
        )
        self.stacker_.fit(X_meta, y_meta)

        if self.verbose:
            oof_pred = self.stacker_.predict(X_meta)
            ba = balanced_accuracy_score(y_meta, oof_pred)
            print(f"  [Stacker] OOF balanced_accuracy: {ba:.4f}")

    def _compute_weights(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fallback: balanced_accuracy 가중치."""
        for name, model in self.models_.items():
            probs = self._predict_proba_single(name, model, X)
            if probs is not None:
                preds = np.argmax(probs, axis=1)
                ba = balanced_accuracy_score(y, preds)
                self.weights_[name] = max(ba, 0.01)

        total = sum(self.weights_.values()) + 1e-12
        for k in self.weights_:
            self.weights_[k] /= total

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """앙상블 확률 예측."""
        if self.stacker_ is not None:
            # Stacking: 모든 모델의 확률을 연결 → meta-learner
            meta_features = []
            for name in self.models_:
                model = self.models_[name]
                probs = self._predict_proba_single(name, model, X)
                if probs is not None:
                    meta_features.append(probs)
                else:
                    meta_features.append(np.ones((len(X), self.n_classes)) / self.n_classes)
            X_meta = np.hstack(meta_features)
            return self.stacker_.predict_proba(X_meta)
        else:
            # Weighted average fallback
            total_probs = np.zeros((len(X), self.n_classes))
            for name, model in self.models_.items():
                probs = self._predict_proba_single(name, model, X)
                if probs is not None:
                    w = self.weights_.get(name, 1.0 / len(self.models_))
                    total_probs += probs * w
            return total_probs

    def predict(self, X: np.ndarray) -> np.ndarray:
        """앙상블 클래스 예측."""
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)

    def get_model_scores(self, X: np.ndarray, y: np.ndarray) -> dict:
        """개별 모델 성능 평가."""
        scores = {}
        for name, model in self.models_.items():
            probs = self._predict_proba_single(name, model, X)
            if probs is not None:
                preds = np.argmax(probs, axis=1)
                ba = balanced_accuracy_score(y, preds)
                scores[name] = round(ba, 4)
        return scores

    def summary(self) -> dict:
        """앙상블 요약."""
        return {
            "n_models": len(self.models_),
            "model_names": list(self.models_.keys()),
            "use_stacking": self.stacker_ is not None,
            "weights": {k: round(v, 4) for k, v in self.weights_.items()} if self.weights_ else "stacking",
            "n_classes": self.n_classes,
        }
