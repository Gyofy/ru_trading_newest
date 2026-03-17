"""Model Store — 학습된 앙상블 + 메타데이터 저장/로드.

학습 완료 후 save_artifact(), 라이브에서 load_artifact().
feature_cols, best_params 포함하여 train/live 일관성 보장.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import joblib

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path("C:/Users/RJ/Desktop/CLAUDE_CRYPTO_AGENT")
DEFAULT_ARTIFACT_DIR = PROJECT_ROOT / "data" / "models" / "production"


def save_artifact(
    symbol: str,
    stage1_ensemble,
    stage2_ensemble,
    feature_cols: list[str],
    params: dict,
    artifact_dir: Path | str = DEFAULT_ARTIFACT_DIR,
) -> Path:
    """학습된 2-stage 앙상블 저장.

    Args:
        symbol: 코인 심볼 (e.g., "BTC")
        stage1_ensemble: Stage1 (Trade/NoTrade) EnhancedEnsemble
        stage2_ensemble: Stage2 (Long/Short) EnhancedEnsemble
        feature_cols: MI 선택된 피처 컬럼 리스트 (순서 보존)
        params: best hyperparameters dict
            - k_upper, k_lower, stage1_threshold
            - max_features, num_leaves, learning_rate, etc.
    """
    artifact_dir = Path(artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    artifact = {
        "symbol": symbol,
        "stage1_ensemble": stage1_ensemble,
        "stage2_ensemble": stage2_ensemble,
        "feature_cols": feature_cols,
        "params": params,
        "saved_at": datetime.now(timezone.utc).isoformat(),
        "version": "v3",
    }

    path = artifact_dir / f"{symbol}_ensemble.joblib"
    joblib.dump(artifact, path, compress=3)
    logger.info(f"[ModelStore] Saved {symbol}: {len(feature_cols)} features, {path}")

    # Also save metadata as JSON (human-readable)
    meta_path = artifact_dir / f"{symbol}_meta.json"
    meta = {
        "symbol": symbol,
        "feature_cols": feature_cols,
        "n_features": len(feature_cols),
        "params": params,
        "saved_at": artifact["saved_at"],
        "version": "v3",
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    return path


def load_artifact(
    symbol: str,
    artifact_dir: Path | str = DEFAULT_ARTIFACT_DIR,
) -> dict:
    """학습된 2-stage 앙상블 로드.

    Returns:
        {
            "symbol": str,
            "stage1_ensemble": EnhancedEnsemble,
            "stage2_ensemble": EnhancedEnsemble,
            "feature_cols": list[str],
            "params": dict,
            "saved_at": str,
        }
    """
    artifact_dir = Path(artifact_dir)
    path = artifact_dir / f"{symbol}_ensemble.joblib"

    if not path.exists():
        raise FileNotFoundError(f"No artifact for {symbol}: {path}")

    artifact = joblib.load(path)
    logger.info(
        f"[ModelStore] Loaded {symbol}: "
        f"{len(artifact['feature_cols'])} features, "
        f"saved {artifact['saved_at']}"
    )
    return artifact


def list_artifacts(
    artifact_dir: Path | str = DEFAULT_ARTIFACT_DIR,
) -> list[dict]:
    """Available model artifacts."""
    artifact_dir = Path(artifact_dir)
    if not artifact_dir.exists():
        return []

    artifacts = []
    for meta_path in artifact_dir.glob("*_meta.json"):
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            # Check corresponding joblib exists
            joblib_path = artifact_dir / f"{meta['symbol']}_ensemble.joblib"
            meta["has_model"] = joblib_path.exists()
            artifacts.append(meta)
        except Exception as e:
            logger.warning(f"[ModelStore] Failed to read {meta_path}: {e}")

    return artifacts
