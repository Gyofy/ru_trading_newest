"""설정 로드 유틸리티.

2-Tier 시간축 설정을 단일 소스에서 제공:
  strategic (1h) : Scanner, Regime, Discussion
  tactical  (4h) : ML/DL 모델, 라벨링, 피처 (1h fetch → 4h resample)
  execution (4h) : 시그널 정책, 포지션 관리
"""

from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIR = PROJECT_ROOT / "config"
DATA_DIR = PROJECT_ROOT / "data"

# .env 파일에서 환경변수 자동 로드
load_dotenv(PROJECT_ROOT / ".env")

_settings_cache: dict | None = None


def load_settings() -> dict[str, Any]:
    """config/settings.yaml 로드 (캐싱).

    SETTINGS_YAML_PATH 환경변수가 있으면 해당 경로를 사용.
    1m 재학습 등 별도 설정이 필요할 때 사용.
    """
    import os
    global _settings_cache
    if _settings_cache is None:
        override = os.environ.get("SETTINGS_YAML_PATH")
        path = Path(override) if override else CONFIG_DIR / "settings.yaml"
        with open(path, encoding="utf-8") as f:
            _settings_cache = yaml.safe_load(f)
    return _settings_cache


def get_data_path(sub: str, date_str: str | None = None) -> Path:
    """data/{sub}/{date}/ 경로 반환. 없으면 생성."""
    path = DATA_DIR / sub
    if date_str:
        path = path / date_str
    path.mkdir(parents=True, exist_ok=True)
    return path


# -------------------------------------------------------------------
# 2-Tier 시간축 헬퍼
# -------------------------------------------------------------------

def get_strategic() -> dict:
    """strategic tier 설정 반환 (scanner, regime, discussion)."""
    return load_settings()["timeframes"]["strategic"]


def get_tactical() -> dict:
    """tactical tier 설정 반환 (models, labeling, features)."""
    return load_settings()["timeframes"]["tactical"]


def get_execution() -> dict:
    """execution tier 설정 반환 (signal policy, position, orders)."""
    return load_settings()["timeframes"]["execution"]


def bar_minutes() -> int:
    """기본 바 간격 (분). 모든 bars→minutes 변환의 기준."""
    return load_settings()["timeframes"]["tactical"]["bar_minutes"]


def horizons() -> list[int]:
    """예측 수평선 (bars 단위). [1, 3, 6, 12]"""
    return load_settings()["timeframes"]["tactical"]["horizons"]


def horizon_labels() -> dict[int, str]:
    """예측 수평선 → 사람 읽기 라벨. {1: '4h', 3: '12h', ...}"""
    bm = bar_minutes()
    result = {}
    for h in horizons():
        total_min = h * bm
        if total_min >= 1440:
            result[h] = f"{total_min // 1440}d"
        elif total_min >= 60:
            result[h] = f"{total_min // 60}h"
        else:
            result[h] = f"{total_min}min"
    return result


def max_horizon() -> int:
    """최대 수평선 (bars). CV gap 등에 사용."""
    return load_settings()["timeframes"]["tactical"]["max_horizon"]


def seq_len() -> int:
    """DL 입력 시퀀스 길이 (bars)."""
    return load_settings()["timeframes"]["tactical"]["seq_len"]


def max_features() -> int:
    """MI 피처 선택 상한."""
    return load_settings()["data"]["feature_selection"]["max_features"]
