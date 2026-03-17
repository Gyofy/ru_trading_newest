"""Feature Policy -- MI=0 피처 제거 + 레짐 필터 정책.

Strategy Diagnosis (2026-03-17) 결과 기반:
- macro 12개: MI=0.000 (3코인 공통 노이즈)
- hilbert 1개: MI < 0.01 (거의 기여 없음)
- RANGE_LOW 레짐: 손실/미미 -> 진입 차단
"""

# MI=0 확인된 노이즈 피처 키워드 (제거 대상)
EXCLUDED_FEATURE_KEYWORDS = [
    # Macro (3코인 공통 MI=0)
    "macro_gold",
    "macro_vix",
    "macro_usd_index",
    "macro_us_10y",
    "macro_sp500",
    "macro_fear_greed",
    "macro_defi_tvl",
    # Hilbert (MI < 0.01)
    "hilbert",
    "inst_phase",
    "inst_freq",
    "inst_amp",
]

# 진입 차단 레짐
BLOCKED_REGIMES = [
    "RANGE_LOW",
]


def is_excluded_feature(col_name: str) -> bool:
    """Check if a feature should be excluded based on MI=0 policy."""
    col_lower = col_name.lower()
    return any(kw in col_lower for kw in EXCLUDED_FEATURE_KEYWORDS)


def is_blocked_regime(regime: str) -> bool:
    """Check if regime should block new entries."""
    return regime in BLOCKED_REGIMES
