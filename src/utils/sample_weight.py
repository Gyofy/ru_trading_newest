"""Sample Weighting -- uniqueness + recency.

Improves training data quality without changing model structure or parameters.
No quality/return-based weighting (overfitting risk with high-R:R strategy).

Usage:
    from src.utils.sample_weight import compute_sample_weights

    weights = compute_sample_weights(
        df=train_df,           # must have DatetimeIndex
        labels=y_labels,       # label array (0/1/2)
        horizon=18,            # max barrier horizon (bars)
        recency_halflife=45,   # days
    )
    # weights shape: (n_samples,), range [0.5, 3.0]
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def compute_sample_weights(
    df: pd.DataFrame,
    labels: np.ndarray,
    horizon: int = 18,
    recency_halflife: float = 45.0,
    alpha_uniq: float = 0.5,
    alpha_recency: float = 0.5,
    clip_min: float = 0.5,
    clip_max: float = 3.0,
) -> np.ndarray:
    """Compute sample weights: uniqueness * recency.

    Args:
        df: training dataframe with DatetimeIndex
        labels: label array
        horizon: max barrier horizon in bars (for overlap calc)
        recency_halflife: half-life in days for recency decay
        alpha_uniq: exponent for uniqueness weight
        alpha_recency: exponent for recency weight
        clip_min: minimum weight
        clip_max: maximum weight

    Returns:
        weight array, shape (n_samples,), clipped to [clip_min, clip_max]
    """
    n = len(df)
    if n == 0:
        return np.ones(0)

    w_uniq = _uniqueness_weights(n, labels, horizon)
    w_recency = _recency_weights(df.index, recency_halflife)

    # Combine: w = w_uniq^alpha * w_recency^alpha
    combined = np.power(w_uniq, alpha_uniq) * np.power(w_recency, alpha_recency)

    # Normalize to mean=1 before clipping
    mean_w = np.mean(combined)
    if mean_w > 0:
        combined = combined / mean_w

    # Clip
    combined = np.clip(combined, clip_min, clip_max)

    return combined


def _uniqueness_weights(n: int, labels: np.ndarray, horizon: int) -> np.ndarray:
    """Compute uniqueness based on barrier window overlap.

    For each sample i, count how many other samples have overlapping
    barrier windows [i, i+horizon]. More overlap = lower weight.
    """
    # Concurrency count: for each bar, how many active barriers overlap
    concurrent = np.ones(n, dtype=float)

    for i in range(n):
        window_end = min(i + horizon, n)
        # Count samples whose barrier window includes bar i
        overlap_start = max(0, i - horizon)
        overlap_count = 0
        for j in range(overlap_start, min(i + 1, n)):
            if j + horizon >= i:
                overlap_count += 1
        concurrent[i] = max(overlap_count, 1)

    # Weight = 1 / average_concurrency_during_own_window
    weights = np.ones(n, dtype=float)
    for i in range(n):
        window_end = min(i + horizon, n)
        if window_end > i:
            avg_conc = np.mean(concurrent[i:window_end])
            weights[i] = 1.0 / max(avg_conc, 1.0)
        else:
            weights[i] = 1.0

    # Normalize to [0, 1] range then scale to mean=1
    if weights.max() > weights.min():
        weights = (weights - weights.min()) / (weights.max() - weights.min())
    weights = weights + 0.5  # shift to [0.5, 1.5]

    return weights


def _recency_weights(index: pd.DatetimeIndex, halflife_days: float) -> np.ndarray:
    """Exponential recency decay.

    More recent samples get higher weight.
    Half-life: weight drops to 0.5 after halflife_days.
    """
    if len(index) == 0:
        return np.ones(0)

    latest = index[-1]
    days_ago = np.array([(latest - t).total_seconds() / 86400 for t in index])

    # Exponential decay: w = exp(-lambda * days)
    # lambda = ln(2) / halflife
    lam = np.log(2) / max(halflife_days, 1.0)
    weights = np.exp(-lam * days_ago)

    # Normalize to mean=1
    mean_w = np.mean(weights)
    if mean_w > 0:
        weights = weights / mean_w

    return weights
