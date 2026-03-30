"""Deflated Sharpe Ratio (DSR) — 다중 검정 보정.

Reference: Bailey & Lopez de Prado,
"The Deflated Sharpe Ratio: Correcting for Selection Bias,
 Backtest Overfitting, and Non-Normality" (JPM, 2014).

시도 횟수(n_trials)가 늘어날수록 요구되는 Sharpe 임계치를 동적으로 높여,
우연히 좋은 성과를 내는 가짜 전략이 실전으로 넘어가는 것을 차단.
"""

from __future__ import annotations

import math
import logging
from typing import Optional

logger = logging.getLogger("dsr")


def _norm_cdf(x: float) -> float:
    """Standard normal CDF (no scipy dependency)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    """Approximate inverse normal CDF (Beasley-Springer-Moro)."""
    if p <= 0.0:
        return -10.0
    if p >= 1.0:
        return 10.0
    if p == 0.5:
        return 0.0

    # Rational approximation for central region
    if 0.08 < p < 0.92:
        r = p - 0.5
        r2 = r * r
        x = r * (2.515517 + 0.802853 * r2) / (1.0 + 0.010328 * r2 + 0.001308 * r2 * r2)
        return x

    # Tail approximation
    if p < 0.5:
        r = math.sqrt(-2.0 * math.log(p))
    else:
        r = math.sqrt(-2.0 * math.log(1.0 - p))

    x = r - (2.515517 + 0.802853 * r + 0.010328 * r * r) / (
        1.0 + 1.432788 * r + 0.189269 * r * r + 0.001308 * r * r * r
    )
    return x if p > 0.5 else -x


def expected_max_sr(n_trials: int, T: int) -> float:
    """E[max(SR)] under null hypothesis of n independent trials.

    Approximation from Bailey & LdP (2014), Eq. (6):
    E[max(SR)] ≈ √(2 ln(n)) - (ln(π) + ln(ln(n))) / (2 √(2 ln(n)))

    Args:
        n_trials: number of independent strategy/parameter trials
        T: number of observations (trades)

    Returns:
        Expected maximum Sharpe ratio under null (annualized)
    """
    if n_trials <= 1:
        return 0.0
    ln_n = math.log(n_trials)
    if ln_n <= 0:
        return 0.0
    sqrt_2ln = math.sqrt(2.0 * ln_n)
    euler_mascheroni = 0.5772156649
    return sqrt_2ln - (math.log(math.pi) + math.log(ln_n)) / (2.0 * sqrt_2ln)


def deflated_sharpe_ratio(
    sr_observed: float,
    sr_benchmark: float,
    T: int,
    skew: float = 0.0,
    kurtosis_excess: float = 0.0,
    n_trials: int = 1,
) -> float:
    """Compute the Deflated Sharpe Ratio.

    DSR = Φ[(SR_obs - SR_benchmark) / σ(SR)]

    where σ(SR) accounts for non-normality:
    σ(SR) = √[(1 - γ₃·SR + (γ₄-1)/4·SR²) / T]

    Args:
        sr_observed: observed Sharpe ratio (annualized)
        sr_benchmark: benchmark SR (default: E[max(SR)] from n_trials)
        T: number of observations
        skew: skewness of returns (γ₃)
        kurtosis_excess: excess kurtosis of returns (γ₄ - 3)
        n_trials: number of independent trials conducted

    Returns:
        DSR p-value in [0, 1]. DSR > 0.95 → statistically significant.
    """
    if T <= 1:
        return 0.0

    # If no explicit benchmark, use expected max SR under null
    if sr_benchmark == 0.0 and n_trials > 1:
        sr_benchmark = expected_max_sr(n_trials, T)

    # SR standard error with non-normality correction
    # Bailey & LdP (2014), Eq. (4)
    sr = sr_observed
    gamma3 = skew
    gamma4 = kurtosis_excess  # excess kurtosis (normal = 0)

    var_sr = (1.0 - gamma3 * sr + (gamma4 / 4.0) * sr * sr) / T
    if var_sr <= 0:
        var_sr = 1.0 / T  # fallback

    sr_std = math.sqrt(var_sr)

    if sr_std <= 0:
        return 0.0

    # PSR (Probabilistic Sharpe Ratio) = Φ[(SR_obs - SR_bench) / σ(SR)]
    z = (sr_observed - sr_benchmark) / sr_std
    return _norm_cdf(z)


class DSREvaluator:
    """Strategy evaluation with Deflated Sharpe Ratio.

    Tracks the number of trials (parameter combinations, strategies tested)
    and adjusts the significance threshold accordingly.
    """

    def __init__(self, significance_level: float = 0.95):
        self.significance_level = significance_level
        self._trials: dict[str, int] = {}  # context → n_trials

    def record_trial(self, context: str = "global") -> int:
        """Record a new trial (parameter test, strategy evaluation)."""
        self._trials[context] = self._trials.get(context, 0) + 1
        return self._trials[context]

    def evaluate(
        self,
        returns: list[float],
        context: str = "global",
        n_trials_override: Optional[int] = None,
    ) -> dict:
        """Evaluate a return series with DSR.

        Args:
            returns: list of per-trade returns (decimal, e.g. 0.008 = 0.8%)
            context: trial context key
            n_trials_override: override trial count

        Returns:
            {
                "sr": float,           # observed Sharpe ratio
                "dsr": float,          # Deflated Sharpe p-value [0,1]
                "significant": bool,   # DSR > significance_level
                "n_trials": int,
                "n_trades": int,
                "sr_benchmark": float, # E[max(SR)] under null
                "skew": float,
                "kurtosis_excess": float,
            }
        """
        n = len(returns)
        if n < 10:
            return {
                "sr": 0.0, "dsr": 0.0, "significant": False,
                "n_trials": 0, "n_trades": n,
                "sr_benchmark": 0.0, "skew": 0.0, "kurtosis_excess": 0.0,
            }

        n_trials = n_trials_override or self._trials.get(context, 1)

        mean_r = sum(returns) / n
        var_r = sum((r - mean_r) ** 2 for r in returns) / (n - 1)
        std_r = math.sqrt(var_r) if var_r > 0 else 1e-10

        sr = (mean_r / std_r) * math.sqrt(252)  # annualized (daily proxy)

        # Skewness
        skew = sum((r - mean_r) ** 3 for r in returns) / (n * std_r ** 3) if std_r > 0 else 0.0

        # Excess kurtosis
        kurt = sum((r - mean_r) ** 4 for r in returns) / (n * std_r ** 4) - 3.0 if std_r > 0 else 0.0

        sr_bench = expected_max_sr(n_trials, n)

        dsr = deflated_sharpe_ratio(
            sr_observed=sr,
            sr_benchmark=sr_bench,
            T=n,
            skew=skew,
            kurtosis_excess=kurt,
            n_trials=n_trials,
        )

        result = {
            "sr": round(sr, 4),
            "dsr": round(dsr, 4),
            "significant": dsr >= self.significance_level,
            "n_trials": n_trials,
            "n_trades": n,
            "sr_benchmark": round(sr_bench, 4),
            "skew": round(skew, 4),
            "kurtosis_excess": round(kurt, 4),
        }

        logger.info(
            f"[DSR] SR={sr:.2f} DSR={dsr:.3f} "
            f"{'PASS' if result['significant'] else 'FAIL'} "
            f"(bench={sr_bench:.2f}, n={n}, trials={n_trials})"
        )
        return result
