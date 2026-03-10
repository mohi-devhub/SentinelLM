"""Statistical regression detection for SentinelLM v2.

Uses Mann-Whitney U test (non-parametric, no normality assumption) and Cohen's d
effect size to determine whether score distributions have regressed between two
eval runs. More robust than a fixed flag-rate delta threshold on small datasets.

Regression is declared when:
  - p_value < alpha (statistically significant shift), AND
  - effect_size >= 'small' (|Cohen's d| >= 0.2, practical significance)

Direction is inferred from the mean shift relative to flag_direction:
  - 'above' evaluators (risk): higher current mean → regression
  - 'below' evaluators (quality): lower current mean → regression

Fallback: when n < 5 for either group, or when scipy is not installed,
returns a result based on flag-rate delta only (mirrors v1 behaviour).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Literal


@dataclass
class StatisticalRegressionResult:
    """Per-evaluator statistical regression summary."""

    mann_whitney_u: float
    p_value: float
    cohens_d: float
    effect_size: Literal["negligible", "small", "medium", "large"]
    significant: bool
    direction: Literal["regression", "improvement", "stable"]
    # Kept for backward compatibility with v1 reporter output
    flag_rate_delta: float
    n_current: int
    n_baseline: int

    def to_dict(self) -> dict:
        return asdict(self)


# ── Internal helpers ──────────────────────────────────────────────────────────


def _cohens_d(a: list[float], b: list[float]) -> float:
    """Cohen's d with pooled standard deviation. Returns 0.0 for n < 2."""
    n_a, n_b = len(a), len(b)
    if n_a < 2 or n_b < 2:
        return 0.0
    mean_a = sum(a) / n_a
    mean_b = sum(b) / n_b
    var_a = sum((x - mean_a) ** 2 for x in a) / (n_a - 1)
    var_b = sum((x - mean_b) ** 2 for x in b) / (n_b - 1)
    pooled_sd = math.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
    if pooled_sd == 0.0:
        return 0.0
    return (mean_a - mean_b) / pooled_sd


def _effect_label(d: float) -> Literal["negligible", "small", "medium", "large"]:
    abs_d = abs(d)
    if abs_d < 0.2:
        return "negligible"
    if abs_d < 0.5:
        return "small"
    if abs_d < 0.8:
        return "medium"
    return "large"


def _fallback(
    current_scores: list[float],
    baseline_scores: list[float],
    current_flags: int,
    baseline_flags: int,
    flag_direction: Literal["above", "below"],
) -> StatisticalRegressionResult:
    """Flag-rate delta fallback when n < 5 or scipy is unavailable."""
    n_cur = max(len(current_scores), 1)
    n_base = max(len(baseline_scores), 1)
    cur_rate = current_flags / n_cur
    base_rate = baseline_flags / n_base
    delta = cur_rate - base_rate

    if flag_direction == "above":
        direction: Literal["regression", "improvement", "stable"] = (
            "regression" if delta > 0.05 else ("improvement" if delta < -0.05 else "stable")
        )
    else:
        direction = (
            "regression" if delta < -0.05 else ("improvement" if delta > 0.05 else "stable")
        )

    return StatisticalRegressionResult(
        mann_whitney_u=0.0,
        p_value=1.0,
        cohens_d=0.0,
        effect_size="negligible",
        significant=False,
        direction=direction,
        flag_rate_delta=delta,
        n_current=len(current_scores),
        n_baseline=len(baseline_scores),
    )


# ── Public API ────────────────────────────────────────────────────────────────


def compute_statistical_regression(
    current_scores: list[float],
    baseline_scores: list[float],
    flag_direction: Literal["above", "below"],
    alpha: float = 0.05,
    current_flags: int = 0,
    baseline_flags: int = 0,
) -> StatisticalRegressionResult:
    """Run Mann-Whitney U + Cohen's d on two score distributions.

    Args:
        current_scores:  Raw evaluator scores from the current run.
        baseline_scores: Raw evaluator scores from the baseline run.
        flag_direction:  'above' (risk evaluators) or 'below' (quality evaluators).
        alpha:           Significance level (default 0.05).
        current_flags:   Flag count used in fallback flag-rate delta computation.
        baseline_flags:  Flag count used in fallback flag-rate delta computation.

    Returns:
        StatisticalRegressionResult with all fields populated.

    Falls back to flag-rate delta when n < 5 for either group or when scipy is
    not installed.
    """
    n_cur = len(current_scores)
    n_base = len(baseline_scores)

    if n_cur < 5 or n_base < 5:
        return _fallback(
            current_scores, baseline_scores, current_flags, baseline_flags, flag_direction
        )

    try:
        from scipy.stats import mannwhitneyu  # noqa: PLC0415

        stat, p_value = mannwhitneyu(current_scores, baseline_scores, alternative="two-sided")
        u_stat = float(stat)
        p_value = float(p_value)
    except ImportError:
        return _fallback(
            current_scores, baseline_scores, current_flags, baseline_flags, flag_direction
        )

    d = _cohens_d(current_scores, baseline_scores)
    effect = _effect_label(d)
    significant = p_value < alpha and effect != "negligible"

    cur_mean = sum(current_scores) / n_cur
    base_mean = sum(baseline_scores) / n_base
    mean_delta = cur_mean - base_mean

    if not significant:
        direction: Literal["regression", "improvement", "stable"] = "stable"
    elif flag_direction == "above":
        direction = "regression" if mean_delta > 0 else "improvement"
    else:
        direction = "regression" if mean_delta < 0 else "improvement"

    n_cur_safe = max(n_cur, 1)
    n_base_safe = max(n_base, 1)
    flag_rate_delta = (current_flags / n_cur_safe) - (baseline_flags / n_base_safe)

    return StatisticalRegressionResult(
        mann_whitney_u=u_stat,
        p_value=p_value,
        cohens_d=d,
        effect_size=effect,
        significant=significant,
        direction=direction,
        flag_rate_delta=flag_rate_delta,
        n_current=n_cur,
        n_baseline=n_base,
    )
