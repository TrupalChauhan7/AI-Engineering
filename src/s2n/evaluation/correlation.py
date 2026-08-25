"""RQ1 statistics — cluster bootstrap + paired Δρ (pre-registration §B2–B3).

WHY the cluster bootstrap: the 285 notes are NOT independent — 5 come from each
consultation. Treating them as 285 would fake precision. So we resample the
*consultations* (57 clusters; effective n ≈ 57), take all their notes, and
recompute. This is the honest confidence interval.

WHY paired Δρ: to ask "does the judge beat SummaC" we must compute BOTH
correlations inside the SAME resample and take the difference (§B2) — never two
separate CIs eyeballed. The paired design cancels shared sampling noise.

ORIENTATION: every metric here is a FAITHFULNESS score (higher = more faithful);
the human target (e.g. errors_total) is higher = WORSE. So raw Spearman is
negative for a good metric. We report the ALIGNED correlation
    aligned ρ = − ρ(metric, target)
so higher = better at tracking unreliability, and Δρ > 0 means "judge better".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


@dataclass
class RhoCI:
    metric: str
    rho: float  # aligned point estimate on the full data
    lo: float  # 2.5th percentile (cluster bootstrap)
    hi: float  # 97.5th percentile


def _aligned_rho(metric: np.ndarray, target: np.ndarray) -> float:
    """Aligned Spearman: −ρ(faithfulness, error-target); higher = better."""
    if len(metric) < 3 or np.all(metric == metric[0]) or np.all(target == target[0]):
        return float("nan")
    return float(-spearmanr(metric, target).statistic)


def _resample_indices(clusters: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Resample CLUSTERS with replacement; return the row indices of their notes."""
    uniq = np.unique(clusters)
    drawn = rng.choice(uniq, size=len(uniq), replace=True)
    # Map each cluster -> its row positions once, then gather.
    pos = {c: np.where(clusters == c)[0] for c in uniq}
    return np.concatenate([pos[c] for c in drawn])


def cluster_bootstrap_rho(
    df: pd.DataFrame,
    metric_col: str,
    target_col: str,
    cluster_col: str = "consultation",
    n_boot: int = 5000,
    seed: int = 42,
) -> RhoCI:
    """Aligned ρ for one metric, with a cluster-bootstrap 95% CI."""
    d = df[[metric_col, target_col, cluster_col]].dropna()
    metric = d[metric_col].to_numpy(float)
    target = d[target_col].to_numpy(float)
    clusters = d[cluster_col].to_numpy()

    point = _aligned_rho(metric, target)
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    for i in range(n_boot):
        idx = _resample_indices(clusters, rng)
        boot[i] = _aligned_rho(metric[idx], target[idx])
    lo, hi = np.nanpercentile(boot, [2.5, 97.5])
    return RhoCI(metric_col, round(point, 4), round(float(lo), 4), round(float(hi), 4))


@dataclass
class DeltaRhoCI:
    judge: str
    baseline: str
    delta_rho: float  # aligned ρ(judge) − aligned ρ(baseline); >0 => judge better
    lo: float
    hi: float
    excludes_zero: bool


def paired_delta_rho(
    df: pd.DataFrame,
    judge_col: str,
    baseline_col: str,
    target_col: str,
    cluster_col: str = "consultation",
    n_boot: int = 5000,
    seed: int = 42,
) -> DeltaRhoCI:
    """Paired Δρ = aligned ρ(judge) − aligned ρ(baseline), cluster-bootstrapped.

    Both correlations are recomputed on the SAME resample (§B2). CI excluding 0
    => the judge is statistically distinguishable from that baseline.
    """
    d = df[[judge_col, baseline_col, target_col, cluster_col]].dropna()
    j = d[judge_col].to_numpy(float)
    b = d[baseline_col].to_numpy(float)
    t = d[target_col].to_numpy(float)
    clusters = d[cluster_col].to_numpy()

    point = _aligned_rho(j, t) - _aligned_rho(b, t)
    rng = np.random.default_rng(seed)
    boot = np.empty(n_boot)
    for i in range(n_boot):
        idx = _resample_indices(clusters, rng)
        boot[i] = _aligned_rho(j[idx], t[idx]) - _aligned_rho(b[idx], t[idx])
    lo, hi = np.nanpercentile(boot, [2.5, 97.5])
    return DeltaRhoCI(
        judge_col,
        baseline_col,
        round(point, 4),
        round(float(lo), 4),
        round(float(hi), 4),
        excludes_zero=bool(lo > 0 or hi < 0),
    )


def correlation_table(
    df: pd.DataFrame,
    metric_cols: list[str],
    target_col: str,
    cluster_col: str = "consultation",
    n_boot: int = 5000,
    seed: int = 42,
) -> pd.DataFrame:
    """Aligned ρ + 95% CI for each metric (sorted best-first)."""
    rows = [
        cluster_bootstrap_rho(df, m, target_col, cluster_col, n_boot, seed).__dict__
        for m in metric_cols
    ]
    return pd.DataFrame(rows).sort_values("rho", ascending=False).reset_index(drop=True)
