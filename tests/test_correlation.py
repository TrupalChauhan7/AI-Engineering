"""Cluster-bootstrap correlation + paired Δρ sanity checks (synthetic data)."""

import numpy as np
import pandas as pd

from s2n.evaluation.correlation import cluster_bootstrap_rho, paired_delta_rho


def _synthetic():
    """A good faithfulness metric (anti-correlated with errors) and a noisy one,
    across 20 consultations x 5 notes."""
    rng = np.random.default_rng(0)
    rows = []
    for c in range(20):
        for _ in range(5):
            errors = rng.integers(0, 20)
            rows.append(
                {
                    "consultation": f"c{c:02d}",
                    "errors_total": errors,
                    "good": 1.0 - errors / 20 + rng.normal(0, 0.03),  # tracks errors well
                    "noise": rng.normal(0, 1),  # unrelated
                }
            )
    return pd.DataFrame(rows)


def test_good_metric_has_positive_aligned_rho():
    df = _synthetic()
    good = cluster_bootstrap_rho(df, "good", "errors_total", n_boot=500)
    noise = cluster_bootstrap_rho(df, "noise", "errors_total", n_boot=500)
    assert good.rho > 0.8  # aligned: higher = better tracking
    assert good.lo > 0  # CI clears zero
    assert abs(noise.rho) < 0.5


def test_paired_delta_favours_the_good_metric():
    df = _synthetic()
    d = paired_delta_rho(df, "good", "noise", "errors_total", n_boot=500)
    assert d.delta_rho > 0 and d.excludes_zero
