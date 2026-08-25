"""RQ2 paired statistics (synthetic data — no models)."""

import numpy as np

from s2n.evaluation.rq2 import dose_response, paired_bootstrap_delta


def test_clear_degradation_is_detected():
    deltas = np.array([0.2, 0.25, 0.15, 0.3, 0.2, 0.25, 0.1, 0.2, 0.3, 0.15])
    d = paired_bootstrap_delta(deltas, "judge", n_boot=1000)
    assert d.mean_delta > 0.15 and d.excludes_zero and d.n == 10


def test_null_effect_ci_contains_zero():
    rng = np.random.default_rng(0)
    deltas = rng.normal(0, 0.1, size=10)  # no systematic degradation
    d = paired_bootstrap_delta(deltas, "judge", n_boot=1000)
    assert not d.excludes_zero  # null-friendly: CI must contain 0


def test_nan_deltas_are_dropped():
    d = paired_bootstrap_delta(np.array([0.1, np.nan, 0.2, 0.15]), "x", n_boot=200)
    assert d.n == 3


def test_dose_response_monotone():
    wer = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
    deltas = wer * 2  # perfectly rank-correlated
    r = dose_response(wer, deltas)
    assert r["rho"] == 1.0 and r["n"] == 5
