"""RQ2 paired statistics — pre-registration §C (+ Amendment 3).

Design (locked): within-consultation PAIRED comparison. Same generator, same
prompt version, same frozen judge on both arms; only the input transcript
differs — (a) clean/gold vs (b) ASR. Per-consultation effect:

    delta = score(clean) - score(asr)      # >0 => ASR degraded the note

Inference: bootstrap the mean delta over consultations (each consultation
contributes exactly one delta, so the cluster bootstrap reduces to resampling
consultations — the same unit-of-analysis discipline as RQ1). 95% percentile
CI; a CI containing 0 is a *finding* (§C is null-friendly: modern LLMs may
absorb minor ASR errors).

Dose-response: Spearman of per-consultation WER vs delta. PRIMARY dose =
content-WER, SECONDARY = verbatim-WER (Amendment 3). Rank correlation tolerates
non-linear / threshold-shaped degradation (§C prior).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import spearmanr


@dataclass
class PairedDelta:
    scorer: str
    mean_delta: float  # mean over consultations; >0 => ASR arm worse
    lo: float
    hi: float
    n: int
    excludes_zero: bool


def paired_bootstrap_delta(
    deltas: np.ndarray, scorer: str, n_boot: int = 5000, seed: int = 42
) -> PairedDelta:
    """Bootstrap the mean per-consultation delta (resample consultations)."""
    deltas = np.asarray(deltas, dtype=float)
    deltas = deltas[~np.isnan(deltas)]
    n = len(deltas)
    if n < 2:
        return PairedDelta(scorer, float("nan"), float("nan"), float("nan"), n, False)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    boot_means = deltas[idx].mean(axis=1)
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    return PairedDelta(
        scorer,
        round(float(deltas.mean()), 4),
        round(float(lo), 4),
        round(float(hi), 4),
        n,
        excludes_zero=bool(lo > 0 or hi < 0),
    )


def dose_response(wer: np.ndarray, deltas: np.ndarray) -> dict:
    """Spearman rho of per-consultation WER vs delta (rank-based, §C)."""
    wer = np.asarray(wer, dtype=float)
    deltas = np.asarray(deltas, dtype=float)
    mask = ~(np.isnan(wer) | np.isnan(deltas))
    if mask.sum() < 3:
        return {"rho": float("nan"), "p": float("nan"), "n": int(mask.sum())}
    r = spearmanr(wer[mask], deltas[mask])
    return {
        "rho": round(float(r.statistic), 4),
        "p": round(float(r.pvalue), 4),
        "n": int(mask.sum()),
    }
