"""Endpoint (human-target) selection — pre-registration §B4.

WHY: critical-error counts can be zero-inflated (sparse), which makes an
unstable correlation target. Before locking the primary target we may inspect
the *marginal distribution* of the candidate targets ON DEV ONLY — their spread
and zero-fraction — but NOT their correlation with any metric (that would leak).

Rule (§B4): pick as primary the target with usable variance on dev
(zero-fraction < 0.60), in preference order:
    total-error count  ->  critical-error count  ->  post-edit time
Log the choice and the dev distribution that justified it, before touching TEST.
"""

from __future__ import annotations

import pandas as pd

# Candidate targets in the pre-registered preference order.
CANDIDATE_TARGETS = ["errors_total", "errors_critical", "Post-edit time"]
MAX_ZERO_FRACTION = 0.60


def describe_targets(table: pd.DataFrame, targets: list[str] = CANDIDATE_TARGETS) -> pd.DataFrame:
    """Per-target distribution summary (spread + zero-fraction), no correlations."""
    rows = []
    for t in targets:
        s = pd.to_numeric(table[t], errors="coerce").dropna()
        zero_frac = float((s == 0).mean()) if len(s) else 1.0
        rows.append(
            {
                "target": t,
                "n": int(len(s)),
                "zero_fraction": round(zero_frac, 3),
                "mean": round(float(s.mean()), 3),
                "std": round(float(s.std()), 3),
                "min": round(float(s.min()), 3),
                "max": round(float(s.max()), 3),
                "usable": zero_frac < MAX_ZERO_FRACTION,
            }
        )
    return pd.DataFrame(rows)


def choose_primary_target(summary: pd.DataFrame) -> str:
    """First candidate (in preference order) that is usable (§B4)."""
    for t in CANDIDATE_TARGETS:
        row = summary[summary["target"] == t]
        if not row.empty and bool(row["usable"].iloc[0]):
            return t
    raise ValueError("No candidate target has usable variance on dev (all >=60% zeros).")
