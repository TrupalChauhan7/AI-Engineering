"""B6 human ceiling (inter-evaluator agreement) + B7 flagging PR-AUC.

Pre-registration §B6 and §B7. Kept separate from the frozen correlation.py.

B6 — the max correlation any metric can reach is capped by how much the human
evaluators agree with each other. With PriMock57's full 5-evaluator overlap we
estimate that ceiling as the mean pairwise Spearman between evaluators on the
error counts, then report each metric's ρ relative to it.

B7 — the reliability as a binary detector: positive class = a note the humans flagged
with a critical error. Report PR-AUC per metric, cluster-bootstrapped over
consultations (the score fed in is the *unreliability* direction, higher = more
likely positive).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import average_precision_score


def inter_evaluator_ceiling(
    human_eval: pd.DataFrame, target: str, consultation_ids: list[str], machine_only: bool = True
) -> dict:
    """Mean pairwise inter-evaluator Spearman on ``target`` = the human ceiling.

    ``human_eval`` is EvaluationLoader.load_human_eval() (one row per
    consultation/model/evaluator). Returns {ceiling, n_pairs, per_pair}.
    """
    df = human_eval[human_eval["Consultation"].isin(consultation_ids)].copy()
    if machine_only:
        df = df[~df["is_doctor"]]
    # (consultation, model) x evaluator matrix of the target.
    wide = df.pivot_table(
        index=["Consultation", "Model"], columns="Evaluator", values=target, aggfunc="mean"
    )
    evaluators = list(wide.columns)
    rhos = []
    for i in range(len(evaluators)):
        for j in range(i + 1, len(evaluators)):
            pair = wide[[evaluators[i], evaluators[j]]].dropna()
            if len(pair) >= 3 and pair.iloc[:, 0].nunique() > 1 and pair.iloc[:, 1].nunique() > 1:
                rhos.append(float(spearmanr(pair.iloc[:, 0], pair.iloc[:, 1]).statistic))
    return {
        "ceiling": float(np.mean(rhos)) if rhos else float("nan"),
        "n_pairs": len(rhos),
        "per_pair": [round(r, 3) for r in rhos],
    }


def pr_auc_bootstrap(
    df: pd.DataFrame,
    score_col: str,
    label_col: str,
    cluster_col: str = "consultation",
    n_boot: int = 5000,
    seed: int = 42,
) -> dict:
    """PR-AUC (average precision) with a cluster-bootstrap 95% CI.

    ``score_col`` must be oriented so HIGHER = more likely POSITIVE (i.e. the
    unreliability direction). ``label_col`` is 0/1. Resamples consultations.
    """
    d = df[[score_col, label_col, cluster_col]].dropna()
    y = d[label_col].to_numpy(int)
    s = d[score_col].to_numpy(float)
    clusters = d[cluster_col].to_numpy()
    base_rate = float(y.mean()) if len(y) else float("nan")

    point = float(average_precision_score(y, s)) if len(np.unique(y)) == 2 else float("nan")

    uniq = np.unique(clusters)
    pos = {c: np.where(clusters == c)[0] for c in uniq}
    rng = np.random.default_rng(seed)
    boot = np.full(n_boot, np.nan)
    for i in range(n_boot):
        drawn = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([pos[c] for c in drawn])
        yb, sb = y[idx], s[idx]
        if len(np.unique(yb)) == 2:
            boot[i] = average_precision_score(yb, sb)
    lo, hi = np.nanpercentile(boot, [2.5, 97.5])
    return {
        "pr_auc": round(point, 4),
        "lo": round(float(lo), 4),
        "hi": round(float(hi), 4),
        "base_rate": round(base_rate, 4),
    }
