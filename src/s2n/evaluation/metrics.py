"""Tier-1 baselines: reference-based traditional metrics (all free, local).

WHY (pre-registration §B1, tier 1): these are the "traditional metrics" the
project contrasts against. Each compares the NOTE to the clinician's GOLD
REFERENCE NOTE — a *different object* from the transcript the judge reads. That
structural difference is the whole point of the three-tier design: beating
ROUGE alone is a weak claim, so SummaC (tier 2, source-grounded) is the real
rival. These three are the Moramarco-2022 baselines (Levenshtein ≈ BERTScore).

Orientation: every metric here returns a FAITHFULNESS score where HIGHER = more
faithful (more similar to the gold note). The correlation harness handles the
sign when correlating against the human error target.
"""

from __future__ import annotations

import Levenshtein
import pandas as pd
from rouge_score import rouge_scorer

_ROUGE = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


def rouge_l(pred: str, ref: str) -> float:
    """ROUGE-L F-measure (longest-common-subsequence overlap). Higher = better."""
    return float(_ROUGE.score(ref, pred)["rougeL"].fmeasure)


def levenshtein_similarity(pred: str, ref: str) -> float:
    """Edit-distance similarity in [0,1] (Levenshtein.ratio). Higher = better."""
    return float(Levenshtein.ratio(pred, ref))


def bertscore(preds: list[str], refs: list[str], model_type: str | None = None) -> list[float]:
    """BERTScore F1 per pair (semantic similarity). Batched — loads the model once.

    Import is local so the (heavy) torch/transformers stack is only pulled in
    when BERTScore is actually used.
    """
    from bert_score import score as _bs

    _, _, f1 = _bs(
        preds,
        refs,
        lang="en",
        model_type=model_type,  # None -> bert-score's default English model
        rescale_with_baseline=False,
        verbose=False,
    )
    return [float(x) for x in f1]


def reference_metrics(
    preds: list[str], refs: list[str], with_bertscore: bool = True
) -> pd.DataFrame:
    """All tier-1 metrics for aligned (pred, ref) lists -> one row per note."""
    out = pd.DataFrame(
        {
            "rougeL": [rouge_l(p, r) for p, r in zip(preds, refs)],
            "levenshtein": [levenshtein_similarity(p, r) for p, r in zip(preds, refs)],
        }
    )
    if with_bertscore:
        out["bertscore"] = bertscore(preds, refs)
    return out
