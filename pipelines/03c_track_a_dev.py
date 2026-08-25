"""RQ1 Track A — full DEV evaluation (DEV ONLY; TEST is never touched here).

Scores every pre-rated DEV note with all three metric tiers, then reports which
metric best tracks the human error target (primary = errors_total, §B4):
  tier 1  ROUGE-L, BERTScore, Levenshtein   (note vs gold reference note)
  tier 2  SummaC-ZS                          (note vs transcript) — primary rival
  tier 3  faithfulness judge v1.0            (note vs transcript) — our method

Headline: paired Δρ(judge − SummaC), cluster-bootstrapped over consultations.
The doctor note is an outlier (open include/exclude decision), so we report
MACHINE-ONLY (primary) and ALL (sensitivity).

Scores are cached to results/track_a_dev_scores.csv (judge + SummaC are slow);
re-run stats instantly with --no-refresh. Nothing here reads the TEST split.

Run:  python pipelines/03c_track_a_dev.py            # compute (slow) + report
      python pipelines/03c_track_a_dev.py --no-refresh  # reuse cached scores
"""

from __future__ import annotations

import argparse

import pandas as pd

from s2n.config import ROOT, load_config
from s2n.evaluation.correlation import correlation_table, paired_delta_rho
from s2n.evaluation.judge import FaithfulnessJudge
from s2n.evaluation.metrics import bertscore, levenshtein_similarity, rouge_l
from s2n.evaluation.summac_metric import SummaCZS
from s2n.evaluation.track_a import build_dev_table

METRIC_COLS = ["rougeL", "bertscore", "levenshtein", "summac", "judge_score"]
TARGET = "errors_total"
SCORES_PATH = ROOT / "results" / "track_a_dev_scores.csv"
CORR_PATH = ROOT / "results" / "track_a_dev_correlation.csv"


def compute_scores(cfg: dict) -> pd.DataFrame:
    # Pin the HISTORICAL v1.0 judge config (llama, holistic JSON). The global
    # config's judge block is frozen to Selene v1.2, but this pipeline is the
    # v1.0 experiment — it must stay reproducible regardless of the freeze.
    # (num_ctx comes from llm config: re-runs after 12 Jul are truncation-free.)
    cfg = {
        **cfg,
        "judge": {"model": "llama3.1:8b", "prompt_version": "v1.0", "mode": "json"},
    }
    t = build_dev_table(cfg)
    notes = t["note_text"].tolist()
    refs = t["reference_note"].tolist()
    srcs = t["transcript"].tolist()

    print(f"Scoring {len(t)} DEV notes with all metric tiers ...")
    t["rougeL"] = [rouge_l(n, r) for n, r in zip(notes, refs)]
    t["levenshtein"] = [levenshtein_similarity(n, r) for n, r in zip(notes, refs)]
    print("  reference metrics done; running BERTScore ...")
    t["bertscore"] = bertscore(notes, refs)
    print("  BERTScore done; running SummaC-ZS ...")
    t["summac"] = SummaCZS(cfg).score_many(srcs, notes)
    print("  SummaC done; running the judge (llama3.1:8b) ...")
    judged = FaithfulnessJudge(cfg).judge_many(srcs, notes)
    t["judge_score"] = [j["score"] for j in judged]
    t["judge_n_halluc"] = [j["n_hallucinations"] for j in judged]
    t["judge_n_omiss"] = [j["n_omissions"] for j in judged]
    t["judge_parse_ok"] = [j["parse_ok"] for j in judged]

    SCORES_PATH.parent.mkdir(parents=True, exist_ok=True)
    keep = [
        "consultation",
        "model",
        "is_doctor",
        TARGET,
        "errors_critical",
        "Post-edit time",
        *METRIC_COLS,
        "judge_n_halluc",
        "judge_n_omiss",
        "judge_parse_ok",
    ]
    t[keep].to_csv(SCORES_PATH, index=False)
    print(f"  cached scores -> {SCORES_PATH.relative_to(ROOT)}")
    return t


def report(df: pd.DataFrame, name: str, seed: int) -> None:
    print(
        f"\n{'=' * 70}\n{name}  (n={len(df)} notes, {df.consultation.nunique()} consultations)"
        f"\n{'=' * 70}"
    )
    tbl = correlation_table(df, METRIC_COLS, TARGET, seed=seed)
    print(f"Aligned Spearman ρ vs {TARGET}  (higher = better flags unreliable notes):")
    print(tbl.to_string(index=False))

    print("\nPaired Δρ (judge − baseline), cluster-bootstrapped; >0 favours the judge:")
    for base in ["summac", "levenshtein", "bertscore", "rougeL"]:
        d = paired_delta_rho(df, "judge_score", base, TARGET, seed=seed)
        tag = " [PRIMARY]" if base == "summac" else ""
        verdict = "distinguishable" if d.excludes_zero else "indistinguishable (CI spans 0)"
        print(
            f"  judge − {base:<11} Δρ={d.delta_rho:+.3f}  95% CI [{d.lo:+.3f}, {d.hi:+.3f}]"
            f"  {verdict}{tag}"
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-refresh", action="store_true", help="reuse cached scores")
    args = ap.parse_args()
    cfg = load_config()
    seed = cfg["seed"]

    if args.no_refresh and SCORES_PATH.exists():
        print(f"Loading cached scores from {SCORES_PATH.relative_to(ROOT)}")
        df = pd.read_csv(SCORES_PATH)
    else:
        df = compute_scores(cfg)

    if not df["judge_parse_ok"].all():
        print(f"\n⚠ judge JSON parse failures: {(~df['judge_parse_ok']).sum()} note(s)")

    report(df[~df.is_doctor], "MACHINE-ONLY DEV (doctor excluded — primary view)", seed)
    report(df, "ALL DEV notes (machine + doctor — sensitivity)", seed)

    # Persist the machine-only correlation table for the write-up.
    correlation_table(df[~df.is_doctor], METRIC_COLS, TARGET, seed=seed).to_csv(
        CORR_PATH, index=False
    )
    print(f"\nSaved correlation table -> {CORR_PATH.relative_to(ROOT)}")
    print("\nNOTE: DEV only. Rubric may be iterated (≤3 versions) before the one-shot TEST run.")


if __name__ == "__main__":
    main()
