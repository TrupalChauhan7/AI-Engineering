"""RQ1 Track A — judge rubric v1.1 on DEV (count-to-count). DEV ONLY.

v1.0 used a holistic 0–1 judge score; llama-8b collapsed it to just 0.6/0.8, so
it couldn't rank notes. v1.1 instead ENUMERATES every hallucination + omission
and uses the COUNT (n_hallucinations + n_omissions) as the signal, correlated
count-to-count against the human `errors_total`.

Orientation: the judge count is higher = WORSE (more errors), the same direction
as errors_total. To sit on the harness's "higher = better" aligned scale we feed
the NEGATED count as the metric, so its aligned ρ = ρ(count, errors_total) — the
count-to-count Spearman the rubric is designed for.

Reuses the cached tier-1/2 scores from 03c (BERTScore, SummaC, ROUGE, Leven.)
and reports the judge against BERTScore (the v1.0 leader) prominently, then
SummaC (the pre-registered primary comparator).

Run:  python pipelines/03d_judge_v11_dev.py               # run v1.1 judge (slow)
      python pipelines/03d_judge_v11_dev.py --no-refresh  # reuse cached v1.1
"""

from __future__ import annotations

import argparse

import pandas as pd

from s2n.config import ROOT, load_config
from s2n.evaluation.correlation import correlation_table, paired_delta_rho
from s2n.evaluation.judge import FaithfulnessJudge
from s2n.evaluation.track_a import build_dev_table

TARGET = "errors_total"
BASE_SCORES = ROOT / "results" / "track_a_dev_scores.csv"  # from 03c
V11_PATH = ROOT / "results" / "track_a_dev_judge_v11.csv"
# Metrics shown in the table (all faithfulness-oriented / higher = better).
METRIC_COLS = ["bertscore", "summac", "rougeL", "levenshtein", "judge_score", "judge_v11_neg_count"]


def run_v11_judge(cfg: dict) -> pd.DataFrame:
    # Pin the HISTORICAL v1.1 judge config (llama, enumerate-and-count JSON).
    # The global judge block is frozen to Selene v1.2; this pipeline is the
    # v1.1 experiment and must stay reproducible regardless of the freeze.
    cfg = {
        **cfg,
        "judge": {"model": "llama3.1:8b", "prompt_version": "v1.1", "mode": "json"},
    }
    t = build_dev_table(cfg)
    print(f"Running judge v1.1 (count mode) on {len(t)} DEV notes ...")
    judged = FaithfulnessJudge(cfg).judge_many(t["transcript"].tolist(), t["note_text"].tolist())
    t["judge_v11_n_halluc"] = [j["n_hallucinations"] for j in judged]
    t["judge_v11_n_omiss"] = [j["n_omissions"] for j in judged]
    t["judge_v11_count"] = t["judge_v11_n_halluc"] + t["judge_v11_n_omiss"]
    t["judge_v11_parse_ok"] = [j["parse_ok"] for j in judged]
    out = t[
        [
            "consultation",
            "model",
            "judge_v11_n_halluc",
            "judge_v11_n_omiss",
            "judge_v11_count",
            "judge_v11_parse_ok",
        ]
    ]
    V11_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(V11_PATH, index=False)
    print(f"  cached v1.1 judge -> {V11_PATH.relative_to(ROOT)}")
    return out


def report(df: pd.DataFrame, name: str, seed: int) -> None:
    print(
        f"\n{'=' * 72}\n{name}  (n={len(df)} notes, {df.consultation.nunique()} consultations)"
        f"\n{'=' * 72}"
    )
    tbl = correlation_table(df, METRIC_COLS, TARGET, seed=seed)
    print(f"Aligned Spearman ρ vs {TARGET}  (higher = better flags unreliable notes):")
    print(tbl.to_string(index=False))

    print("\nPaired Δρ (judge v1.1 count − baseline); >0 favours the judge:")
    for base in ["bertscore", "summac", "levenshtein", "rougeL", "judge_score"]:
        d = paired_delta_rho(df, "judge_v11_neg_count", base, TARGET, seed=seed)
        tag = {"bertscore": " [vs v1.0 LEADER]", "summac": " [PRIMARY comparator]"}.get(base, "")
        verdict = "distinguishable" if d.excludes_zero else "indistinguishable (CI spans 0)"
        print(
            f"  v1.1 − {base:<12} Δρ={d.delta_rho:+.3f}  95% CI [{d.lo:+.3f}, {d.hi:+.3f}]"
            f"  {verdict}{tag}"
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-refresh", action="store_true", help="reuse cached v1.1 judge scores")
    args = ap.parse_args()
    cfg = load_config()
    seed = cfg["seed"]

    if not BASE_SCORES.exists():
        raise SystemExit("Run pipelines/03c_track_a_dev.py first (needs cached tier-1/2 scores).")
    base = pd.read_csv(BASE_SCORES)

    if args.no_refresh and V11_PATH.exists():
        print(f"Loading cached v1.1 judge from {V11_PATH.relative_to(ROOT)}")
        v11 = pd.read_csv(V11_PATH)
    else:
        v11 = run_v11_judge(cfg)

    df = base.merge(v11, on=["consultation", "model"], how="left")
    df["judge_v11_neg_count"] = -df["judge_v11_count"]  # higher = better (fewer errors)
    if not df["judge_v11_parse_ok"].all():
        print(f"\n⚠ v1.1 judge parse failures: {(~df['judge_v11_parse_ok']).sum()} note(s)")

    print(
        f"\njudge v1.1 count spread: min={df['judge_v11_count'].min()} "
        f"max={df['judge_v11_count'].max()} nunique={df['judge_v11_count'].nunique()} "
        f"(v1.0 judge_score had 2 distinct values)"
    )

    report(df[~df.is_doctor], "MACHINE-ONLY DEV (doctor excluded — primary view)", seed)
    report(df, "ALL DEV notes (machine + doctor — sensitivity)", seed)
    print("\nNOTE: DEV only. This is rubric v1.1 of ≤3 permitted before the one-shot TEST run.")


if __name__ == "__main__":
    main()
