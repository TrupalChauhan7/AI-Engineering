"""RQ1 Track A — FINAL judge config on DEV: Selene-Mini absolute rubric (v1.2).

Config 3/3 of the ≤3 allowed on DEV (pre-registration §B5). After this the judge
is FROZEN — win or lose — and the only remaining step is the one-shot TEST run.
DEV ONLY; TEST is never read here.

Selene-Mini (atla/selene-mini) is a purpose-built evaluator, Llama-3.1-8B family
(different family from the MedGemma generator → self-enhancement rule holds; ~8B
fits the RTX-5060 8GB ceiling). Used in its trained absolute-rubric format: a
calibrated 1–5 faithfulness score → (N−1)/4 ∈ [0,1] (higher = more faithful).

Reuses the cached tier-1/2 scores (BERTScore, SummaC, ROUGE, Leven.) from 03c.
Reports the judge's aligned ρ vs errors_total, plus paired Δρ vs BERTScore (the
real bar — v1.0/v1.1 leader) and vs SummaC (the pre-registered comparator).

Run:  python pipelines/03e_judge_selene_dev.py               # run Selene (slow)
      python pipelines/03e_judge_selene_dev.py --no-refresh  # reuse cached
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
SELENE_PATH = ROOT / "results" / "track_a_dev_judge_selene.csv"
METRIC_COLS = ["bertscore", "summac", "rougeL", "levenshtein", "judge_selene"]


def run_selene(cfg: dict) -> pd.DataFrame:
    t = build_dev_table(cfg)
    print(f"Running Selene-Mini judge (v1.2 rubric) on {len(t)} DEV notes ...")
    judged = FaithfulnessJudge(cfg).judge_many(t["transcript"].tolist(), t["note_text"].tolist())
    t["judge_selene"] = [j["score"] for j in judged]
    t["selene_rubric"] = [j["rubric_result"] for j in judged]
    t["selene_parse_ok"] = [j["parse_ok"] for j in judged]
    out = t[["consultation", "model", "judge_selene", "selene_rubric", "selene_parse_ok"]]
    SELENE_PATH.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(SELENE_PATH, index=False)
    print(f"  cached Selene scores -> {SELENE_PATH.relative_to(ROOT)}")
    return out


def report(df: pd.DataFrame, name: str, seed: int) -> None:
    print(
        f"\n{'=' * 72}\n{name}  (n={len(df)} notes, {df.consultation.nunique()} consultations)"
        f"\n{'=' * 72}"
    )
    tbl = correlation_table(df, METRIC_COLS, TARGET, seed=seed)
    print(f"Aligned Spearman ρ vs {TARGET}  (higher = better flags unreliable notes):")
    print(tbl.to_string(index=False))

    print("\nPaired Δρ (Selene judge − baseline); >0 favours the judge:")
    for base in ["bertscore", "summac", "levenshtein", "rougeL"]:
        d = paired_delta_rho(df, "judge_selene", base, TARGET, seed=seed)
        tag = {"bertscore": " [THE BAR]", "summac": " [PRE-REG comparator]"}.get(base, "")
        verdict = "distinguishable" if d.excludes_zero else "indistinguishable (CI spans 0)"
        print(
            f"  Selene − {base:<11} Δρ={d.delta_rho:+.3f}  95% CI [{d.lo:+.3f}, {d.hi:+.3f}]"
            f"  {verdict}{tag}"
        )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-refresh", action="store_true", help="reuse cached Selene scores")
    args = ap.parse_args()
    cfg = load_config()
    seed = cfg["seed"]
    assert cfg["judge"]["model"] == "atla/selene-mini", "config judge.model must be Selene"

    if not BASE_SCORES.exists():
        raise SystemExit("Run pipelines/03c_track_a_dev.py first (needs cached tier-1/2 scores).")
    base = pd.read_csv(BASE_SCORES)

    if args.no_refresh and SELENE_PATH.exists():
        print(f"Loading cached Selene scores from {SELENE_PATH.relative_to(ROOT)}")
        sel = pd.read_csv(SELENE_PATH)
    else:
        sel = run_selene(cfg)

    df = base.merge(sel, on=["consultation", "model"], how="left")
    if not df["selene_parse_ok"].all():
        print(f"\n⚠ Selene parse failures: {(~df['selene_parse_ok']).sum()} note(s)")
    print(f"\nSelene rubric spread: {df['selene_rubric'].value_counts().sort_index().to_dict()}")

    report(df[~df.is_doctor], "MACHINE-ONLY DEV (doctor excluded — primary view)", seed)
    report(df, "ALL DEV notes (machine + doctor — sensitivity)", seed)
    print("\nJUDGE FROZEN after this run (config 3/3). Next step is the one-shot TEST only.")


if __name__ == "__main__":
    main()
