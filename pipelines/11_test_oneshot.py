"""ONE-SHOT TEST harness (pre-registration §D) — DEV by default; TEST once.

Single harness that scores a split's pre-rated notes with EVERY frozen metric
(reusing the exact dev-pipeline code paths) and runs the pre-registered RQ1
analyses. Default split = dev, for dry-runs. TEST is scored EXACTLY ONCE and is
protected by a one-shot guard.

Metrics (all reused, no reimplementation):
  * ROUGE-L, Levenshtein, BERTScore   -> s2n.evaluation.metrics
  * SummaC-ZS                         -> s2n.evaluation.summac_metric
  * frozen judge Selene v1.2          -> s2n.evaluation.judge (config: atla/selene-mini)
  * claim verifier LOCKED config      -> s2n.evaluation.claim_verifier
      claim_decompose_v1 + claim_verify_v1 + transcript_facts_v2 +
      omission_verify_v1 + claim_criticality_v1 ; combined = unsupported + omitted
Inference constraints come from config.yaml: num_ctx 8192, temperature 0,
reasoning off (already wired into LLMClient / judge / verifier).

Analyses (pre-registration):
  * §B2/B3 aligned Spearman ρ vs errors_total, 5000× cluster-bootstrap
  * paired Δρ: (combined − BERTScore) HEADLINE; (combined − SummaC);
    (combined − Selene); and (Selene − SummaC) for the frozen-judge test
  * Amendment 2 population: machine-only PRIMARY + doctor-inclusive SENSITIVITY
  * §B6 human ceiling (inter-evaluator agreement) + ρ relative to it
  * §B7 flagging PR-AUC (positive = note has ≥1 critical human error)
  * EXPLORATORY (Amendment 5, pre-TEST): the same PR-AUC with positive =
    worst-quartile errors_total, τ frozen from DEV in config.flagging —
    reported ALONGSIDE the pre-registered §B7, never instead of it

Run:  python pipelines/11_test_oneshot.py                 # dev dry-run (default)
      python pipelines/11_test_oneshot.py --no-refresh    # reuse cached scores
      python pipelines/11_test_oneshot.py --split test    # ONE-SHOT (guarded)
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from s2n.config import ROOT, load_config
from s2n.data.evaluation_data import EvaluationLoader
from s2n.data.splits import load_or_create_split
from s2n.evaluation.ceiling_flagging import inter_evaluator_ceiling, pr_auc_bootstrap
from s2n.evaluation.claim_verifier import ClaimVerifier
from s2n.evaluation.correlation import correlation_table, paired_delta_rho
from s2n.evaluation.judge import FaithfulnessJudge
from s2n.evaluation.metrics import bertscore, levenshtein_similarity, rouge_l
from s2n.evaluation.summac_metric import SummaCZS
from s2n.evaluation.track_a import build_scoring_table

TARGET = "errors_total"
# faithfulness-oriented columns (higher = more faithful) for the correlation table
METRIC_COLS = [
    "bertscore",
    "summac",
    "rougeL",
    "levenshtein",
    "judge_selene",
    "claims_neg_combined",
]
LABELS = {
    "bertscore": "BERTScore",
    "summac": "SummaC",
    "rougeL": "ROUGE-L",
    "levenshtein": "Levenshtein",
    "judge_selene": "Selene judge",
    "claims_neg_combined": "Claim verifier (combined)",
}


def _paths(split: str):
    r = ROOT / "results"
    return (
        r / f"oneshot_{split}_scores.csv",
        r / f"oneshot_{split}_correlation.csv",
        r / f"oneshot_{split}_facts.json",
    )


def compute_scores(cfg: dict, ids: list[str], facts_path) -> pd.DataFrame:
    t = build_scoring_table(ids, cfg)
    notes = t["note_text"].tolist()
    refs = t["reference_note"].tolist()
    srcs = t["transcript"].tolist()

    print(f"Scoring {len(t)} notes ({t.consultation.nunique()} consultations) — all tiers ...")
    t["rougeL"] = [rouge_l(n, r) for n, r in zip(notes, refs)]
    t["levenshtein"] = [levenshtein_similarity(n, r) for n, r in zip(notes, refs)]
    print("  reference metrics done; BERTScore ...")
    t["bertscore"] = bertscore(notes, refs)
    print("  BERTScore done; SummaC-ZS ...")
    t["summac"] = SummaCZS(cfg).score_many(srcs, notes)
    print("  SummaC done; Selene judge (frozen v1.2) ...")
    judged = FaithfulnessJudge(cfg).judge_many(srcs, notes)
    t["judge_selene"] = [j["score"] for j in judged]

    print("  Judge done; claim verifier (locked) ...")
    verifier = ClaimVerifier(cfg)
    facts = json.loads(facts_path.read_text()) if facts_path.exists() else {}
    for cid in t.consultation.unique():
        if cid not in facts:
            print(f"    facts(v2) {cid} ...")
            src = t.loc[t.consultation == cid, "transcript"].iloc[0]
            facts[cid] = verifier.decompose_transcript(src)
            facts_path.parent.mkdir(parents=True, exist_ok=True)
            facts_path.write_text(json.dumps(facts, indent=1))

    n_unsup, n_omit, n_crit, n_claims, n_facts = [], [], [], [], []
    for i, row in t.iterrows():
        print(f"  [{len(n_unsup) + 1}/{len(t)}] verify {row.consultation}/{row.model} ...")
        rep = verifier.score(row.transcript, row.note_text)
        omit = verifier.check_omissions(row.note_text, facts[row.consultation])
        unsupported = [c for c, v in zip(rep.claims, rep.verdicts) if v != "supported"]
        n_unsup.append(rep.n_unsupported)
        n_omit.append(omit.n_omitted)
        n_crit.append(verifier.rate_criticality(row.transcript, unsupported))
        n_claims.append(rep.n_claims)
        n_facts.append(omit.n_facts)
    t["claims_unsupported"] = n_unsup
    t["n_omitted"] = n_omit
    t["n_critical_unsupported"] = n_crit
    t["n_claims"] = n_claims
    t["n_facts"] = n_facts
    return t


def add_orientation(df: pd.DataFrame) -> pd.DataFrame:
    df["claims_combined"] = df["claims_unsupported"] + df["n_omitted"]
    df["claims_neg_combined"] = -df["claims_combined"]
    # is_critical (§B7 positive class): note has >=1 critical human error.
    df["is_critical"] = (df["errors_critical"] >= 1.0).astype(int)
    return df


def report(df: pd.DataFrame, name: str, seed: int) -> pd.DataFrame:
    print(
        f"\n{'=' * 74}\n{name}  (n={len(df)} notes, {df.consultation.nunique()} consultations)"
        f"\n{'=' * 74}"
    )
    tbl = correlation_table(df, METRIC_COLS, TARGET, seed=seed)
    tbl["label"] = tbl["metric"].map(LABELS)
    print(f"Aligned Spearman ρ vs {TARGET} (higher = better flags unreliable notes):")
    print(tbl[["label", "rho", "lo", "hi"]].to_string(index=False))

    print("\nPaired Δρ, cluster-bootstrapped; >0 favours the FIRST term:")
    pairs = [
        ("claims_neg_combined", "bertscore", "HEADLINE"),
        ("claims_neg_combined", "summac", ""),
        ("claims_neg_combined", "judge_selene", ""),
        ("judge_selene", "summac", "frozen-judge test"),
    ]
    for a, b, tag in pairs:
        d = paired_delta_rho(df, a, b, TARGET, seed=seed)
        verdict = "distinguishable" if d.excludes_zero else "indistinguishable (CI spans 0)"
        t = f"  [{tag}]" if tag else ""
        print(
            f"  {LABELS[a]} − {LABELS[b]:<25} Δρ={d.delta_rho:+.3f}  "
            f"CI [{d.lo:+.3f}, {d.hi:+.3f}]  {verdict}{t}"
        )
    return tbl


def report_ceiling(df: pd.DataFrame, tbl: pd.DataFrame, human_eval, ids, seed: int) -> None:
    print("\n--- §B6 human ceiling (inter-evaluator agreement, machine-only) ---")
    for tgt in ("errors_total", "errors_critical"):
        c = inter_evaluator_ceiling(human_eval, tgt, ids, machine_only=True)
        print(
            f"  ceiling on {tgt}: mean pairwise Spearman = {c['ceiling']:.3f} "
            f"(n_pairs={c['n_pairs']})"
        )
    ceil = inter_evaluator_ceiling(human_eval, TARGET, ids, machine_only=True)["ceiling"]
    print(f"  metric ρ relative to the {TARGET} ceiling ({ceil:.3f}):")
    for _, r in tbl.iterrows():
        frac = r["rho"] / ceil if ceil and ceil == ceil else float("nan")
        print(f"    {r['label']:<26} ρ={r['rho']:+.3f}  = {frac:.2f}× ceiling")


def report_flagging(df: pd.DataFrame, seed: int) -> None:
    print("\n--- §B7 flagging PR-AUC (positive = note has ≥1 critical human error) ---")
    base = df["is_critical"].mean()
    print(f"  positive base rate = {base:.3f} (n_pos={int(df.is_critical.sum())}/{len(df)})")
    if base > 0.9 or base < 0.1:
        print(
            "  ⚠ DEGENERATE base rate — 'the alarm' PR-AUC is near-trivial on this split; "
            "flag for review (most machine notes have ≥1 critical error)."
        )
    for col in METRIC_COLS:
        df["_unrel"] = -df[col]  # higher = more likely unreliable/positive
        out = pr_auc_bootstrap(df, "_unrel", "is_critical", seed=seed)
        print(
            f"    {LABELS[col]:<26} PR-AUC={out['pr_auc']:.3f}  "
            f"CI [{out['lo']:.3f}, {out['hi']:.3f}]"
        )


def report_flagging_exploratory(df: pd.DataFrame, cfg: dict, seed: int) -> None:
    """EXPLORATORY companion to §B7 (Amendment 5, pre-TEST) — post-hoc, NOT pre-registered.

    §B7's pre-registered positive class (errors_critical >= 1) is saturated on DEV
    machine notes (base rate 1.000 -> PR-AUC undefined). This adds ONE clearly
    labelled alternative class: positive = worst-quartile ``errors_total``.

    The threshold tau was derived on DEV machine notes and frozen as a literal in
    config. It is READ from config here and NEVER recomputed from ``df`` — that is
    precisely what makes it DEV-frozen and keeps the TEST application honest.
    """
    tau = cfg["flagging"]["exploratory_errors_total_threshold"]
    print(
        "\n--- EXPLORATORY flagging (post-hoc; positive = worst-quartile errors_total, "
        f"τ={tau} frozen from DEV) — NOT pre-registered §B7 ---"
    )
    d = df.copy()  # copy: never disturb the frozen §B7 frame
    d["is_worst_quartile"] = (d["errors_total"] >= tau).astype(int)
    base = d["is_worst_quartile"].mean()
    print(f"  positive base rate = {base:.3f} (n_pos={int(d.is_worst_quartile.sum())}/{len(d)})")
    for col in METRIC_COLS:
        d["_unrel"] = -d[col]  # higher = more likely unreliable/positive
        out = pr_auc_bootstrap(d, "_unrel", "is_worst_quartile", seed=seed)
        print(
            f"    {LABELS[col]:<26} PR-AUC={out['pr_auc']:.3f}  "
            f"CI [{out['lo']:.3f}, {out['hi']:.3f}]"
        )


def report_alarm_threshold(df: pd.DataFrame, cfg: dict) -> None:
    """DEV-only: describe the combined distribution + a recommended threshold."""
    c = df["claims_combined"]
    print("\n--- Alarm threshold selection (DEV only) ---")
    print(
        f"  combined error count: min {c.min()}, q1 {c.quantile(.25):.0f}, "
        f"median {c.median():.0f}, q3 {c.quantile(.75):.0f}, max {c.max()}"
    )
    rec = int(round(c.median()))
    lo, hi = cfg["alarm"]["reliable_max"], cfg["alarm"]["unreliable_min"]
    print(f"  config alarm bands: reliable <= {lo} | review | unreliable >= {hi}")
    print(f"  RECOMMENDED (DEV median combined) = {rec}  → set in config before the TEST run")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["dev", "test"], default="dev")
    ap.add_argument("--no-refresh", action="store_true", help="reuse cached scores")
    args = ap.parse_args()
    cfg = load_config()
    seed = cfg["seed"]
    scores_csv, corr_csv, facts_json = _paths(args.split)

    # ---- TASK C: one-shot guard -----------------------------------------
    if args.split == "test" and scores_csv.exists() and not args.no_refresh:
        raise SystemExit(
            f"\n*** ONE-SHOT GUARD: {scores_csv.relative_to(ROOT)} already exists. ***\n"
            "The TEST split has already been scored. Re-scoring would violate the one-shot\n"
            "rule (pre-registration §B5/§D). To re-print from the existing scores use\n"
            "  python pipelines/11_test_oneshot.py --split test --no-refresh\n"
            "If you truly must re-score, delete the file deliberately first.\n"
        )

    ids = getattr(load_or_create_split(cfg), args.split)
    print(f"SPLIT = {args.split.upper()}  ({len(ids)} consultations)")

    if args.no_refresh and scores_csv.exists():
        print(f"Loading cached scores from {scores_csv.relative_to(ROOT)}")
        df = pd.read_csv(scores_csv)
    else:
        df = compute_scores(cfg, ids, facts_json)
        scores_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(scores_csv, index=False)
        print(f"  cached scores -> {scores_csv.relative_to(ROOT)}")

    df = add_orientation(df)
    human_eval = EvaluationLoader.from_config(cfg).load_human_eval()

    machine = df[~df.is_doctor].reset_index(drop=True)
    tbl = report(machine, "MACHINE-ONLY (doctor excluded — PRIMARY, Amendment 2)", seed)
    report_ceiling(machine, tbl, human_eval, ids, seed)
    report_flagging(machine, seed)
    report_flagging_exploratory(machine, cfg, seed)
    report(df, "ALL NOTES (machine + doctor — SENSITIVITY)", seed)
    if args.split == "dev":
        report_alarm_threshold(machine, cfg)

    tbl.to_csv(corr_csv, index=False)
    print(f"\nSaved correlation table -> {corr_csv.relative_to(ROOT)}")
    print(
        "\n>>> CORRECTNESS CHECK (machine-only, expect ≈): "
        f"combined {tbl.loc[tbl.metric=='claims_neg_combined','rho'].iloc[0]:.2f}~0.67 | "
        f"BERTScore {tbl.loc[tbl.metric=='bertscore','rho'].iloc[0]:.2f}~0.56 | "
        f"Selene {tbl.loc[tbl.metric=='judge_selene','rho'].iloc[0]:.2f}~0.03"
    )
    print(f"\n{args.split.upper()} run complete. TEST stays sealed unless --split test is chosen.")


if __name__ == "__main__":
    main()
