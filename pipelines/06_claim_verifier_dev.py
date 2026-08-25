"""EXPLORATORY claim-level verifier on DEV — NOT part of frozen RQ1. DEV ONLY.

Scores the 50 pre-rated DEV notes with the decompose->verify->count metric and
evaluates it the same honest way as every other metric: aligned Spearman rho vs
errors_total, cluster-bootstrapped, machine-only primary + all-notes
sensitivity, paired delta-rho vs BERTScore (the RQ1 leader) and vs the frozen
Selene judge. Also reported:
  * claimnli_unsupported — the SummaC NLI backbone applied per claim (does
    decomposition rescue NLI, which failed on raw shorthand?)
  * a DIAGNOSTIC correlation vs incorrect_total — this metric measures the
    incorrectness axis only (omissions deferred), so errors_total (which
    includes omissions) caps its achievable rho; the diagnostic shows the
    mechanism cleanly. The PRIMARY comparison stays vs errors_total — same
    target as every other metric, no special pleading.

Frozen RQ1 is untouched; TEST is never read.

Run:  python pipelines/06_claim_verifier_dev.py               # slow (LLM)
      python pipelines/06_claim_verifier_dev.py --no-refresh  # reuse cache

NOTE (added later): this script ran BEFORE Amendment 4. Its "EXPLORATORY"
wording was accurate at the time and is left intact as the provenance trail —
the verifier was subsequently LOCKED and pre-registered as the RQ1 TEST
endpoint (Amendment 4, 19 Jul 2026), before the one-shot TEST run.
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from s2n.config import ROOT, load_config
from s2n.evaluation.claim_verifier import ClaimNLI, ClaimVerifier
from s2n.evaluation.correlation import correlation_table, paired_delta_rho
from s2n.evaluation.track_a import build_dev_table

BASE_SCORES = ROOT / "results" / "track_a_dev_scores.csv"  # tier 1/2 + v1.0 judge
SELENE_SCORES = ROOT / "results" / "track_a_dev_judge_selene.csv"
CLAIMS_CSV = ROOT / "results" / "claim_verifier_dev.csv"
CLAIMS_DETAIL = ROOT / "results" / "claim_verifier_dev_detail.jsonl"
TARGET = "errors_total"
METRIC_COLS = [
    "bertscore",
    "judge_selene",
    "summac",
    "claims_neg_unsupported",
    "claimnli_neg_unsupported",
]


def run_verifier(cfg: dict) -> pd.DataFrame:
    t = build_dev_table(cfg)
    verifier = ClaimVerifier(cfg)
    nli = ClaimNLI(cfg)

    rows, details = [], []
    for i, row in t.iterrows():
        print(f"  [{len(rows) + 1}/{len(t)}] {row.consultation}/{row.model} ...")
        rep = verifier.score(row.transcript, row.note_text)
        nli_unsupported = nli.n_unsupported(row.transcript, rep.claims)
        rows.append(
            {
                "consultation": row.consultation,
                "model": row.model,
                "n_claims": rep.n_claims,
                "n_supported": rep.count("supported"),
                "n_contradicted": rep.count("contradicted"),
                "n_not_mentioned": rep.count("not_mentioned"),
                "claims_unsupported": rep.n_unsupported,
                "claimnli_unsupported": nli_unsupported,
                "used_fallback": rep.used_fallback,
                "parse_ok": rep.parse_ok,
            }
        )
        details.append(
            {
                "consultation": row.consultation,
                "model": row.model,
                "claims": rep.claims,
                "verdicts": rep.verdicts,
            }
        )

    out = pd.DataFrame(rows)
    CLAIMS_CSV.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(CLAIMS_CSV, index=False)
    with open(CLAIMS_DETAIL, "w") as f:
        for d in details:
            f.write(json.dumps(d) + "\n")
    print(f"  cached -> {CLAIMS_CSV.relative_to(ROOT)} (+ per-claim detail jsonl)")
    return out


def report(df: pd.DataFrame, name: str, seed: int) -> None:
    print(
        f"\n{'=' * 72}\n{name}  (n={len(df)} notes, {df.consultation.nunique()} consultations)"
        f"\n{'=' * 72}"
    )
    tbl = correlation_table(df, METRIC_COLS, TARGET, seed=seed)
    print(f"Aligned Spearman rho vs {TARGET} (higher = better flags unreliable notes):")
    print(tbl.to_string(index=False))

    print("\nPaired delta-rho (claim verifier − baseline); >0 favours the verifier:")
    for base in ["bertscore", "judge_selene", "summac"]:
        d = paired_delta_rho(df, "claims_neg_unsupported", base, TARGET, seed=seed)
        tag = {"bertscore": " [RQ1 leader]", "judge_selene": " [frozen judge]"}.get(base, "")
        verdict = "distinguishable" if d.excludes_zero else "indistinguishable (CI spans 0)"
        print(
            f"  claims − {base:<13} Δρ={d.delta_rho:+.3f}  95% CI [{d.lo:+.3f}, {d.hi:+.3f}]"
            f"  {verdict}{tag}"
        )

    # Diagnostic: the axis the metric actually measures (incorrectness only).
    diag = correlation_table(df, ["claims_neg_unsupported"], "incorrect_total", seed=seed)
    r = diag.iloc[0]
    print(
        f"\nDIAGNOSTIC vs incorrect_total (the measured axis; omissions deferred): "
        f"rho={r['rho']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}]"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-refresh", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    seed = cfg["seed"]

    if args.no_refresh and CLAIMS_CSV.exists():
        claims = pd.read_csv(CLAIMS_CSV)
    else:
        print("Running claim verifier on 50 DEV notes (decompose + verify) ...")
        claims = run_verifier(cfg)

    base = pd.read_csv(BASE_SCORES)
    selene = pd.read_csv(SELENE_SCORES)[["consultation", "model", "judge_selene"]]
    df = base.merge(selene, on=["consultation", "model"]).merge(
        claims, on=["consultation", "model"]
    )
    # incorrect_total (diagnostic target) isn't in the cached scores CSV — pull
    # it from the answer key (evaluation side; this is scoring, not generation).
    from s2n.data.evaluation_data import EvaluationLoader

    inc = EvaluationLoader.from_config(cfg).aggregate_by_note()[
        ["Consultation", "Model", "incorrect_total"]
    ]
    inc.columns = ["consultation", "model", "incorrect_total"]
    df = df.merge(inc, on=["consultation", "model"], how="left")
    # Aligned orientation: higher = more faithful (like every other metric).
    df["claims_neg_unsupported"] = -df["claims_unsupported"]
    df["claimnli_neg_unsupported"] = -df["claimnli_unsupported"]

    n_fallback = int(df["used_fallback"].sum())
    n_parse_fail = int((~df["parse_ok"]).sum())
    print(
        f"\nclaims/note: mean {df.n_claims.mean():.1f} (min {df.n_claims.min()}, "
        f"max {df.n_claims.max()}) | unsupported: mean {df.claims_unsupported.mean():.1f} "
        f"| batched-verify fallbacks: {n_fallback}/50 | decompose failures: {n_parse_fail}"
    )

    report(df[~df.is_doctor], "MACHINE-ONLY DEV (doctor excluded — primary view)", seed)
    report(df, "ALL DEV notes (machine + doctor — sensitivity)", seed)
    print("\nEXPLORATORY metric — frozen RQ1 unchanged; TEST sealed.")


if __name__ == "__main__":
    main()
