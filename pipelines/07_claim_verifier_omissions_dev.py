"""EXPLORATORY claim verifier + OMISSION axis on DEV — NOT frozen RQ1. DEV ONLY.

Adds the omission axis to the claim verifier and evaluates the COMBINED score:
  1. decompose each DEV TRANSCRIPT into clinically important atomic facts
     (transcript_facts_v1; cached per consultation — reused across its 5 notes)
  2. for each note, check which facts are ABSENT (omission_verify_v1, batched
     + per-fact fallback) -> n_omitted
  3. combined = claims_unsupported (from pipeline 06's cache) + n_omitted —
     mirroring the human target's structure (errors_total = incorrect + omissions)

Evaluated the same honest way: aligned Spearman rho vs errors_total,
cluster-bootstrapped, machine-only primary; paired delta-rho vs BERTScore and
the frozen Selene judge. Mechanism diagnostics per component:
omitted-only vs omission_total, unsupported-only vs incorrect_total.

Frozen RQ1 unchanged; TEST never read.

Run:  python pipelines/07_claim_verifier_omissions_dev.py
      python pipelines/07_claim_verifier_omissions_dev.py --no-refresh

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
from s2n.data.evaluation_data import EvaluationLoader
from s2n.evaluation.claim_verifier import ClaimVerifier
from s2n.evaluation.correlation import correlation_table, paired_delta_rho
from s2n.evaluation.track_a import build_dev_table

BASE_SCORES = ROOT / "results" / "track_a_dev_scores.csv"
SELENE_SCORES = ROOT / "results" / "track_a_dev_judge_selene.csv"
CLAIMS_CSV = ROOT / "results" / "claim_verifier_dev.csv"  # pipeline 06 cache
FACTS_JSON = ROOT / "results" / "transcript_facts_dev.json"  # per-consultation cache
OMIT_CSV = ROOT / "results" / "claim_omissions_dev.csv"
TARGET = "errors_total"
METRIC_COLS = [
    "bertscore",
    "judge_selene",
    "claims_neg_combined",
    "claims_neg_unsupported",
    "claims_neg_omitted",
]


def get_transcript_facts(cfg: dict, table: pd.DataFrame, verifier: ClaimVerifier) -> dict:
    """Facts per consultation, cached to disk (one decomposition per transcript)."""
    if FACTS_JSON.exists():
        facts = json.loads(FACTS_JSON.read_text())
    else:
        facts = {}
    transcripts = table.drop_duplicates("consultation")[["consultation", "transcript"]]
    for _, row in transcripts.iterrows():
        if row.consultation not in facts:
            print(f"  decomposing transcript {row.consultation} ...")
            facts[row.consultation] = verifier.decompose_transcript(row.transcript)
            FACTS_JSON.parent.mkdir(parents=True, exist_ok=True)
            FACTS_JSON.write_text(json.dumps(facts, indent=1))
    return facts


def run_omissions(cfg: dict) -> pd.DataFrame:
    t = build_dev_table(cfg)
    verifier = ClaimVerifier(cfg)
    facts = get_transcript_facts(cfg, t, verifier)

    rows = []
    for _, row in t.iterrows():
        print(f"  [{len(rows) + 1}/{len(t)}] omissions {row.consultation}/{row.model} ...")
        rep = verifier.check_omissions(row.note_text, facts[row.consultation])
        rows.append(
            {
                "consultation": row.consultation,
                "model": row.model,
                "n_facts": rep.n_facts,
                "n_omitted": rep.n_omitted,
                "omit_used_fallback": rep.used_fallback,
                "omit_parse_ok": rep.parse_ok,
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(OMIT_CSV, index=False)
    print(f"  cached -> {OMIT_CSV.relative_to(ROOT)}")
    return out


def report(df: pd.DataFrame, name: str, seed: int) -> None:
    print(
        f"\n{'=' * 72}\n{name}  (n={len(df)} notes, {df.consultation.nunique()} consultations)"
        f"\n{'=' * 72}"
    )
    tbl = correlation_table(df, METRIC_COLS, TARGET, seed=seed)
    print(f"Aligned Spearman rho vs {TARGET} (higher = better flags unreliable notes):")
    print(tbl.to_string(index=False))

    print("\nPaired delta-rho (COMBINED claim score − baseline); >0 favours the verifier:")
    for base in ["bertscore", "judge_selene"]:
        d = paired_delta_rho(df, "claims_neg_combined", base, TARGET, seed=seed)
        tag = {"bertscore": " [RQ1 leader]", "judge_selene": " [frozen judge]"}[base]
        verdict = "distinguishable" if d.excludes_zero else "indistinguishable (CI spans 0)"
        print(
            f"  combined − {base:<12} Δρ={d.delta_rho:+.3f}  95% CI [{d.lo:+.3f}, {d.hi:+.3f}]"
            f"  {verdict}{tag}"
        )

    print("\nMechanism diagnostics (each component vs the axis it measures):")
    for col, target in [
        ("claims_neg_omitted", "omission_total"),
        ("claims_neg_unsupported", "incorrect_total"),
    ]:
        r = correlation_table(df, [col], target, seed=seed).iloc[0]
        print(f"  {col:<24} vs {target:<15} rho={r['rho']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}]")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-refresh", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    seed = cfg["seed"]

    if not CLAIMS_CSV.exists():
        raise SystemExit("Run pipelines/06_claim_verifier_dev.py first (incorrectness axis).")

    if args.no_refresh and OMIT_CSV.exists():
        omit = pd.read_csv(OMIT_CSV)
    else:
        print("Running omission axis on 50 DEV notes ...")
        omit = run_omissions(cfg)

    base = pd.read_csv(BASE_SCORES)
    selene = pd.read_csv(SELENE_SCORES)[["consultation", "model", "judge_selene"]]
    claims = pd.read_csv(CLAIMS_CSV)
    df = (
        base.merge(selene, on=["consultation", "model"])
        .merge(claims, on=["consultation", "model"])
        .merge(omit, on=["consultation", "model"])
    )
    # Human per-axis targets for the mechanism diagnostics.
    ax = EvaluationLoader.from_config(cfg).aggregate_by_note()[
        ["Consultation", "Model", "incorrect_total", "omission_total"]
    ]
    ax.columns = ["consultation", "model", "incorrect_total", "omission_total"]
    df = df.merge(ax, on=["consultation", "model"], how="left")

    df["claims_combined"] = df["claims_unsupported"] + df["n_omitted"]
    for src, dst in [
        ("claims_combined", "claims_neg_combined"),
        ("claims_unsupported", "claims_neg_unsupported"),
        ("n_omitted", "claims_neg_omitted"),
    ]:
        df[dst] = -df[src]

    print(
        f"\nfacts/consultation: mean {df.groupby('consultation').n_facts.first().mean():.1f} | "
        f"omitted/note: mean {df.n_omitted.mean():.1f} | "
        f"omission batched-verify fallbacks: {int(df.omit_used_fallback.sum())}/50 | "
        f"combined score: mean {df.claims_combined.mean():.1f}"
    )

    report(df[~df.is_doctor], "MACHINE-ONLY DEV (doctor excluded — primary view)", seed)
    report(df, "ALL DEV notes (machine + doctor — sensitivity)", seed)
    print("\nEXPLORATORY metric — frozen RQ1 unchanged; TEST sealed.")


if __name__ == "__main__":
    main()
