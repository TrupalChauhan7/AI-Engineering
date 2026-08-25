"""EXPLORATORY claim verifier — FINAL (locked) version, one DEV re-run.

The one bounded improvement pass before locking:
  1. OPS FIX: exact-length verdict schemas (minItems==maxItems) compiled into
     the decoding grammar -> misaligned batches impossible; fallback ~never
     fires. Validity-neutral; needed for TEST feasibility (~235 notes).
  2. transcript_facts_v2 (BLIND tightening of the weak omission axis; v1 kept).
  3. SECONDARY criticality column: unsupported claims rated critical/minor in a
     separate pass (locked verify path untouched); n_critical_unsupported is
     reported vs errors_critical ONLY. Primary endpoint unchanged.

After this run the verifier is LOCKED — no further DEV iteration, regardless
of the number (10 clusters cannot validate small gains). Still EXPLORATORY:
frozen RQ1 unchanged, TEST sealed.

Run:  python pipelines/08_claim_verifier_final_dev.py
      python pipelines/08_claim_verifier_final_dev.py --no-refresh

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
FINAL_CSV = ROOT / "results" / "claim_verifier_final_dev.csv"
FACTS_JSON = ROOT / "results" / "transcript_facts_v2_dev.json"
TARGET = "errors_total"
METRIC_COLS = [
    "bertscore",
    "judge_selene",
    "claims_neg_combined",
    "claims_neg_unsupported",
    "claims_neg_omitted",
]


def run_final(cfg: dict) -> pd.DataFrame:
    t = build_dev_table(cfg)
    verifier = ClaimVerifier(cfg)

    # Transcript facts (v2), one decomposition per consultation, cached.
    facts = json.loads(FACTS_JSON.read_text()) if FACTS_JSON.exists() else {}
    for _, row in t.drop_duplicates("consultation").iterrows():
        if row.consultation not in facts:
            print(f"  facts(v2) {row.consultation} ...")
            facts[row.consultation] = verifier.decompose_transcript(row.transcript)
            FACTS_JSON.parent.mkdir(parents=True, exist_ok=True)
            FACTS_JSON.write_text(json.dumps(facts, indent=1))

    rows = []
    for _, row in t.iterrows():
        print(f"  [{len(rows) + 1}/{len(t)}] {row.consultation}/{row.model} ...")
        rep = verifier.score(row.transcript, row.note_text)
        omit = verifier.check_omissions(row.note_text, facts[row.consultation])
        unsupported_claims = [c for c, v in zip(rep.claims, rep.verdicts) if v != "supported"]
        n_critical = verifier.rate_criticality(row.transcript, unsupported_claims)
        rows.append(
            {
                "consultation": row.consultation,
                "model": row.model,
                "n_claims": rep.n_claims,
                "claims_unsupported": rep.n_unsupported,
                "n_critical_unsupported": n_critical,
                "n_facts": omit.n_facts,
                "n_omitted": omit.n_omitted,
                "verify_fallback": rep.used_fallback,
                "omit_fallback": omit.used_fallback,
                "parse_ok": rep.parse_ok and omit.parse_ok,
            }
        )
    out = pd.DataFrame(rows)
    out.to_csv(FINAL_CSV, index=False)
    print(f"  cached -> {FINAL_CSV.relative_to(ROOT)}")
    return out


def report(df: pd.DataFrame, name: str, seed: int) -> None:
    print(
        f"\n{'=' * 72}\n{name}  (n={len(df)} notes, {df.consultation.nunique()} consultations)"
        f"\n{'=' * 72}"
    )
    tbl = correlation_table(df, METRIC_COLS, TARGET, seed=seed)
    print(f"Aligned Spearman rho vs {TARGET} (higher = better flags unreliable notes):")
    print(tbl.to_string(index=False))

    print("\nPaired delta-rho (COMBINED − baseline); >0 favours the verifier:")
    for base in ["bertscore", "judge_selene"]:
        d = paired_delta_rho(df, "claims_neg_combined", base, TARGET, seed=seed)
        tag = {"bertscore": " [RQ1 leader]", "judge_selene": " [frozen judge]"}[base]
        verdict = "distinguishable" if d.excludes_zero else "indistinguishable (CI spans 0)"
        print(
            f"  combined − {base:<12} Δρ={d.delta_rho:+.3f}  95% CI [{d.lo:+.3f}, {d.hi:+.3f}]"
            f"  {verdict}{tag}"
        )

    print("\nPer-axis diagnostics + SECONDARY criticality column:")
    for col, target in [
        ("claims_neg_unsupported", "incorrect_total"),
        ("claims_neg_omitted", "omission_total"),
        ("neg_critical_unsupported", "errors_critical"),  # secondary, its own axis
    ]:
        r = correlation_table(df, [col], target, seed=seed).iloc[0]
        print(f"  {col:<26} vs {target:<16} rho={r['rho']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}]")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-refresh", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    seed = cfg["seed"]

    if args.no_refresh and FINAL_CSV.exists():
        final = pd.read_csv(FINAL_CSV)
    else:
        print("Running FINAL claim verifier (both axes + criticality) on 50 DEV notes ...")
        final = run_final(cfg)

    base = pd.read_csv(BASE_SCORES)
    selene = pd.read_csv(SELENE_SCORES)[["consultation", "model", "judge_selene"]]
    df = base.merge(selene, on=["consultation", "model"]).merge(final, on=["consultation", "model"])
    ax = EvaluationLoader.from_config(cfg).aggregate_by_note()[
        ["Consultation", "Model", "incorrect_total", "omission_total"]
    ]
    ax.columns = ["consultation", "model", "incorrect_total", "omission_total"]
    df = df.merge(ax, on=["consultation", "model"], how="left")

    df["claims_combined"] = df["claims_unsupported"] + df["n_omitted"]
    df["claims_neg_combined"] = -df["claims_combined"]
    df["claims_neg_unsupported"] = -df["claims_unsupported"]
    df["claims_neg_omitted"] = -df["n_omitted"]
    df["neg_critical_unsupported"] = -df["n_critical_unsupported"]

    print(
        f"\nOPS: verify fallbacks {int(df.verify_fallback.sum())}/50 (was 36/50), "
        f"omission fallbacks {int(df.omit_fallback.sum())}/50 (was 17/50) | "
        f"facts/consultation mean {df.groupby('consultation').n_facts.first().mean():.1f} "
        f"(v1: 13.7) | omitted/note mean {df.n_omitted.mean():.1f} | "
        f"critical unsupported/note mean {df.n_critical_unsupported.mean():.1f}"
    )

    report(df[~df.is_doctor], "MACHINE-ONLY DEV (doctor excluded — primary view)", seed)
    report(df, "ALL DEV notes (machine + doctor — sensitivity)", seed)
    print("\nVERIFIER LOCKED after this run. EXPLORATORY — frozen RQ1 unchanged; TEST sealed.")


if __name__ == "__main__":
    main()
