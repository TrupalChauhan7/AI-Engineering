"""Track B #3 — base vs fine-tuned generator on PriMock57 DEV. DEV ONLY.

Generates a SOAP note for each of the 10 DEV consultations with (a) the base
MLX MedGemma-4B and (b) the same model + the LoRA adapter — IDENTICAL prompt
(v1.0) and decoding (greedy, temp 0), the ONLY difference being the adapter.
Scores both faithfulness-wise and reports the within-consultation paired delta.

Scorers (same as the RQ1/claim-verifier work):
  * claim verifier COMBINED error count (unsupported + omitted; LOWER = better)
    — reuses the cached transcript facts (transcript_facts_v2_dev.json)
  * BERTScore vs the gold clinician note (HIGHER = better)

Paired delta per consultation, bootstrapped over the 10 (s2n.evaluation.rq2).
A neutral/negative transfer result is expected and fine (snippet->section train
vs full-note eval mismatch) — reported honestly either way.

Train = MTS-Dialog, eval = PriMock57 (disjoint). TEST never read; frozen RQ1
unchanged.

Run:  python pipelines/10_finetune_eval_dev.py --ft-adapter <best_val_ckpt_dir>
      python pipelines/10_finetune_eval_dev.py --ft-adapter ... --no-refresh
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from s2n.config import ROOT, load_config
from s2n.data.evaluation_data import EvaluationLoader
from s2n.data.generation_data import GenerationLoader
from s2n.data.splits import load_or_create_split
from s2n.evaluation.claim_verifier import ClaimVerifier
from s2n.evaluation.metrics import bertscore
from s2n.evaluation.rq2 import paired_bootstrap_delta

MODEL = ROOT / "models" / "medgemma-4b-it-4bit"
NOTES_DIR = ROOT / "results" / "finetune" / "eval_notes"
SCORES_CSV = ROOT / "results" / "finetune" / "eval_dev_scores.csv"
FACTS_JSON = ROOT / "results" / "transcript_facts_v2_dev.json"  # reuse Phase 14c cache


def generate_arm(arm: str, adapter: str | None, dev_ids: list[str], gen_loader) -> dict[str, str]:
    """Generate (or load cached) notes for one arm across DEV consultations."""
    from s2n.finetune.mlx_generate import MLXNoteGenerator

    NOTES_DIR.mkdir(parents=True, exist_ok=True)
    todo = {c: NOTES_DIR / f"{c}.{arm}.md" for c in dev_ids}
    if all(p.exists() for p in todo.values()):
        return {c: p.read_text() for c, p in todo.items()}

    generator = MLXNoteGenerator(str(MODEL), adapter_path=adapter)
    notes = {}
    for cid in dev_ids:
        cache = todo[cid]
        if cache.exists():
            notes[cid] = cache.read_text()
        else:
            print(f"  [{arm}] generating {cid} ...")
            note = generator.generate(gen_loader.load(cid).dialogue)
            cache.write_text(note)
            notes[cid] = note
    return notes


def score_notes(cfg, dev_ids, transcripts, notes_by_arm, gold, facts) -> pd.DataFrame:
    verifier = ClaimVerifier(cfg)
    rows = []
    for cid in dev_ids:
        row = {"consultation": cid}
        for arm, notes in notes_by_arm.items():
            rep = verifier.score(transcripts[cid], notes[cid])
            omit = verifier.check_omissions(notes[cid], facts.get(cid, []))
            row[f"{arm}_unsupported"] = rep.n_unsupported
            row[f"{arm}_omitted"] = omit.n_omitted
            row[f"{arm}_combined"] = rep.n_unsupported + omit.n_omitted
        rows.append(row)
    df = pd.DataFrame(rows)
    # BERTScore per arm (batched) vs the gold clinician note.
    refs = [gold[c] for c in dev_ids]
    for arm, notes in notes_by_arm.items():
        df[f"{arm}_bertscore"] = bertscore([notes[c] for c in dev_ids], refs)
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ft-adapter", required=True, help="best-val LoRA adapter dir")
    ap.add_argument("--no-refresh", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    seed = cfg["seed"]
    dev_ids = load_or_create_split(cfg).dev
    gen_loader = GenerationLoader.from_config(cfg)
    ev = EvaluationLoader.from_config(cfg)

    if args.no_refresh and SCORES_CSV.exists():
        df = pd.read_csv(SCORES_CSV)
    else:
        transcripts = {c: gen_loader.load(c).dialogue for c in dev_ids}
        gold = {c: ev.load_note(c)["note"] for c in dev_ids}
        facts = json.loads(FACTS_JSON.read_text()) if FACTS_JSON.exists() else {}
        print("Generating BASE notes (MLX, no adapter) ...")
        base_notes = generate_arm("base", None, dev_ids, gen_loader)
        print("Generating FINE-TUNED notes (MLX + LoRA adapter) ...")
        ft_notes = generate_arm("ft", args.ft_adapter, dev_ids, gen_loader)
        print("Scoring both arms ...")
        df = score_notes(
            cfg, dev_ids, transcripts, {"base": base_notes, "ft": ft_notes}, gold, facts
        )
        SCORES_CSV.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(SCORES_CSV, index=False)

    print(f"\n=== Track B: base vs fine-tuned on PriMock57 DEV (n={len(df)}) ===")
    print(f"adapter: {args.ft_adapter}")
    show = ["consultation", "base_combined", "ft_combined", "base_bertscore", "ft_bertscore"]
    print(df[show].round(3).to_string(index=False))

    print("\nPaired deltas (bootstrapped over the 10 consultations):")
    # Faithfulness error count: LOWER = better -> improvement = base - ft (>0 good).
    for col in ["combined", "unsupported", "omitted"]:
        d = paired_bootstrap_delta(
            (df[f"base_{col}"] - df[f"ft_{col}"]).to_numpy(), f"{col}_reduction", seed=seed
        )
        verdict = (
            "FT reduces errors"
            if d.excludes_zero and d.mean_delta > 0
            else ("FT worse" if d.excludes_zero else "no difference (CI spans 0)")
        )
        print(
            f"  {col:<11} error reduction (base−ft) Δ={d.mean_delta:+.3f}  "
            f"CI [{d.lo:+.3f}, {d.hi:+.3f}]  {verdict}"
        )
    # BERTScore: HIGHER = better -> improvement = ft - base (>0 good).
    d = paired_bootstrap_delta(
        (df["ft_bertscore"] - df["base_bertscore"]).to_numpy(), "bertscore_gain", seed=seed
    )
    verdict = (
        "FT better"
        if d.excludes_zero and d.mean_delta > 0
        else ("FT worse" if d.excludes_zero else "no difference (CI spans 0)")
    )
    print(
        f"  bertscore   gain (ft−base)         Δ={d.mean_delta:+.4f}  "
        f"CI [{d.lo:+.4f}, {d.hi:+.4f}]  {verdict}"
    )

    print(
        "\nDEV only; TEST sealed; frozen RQ1 unchanged. "
        "Neutral/negative transfer is an expected, valid result (snippet→section "
        "train vs full-note eval)."
    )


if __name__ == "__main__":
    main()
