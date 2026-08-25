"""RQ2 paired comparison on DEV — pre-registration §C + Amendment 3. DEV ONLY.

BUILT, PENDING DESIGN REVIEW — do not run until the harness design is approved.

Per DEV consultation (paired, within-consultation):
  arm A (clean): gold dialogue  -> NoteGenerator -> note_clean
  arm B (asr):   ASR transcript -> NoteGenerator -> note_asr
Same generator (MedGemma-4B), same prompt (v1.0), same decoding (temp 0,
reasoning off, num_ctx 8192) on both arms — ONLY the transcript differs, so any
systematic difference is attributable to ASR.

Scoring both arms, two scorers:
  * judge_selene — the FROZEN v1.2 judge (pre-registered §C scorer: "same judge")
  * bertscore    — note vs gold clinician note (SCORING-side use of the answer
    key; legitimate — evaluation path only). Reported as corroborating signal:
    RQ1 DEV showed BERTScore is empirically the most valid faithfulness proxy,
    and Selene's 1-5 rubric gives coarse deltas (steps of 0.25). Divergence
    between the two scorers is itself reported.

Stats (s2n.evaluation.rq2): per-consultation delta = score(clean) - score(asr);
bootstrap mean delta over consultations, 95% CI (null-friendly); dose-response
Spearman vs content-WER (PRIMARY, Amendment 3) and verbatim-WER (secondary).

Artifacts: results/rq2_dev/notes/{cid}.{clean|asr}.md, results/rq2_dev_scores.csv.
Leak-safety: the GENERATOR sees only transcripts (gold or ASR). Gold notes are
touched only by the scoring step. TEST is never read.

Run:  python pipelines/05_rq2_paired_dev.py             # generate + score + stats
      python pipelines/05_rq2_paired_dev.py --no-refresh  # reuse cached notes/scores
"""

from __future__ import annotations

import argparse

import pandas as pd

from s2n.config import ROOT, load_config
from s2n.data.evaluation_data import EvaluationLoader
from s2n.data.generation_data import GenerationLoader
from s2n.data.splits import load_or_create_split
from s2n.evaluation.judge import FaithfulnessJudge
from s2n.evaluation.metrics import bertscore
from s2n.evaluation.rq2 import dose_response, paired_bootstrap_delta
from s2n.generation.generator import NoteGenerator

ASR_DIR = ROOT / "results" / "asr_dev"
NOTES_DIR = ROOT / "results" / "rq2_dev" / "notes"
SCORES_CSV = ROOT / "results" / "rq2_dev_scores.csv"
WER_CSV = ROOT / "results" / "wer_dev.csv"


def generate_notes(cfg: dict, dev_ids: list[str]) -> pd.DataFrame:
    """Both arms per consultation, same generator config; cached to disk."""
    gen_loader = GenerationLoader.from_config(cfg)
    generator = NoteGenerator(cfg)
    NOTES_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for cid in dev_ids:
        transcripts = {
            "clean": gen_loader.load(cid).dialogue,
            "asr": (ASR_DIR / f"{cid}.txt").read_text(),
        }
        row = {"consultation": cid}
        for arm, transcript in transcripts.items():
            cache = NOTES_DIR / f"{cid}.{arm}.md"
            if cache.exists():
                note = cache.read_text()
            else:
                print(f"  {cid} [{arm}]: generating ...")
                note = generator.generate(transcript, cid).note
                cache.write_text(note)
            row[f"note_{arm}"] = note
        rows.append(row)
    return pd.DataFrame(rows)


def score_notes(cfg: dict, notes: pd.DataFrame) -> pd.DataFrame:
    """Score both arms with the frozen judge + BERTScore-vs-gold-note."""
    judge = FaithfulnessJudge(cfg)  # frozen Selene v1.2 from config
    gen_loader = GenerationLoader.from_config(cfg)
    ev = EvaluationLoader.from_config(cfg)  # scoring side only

    for arm in ("clean", "asr"):
        print(f"  judging arm: {arm} ...")
        # Judge is source-grounded on the GOLD dialogue for BOTH arms: the
        # question is whether the ASR-derived note is faithful to what was
        # actually said, not to the corrupted transcript it was made from.
        judged = [
            judge.judge(gen_loader.load(cid).dialogue, note)
            for cid, note in zip(notes["consultation"], notes[f"note_{arm}"])
        ]
        notes[f"judge_{arm}"] = [j["score"] for j in judged]
        notes[f"judge_parse_ok_{arm}"] = [j["parse_ok"] for j in judged]

        refs = [ev.load_note(cid)["note"] for cid in notes["consultation"]]
        notes[f"bertscore_{arm}"] = bertscore(notes[f"note_{arm}"].tolist(), refs)

    for scorer in ("judge", "bertscore"):
        notes[f"delta_{scorer}"] = notes[f"{scorer}_clean"] - notes[f"{scorer}_asr"]
    return notes


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-refresh", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    seed = cfg["seed"]
    dev_ids = load_or_create_split(cfg).dev

    missing = [c for c in dev_ids if not (ASR_DIR / f"{c}.txt").exists()]
    if missing:
        raise SystemExit(f"ASR transcripts missing (run 04 first): {missing}")

    if args.no_refresh and SCORES_CSV.exists():
        df = pd.read_csv(SCORES_CSV)
    else:
        print("Generating paired notes (clean vs ASR) ...")
        df = generate_notes(cfg, dev_ids)
        print("Scoring both arms ...")
        df = score_notes(cfg, df)
        df.drop(columns=[c for c in df.columns if c.startswith("note_")]).to_csv(
            SCORES_CSV, index=False
        )

    wer = pd.read_csv(WER_CSV)[["consultation", "wer_primock", "wer_whisper"]]
    df = df.merge(wer, on="consultation", how="left")

    print(
        f"\n=== RQ2 paired DEV (n={len(df)}; generator={cfg['llm']['model']}, "
        f"prompt {cfg['generation']['prompt_version']}, judge frozen v1.2) ==="
    )
    print(
        df[
            [
                "consultation",
                "judge_clean",
                "judge_asr",
                "delta_judge",
                "bertscore_clean",
                "bertscore_asr",
                "delta_bertscore",
                "wer_whisper",
            ]
        ]
        .round(3)
        .to_string(index=False)
    )

    print(
        "\nPaired mean Δ (clean − ASR); >0 => ASR degraded the note "
        "(CI containing 0 is a finding, §C):"
    )
    for scorer in ("judge", "bertscore"):
        d = paired_bootstrap_delta(df[f"delta_{scorer}"].to_numpy(), scorer, seed=seed)
        tag = "pre-registered scorer" if scorer == "judge" else "corroborating"
        verdict = "degradation detected" if d.excludes_zero else "CI contains 0"
        print(
            f"  {scorer:<10} Δ={d.mean_delta:+.4f}  95% CI [{d.lo:+.4f}, {d.hi:+.4f}]"
            f"  {verdict}  [{tag}]"
        )

    print("\nDose-response Spearman(WER, Δ)  [PRIMARY dose = content-WER]:")
    for scorer in ("judge", "bertscore"):
        for dose, col in [("content", "wer_whisper"), ("verbatim", "wer_primock")]:
            r = dose_response(df[col].to_numpy(), df[f"delta_{scorer}"].to_numpy())
            print(f"  {scorer:<10} vs {dose:<8} ρ={r['rho']:+.3f}  (p={r['p']:.3f}, n={r['n']})")

    print(
        "\nDEV only; TEST sealed. Selene Δ granularity is 0.25 (1-5 rubric) — "
        "expect coarse deltas at n=10."
    )


if __name__ == "__main__":
    main()
