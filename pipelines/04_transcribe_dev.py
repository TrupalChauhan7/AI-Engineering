"""RQ2 stage 1 — transcribe the 10 DEV consultations, measure WER. DEV ONLY.

Per consultation: mix doctor+patient channels (sample average, like the dataset
authors' sox -m) → Whisper (config model, temp 0) → WER vs the gold TextGrid
dialogue (speaker labels stripped).

Reports TWO WER columns (approved decision 2):
  * wer_primock — PriMock57's own normalization (PRIMARY)
  * wer_whisper — Whisper's EnglishTextNormalizer (numbers canonicalised);
    the gap between them = the digit-vs-word effect on this medical audio.
Plus a repetition-loop flag per consultation (approved decision 4): a looping
consultation's WER is unreliable and must not be averaged in silently.

Artifacts (results/ is gitignored):
  results/asr_dev/{cid}.txt      cached ASR transcripts (re-runs are free)
  results/wer_dev.csv            per-consultation WER table

TEST is never touched. Diarization = planned refinement (mixed→diarizer→Whisper).

Run:  python pipelines/04_transcribe_dev.py
"""

from __future__ import annotations

import re

import pandas as pd

from s2n.config import ROOT, load_config
from s2n.data.generation_data import GenerationLoader
from s2n.data.splits import load_or_create_split
from s2n.transcription.wer import (
    detect_repetition_loop,
    normalize_primock,
    normalize_whisper,
    word_error_rate,
)
from s2n.transcription.whisper_asr import WhisperTranscriber, mix_channels

ASR_DIR = ROOT / "results" / "asr_dev"
MIX_DIR = ROOT / "results" / "mixed_audio_dev"
WER_CSV = ROOT / "results" / "wer_dev.csv"
_SPEAKER_RE = re.compile(r"^(Doctor|Patient):\s*", re.MULTILINE)


def gold_plain_text(loader: GenerationLoader, cid: str) -> str:
    """Gold dialogue without speaker labels — the WER reference."""
    return _SPEAKER_RE.sub("", loader.load(cid).dialogue)


def main() -> None:
    cfg = load_config()
    dev_ids = load_or_create_split(cfg).dev
    loader = GenerationLoader.from_config(cfg)
    audio_dir = ROOT / cfg["paths"]["audio"]
    asr = WhisperTranscriber(cfg)
    ASR_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    for cid in dev_ids:
        cache = ASR_DIR / f"{cid}.txt"
        if cache.exists():
            hyp_raw = cache.read_text()
            print(f"{cid}: cached")
        else:
            print(f"{cid}: mixing + transcribing ...")
            mixed = mix_channels(
                audio_dir / f"{cid}_doctor.wav",
                audio_dir / f"{cid}_patient.wav",
                MIX_DIR / f"{cid}.wav",
            )
            hyp_raw = asr.transcribe(mixed)
            cache.write_text(hyp_raw)

        ref_raw = gold_plain_text(loader, cid)
        loop = detect_repetition_loop(hyp_raw)
        row = {"consultation": cid, "loop_flag": loop.looping, "loop_ngram": loop.ngram or ""}
        for tag, norm in [("primock", normalize_primock), ("whisper", normalize_whisper)]:
            r = word_error_rate(norm(ref_raw), norm(hyp_raw))
            row[f"wer_{tag}"] = round(r["wer"], 4)
            if tag == "primock":
                row.update(
                    subs=r["substitutions"],
                    dels=r["deletions"],
                    ins=r["insertions"],
                    n_ref_words=r["n_ref_words"],
                )
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(WER_CSV, index=False)

    print(
        f"\n=== DEV WER (n={len(df)}; model={cfg['transcription']['model']}, "
        f"mixed audio, no diarization) ==="
    )
    cols = [
        "consultation",
        "wer_primock",
        "wer_whisper",
        "subs",
        "dels",
        "ins",
        "n_ref_words",
        "loop_flag",
    ]
    print(df[cols].to_string(index=False))
    clean = df[~df.loop_flag]
    print(
        f"\nmean WER (primock norm):  {df.wer_primock.mean():.3f}"
        f"   [loop-free only: {clean.wer_primock.mean():.3f}, n={len(clean)}]"
    )
    print(
        f"mean WER (whisper norm):  {df.wer_whisper.mean():.3f}"
        f"   digit/word effect: {df.wer_primock.mean() - df.wer_whisper.mean():+.3f}"
    )
    n_loops = int(df.loop_flag.sum())
    if n_loops:
        print(
            f"\n⚠ {n_loops} consultation(s) show repetition loops — their WER is "
            f"unreliable: {', '.join(df[df.loop_flag].consultation)}"
        )
        if n_loops > 1:
            print("  >1 loop => reconsider the temperature-fallback decision (see log).")
    print(f"\nSaved -> {WER_CSV.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
