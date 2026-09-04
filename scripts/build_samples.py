"""Pre-compute Clarion's sample consultations — run once, locally.

WHY: the demo must feel instant and be reproducible on camera. Running Whisper +
MedGemma + the verifier live takes minutes per consultation, so we run it ONCE
here and cache the result; the UI then replays the staged reveal with scripted
timing. Uploads still run the real pipeline live.

DATA SAFETY: the output JSON contains transcript and note text derived from
PriMock57, so ``app/web/public/samples/`` is GITIGNORED. This script writes
there and nowhere else. Never commit its output.

Leak-safe: uses ``s2n.service.run_pipeline`` only — audio -> transcript -> note
-> verifier. It never touches primock57/notes/ or human_eval_data/.

Usage:
    python scripts/build_samples.py                 # all DEV consultations
    python scripts/build_samples.py --limit 3       # just the first few
    python scripts/build_samples.py --force         # recompute existing
"""

from __future__ import annotations

import argparse
import json
import time
import wave
from pathlib import Path

from app.api.spans import locate_claims
from s2n.config import ROOT, load_config
from s2n.data.splits import load_or_create_split
from s2n.service import run_pipeline

OUT_DIR = ROOT / "app" / "web" / "public" / "samples"
MIXED_DIR = ROOT / "results" / "mixed_audio_dev"


def _duration_label(wav_path: Path) -> str:
    try:
        with wave.open(str(wav_path), "rb") as w:
            seconds = w.getnframes() / float(w.getframerate())
    except (OSError, wave.Error):
        return ""
    return f"{int(seconds // 60)}:{int(seconds % 60):02d}"


def _pretty_label(consultation_id: str) -> str:
    """day1_consultation02 -> 'Day 1 · Consultation 02'."""
    day, _, cons = consultation_id.partition("_")
    return f"{day.replace('day', 'Day ')} · Consultation {cons.replace('consultation', '')}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="only the first N consultations")
    ap.add_argument("--force", action="store_true", help="recompute samples that already exist")
    args = ap.parse_args()

    cfg = load_config()
    ids = load_or_create_split(cfg).dev  # DEV only — TEST audio stays untouched
    if args.limit:
        ids = ids[: args.limit]

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Building {len(ids)} sample(s) -> {OUT_DIR.relative_to(ROOT)}  (gitignored)\n")

    for i, cid in enumerate(ids, 1):
        dest = OUT_DIR / f"{cid}.json"
        if dest.exists() and not args.force:
            print(f"[{i}/{len(ids)}] {cid}: cached, skipping (--force to redo)")
            continue

        audio = MIXED_DIR / f"{cid}.wav"
        if not audio.is_file():
            print(f"[{i}/{len(ids)}] {cid}: NO MIXED AUDIO at {audio.relative_to(ROOT)} — skipping")
            print("           run pipelines/04_transcribe_dev.py first to build the mix")
            continue

        print(f"[{i}/{len(ids)}] {cid}: running pipeline (Whisper -> MedGemma -> verifier) ...")
        t0 = time.perf_counter()
        result = run_pipeline(audio_path=audio, cfg=cfg)
        # Spans let the UI underline flagged text during the verification sweep.
        # The live SSE route computes these too — cache them so the sample path
        # replays the identical animation. Pure post-processing: no score changes.
        result["spans"] = locate_claims(result["note"], result["reliability"]["unsupported_claims"])
        result["id"] = cid
        result["label"] = _pretty_label(cid)
        result["durationLabel"] = _duration_label(audio)
        dest.write_text(json.dumps(result, indent=1))

        reliability = result["reliability"]
        print(
            f"           done in {time.perf_counter() - t0:.0f}s — "
            f"{reliability['verdict'].upper()}: {reliability['n_unsupported']} unsupported, "
            f"{reliability['n_omitted']} omitted (combined {reliability['combined']})"
        )

    n = len(list(OUT_DIR.glob("*.json")))
    print(f"\n{n} sample(s) cached. These are gitignored — never commit them.")


if __name__ == "__main__":
    main()
