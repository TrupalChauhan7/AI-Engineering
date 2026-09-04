"""The pipeline as ONE callable — audio/transcript -> note -> reliability, as plain JSON.

WHY this module exists: the demo needs a single entry point, and the demo must
not dictate the architecture. ``run_pipeline`` is a plain function returning a
plain dict, so a FastAPI route can ``return run_pipeline(...)`` directly and a
frontend can render it without the backend knowing a UI exists. No web, UI, or
serialisation dependency is imported here — deliberately.

TWO SHAPES, ONE IMPLEMENTATION. ``run_pipeline_staged`` yields an event per
stage so a streaming transport (SSE) can animate each as it lands;
``run_pipeline`` drains that same generator and returns the assembled result.
The scoring logic exists exactly once. The staged form yields plain dicts and
knows nothing about SSE — the transport formats them.

LEAK SAFETY (project rule 1). This module imports the transcription,
generation, and verification paths ONLY. It does not — and must not — import
``s2n.data.evaluation_data`` or any loader that can reach ``primock57/notes/``
or ``primock57/human_eval_data/``. Those are the answer key. The reliability's whole
claim is that it judges a note using nothing but the transcript that produced
it, so the structural absence of an import is the guarantee; a test asserts it.

WHAT THE RELIABILITY FLAG IS. The claim verifier scores two axes against the transcript:
  * incorrectness — note claims the transcript does not support
  * omission      — transcript facts the note dropped
Their sum is the COMBINED error count, the signal that ranked first on the
held-out TEST split. ``s2n.reliability.flagger.verdict`` turns that count into a
traffic light using the DEV-calibrated bands in config.

HONEST SCOPE: those bands are a prototype (see the config comment). On TEST the
verifier reached only ~0.48x the human inter-rater ceiling, so this flags notes
worth a second look — it does not certify one as safe.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path

from s2n.config import load_config
from s2n.evaluation.claim_verifier import ClaimVerifier
from s2n.generation.generator import NoteGenerator
from s2n.reliability.flagger import verdict
from s2n.transcription.whisper_asr import WhisperTranscriber


def run_pipeline_staged(
    audio_path: str | Path | None = None,
    transcript: str | None = None,
    cfg: dict | None = None,
    *,
    transcriber: WhisperTranscriber | None = None,
    generator: NoteGenerator | None = None,
    verifier: ClaimVerifier | None = None,
) -> Iterator[dict]:
    """Run the pipeline, yielding one plain dict per stage boundary.

    Yields, in order::

        {"stage": "transcribe", "status": "running"}
        {"stage": "transcribe", "status": "done", "transcript": ...}
        {"stage": "generate",   "status": "running"}
        {"stage": "generate",   "status": "done", "note": ...}
        {"stage": "reliability",      "status": "running"}
        {"stage": "reliability",      "status": "done", "reliability": {...}}
        {"stage": "done",       "timings": {...}}

    Every payload is JSON-serialisable. A caller that wants the whole result in
    one piece should use ``run_pipeline`` instead.
    """
    if audio_path is None and transcript is None:
        raise ValueError("run_pipeline needs either audio_path or transcript")
    cfg = cfg or load_config()

    # --- 1. audio -> transcript -------------------------------------------
    yield {"stage": "transcribe", "status": "running"}
    t0 = time.perf_counter()
    if transcript is None:
        transcript = (transcriber or WhisperTranscriber(cfg)).transcribe(audio_path)
    transcribe_s = time.perf_counter() - t0
    yield {"stage": "transcribe", "status": "done", "transcript": transcript}

    # --- 2. transcript -> SOAP note ---------------------------------------
    yield {"stage": "generate", "status": "running"}
    t0 = time.perf_counter()
    note = (generator or NoteGenerator(cfg)).generate(transcript).note
    generate_s = time.perf_counter() - t0
    yield {"stage": "generate", "status": "done", "note": note}

    # --- 3. note + transcript -> reliability ------------------------------------
    yield {"stage": "reliability", "status": "running"}
    t0 = time.perf_counter()
    verifier = verifier or ClaimVerifier(cfg)
    # incorrectness axis: which note claims the transcript does not support
    report = verifier.score(transcript, note)
    unsupported = [c for c, v in zip(report.claims, report.verdicts) if v != "supported"]
    # omission axis: which transcript facts never made it into the note
    facts = verifier.decompose_transcript(transcript)
    omissions = verifier.check_omissions(note, facts)
    omitted = [f for f, v in zip(omissions.facts, omissions.verdicts) if v == "absent"]
    reliability_s = time.perf_counter() - t0

    n_unsupported, n_omitted = len(unsupported), len(omitted)
    band = verdict({"n_unsupported": n_unsupported, "n_omitted": n_omitted}, cfg)
    yield {
        "stage": "reliability",
        "status": "done",
        "reliability": {
            "unsupported_claims": unsupported,
            "omitted_facts": omitted,
            "n_unsupported": n_unsupported,
            "n_omitted": n_omitted,
            "combined": band["combined"],
            "verdict": band["verdict"],
            "score_note": band["score_note"],
        },
    }

    yield {
        "stage": "done",
        "timings": {
            "transcribe_s": round(transcribe_s, 3),
            "generate_s": round(generate_s, 3),
            "reliability_s": round(reliability_s, 3),
        },
    }


def run_pipeline(
    audio_path: str | Path | None = None,
    transcript: str | None = None,
    cfg: dict | None = None,
    *,
    transcriber: WhisperTranscriber | None = None,
    generator: NoteGenerator | None = None,
    verifier: ClaimVerifier | None = None,
) -> dict:
    """Run audio/transcript -> SOAP note -> reliability flag, all in one go.

    Supply ``audio_path`` to transcribe with Whisper, or ``transcript`` to skip
    ASR (the demo pre-caches sample transcripts so it can respond instantly).
    Passing both skips ASR and uses the given transcript.

    The ``transcriber`` / ``generator`` / ``verifier`` keyword arguments exist so
    callers (and tests) can inject already-warm or fake components; by default
    each is built from config.

    Returns a JSON-serialisable dict — only str/int/float/list/dict — so a
    FastAPI route can return it unchanged::

        {"transcript": str,
         "note": str,
         "reliability": {"unsupported_claims": [...], "omitted_facts": [...],
                   "n_unsupported": int, "n_omitted": int, "combined": int,
                   "verdict": "reliable"|"review"|"unreliable",
                   "score_note": str},
         "timings": {"transcribe_s": float, "generate_s": float, "reliability_s": float}}
    """
    out: dict = {}
    for event in run_pipeline_staged(
        audio_path,
        transcript,
        cfg,
        transcriber=transcriber,
        generator=generator,
        verifier=verifier,
    ):
        if event.get("status") != "done" and event["stage"] != "done":
            continue
        # each terminal event carries exactly the key we still need
        for key in ("transcript", "note", "reliability", "timings"):
            if key in event:
                out[key] = event[key]
    return out
