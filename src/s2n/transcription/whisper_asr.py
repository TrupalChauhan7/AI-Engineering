"""Transcribe consultation audio with Whisper (local, free) — RQ2's ASR stage.

WHY isolated: RQ2 compares notes generated from ASR transcripts vs the gold
transcript (paired, pre-registration §C). Keeping Whisper here lets us swap
model sizes via config and keeps every ASR error attributable to this stage.

Channel handling (approved decision): PriMock57 ships two mono 16 kHz WAVs per
consultation (doctor + patient, aligned timelines). We MIX them to one mono
track by sample averaging — the same operation as the dataset authors' own
``mix_audio.sh`` (``sox -m``) — because RQ2's locked design is
*mixed → diarizer → Whisper* ("real scribes hear one mixed mic"). Diarization
is a planned later refinement; today we transcribe the mix plainly.

Decoding (Amendment 3): Whisper's STANDARD temperature-fallback schedule.
We first tried strict ``temperature=0`` for determinism; it produced
trailing-silence repetition loops on 4/10 DEV consultations (one pathological:
"Right."x109 -> WER 0.67). The fallback fires only on segments that fail
Whisper's internal quality thresholds, so nondeterminism is confined to
exactly the segments that would otherwise loop. The schedule lives in config
(``transcription.temperature``) and the WER pipeline still scans for loops.

16 kHz in, 16 kHz out: Whisper's native rate, so no resampling anywhere.
"""

from __future__ import annotations

import wave
from pathlib import Path

import numpy as np

from s2n.config import load_config


def _read_mono_wav(path: str | Path) -> tuple[np.ndarray, int]:
    """Read a mono 16-bit WAV into an int16 array + its sample rate."""
    with wave.open(str(path), "rb") as w:
        assert w.getnchannels() == 1, f"{path}: expected mono"
        assert w.getsampwidth() == 2, f"{path}: expected 16-bit"
        rate = w.getframerate()
        data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    return data, rate


def mix_channels(doctor_wav: str | Path, patient_wav: str | Path, out_wav: str | Path) -> Path:
    """Average the doctor + patient tracks into one mono WAV (like `sox -m`).

    Averaging (not summing) keeps headroom — no clipping is possible when both
    inputs are int16. Tracks are zero-padded to equal length (they are recorded
    in parallel, so lengths match to within a frame or two).
    """
    doc, rate_d = _read_mono_wav(doctor_wav)
    pat, rate_p = _read_mono_wav(patient_wav)
    assert rate_d == rate_p, "sample-rate mismatch between channels"
    n = max(len(doc), len(pat))
    doc = np.pad(doc, (0, n - len(doc)))
    pat = np.pad(pat, (0, n - len(pat)))
    mixed = ((doc.astype(np.int32) + pat.astype(np.int32)) // 2).astype(np.int16)

    out_wav = Path(out_wav)
    out_wav.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(out_wav), "wb") as w:
        w.setparams((1, 2, rate_d, 0, "NONE", "no compression"))
        w.writeframes(mixed.tobytes())
    return out_wav


class WhisperTranscriber:
    """Config-driven Whisper wrapper; loads the model once, reuses it."""

    def __init__(self, cfg: dict | None = None):
        tcfg = (cfg or load_config())["transcription"]
        self.model_name = tcfg["model"]
        self.language = tcfg.get("language", "en")
        # Config list -> tuple; default = Whisper's standard fallback schedule.
        self.temperature = tuple(tcfg.get("temperature", [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]))
        self._model = None  # lazy: loading takes seconds + RAM

    @property
    def model(self):
        if self._model is None:
            import whisper  # local import: heavy

            self._model = whisper.load_model(self.model_name)
        return self._model

    def transcribe(self, audio_path: str | Path) -> str:
        """Return the ASR transcript text for one audio file (deterministic)."""
        result = self.model.transcribe(
            str(audio_path),
            language=self.language,
            temperature=self.temperature,  # fallback schedule (Amendment 3)
            condition_on_previous_text=True,
            verbose=None,
        )
        return result["text"].strip()
