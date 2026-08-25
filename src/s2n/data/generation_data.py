"""GENERATION-side data access: transcripts ONLY.

WHY (project rule 1 — no data leakage): the note generator must see nothing
but the transcript. This loader is that guarantee made *structural*: it is
constructed with the transcripts path alone and has no attribute, import, or
method that can reach ``notes/`` or ``human_eval/``. The answers are simply not
reachable from this object, so leakage is impossible by construction rather
than by discipline. (See ``tests/test_leak_safety.py`` for the guard.)

Consultation IDs are derived from transcript filenames — a leak-safe source.
The EVALUATION side lives in a separate module: ``s2n.data.evaluation_data``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from s2n.config import ROOT, load_config
from s2n.data.textgrid import TagCounts, dialogue_to_text, load_dialogue


@dataclass
class Consultation:
    """One consultation's transcript, ready for the generator."""

    consultation_id: str  # e.g. "day1_consultation01"
    dialogue: str  # speaker-labelled transcript text
    tag_counts: TagCounts  # <UNSURE>/inaudible frequencies for this consultation


class GenerationLoader:
    """Yields transcripts and NOTHING else.

    Deliberately takes only a transcripts directory — never the config's
    ``notes_gold`` / ``human_eval`` paths — so this class cannot open an answer.
    """

    _DOCTOR_SUFFIX = "_doctor.TextGrid"
    _PATIENT_SUFFIX = "_patient.TextGrid"

    def __init__(self, transcripts_dir: str | Path):
        self.transcripts_dir = Path(transcripts_dir)
        if not self.transcripts_dir.is_dir():
            raise FileNotFoundError(f"Transcripts dir not found: {self.transcripts_dir}")

    @classmethod
    def from_config(cls, cfg: dict | None = None) -> "GenerationLoader":
        """Build from config, passing ONLY the transcripts path (leak-safe)."""
        cfg = cfg or load_config()
        return cls(ROOT / cfg["paths"]["transcripts_gold"])

    def consultation_ids(self) -> list[str]:
        """All consultation IDs, sorted — derived from doctor TextGrid names."""
        ids = [
            p.name[: -len(self._DOCTOR_SUFFIX)]
            for p in self.transcripts_dir.glob(f"*{self._DOCTOR_SUFFIX}")
        ]
        return sorted(ids)

    def _paths(self, consultation_id: str) -> tuple[Path, Path]:
        doctor = self.transcripts_dir / f"{consultation_id}{self._DOCTOR_SUFFIX}"
        patient = self.transcripts_dir / f"{consultation_id}{self._PATIENT_SUFFIX}"
        for p in (doctor, patient):
            if not p.exists():
                raise FileNotFoundError(f"Missing transcript: {p}")
        return doctor, patient

    def load(self, consultation_id: str) -> Consultation:
        """Load one consultation's merged, tag-cleaned dialogue."""
        doctor, patient = self._paths(consultation_id)
        utterances, tags = load_dialogue(doctor, patient)
        return Consultation(consultation_id, dialogue_to_text(utterances), tags)

    def iter_consultations(self, ids: list[str] | None = None):
        """Iterate consultations (optionally restricted to a split's IDs)."""
        for cid in ids or self.consultation_ids():
            yield self.load(cid)
