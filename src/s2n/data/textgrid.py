"""Parse Praat .TextGrid transcripts into a speaker-labelled dialogue.

WHY: the gold transcripts are one TextGrid per speaker (``*_doctor.TextGrid``,
``*_patient.TextGrid``). Each is an IntervalTier of ``(xmin, xmax, text)``;
empty intervals are silence. We merge the two speakers by start time into a
single "Doctor: ... / Patient: ..." dialogue — the same logic PriMock57's own
``scripts/textgrid_to_transcript.py`` uses, so the merge is defensible.

Tag policy (decided with supervisor, logged for the viva):
  * ``<UNSURE>word</UNSURE>`` -> KEEP the word, drop only the tags. The
    transcriber's best guess is still real, useful content.
  * ``<UNIN/>`` / ``<INAUDIBLE_SPEECH/>`` -> STRIP entirely (no content).
We also COUNT every tag occurrence (see ``count_tags``) so we can defend the
choice ("inaudible speech is negligible, N=X") and use it as a free
RQ2/limitations data point.

This module reads ONLY transcripts. It never touches notes/ or human_eval/.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from praatio import textgrid as _tg

# Tags that carry NO content -> removed completely.
_EMPTY_TAGS = ("<UNIN/>", "<INAUDIBLE_SPEECH/>")
# Wrapper tags around a transcriber's best guess -> unwrap, keep the inner text.
_UNSURE_OPEN, _UNSURE_CLOSE = "<UNSURE>", "</UNSURE>"

# For frequency logging: every distinct tag we care about.
_ALL_TAGS = (_UNSURE_OPEN, _UNSURE_CLOSE, *_EMPTY_TAGS)


@dataclass
class Utterance:
    """One non-empty interval from a TextGrid, with its speaker and timing."""

    speaker: str
    start: float
    end: float
    text: str  # tag-cleaned text


@dataclass
class TagCounts:
    """How often each transcript tag occurred (for the tag-policy defence)."""

    unsure: int = 0  # <UNSURE>...</UNSURE> spans (guessed words we KEPT)
    inaudible: int = 0  # <UNIN/> + <INAUDIBLE_SPEECH/> markers (content we DROPPED)
    per_tag: dict = field(default_factory=dict)

    def merge(self, other: "TagCounts") -> "TagCounts":
        merged = TagCounts(
            unsure=self.unsure + other.unsure,
            inaudible=self.inaudible + other.inaudible,
        )
        keys = set(self.per_tag) | set(other.per_tag)
        merged.per_tag = {k: self.per_tag.get(k, 0) + other.per_tag.get(k, 0) for k in keys}
        return merged


def count_tags(raw_text: str) -> TagCounts:
    """Count each tag in a raw interval string (before cleaning)."""
    per_tag = {t: raw_text.count(t) for t in _ALL_TAGS}
    return TagCounts(
        unsure=per_tag.get(_UNSURE_OPEN, 0),
        inaudible=per_tag.get("<UNIN/>", 0) + per_tag.get("<INAUDIBLE_SPEECH/>", 0),
        per_tag={t: n for t, n in per_tag.items() if n},
    )


def clean_text(raw_text: str) -> str:
    """Apply the tag policy: keep <UNSURE> content, drop inaudible markers."""
    text = raw_text.replace(_UNSURE_OPEN, "").replace(_UNSURE_CLOSE, "")
    for tag in _EMPTY_TAGS:
        text = text.replace(tag, "")
    return re.sub(r"\s+", " ", text).strip()


def _read_utterances(path: Path, speaker: str) -> tuple[list[Utterance], TagCounts]:
    tg = _tg.openTextgrid(str(path), includeEmptyIntervals=False)
    utterances: list[Utterance] = []
    tags = TagCounts()
    for tier_name in tg.tierNames:
        tier = tg.getTier(tier_name)
        for start, end, label in tier.entries:
            tags = tags.merge(count_tags(label))
            cleaned = clean_text(label)
            if cleaned:  # skip intervals that were pure silence/inaudible
                utterances.append(Utterance(speaker, float(start), float(end), cleaned))
    return utterances, tags


def load_dialogue(
    doctor_path: str | Path, patient_path: str | Path
) -> tuple[list[Utterance], TagCounts]:
    """Merge doctor + patient TextGrids into one time-ordered utterance list.

    Returns (utterances, tag_counts). Utterances are sorted by start time so
    the two channels interleave into a natural back-and-forth dialogue.
    """
    doc_utts, doc_tags = _read_utterances(Path(doctor_path), "Doctor")
    pat_utts, pat_tags = _read_utterances(Path(patient_path), "Patient")
    utterances = sorted(doc_utts + pat_utts, key=lambda u: u.start)
    return utterances, doc_tags.merge(pat_tags)


def dialogue_to_text(utterances: list[Utterance]) -> str:
    """Render utterances as 'Speaker: text' lines — the generator's input."""
    return "\n".join(f"{u.speaker}: {u.text}" for u in utterances)


def textgrid_to_text(doctor_path: str | Path, patient_path: str | Path) -> str:
    """Convenience: two TextGrid paths -> a speaker-labelled dialogue string."""
    utterances, _ = load_dialogue(doctor_path, patient_path)
    return dialogue_to_text(utterances)
