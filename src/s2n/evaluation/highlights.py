"""Key-fact coverage using the clinician 'highlights' from each note JSON.

WHY: 'highlights' are the facts the clinician deemed important — a
ready-made checklist for detecting OMISSIONS objectively.

TODO:
  - highlight_coverage(generated_note, highlights) -> float (0..1)
"""

from __future__ import annotations


def highlight_coverage(generated_note: str, highlights: list[str]) -> float:
    raise NotImplementedError("TODO: check which highlights appear in the note")
