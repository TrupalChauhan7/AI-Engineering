"""Canonical consultation IDs.

WHY: PriMock57 is internally inconsistent about zero-padding. The transcripts
and notes use a 2-digit index (``day1_consultation10``), but human_eval's
results.csv writes ``"0" + number`` (``day1_consultation010`` for n>=10, while
n<10 happens to coincide as ``day1_consultation01``). Joining the answer key to
the transcripts/notes on the raw string SILENTLY drops the 11 consultations
numbered >=10 (19% of the data). We normalise everything to ONE canonical form.

Canonical form = ``day{D}_consultation{NN}`` with NN zero-padded to 2 digits,
matching the transcripts/notes (the natural form we split on).
"""

from __future__ import annotations

import re

_ID_RE = re.compile(r"day(\d+)_consultation0*(\d+)", re.IGNORECASE)


def canonical_consultation_id(raw: str) -> str:
    """Normalise any PriMock57 consultation ID to ``day{D}_consultation{NN}``."""
    m = _ID_RE.fullmatch(str(raw).strip())
    if not m:
        raise ValueError(f"Unrecognised consultation id: {raw!r}")
    day, num = int(m.group(1)), int(m.group(2))
    return f"day{day}_consultation{num:02d}"
