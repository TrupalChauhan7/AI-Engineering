"""Map flagged claims back to character spans in the note.

WHY this is best-effort and lives in the app layer, not in ``s2n``: the verifier
decomposes a note into *atomic paraphrases* ("Patient reports diarrhoea for
three days"), which rarely appear verbatim in the note text ("3/7 hx of
diarrhea"). There is no exact substring to find. So this scores each note line
by content-word overlap with the claim and returns the best line's span.

That is enough for the UI's purpose — draw the underline on the line the flag
came from — and it is deliberately kept out of the scoring path: nothing here
changes a count, a verdict, or any number in the results. If a claim cannot be
matched confidently the span is simply omitted and the UI shows the flag in the
rail without highlighting.
"""

from __future__ import annotations

import re

# words too common to carry matching signal in a clinical note
STOPWORDS = frozenset(
    """a an and are as at be been by for from had has have he her his in is it its of on
    or patient reports states that the their there they this to was were which who with
    no not any""".split()
)
MIN_OVERLAP = 0.30  # fraction of the claim's content words that must appear
# 0.30 keeps 1-of-3 paraphrase matches ("no blood in the stool" -> "No blood.").
# Being slightly permissive is the right failure mode: a near-miss underlines a
# neighbouring line, whereas a miss drops the highlight entirely.


def _content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", text.lower()) if w not in STOPWORDS and len(w) > 2}


# Sentence boundary: a ., ? or ! followed by whitespace. Deliberately simple —
# the cost of a bad split is an underline one clause too long, not a wrong count.
_SENTENCE_END = re.compile(r"(?<=[.?!])\s+")


def _unit_spans(note: str) -> list[tuple[int, int, str]]:
    """(start, end, text) for each SENTENCE in the note.

    Sentence granularity, not line: a generated SOAP note puts each whole
    section ("**Subjective:** ...") on one long line, so matching per line
    would paint an entire section red for a single flagged claim. Sentences
    keep the highlight on the clause the flag is actually about.
    """
    out, pos = [], 0
    for line in note.splitlines(keepends=True):
        if line.strip():
            offset = 0
            for piece in _SENTENCE_END.split(line):
                if piece.strip():
                    lead = len(piece) - len(piece.lstrip())
                    text = piece.strip()
                    start = pos + offset + lead
                    out.append((start, start + len(text), text))
                offset += len(piece)
                # account for the whitespace the split consumed
                while pos + offset < len(note) and note[pos + offset].isspace():
                    if note[pos + offset] == "\n":
                        break
                    offset += 1
        pos += len(line)
    return out


def locate_claims(note: str, claims: list[str]) -> list[dict]:
    """Best-effort span for each claim: ``{claim, start, end, line}``.

    ``line`` is the 0-based index of the matched sentence. Claims with no
    confident match are returned with ``start == -1`` so the caller can still
    list them without highlighting anything.
    """
    lines = _unit_spans(note)
    line_words = [_content_words(text) for _, _, text in lines]
    results = []
    for claim in claims:
        want = _content_words(claim)
        best_i, best_score = -1, 0.0
        if want:
            for i, have in enumerate(line_words):
                score = len(want & have) / len(want)
                if score > best_score:
                    best_i, best_score = i, score
        if best_i >= 0 and best_score >= MIN_OVERLAP:
            start, end, _ = lines[best_i]
            results.append({"claim": claim, "start": start, "end": end, "line": best_i})
        else:
            results.append({"claim": claim, "start": -1, "end": -1, "line": -1})
    return results
