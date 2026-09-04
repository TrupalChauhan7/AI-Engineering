"""Fact-controlled error injection — EXPLORATORY external-validity probe.

STATUS: NOT part of the pre-registered RQ1/RQ2. This module adds a *new*
pipeline around the LOCKED generator / claim-verifier / reliability; it does not
modify, re-tune, or re-implement any of them.

WHY THIS EXISTS. On PriMock57 the reliability was validated by rank-correlation
against NOISY human labels (rho ~0.26, about half the human ceiling). A
correlation against a noisy target can only ever be a lower bound on detection,
and it cannot say *which* errors were caught. Here we author the errors
ourselves, so the label is EXACT: we know precisely which sentences were planted
and which fact was deleted. That turns "does the score track human judgement?"
into "does the reliability respond to a known, dosed perturbation?".

THE DEFENSIBILITY POINT: NO LLM IN THE LABELLING LOOP. The hallucination
catalogue is hand-written (config/injection_catalogue.yaml) and the absence
check is a plain case-insensitive substring test. The omission target is chosen
by deterministic token-overlap arithmetic. If a model chose or authored the
planted errors, the "exact" label would inherit that model's mistakes and we
would be back to noisy supervision.

WHAT THE ABSENCE CHECK PROVES (stated precisely, because a viva will ask). A
catalogue entry is only planted in a consultation whose transcript does not
contain its keyword. That proves the planted fact is absent from the transcript
*as stated*. A paraphrase collision is possible in principle -- a transcript
saying "BP was 180 over 100" without the words "blood pressure" -- but rare, and
every skip is logged so the filter's behaviour is visible rather than assumed.

CIRCULARITY AVOIDANCE (the omission axis). The reliability's omission axis scores a
note against the fact list that ``ClaimVerifier.decompose_transcript`` extracts
from the transcript. If we picked the fact to DELETE from that same list, a
detection would be close to guaranteed and the number would be near-circular.
So ``inject_omission`` chooses its target INDEPENDENTLY of the verifier: a
salient sentence of the base note, matched back to the transcript by token
overlap only -- arithmetic, no model. The verifier's fact list is used solely to
*annotate* the chosen target afterwards (``in_verifier_facts``), which lets the
pipeline report the fact-list-conditioned rate as a clearly labelled UPPER-BOUND
diagnostic alongside the independent headline rate.

SCOPE OF THE OMISSION TARGET (pre-declared before the full run, after a 3-case
smoke run showed the problem). Candidates are restricted to the SUBJECTIVE and
OBJECTIVE sections -- patient history, symptoms, examination findings -- and
Plan/advice sentences are excluded. The reason is construct validity, not
results: the verifier's omission axis is built to check whether a note COVERS
the clinically important facts the patient stated, so deleting management advice
would test the axis outside what it was designed to measure and a non-detection
would say nothing about the blind spot. Restricting to S/O makes the deletion a
valid probe of that axis. The restriction was also necessary because the
grounding heuristic is biased toward Plan lines: a doctor voices the plan almost
verbatim in one turn, so it scores high, whereas a Subjective paragraph
synthesises across many turns and scores lower.

Selection remains INDEPENDENT of the verifier's fact list -- the restriction is
by note SECTION, which the verifier never sees.

Everything here is deterministic under a seed: same seed and same inputs give
byte-identical notes and logs. ``tests/test_fact_injection.py`` asserts it.
"""

from __future__ import annotations

import re
import zlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from s2n.config import ROOT, load_config

# Section headers of the SOAP note produced by prompts/note_generation/v1.0.
# Matched leniently: the generator may emit "Subjective:", "**Subjective**",
# "## Subjective", or "S:" style headers.
_SECTIONS = ("subjective", "objective", "assessment", "plan")
_HEADER_RE = re.compile(
    r"^\s*(?:[*#>\-\s]*)(" + "|".join(_SECTIONS) + r")\b\s*[:\-]?\s*(?:\*+)?\s*$",
    re.IGNORECASE,
)
# The generator often puts the header and its content on ONE line
# ("**Subjective:** Patient reports chest pressure ..."), so a header-only
# regex is not enough: we need to find the section AND keep its label out of
# the deletable text.
_INLINE_HEADER_RE = re.compile(
    r"^\s*(?:[*#>\-\s]*)(" + "|".join(_SECTIONS) + r")\s*[:\-]\s*(?:\*+)?\s*",
    re.IGNORECASE,
)
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")

# Sections an omission target may be drawn from. See the module docstring:
# the omission axis measures coverage of patient-stated clinical facts, so
# Plan/advice sentences are out of scope for a valid blind-spot test.
TARGET_SECTIONS = ("subjective", "objective")


def _section_of(line: str) -> str | None:
    """Which SOAP section this line starts, header-only or inline, else None."""
    m = _HEADER_RE.match(line) or _INLINE_HEADER_RE.match(line)
    return m.group(1).lower() if m else None


# Deliberately tiny. A big stopword list is a tuning knob, and a tuning knob on
# the labelling path is exactly what this module is trying not to have.
_STOP = {
    "the",
    "a",
    "an",
    "and",
    "or",
    "but",
    "for",
    "with",
    "without",
    "of",
    "to",
    "in",
    "on",
    "at",
    "is",
    "was",
    "are",
    "were",
    "be",
    "been",
    "has",
    "have",
    "had",
    "that",
    "this",
    "it",
    "its",
    "as",
    "by",
    "from",
    "not",
    "no",
    "any",
    "some",
    "he",
    "she",
    "they",
    "his",
    "her",
    "their",
    "patient",
    "patients",
    "reports",
    "report",
    "reported",
    "denies",
    "denied",
    "states",
    "stated",
}


def _tokens(text: str) -> set[str]:
    """Content tokens: lowercase alphanumerics, >=3 chars, minus stopwords."""
    return {t for t in re.findall(r"[a-z0-9]+", text.lower()) if len(t) >= 3 and t not in _STOP}


def consultation_rng(seed: int, consultation_id: str) -> np.random.Generator:
    """A per-consultation generator derived from the run seed.

    WHY not one shared rng: with a shared stream, ``--limit 3`` would give
    consultation 3 a different draw than a full run does, so a smoke run and the
    real run would disagree. Deriving the stream from (seed, id) makes each
    consultation's injection independent of how many others ran. ``crc32`` is
    used rather than ``hash()`` because Python string hashing is salted per
    process and would not reproduce across runs.
    """
    return np.random.default_rng([seed, zlib.crc32(consultation_id.encode())])


# --------------------------------------------------------------------------
# catalogue
# --------------------------------------------------------------------------
def load_catalogue(cfg: dict | None = None) -> list[dict]:
    """Load the hand-written hallucination catalogue named in config."""
    cfg = cfg or load_config()
    path = Path(cfg["synthetic_probe"]["injection_catalogue"])
    if not path.is_absolute():
        path = ROOT / path
    entries = yaml.safe_load(path.read_text())["catalogue"]
    return [{"sentence": e["sentence"], "keyword": e["keyword"]} for e in entries]


def usable_entries(catalogue: list[dict], transcript: str) -> tuple[list[dict], list[dict]]:
    """Split the catalogue into (usable, skipped) for one transcript.

    Usable = the keyword does NOT occur in the transcript, so planting the
    sentence makes it a true hallucination *as stated* for this consultation.
    """
    low = transcript.lower()
    usable = [e for e in catalogue if e["keyword"].lower() not in low]
    skipped = [e for e in catalogue if e["keyword"].lower() in low]
    return usable, skipped


# --------------------------------------------------------------------------
# hallucination injection
# --------------------------------------------------------------------------
def _subjective_body(lines: list[str]) -> tuple[int, int]:
    """Line range [start, end) of the Subjective body, or the whole note."""
    start = None
    for i, line in enumerate(lines):
        name = _section_of(line)
        if name is None:
            continue
        if name == "subjective":
            # +1 either way: after a standalone "Subjective:" header, or after
            # the single line that carries the header AND its content.
            start = i + 1
        elif start is not None:
            return start, i
    if start is None:
        return len(lines), len(lines)  # no header found -> append at the end
    return start, len(lines)


@dataclass
class InjectionLog:
    """Exactly what was done to one note. CONTAINS CLINICAL TEXT -> gitignored."""

    consultation: str = ""
    condition: str = ""
    inserted: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    skipped_keywords: list[str] = field(default_factory=list)
    n_usable: int = 0
    match_score: float | None = None
    n_candidates: int = 0  # grounded S/O sentences that were eligible as targets
    in_verifier_facts: bool | None = None
    note: str = ""


def inject_hallucinations(
    note: str,
    transcript: str,
    k: int,
    rng: np.random.Generator,
    catalogue: list[dict],
) -> tuple[str, InjectionLog]:
    """Insert ``k`` catalogue sentences absent from ``transcript`` into ``note``.

    Doses NEST: called with a freshly-seeded rng per dose, the permutation is
    identical, so the +2H set is the +1H set plus one. That is what makes the
    dose axis paired and monotone BY CONSTRUCTION OF THE INJECTION -- the open
    question this study asks is whether the reliability flag's response rises with it.

    Returns the new note and a log of exactly which sentences were inserted.
    """
    usable, skipped = usable_entries(catalogue, transcript)
    if k > len(usable):
        raise ValueError(
            f"dose {k} exceeds {len(usable)} usable catalogue entries for this "
            "transcript; extend config/injection_catalogue.yaml"
        )
    order = rng.permutation(len(usable))
    chosen = [usable[i]["sentence"] for i in order[:k]]

    lines = note.split("\n")
    start, end = _subjective_body(lines)
    body = [i for i in range(start, end) if lines[i].strip()]
    # Insert as a block at a seeded position among the Subjective body lines.
    at = int(rng.integers(0, len(body) + 1)) if body else start
    at = body[at] if at < len(body) else (body[-1] + 1 if body else start)
    out = lines[:at] + list(chosen) + lines[at:]

    return "\n".join(out), InjectionLog(
        inserted=chosen,
        skipped_keywords=[e["keyword"] for e in skipped],
        n_usable=len(usable),
    )


# --------------------------------------------------------------------------
# omission injection
# --------------------------------------------------------------------------
def _note_units(
    note: str, sections: tuple[str, ...] | None = TARGET_SECTIONS
) -> list[tuple[int, str]]:
    """(line index, sentence) for the note's sentences in ``sections``.

    Sentence-level, not line-level, so deleting the target removes the target
    and nothing else. Over-deletion would drop extra facts and inflate the very
    omission count we are measuring.

    ``sections=None`` returns every sentence; the default keeps only Subjective
    and Objective (see the module docstring on why Plan is out of scope).
    """
    units = []
    current = None
    for i, line in enumerate(note.split("\n")):
        if not line.strip():
            continue
        name = _section_of(line)
        if name is not None:
            current = name  # standalone header, or a header inline with content
        if _HEADER_RE.match(line):
            continue  # header-only line carries no content of its own
        if sections is not None and current not in sections:
            continue
        # Strip an inline "**Subjective:**" label before splitting, so the label
        # is never part of a deletable unit — deleting a fact must not decapitate
        # the section it lived in.
        body = _INLINE_HEADER_RE.sub("", line.strip())
        for s in _SENT_SPLIT_RE.split(body):
            s = s.strip()
            if s:
                units.append((i, s))
    return units


def _grounding(unit: str, transcript_lines: list[list[str]]) -> float:
    """Max fraction of a note sentence's content tokens found in ONE transcript line.

    Matching against single lines rather than the whole transcript is what makes
    this a real grounding test: any short sentence overlaps *something* across a
    2000-word transcript, but only a genuinely transcribed fact overlaps one
    turn.
    """
    toks = _tokens(unit)
    if not toks:
        return 0.0
    return max((len(toks & set(t)) / len(toks) for t in transcript_lines), default=0.0)


def inject_omission(
    note: str,
    transcript: str,
    rng: np.random.Generator,
    threshold: float,
    facts: list[str] | None = None,
    n_salient: int = 5,
    sections: tuple[str, ...] | None = TARGET_SECTIONS,
) -> tuple[str, InjectionLog]:
    """Delete one grounded, salient sentence from ``note``.

    Target selection is INDEPENDENT of the verifier's fact list (see the module
    docstring on circularity): candidates are note sentences from ``sections``
    -- Subjective/Objective by default, so the target is a patient-stated
    clinical fact rather than management advice -- whose content tokens are
    corroborated by a single transcript turn at or above ``threshold``. The
    most-grounded, most-content-bearing few form a salient pool and the seeded
    rng picks one. Restricting by note SECTION keeps selection independent of
    the verifier, which never sees the note's section structure.

    ``facts`` (the verifier's transcript facts) is used ONLY to annotate the
    chosen target with ``in_verifier_facts``, enabling the fact-list-conditioned
    UPPER-BOUND diagnostic. It never influences the choice.
    """
    tlines = [list(_tokens(line)) for line in transcript.split("\n") if line.strip()]
    scored = []
    for idx, unit in _note_units(note, sections):
        toks = _tokens(unit)
        if len(toks) < 3:
            continue
        g = _grounding(unit, tlines)
        if g >= threshold:
            scored.append((g, len(toks), unit, idx))
    if not scored:
        return note, InjectionLog(removed=[], match_score=None, n_candidates=0)

    # Deterministic ordering before any random draw: most grounded, then most
    # content-bearing, then alphabetical as the final tiebreak.
    scored.sort(key=lambda r: (-r[0], -r[1], r[2]))
    pool = scored[:n_salient]
    g, _, target, line_idx = pool[int(rng.integers(0, len(pool)))]

    lines = note.split("\n")
    remainder = lines[line_idx].replace(target, "").strip()
    if _tokens(remainder):
        lines[line_idx] = remainder
    else:
        del lines[line_idx]  # line held nothing but the target

    in_facts = None
    if facts is not None:
        ttoks = _tokens(target)
        in_facts = (
            any(len(ttoks & _tokens(f)) / len(ttoks) >= threshold for f in facts)
            if ttoks
            else False
        )

    return "\n".join(lines), InjectionLog(
        removed=[target],
        match_score=round(float(g), 4),
        n_candidates=len(scored),
        in_verifier_facts=in_facts,
    )
