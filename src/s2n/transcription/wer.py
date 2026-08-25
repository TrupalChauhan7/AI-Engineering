"""Word Error Rate — in-house, transparent, dual-normalized.

WER = (substitutions + deletions + insertions) / reference_words, computed by
the classic word-level Levenshtein DP. Implemented in-house (≈30 lines) rather
than adding a dependency to the shared env (approved decision 3; validated in
tests against a hand-worked example).

TWO normalizations, reported side by side (approved decision 2):
  * PRIMARY  — PriMock57's own convention (their scripts/utils.preprocess_text):
    lowercase, dashes→spaces, keep only letters+apostrophes.
  * SECONDARY — Whisper's EnglishTextNormalizer (ships with openai-whisper, no
    new dependency): additionally canonicalises numbers on BOTH sides
    ("nineteen"→"19"), so digit-vs-word mismatches cancel. Medical audio is
    number-heavy; the gap between the two columns measures that effect.

Also: a repetition-loop detector (approved decision 4) — with temperature
pinned to 0, Whisper can loop on garbled audio. A looping consultation's WER is
unreliable and must be flagged, not averaged in silently.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

_KEEP_RE = re.compile(r"[^a-z ']+")


def normalize_primock(text: str) -> str:
    """PriMock57's own ASR-eval normalization (scripts/utils.preprocess_text)."""
    text = text.lower().replace("-", " ")
    text = _KEEP_RE.sub("", text)
    return re.sub(r"\s+", " ", text).strip()


def normalize_whisper(text: str) -> str:
    """Whisper's EnglishTextNormalizer (numbers canonicalised on both sides)."""
    from whisper.normalizers import EnglishTextNormalizer  # local: heavy-ish

    return EnglishTextNormalizer()(text)


def word_error_rate(reference: str, hypothesis: str) -> dict:
    """Word-level Levenshtein WER with edit-op counts.

    Returns {wer, substitutions, deletions, insertions, n_ref_words}.
    """
    ref = reference.split()
    hyp = hypothesis.split()
    R, H = len(ref), len(hyp)
    if R == 0:
        return {
            "wer": float("nan"),
            "substitutions": 0,
            "deletions": 0,
            "insertions": H,
            "n_ref_words": 0,
        }

    # dp[i][j] = min edits to turn ref[:i] into hyp[:j]
    dp = np.zeros((R + 1, H + 1), dtype=np.int32)
    dp[:, 0] = np.arange(R + 1)  # delete all
    dp[0, :] = np.arange(H + 1)  # insert all
    for i in range(1, R + 1):
        for j in range(1, H + 1):
            cost = 0 if ref[i - 1] == hyp[j - 1] else 1
            dp[i, j] = min(
                dp[i - 1, j - 1] + cost,  # match/substitute
                dp[i - 1, j] + 1,  # delete ref word
                dp[i, j - 1] + 1,
            )  # insert hyp word

    # Backtrace to split the distance into S/D/I (deterministic tie-break).
    i, j = R, H
    subs = dels = ins = 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and dp[i, j] == dp[i - 1, j - 1] + (ref[i - 1] != hyp[j - 1]):
            subs += ref[i - 1] != hyp[j - 1]
            i, j = i - 1, j - 1
        elif i > 0 and dp[i, j] == dp[i - 1, j] + 1:
            dels += 1
            i -= 1
        else:
            ins += 1
            j -= 1

    return {
        "wer": float(dp[R, H]) / R,
        "substitutions": int(subs),
        "deletions": int(dels),
        "insertions": int(ins),
        "n_ref_words": R,
    }


@dataclass
class LoopFlag:
    looping: bool
    ngram: str | None = None
    repeats: int = 0


def detect_repetition_loop(text: str, max_ngram: int = 5, threshold: int = 5) -> LoopFlag:
    """Flag pathological repetition: any 1..max_ngram word sequence repeated
    >= threshold times CONSECUTIVELY (temp-0 Whisper's known failure mode).

    Normal speech repeats words ("yeah, yeah") but not 5+ times in a row.
    """
    words = text.lower().split()
    for n in range(1, max_ngram + 1):
        for i in range(len(words) - n * threshold + 1):
            gram = words[i : i + n]
            run = 1
            j = i + n
            while j + n <= len(words) and words[j : j + n] == gram:
                run += 1
                j += n
            if run >= threshold:
                return LoopFlag(True, " ".join(gram), run)
    return LoopFlag(False)
