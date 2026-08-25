"""Tier-2 baseline: SummaC-ZS (source-grounded, NLI-based) — our own impl.

WHY (pre-registration §B1 tier 2, §A1.3): SummaC is the PRIMARY comparator — the
fair rival, because like the judge it reads the TRANSCRIPT (not a reference
note). The sharper thesis is "does an expensive LLM judge beat a cheap NLI
metric that also reads the source?".

We implement SummaC-ZS (Laban et al., TACL 2022) directly rather than using the
stale `summac` pip package (its old pins broke the shared env). The algorithm:
  * split source (transcript) and summary (note) into sentences;
  * for each NOTE sentence (hypothesis), take the MAX entailment probability
    over all SOURCE sentences (premise)  [op1 = max];
  * the note score is the MEAN of those per-sentence maxima  [op2 = mean].
Score in [0,1]; HIGHER = more faithful.

Implementing it ourselves also fixes the A1.1 trap: we own the CLINICAL
sentence segmentation instead of fighting a general-domain splitter.
"""

from __future__ import annotations

import re

import numpy as np

from s2n.config import load_config

# Entailment is index 1 for cross-encoder NLI models
# (id2label = {0: contradiction, 1: entailment, 2: neutral}).
_ENTAIL_IDX = 1

_SPEAKER_RE = re.compile(r"^(Doctor|Patient)\s*:\s*", re.IGNORECASE)
_SENT_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")


def split_clinical_sentences(text: str, min_len: int = 3) -> list[str]:
    """Segment dialogue / clinical shorthand into sentences.

    Clinical notes lean on NEWLINES (one finding per line) and terse
    punctuation, which general splitters mishandle (A1.1). We split on newlines
    first, strip speaker labels, then split each line on sentence punctuation.
    """
    sents: list[str] = []
    for line in str(text).split("\n"):
        line = _SPEAKER_RE.sub("", line.strip())
        if not line:
            continue
        for s in _SENT_SPLIT_RE.split(line):
            s = s.strip()
            if len(s) >= min_len:
                sents.append(s)
    return sents


class SummaCZS:
    """SummaC-ZS scorer over an NLI cross-encoder (loaded once, reused)."""

    def __init__(self, cfg: dict | None = None):
        cfg = (cfg or load_config())["summac"]
        self.batch_size = cfg.get("batch_size", 256)
        # Local import so torch/transformers load only when SummaC is used.
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(cfg["model"])

    def score_one(self, source: str, note: str) -> float:
        src = split_clinical_sentences(source)
        gen = split_clinical_sentences(note)
        if not src or not gen:
            return float("nan")
        # All (premise=source, hypothesis=note) pairs; entailment prob per pair.
        pairs = [(s, g) for g in gen for s in src]
        probs = self.model.predict(
            pairs, apply_softmax=True, batch_size=self.batch_size, show_progress_bar=False
        )
        ent = np.asarray(probs)[:, _ENTAIL_IDX].reshape(len(gen), len(src))
        return float(ent.max(axis=1).mean())  # max over source, mean over note

    def score_many(self, sources: list[str], notes: list[str]) -> list[float]:
        return [self.score_one(s, n) for s, n in zip(sources, notes)]
