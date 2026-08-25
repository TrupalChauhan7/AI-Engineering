"""RQ1 Track A: assemble the scoring table for the PRE-RATED notes.

WHY (the whole point of Track A): PriMock57 already gives us machine notes with
human hallucination/omission labels. We do NOT need our own generator to work —
we just re-score those existing notes with each metric and ask which metric's
score best tracks the human error counts. This is the defensible RQ1 result for
A2 (the "insurance" track).

For each (consultation, model) note we gather THREE text objects + the target:
  * ``note_text``      — the pre-rated Model Note (what every metric scores)
  * ``reference_note`` — the clinician's gold note (for ROUGE/BERTScore/Leven.)
  * ``transcript``     — the gold dialogue (for SummaC and the judge)
  * human targets      — aggregated error counts (mean over the 5 evaluators)

DEV/TEST discipline: this assembler takes an explicit list of consultation IDs.
The harness passes the DEV ids only; TEST is never assembled until the one-shot
final run (pre-registration §D).
"""

from __future__ import annotations

import pandas as pd

from s2n.data.evaluation_data import EvaluationLoader
from s2n.data.generation_data import GenerationLoader
from s2n.data.splits import load_or_create_split


def build_scoring_table(
    consultation_ids: list[str],
    cfg: dict | None = None,
) -> pd.DataFrame:
    """Return one row per (consultation, model) note for the given IDs."""
    ev = EvaluationLoader.from_config(cfg)
    gen = GenerationLoader.from_config(cfg)  # transcripts only (leak-safe)

    # Human targets (means over evaluators) + note text, restricted to the IDs.
    targets = ev.aggregate_by_note()
    texts = ev.note_texts()
    table = targets.merge(texts, on=["Consultation", "Model"], how="left")
    table = table[table["Consultation"].isin(consultation_ids)].copy()

    # Attach the gold reference note and the gold transcript per consultation.
    ref_notes = {cid: ev.load_note(cid)["note"] for cid in consultation_ids}
    transcripts = {cid: gen.load(cid).dialogue for cid in consultation_ids}
    table["reference_note"] = table["Consultation"].map(ref_notes)
    table["transcript"] = table["Consultation"].map(transcripts)

    table = table.rename(columns={"Consultation": "consultation", "Model": "model"})
    return table.reset_index(drop=True)


def build_dev_table(cfg: dict | None = None) -> pd.DataFrame:
    """Convenience: the scoring table for the frozen DEV split only."""
    return build_scoring_table(load_or_create_split(cfg).dev, cfg)
