"""EVALUATION-side data access: the ANSWER KEY (notes/ + human_eval/).

WHY (project rule 1): these are the gold answers used ONLY for scoring, never
as generator input. Keeping them in a *separate module* from
``generation_data`` is the structural firewall: the generation path never
imports this file, so it cannot read an answer.

What lives here:
  * ``load_note`` / ``load_notes``     -> clinician SOAP note + highlights (JSON)
  * ``load_human_eval``                -> results.csv, one row per (consultation,
                                          model, evaluator), with the "!"/"-"
                                          criticality flags parsed into counts
  * ``aggregate_by_note``              -> mean counts per (consultation, model)
                                          across its 5 evaluators, PLUS the
                                          per-evaluator rows kept for the human
                                          ceiling / IAA (pre-registration §B6)

DOCTOR-NOTE DECISION (do not let this slip through silently):
  results.csv includes a pseudo-model literally named "doctor" — the human note
  scored as if a model. It has ~zero errors and sits at the extreme of the
  metric-vs-human relationship, so it can distort the correlation study.
  Policy here: KEEP it in the data (exposed via the ``is_doctor`` column and
  ``DOCTOR_MODEL``); whether to INCLUDE or EXCLUDE it in the RQ1 correlation is
  an analysis decision to pre-register before touching TEST. Use
  ``machine_notes_only()`` to get the exclude-doctor view.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from s2n.config import ROOT, load_config
from s2n.data.ids import canonical_consultation_id

# The pseudo-model in results.csv that is actually the human note (see above).
DOCTOR_MODEL = "doctor"

# results.csv free-text fields whose lines carry "!"/"-" criticality flags.
_FLAG_FIELDS = ("Incorrect statements", "Omissions")


def parse_flagged_items(cell: object) -> dict[str, int]:
    """Count critical ('!') vs non-critical ('-') items in one CSV cell.

    Each field is a newline-separated list where every item is prefixed by
    ``!`` (critical) or ``-`` (non-critical). Blank / NaN -> all zeros.
    """
    critical = noncritical = unflagged = 0
    if isinstance(cell, str):
        for raw in cell.splitlines():
            item = raw.strip()
            if not item:
                continue
            if item.startswith("!"):
                critical += 1
            elif item.startswith("-"):
                noncritical += 1
            else:
                unflagged += 1  # defensive: item present but no leading flag
    return {
        "critical": critical,
        "noncritical": noncritical,
        "unflagged": unflagged,
        "total": critical + noncritical + unflagged,
    }


class EvaluationLoader:
    """Reads gold notes and the human-evaluation answer key."""

    def __init__(self, notes_dir: str | Path, human_eval_dir: str | Path):
        self.notes_dir = Path(notes_dir)
        self.human_eval_dir = Path(human_eval_dir)

    @classmethod
    def from_config(cls, cfg: dict | None = None) -> "EvaluationLoader":
        cfg = cfg or load_config()
        return cls(
            ROOT / cfg["paths"]["notes_gold"],
            ROOT / cfg["paths"]["human_eval"],
        )

    # ---- gold clinician notes -------------------------------------------
    def load_note(self, consultation_id: str) -> dict:
        """Return the clinician note JSON (note text + highlights + meta)."""
        path = self.notes_dir / f"{consultation_id}.json"
        with open(path) as f:
            return json.load(f)

    def load_notes(self, ids: list[str] | None = None) -> dict[str, dict]:
        ids = ids or [p.stem for p in sorted(self.notes_dir.glob("*.json"))]
        return {cid: self.load_note(cid) for cid in ids}

    # ---- human evaluation answer key ------------------------------------
    def _read_results(self) -> pd.DataFrame:
        """Read results.csv and normalise the Consultation column to canonical
        IDs so it joins cleanly to transcripts/notes (see s2n.data.ids)."""
        df = pd.read_csv(self.human_eval_dir / "results.csv")
        df["Consultation"] = df["Consultation"].map(canonical_consultation_id)
        return df

    def load_human_eval(self) -> pd.DataFrame:
        """results.csv, one row per (consultation, model, evaluator), with the
        criticality flags parsed into numeric columns."""
        df = self._read_results()
        for field in _FLAG_FIELDS:
            counts = df[field].apply(parse_flagged_items).apply(pd.Series)
            prefix = "incorrect" if field.startswith("Incorrect") else "omission"
            counts.columns = [f"{prefix}_{c}" for c in counts.columns]
            df = pd.concat([df, counts], axis=1)
        # Convenience roll-ups across both axes.
        df["errors_critical"] = df["incorrect_critical"] + df["omission_critical"]
        df["errors_total"] = df["incorrect_total"] + df["omission_total"]
        df["is_doctor"] = df["Model"] == DOCTOR_MODEL
        return df

    def note_texts(self) -> pd.DataFrame:
        """One row per (consultation, model) with the machine note TEXT.

        The note we score in Track A is the pre-rated ``Model Note`` from
        results.csv (identical across a note's 5 evaluator rows), so we take the
        first. This is the text the baselines and judge are evaluated on.
        """
        df = self._read_results()
        out = (
            df.groupby(["Consultation", "Model"], as_index=False)["Model Note"]
            .first()
            .rename(columns={"Model Note": "note_text"})
        )
        return out

    def aggregate_by_note(self) -> pd.DataFrame:
        """Mean counts per (consultation, model) across its evaluators.

        Also returns ``n_evaluators`` — full 5x overlap in this dataset means
        the human ceiling / IAA is computable (pre-registration §B6).
        """
        df = self.load_human_eval()
        count_cols = [c for c in df.columns if c.startswith(("incorrect_", "omission_"))]
        count_cols += ["errors_critical", "errors_total"]
        num_cols = count_cols + ["Post-edit time"]
        for c in num_cols:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        agg = df.groupby(["Consultation", "Model"], as_index=False).agg(
            n_evaluators=("Evaluator", "nunique"), **{c: (c, "mean") for c in num_cols}
        )
        agg["is_doctor"] = agg["Model"] == DOCTOR_MODEL
        return agg

    def machine_notes_only(self, agg: pd.DataFrame | None = None) -> pd.DataFrame:
        """The exclude-doctor view (see DOCTOR-NOTE DECISION at module top)."""
        agg = agg if agg is not None else self.aggregate_by_note()
        return agg[~agg["is_doctor"]].reset_index(drop=True)
