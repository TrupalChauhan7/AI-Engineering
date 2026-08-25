"""Format MTS-Dialog into chat-format instruction pairs for MLX-LM LoRA.

WHY (Track B, component #3): the generator is fine-tuned to improve content
selection and faithful clinical register. MTS-Dialog gives (dialogue snippet ->
one note section) pairs — a DIFFERENT corpus from PriMock57, so training here
and evaluating on PriMock57 stays leakage-free (disjoint institutions; the
generator never sees a PriMock note).

Mapping (approved): each row -> a chat example whose system message mirrors the
v1.0 generation prompt's faithfulness rule ("use ONLY the dialogue"), user asks
for the named section, assistant is the gold section text. MLX-LM applies the
model's own chat template at train time, so we emit the `messages` schema.

The training objective (snippet->section) differs from inference (full dialogue
->full SOAP) — an accepted, documented gap: LoRA transfers style/faithfulness,
and MTS rows are independent encounters that cannot be reassembled into full
notes. Evaluation is still full-dialogue->full-SOAP on PriMock57.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

# MTS-Dialog section-header codes -> human-readable clinical section names.
SECTION_NAMES = {
    "CC": "Chief Complaint",
    "GENHX": "History of Present Illness",
    "PASTMEDICALHX": "Past Medical History",
    "PASTSURGICAL": "Past Surgical History",
    "FAM/SOCHX": "Family and Social History",
    "ALLERGY": "Allergies",
    "ROS": "Review of Systems",
    "MEDICATIONS": "Medications",
    "ASSESSMENT": "Assessment",
    "EXAM": "Examination",
    "DIAGNOSIS": "Diagnosis",
    "DISPOSITION": "Disposition",
    "PLAN": "Plan",
    "EDCOURSE": "Emergency Department Course",
    "IMMUNIZATIONS": "Immunizations",
    "IMAGING": "Imaging",
    "GYNHX": "Gynecologic History",
    "PROCEDURES": "Procedures",
    "OTHER_HISTORY": "Other History",
    "LABS": "Laboratory Results",
}

SYSTEM = (
    "You are a clinical scribe. Write the requested section of a clinical note "
    "from the consultation dialogue. Use ONLY information present in the "
    "dialogue. Do not invent findings, medications, or history. Be concise and "
    "use standard clinical style."
)


def section_name(code: str) -> str:
    return SECTION_NAMES.get(str(code).strip(), str(code).strip().title())


def row_to_chat(dialogue: str, section_header: str, section_text: str) -> dict:
    """One MTS-Dialog row -> a chat example (MLX-LM `messages` schema)."""
    section = section_name(section_header)
    user = (
        f"Write the {section} section of the clinical note based on this "
        f"consultation dialogue.\n\nDialogue:\n{dialogue.strip()}"
    )
    return {
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": user},
            {"role": "assistant", "content": str(section_text).strip()},
        ]
    }


def _valid_row(row: pd.Series) -> bool:
    if pd.isna(row.get("dialogue")) or pd.isna(row.get("section_text")):
        return False  # pandas reads empty CSV cells as NaN
    d, t = str(row["dialogue"]).strip(), str(row["section_text"]).strip()
    return len(d) > 0 and len(t) > 0 and t.lower() not in ("none", "nan")


def format_csv(csv_path: str | Path, out_jsonl: str | Path, limit: int | None = None) -> int:
    """Convert one MTS-Dialog CSV to chat-format JSONL. Returns rows written."""
    df = pd.read_csv(csv_path)
    if limit is not None:
        df = df.head(limit)
    out_jsonl = Path(out_jsonl)
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out_jsonl, "w") as f:
        for _, row in df.iterrows():
            if not _valid_row(row):
                continue
            example = row_to_chat(row["dialogue"], row["section_header"], row["section_text"])
            f.write(json.dumps(example) + "\n")
            n += 1
    return n
