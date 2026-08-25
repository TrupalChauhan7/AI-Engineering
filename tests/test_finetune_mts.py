"""MTS-Dialog -> chat-format instruction pairs (deterministic; no training)."""

import json

from s2n.finetune.mts import format_csv, row_to_chat, section_name


def test_section_name_maps_codes():
    assert section_name("GENHX") == "History of Present Illness"
    assert section_name("FAM/SOCHX") == "Family and Social History"
    assert section_name("UNKNOWN_CODE") == "Unknown_Code"  # graceful fallback


def test_row_to_chat_structure_and_faithfulness_instruction():
    ex = row_to_chat("Doctor: Any pain?\nPatient: Yes, my knee.", "CC", "Knee pain.")
    roles = [m["role"] for m in ex["messages"]]
    assert roles == ["system", "user", "assistant"]
    assert "ONLY information present in the dialogue" in ex["messages"][0]["content"]
    assert "Chief Complaint" in ex["messages"][1]["content"]
    assert ex["messages"][2]["content"] == "Knee pain."


def test_format_csv_skips_empty_and_writes_jsonl(tmp_path):
    import pandas as pd

    csv = tmp_path / "mts.csv"
    pd.DataFrame(
        {
            "ID": [0, 1, 2],
            "section_header": ["CC", "GENHX", "ALLERGY"],
            "section_text": ["Knee pain.", "", "NKDA"],  # middle row is empty -> skipped
            "dialogue": ["Doctor: hi\nPatient: knee", "Doctor: hi", "Doctor: allergies?"],
        }
    ).to_csv(csv, index=False)

    out = tmp_path / "train.jsonl"
    n = format_csv(csv, out)
    assert n == 2  # empty section_text row skipped
    lines = out.read_text().strip().splitlines()
    assert len(lines) == 2
    assert all("messages" in json.loads(line) for line in lines)
