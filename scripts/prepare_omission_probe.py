#!/usr/bin/env python3
"""
prepare_omission_probe.py — build the omission probe from TofuEval completeness.

Each TofuEval completeness row gives, per (doc_id, topic), the human-written
`key_points` = the essential facts a complete summary must contain. We keep items
with >=2 key_points (so we can remove one and still have a summary).

RUN (from project root):
    python scripts/prepare_omission_probe.py
Writes data/tofueval/prepared_completeness.jsonl.
"""

import ast
import json
import os
import re
import statistics

OUT = "data/tofueval"


def main() -> None:
    import pandas as pd

    df = pd.read_csv(os.path.join(OUT, "meetingbank_completeness_final.csv"))
    out = []
    for r in df.itertuples():
        try:
            kps = ast.literal_eval(r.key_points)
        except (ValueError, SyntaxError):
            continue
        # strip leading "1. " / "2. " numbering and blanks
        kps = [re.sub(r"^\s*\d+\.\s*", "", str(k)).strip() for k in kps]
        kps = [k for k in kps if k]
        if len(kps) >= 2:
            out.append({"doc_id": str(r.doc_id), "topic": str(r.topic), "key_points": kps})

    path = os.path.join(OUT, "prepared_completeness.jsonl")
    with open(path, "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    counts = [len(r["key_points"]) for r in out]
    print(f"{len(out)} (doc,topic) items with >=2 key_points -> {path}")
    print(
        f"key_points per item: mean {statistics.mean(counts):.1f}, "
        f"min {min(counts)}, max {max(counts)}"
    )


if __name__ == "__main__":
    main()
