#!/usr/bin/env python3
"""
prepare_tofueval.py — join TofuEval MeetingBank annotations to their source
transcripts and save prepared per-summary eval files.

doc_id == MeetingBank meeting_id (exact match across all MeetingBank splits).
Each output record = one summary: {doc_id, topic, model_name, transcript,
sentences:[{sent_idx, summ_sent, sent_label, type}]}.

RUN (from the project folder):
    python scripts/prepare_tofueval.py
Writes data/tofueval/prepared_factual_dev.jsonl and _test.jsonl.
"""

import json
import os

import pandas as pd
from datasets import load_dataset

OUT = "data/tofueval"


def build_transcript_map() -> dict[str, str]:
    ds = load_dataset("lytang/MeetingBank-transcript")
    m: dict[str, str] = {}
    for sp in ds:
        for mid, src in zip(ds[sp]["meeting_id"], ds[sp]["source"]):
            m[str(mid)] = src
    return m


def prepare(split: str, tmap: dict[str, str]) -> None:
    df = pd.read_csv(os.path.join(OUT, f"meetingbank_factual_eval_{split}.csv"))
    out, missing = [], set()
    for (doc, topic, model), g in df.groupby(["doc_id", "topic", "model_name"]):
        doc = str(doc)
        if doc not in tmap:
            missing.add(doc)
            continue
        g = g.sort_values("sent_idx")
        sents = [
            {
                "sent_idx": int(r.sent_idx),
                "summ_sent": str(r.summ_sent),
                "sent_label": str(r.sent_label),
                "type": (None if pd.isna(r.type) else str(r.type)),
            }
            for r in g.itertuples()
        ]
        out.append(
            {
                "doc_id": doc,
                "topic": str(topic),
                "model_name": str(model),
                "transcript": tmap[doc],
                "sentences": sents,
            }
        )
    path = os.path.join(OUT, f"prepared_factual_{split}.jsonl")
    with open(path, "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    n_sent = sum(len(r["sentences"]) for r in out)
    print(f"{split}: {len(out)} summaries, {n_sent} sentences -> {path}")
    if missing:
        print(f"  WARNING: {len(missing)} docs had no transcript match: {sorted(missing)[:5]}")


def main() -> None:
    print("Loading MeetingBank transcripts (cached after first run)...")
    tmap = build_transcript_map()
    print(f"  {len(tmap)} meeting transcripts")
    for split in ("dev", "test"):
        prepare(split, tmap)
    print("\nDone.")


if __name__ == "__main__":
    main()
