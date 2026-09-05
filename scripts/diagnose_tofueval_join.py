#!/usr/bin/env python3
"""Diagnose how TofuEval doc_ids map to MeetingBank meeting_ids (they don't match
exactly). Prints normalization results + prefix candidates so we can write the
correct join. Read-only; saves nothing."""

import os

import pandas as pd
from datasets import load_dataset

OUT = "data/tofueval"

# 1. TofuEval doc_ids
ids: set[str] = set()
for f in (
    "meetingbank_factual_eval_dev.csv",
    "meetingbank_factual_eval_test.csv",
    "meetingbank_completeness_final.csv",
):
    ids |= set(pd.read_csv(os.path.join(OUT, f))["doc_id"].astype(str))
print(f"TofuEval unique doc_ids: {len(ids)}")

# 2. MeetingBank meeting_ids (all splits)
ds = load_dataset("lytang/MeetingBank-transcript")
mb: set[str] = set()
for sp in ds:
    mb |= {str(x) for x in ds[sp]["meeting_id"]}
print(f"MeetingBank meeting_ids: {len(mb)}")


def norm(s: str) -> str:
    return s.replace(" ", "").replace("-", "").replace("_", "").lower()


mb_norm = {norm(x): x for x in mb}

exact = len(ids & mb)
normed = sum(1 for i in ids if norm(i) in mb_norm)
print(f"\nexact matches: {exact}/{len(ids)}")
print(f"normalized (strip spaces/dashes/underscores, lowercase) matches: {normed}/{len(ids)}")

# 3. show a few unmatched with prefix candidates so we can see the real pattern
print("\n--- examples: TofuEval id vs MeetingBank ids sharing the City_date prefix ---")
shown = 0
for i in sorted(ids):
    if i in mb:
        continue
    parts = i.split("_")
    pref = "_".join(parts[:2]) if len(parts) >= 2 else i
    cands = sorted(x for x in mb if x.startswith(pref))
    print(f"\nTOFU : {i!r}")
    print(f"  norm: {norm(i)!r}")
    print(f"  MB candidates with prefix {pref!r} ({len(cands)}): {cands[:10]}")
    if cands:
        print(f"  MB cand norms: {[norm(c) for c in cands[:10]]}")
    shown += 1
    if shown >= 6:
        break
