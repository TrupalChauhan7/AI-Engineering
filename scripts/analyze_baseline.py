#!/usr/bin/env python3
"""
analyze_baseline.py — failure analysis of the TofuEval-MeetingBank baseline.

Joins the per-sentence baseline results back to the TofuEval annotations to show
WHY the verifier misses hallucinations: the missed sentences, the human
explanation, and the error-type breakdown of missed vs caught. Read-only.

RUN (conda ds, from project root):
    python scripts/analyze_baseline.py
"""

import pandas as pd

res = pd.read_csv("results/tofueval_baseline_dev.csv")
fac = pd.read_csv("data/tofueval/meetingbank_factual_eval_dev.csv")

keys = ["doc_id", "topic", "model_name", "sent_idx"]
m = res.merge(fac[keys + ["summ_sent", "exp", "type"]], on=keys, how="left")

# pred_no may be serialised as the strings "True"/"False"
m["pred_no"] = m["pred_no"].astype(str).str.lower().eq("true")

miss = m[(m["gold"] == "no") & (~m["pred_no"])]  # missed hallucinations (FN)
caught = m[(m["gold"] == "no") & (m["pred_no"])]  # caught (TP)
false_alarm = m[(m["gold"] == "yes") & (m["pred_no"])]  # FP

print(
    f"missed (FN): {len(miss)}   caught (TP): {len(caught)}   false-alarm (FP): {len(false_alarm)}"
)

print("\n== MISSED hallucinations by error type ==")
print(miss["type"].fillna("(none)").value_counts().to_string())
print("\n== CAUGHT hallucinations by error type ==")
print(caught["type"].fillna("(none)").value_counts().to_string())

print("\n== what verdict did the verifier give on MISSED cases? ==")
print(miss["verdict"].value_counts().to_string())

print("\n== 12 MISSED hallucination examples ==")
for r in miss.head(12).itertuples():
    exp = str(r.exp)[:220]
    print(f"\n[type: {r.type}] topic: {r.topic}")
    print(f"  sentence : {r.summ_sent}")
    print(f"  human why: {exp}")
    print(f"  verifier : {r.verdict}")
