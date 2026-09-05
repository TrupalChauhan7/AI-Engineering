#!/usr/bin/env python3
"""
eval_omission_probe.py — OMISSION-detection baseline via a controlled key-points probe.

For each (doc, topic): build a "complete summary" from the human key_points, remove
ONE known key_point, and check whether the verifier's omission axis flags the removed
fact as absent (recall) without wrongly flagging kept facts as absent (false-omission).

Ground truth is exact (we control the deletion) — mirrors the academic self-built
omission probe, so it carries the same honest caveat: deletions are obvious, so recall
here is an UPPER BOUND for subtler real omissions.

--omission-prompt swaps the omission prompt (e.g. a future v2) without editing config.

RUN (conda ds, from project root, Ollama running):
    python scripts/eval_omission_probe.py --limit 20   # sanity
    python scripts/eval_omission_probe.py              # full baseline
"""

import argparse
import csv
import json
import os
import random
import time

from s2n.config import load_config
from s2n.evaluation.claim_verifier import ClaimVerifier
from s2n.generation.prompts import load_prompt


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="only first N items (0=all)")
    ap.add_argument("--omission-prompt", default=None, help="override omission prompt")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    with open("data/tofueval/prepared_completeness.jsonl") as f:
        data = [json.loads(line) for line in f]
    if args.limit:
        data = data[: args.limit]

    rng = random.Random(args.seed)
    cfg = load_config()
    v = ClaimVerifier(cfg)
    prompt_ver = args.omission_prompt or cfg["claim_verifier"]["omission_verify_prompt"]
    if args.omission_prompt:
        v.omission_prompt = load_prompt("judge", args.omission_prompt)

    print(f"omission prompt: {prompt_ver} | items: {len(data)}\n")

    caught = total_removed = false_om = total_kept = 0
    rows = []
    t0 = time.perf_counter()
    for i, item in enumerate(data, 1):
        kps = item["key_points"]
        ridx = rng.randrange(len(kps))
        kept = [k for j, k in enumerate(kps) if j != ridx]
        summary = " ".join(kept)  # a "complete" summary missing exactly one known fact
        rep = v.check_omissions(summary, kps)
        verdicts = rep.verdicts if rep.facts else ["present"] * len(kps)
        for j, (k, verd) in enumerate(zip(kps, verdicts)):
            removed = j == ridx
            if removed:
                total_removed += 1
                caught += verd == "absent"
            else:
                total_kept += 1
                false_om += verd == "absent"
            rows.append(
                {
                    "doc_id": item["doc_id"],
                    "topic": item["topic"],
                    "removed": removed,
                    "verdict": verd,
                }
            )
        if i % 25 == 0:
            print(f"  {i}/{len(data)}...")
    dt = time.perf_counter() - t0

    recall = caught / total_removed if total_removed else 0.0
    far = false_om / total_kept if total_kept else 0.0
    print(f"\n=== OMISSION baseline [{prompt_ver}] (single-deletion probe) ===")
    print(
        f"  omission recall (removed facts caught absent): {caught}/{total_removed} = {recall:.3f}"
    )
    print(f"  false-omission rate (kept facts wrongly absent): {false_om}/{total_kept} = {far:.3f}")
    print(f"  runtime: {dt / 60:.1f} min")

    os.makedirs("results", exist_ok=True)
    out = f"results/omission_probe_{prompt_ver.replace('/', '_')}.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  per-fact results -> {out}")


if __name__ == "__main__":
    main()
