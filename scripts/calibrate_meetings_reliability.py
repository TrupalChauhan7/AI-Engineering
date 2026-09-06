#!/usr/bin/env python3
"""
calibrate_meetings_reliability.py — set the meetings reliability bands from data.

The reliability verdict thresholds are population-specific (the academic project
calibrated them on the clinical DEV machine-note distribution). This measures the
combined flag count (n_unsupported + n_omitted) over generated MEETING notes and
suggests bands from the quartiles — the same method: reliable_max = Q1,
unreliable_min = Q3.

Runs the full meetings stack (qwen3:14b + claim_verify_v2 + omission_verify_v2).

RUN (conda ds, project root, Ollama running):
    python scripts/calibrate_meetings_reliability.py --limit 20   # ~20 min, approximate
    python scripts/calibrate_meetings_reliability.py              # all 35 dev meetings
"""

import argparse
import csv
import json
import os
import statistics

os.environ.setdefault("S2N_DOMAIN", "meetings")

from s2n.config import load_config  # noqa: E402
from s2n.evaluation.claim_verifier import ClaimVerifier  # noqa: E402
from s2n.generation.generator import NoteGenerator  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="only first N meetings (0=all)")
    ap.add_argument("--max-words", type=int, default=1200, help="truncate long transcripts")
    args = ap.parse_args()

    # unique transcripts by doc_id
    seen: dict[str, str] = {}
    with open("data/tofueval/prepared_factual_dev.jsonl") as f:
        for line in f:
            r = json.loads(line)
            seen.setdefault(r["doc_id"], r["transcript"])
    items = list(seen.items())
    if args.limit:
        items = items[: args.limit]

    cfg = load_config()
    gen = NoteGenerator(cfg)
    ver = ClaimVerifier(cfg)
    print(
        f"domain: {cfg.get('active_domain')} | generator: {cfg['llm']['model']} "
        f"| {len(items)} meetings\n"
    )

    combined, rows = [], []
    for i, (doc, tr) in enumerate(items, 1):
        w = tr.split()
        if args.max_words and len(w) > args.max_words:
            tr = " ".join(w[: args.max_words])
        note = gen.generate(tr).note
        rep = ver.score(tr, note)
        n_u = sum(1 for v in rep.verdicts if v != "supported")
        facts = ver.decompose_transcript(tr)
        om = ver.check_omissions(note, facts)
        n_o = sum(1 for v in om.verdicts if v == "absent")
        c = n_u + n_o
        combined.append(c)
        rows.append({"doc_id": doc, "n_unsupported": n_u, "n_omitted": n_o, "combined": c})
        print(f"  {i}/{len(items)} {doc}: combined={c} ({n_u} unsupported + {n_o} omitted)")

    q1, med, q3 = statistics.quantiles(combined, n=4)
    print("\n=== combined flag-count distribution (meeting notes) ===")
    print(
        f"  n={len(combined)}  min={min(combined)}  q1={q1:.1f}  "
        f"median={med:.1f}  q3={q3:.1f}  max={max(combined)}"
    )
    print(f"  SUGGESTED meetings bands: reliable_max={round(q1)}  unreliable_min={round(q3)}")

    os.makedirs("results", exist_ok=True)
    with open("results/meetings_calibration.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        wr.writeheader()
        wr.writerows(rows)
    print("  -> results/meetings_calibration.csv")


if __name__ == "__main__":
    main()
