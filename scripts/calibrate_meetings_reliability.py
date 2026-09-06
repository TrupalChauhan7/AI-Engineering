#!/usr/bin/env python3
"""
calibrate_meetings_reliability.py — set the meetings reliability bands from data.

The reliability verdict thresholds are population-specific (the academic project
calibrated them on the clinical DEV machine-note distribution). This measures the
combined flag count (n_unsupported + n_omitted) over generated MEETING notes and
suggests bands from the quartiles: reliable_max = Q1, unreliable_min = Q3.

Runs in TWO PHASES so only one model is in memory at a time (avoids Qwen<->Llama
swap thrash on a 24 GB Mac):
  Phase A: generate ALL notes with the generator (qwen3:14b), cache them.
  Phase B: verify + omission on ALL notes with the verifier (llama3.1:8b).
Notes are cached, so re-running (e.g. after a prompt change) can reuse Phase A.

RUN (conda ds, project root, Ollama running):
    python scripts/calibrate_meetings_reliability.py --limit 20
    python scripts/calibrate_meetings_reliability.py            # all 35 dev meetings
    python scripts/calibrate_meetings_reliability.py --regen    # ignore cached notes
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

NOTES_CACHE = "results/meetings_calib_notes.jsonl"


def load_transcripts(limit: int, max_words: int) -> list[tuple[str, str]]:
    seen: dict[str, str] = {}
    with open("data/tofueval/prepared_factual_dev.jsonl") as f:
        for line in f:
            r = json.loads(line)
            seen.setdefault(r["doc_id"], r["transcript"])
    items = list(seen.items())
    if limit:
        items = items[:limit]
    out = []
    for doc, tr in items:
        w = tr.split()
        out.append((doc, " ".join(w[:max_words]) if max_words and len(w) > max_words else tr))
    return out


def phase_a_generate(items, cfg, regen: bool) -> dict[str, str]:
    """Generate a note per meeting (generator model only). Cached."""
    cache: dict[str, str] = {}
    if os.path.exists(NOTES_CACHE) and not regen:
        with open(NOTES_CACHE) as f:
            for line in f:
                r = json.loads(line)
                cache[r["doc_id"]] = r["note"]
    todo = [(d, t) for d, t in items if d not in cache]
    if todo:
        gen = NoteGenerator(cfg)
        print(f"Phase A: generating {len(todo)} notes (qwen)...")
        with open(NOTES_CACHE, "a") as f:
            for i, (doc, tr) in enumerate(todo, 1):
                try:
                    note = gen.generate(tr).note
                except Exception as e:  # noqa: BLE001
                    print(f"  ! gen failed {doc}: {e}")
                    continue
                cache[doc] = note
                f.write(json.dumps({"doc_id": doc, "note": note}) + "\n")
                print(f"  A {i}/{len(todo)} {doc}")
    else:
        print(f"Phase A: all {len(items)} notes cached.")
    return cache


def phase_b_verify(items, notes, cfg) -> list[dict]:
    """Verify + omission per meeting (verifier model only)."""
    ver = ClaimVerifier(cfg)
    print("\nPhase B: verify + omission (llama)...")
    rows = []
    for i, (doc, tr) in enumerate(items, 1):
        note = notes.get(doc)
        if not note:
            continue
        try:
            rep = ver.score(tr, note)
            n_u = sum(1 for v in rep.verdicts if v != "supported")
            facts = ver.decompose_transcript(tr)
            om = ver.check_omissions(note, facts)
            n_o = sum(1 for v in om.verdicts if v == "absent")
        except Exception as e:  # noqa: BLE001
            print(f"  ! verify failed {doc}: {e}")
            continue
        rows.append({"doc_id": doc, "n_unsupported": n_u, "n_omitted": n_o, "combined": n_u + n_o})
        print(f"  B {i}/{len(items)} {doc}: combined={n_u + n_o} ({n_u}+{n_o})")
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="only first N meetings (0=all)")
    ap.add_argument("--max-words", type=int, default=1200, help="truncate long transcripts")
    ap.add_argument("--regen", action="store_true", help="ignore cached notes, regenerate")
    ap.add_argument("--timeout", type=int, default=300, help="per-call Ollama timeout (s)")
    args = ap.parse_args()

    cfg = load_config()
    cfg["llm"]["timeout"] = args.timeout  # fail fast on a hang instead of 20 min
    print(
        f"domain: {cfg.get('active_domain')} | generator: {cfg['llm']['model']} "
        f"| timeout {args.timeout}s"
    )

    items = load_transcripts(args.limit, args.max_words)
    notes = phase_a_generate(items, cfg, args.regen)
    rows = phase_b_verify(items, notes, cfg)

    combined = sorted(r["combined"] for r in rows)
    if len(combined) < 4:
        print(f"\nOnly {len(combined)} results — need >=4 for quartiles.")
        return
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
