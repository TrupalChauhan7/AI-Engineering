#!/usr/bin/env python3
"""
end_to_end_meetings.py — the FULL meetings pipeline on one real MeetingBank transcript.

Runs transcript -> minutes (Qwen 3) -> verifier (claim_verify_v2) -> omission
(omission_verify_v2) -> reliability verdict, all under the meetings domain profile.
Defaults S2N_DOMAIN=meetings so the improved stack is used automatically.

RUN (conda ds, project root, Ollama running):
    python scripts/end_to_end_meetings.py            # first MeetingBank dev transcript
    python scripts/end_to_end_meetings.py --index 3
"""

import argparse
import json
import os

os.environ.setdefault("S2N_DOMAIN", "meetings")  # this demo is the meetings domain

from s2n.config import load_config  # noqa: E402  (must set env before importing)
from s2n.service import run_pipeline  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=int, default=0, help="which MeetingBank dev transcript")
    ap.add_argument(
        "--max-words", type=int, default=1200, help="truncate long transcripts for a quick run"
    )
    args = ap.parse_args()

    with open("data/tofueval/prepared_factual_dev.jsonl") as f:
        data = [json.loads(line) for line in f]
    item = data[args.index]
    transcript = item["transcript"]
    words = transcript.split()
    if args.max_words and len(words) > args.max_words:
        transcript = " ".join(words[: args.max_words])

    cfg = load_config()
    print(f"domain: {cfg.get('active_domain')} | generator: {cfg['llm']['model']}")
    print(
        f"verify prompt: {cfg['claim_verifier']['verify_prompt']} | "
        f"omission prompt: {cfg['claim_verifier']['omission_verify_prompt']}"
    )
    print(f"meeting: {item['doc_id']}  ({len(transcript.split())} transcript words)\n")

    result = run_pipeline(transcript=transcript, cfg=cfg)

    print("=" * 70)
    print("GENERATED MINUTES")
    print("=" * 70)
    print(result["note"])

    r = result["reliability"]
    print("\n" + "=" * 70)
    print("RELIABILITY")
    print("=" * 70)
    print(
        f"  verdict: {r['verdict'].upper()}  (combined = {r['combined']}: "
        f"{r['n_unsupported']} unsupported + {r['n_omitted']} omitted)"
    )
    print(
        "  NOTE: verdict bands are clinical-calibrated (reliable<=7 / unreliable>=12); "
        "per-domain calibration is a TODO — read the COUNTS, not the label."
    )
    if r["unsupported_claims"]:
        print("\n  Unsupported claims (possible hallucinations):")
        for c in r["unsupported_claims"][:8]:
            print(f"   - {c}")
    if r["omitted_facts"]:
        print("\n  Omitted facts (in transcript, missing from minutes):")
        for fct in r["omitted_facts"][:8]:
            print(f"   - {fct}")
    print(f"\n  timings: {result['timings']}")


if __name__ == "__main__":
    main()
