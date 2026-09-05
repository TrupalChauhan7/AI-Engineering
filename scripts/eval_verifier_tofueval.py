#!/usr/bin/env python3
"""
eval_verifier_tofueval.py — verifier hallucination-detection eval on TofuEval-MeetingBank.

Runs the claim-verifier over each summary sentence, checks it against the meeting
transcript, and scores against TofuEval's human sent_label:
    verdict != "supported"  ==>  predicts "inconsistent" (a flag)
    sent_label == "no"      ==>  human says inconsistent
Metric = precision / recall / F1 on the 'no' (hallucination) class + accuracy.

--verify-prompt lets us swap the verify prompt (e.g. claim_verify_v1 vs v2)
WITHOUT editing config, so we can A/B improvements against the baseline.

RUN (conda ds, from project root, Ollama running):
    python scripts/eval_verifier_tofueval.py --split dev --limit 10          # sanity
    python scripts/eval_verifier_tofueval.py --split dev                     # baseline (v1)
    python scripts/eval_verifier_tofueval.py --split dev --verify-prompt claim_verify_v2
"""

import argparse
import csv
import json
import os
import time

from s2n.config import load_config
from s2n.evaluation.claim_verifier import ClaimVerifier
from s2n.generation.prompts import load_prompt


def load_prepared(split: str) -> list[dict]:
    path = f"data/tofueval/prepared_factual_{split}.jsonl"
    with open(path) as f:
        return [json.loads(line) for line in f]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="dev", choices=["dev", "test"])
    ap.add_argument("--limit", type=int, default=0, help="only first N summaries (0=all)")
    ap.add_argument(
        "--verify-prompt", default=None, help="override verify prompt, e.g. claim_verify_v2"
    )
    args = ap.parse_args()

    data = load_prepared(args.split)
    if args.limit:
        data = data[: args.limit]

    cfg = load_config()
    v = ClaimVerifier(cfg)
    prompt_ver = args.verify_prompt or cfg["claim_verifier"]["verify_prompt"]
    if args.verify_prompt:
        # override the verify prompt without touching config
        v.verify_prompt = load_prompt("judge", args.verify_prompt)

    print(f"verifier model: {cfg['claim_verifier']['model']} | verify prompt: {prompt_ver}")
    print(f"scoring {len(data)} summaries from TofuEval-MeetingBank [{args.split}]\n")

    tp = fp = fn = tn = 0
    rows = []
    t0 = time.perf_counter()
    for i, summ in enumerate(data, 1):
        sents = [s["summ_sent"] for s in summ["sentences"]]
        verdicts = v._verify_batched(summ["transcript"], sents)
        if verdicts is None:
            verdicts = [v._verify_one(summ["transcript"], s) for s in sents]
        for s, verd in zip(summ["sentences"], verdicts):
            pred_no = verd != "supported"
            gold_no = s["sent_label"] == "no"
            tp += pred_no and gold_no
            fp += pred_no and not gold_no
            fn += (not pred_no) and gold_no
            tn += (not pred_no) and not gold_no
            rows.append(
                {
                    "doc_id": summ["doc_id"],
                    "topic": summ["topic"],
                    "model_name": summ["model_name"],
                    "sent_idx": s["sent_idx"],
                    "gold": s["sent_label"],
                    "verdict": verd,
                    "pred_no": pred_no,
                }
            )
        if i % 25 == 0:
            print(f"  {i}/{len(data)} summaries...")
    dt = time.perf_counter() - t0

    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    acc = (tp + tn) / (tp + fp + fn + tn) if (tp + fp + fn + tn) else 0.0

    print(f"\n=== hallucination detection [{prompt_ver}] (positive class = 'no') ===")
    print(f"  TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    print(f"  precision={prec:.3f}  recall={rec:.3f}  F1={f1:.3f}  accuracy={acc:.3f}")
    print(f"  runtime: {dt / 60:.1f} min for {len(rows)} sentences")

    os.makedirs("results", exist_ok=True)
    tag = prompt_ver.replace("/", "_")
    out = f"results/tofueval_{tag}_{args.split}.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"  per-sentence results -> {out}")


if __name__ == "__main__":
    main()
