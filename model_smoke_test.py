#!/usr/bin/env python3
"""
model_smoke_test.py — Stage-1 model check (per Prof. Rauf's guidance).

Runs a FEW real PriMock57 consultations through a LOCAL Ollama model so you can SEE:
  (1) what SOAP note the model actually produces from a real transcript,
  (2) how fast it runs + tokens/sec (your "GPU constraints" reality check),
  (3) the model's general ability on OUR task (generation, not medical Q&A).

This is the empirical first step of the model-selection funnel — look at real
outputs before committing.

------------------------------------------------------------------------------
BEFORE RUNNING (on your Mac):
  1. Install Ollama (brew isn't on your machine, so use the app): https://ollama.com
  2. Pull the model you want to test:
        ollama pull medgemma:4b        # medical generator, 3.3 GB
        # (optional others to compare:)
        ollama pull llama3.2:1b
        ollama pull qwen3:1.7b
  3. From the project folder, run:
        python model_smoke_test.py                 # defaults to medgemma:4b, 3 examples
        MODEL=llama3.2:1b python model_smoke_test.py   # try another model
        N=2 MODEL=qwen3:1.7b python model_smoke_test.py

No extra pip installs — this talks to Ollama's local REST API (localhost:11434).
Outputs are also saved to results/ so you can review or show them.
------------------------------------------------------------------------------
"""

import glob
import json
import os
import re
import sys
import urllib.error
import urllib.request

# ---------------- config (override via environment variables) ----------------
ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL = os.environ.get("MODEL", "medgemma:4b")
N_EXAMPLES = int(os.environ.get("N", "3"))
TRANSCRIPTS = os.path.join(ROOT, "primock57", "transcripts")
PROMPT_FILE = os.path.join(ROOT, "prompts", "note_generation", "v1.0.md")
OLLAMA_URL = "http://localhost:11434/api/generate"
OUT_DIR = os.path.join(ROOT, "results")


# ---------------- 1. read a PriMock57 transcript (.TextGrid) -----------------
def read_utterances(path, speaker):
    """Pair each interval's start time (xmin) with its text; skip empty ones."""
    utts, cur_xmin = [], None
    with open(path, encoding="utf-8", errors="ignore") as fh:
        for line in fh:
            line = line.strip()
            m = re.match(r"xmin\s*=\s*([\d.]+)", line)
            if m:
                cur_xmin = float(m.group(1))
            m = re.match(r'text\s*=\s*"(.*)"', line)
            if m and cur_xmin is not None:
                text = m.group(1).strip()
                if text:
                    utts.append((cur_xmin, speaker, text))
    return utts


def load_transcript(prefix):
    """Merge the doctor + patient TextGrids into one time-ordered dialogue."""
    doc = read_utterances(prefix + "_doctor.TextGrid", "Doctor")
    pat = read_utterances(prefix + "_patient.TextGrid", "Patient")
    merged = sorted(doc + pat, key=lambda x: x[0])
    return "\n".join(f"{spk}: {txt}" for _, spk, txt in merged)


# ---------------- 2. call the local Ollama model -----------------------------
def generate(prompt, model):
    payload = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0},  # deterministic, for a fair look
        }
    ).encode()
    req = urllib.request.Request(
        OLLAMA_URL, data=payload, headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=1200) as r:
        return json.load(r)


def stats(resp):
    """Turn Ollama's nanosecond timings into human numbers (the GPU reality)."""
    ev = resp.get("eval_count", 0)
    ev_ns = resp.get("eval_duration", 1) or 1
    load_s = resp.get("load_duration", 0) / 1e9
    total_s = resp.get("total_duration", 0) / 1e9
    tok_s = ev / (ev_ns / 1e9) if ev_ns else 0
    return f"{total_s:.1f}s total | model load {load_s:.1f}s | {ev} tokens @ {tok_s:.1f} tok/s"


# ---------------- 3. run a few examples --------------------------------------
def main():
    if not os.path.exists(PROMPT_FILE):
        sys.exit(f"Prompt not found: {PROMPT_FILE}")
    prompt_template = open(PROMPT_FILE).read()

    doctors = sorted(glob.glob(os.path.join(TRANSCRIPTS, "*_doctor.TextGrid")))[:N_EXAMPLES]
    if not doctors:
        sys.exit(f"No transcripts found in {TRANSCRIPTS}")

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"smoke_test_{MODEL.replace(':', '_').replace('/', '_')}.md")
    out = open(out_path, "w")
    out.write(f"# Stage-1 model smoke test — {MODEL}\n\n")

    print(f"Model: {MODEL} | examples: {len(doctors)}\n")
    for d in doctors:
        prefix = d[: -len("_doctor.TextGrid")]
        name = os.path.basename(prefix)
        transcript = load_transcript(prefix)
        prompt = prompt_template.replace("{{transcript}}", transcript)

        print("=" * 72)
        print(f"CONSULTATION: {name}  (transcript {len(transcript)} chars)")
        try:
            resp = generate(prompt, MODEL)
        except urllib.error.URLError as e:
            sys.exit(
                f"\nCould not reach Ollama ({e}).\n"
                f"Is it running, and did you `ollama pull {MODEL}`?"
            )
        note = resp.get("response", "").strip()
        print(f"[{stats(resp)}]\n")
        print(f"--- {MODEL} SOAP note ---\n{note}\n")

        out.write(f"## {name}\n\n**Stats:** {stats(resp)}\n\n")
        out.write(f"### Generated note\n\n{note}\n\n---\n\n")

    out.close()
    print("=" * 72)
    print(f"Saved all outputs to: {out_path}")


if __name__ == "__main__":
    main()
