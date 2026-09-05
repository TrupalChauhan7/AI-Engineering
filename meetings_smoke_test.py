#!/usr/bin/env python3
"""
meetings_smoke_test.py — Phase-2 Stage-1 check for the MEETINGS generator.

Mirrors the academic model_smoke_test.py: run a real prompt through a local
Ollama model and SEE the output before committing to it. Here we check whether
Qwen 3 14B writes good MEETING MINUTES in our locked output spec (DESIGN_DECISIONS
D4): Summary / Key decisions / Action items (owner + due date) / Open questions.

No project dependencies — just Python stdlib talking to Ollama's local API.

RUN (from the project folder, with Ollama running):
    python meetings_smoke_test.py                 # defaults to qwen3:14b
    MODEL=qwen3:8b python meetings_smoke_test.py   # try another size
Output is printed and also saved to results/meetings_smoke_<model>.md
"""

import json
import os
import re
import time
import urllib.request

MODEL = os.environ.get("MODEL", "qwen3:14b")
OLLAMA_URL = "http://localhost:11434/api/chat"
OUT_DIR = "results"

# --- the locked meeting-note spec (DESIGN_DECISIONS D4) ----------------------
SYSTEM = (
    "You are a meeting-minutes assistant. From the meeting transcript, write "
    "structured minutes using ONLY information explicitly stated in the "
    "transcript. Do not invent decisions, owners, dates, or facts. If a due "
    "date or owner is not stated, write 'not specified'."
)
INSTRUCTION = (
    "Write the minutes with EXACTLY these four sections and nothing else:\n"
    "## Summary\n(2-4 sentences overview)\n"
    "## Key decisions\n(bulleted list of decisions actually made)\n"
    "## Action items\n(bulleted list, each as: task - owner - due date)\n"
    "## Open questions\n(bulleted list of things raised but left unresolved)"
)

# --- a realistic sample meeting (synthetic; smoke-test only, NOT eval data) ---
# Deliberately contains: 2 clear decisions, 3 action items (2 with owners+dates,
# 1 with owner but no date), and 1 open question deferred to next time. This lets
# us judge whether the model extracts each element correctly and honestly marks
# the missing due date as 'not specified' rather than inventing one.
TRANSCRIPT = """\
Alice (PM): Okay, let's start. Main thing today is the v2.1 release. Ben, where are we on the login bug?
Ben (Eng): It's the session-timeout issue. I've reproduced it, I should have a fix by Wednesday.
Alice: Good. Let's lock that in - Ben fixes the login bug by Wednesday.
Ben: Works for me.
Alice: Chandra, the new icon set?
Chandra (Design): Almost done. I can hand over the final icons Thursday.
Alice: Great, Chandra delivers the icons Thursday. So if the bug's fixed Wednesday and icons land Thursday, I think we ship v2.1 next Friday. Any objections?
Dev (Marketing): No objection. I'll draft the release announcement email.
Alice: Perfect, Dev drafts the release email - Dev, when can you have it?
Dev: I'll get to it, not totally sure on timing yet.
Alice: Okay, owner is Dev, timing to be confirmed. Decision: we ship v2.1 next Friday.
Chandra: One thing - are we still including dark mode in this release?
Alice: Good question. Honestly I think dark mode isn't ready. Let's cut it from 2.1.
Ben: Agreed, it needs more testing.
Alice: Decided - dark mode is dropped from v2.1, we'll revisit for 2.2.
Dev: Separate thing - should we raise the pricing tier with this release? Marketing's been asking.
Alice: That's a bigger conversation. Let's not decide that today - we'll pick it up next meeting.
Alice: Anything else? No? Thanks everyone.
"""


def ask(model: str, system: str, user: str) -> tuple[str, dict]:
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "stream": False,
        "think": False,  # qwen3 is a thinking model; ask for a clean answer
        "options": {"temperature": 0},
    }
    data = json.dumps(payload).encode()
    headers = {"Content-Type": "application/json"}
    req = urllib.request.Request(OLLAMA_URL, data=data, headers=headers)
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=600) as resp:
        body = json.loads(resp.read())
    dt = time.perf_counter() - t0
    text = body.get("message", {}).get("content", "")
    # defensively strip any <think>...</think> block if the model emitted one
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    stats = {
        "seconds": round(dt, 1),
        "eval_count": body.get("eval_count"),
        "tok_per_s": round((body.get("eval_count") or 0) / dt, 1) if dt else None,
    }
    return text, stats


def main() -> None:
    user = f"MEETING TRANSCRIPT:\n{TRANSCRIPT}\n\n{INSTRUCTION}"
    print(f"== Meetings smoke test — model: {MODEL} ==\n")
    try:
        note, stats = ask(MODEL, SYSTEM, user)
    except urllib.error.URLError as e:
        raise SystemExit(
            f"Could not reach Ollama at {OLLAMA_URL} ({e}).\n"
            "Is Ollama running? Start it, then re-run."
        )
    print(note)
    print("\n---")
    print(f"stats: {stats['seconds']}s | {stats['eval_count']} tokens | {stats['tok_per_s']} tok/s")

    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, f"meetings_smoke_{MODEL.replace(':', '_').replace('/', '_')}.md")
    with open(out, "w") as fh:
        fh.write(f"# Meetings smoke test - {MODEL}\n\n{note}\n\n---\nstats: {stats}\n")
    print(f"\nsaved -> {out}")


if __name__ == "__main__":
    main()
