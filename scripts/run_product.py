#!/usr/bin/env python3
"""One command to run the Clarion product locally, with preflight checks.

    python scripts/run_product.py           # check everything, then launch
    python scripts/run_product.py --check   # preflight only, don't launch

WHY this exists on top of ``scripts/dev.py``: ``dev.py`` just starts the two
processes. This wraps it with the checks that turn "it crashed with a confusing
error" into "here is exactly what is missing and how to fix it" — is Ollama up,
are the models pulled, is Node present. It fixes what it safely can (pulls
missing models) and explains what it cannot.

The models run in Ollama on this machine; ``S2N_OLLAMA_HOST`` (default
http://localhost:11434) says where. Whisper downloads its own (small) weights on
first use — nothing to pre-pull.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
OLLAMA_HOST = os.environ.get("S2N_OLLAMA_HOST", "http://localhost:11434").rstrip("/")

# The Ollama models the product can use across both domains (clinical +
# meetings) plus the shared verifier. Whisper is NOT here — it is not an Ollama
# model and self-downloads.
REQUIRED_MODELS = ("medgemma:4b", "qwen3:14b", "llama3.1:8b")


def _ollama_tags() -> list[str] | None:
    """Model names Ollama reports, or None if Ollama is unreachable."""
    try:
        resp = requests.get(f"{OLLAMA_HOST}/api/tags", timeout=5)
        resp.raise_for_status()
        return [m["name"] for m in resp.json().get("models", [])]
    except requests.RequestException:
        return None


def _pull(model: str) -> bool:
    """Pull one model, streaming Ollama's progress. True on success."""
    print(f"   pulling {model} (first time only, this can take a while)…")
    return subprocess.run(["ollama", "pull", model]).returncode == 0


def preflight() -> bool:
    """Check Ollama + models + Node. Pull missing models. Return True if ready."""
    ok = True

    print(f"1) Ollama at {OLLAMA_HOST} …")
    tags = _ollama_tags()
    if tags is None:
        print("   ✗ not reachable. Start it with `ollama serve` (or open the Ollama app),")
        print("     then re-run. Install: https://ollama.com/download")
        return False  # nothing else can be checked without it
    print("   ✓ up")

    print("2) required models …")
    have = {t.split(":")[0]: t for t in tags}
    for model in REQUIRED_MODELS:
        base = model.split(":")[0]
        if model in tags or base in have:
            print(f"   ✓ {model}")
            continue
        if os.environ.get("S2N_OLLAMA_HOST", "").startswith("http://host.docker"):
            print(f"   ✗ {model} missing (cannot auto-pull a remote host) — pull it there")
            ok = False
        elif not _pull(model):
            print(f"   ✗ {model} failed to pull")
            ok = False

    print("3) Node.js (for the web UI) …")
    from shutil import which

    if which("node") and (which("npm") or which("npm.cmd")):
        print("   ✓ found")
    else:
        print("   ! Node.js 20+ not found — the API will still run, but the web UI won't.")
        print("     Install Node, or use the API directly at http://localhost:8000")

    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the Clarion product with preflight checks.")
    ap.add_argument("--check", action="store_true", help="run checks only, do not launch")
    args = ap.parse_args()

    print("Clarion preflight\n" + "-" * 40)
    ready = preflight()
    print("-" * 40)

    if not ready:
        print("Not ready — fix the ✗ items above, then re-run.")
        sys.exit(1)
    print("All set." + ("" if args.check else " Launching API + web…\n"))

    if args.check:
        return
    # hand off to the existing cross-platform launcher
    subprocess.run([sys.executable, str(ROOT / "scripts" / "dev.py")])


if __name__ == "__main__":
    main()
