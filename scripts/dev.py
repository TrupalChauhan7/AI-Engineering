#!/usr/bin/env python3
"""Cross-platform demo launcher — Windows, macOS, Linux.

Starts the FastAPI backend (:8000) and the Next.js front end (:3000) together, so
`python scripts/dev.py` works everywhere the Unix-only `scripts/dev.sh` used to.
Press Ctrl-C to stop both.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "app" / "web"
NPM = "npm.cmd" if os.name == "nt" else "npm"  # Windows npm is npm.cmd


def main() -> None:
    if shutil.which(NPM) is None or shutil.which("node") is None:
        sys.exit("Node.js 20+ not found on PATH — install it, then re-run (see README).")
    if not (WEB / "node_modules").exists():
        print("-> installing web dependencies (first run)…")
        subprocess.run([NPM, "install"], cwd=WEB, check=True)

    print("-> API  http://127.0.0.1:8000")
    api = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "app.api.main:app", "--port", "8000", "--reload"],
        cwd=ROOT,
    )
    print("-> Web  http://localhost:3000")
    web = subprocess.Popen([NPM, "run", "dev"], cwd=WEB)
    import time

    try:
        while api.poll() is None and web.poll() is None:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        for p in (web, api):
            if p.poll() is None:
                p.terminate()
        print("\nstopped.")


if __name__ == "__main__":
    main()
