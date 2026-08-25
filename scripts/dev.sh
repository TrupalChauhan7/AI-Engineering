#!/usr/bin/env bash
# Run the Clarion demo: FastAPI (:8000) + Next.js (:3000) together.
# Ctrl-C stops both.
set -euo pipefail
cd "$(dirname "$0")/.."

export PATH="$HOME/.local/node/bin:$PATH"
command -v node >/dev/null || { echo "node not found — see README 'Run the demo'"; exit 1; }

[ -d app/web/node_modules ] || (cd app/web && npm install)

cleanup() { kill 0 2>/dev/null || true; }
trap cleanup EXIT INT TERM

echo "→ API  http://127.0.0.1:8000"
uvicorn app.api.main:app --port 8000 --reload &

echo "→ Web  http://localhost:3000"
(cd app/web && npm run dev) &

wait
