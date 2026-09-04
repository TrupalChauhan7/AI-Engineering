#!/bin/bash
cd "$(dirname "$0")"

echo "=================================================="
echo "  FM-05 Speech-to-Clinical-Note - Mac setup"
echo "  Run this once to install everything and launch."
echo "=================================================="
echo ""

if ! command -v brew >/dev/null 2>&1; then
  echo "Homebrew is needed to install ffmpeg + Ollama."
  echo "Install it from https://brew.sh then run this file again."
  exit 1
fi

echo "[1/5] ffmpeg..."
command -v ffmpeg >/dev/null 2>&1 || brew install ffmpeg
echo "    ffmpeg ready."
echo ""

echo "[2/5] Ollama..."
command -v ollama >/dev/null 2>&1 || brew install ollama
echo "    Ollama ready."
echo ""

echo "[3/5] Starting the Ollama engine in the background..."
ollama serve >/dev/null 2>&1 &
sleep 5
echo ""

echo "[4/5] Downloading the 3 AI models (large; only fetches what is missing)..."
ollama pull medgemma:4b
ollama pull llama3.1:8b
ollama pull atla/selene-mini
echo ""

echo "[5/5] Python environment + project install..."
[ -d .venv ] || python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
echo ""

echo "Launching the demo at http://localhost:3000 ..."
python scripts/dev.py
