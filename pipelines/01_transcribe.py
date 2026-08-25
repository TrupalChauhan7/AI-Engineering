"""Pipeline step 1 — transcribe all audio with Whisper (RQ2).

Run:  python pipelines/01_transcribe.py   (or: make transcribe)

Reads settings from config/config.yaml, writes transcripts to results/.
"""

from s2n.config import load_config


def main():
    cfg = load_config()
    print("TODO: loop audio ->", cfg["transcription"]["model"], "-> results/transcripts/")


if __name__ == "__main__":
    main()
