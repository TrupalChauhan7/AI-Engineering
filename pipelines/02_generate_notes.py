"""Pipeline step 2 — generate SOAP notes from transcripts.

Run:  python pipelines/02_generate_notes.py   (or: make generate)
"""

from s2n.config import load_config


def main():
    cfg = load_config()
    print("TODO: transcript + prompt", cfg["generation"]["prompt_version"], "-> notes")


if __name__ == "__main__":
    main()
