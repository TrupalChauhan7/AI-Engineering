#!/usr/bin/env python3
"""
fetch_tofueval.py — download + INSPECT the TofuEval MeetingBank subset.

Phase-2 step: get the verifier's second-domain evaluation data in place and
REPORT its real structure so we can build the baseline harness against the
actual column names (not assumptions).

What it does:
  1. Downloads TofuEval's MeetingBank annotation files from the GitHub repo
     (factual_consistency + completeness + the dev/test doc-id split) into
     data/tofueval/.
  2. Loads them with pandas and prints columns, row counts, a sample row,
     and the yes/no label balance.
  3. Loads the MeetingBank transcripts from HuggingFace (lytang/MeetingBank-
     transcript) and reports its columns + how well its ids overlap the
     TofuEval doc_ids (so we know the join works before writing the harness).

TofuEval rule: evaluation ONLY — its annotations must not train any model.

RUN (from the project folder):
    python scripts/fetch_tofueval.py
Deps: pandas (already in the env). For the MeetingBank overlap check also:
    pip install datasets
"""

import json
import os
import urllib.request

OUT = "data/tofueval"
API = "https://api.github.com/repos/amazon-science/tofueval/contents"
RAW_ROOT = "https://raw.githubusercontent.com/amazon-science/tofueval/main"


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "clarion-dev"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def gh_list(subdir: str) -> list[dict]:
    """List files in a repo subdirectory via the GitHub contents API."""
    try:
        return json.loads(_get(f"{API}/{subdir}"))
    except Exception as e:  # noqa: BLE001
        print(f"  ! could not list {subdir}: {e}")
        return []


def fetch_meetingbank_files() -> list[str]:
    os.makedirs(OUT, exist_ok=True)
    saved = []
    # the dev/test split map lives at the repo root
    try:
        dest = os.path.join(OUT, "document_ids_dev_test_split.json")
        open(dest, "wb").write(_get(f"{RAW_ROOT}/document_ids_dev_test_split.json"))
        saved.append(dest)
        print(f"  saved {dest}")
    except Exception as e:  # noqa: BLE001
        print(f"  ! document_ids_dev_test_split.json: {e}")
    # meetingbank annotation files in these two dirs
    for subdir in ("factual_consistency", "completeness"):
        for item in gh_list(subdir):
            name = item.get("name", "")
            if "meetingbank" in name.lower() and item.get("download_url"):
                dest = os.path.join(OUT, name)
                try:
                    open(dest, "wb").write(_get(item["download_url"]))
                    saved.append(dest)
                    print(f"  saved {dest}")
                except Exception as e:  # noqa: BLE001
                    print(f"  ! {name}: {e}")
    return saved


def inspect_csvs(saved: list[str]) -> set[str]:
    import pandas as pd

    doc_ids: set[str] = set()
    for path in saved:
        if not path.endswith(".csv"):
            continue
        print(f"\n=== {os.path.basename(path)} ===")
        df = pd.read_csv(path)
        print(f"  rows: {len(df)}  columns: {list(df.columns)}")
        if "doc_id" in df.columns:
            ids = set(df["doc_id"].astype(str))
            doc_ids |= ids
            print(f"  unique doc_id: {len(ids)}")
        for lbl in ("sent_label", "label"):
            if lbl in df.columns:
                print(f"  {lbl} balance: {df[lbl].value_counts().to_dict()}")
        print("  sample row:")
        print(df.head(1).to_dict("records")[0])
    return doc_ids


def inspect_meetingbank(doc_ids: set[str]) -> None:
    print("\n=== MeetingBank transcripts (HuggingFace: lytang/MeetingBank-transcript) ===")
    try:
        from datasets import load_dataset
    except ImportError:
        print("  (datasets not installed — run `pip install datasets` to check the join)")
        return
    ds = load_dataset("lytang/MeetingBank-transcript")
    print(f"  splits: {list(ds.keys())}")
    split = next(iter(ds.keys()))
    cols = ds[split].column_names
    print(f"  columns: {cols}")
    print(f"  rows in '{split}': {ds[split].num_rows}")
    print("  sample row keys/values (truncated):")
    row = ds[split][0]
    for k, v in row.items():
        sv = str(v)
        print(f"    {k}: {sv[:120]}{'...' if len(sv) > 120 else ''}")
    # overlap check across candidate id columns
    print("\n  doc_id overlap vs TofuEval:")
    for cand in ("meeting_id", "id", "uid", "doc_id"):
        if cand in cols:
            mb_ids = {str(x) for x in ds[split][cand]}
            hit = len(doc_ids & mb_ids)
            print(f"    column '{cand}': {hit}/{len(doc_ids)} TofuEval doc_ids matched")


def main() -> None:
    print("Fetching TofuEval MeetingBank annotations ->", OUT)
    saved = fetch_meetingbank_files()
    if not saved:
        raise SystemExit("No files fetched — check network / repo path.")
    doc_ids = inspect_csvs(saved)
    inspect_meetingbank(doc_ids)
    print("\nDone. Paste this whole output back so the harness is built on the real schema.")


if __name__ == "__main__":
    main()
