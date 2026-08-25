#!/usr/bin/env python3
"""
download_audio.py  --  Fetch the PriMock57 audio (.wav) files.

WHY THIS EXISTS
---------------
The 114 audio files (~2 GB) are stored with "Git LFS" (Large File Storage).
In the repo they are just tiny *pointer* text files, e.g.:

    version https://git-lfs.github.com/spec/v1
    oid sha256:a93ca69f...        <- unique fingerprint of the real file
    size 14653484                 <- real file size in bytes

This script reads each pointer, asks GitHub's LFS server for a temporary
download link (the "batch API"), downloads the real .wav, and checks the
fingerprint matches. No git-lfs install, no Homebrew needed -- just Python 3.

USAGE (from inside the primock57 folder):
    python3 download_audio.py

Already-downloaded files are skipped, so it is safe to re-run if it stops.
"""

import hashlib
import json
import os
import sys
import urllib.request

REPO_LFS = "https://github.com/babylonhealth/primock57.git/info/lfs/objects/batch"
BATCH_SIZE = 50  # how many links to request per API call
HEADERS = {
    "Accept": "application/vnd.git-lfs+json",
    "Content-Type": "application/vnd.git-lfs+json",
}


def is_pointer(path):
    """Return (oid, size) if 'path' is still an LFS pointer, else None."""
    try:
        with open(path, "r", errors="ignore") as fh:
            head = fh.read(200)
    except (UnicodeDecodeError, OSError):
        return None
    if not head.startswith("version https://git-lfs"):
        return None
    oid = size = None
    for line in head.splitlines():
        if line.startswith("oid sha256:"):
            oid = line.split(":")[1].strip()
        elif line.startswith("size"):
            size = int(line.split()[1])
    return (oid, size) if oid and size else None


def find_pointers(root="."):
    out = []
    for folder, _, files in os.walk(root):
        if os.sep + ".git" in folder:
            continue
        for name in files:
            if name.endswith(".wav"):
                p = os.path.join(folder, name)
                info = is_pointer(p)
                if info:
                    out.append((p, info[0], info[1]))
    return out


def get_links(batch):
    """Ask the LFS server for download URLs for a batch of (path, oid, size)."""
    objects = [{"oid": oid, "size": size} for _, oid, size in batch]
    payload = json.dumps(
        {"operation": "download", "transfers": ["basic"], "objects": objects}
    ).encode()
    req = urllib.request.Request(REPO_LFS, data=payload, headers=HEADERS)
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.load(resp)
    return {o["oid"]: o for o in data["objects"]}


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    pointers = find_pointers(".")
    if not pointers:
        print(
            "No LFS pointer files found. Audio may already be downloaded. "
            "Run this from inside the primock57 folder."
        )
        return

    total_gb = sum(s for _, _, s in pointers) / 1e9
    print(f"Found {len(pointers)} audio files to download (~{total_gb:.1f} GB).\n")

    done = 0
    for i in range(0, len(pointers), BATCH_SIZE):
        batch = pointers[i : i + BATCH_SIZE]
        links = get_links(batch)
        for path, oid, size in batch:
            done += 1
            obj = links.get(oid, {})
            href = obj.get("actions", {}).get("download", {}).get("href")
            if not href:
                print(f"[{done}/{len(pointers)}] SKIP (no link): {path}")
                continue
            tmp = path + ".part"
            try:
                urllib.request.urlretrieve(href, tmp)
                got = sha256(tmp)
                if got != oid:
                    os.remove(tmp)
                    print(f"[{done}/{len(pointers)}] FAILED checksum: {path}")
                    continue
                os.replace(tmp, path)
                print(f"[{done}/{len(pointers)}] OK  {path}  ({size/1e6:.1f} MB)")
            except Exception as e:
                if os.path.exists(tmp):
                    os.remove(tmp)
                print(f"[{done}/{len(pointers)}] ERROR {path}: {e}")

    print("\nDone. Re-run this script if any file failed -- finished ones are skipped.")


if __name__ == "__main__":
    sys.exit(main())
