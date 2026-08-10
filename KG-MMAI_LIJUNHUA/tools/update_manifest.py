#!/usr/bin/env python3
"""Regenerate the SHA-256 manifest for the public release.

The authorised local raw corpus (data/train.txt) is deliberately excluded.
"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

__author__ = "LIJUNHUA"

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "MANIFEST_SHA256.csv"
EXCLUDED_DIRS = {".git", ".venv", "venv", "__pycache__", "logs"}
EXCLUDED_FILES = {"data/train.txt", "MANIFEST_SHA256.csv"}


def sha256(path, chunk_size=1024 * 1024):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tracked_files():
    files = []
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(ROOT).as_posix()
        if rel in EXCLUDED_FILES:
            continue
        if any(part in EXCLUDED_DIRS for part in path.relative_to(ROOT).parts):
            continue
        files.append(path)
    return sorted(files, key=lambda p: p.relative_to(ROOT).as_posix())


def main():
    files = tracked_files()
    with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path", "bytes", "sha256"])
        for path in files:
            writer.writerow([
                path.relative_to(ROOT).as_posix(),
                path.stat().st_size,
                sha256(path),
            ])
    print(f"Updated {OUTPUT.relative_to(ROOT)} ({len(files)} files)")


if __name__ == "__main__":
    main()
