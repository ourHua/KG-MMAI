#!/usr/bin/env python3
"""Verify a fresh release copy against ``MANIFEST_SHA256.csv``."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

__author__ = "LIJUNHUA"

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "MANIFEST_SHA256.csv"
EXCLUDED_DIRS = {
    ".git",
    ".idea",
    ".pytest_cache",
    ".venv",
    "venv",
    "__pycache__",
    "logs",
}
EXCLUDED_FILES = {
    ".DS_Store",
    "data/train.txt",
    "MANIFEST_SHA256.csv",
    "results/revision_claims.csv",
}


def sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def release_paths() -> set[str]:
    paths = set()
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT)
        rel = relative.as_posix()
        if rel in EXCLUDED_FILES:
            continue
        if any(part in EXCLUDED_DIRS for part in relative.parts):
            continue
        paths.add(rel)
    return paths


def main() -> int:
    failures: list[str] = []
    with MANIFEST.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    manifest_paths = {row["path"] for row in rows}
    actual_paths = release_paths()
    for path in sorted(actual_paths - manifest_paths):
        failures.append(f"unlisted release file: {path}")
    for path in sorted(manifest_paths - actual_paths):
        failures.append(f"manifest entry not present in release: {path}")

    for row in rows:
        path = ROOT / row["path"]
        if not path.exists():
            continue
        actual_size = path.stat().st_size
        if actual_size != int(row["bytes"]):
            failures.append(
                f"size mismatch: {row['path']} ({actual_size} != {row['bytes']})"
            )
            continue
        if sha256(path) != row["sha256"]:
            failures.append(f"hash mismatch: {row['path']}")

    if failures:
        print("Manifest verification failed:")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print(f"Manifest verification passed ({len(rows)} files).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
