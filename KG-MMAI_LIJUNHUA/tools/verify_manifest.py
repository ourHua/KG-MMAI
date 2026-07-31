#!/usr/bin/env python3
"""Verify repository files against MANIFEST_SHA256.csv.

Author: LIJUNHUA
"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

__author__ = "LIJUNHUA"

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "MANIFEST_SHA256.csv"


def sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    failures: list[str] = []
    with MANIFEST.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            path = ROOT / row["path"]
            if not path.exists():
                failures.append(f"missing: {row['path']}")
                continue
            actual_size = path.stat().st_size
            if actual_size != int(row["bytes"]):
                failures.append(
                    f"size mismatch: {row['path']} ({actual_size} != {row['bytes']})"
                )
                continue
            actual_hash = sha256(path)
            if actual_hash != row["sha256"]:
                failures.append(f"hash mismatch: {row['path']}")

    if failures:
        print("Manifest verification failed:")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print("Manifest verification passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
