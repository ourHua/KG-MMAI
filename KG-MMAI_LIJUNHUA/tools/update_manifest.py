#!/usr/bin/env python3
"""Regenerate the SHA-256 manifest for the repository.

Author: LIJUNHUA
"""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path

__author__ = "LIJUNHUA"

ROOT = Path(__file__).resolve().parent.parent
OUTPUT = ROOT / "MANIFEST_SHA256.csv"
EXCLUDED_DIRS = {".git", ".venv", "__pycache__"}


def sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def tracked_files() -> list[Path]:
    files: list[Path] = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or path == OUTPUT:
            continue
        if any(part in EXCLUDED_DIRS for part in path.relative_to(ROOT).parts):
            continue
        files.append(path)
    return sorted(files, key=lambda item: item.relative_to(ROOT).as_posix())


def main() -> None:
    with OUTPUT.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["path", "bytes", "sha256"])
        for path in tracked_files():
            writer.writerow([
                path.relative_to(ROOT).as_posix(),
                path.stat().st_size,
                sha256(path),
            ])
    print(f"Updated {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
