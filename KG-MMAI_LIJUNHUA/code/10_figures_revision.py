#!/usr/bin/env python3
"""Regenerate the revised-manuscript Figures 6--8.

The figure-specific implementations live in ``code/figures``.  This numbered
script is kept as a workflow entry point for older commands and simply runs the
three canonical figure scripts in manuscript order.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

__author__ = "LIJUNHUA"

HERE = Path(__file__).resolve().parent
FIGURE_DIR = HERE / "figures"
SCRIPTS = (
    "fig06_annotation_sensitivity.py",
    "fig07_objective_ablation.py",
    "fig08_relation_lift_exact.py",
)


def main() -> int:
    for name in SCRIPTS:
        script = FIGURE_DIR / name
        if not script.is_file():
            print(f"missing figure script: {script}", file=sys.stderr)
            return 1
        completed = subprocess.run([sys.executable, str(script)], check=False)
        if completed.returncode:
            return completed.returncode
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
