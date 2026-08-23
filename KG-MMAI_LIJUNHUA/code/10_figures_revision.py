#!/usr/bin/env python3
"""Generate revised-manuscript Figures 6--8.

The actual plotting code is kept in ``code/figures``. This script remains as a
numbered workflow entry point and runs the three figure scripts in manuscript
order.
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
