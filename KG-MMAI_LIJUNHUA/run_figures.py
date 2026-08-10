#!/usr/bin/env python3
"""Render the ten figures used in the revised manuscript."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

__author__ = "LIJUNHUA"

ROOT = Path(__file__).resolve().parent
FIGURE_DIR = ROOT / "code" / "figures"

FIGURE_SCRIPTS = (
    "fig01_schema.py",
    "fig02_extraction_funnel.py",
    "fig03_relation_composition.py",
    "fig04_threshold_sensitivity.py",
    "fig05_degree_structure.py",
    "fig06_annotation_sensitivity.py",
    "fig07_objective_ablation.py",
    "fig08_relation_lift_exact.py",
    "fig09_graph_map.py",
    "fig10_kgmmai_design.py",
)


def main() -> None:
    for script_name in FIGURE_SCRIPTS:
        script_path = FIGURE_DIR / script_name
        print(f"\n=== {script_name} ===", flush=True)
        subprocess.run(
            [sys.executable, str(script_path)],
            cwd=ROOT,
            check=True,
        )

    print("\nAll ten manuscript figures were generated.")


if __name__ == "__main__":
    main()
