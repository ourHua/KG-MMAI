#!/usr/bin/env python3
"""Run the KG-MMAI experiment workflow from the repository root.

Author: LIJUNHUA
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

__author__ = "LIJUNHUA"

ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class Step:
    name: str
    script: str
    description: str


STEPS = (
    Step("structure", "code/01_structural_analysis.py", "structural analysis"),
    Step("link-prediction", "code/02_link_prediction.py", "link prediction"),
    Step("robustness", "code/03_ranking_robustness.py", "ranking robustness"),
    Step("statistics", "code/04_statistics.py", "statistical analysis"),
    Step("figures-structure", "code/05_figures_structure.py", "structural figures"),
    Step("figures-results", "code/06_figures_results.py", "result figures"),
)
STEP_BY_NAME = {step.name: step for step in STEPS}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run all or selected KG-MMAI experiment stages."
    )
    parser.add_argument(
        "--steps",
        nargs="+",
        choices=tuple(STEP_BY_NAME),
        help="Run only the named stages. The default is the complete workflow.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue to later stages when one stage fails.",
    )
    return parser.parse_args()


def run_step(step: Step) -> bool:
    script = ROOT / step.script
    if not script.exists():
        print(f"[ERROR] Missing script: {script}", file=sys.stderr)
        return False

    started = time.perf_counter()
    print(f"\n[{step.name}] {step.description}")
    completed = subprocess.run([sys.executable, str(script)], cwd=ROOT, check=False)
    elapsed = time.perf_counter() - started

    if completed.returncode == 0:
        print(f"[{step.name}] completed in {elapsed:.1f} s")
        return True

    print(
        f"[{step.name}] failed with exit code {completed.returncode} after {elapsed:.1f} s",
        file=sys.stderr,
    )
    return False


def main() -> int:
    args = parse_args()
    selected = [STEP_BY_NAME[name] for name in args.steps] if args.steps else list(STEPS)

    print(f"KG-MMAI workflow | author: {__author__}")
    for step in selected:
        if not run_step(step) and not args.continue_on_error:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
