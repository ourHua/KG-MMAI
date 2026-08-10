#!/usr/bin/env python3
"""Run the KG-MMAI experiment workflow from the repository root.

``python run_experiments.py`` preserves the original released workflow.
``python run_experiments.py --revision`` runs the reviewer-requested revision
pipeline (Scripts 08, 07, 09, the three corrected-graph link-prediction runs,
and Script 10) with one command.

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
    Step("statistics", "code/04_statistics.py", "original statistical analysis"),
    Step("figures-structure", "code/05_figures_structure.py", "structural figures"),
    Step("figures-results", "code/06_figures_results.py", "original result figures"),
    Step("objective-ablation", "code/07_objective_ablation.py", "controlled objective ablation"),
    Step("annotation-sensitivity", "code/08_annotation_sensitivity.py", "annotation audit and graph correction"),
    Step("statistics-revised", "code/09_statistics_revised.py", "triple-level clustered statistics"),
    Step("figures-revision", "code/10_figures_revision.py", "revision figures"),
)
STEP_BY_NAME = {step.name: step for step in STEPS}

REVISION_PRE_STEPS = (
    "annotation-sensitivity",
    "objective-ablation",
    "statistics-revised",
)
REVISION_POST_STEPS = ("figures-revision",)

SENSITIVITY_GRAPHS = {
    "S0": "results/sensitivity/edges_S0_as_annotated.csv",
    "S1": "results/sensitivity/edges_S1_expert_corrected.csv",
    "S2": "results/sensitivity/edges_S2_majority_harmonised.csv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run KG-MMAI experiment stages.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--steps",
        nargs="+",
        choices=tuple(STEP_BY_NAME),
        help="Run only the named stages.",
    )
    group.add_argument(
        "--revision",
        action="store_true",
        help="Run the complete reviewer-requested revision workflow with one command.",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue to later stages when one stage fails.",
    )
    return parser.parse_args()


def run_command(name: str, description: str, command: list[str]) -> bool:
    started = time.perf_counter()
    print(f"\n[{name}] {description}")
    completed = subprocess.run(command, cwd=ROOT, check=False)
    elapsed = time.perf_counter() - started
    if completed.returncode == 0:
        print(f"[{name}] completed in {elapsed:.1f} s")
        return True
    print(
        f"[{name}] failed with exit code {completed.returncode} after {elapsed:.1f} s",
        file=sys.stderr,
    )
    return False


def run_step(step: Step) -> bool:
    script = ROOT / step.script
    if not script.exists():
        print(f"[ERROR] Missing script: {script}", file=sys.stderr)
        return False
    return run_command(step.name, step.description, [sys.executable, str(script)])


def run_sensitivity_linkpred(condition: str, relative_edges: str) -> bool:
    script = ROOT / "code" / "11_sensitivity_linkpred.py"
    edge_path = ROOT / relative_edges
    if not edge_path.exists():
        print(f"[ERROR] Missing corrected edge table: {edge_path}", file=sys.stderr)
        return False
    return run_command(
        f"sensitivity-linkpred-{condition}",
        f"primary link prediction on {condition}",
        [
            sys.executable,
            str(script),
            "--condition",
            condition,
            "--edges",
            str(edge_path),
        ],
    )


def run_revision(continue_on_error: bool) -> int:
    raw_corpus = ROOT / "data" / "train.txt"
    if not raw_corpus.exists():
        print(
            "[ERROR] The full revision workflow starts from data/train.txt, but the raw BIO "
            "corpus is intentionally not redistributed. Place an authorised local copy at "
            "KG-MMAI_LIJUNHUA/data/train.txt and rerun with --revision.",
            file=sys.stderr,
        )
        return 2

    for name in REVISION_PRE_STEPS:
        if not run_step(STEP_BY_NAME[name]) and not continue_on_error:
            return 1

    for condition, edges in SENSITIVITY_GRAPHS.items():
        if not run_sensitivity_linkpred(condition, edges) and not continue_on_error:
            return 1

    for name in REVISION_POST_STEPS:
        if not run_step(STEP_BY_NAME[name]) and not continue_on_error:
            return 1
    return 0


def main() -> int:
    args = parse_args()
    print(f"KG-MMAI workflow | author: {__author__}")

    if args.revision:
        return run_revision(args.continue_on_error)

    selected = [STEP_BY_NAME[name] for name in args.steps] if args.steps else list(STEPS[:6])
    for step in selected:
        if not run_step(step) and not args.continue_on_error:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
