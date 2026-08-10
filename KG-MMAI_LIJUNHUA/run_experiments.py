#!/usr/bin/env python3
"""Orchestrate the complete KG-MMAI experiment and figure workflow.

Default behaviour:
    python run_experiments.py
runs the complete manuscript workflow and verifies that every expected PNG/PDF
figure has been written.

Useful alternatives:
    python run_experiments.py --original
    python run_experiments.py --revision
    python run_experiments.py --figures-only
    python run_experiments.py --steps structure figures-structure

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
FIG = ROOT / "figures"


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

ORIGINAL_ANALYSIS = (
    "structure",
    "link-prediction",
    "robustness",
    "statistics",
)
ORIGINAL_FIGURE_STEPS = (
    "figures-structure",
    "figures-results",
)
REVISION_ANALYSIS = (
    "annotation-sensitivity",
    "objective-ablation",
    "statistics-revised",
)
REVISION_FIGURE_STEPS = ("figures-revision",)

SENSITIVITY_GRAPHS = {
    "S0": "results/sensitivity/edges_S0_as_annotated.csv",
    "S1": "results/sensitivity/edges_S1_expert_corrected.csv",
    "S2": "results/sensitivity/edges_S2_majority_harmonised.csv",
}

CANONICAL_FIGURES = (
    "fig01_schema",
    "fig02_extraction_funnel",
    "fig03_relation_composition",
    "fig04_threshold_sensitivity",
    "fig05_degree_structure",
    "fig06_ranking_robustness",
    "fig07_relation_difficulty",
    "fig08_small_sample",
    "fig09_graph_map",
    "fig10_annotation_sensitivity",
    "fig11_objective_ablation",
    "fig12_relation_lift_exact",
)

REVISION_NUMBER_ALIASES = (
    "fig06_annotation_sensitivity",
    "fig07_objective_ablation",
    "fig08_relation_lift_exact",
)

ORIGINAL_FIGURES = CANONICAL_FIGURES[:9]
REVISION_FIGURES = CANONICAL_FIGURES[9:] + REVISION_NUMBER_ALIASES
ALL_FIGURES = CANONICAL_FIGURES + REVISION_NUMBER_ALIASES
FORMATS = ("png", "pdf")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run KG-MMAI analyses and regenerate publication figures. "
            "With no mode flag, the complete workflow is executed."
        )
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--steps",
        nargs="+",
        choices=tuple(STEP_BY_NAME),
        help="Run only the named stages; no implicit prerequisite stages are added.",
    )
    mode.add_argument(
        "--original",
        action="store_true",
        help="Run Scripts 01-06 and verify Figures 01-09.",
    )
    mode.add_argument(
        "--revision",
        action="store_true",
        help="Run reviewer-requested Scripts 07-11 and verify the revised figures.",
    )
    mode.add_argument(
        "--figures-only",
        action="store_true",
        help="Regenerate all figure groups from result files that already exist.",
    )
    mode.add_argument(
        "--all",
        action="store_true",
        help="Run the complete workflow (same as invoking the runner with no mode flag).",
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue to later stages after a failure; final exit status remains non-zero.",
    )
    return parser.parse_args()


def run_command(name: str, description: str, command: list[str]) -> bool:
    started = time.perf_counter()
    print(f"\n[{name}] {description}", flush=True)
    print("  $", " ".join(str(part) for part in command), flush=True)
    completed = subprocess.run(command, cwd=ROOT, check=False)
    elapsed = time.perf_counter() - started
    if completed.returncode == 0:
        print(f"[{name}] completed in {elapsed:.1f} s", flush=True)
        return True
    print(
        f"[{name}] FAILED with exit code {completed.returncode} after {elapsed:.1f} s",
        file=sys.stderr,
        flush=True,
    )
    return False


def run_step(step: Step) -> bool:
    script = ROOT / step.script
    if not script.exists():
        print(f"[ERROR] Missing script: {script}", file=sys.stderr)
        return False
    return run_command(step.name, step.description, [sys.executable, str(script)])


def run_named_steps(names: tuple[str, ...], continue_on_error: bool) -> bool:
    ok = True
    for name in names:
        passed = run_step(STEP_BY_NAME[name])
        ok = passed and ok
        if not passed and not continue_on_error:
            return False
    return ok


def require_raw_corpus() -> bool:
    raw_corpus = ROOT / "data" / "train.txt"
    if raw_corpus.exists():
        return True
    print(
        "[ERROR] The annotation-sensitivity workflow requires data/train.txt. "
        "Place an authorised local copy at KG-MMAI_LIJUNHUA/data/train.txt "
        "before running the full or revision workflow.",
        file=sys.stderr,
    )
    return False


def run_sensitivity_linkpred(condition: str, relative_edges: str) -> bool:
    script = ROOT / "code" / "11_sensitivity_linkpred.py"
    edge_path = ROOT / relative_edges
    if not script.exists():
        print(f"[ERROR] Missing script: {script}", file=sys.stderr)
        return False
    if not edge_path.exists():
        print(f"[ERROR] Missing corrected edge table: {edge_path}", file=sys.stderr)
        return False
    return run_command(
        f"sensitivity-linkpred-{condition}",
        f"primary link prediction on annotation condition {condition}",
        [
            sys.executable,
            str(script),
            "--condition",
            condition,
            "--edges",
            str(edge_path),
        ],
    )


def run_sensitivity_suite(continue_on_error: bool) -> bool:
    ok = True
    for condition, edges in SENSITIVITY_GRAPHS.items():
        passed = run_sensitivity_linkpred(condition, edges)
        ok = passed and ok
        if not passed and not continue_on_error:
            return False
    return ok


def verify_figures(stems: tuple[str, ...]) -> bool:
    missing = []
    for stem in stems:
        for ext in FORMATS:
            path = FIG / f"{stem}.{ext}"
            if not path.exists() or path.stat().st_size == 0:
                missing.append(path.relative_to(ROOT))

    print("\n[figure-check]")
    if missing:
        print("  Missing or empty expected figure files:", file=sys.stderr)
        for path in missing:
            print(f"    - {path}", file=sys.stderr)
        return False

    print(f"  verified {len(stems)} figure stems / {len(stems) * len(FORMATS)} files")
    for stem in stems:
        print(f"    OK  figures/{stem}.png + .pdf")
    return True


def run_original(continue_on_error: bool) -> bool:
    ok = run_named_steps(ORIGINAL_ANALYSIS, continue_on_error)
    if not ok and not continue_on_error:
        return False
    figures_ok = run_named_steps(ORIGINAL_FIGURE_STEPS, continue_on_error)
    ok = figures_ok and ok
    return verify_figures(ORIGINAL_FIGURES) and ok


def run_revision(continue_on_error: bool) -> bool:
    if not require_raw_corpus():
        return False

    ok = run_named_steps(REVISION_ANALYSIS, continue_on_error)
    if not ok and not continue_on_error:
        return False

    sensitivity_ok = run_sensitivity_suite(continue_on_error)
    ok = sensitivity_ok and ok
    if not sensitivity_ok and not continue_on_error:
        return False

    figures_ok = run_named_steps(REVISION_FIGURE_STEPS, continue_on_error)
    ok = figures_ok and ok
    return verify_figures(REVISION_FIGURES) and ok


def run_all(continue_on_error: bool) -> bool:
    """Run all analyses first, then render and verify every publication figure."""
    if not require_raw_corpus():
        return False

    ok = run_named_steps(ORIGINAL_ANALYSIS, continue_on_error)
    if not ok and not continue_on_error:
        return False

    revision_ok = run_named_steps(REVISION_ANALYSIS, continue_on_error)
    ok = revision_ok and ok
    if not revision_ok and not continue_on_error:
        return False

    sensitivity_ok = run_sensitivity_suite(continue_on_error)
    ok = sensitivity_ok and ok
    if not sensitivity_ok and not continue_on_error:
        return False

    figure_steps = ORIGINAL_FIGURE_STEPS + REVISION_FIGURE_STEPS
    figures_ok = run_named_steps(figure_steps, continue_on_error)
    ok = figures_ok and ok
    return verify_figures(ALL_FIGURES) and ok


def run_figures_only(continue_on_error: bool) -> bool:
    figure_steps = ORIGINAL_FIGURE_STEPS + REVISION_FIGURE_STEPS
    ok = run_named_steps(figure_steps, continue_on_error)
    return verify_figures(ALL_FIGURES) and ok


def main() -> int:
    args = parse_args()
    print(f"KG-MMAI workflow | author: {__author__}")
    print(f"repository root: {ROOT}")

    if args.steps:
        ok = run_named_steps(tuple(args.steps), args.continue_on_error)
        return 0 if ok else 1

    if args.original:
        ok = run_original(args.continue_on_error)
    elif args.revision:
        ok = run_revision(args.continue_on_error)
    elif args.figures_only:
        ok = run_figures_only(args.continue_on_error)
    else:
        ok = run_all(args.continue_on_error)

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
