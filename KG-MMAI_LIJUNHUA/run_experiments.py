#!/usr/bin/env python3
"""Run the KG-MMAI workflow in the order used by the revised manuscript.

Modes
-----
python run_experiments.py
    Public-release workflow. Recomputes all experiments that do not require the
    withheld raw BIO corpus, regenerates available figures, generates the
    conceptual Figure 10, and verifies manuscript figure assets.

python run_experiments.py --full-local
    Full end-to-end workflow. Requires an authorised local data/train.txt,
    rebuilds S0/S1/S2, reruns link prediction on all three graphs, and applies
    strict manuscript-alignment checks.

python run_experiments.py --figures-only
    Regenerate figures from result tables already present.

The final manuscript uses Figures 1--10. Historical diagnostic plots created by
Script 06 are redirected to figures/supplementary/ by figstyle.py.
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
    Step("link-prediction", "code/02_link_prediction.py", "typed filtered link prediction"),
    Step("robustness", "code/03_ranking_robustness.py", "training-budget robustness"),
    Step("statistics-original", "code/04_statistics.py", "legacy/original statistical tables"),
    Step("figures-structure", "code/05_figures_structure.py", "manuscript Figures 1--4"),
    Step("figures-results", "code/06_figures_results.py", "manuscript Figures 5 and 9 + supplementary diagnostics"),
    Step("objective-ablation", "code/07_objective_ablation.py", "72-run controlled objective ablation"),
    Step("annotation-sensitivity", "code/08_annotation_sensitivity.py", "S0/S1/S2 annotation audit"),
    Step("statistics-revised", "code/09_statistics_revised.py", "triple-level clustered inference and exact baseline"),
    Step("figures-revision", "code/10_figures_revision.py", "manuscript Figures 6--8"),
    Step("revision-audit", "code/12_revision_audit.py", "manuscript-to-result consistency gate"),
    Step("figure-design", "code/13_figure_design.py", "manuscript Figure 10 design specification"),
)
STEP_BY_NAME = {step.name: step for step in STEPS}

SENSITIVITY_GRAPHS = {
    "S0": "results/sensitivity/edges_S0_as_annotated.csv",
    "S1": "results/sensitivity/edges_S1_expert_corrected.csv",
    "S2": "results/sensitivity/edges_S2_majority_harmonised.csv",
}

MANUSCRIPT_FIGURES = (
    "fig01_schema",
    "fig02_extraction_funnel",
    "fig03_relation_composition",
    "fig04_threshold_sensitivity",
    "fig05_degree_structure",
    "fig06_annotation_sensitivity",
    "fig07_objective_ablation",
    "fig08_relation_lift_exact",
    "fig09_graph_map",
    "fig10_kgmmai_design",
)
FORMATS = ("png", "pdf")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--public", action="store_true",
                      help="Run the public-release workflow (default).")
    mode.add_argument("--full-local", action="store_true",
                      help="Require local data/train.txt and regenerate S0/S1/S2 end to end.")
    mode.add_argument("--revision", action="store_true",
                      help="Run revision analyses; use raw corpus if locally available.")
    mode.add_argument("--original", action="store_true",
                      help="Run original structural/KGE analyses and Figures 1--5,9.")
    mode.add_argument("--figures-only", action="store_true",
                      help="Regenerate figures from existing result tables.")
    mode.add_argument("--steps", nargs="+", choices=tuple(STEP_BY_NAME),
                      help="Run only selected named stages; prerequisites are not added.")
    parser.add_argument("--continue-on-error", action="store_true",
                        help="Continue after failures; final exit status remains non-zero.")
    parser.add_argument("--skip-figure-check", action="store_true",
                        help="Do not verify manuscript PNG/PDF assets at the end.")
    return parser.parse_args()


def run_command(name, description, command):
    started = time.perf_counter()
    print(f"\n=== {name} ===\n{description}")
    completed = subprocess.run(command, cwd=ROOT, check=False)
    elapsed = time.perf_counter() - started
    if completed.returncode == 0:
        print(f"[{name}] completed in {elapsed:.1f} s")
        return True
    print(f"[{name}] FAILED ({completed.returncode}) after {elapsed:.1f} s",
          file=sys.stderr)
    return False


def run_step(name, extra=None):
    step = STEP_BY_NAME[name]
    script = ROOT / step.script
    if not script.is_file():
        print(f"[ERROR] missing script: {script}", file=sys.stderr)
        return False
    command = [sys.executable, str(script)]
    if extra:
        command.extend(extra)
    return run_command(step.name, step.description, command)


def run_sensitivity_linkpred(condition, relative_edges):
    script = ROOT / "code" / "11_sensitivity_linkpred.py"
    edge_path = ROOT / relative_edges
    if not script.is_file() or not edge_path.is_file():
        print(f"[ERROR] missing sensitivity input for {condition}: {edge_path}",
              file=sys.stderr)
        return False
    return run_command(
        f"sensitivity-linkpred-{condition}",
        f"60-epoch O3 link prediction on {condition}",
        [sys.executable, str(script), "--condition", condition,
         "--edges", str(edge_path)],
    )


def verify_figures(stems=MANUSCRIPT_FIGURES):
    missing = []
    print("\n=== figure-check ===")
    for stem in stems:
        paths = [FIG / f"{stem}.{ext}" for ext in FORMATS]
        bad = [p for p in paths if not p.is_file() or p.stat().st_size == 0]
        if bad:
            missing.extend(bad)
            print(f"MISS {stem}: " + ", ".join(p.name for p in bad))
        else:
            print(f"OK   {stem}.png / .pdf")
    if missing:
        print(f"[figure-check] {len(missing)} required asset(s) missing.", file=sys.stderr)
        return False
    print(f"[figure-check] verified {len(stems)} manuscript figures / "
          f"{len(stems)*len(FORMATS)} files.")
    return True


def run_sequence(names, continue_on_error=False):
    any_failed = False
    for name in names:
        ok = run_step(name)
        if not ok:
            any_failed = True
            if not continue_on_error:
                return False
    return not any_failed


def public_workflow(continue_on_error=False):
    # Does not invoke Script 08 because the public release intentionally omits
    # the raw corpus. Figure 6 remains a committed publication asset unless the
    # source-derived sensitivity tables are locally available.
    names = (
        "structure", "link-prediction", "robustness", "statistics-original",
        "objective-ablation", "statistics-revised",
        "figures-structure", "figures-results", "figures-revision",
        "figure-design", "revision-audit",
    )
    return run_sequence(names, continue_on_error)


def revision_workflow(full_local=False, continue_on_error=False):
    failed = False
    raw = ROOT / "data" / "train.txt"

    if full_local or raw.is_file():
        if not raw.is_file():
            print(
                "[ERROR] --full-local requires an authorised data/train.txt. "
                "The public release intentionally excludes it.",
                file=sys.stderr,
            )
            return False
        if not run_step("annotation-sensitivity", ["--strict-manuscript"]):
            if not continue_on_error:
                return False
            failed = True
    else:
        print(
            "\n[annotation-sensitivity] SKIPPED: data/train.txt is intentionally "
            "absent from the public release. The committed Figure 6 remains "
            "available; use --full-local with an authorised copy for end-to-end regeneration."
        )

    for name in ("objective-ablation", "statistics-revised"):
        if not run_step(name):
            if not continue_on_error:
                return False
            failed = True

    if raw.is_file():
        for condition, edges in SENSITIVITY_GRAPHS.items():
            if not run_sensitivity_linkpred(condition, edges):
                if not continue_on_error:
                    return False
                failed = True

    for name in ("figures-revision", "figure-design"):
        if not run_step(name):
            if not continue_on_error:
                return False
            failed = True

    audit_args = ["--strict-sensitivity"] if raw.is_file() else []
    if not run_step("revision-audit", audit_args):
        if not continue_on_error:
            return False
        failed = True
    return not failed


def main():
    args = parse_args()
    print(f"KG-MMAI manuscript workflow | author: {__author__}")

    if args.steps:
        ok = run_sequence(args.steps, args.continue_on_error)
        check = True
    elif args.figures_only:
        ok = run_sequence(
            ("figures-structure", "figures-results", "figures-revision", "figure-design"),
            args.continue_on_error,
        )
        check = True
    elif args.original:
        ok = run_sequence(
            ("structure", "link-prediction", "robustness", "statistics-original",
             "figures-structure", "figures-results"),
            args.continue_on_error,
        )
        check = False
        if not args.skip_figure_check:
            original_stems = MANUSCRIPT_FIGURES[:5] + ("fig09_graph_map",)
            check = verify_figures(original_stems)
    elif args.full_local:
        ok = run_sequence(
            ("structure", "link-prediction", "robustness", "statistics-original"),
            args.continue_on_error,
        )
        if ok or args.continue_on_error:
            ok = revision_workflow(True, args.continue_on_error) and ok
        if ok or args.continue_on_error:
            ok = run_sequence(("figures-structure", "figures-results"),
                              args.continue_on_error) and ok
        check = True
    elif args.revision:
        ok = revision_workflow(False, args.continue_on_error)
        check = True
    else:
        ok = public_workflow(args.continue_on_error)
        check = True

    if check and not args.skip_figure_check:
        ok = verify_figures() and ok
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
