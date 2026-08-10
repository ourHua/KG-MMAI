#!/usr/bin/env python3
"""Run the KG-MMAI workflow used by the revised IJASC manuscript.

Default: public-release workflow (no raw BIO corpus required).
Use ``--full-local`` with an authorised ``data/train.txt`` for source-level
S0/S1/S2 reconstruction and strict sensitivity validation.
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
    Step("figures-results", "code/06_figures_results.py", "Figures 5 and 9 plus supplementary plots"),
    Step("objective-ablation", "code/07_objective_ablation.py", "72-run controlled objective ablation"),
    Step("annotation-sensitivity", "code/08_annotation_sensitivity.py", "S0/S1/S2 annotation audit"),
    Step("statistics-revised", "code/09_statistics_revised.py", "triple-level clustered inference"),
    Step("figures-revision", "code/10_figures_revision.py", "manuscript Figures 6--8"),
    Step("revision-audit", "code/12_revision_audit.py", "manuscript/result consistency gate"),
    Step("figure-design", "code/13_figure_design.py", "manuscript Figure 10 design"),
    Step("figure-formats", "code/14_ensure_figure_formats.py", "complete PNG/PDF figure pairs"),
)
STEP_BY_NAME = {s.name: s for s in STEPS}

SENSITIVITY_GRAPHS = {
    "S0": "results/sensitivity/edges_S0_as_annotated.csv",
    "S1": "results/sensitivity/edges_S1_expert_corrected.csv",
    "S2": "results/sensitivity/edges_S2_majority_harmonised.csv",
}
MANUSCRIPT_FIGURES = (
    "fig01_schema", "fig02_extraction_funnel", "fig03_relation_composition",
    "fig04_threshold_sensitivity", "fig05_degree_structure",
    "fig06_annotation_sensitivity", "fig07_objective_ablation",
    "fig08_relation_lift_exact", "fig09_graph_map", "fig10_kgmmai_design",
)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--public", action="store_true", help="public workflow (default)")
    mode.add_argument("--full-local", action="store_true", help="require local data/train.txt")
    mode.add_argument("--revision", action="store_true", help="revision analyses and figures")
    mode.add_argument("--original", action="store_true", help="original structural/KGE workflow")
    mode.add_argument("--figures-only", action="store_true", help="regenerate figures from existing results")
    mode.add_argument("--steps", nargs="+", choices=tuple(STEP_BY_NAME))
    p.add_argument("--continue-on-error", action="store_true")
    p.add_argument("--skip-figure-check", action="store_true")
    return p.parse_args()


def run_command(name, description, command):
    started = time.perf_counter()
    print(f"\n=== {name} ===\n{description}")
    done = subprocess.run(command, cwd=ROOT, check=False)
    elapsed = time.perf_counter() - started
    if done.returncode == 0:
        print(f"[{name}] completed in {elapsed:.1f} s")
        return True
    print(f"[{name}] FAILED ({done.returncode}) after {elapsed:.1f} s", file=sys.stderr)
    return False


def run_step(name, extra=None):
    step = STEP_BY_NAME[name]
    script = ROOT / step.script
    if not script.is_file():
        print(f"[ERROR] missing script: {script}", file=sys.stderr)
        return False
    cmd = [sys.executable, str(script)] + (extra or [])
    return run_command(step.name, step.description, cmd)


def run_sequence(names, continue_on_error=False):
    failed = False
    for name in names:
        if not run_step(name):
            failed = True
            if not continue_on_error:
                return False
    return not failed


def run_sensitivity_linkpred(condition, rel_edges):
    script = ROOT / "code/11_sensitivity_linkpred.py"
    edges = ROOT / rel_edges
    if not script.is_file() or not edges.is_file():
        print(f"[ERROR] missing sensitivity input: {edges}", file=sys.stderr)
        return False
    return run_command(
        f"sensitivity-linkpred-{condition}",
        f"60-epoch O3 link prediction on {condition}",
        [sys.executable, str(script), "--condition", condition, "--edges", str(edges)],
    )


def verify_figures(stems=MANUSCRIPT_FIGURES):
    print("\n=== figure-check ===")
    missing = []
    for stem in stems:
        bad = []
        for ext in ("png", "pdf"):
            path = FIG / f"{stem}.{ext}"
            if not path.is_file() or path.stat().st_size == 0:
                bad.append(path.name); missing.append(path)
        print(("MISS " if bad else "OK   ") + stem + (": " + ", ".join(bad) if bad else ""))
    if missing:
        print(f"[figure-check] {len(missing)} required asset(s) missing", file=sys.stderr)
        return False
    print(f"[figure-check] verified 10 manuscript figures / 20 files")
    return True


def revision_workflow(full_local=False, continue_on_error=False):
    failed = False
    raw = ROOT / "data/train.txt"
    if full_local or raw.is_file():
        if not raw.is_file():
            print("[ERROR] --full-local requires an authorised data/train.txt", file=sys.stderr)
            return False
        if not run_step("annotation-sensitivity", ["--strict-manuscript"]):
            if not continue_on_error: return False
            failed = True
    else:
        print("\n[annotation-sensitivity] skipped: raw corpus is intentionally absent from public release")

    for name in ("objective-ablation", "statistics-revised"):
        if not run_step(name):
            if not continue_on_error: return False
            failed = True

    if raw.is_file():
        for condition, edges in SENSITIVITY_GRAPHS.items():
            if not run_sensitivity_linkpred(condition, edges):
                if not continue_on_error: return False
                failed = True

    for name in ("figures-revision", "figure-design", "figure-formats"):
        if not run_step(name):
            if not continue_on_error: return False
            failed = True

    audit_args = ["--strict-sensitivity"] if raw.is_file() else []
    if not run_step("revision-audit", audit_args):
        if not continue_on_error: return False
        failed = True
    return not failed


def public_workflow(continue_on_error=False):
    # Script 08 is omitted because the public branch intentionally withholds the
    # raw BIO corpus. The distributed Figure 6 PNG is retained and Script 14
    # creates a PDF wrapper when a native source-regenerated PDF is unavailable.
    return run_sequence((
        "structure", "link-prediction", "robustness", "statistics-original",
        "objective-ablation", "statistics-revised", "figures-structure",
        "figures-results", "figures-revision", "figure-design",
        "figure-formats", "revision-audit",
    ), continue_on_error)


def main():
    args = parse_args()
    print(f"KG-MMAI manuscript workflow | author: {__author__}")

    if args.steps:
        ok = run_sequence(args.steps, args.continue_on_error)
        needs_check = True
    elif args.figures_only:
        ok = run_sequence(("figures-structure", "figures-results", "figures-revision",
                           "figure-design", "figure-formats"), args.continue_on_error)
        needs_check = True
    elif args.original:
        ok = run_sequence(("structure", "link-prediction", "robustness", "statistics-original",
                           "figures-structure", "figures-results"), args.continue_on_error)
        needs_check = False
        if not args.skip_figure_check:
            ok = verify_figures(MANUSCRIPT_FIGURES[:5] + ("fig09_graph_map",)) and ok
    elif args.full_local:
        ok = run_sequence(("structure", "link-prediction", "robustness", "statistics-original"),
                          args.continue_on_error)
        if ok or args.continue_on_error:
            ok = revision_workflow(True, args.continue_on_error) and ok
        if ok or args.continue_on_error:
            ok = run_sequence(("figures-structure", "figures-results", "figure-formats"),
                              args.continue_on_error) and ok
        needs_check = True
    elif args.revision:
        ok = revision_workflow(False, args.continue_on_error)
        needs_check = True
    else:
        ok = public_workflow(args.continue_on_error)
        needs_check = True

    if needs_check and not args.skip_figure_check:
        ok = verify_figures() and ok
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
