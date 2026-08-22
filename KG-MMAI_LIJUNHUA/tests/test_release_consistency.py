"""Small regression checks for release-level naming and metadata."""

from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_CONDITIONS = (
    "S0_as_annotated",
    "S1_adjudicated",
    "S2_majority_harmonised",
)
OLD_S1_CONDITION = "S1_" + "expert_corrected"


def test_old_s1_name_is_absent():
    checked = []
    for pattern in ("*.py", "*.csv", "*.md", "*.cff"):
        checked.extend(ROOT.rglob(pattern))
    offenders = []
    for path in checked:
        if any(part in {".git", ".venv", "venv", "__pycache__"} for part in path.parts):
            continue
        if OLD_S1_CONDITION in path.read_text(encoding="utf-8", errors="ignore"):
            offenders.append(path.relative_to(ROOT).as_posix())
    assert not offenders, f"old S1 condition name remains in: {offenders}"


def test_runner_uses_adjudicated_s1_edges():
    text = (ROOT / "run_experiments.py").read_text(encoding="utf-8")
    assert "results/sensitivity/edges_S1_adjudicated.csv" in text


def test_figure6_and_reference_table_use_same_conditions():
    figure_text = (ROOT / "code/figures/fig06_annotation_sensitivity.py").read_text(encoding="utf-8")
    for condition in EXPECTED_CONDITIONS:
        assert condition in figure_text

    path = ROOT / "results/manuscript_reference/annotation_sensitivity_structure.csv"
    with path.open(newline="", encoding="utf-8") as handle:
        conditions = tuple(row["condition"] for row in csv.DictReader(handle))
    assert conditions == EXPECTED_CONDITIONS


def test_release_metadata_files_exist():
    required = (
        "LICENSE",
        "PROVENANCE.md",
        "CITATION.cff",
        "MANIFEST_SHA256.csv",
        "tools/verify_manifest.py",
    )
    assert all((ROOT / name).is_file() for name in required)


def test_citation_metadata_uses_cff_12_fields():
    text = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    assert "cff-version: 1.2.0" in text
    assert "date-released:" in text
    assert "version: \"1.2.0\"" in text
    assert "license: MIT" in text
    assert not any(line.startswith("year:") for line in text.splitlines())


def test_manifest_covers_release_files():
    excluded_dirs = {
        ".git", ".idea", ".pytest_cache", ".venv", "venv", "__pycache__", "logs"
    }
    excluded_files = {
        ".DS_Store", "data/train.txt", "MANIFEST_SHA256.csv", "results/revision_claims.csv"
    }

    actual = set()
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT)
        rel = relative.as_posix()
        if rel in excluded_files or any(part in excluded_dirs for part in relative.parts):
            continue
        actual.add(rel)

    with (ROOT / "MANIFEST_SHA256.csv").open(newline="", encoding="utf-8") as handle:
        listed = {row["path"] for row in csv.DictReader(handle)}
    assert listed == actual
