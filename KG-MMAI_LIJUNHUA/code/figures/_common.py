"""Shared figure utilities. Every manuscript figure still has its own entry script."""
from __future__ import annotations
__author__ = "LIJUNHUA"
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CODE = ROOT / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))
DATA = ROOT / "data"
RES = ROOT / "results"
FIG = ROOT / "figures"
REF = RES / "manuscript_reference"
FIG.mkdir(parents=True, exist_ok=True)

from figstyle import *  # noqa: F401,F403,E402

def choose(machine: Path, reference: Path | None = None) -> Path:
    if machine.is_file(): return machine
    if reference is not None and reference.is_file():
        print(f"[reference fallback] {reference.relative_to(ROOT)}")
        return reference
    raise FileNotFoundError(f"Missing required result: {machine}")
