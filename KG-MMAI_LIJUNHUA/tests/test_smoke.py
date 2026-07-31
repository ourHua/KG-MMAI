"""Small deterministic checks for the KG-MMAI experiment code.

Author: LIJUNHUA
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

__author__ = "LIJUNHUA"

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "code"))

from kge_core import (  # noqa: E402
    MODELS,
    N_E,
    N_R,
    TAG,
    evaluate,
    te,
    train_epochs,
)


class ExperimentSmokeTests(unittest.TestCase):
    def test_graph_and_split_sizes(self) -> None:
        self.assertEqual(N_E, 1905)
        self.assertEqual(N_R, 5)
        counts = {name: int(np.sum(TAG == name)) for name in ("train", "valid", "test")}
        self.assertEqual(counts, {"train": 7772, "valid": 886, "test": 886})

    def test_each_model_runs_for_one_epoch(self) -> None:
        for model_name in MODELS:
            with self.subTest(model=model_name):
                model = train_epochs(model_name, seed=42, epochs=1)
                metrics = evaluate(model, te[:20])
                self.assertTrue(np.isfinite(metrics["MRR"]))
                self.assertGreaterEqual(metrics["MRR"], 0.0)
                self.assertLessEqual(metrics["MRR"], 1.0)
                self.assertEqual(metrics["queries"], 40)


if __name__ == "__main__":
    unittest.main()
