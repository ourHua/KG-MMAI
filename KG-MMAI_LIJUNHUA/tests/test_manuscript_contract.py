"""Static release-contract checks for the revised IJASC manuscript."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent


def load_runner():
    spec = importlib.util.spec_from_file_location("kgmmai_runner", ROOT / "run_experiments.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ManuscriptContractTests(unittest.TestCase):
    def test_final_figure_numbering(self):
        runner = load_runner()
        self.assertEqual(len(runner.MANUSCRIPT_FIGURES), 10)
        self.assertEqual(runner.MANUSCRIPT_FIGURES[5], "fig06_annotation_sensitivity")
        self.assertEqual(runner.MANUSCRIPT_FIGURES[6], "fig07_objective_ablation")
        self.assertEqual(runner.MANUSCRIPT_FIGURES[7], "fig08_relation_lift_exact")
        self.assertEqual(runner.MANUSCRIPT_FIGURES[8], "fig09_graph_map")
        self.assertEqual(runner.MANUSCRIPT_FIGURES[9], "fig10_kgmmai_design")

    def test_revision_scripts_present(self):
        for number in range(7, 15):
            matches = list((ROOT / "code").glob(f"{number:02d}_*.py"))
            self.assertTrue(matches, f"missing Script {number:02d}")

    def test_reference_ablation_orderings(self):
        path = ROOT / "results/manuscript_reference/objective_ablation_60ep.csv"
        table = pd.read_csv(path)
        expected = {
            "margin": ["DistMult", "TransE", "ComplEx", "RotatE"],
            "logistic": ["RotatE", "TransE", "DistMult", "ComplEx"],
            "selfadv": ["RotatE", "TransE", "DistMult", "ComplEx"],
        }
        for objective, ordering in expected.items():
            sub = table[table.objective == objective].sort_values("rank_60ep")
            self.assertEqual(sub.model.tolist(), ordering)

    def test_reference_structural_fingerprint(self):
        path = ROOT / "results/manuscript_reference/annotation_sensitivity_structure.csv"
        table = pd.read_csv(path).set_index("condition")
        s0 = table.loc["S0_as_annotated"]
        self.assertEqual(int(s0.unique_entities), 8024)
        self.assertEqual(int(s0.candidate_triples), 48566)
        self.assertEqual(int(s0.core_entities), 1905)
        self.assertEqual(int(s0.core_triples), 9544)
        self.assertAlmostEqual(float(s0.largest_component_pct), 99.48, places=2)


if __name__ == "__main__":
    unittest.main()
