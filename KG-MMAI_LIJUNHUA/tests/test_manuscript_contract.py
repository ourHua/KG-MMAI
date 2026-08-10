"""Static release-contract checks for the revised IJASC manuscript."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import run_experiments as runner  # noqa: E402


class ManuscriptContractTests(unittest.TestCase):
    def test_final_figure_numbering(self):
        self.assertEqual(len(runner.MANUSCRIPT_FIGURES), 10)
        self.assertEqual(runner.MANUSCRIPT_FIGURES[5:], (
            "fig06_annotation_sensitivity",
            "fig07_objective_ablation",
            "fig08_relation_lift_exact",
            "fig09_graph_map",
            "fig10_kgmmai_design",
        ))

    def test_revision_scripts_present(self):
        for number in range(7, 15):
            matches = list((ROOT / "code").glob(f"{number:02d}_*.py"))
            self.assertTrue(matches, f"missing Script {number:02d}")

    def test_reference_ablation_orderings(self):
        table = pd.read_csv(ROOT / "results/manuscript_reference/objective_ablation_60ep.csv")
        expected = {
            "margin": ["DistMult", "TransE", "ComplEx", "RotatE"],
            "logistic": ["RotatE", "TransE", "DistMult", "ComplEx"],
            "selfadv": ["RotatE", "TransE", "DistMult", "ComplEx"],
        }
        for objective, ordering in expected.items():
            sub = table[table.objective == objective].sort_values("rank_60ep")
            self.assertEqual(sub.model.tolist(), ordering)

    def test_reference_structural_fingerprint(self):
        table = pd.read_csv(
            ROOT / "results/manuscript_reference/annotation_sensitivity_structure.csv"
        ).set_index("condition")
        s0 = table.loc["S0_as_annotated"]
        self.assertEqual(int(s0.unique_entities), 8024)
        self.assertEqual(int(s0.candidate_triples), 48566)
        self.assertEqual(int(s0.core_entities), 1905)
        self.assertEqual(int(s0.core_triples), 9544)
        self.assertAlmostEqual(float(s0.largest_component_pct), 99.48, places=2)


if __name__ == "__main__":
    unittest.main()
