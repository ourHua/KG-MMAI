# KG-MMAI experiment package

This directory contains the reproducibility code for the revised KG-MMAI knowledge-layer study. The maintained code is attributed to **LIJUNHUA**. The analysis scripts and plotting scripts are kept separate so that each reported result can be traced to a specific step.

The study evaluates the induced TCM knowledge graph, annotation sensitivity and knowledge-graph embedding behaviour. Figure 10 is an intended architecture diagram only; it is not evidence that the multimodal diagnostic model has already been trained or clinically validated.

## Experiment scripts

The existing numbered analysis scripts remain the experiment entry points:

- `code/01_structural_analysis.py` — structural profile and threshold analysis.
- `code/02_link_prediction.py` — typed, filtered link-prediction benchmark.
- `code/03_ranking_robustness.py` — ranking robustness checks.
- `code/04_statistics.py` — original statistical outputs retained for provenance.
- `code/07_objective_ablation.py` — controlled objective ablation.
- `code/08_annotation_sensitivity.py` — S0/S1/S2 annotation sensitivity analysis.
- `code/09_statistics_revised.py` — triple-level statistics, clustered bootstrap and exact random-ranking baseline.
- `code/11_sensitivity_linkpred.py` — link prediction on the corrected graphs.
- `code/12_revision_audit.py` — checks headline manuscript values against regenerated outputs.

## One figure = one Python script

The canonical figure code is now in `code/figures/`. Each manuscript figure has one dedicated entry script and writes only its own PNG/PDF pair.

```text
code/figures/
├── fig01_schema.py
├── fig02_extraction_funnel.py
├── fig03_relation_composition.py
├── fig04_threshold_sensitivity.py
├── fig05_degree_structure.py
├── fig06_annotation_sensitivity.py
├── fig07_objective_ablation.py
├── fig08_relation_lift_exact.py
├── fig09_graph_map.py
└── fig10_kgmmai_design.py
```

Run all ten figure scripts with:

```bash
python run_figures.py
```

See `FIGURE_MAP.md` for the figure-to-script mapping.

## Data boundary

The raw BIO-tagged `train.txt` is not redistributed because its original redistribution licence has not been established. Derived graph tables and manuscript-reference result tables remain in the repository. If an authorised local copy of the corpus is available, the annotation-sensitivity workflow can rebuild the corresponding graph variants from source.

## Validation

The reorganised figure code was compiled and the repository smoke tests were run locally before submission. The figure scripts were also executed independently to confirm that all ten PNG/PDF pairs can be regenerated.

## Author

Code maintainer: **LIJUNHUA**.
