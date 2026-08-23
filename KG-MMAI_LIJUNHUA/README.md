# KG-MMAI experiment package

This folder contains the code, result tables and figure scripts used for the revised KG-MMAI study. The maintained code is attributed to **LIJUNHUA**. Analysis and plotting are kept separate so that each reported result can be traced to the script that produced it.

The experiments cover the induced TCM knowledge graph, annotation sensitivity and knowledge-graph embedding performance. Figure 10 is a design diagram for the proposed multimodal extension. It should not be read as evidence that a complete multimodal diagnostic system was trained or clinically validated in this study.

## Experiment scripts

The numbered scripts are the main workflow entry points:

- `code/01_structural_analysis.py` — graph structure and threshold analysis
- `code/02_link_prediction.py` — typed, filtered link prediction
- `code/03_ranking_robustness.py` — ranking robustness checks
- `code/04_statistics.py` — original statistical outputs retained for record keeping
- `code/07_objective_ablation.py` — controlled objective ablation
- `code/08_annotation_sensitivity.py` — S0/S1/S2 annotation sensitivity analysis
- `code/09_statistics_revised.py` — triple-level tests, clustered bootstrap and exact random-ranking baseline
- `code/11_sensitivity_linkpred.py` — link prediction on the rebuilt sensitivity graphs
- `code/12_revision_audit.py` — checks key manuscript values against available outputs

## Figure scripts

The manuscript figures are generated from `code/figures/`:

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

To regenerate all ten figures:

```bash
python run_figures.py
```

`code/10_figures_revision.py` remains for compatibility with the earlier workflow and calls the Figure 6–8 scripts above. `FIGURE_MAP.md` lists the figure-to-script mapping.

## Data source

The raw BIO-tagged `train.txt` is not redistributed in this repository. The analysis used the training split of the public TCMNER dataset. The exact file used in the experiments is identified in [`PROVENANCE.md`](PROVENANCE.md) by source, byte size and SHA-256 digest.

If you already have an authorised copy of the corpus, place it at `data/train.txt` and run:

```bash
python run_experiments.py --full-local
```

For the public package, use:

```bash
python run_experiments.py --public
```

The public workflow uses the archived manuscript-reference tables where the raw corpus is required for source-level rebuilding.

## Release checks

Before making a release, run:

```bash
python tools/verify_manifest.py
python -m pytest tests/ -q
python code/12_revision_audit.py
```

`MANIFEST_SHA256.csv` is for checking a fresh copy of the release. If any tracked release file changes, regenerate the manifest after all other edits are complete:

```bash
python tools/update_manifest.py
```

## Licence

Code is released under the MIT License. Derived tables and figures are released under CC BY 4.0. The upstream raw corpus is not included and is not covered by either licence. See [`LICENSE`](LICENSE) for the scope of each licence.

## Author

Code maintainer: **LIJUNHUA**.
