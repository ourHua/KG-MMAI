# KG-MMAI experiment package

This directory contains the code and release artefacts used for the revised
KG-MMAI knowledge-layer study.  The maintained code is attributed to
**LIJUNHUA**.  Analysis and plotting are kept as separate steps so a reported
result can be traced back to the script that produced it.

The study evaluates the induced TCM knowledge graph, annotation sensitivity,
and knowledge-graph embedding behaviour.  Figure 10 is an architecture diagram
for the proposed multimodal extension; it is not evidence of a trained or
clinically validated multimodal diagnostic system.

## Experiment scripts

The numbered analysis scripts remain the workflow entry points:

- `code/01_structural_analysis.py` — structural profile and threshold analysis
- `code/02_link_prediction.py` — typed, filtered link-prediction benchmark
- `code/03_ranking_robustness.py` — ranking robustness checks
- `code/04_statistics.py` — original statistical outputs retained for provenance
- `code/07_objective_ablation.py` — controlled objective ablation
- `code/08_annotation_sensitivity.py` — S0/S1/S2 annotation sensitivity analysis
- `code/09_statistics_revised.py` — triple-level statistics, clustered bootstrap, and exact random-ranking baseline
- `code/11_sensitivity_linkpred.py` — link prediction on the rebuilt sensitivity graphs
- `code/12_revision_audit.py` — checks headline manuscript values against regenerated outputs

## One figure, one script

The canonical manuscript figure code is in `code/figures/`:

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

Regenerate all ten figures with:

```bash
python run_figures.py
```

`code/10_figures_revision.py` is retained as a compatibility entry point and
now delegates Figures 6–8 to those canonical scripts.  See `FIGURE_MAP.md` for
the full figure-to-script mapping.

## Data and provenance

The raw BIO-tagged `train.txt` is not redistributed.  The study used the
training split of the public TCMNER dataset, and the exact analysed copy is
identified by file size and SHA-256 digest in [`PROVENANCE.md`](PROVENANCE.md).
That record also gives the upstream repository and the reproducible corpus
fingerprint used in the manuscript.

Researchers who already have an authorised copy may place it at
`data/train.txt` and run:

```bash
python run_experiments.py --full-local
```

The public workflow does not require the raw corpus:

```bash
python run_experiments.py --public
```

## Release checks

The repository includes three lightweight release checks:

```bash
python tools/verify_manifest.py
python -m pytest tests/ -q
python code/12_revision_audit.py
```

`MANIFEST_SHA256.csv` records the released files and is intended to verify a
freshly downloaded copy.  If tracked release files are changed, regenerate the
manifest only after the other edits are finished:

```bash
python tools/update_manifest.py
```

## Licence

Code is released under the MIT License.  Derived tables and figures are
released under CC BY 4.0.  The upstream raw corpus is not included and is not
covered by either grant.  See [`LICENSE`](LICENSE) for the exact scope.

## Author

Code maintainer: **LIJUNHUA**.
