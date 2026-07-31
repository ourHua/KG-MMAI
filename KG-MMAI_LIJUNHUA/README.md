# KG-MMAI Experiment Package

**Author:** LIJUNHUA  
**Repository purpose:** Reproduce the knowledge-graph experiments, statistical analyses, and figures reported in the accompanying manuscript.

This repository contains the processed graph tables, analysis scripts, numerical results, and publication figures for the KG-MMAI study. The workflow is CPU-only and uses standard scientific Python packages. Each script has a focused responsibility so that structural analysis, link prediction, statistical testing, and figure generation can be checked separately.

## Repository layout

```text
KG-MMAI_LIJUNHUA/
├── code/                 Experiment scripts and shared KGE implementation
├── data/                 Processed node and edge tables
├── results/              Tables, cached layout, and machine-readable outputs
├── figures/              PNG and PDF figures
├── tests/                Deterministic smoke tests
├── tools/                Manifest maintenance utility
├── run_experiments.py    Command-line runner for the workflow
├── GITHUB_UPLOAD_GUIDE_CN.md  Chinese upload instructions for ourHua
├── requirements.txt      Python dependencies
├── CITATION.cff          Citation metadata
└── MANIFEST_SHA256.csv   File sizes and SHA-256 checksums
```

## Environment

Python 3.10 or newer is recommended.

```bash
python -m venv .venv
source .venv/bin/activate          # macOS or Linux
# .venv\Scripts\activate           # Windows PowerShell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

The main dependencies are NumPy, pandas, SciPy, NetworkX, Matplotlib, and pypinyin. A GPU is not required.

## Running the experiments

Run the complete workflow from the repository root:

```bash
python run_experiments.py
```

Run selected stages only:

```bash
python run_experiments.py --steps structure figures-structure
python run_experiments.py --steps link-prediction robustness statistics figures-results
```

Run the deterministic smoke tests after installation:

```bash
python -m unittest discover -s tests -v
```

The original scripts can also be executed directly:

```bash
python code/01_structural_analysis.py
python code/02_link_prediction.py
python code/03_ranking_robustness.py
python code/04_statistics.py
python code/05_figures_structure.py
python code/06_figures_results.py
```

## Script overview

| Script | Main responsibility | Key outputs |
|---|---|---|
| `01_structural_analysis.py` | Graph structure, threshold sensitivity, components, and degree statistics | structural and distribution tables |
| `02_link_prediction.py` | Typed and filtered link-prediction experiment | data split, per-seed metrics, training curves |
| `kge_core.py` | Shared data preparation, KGE scoring, training, and evaluation | imported by Scripts 02–04 |
| `03_ranking_robustness.py` | Model-ranking stability under two training budgets | robustness tables and relation difficulty |
| `04_statistics.py` | Pairwise tests, effect sizes, bootstrap intervals, and small-sample precision | statistical result tables |
| `05_figures_structure.py` | Structural figures | Figures 1–5 |
| `06_figures_results.py` | Model and robustness figures | Figures 6–9 |
| `figstyle.py` | Shared plotting settings and text-overlap checks | reusable figure utilities |
| `labels.py`, `labels_en.py` | Traceable English and pinyin labels for TCM entities | entity label tables |

## Core data checks

The structural script recomputes the main graph statistics from `data/nodes.csv` and `data/edges.csv`:

| Quantity | Recomputed value |
|---|---:|
| Total entities | 8,024 |
| Candidate relations | 48,566 |
| Core entities at weight ≥ 2 | 1,905 |
| Core relations at weight ≥ 2 | 9,544 |
| Largest core component | 99.48% |
| Core components | 6 |
| Single-occurrence relations | 80.35% |
| Schema violations | 0 |
| Duplicate triples | 0 |
| Train / validation / test | 7,772 / 886 / 886 |

## Link-prediction protocol

The included NumPy implementation uses:

- a relation-stratified 80/10/10 split with entity-coverage repair;
- typed negative sampling with eight negatives per positive triple;
- 64-dimensional embeddings;
- seeds 42, 1337, and 2024;
- typed, filtered ranking for both head and tail prediction;
- TransE, DistMult, ComplEx, and RotatE scoring functions.

The repository distinguishes two experimental configurations. Configuration A refers to the earlier PyTorch workflow described in the manuscript and is not included here. Configuration B is the NumPy workflow in `code/02_link_prediction.py` and `code/03_ranking_robustness.py`. Because the training objectives differ, the two configurations should be interpreted as a robustness comparison rather than as numerically identical implementations.

## Figure validation

`figstyle.py` includes a text-overlap check. Each plotting script renders the figure, evaluates visible text bounding boxes, and reports potential overlaps before saving PNG and PDF versions. The large graph map stores its deterministic node layout in `results/graph_layout_seed7.csv`, which makes later figure regeneration substantially faster.

## Entity labels

Herbs and prescriptions are displayed in pinyin. Symptoms, signs, pathogenesis terms, pulse qualities, and treatment actions use curated English glosses where available. Unmapped terms fall back to automatic transliteration. Entity IDs remain the stable identifiers, and the complete label mapping is stored in `results/entity_labels.csv`.

## Data and research-ethics note

The raw source corpus is not distributed in this repository. The package contains processed node and edge tables that are sufficient for the reported graph analyses. Before public release or journal submission, the author should retain documentation for source permissions, data provenance, redistribution rights, and any required ethics or institutional review determination.

## Reproducibility

`MANIFEST_SHA256.csv` records the byte size and SHA-256 checksum of every released file. Regenerate it after changing any tracked file:

```bash
python tools/update_manifest.py
python tools/verify_manifest.py
```

## Citation

Use the metadata in `CITATION.cff`. A general citation can be written as:

> LIJUNHUA. *KG-MMAI Experiment Package: Knowledge-Graph Experiment Reproduction Code and Data*. 2026.

## Authorship and use

The repository is maintained under the author name **LIJUNHUA**. No software license has been selected in this package; add an appropriate license before encouraging third-party reuse.
