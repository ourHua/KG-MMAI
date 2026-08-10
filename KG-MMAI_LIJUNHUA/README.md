# KG-MMAI Experiment Package

**Author:** LIJUNHUA  
**Purpose:** Reproduce the knowledge-graph construction, link-prediction experiments, statistical analyses, reviewer-requested sensitivity analyses, and publication figures for the KG-MMAI manuscript.

The repository is organised as a reproducible scientific workflow. The analysis scripts generate machine-readable result tables first; the plotting scripts then regenerate the publication figures from those results. The top-level `run_experiments.py` coordinates the dependency order and checks that every expected PNG/PDF figure has actually been written.

## Repository layout

```text
KG-MMAI_LIJUNHUA/
├── code/                    Analysis, KGE, statistics, and figure scripts
├── data/                    Graph tables and the local BIO corpus input
├── results/                 Machine-readable experiment outputs
├── figures/                 Publication figures in PNG and PDF
├── tests/                   Deterministic smoke tests
├── tools/                   Manifest update / verification utilities
├── run_experiments.py       Workflow runner and figure-output verifier
├── requirements.txt         Python dependencies
├── CITATION.cff             Citation metadata
└── MANIFEST_SHA256.csv      File sizes and SHA-256 checksums
```

## Environment

Python 3.10 or newer is recommended. The workflow is CPU-only; a GPU is not required.

```bash
python -m venv .venv
source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\Activate.ps1       # Windows PowerShell

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Core dependencies include NumPy, pandas, SciPy, NetworkX, Matplotlib, and pypinyin.

## Recommended: reproduce everything

From the `KG-MMAI_LIJUNHUA` directory, run:

```bash
python run_experiments.py
```

With no mode flag, the runner executes the **complete manuscript workflow**:

1. structural analysis;
2. original link prediction;
3. ranking robustness;
4. original statistical analysis;
5. annotation-sensitivity reconstruction;
6. controlled objective ablation;
7. revised triple-level statistical analysis;
8. S0/S1/S2 corrected-graph link-prediction reruns;
9. all three figure-generation scripts;
10. an explicit existence/size check for every expected PNG and PDF.

The equivalent explicit command is:

```bash
python run_experiments.py --all
```

A successful complete run finishes with a `[figure-check]` block confirming all expected figure files.

## Other execution modes

Run only the original Scripts 01-06 and verify Figures 01-09:

```bash
python run_experiments.py --original
```

Run only the reviewer-requested revision experiments and revised figures:

```bash
python run_experiments.py --revision
```

Regenerate the figure files from result tables that already exist:

```bash
python run_experiments.py --figures-only
```

Run selected stages only:

```bash
python run_experiments.py --steps structure figures-structure
python run_experiments.py --steps link-prediction robustness statistics figures-results
python run_experiments.py --steps objective-ablation statistics-revised figures-revision
```

`--steps` does **not** add prerequisite stages automatically. Use it only when the required input/result files are already present.

To continue after an error while still returning a non-zero final status:

```bash
python run_experiments.py --all --continue-on-error
```

## Script map and dependencies

| Script | Responsibility | Principal outputs / consumers |
|---|---|---|
| `01_structural_analysis.py` | Core-graph structure, threshold sensitivity, components, degree statistics | Structural tables used by Scripts 05-06 |
| `02_link_prediction.py` | Typed, filtered link prediction | Split/metrics used by later analyses |
| `03_ranking_robustness.py` | Ranking stability across training budgets | Robustness curves and relation-difficulty tables |
| `04_statistics.py` | Original pairwise tests, effect sizes, bootstrap intervals, small-sample precision | Tables used by Script 06 |
| `05_figures_structure.py` | Structural visualisation | `fig01`-`fig04` |
| `06_figures_results.py` | Degree structure, model robustness, relation difficulty, precision, graph map | `fig05`-`fig09` |
| `07_objective_ablation.py` | Controlled O1/O2/O3 objective ablation in one code base | `results/ablation/*` |
| `08_annotation_sensitivity.py` | S0/S1/S2 annotation audit and graph rebuilding | `results/sensitivity/*` |
| `09_statistics_revised.py` | Triple-level inference, clustered bootstrap, Holm adjustment, exact random baseline | `results/statistics/*` |
| `10_figures_revision.py` | Revision figures | canonical `fig10`-`fig12` aliases plus revised-manuscript aliases |
| `11_sensitivity_linkpred.py` | Re-run primary link prediction on S0/S1/S2 corrected graphs | `results/sensitivity/linkpred_*` |
| `kge_core.py` | Shared KGE data preparation, scoring, training, and evaluation | Imported by link-prediction scripts |
| `figstyle.py` | Shared plotting style, PNG/PDF saving, text-overlap reporting | Imported by figure scripts |
| `labels.py`, `labels_en.py` | Traceable Latin-script / English entity labels | Used in publication figures |

## Figure outputs

The release keeps a simple canonical sequence `fig01`-`fig12`. Every canonical figure is expected in **both PNG and PDF**:

| Canonical stem | Generated by |
|---|---|
| `fig01_schema` | Script 05 |
| `fig02_extraction_funnel` | Script 05 |
| `fig03_relation_composition` | Script 05 |
| `fig04_threshold_sensitivity` | Script 05 |
| `fig05_degree_structure` | Script 06 |
| `fig06_ranking_robustness` | Script 06 |
| `fig07_relation_difficulty` | Script 06 |
| `fig08_small_sample` | Script 06 |
| `fig09_graph_map` | Script 06 |
| `fig10_annotation_sensitivity` | Script 10 |
| `fig11_objective_ablation` | Script 10 |
| `fig12_relation_lift_exact` | Script 10 |

Script 10 also writes three **intentional aliases** that follow the numbering used inside the revised manuscript:

```text
fig06_annotation_sensitivity.png / .pdf
fig07_objective_ablation.png / .pdf
fig08_relation_lift_exact.png / .pdf
```

Therefore a complete run verifies **15 figure stems / 30 files**: the 12 canonical assets plus the 3 revised-manuscript aliases.

### Why both PNG and PDF?

- **PNG** is convenient for GitHub preview, Word insertion, and rapid visual checking.
- **PDF** preserves vector graphics for publication-quality export.

Scripts 05 and 06 save through `figstyle.save_checked()`, which renders the canvas, reports possible text overlaps, and then writes both formats. Script 10 likewise writes both PNG and PDF for every revised figure and alias.

## Core graph checks

`01_structural_analysis.py` recomputes the principal graph quantities from `data/nodes.csv` and `data/edges.csv`:

| Quantity | Recomputed value |
|---|---:|
| Total entities | 8,024 |
| Candidate relations | 48,566 |
| Core entities at weight >= 2 | 1,905 |
| Core relations at weight >= 2 | 9,544 |
| Largest core component | 99.48% |
| Core components | 6 |
| Single-occurrence relations | 80.35% |
| Schema violations | 0 |
| Duplicate triples | 0 |
| Train / validation / test | 7,772 / 886 / 886 |

These values provide a fast structural fingerprint before the more expensive embedding experiments are run.

## Link-prediction protocol

The shared KGE implementation uses:

- relation-stratified 80/10/10 splitting with entity-coverage repair;
- typed negative sampling with eight negatives per positive triple;
- 64-dimensional embeddings;
- seeds 42, 1337, and 2024;
- typed, filtered head and tail ranking;
- TransE, DistMult, ComplEx, and RotatE scoring functions.

The controlled revision experiment in Script 07 keeps the code base, split, sampler, seeds, optimiser settings, and evaluation protocol fixed while varying only the objective:

- **O1:** pairwise margin-ranking loss;
- **O2:** binary logistic loss with uniform negative weights;
- **O3:** binary logistic loss with self-adversarial negative weights.

## Annotation-sensitivity workflow

Script 08 rebuilds the graph under three conditions:

- **S0:** as annotated;
- **S1:** expert correction of the specified PRE/HER collisions;
- **S2:** majority harmonisation of multi-type surface forms.

Script 11 is then invoked once for each rebuilt graph. `run_experiments.py` passes the corresponding edge table explicitly with `--condition` and `--edges`, ensuring that S0/S1/S2 are actually trained and evaluated on different corrected graphs rather than silently falling back to `data/edges.csv`.

## Raw corpus and release policy

The annotation audit in Script 08 requires:

```text
data/train.txt
```

The runner checks for this file before `--revision` or the complete workflow begins.

**Important for archival/public release:** the code requirement and the redistribution policy are separate issues. Before creating a GitHub Release or Zenodo version, confirm that you have the right to redistribute `data/train.txt`. If redistribution is not permitted, remove it from the public release and keep an authorised local copy at the same path when reproducing the revision workflow. The processed `nodes.csv` and `edges.csv` remain sufficient for the original graph analyses.

## Figure-only regeneration

After all analysis result files have been generated once, publication graphics can be recreated without retraining:

```bash
python run_experiments.py --figures-only
```

This runs Scripts 05, 06, and 10 and then checks every expected PNG/PDF output. If a prerequisite CSV is missing, the responsible plotting script fails instead of silently producing a partial figure set.

## Smoke tests

Run:

```bash
python -m unittest discover -s tests -v
```

These tests are intended as quick deterministic checks; they are not a replacement for re-running the full experimental workflow.

## Reproducibility manifest

After changing tracked code, results, or figures, regenerate and verify the manifest:

```bash
python tools/update_manifest.py
python tools/verify_manifest.py
```

Do this **after** the final experiment/figure run so the checksums correspond to the files actually archived.

## Suggested final-release sequence

```bash
python -m unittest discover -s tests -v
python run_experiments.py --all
python tools/update_manifest.py
python tools/verify_manifest.py
```

Then inspect `figures/` visually, commit the regenerated results/figures, create the GitHub release, and archive that exact release on Zenodo.

## Citation

Use the metadata in `CITATION.cff`. A general software citation can be written as:

> LIJUNHUA. *KG-MMAI Experiment Package: Knowledge-Graph Experiment Reproduction Code and Data*. 2026.

## Authorship

Repository maintenance and experiment-package authorship are recorded under **LIJUNHUA**.
