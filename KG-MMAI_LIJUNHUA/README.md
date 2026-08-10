# KG-MMAI — Manuscript Reproducibility Package

**Author:** Junhua Li (LIJUNHUA)  
**Manuscript:** *Constructing and Auditing a Weakly Supervised Traditional Chinese Medicine Knowledge Graph: Structural Profile, Annotation Sensitivity, and a Controlled Analysis of Embedding-Model Selection*  
**Archived release DOI:** `10.5281/zenodo.21731543`

This directory is aligned to the **revised IJASC manuscript**. It contains the processed graph tables, deterministic KGE implementation, controlled objective ablation, statistical analysis, annotation-sensitivity code, manuscript figures, and release-audit utilities.

## Scientific scope

The code evaluates the **knowledge layer**, not a completed multimodal diagnostic system. The present graph contains five entity types (`SYM`, `CAU`, `PRE`, `HER`, `EFF`) and five directional relation types. It does **not** contain a syndrome entity type or a syndrome-adjacency relation. Consequently, the KG-MMAI multimodal architecture shown in Figure 10 is a **design specification only**; no multimodal classifier or knowledge-constraint term is implemented or evaluated here.

The structural fingerprint reported in the manuscript is:

| Quantity | Value |
|---|---:|
| BIO samples | 6,199 |
| Character tokens | 307,398 |
| Entity mentions | 41,262 |
| Unique type-name entities | 8,024 |
| Candidate relations | 48,566 |
| Core entities (`weight >= 2`) | 1,905 |
| Core relations (`weight >= 2`) | 9,544 |
| Largest core component | 99.48% |
| Validation / test triples | 886 / 886 |
| Ranking queries per seed | 1,772 |

## Repository layout

```text
KG-MMAI_LIJUNHUA/
├── code/
│   ├── 01_structural_analysis.py
│   ├── 02_link_prediction.py
│   ├── 03_ranking_robustness.py
│   ├── 04_statistics.py
│   ├── 05_figures_structure.py
│   ├── 06_figures_results.py
│   ├── 07_objective_ablation.py
│   ├── 08_annotation_sensitivity.py
│   ├── 09_statistics_revised.py
│   ├── 10_figures_revision.py
│   ├── 11_sensitivity_linkpred.py
│   ├── 12_revision_audit.py
│   ├── 13_figure_design.py
│   ├── kge_core.py
│   ├── figstyle.py
│   ├── labels.py
│   └── labels_en.py
├── data/                 processed public graph tables; raw BIO corpus withheld
├── results/              machine-readable outputs created by analyses
├── figures/              final manuscript Figures 1–10 (PNG + PDF)
├── tests/
├── tools/
├── run_experiments.py
├── requirements.txt
├── CITATION.cff
└── MANIFEST_SHA256.csv
```

## Environment

Python 3.10+ is recommended. A GPU is not required.

```bash
python -m venv .venv
source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\Activate.ps1       # Windows PowerShell

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run the public-release workflow

```bash
python run_experiments.py
```

The default workflow recomputes analyses that can be reproduced from the released processed graph tables, regenerates available figures, generates Figure 10, runs the manuscript/result audit, and checks all committed manuscript figure assets.

The raw BIO corpus is **not** required for the public workflow.

## Full local end-to-end reproduction

The annotation audit and graph reconstruction in Script 08 start from the original BIO corpus. The revised manuscript explicitly does not redistribute that file because redistribution rights were not established.

Researchers who already have authorised access can place the file locally at:

```text
data/train.txt
```

and run:

```bash
python run_experiments.py --full-local
```

This mode:

1. rebuilds S0/S1/S2 from the BIO annotations;
2. checks the manuscript structural values;
3. reruns the 72 controlled ablation runs;
4. performs triple-level clustered inference;
5. reruns the 60-epoch O3 link-prediction protocol on S0/S1/S2;
6. regenerates Figures 1–10; and
7. applies the strict manuscript-alignment audit.

## Other useful commands

```bash
# Revision analyses only
python run_experiments.py --revision

# Original structural/KGE workflow
python run_experiments.py --original

# Regenerate figures from existing result tables
python run_experiments.py --figures-only

# Selected stages, with no implicit prerequisites
python run_experiments.py --steps objective-ablation statistics-revised figures-revision

# Unit tests
python -m unittest discover -s tests -v
```

## Manuscript figure map

The final revised manuscript uses **Figures 1–10**:

| Manuscript figure | Repository asset | Generator |
|---|---|---|
| Figure 1 | `fig01_schema.png/.pdf` | Script 05 |
| Figure 2 | `fig02_extraction_funnel.png/.pdf` | Script 05 |
| Figure 3 | `fig03_relation_composition.png/.pdf` | Script 05 |
| Figure 4 | `fig04_threshold_sensitivity.png/.pdf` | Script 05 |
| Figure 5 | `fig05_degree_structure.png/.pdf` | Script 06 |
| Figure 6 | `fig06_annotation_sensitivity.png/.pdf` | Script 10 |
| Figure 7 | `fig07_objective_ablation.png/.pdf` | Script 10 |
| Figure 8 | `fig08_relation_lift_exact.png/.pdf` | Script 10 |
| Figure 9 | `fig09_graph_map.png/.pdf` | Script 06 |
| Figure 10 | `fig10_kgmmai_design.png/.pdf` | Script 13 |

Historical diagnostic plots formerly numbered 6–8 in an earlier draft are **not manuscript figures**. `figstyle.py` redirects those outputs to `figures/supplementary/` so they cannot overwrite the final manuscript assets.

## Controlled objective ablation

Script 07 holds constant the code base, graph, deterministic split, typed negative sampler, random-number stream, embedding dimension, optimiser, learning rate, batch size, seeds, budgets, and filtered typed evaluation. Only the objective changes:

- **O1 margin** — pairwise margin-ranking loss with uniform negative weights;
- **O2 logistic** — binary logistic loss with uniform negative weights;
- **O3 self-adversarial** — the same logistic loss with self-adversarial negative weighting.

Four models × three objectives × three seeds × two budgets = **72 training runs**.

At 60 epochs the revised manuscript reports:

| Objective | 1st | 2nd | 3rd | 4th |
|---|---|---|---|---|
| O1 margin | DistMult 0.0991 | TransE 0.0942 | ComplEx 0.0932 | RotatE 0.0845 |
| O2 logistic | RotatE 0.2711 | TransE 0.2577 | DistMult 0.1627 | ComplEx 0.1546 |
| O3 self-adversarial | RotatE 0.2119 | TransE 0.1930 | DistMult 0.1859 | ComplEx 0.1831 |

Script 12 checks these values before a release is tagged.

## Statistical protocol

Script 09 follows the revised analysis contract:

- primary unit: **886 held-out triples**, not 1,772 queries;
- 5,000 cluster-bootstrap resamples of whole triples;
- secondary bootstrap blocking on **343 shared head entities**;
- six paired model comparisons;
- Holm-Bonferroni adjustment reported for **both** paired t-tests and Wilcoxon tests;
- paired Cohen's `d`;
- exact per-query random-ranking baseline `H_m / m`.

The exact-baseline relation analysis is therefore not comparable to the older pooled approximation.

## Annotation audit

Script 08 detects surface forms assigned more than one entity type, reconstructs three graph conditions, and writes a direct comparison against S0:

- **S0** — as annotated;
- **S1** — expert correction of the five PRE/HER collisions;
- **S2** — majority harmonisation of every multi-type surface form.

The script distinguishes two quantities that should not be conflated:

- the **net core-edge-count change** (the quantity behind the manuscript's 104-triple / approximately 1.1% S1 statement);
- the **symmetric semantic edge-set difference** (a larger membership-change measure).

`--strict-manuscript` checks the final revised Table 6 values.

## Data governance

`data/train.txt` is intentionally excluded from the current public release. The manuscript states that the source archive supplied no explicit redistribution licence; therefore source analysis and source redistribution are treated separately.

The public package retains the processed node/edge tables and the code necessary for authorised users to rebuild them from a local source copy. See `data/README.md`.

> **Repository-history note:** deleting a file from the current branch does not erase earlier Git history. Before creating the archival release, maintainers should verify that the raw corpus is absent from the released archive and, if required by the data-governance decision, purge historical blobs using an appropriate Git-history rewriting procedure.

## Release audit and manifest

After a local full rerun:

```bash
python code/12_revision_audit.py --strict-sensitivity
python tools/update_manifest.py
python tools/verify_manifest.py
```

The manifest utility deliberately excludes `data/train.txt` even if an authorised local copy is present.

## Citation

Use `CITATION.cff`. The archived reproducibility package is associated with:

```text
DOI: 10.5281/zenodo.21731543
Repository: https://github.com/ourHua/KG-MMAI
```

## Interpretation boundary

This package supports claims about graph construction, structural quality, annotation sensitivity, typed filtered link prediction, objective sensitivity, and statistical robustness. It does **not** provide evidence of KG-MMAI diagnostic accuracy, clinical benefit, or multimodal performance.
