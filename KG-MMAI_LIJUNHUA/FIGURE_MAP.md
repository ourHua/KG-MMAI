# Figure-to-code map

Each manuscript figure is owned by one Python entry script. Shared helpers provide styling and file lookup only; they do not emit manuscript figures.

| Figure | Script | Main input |
|---|---|---|
| 1 | `code/figures/fig01_schema.py` | node tables and relation distribution |
| 2 | `code/figures/fig02_extraction_funnel.py` | node/edge tables and weight distribution |
| 3 | `code/figures/fig03_relation_composition.py` | relation distribution |
| 4 | `code/figures/fig04_threshold_sensitivity.py` | structural profile |
| 5 | `code/figures/fig05_degree_structure.py` | core-node degree table |
| 6 | `code/figures/fig06_annotation_sensitivity.py` | S0/S1/S2 sensitivity outputs |
| 7 | `code/figures/fig07_objective_ablation.py` | ablation and statistical outputs |
| 8 | `code/figures/fig08_relation_lift_exact.py` | exact relation-lift table |
| 9 | `code/figures/fig09_graph_map.py` | core graph and deterministic layout |
| 10 | `code/figures/fig10_kgmmai_design.py` | design specification only |

`run_figures.py` is only an orchestrator and does not draw a figure itself.
