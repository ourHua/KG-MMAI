# Manuscript reference tables

These compact CSV files transcribe headline numerical values from the **revised IJASC manuscript** so that the repository has an explicit release contract.

They are **reference records**, not substitutes for machine-generated detailed output. The experiment scripts write their regenerated outputs to:

- `results/ablation/`
- `results/statistics/`
- `results/sensitivity/`

`code/12_revision_audit.py` compares regenerated output with the manuscript values and fails when a hard claim is inconsistent.

The source-derived S0/S1/S2 tables can only be regenerated end to end by an authorised user who has the withheld BIO corpus locally at `data/train.txt`.
