# Data directory and redistribution boundary

This directory contains processed graph tables used by the public KG-MMAI
reproducibility package.  The raw BIO-tagged corpus is intentionally absent.

The study analysed `train.txt`, the training split of the public TCMNER dataset.
The upstream location, exact byte size, complete SHA-256 digest, and corpus
fingerprint are recorded in [`../PROVENANCE.md`](../PROVENANCE.md).

Researchers who already have an authorised copy may place it at:

```text
KG-MMAI_LIJUNHUA/data/train.txt
```

and run:

```bash
python run_experiments.py --full-local
```

Script 08 will then rebuild the S0/S1/S2 graph variants from source and check
the headline structural values used in the manuscript.

Corpus fingerprint:

- 6,199 blank-line-delimited samples
- 307,398 character-level rows
- 41,262 entity mentions after deterministic repair
- 8,024 unique type-name pairs
- five BIO entity types: SYM, CAU, PRE, HER, EFF
- one malformed source label (`SYM[@`) repaired deterministically

`data/train.txt` is ignored by Git and excluded from the release manifest.  Do
not add the raw file to a public archive unless its redistribution rights have
been established separately.
