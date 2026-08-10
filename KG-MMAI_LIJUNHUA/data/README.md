# Data directory and redistribution boundary

This directory contains **processed, derived graph tables** used by the public KG-MMAI reproducibility package.

The original BIO-tagged corpus was analysed as `train.txt`, but the revised manuscript states that the supplied archive did not include a source citation or an explicit redistribution licence. The public release therefore does **not** redistribute the raw corpus.

Researchers who already have authorised access may place a local copy at:

```text
KG-MMAI_LIJUNHUA/data/train.txt
```

and run:

```bash
python run_experiments.py --full-local
```

Script 08 will then reconstruct the graph and audit the source annotations.

Corpus fingerprint reported in the manuscript:

- 6,199 blank-line-delimited samples
- 307,398 character-level tokens
- 41,262 entity mentions after deterministic repair
- 8,024 unique type-name pairs
- five BIO entity types: SYM, CAU, PRE, HER, EFF
- one malformed source label (`SYM[@`) repaired deterministically

`data/train.txt` is ignored by Git and excluded by `tools/update_manifest.py`. Do not add it to a public release unless redistribution rights have been separately established.
