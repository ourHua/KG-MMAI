# Source corpus provenance

This repository does not redistribute the raw BIO corpus.  The analysis used the
`train.txt` training split of the public **TCMNER** dataset released by Cao and
Wu with their MMM 2024 work.

- Upstream repository: https://github.com/cshan-github/TCM_NER_datasets
- Analysed file: `train.txt`
- File size: **2,318,558 bytes**
- SHA-256: `5b4560805e6cd49295e82ae2646c3d1b116e213ae074f1474a3d452229064d71`

The identity of the analysed file was checked by recomputing the corpus-level
counts reported in the manuscript and by comparing the byte size and SHA-256
digest with the public source copy.  The training split contains 6,199
blank-line-delimited samples, 307,398 character-level rows, 41,262 entity
mentions after deterministic label repair, and 8,024 unique type-name pairs.

The dataset authors describe five BIO entity categories: symptom (SYM), cause
(CAU), prescription (PRE), herb (HER), and effect (EFF).  The present study uses
the released annotations as source data and claims no credit for their
collection or annotation.

## Verify a local copy

On Linux:

```bash
sha256sum data/train.txt
```

On macOS:

```bash
shasum -a 256 data/train.txt
```

The output should be:

```text
5b4560805e6cd49295e82ae2646c3d1b116e213ae074f1474a3d452229064d71
```

## Redistribution boundary

At the time this release was prepared, the upstream repository did not provide
an explicit redistribution licence for the raw corpus.  For that reason,
`data/train.txt` is cited and fingerprinted here but is not included in this
repository or in its public archive.  The repository contains only derived
analysis tables, result summaries, code, and publication figures.
