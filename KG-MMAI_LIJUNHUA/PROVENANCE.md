# Source corpus provenance

The raw BIO corpus is not included in this repository. The experiments used `train.txt`, the training split of the public **TCMNER** dataset released by Cao and Wu with their MMM 2024 work.

## File used in this study

- Upstream repository: https://github.com/cshan-github/TCM_NER_datasets
- File: `train.txt`
- Size: **2,318,558 bytes**
- SHA-256: `5b4560805e6cd49295e82ae2646c3d1b116e213ae074f1474a3d452229064d71`

The file identity was checked against the public source copy and against the corpus-level counts used in the manuscript. After the same deterministic label repair used by the analysis code, the file gives 6,199 samples, 307,398 character-level rows, 41,262 entity mentions and 8,024 unique type-name pairs.

The five BIO entity types are symptom (SYM), cause (CAU), prescription (PRE), herb (HER) and effect (EFF). These annotations come from the source dataset; this study did not create the original corpus or its labels.

## Checking a local copy

Linux:

```bash
sha256sum data/train.txt
```

macOS:

```bash
shasum -a 256 data/train.txt
```

Expected digest:

```text
5b4560805e6cd49295e82ae2646c3d1b116e213ae074f1474a3d452229064d71
```

## Redistribution

The upstream repository did not provide an explicit redistribution licence for the raw corpus when this release was prepared. For that reason, `data/train.txt` is cited and fingerprinted here but is not copied into this repository or its public archive. The release contains code, derived tables, result summaries and publication figures only.
