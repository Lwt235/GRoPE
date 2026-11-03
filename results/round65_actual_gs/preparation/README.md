---
license: other
task_categories:
  - other
tags:
  - point-cloud
  - point-cloud-classification
  - 3d
pretty_name: GSModel60
size_categories:
  - 10K<n<100K
---

# GSModel60

GSModel60 is the fixed 60-class Gaussian-splat point-cloud classification
dataset used by GAPrompt++. It contains 13,847 PLY files and preserves the
paper's original train/test split.

## Splits

| Split | Samples |
|---|---:|
| train | 7,757 |
| test | 6,090 |
| total | 13,847 |

`train.json` and `test.json` are part of the dataset definition and must not be
regenerated. `metadata/*.jsonl` contains labels, model IDs, relative paths,
source groups, file sizes, and available source hashes.

## Composition

- Classes 0--29: MACGS-derived Gaussian-splat point clouds.
- Classes 30--59: ModelNet_Splats-derived Gaussian-splat point clouds.

The dataset maintainer confirmed that the complete GSModel60 package may be
published and redistributed. The release uses the Hugging Face `other` license
identifier because no standard SPDX license identifier was supplied. See
`LICENSE_NOTICE.md` for the release statement.

## Loading

Use this directory as both `DATA_PATH` and `GS_PATH` in the GAPrompt++
classification config. The loader reads `category.txt`, `train.json`, and
`test.json` directly. Evaluation sampling is deterministic and uses
`EVAL_SEED + CRC32(model_id)`.

## Integrity

`FILES_SHA256SUMS.txt` validates every controlled file. Run:

```bash
sha256sum -c FILES_SHA256SUMS.txt
```

The GAPromptPlus repository also provides:

```bash
python scripts/verify_dataset_bundle.py /path/to/GSModel60   --dataset gsmodel60 --check-files
python scripts/smoke_test_classification_dataset.py /path/to/GSModel60   --dataset gsmodel60
```
