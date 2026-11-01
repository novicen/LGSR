# LGSR

This directory contains the reproducible implementation of **Logit-Guided
Stopping Regions (LGSR)** for dense liver-tumor segmentation.
Datasets, preprocessed arrays, checkpoints, predictions, and experiment logs
are intentionally excluded.

## Method configuration

The released trainer is `nnUNetTrainer_lgsr` and keeps the paper's primary
configuration:

- 32-dimensional penultimate decoder feature;
- class components `(1, 3, 3)` for background, liver, and tumor;
- identity covariance and uniform within-class component weights;
- strict logit-certified component region with rates `(1, 1, 1)`;
- active CE plus `0.01 *` class-balanced active true-class NLL;
- active foreground Dice with weight `1.0`;
- geometry-only gate (`require_safe_prediction_match=false`);
- gate applied during training and removed during inference.

The formulas are implemented in `src/lgsr/geometry`, `src/lgsr/models`,
and `src/lgsr/losses`. The nnU-Net runtime is vendored under
`vendor/nnunetv2` and contains no project data.

## Dataset Preparation

Dataset conversion, phase-2 selection, legacy train/test splits, nnU-Net
fold generation, and validation commands are documented in
[`data/README.md`](data/README.md). The data tools write into caller-provided
directories; dataset files are not included in this release.

## Environment

Use an environment containing the packages in `requirements.txt`, then set:

```bash
export nnUNet_raw=/path/to/nnUNet_raw
export nnUNet_preprocessed=/path/to/nnUNet_preprocessed
export nnUNet_results=/path/to/nnUNet_results
```

The roots must contain the standard nnU-Net dataset, preprocessed, and model
directory structures. No data is downloaded or copied by this package.
The LGSR source code, data-preparation tools, scripts, configurations, and
documentation are licensed under Apache-2.0; see the root `LICENSE` file.
The vendored nnU-Net runtime retains its Apache-2.0 license and attribution;
see `THIRD_PARTY_NOTICES.md`.

## Training

```bash
cd lgsr_release
bash scripts/train_lgsr.sh DatasetXXX 3d_fullres 4
```

Supported options include `--continue-training`, `--validation-only`,
`--pretrained-weights PATH`, `--save-probabilities`,
`--disable-checkpointing`, `--device cuda|cpu|mps`, and `-p PLAN_NAME`.

## Inference

```bash
export PYTHONPATH="$PWD/src:$PWD/vendor:$PYTHONPATH"
python scripts/predict_lgsr.py \
  -i /path/to/imagesTs \
  -o /path/to/predictions \
  -m /path/to/nnUNet_results/DatasetXXX/nnUNetTrainer_lgsr__nnUNetPlans__3d_fullres \
  -f 4
```

## Verification

Run the deterministic equivalence and architecture checks:

```bash
PYTHONPATH="$PWD/src:$PWD/vendor" python -m pytest -q tests
```

The tests compare the released head, certified gate, loss, gradients, and
network checkpoint state against the current project implementation.
