# LGSR Dataset Preparation

These scripts create nnU-Net datasets from the project data. They never
download, move, or include images in the release. Each command requires
explicit input and output paths, writes into a new destination, and stops if
the destination already exists.

Install `requirements.txt` for data conversion and `requirements-dev.txt` to
run the regression suite. Run commands from `lgsr_release`.

## HCC-TACE-SEG

`hcc-intermediate` reads the organized NIfTI structure produced from the TCIA
cases. Each case has `PreTACE` phase folders containing `CT.nii.gz` and
`Segmentation/{liver,tumor,vessels,aorta}/seg.nii.gz`.

```bash
python -m data.cli hcc-intermediate \
  --normalized-dir /path/to/patients_adjust \
  --output-dir /path/to/msd/Dataset001_Hcc_tace_seg

python -m data.cli hcc-phase2 \
  --intermediate-dir /path/to/msd/Dataset001_Hcc_tace_seg \
  --output-dir /path/to/nnUNet_raw/Dataset007_hcc_train2
```

The intermediate cases are split 73/32 with the historical seed 42. The
phase-2 export has 72 training and 32 test cases because HCC_089 lacks a
phase-2 image. The label conversion is `3 -> 1`, `4 and above -> 0`; labels
0, 1, and 2 remain unchanged.

The input to this command is the organized NIfTI tree produced by the earlier
TCIA DICOM/SEG normalization stage. The original TCIA manifest used by that
stage is not present in the current workspace, so that upstream extraction
cannot be recreated or regression checked from this checkout. The organized
input conversion above has been checked against HCC_001; the label mapping
was compared against all 104 existing output pairs.

## WAW-TACE

The raw input root must contain `tumor_masks_wawtace_v1_08_05_2024`,
`organ_masks_wawtace_09_05_24`, and `ct_scans_1_4_wawtace_09_05_24` through
`ct_scans_4_4_wawtace_09_05_24`.

```bash
python -m data.cli waw-intermediate \
  --waw-root /path/to/WAW-TACE \
  --output-dir /path/to/msd/Dataset006_Waw_allphase

python -m data.cli waw-phase2 \
  --intermediate-dir /path/to/msd/Dataset006_Waw_allphase \
  --output-dir /path/to/nnUNet_raw/Dataset009_waw_train2
```

The outer split is 163/70 with the historical seed 42 and numeric case-ID
ordering. Phase-2 export keeps 161/70 pairs because two training cases lack
that phase. The release includes raw NRRD/NIfTI mask merging and multi-phase
CT alignment. The original WAW input directory is absent locally, so raw
conversion is tested with synthetic volumes; phase selection is checked
against the current all-phase and nnU-Net datasets.

## MSD Task03_Liver

```bash
python -m data.cli task03 \
  --source-dir /path/to/Task03_Liver \
  --output-dir /path/to/nnUNet_raw/Dataset002_Task03_liver
```

The converter preserves source `dataset.json` order, creates the historical
91/40 outer split with seed 42, and applies the original orientation, crop,
and resampling operations. One representative case is checked voxel by voxel
against the existing nnU-Net output.

## Folds and Validation

After a raw dataset is ready, create the same five folds as nnU-Net:

```bash
python -m data.cli folds \
  --dataset-dir /path/to/nnUNet_raw/Dataset007_hcc_train2
```

This writes `splits_final.json` using sorted training IDs,
`KFold(n_splits=5, shuffle=True, random_state=12345)`. Existing files are
preserved unless `--overwrite` is supplied. Run output checks with:

```bash
python -m data.cli validate \
  --dataset-dir /path/to/nnUNet_raw/Dataset007_hcc_train2
```

The default validator checks phase suffixes, train/test image-label pairing,
and spatial geometry from NIfTI headers without loading CT voxels. For a full
label-value scan, add `--check-label-values`; this reads every label volume
and verifies its values against `dataset.json`.

## Regression Tests

Run tests from this release directory:

```bash
PYTHONPATH="$PWD/src:$PWD/vendor" python -m pytest -q tests
```

Install the test runner first with `python -m pip install -r requirements-dev.txt`.

The local workspace tests also compare case lists and available NIfTI outputs
with the historical datasets. Tests requiring local datasets are skipped
when those source directories are absent.
