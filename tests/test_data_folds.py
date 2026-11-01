from __future__ import annotations

import json
from pathlib import Path

import pytest

from data.folds import generate_nnunet_folds


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "dataset_name",
    ["Dataset007_hcc_train2", "Dataset002_Task03_liver", "Dataset009_waw_train2"],
)
def test_generated_folds_match_saved_nnunet_splits(dataset_name):
    dataset = PROJECT_ROOT / "workdir/nnunet/nnUNet_raw" / dataset_name
    saved = PROJECT_ROOT / "workdir/nnunet/nnUNet_preprocessed" / dataset_name / "splits_final.json"
    if not dataset.is_dir() or not saved.is_file():
        pytest.skip(f"Local parity data unavailable for {dataset_name}")
    case_ids = [path.name[:-12] for path in (dataset / "imagesTr").glob("*_0002.nii.gz")]

    assert generate_nnunet_folds(case_ids) == json.loads(saved.read_text())


def test_generate_nnunet_folds_sorts_case_ids_before_kfold():
    folds = generate_nnunet_folds(["case_3", "case_1", "case_2"], n_splits=3)

    assert len(folds) == 3
    assert all(sorted(fold["train"] + fold["val"]) == ["case_1", "case_2", "case_3"] for fold in folds)
