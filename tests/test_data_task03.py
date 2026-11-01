from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

from data.datasets.task03_liver import build_task03_cases, preprocess_task03_case
from data.splitting import legacy_train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE = PROJECT_ROOT / "Task03_Liver"
OUTPUT = PROJECT_ROOT / "workdir/nnunet/nnUNet_raw/Dataset002_Task03_liver"


@pytest.mark.skipif(not (SOURCE / "dataset.json").is_file(), reason="Task03 source data unavailable")
def test_task03_split_matches_existing_phase2_case_ids_in_source_order():
    dataset = json.loads((SOURCE / "dataset.json").read_text())
    source_ids = [Path(item["image"]).name.removesuffix(".nii.gz") for item in dataset["training"]]
    names = [f"liver_{int(case_id.split('_')[1]):03d}" for case_id in source_ids]
    train_names, test_names = legacy_train_test_split(names)
    train_cases, test_cases = build_task03_cases(SOURCE)

    assert [case.case_id for case in train_cases] == train_names
    assert [case.case_id for case in test_cases] == test_names
    assert {case.case_id for case in train_cases} == {
        path.name[:-7] for path in (OUTPUT / "labelsTr").glob("*.nii.gz")
    }
    assert {case.case_id for case in test_cases} == {
        path.name[:-7] for path in (OUTPUT / "labelsTs").glob("*.nii.gz")
    }


@pytest.mark.skipif(not (SOURCE / "dataset.json").is_file(), reason="Task03 source data unavailable")
def test_task03_case_preprocessing_matches_current_export():
    train_cases, test_cases = build_task03_cases(SOURCE)
    case = next(item for item in train_cases + test_cases if item.case_id == "liver_014")
    split = "Tr" if case in train_cases else "Ts"
    image, label = preprocess_task03_case(case)
    image_ref = sitk.ReadImage(str(OUTPUT / f"images{split}" / "liver_014_0002.nii.gz"))
    label_ref = sitk.ReadImage(str(OUTPUT / f"labels{split}" / "liver_014.nii.gz"))

    np.testing.assert_array_equal(sitk.GetArrayFromImage(image), sitk.GetArrayFromImage(image_ref))
    np.testing.assert_array_equal(sitk.GetArrayFromImage(label), sitk.GetArrayFromImage(label_ref))
    assert image.GetOrigin() == image_ref.GetOrigin()
    assert image.GetSpacing() == image_ref.GetSpacing()
    assert image.GetDirection() == image_ref.GetDirection()
    assert label.GetOrigin() == label_ref.GetOrigin()
    assert label.GetSpacing() == label_ref.GetSpacing()
    assert label.GetDirection() == label_ref.GetDirection()
