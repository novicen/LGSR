from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

from data.datasets.hcc_tace import assemble_hcc_label, build_hcc_cases, preprocess_hcc_image
from data.splitting import legacy_train_test_split


PROJECT_ROOT = Path(__file__).resolve().parents[2]
SOURCE = PROJECT_ROOT / "workdir/patients_adjust"
INTERMEDIATE = PROJECT_ROOT / "workdir/MSD/Dataset001_Hcc_tace_seg"


@pytest.mark.skipif(not SOURCE.is_dir(), reason="Normalized HCC source unavailable")
def test_hcc_case_order_and_split_match_existing_intermediate_dataset():
    cases = build_hcc_cases(SOURCE)
    train_ids, test_ids = legacy_train_test_split([case.case_id for case in cases])

    assert len(train_ids) == 73
    assert len(test_ids) == 32
    assert set(train_ids) == {path.name[:-7] for path in (INTERMEDIATE / "labelsTr").glob("*.nii.gz")}
    assert set(test_ids) == {path.name[:-7] for path in (INTERMEDIATE / "labelsTs").glob("*.nii.gz")}


@pytest.mark.skipif(not SOURCE.is_dir(), reason="Normalized HCC source unavailable")
def test_hcc_label_assembly_and_phase2_alignment_match_existing_intermediate():
    case = next(case for case in build_hcc_cases(SOURCE) if case.case_id == "HCC_001")
    label = assemble_hcc_label(case)
    image = preprocess_hcc_image(case.phase_images[2], label)
    label_ref = sitk.ReadImage(str(INTERMEDIATE / "labelsTr/HCC_001.nii.gz"))
    image_ref = sitk.ReadImage(str(INTERMEDIATE / "imagesTr/HCC_001_0002.nii.gz"))

    np.testing.assert_array_equal(sitk.GetArrayFromImage(label), sitk.GetArrayFromImage(label_ref))
    np.testing.assert_array_equal(sitk.GetArrayFromImage(image), sitk.GetArrayFromImage(image_ref))
    assert label.GetOrigin() == label_ref.GetOrigin()
    assert label.GetSpacing() == label_ref.GetSpacing()
    assert label.GetDirection() == label_ref.GetDirection()
    assert image.GetOrigin() == image_ref.GetOrigin()
    assert image.GetSpacing() == image_ref.GetSpacing()
    assert image.GetDirection() == image_ref.GetDirection()
