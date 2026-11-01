from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

from data.datasets.hcc_tace import build_hcc_phase2_cases
from data.datasets.waw_tace import WAWCase, build_waw_phase2_cases, split_waw_cases


PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.skipif(
    not (PROJECT_ROOT / "workdir/MSD/Dataset001_Hcc_tace_seg").is_dir(),
    reason="HCC intermediate NIfTI data unavailable",
)
def test_hcc_phase2_selection_matches_dataset007():
    source = PROJECT_ROOT / "workdir/MSD/Dataset001_Hcc_tace_seg"
    final = PROJECT_ROOT / "workdir/nnunet/nnUNet_raw/Dataset007_hcc_train2"
    train_cases, test_cases, excluded = build_hcc_phase2_cases(source)

    assert len(train_cases) == 72
    assert len(test_cases) == 32
    assert excluded["Tr"] == ["HCC_089"]
    assert {case.case_id for case in train_cases} == {
        path.name[:-7] for path in (final / "labelsTr").glob("*.nii.gz")
    }
    assert {case.case_id for case in test_cases} == {
        path.name[:-7] for path in (final / "labelsTs").glob("*.nii.gz")
    }

    for split in ("Tr", "Ts"):
        for final_label_path in (final / f"labels{split}").glob("*.nii.gz"):
            intermediate = sitk.ReadImage(
                str(source / f"labels{split}" / final_label_path.name)
            )
            exported = sitk.ReadImage(str(final_label_path))
            source_array = sitk.GetArrayFromImage(intermediate)
            expected = np.where(source_array >= 4, 0, np.where(source_array == 3, 1, source_array))
            np.testing.assert_array_equal(sitk.GetArrayFromImage(exported), expected)
            assert exported.GetOrigin() == intermediate.GetOrigin()
            assert exported.GetSpacing() == intermediate.GetSpacing()
            assert exported.GetDirection() == intermediate.GetDirection()


@pytest.mark.skipif(
    not (PROJECT_ROOT / "workdir/MSD/Dataset006_Waw_allphase").is_dir(),
    reason="WAW intermediate NIfTI data unavailable",
)
def test_waw_phase2_selection_matches_dataset009():
    source = PROJECT_ROOT / "workdir/MSD/Dataset006_Waw_allphase"
    final = PROJECT_ROOT / "workdir/nnunet/nnUNet_raw/Dataset009_waw_train2"
    train_cases, test_cases, excluded = build_waw_phase2_cases(source)

    assert len(train_cases) == 161
    assert len(test_cases) == 70
    assert len(excluded["Tr"]) == 2
    assert excluded["Ts"] == []
    assert {case.case_id for case in train_cases} == {
        path.name[:-7] for path in (final / "labelsTr").glob("*.nii.gz")
    }
    assert {case.case_id for case in test_cases} == {
        path.name[:-7] for path in (final / "labelsTs").glob("*.nii.gz")
    }

    intermediate = PROJECT_ROOT / "workdir/MSD/Dataset006_Waw_allphase"
    outer_cases = sorted(
        {
            path.name[:-7]
            for split in ("Tr", "Ts")
            for path in (intermediate / f"labels{split}").glob("*.nii.gz")
        },
        key=lambda case_id: int(case_id.split("_")[1]),
    )
    synthetic_cases = [WAWCase(case_id.split("_")[1], Path("."), Path("."), Path(".")) for case_id in outer_cases]
    legacy_train, legacy_test = split_waw_cases(synthetic_cases)
    assert {f"waw_{int(case.case_id):03d}" for case in legacy_train} == {
        path.name[:-7] for path in (intermediate / "labelsTr").glob("*.nii.gz")
    }
    assert {f"waw_{int(case.case_id):03d}" for case in legacy_test} == {
        path.name[:-7] for path in (intermediate / "labelsTs").glob("*.nii.gz")
    }
