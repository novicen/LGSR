from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import SimpleITK as sitk

from data.export import ExportCase, export_dataset
from data.validation import validate_dataset


def _write_volume(path: Path, array: np.ndarray) -> sitk.Image:
    image = sitk.GetImageFromArray(array)
    image.SetOrigin((13.5, -4.0, 29.0))
    image.SetSpacing((0.8, 0.9, 2.5))
    image.SetDirection((-1.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 1.0))
    sitk.WriteImage(image, str(path))
    return image


def _case(tmp_path: Path, case_id: str, labels: np.ndarray) -> ExportCase:
    image_path = tmp_path / f"{case_id}_source.nii.gz"
    label_path = tmp_path / f"{case_id}_label.nii.gz"
    _write_volume(image_path, np.arange(labels.size, dtype=np.int16).reshape(labels.shape))
    _write_volume(label_path, labels.astype(np.uint8))
    return ExportCase(case_id, image_path, label_path)


def test_export_preserves_image_and_label_values_geometry_and_metadata(tmp_path):
    labels = np.array([[[0, 1, 2], [3, 4, 0]]], dtype=np.uint8)
    train_case = _case(tmp_path, "case_001", labels)
    test_case = _case(tmp_path, "case_002", labels)
    output = tmp_path / "Dataset001_Test"

    export_dataset(
        output,
        train_cases=[train_case],
        test_cases=[test_case],
        label_names={"background": 0, "liver": 1, "cancer": 2},
        label_map={3: 1, 4: 0},
    )

    expected = np.array([[[0, 1, 2], [1, 0, 0]]], dtype=np.uint8)
    for split in ("Tr", "Ts"):
        image_path = output / f"images{split}" / f"case_00{1 if split == 'Tr' else 2}_0002.nii.gz"
        label_path = output / f"labels{split}" / f"case_00{1 if split == 'Tr' else 2}.nii.gz"
        input_image = sitk.ReadImage(str(train_case.image_path if split == "Tr" else test_case.image_path))
        exported_image = sitk.ReadImage(str(image_path))
        exported_label = sitk.ReadImage(str(label_path))
        np.testing.assert_array_equal(
            sitk.GetArrayFromImage(exported_image), sitk.GetArrayFromImage(input_image)
        )
        np.testing.assert_array_equal(sitk.GetArrayFromImage(exported_label), expected)
        assert exported_image.GetOrigin() == input_image.GetOrigin()
        assert exported_image.GetSpacing() == input_image.GetSpacing()
        assert exported_image.GetDirection() == input_image.GetDirection()
        assert exported_label.GetOrigin() == input_image.GetOrigin()
        assert exported_label.GetSpacing() == input_image.GetSpacing()
        assert exported_label.GetDirection() == input_image.GetDirection()

    dataset = json.loads((output / "dataset.json").read_text())
    assert dataset["channel_names"] == {"2": "CT"}
    assert dataset["labels"] == {"background": 0, "liver": 1, "cancer": 2}
    assert dataset["numTraining"] == 1
    assert dataset["file_ending"] == ".nii.gz"


def test_export_rejects_duplicate_ids_before_creating_destination(tmp_path):
    case = _case(tmp_path, "duplicate", np.zeros((1, 2, 2), dtype=np.uint8))
    output = tmp_path / "Dataset002_Duplicate"

    with pytest.raises(ValueError, match="duplicate"):
        export_dataset(
            output,
            train_cases=[case],
            test_cases=[case],
            label_names={"background": 0},
        )

    assert not output.exists()


def test_validator_reports_geometry_mismatch(tmp_path):
    case = _case(tmp_path, "geometry", np.zeros((1, 2, 2), dtype=np.uint8))
    output = tmp_path / "Dataset003_Geometry"
    export_dataset(
        output,
        train_cases=[case],
        test_cases=[],
        label_names={"background": 0},
    )

    label_path = output / "labelsTr" / "geometry.nii.gz"
    wrong_geometry = sitk.ReadImage(str(label_path))
    wrong_geometry.SetOrigin((0.0, 0.0, 0.0))
    sitk.WriteImage(wrong_geometry, str(label_path))

    with pytest.raises(ValueError, match="geometry"):
        validate_dataset(output)


def test_validator_checks_geometry_from_headers_without_loading_ct_arrays(tmp_path, monkeypatch):
    case = _case(tmp_path, "header_only", np.zeros((1, 2, 2), dtype=np.uint8))
    output = tmp_path / "Dataset004_HeaderOnly"
    export_dataset(
        output,
        train_cases=[case],
        test_cases=[],
        label_names={"background": 0},
    )

    def fail_read_image(_):
        raise AssertionError("validator should inspect NIfTI headers without loading voxels")

    monkeypatch.setattr(sitk, "ReadImage", fail_read_image)
    assert validate_dataset(output) == {"Tr": 1, "Ts": 0}


def test_export_copies_ct_without_decompressing_its_voxels(tmp_path, monkeypatch):
    case = _case(tmp_path, "copy_ct", np.zeros((1, 2, 2), dtype=np.uint8))
    output = tmp_path / "Dataset005_CopyCT"
    original_read_image = sitk.ReadImage

    def reject_ct_read(path):
        if Path(path) == case.image_path:
            raise AssertionError("export should copy the CT file after checking its header")
        return original_read_image(str(path))

    monkeypatch.setattr(sitk, "ReadImage", reject_ct_read)
    export_dataset(
        output,
        train_cases=[case],
        test_cases=[],
        label_names={"background": 0},
    )
