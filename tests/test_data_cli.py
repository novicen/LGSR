from __future__ import annotations

import json

import numpy as np
import pytest
import SimpleITK as sitk

from data.cli import main
from data.folds import generate_nnunet_folds


def test_cli_writes_reproducible_nnunet_folds(tmp_path):
    dataset = tmp_path / "Dataset001_Test"
    (dataset / "imagesTr").mkdir(parents=True)
    (dataset / "labelsTr").mkdir()
    for case_id in ["case_4", "case_2", "case_0", "case_3", "case_1"]:
        image = sitk.GetImageFromArray(np.zeros((1, 2, 2), dtype=np.int16))
        sitk.WriteImage(image, str(dataset / "imagesTr" / f"{case_id}_0002.nii.gz"))
    folds_path = dataset / "splits_final.json"

    assert main(["folds", "--dataset-dir", str(dataset)]) == 0

    actual = json.loads(folds_path.read_text())
    assert actual == generate_nnunet_folds([f"case_{i}" for i in range(5)])


def test_cli_does_not_overwrite_existing_folds_without_explicit_flag(tmp_path):
    dataset = tmp_path / "Dataset002_Test"
    (dataset / "imagesTr").mkdir(parents=True)
    for case_id in range(5):
        image = sitk.GetImageFromArray(np.zeros((1, 1, 1), dtype=np.int16))
        sitk.WriteImage(image, str(dataset / "imagesTr" / f"case_{case_id}_0002.nii.gz"))
    folds_path = dataset / "splits_final.json"
    folds_path.write_text("[]\n")

    with pytest.raises(FileExistsError, match="splits_final.json"):
        main(["folds", "--dataset-dir", str(dataset)])

    assert folds_path.read_text() == "[]\n"


def test_cli_label_value_validation_is_opt_in(tmp_path, capsys):
    dataset = tmp_path / "Dataset003_Labels"
    images = dataset / "imagesTr"
    labels = dataset / "labelsTr"
    images.mkdir(parents=True)
    labels.mkdir()
    image = sitk.GetImageFromArray(np.zeros((1, 2, 2), dtype=np.int16))
    label = sitk.GetImageFromArray(np.full((1, 2, 2), 2, dtype=np.uint8))
    label.CopyInformation(image)
    sitk.WriteImage(image, str(images / "case_0_0002.nii.gz"))
    sitk.WriteImage(label, str(labels / "case_0.nii.gz"))
    (dataset / "dataset.json").write_text(
        json.dumps(
            {
                "channel_names": {"2": "CT"},
                "labels": {"background": 0},
                "numTraining": 1,
            }
        )
    )

    assert main(["validate", "--dataset-dir", str(dataset)]) == 0
    capsys.readouterr()
    with pytest.raises(ValueError, match="unexpected label values"):
        main(["validate", "--dataset-dir", str(dataset), "--check-label-values"])
