from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import SimpleITK as sitk

from data.datasets.waw_tace import merge_waw_label
from data.datasets.waw_tace import prepare_waw_intermediate_dataset


def test_waw_mask_merge_preserves_organ_geometry_and_tumor_priority(tmp_path, monkeypatch):
    organ_array = np.zeros((2, 3, 4), dtype=np.uint8)
    organ_array[:, :, :2] = 5
    organ = sitk.GetImageFromArray(organ_array)
    organ.SetOrigin((10.0, -3.0, 21.0))
    organ.SetSpacing((0.7, 0.8, 2.0))
    organ_path = tmp_path / "organ_2.nii.gz"
    sitk.WriteImage(organ, str(organ_path))

    tumor_array = np.zeros((4, 3, 2), dtype=np.uint8)
    tumor_array[1, 1, 0] = 1
    tumor_path = tmp_path / "case_2_0_tumor_seg.nrrd"
    tumor_path.write_bytes(b"NRRD fixture")
    tumor_header = {
        "space origin": np.array([10.0, -3.0, 21.0]),
        "space directions": np.diag([0.7, 0.8, 2.0]),
    }
    monkeypatch.setitem(
        sys.modules,
        "nrrd",
        SimpleNamespace(read=lambda _: (tumor_array, tumor_header)),
    )

    merged = merge_waw_label(organ_path, [tumor_path])
    organ = sitk.ReadImage(str(organ_path))

    expected = np.zeros_like(organ_array)
    expected[:, :, :2] = 1
    expected[0, 1, 1] = 2
    np.testing.assert_array_equal(sitk.GetArrayFromImage(merged), expected)
    assert merged.GetOrigin() == organ.GetOrigin()
    assert merged.GetSpacing() == organ.GetSpacing()
    assert merged.GetDirection() == organ.GetDirection()


def test_waw_raw_dataset_conversion_writes_phase_files_and_outer_split(tmp_path, monkeypatch):
    waw_root = tmp_path / "WAW-TACE"
    tumor_dir = waw_root / "tumor_masks_wawtace_v1_08_05_2024" / "1"
    organ_dir = waw_root / "organ_masks_wawtace_09_05_24" / "1"
    ct_dir = waw_root / "ct_scans_4_4_wawtace_09_05_24" / "1"
    tumor_dir.mkdir(parents=True)
    organ_dir.mkdir(parents=True)
    ct_dir.mkdir(parents=True)
    tumor_path = tumor_dir / "1_2_0_tumor_seg.nrrd"
    tumor_path.write_bytes(b"NRRD fixture")

    organ_array = np.zeros((2, 3, 4), dtype=np.uint8)
    organ_array[:, :, :2] = 5
    organ = sitk.GetImageFromArray(organ_array)
    organ_path = organ_dir / "organ_2_merged.nii.gz"
    sitk.WriteImage(organ, str(organ_path))
    image = sitk.GetImageFromArray(np.arange(24, dtype=np.int16).reshape(2, 3, 4))
    image.CopyInformation(sitk.ReadImage(str(organ_path)))
    sitk.WriteImage(image, str(ct_dir / "scan_2_volume.nii.gz"))

    tumor_array = np.zeros((4, 3, 2), dtype=np.uint8)
    tumor_array[1, 1, 0] = 1
    tumor_header = {
        "space origin": np.array([0.0, 0.0, 0.0]),
        "space directions": np.diag([1.0, 1.0, 1.0]),
    }
    monkeypatch.setitem(
        sys.modules,
        "nrrd",
        SimpleNamespace(read=lambda _: (tumor_array, tumor_header)),
    )

    output = tmp_path / "Dataset006_Waw_allphase"
    counts = prepare_waw_intermediate_dataset(waw_root, output)

    assert counts == {"Tr": 0, "Ts": 1}
    assert (output / "imagesTs/waw_001_0002.nii.gz").is_file()
    exported_label = sitk.GetArrayFromImage(sitk.ReadImage(str(output / "labelsTs/waw_001.nii.gz")))
    assert exported_label[0, 1, 1] == 2
