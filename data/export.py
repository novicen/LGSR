from __future__ import annotations

import json
import shutil
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from data.validation import read_image_geometry, validate_dataset


@dataclass(frozen=True)
class ExportCase:
    case_id: str
    image_path: Path
    label_path: Path


CasePreprocessor = Callable[[ExportCase], tuple[sitk.Image, sitk.Image]]


def export_dataset(
    output_dir: str | Path,
    *,
    train_cases: Sequence[ExportCase],
    test_cases: Sequence[ExportCase],
    label_names: Mapping[str, int],
    label_map: Mapping[int, int] | None = None,
    channel_id: int = 2,
    channel_name: str = "CT",
    case_preprocessor: CasePreprocessor | None = None,
) -> dict[str, int]:
    """Write paired nnU-Net cases to a new directory and validate the result."""
    output_dir = Path(output_dir).resolve()
    train_cases = list(train_cases)
    test_cases = list(test_cases)
    _validate_case_lists(train_cases, test_cases)
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")

    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent))
    try:
        for split_name, cases in (("Tr", train_cases), ("Ts", test_cases)):
            image_dir = staging / f"images{split_name}"
            label_dir = staging / f"labels{split_name}"
            image_dir.mkdir()
            label_dir.mkdir()
            for case in cases:
                _export_case(
                    case,
                    image_dir,
                    label_dir,
                    channel_id,
                    label_map or {},
                    case_preprocessor,
                )

        dataset_json = {
            "channel_names": {str(channel_id): channel_name},
            "labels": dict(label_names),
            "numTraining": len(train_cases),
            "file_ending": ".nii.gz",
            "overwrite_image_reader_writer": "SimpleITKIO",
        }
        (staging / "dataset.json").write_text(
            json.dumps(dataset_json, indent=2) + "\n", encoding="utf-8"
        )
        summary = validate_dataset(staging, allowed_label_values=set(label_names.values()))
        staging.rename(output_dir)
        return summary
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _validate_case_lists(
    train_cases: Sequence[ExportCase], test_cases: Sequence[ExportCase]
) -> None:
    all_cases = [*train_cases, *test_cases]
    case_ids = [case.case_id for case in all_cases]
    duplicates = sorted({case_id for case_id in case_ids if case_ids.count(case_id) > 1})
    if duplicates:
        raise ValueError(f"Duplicate case IDs across train/test: {', '.join(duplicates)}")
    for case in all_cases:
        if not case.case_id or "/" in case.case_id or "\\" in case.case_id:
            raise ValueError(f"Invalid case ID: {case.case_id!r}")
        if not Path(case.image_path).is_file():
            raise FileNotFoundError(f"{case.case_id}: missing image {case.image_path}")
        if not Path(case.label_path).is_file():
            raise FileNotFoundError(f"{case.case_id}: missing label {case.label_path}")


def _export_case(
    case: ExportCase,
    image_dir: Path,
    label_dir: Path,
    channel_id: int,
    label_map: Mapping[int, int],
    case_preprocessor: CasePreprocessor | None,
) -> None:
    if case_preprocessor is None:
        label = sitk.ReadImage(str(case.label_path))
        image_geometry = read_image_geometry(case.image_path)
        _assert_same_geometry(case.case_id, image_geometry, label)
    else:
        image, label = case_preprocessor(case)
        image_geometry = _geometry(image)
        _assert_same_geometry(case.case_id, image_geometry, label)

    label_array = sitk.GetArrayFromImage(label)
    mapped = label_array.copy()
    for source_value, target_value in label_map.items():
        mapped[label_array == source_value] = target_value
    if not np.array_equal(mapped, label_array):
        remapped_label = sitk.GetImageFromArray(mapped)
        remapped_label.CopyInformation(label)
        label = remapped_label

    image_path = image_dir / f"{case.case_id}_{channel_id:04d}.nii.gz"
    if case_preprocessor is None:
        shutil.copy2(case.image_path, image_path)
    else:
        sitk.WriteImage(image, str(image_path))
    sitk.WriteImage(label, str(label_dir / f"{case.case_id}.nii.gz"))


def _geometry(image: sitk.Image) -> tuple[tuple, tuple, tuple, tuple]:
    return image.GetSize(), image.GetOrigin(), image.GetSpacing(), image.GetDirection()


def _assert_same_geometry(
    case_id: str,
    image_geometry: tuple[tuple, tuple, tuple, tuple],
    label: sitk.Image,
) -> None:
    label_geometry = _geometry(label)
    if image_geometry[0] != label_geometry[0]:
        raise ValueError(f"{case_id}: image/label size mismatch")
    if image_geometry[1] != label_geometry[1]:
        raise ValueError(f"{case_id}: image/label origin mismatch")
    if image_geometry[2] != label_geometry[2]:
        raise ValueError(f"{case_id}: image/label spacing mismatch")
    if image_geometry[3] != label_geometry[3]:
        raise ValueError(f"{case_id}: image/label direction mismatch")
