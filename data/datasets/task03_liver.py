from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from data.export import ExportCase, export_dataset
from data.splitting import legacy_train_test_split


TASK03_LABELS = {"0": "background", "1": "liver", "2": "cancer"}


def build_task03_cases(source_dir: str | Path) -> tuple[list[ExportCase], list[ExportCase]]:
    source_dir = Path(source_dir)
    metadata = json.loads((source_dir / "dataset.json").read_text(encoding="utf-8"))
    records = metadata["training"]
    original_ids = [Path(record["image"]).name.removesuffix(".nii.gz") for record in records]
    output_ids = [_format_case_id(case_id) for case_id in original_ids]
    train_ids, test_ids = legacy_train_test_split(output_ids, ratio=0.7, seed=42)
    by_id = {}
    for record, original_id, output_id in zip(records, original_ids, output_ids):
        if output_id in by_id:
            raise ValueError(f"Duplicate Task03 case ID after formatting: {output_id}")
        image_path = source_dir / record["image"]
        label_path = source_dir / record["label"]
        if not image_path.is_file() or not label_path.is_file():
            raise FileNotFoundError(f"{original_id}: source image or label is missing")
        by_id[output_id] = ExportCase(output_id, image_path, label_path)
    return [by_id[name] for name in train_ids], [by_id[name] for name in test_ids]


def preprocess_task03_case(case: ExportCase) -> tuple[sitk.Image, sitk.Image]:
    image = sitk.ReadImage(str(case.image_path))
    label = sitk.ReadImage(str(case.label_path))

    if label.GetDirection()[0] == -1.0 or label.GetDirection()[4] == -1.0:
        label = _make_label_direction_positive(label)
    label = _crop_legacy_label(label, image)
    if not _same_geometry(image, label):
        image = _resample_to_reference(label, image)
    return image, label


def prepare_task03_dataset(
    source_dir: str | Path, output_dir: str | Path
) -> dict[str, int]:
    train_cases, test_cases = build_task03_cases(source_dir)
    return export_dataset(
        output_dir,
        train_cases=train_cases,
        test_cases=test_cases,
        label_names=TASK03_LABELS,
        case_preprocessor=preprocess_task03_case,
    )


def _format_case_id(case_id: str) -> str:
    prefix, patient_id = case_id.rsplit("_", maxsplit=1)
    return f"{prefix}_{patient_id.zfill(3)}"


def _make_label_direction_positive(label: sitk.Image) -> sitk.Image:
    direction = label.GetDirection()
    size = label.GetSize()
    spacing = label.GetSpacing()
    origin = list(label.GetOrigin())
    for axis, diagonal in ((0, 0), (1, 4)):
        if direction[diagonal] == -1.0:
            origin[axis] -= spacing[axis] * size[axis]

    resampler = sitk.ResampleImageFilter()
    resampler.SetOutputDirection((1, 0, 0, 0, 1, 0, 0, 0, 1))
    resampler.SetOutputOrigin(origin)
    resampler.SetOutputSpacing(spacing)
    resampler.SetSize(size)
    resampler.SetInterpolator(_legacy_interpolator(label))
    return resampler.Execute(label)


def _crop_legacy_label(label: sitk.Image, image: sitk.Image) -> sitk.Image:
    label_array = sitk.GetArrayFromImage(label)
    image_array = sitk.GetArrayViewFromImage(image)
    label_start = _first_nonzero_slice(label_array, positive_only=False)
    image_start = _first_nonzero_slice(image_array, positive_only=True)
    start = min(label_start, image_start)
    end = label_array.shape[0]
    cropped = sitk.GetImageFromArray(label_array[start : end + 1])
    cropped.SetDirection(label.GetDirection())
    cropped.SetSpacing(label.GetSpacing())
    origin = list(label.GetOrigin())
    step = label.GetSpacing()[-1] * (1 if label.GetDirection()[-1] == 1.0 else -1)
    origin[-1] += start * step
    cropped.SetOrigin(origin)
    return cropped


def _first_nonzero_slice(array: np.ndarray, *, positive_only: bool) -> int:
    predicate = np.any(array > 0, axis=tuple(range(1, array.ndim))) if positive_only else np.any(
        array != 0, axis=tuple(range(1, array.ndim))
    )
    indices = np.flatnonzero(predicate)
    return int(indices[0]) if len(indices) else 0


def _resample_to_reference(reference: sitk.Image, moving: sitk.Image) -> sitk.Image:
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference)
    resampler.SetTransform(sitk.AffineTransform(reference.GetDimension()))
    resampler.SetDefaultPixelValue(0)
    resampler.SetInterpolator(_legacy_interpolator(moving))
    return resampler.Execute(moving)


def _legacy_interpolator(image: sitk.Image) -> int:
    if image.GetPixelID() in (sitk.sitkInt8, sitk.sitkInt16):
        return sitk.sitkNearestNeighbor
    return sitk.sitkLinear


def _same_geometry(first: sitk.Image, second: sitk.Image) -> bool:
    return (
        first.GetOrigin() == second.GetOrigin()
        and first.GetDirection() == second.GetDirection()
        and first.GetSpacing() == second.GetSpacing()
        and first.GetSize() == second.GetSize()
    )
