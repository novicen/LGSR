from __future__ import annotations

import json
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import SimpleITK as sitk

from data.export import ExportCase, export_dataset
from data.phase import build_phase_cases
from data.splitting import legacy_train_test_split


HCC_SEGMENTS = ("liver", "tumor", "vessels", "aorta")
HCC_INTERMEDIATE_LABELS = {
    "background": 0,
    "liver": 1,
    "tumor": 2,
    "vessels": 3,
    "aorta": 4,
}


@dataclass(frozen=True)
class HCCCase:
    case_id: str
    phase_images: dict[int, Path]
    segment_paths: tuple[Path, ...]


def build_hcc_cases(normalized_root: str | Path) -> list[HCCCase]:
    normalized_root = Path(normalized_root)
    cases = []
    for patient_dir in sorted(path for path in normalized_root.iterdir() if path.is_dir()):
        pretace_dir = patient_dir / "PreTACE"
        segment_paths = tuple(
            patient_dir / "Segmentation" / segment / "seg.nii.gz"
            for segment in HCC_SEGMENTS
        )
        if not pretace_dir.is_dir() or not all(path.is_file() for path in segment_paths):
            continue

        phase_images: dict[int, Path] = {}
        series_dirs = sorted((path for path in pretace_dir.iterdir() if path.is_dir()), key=lambda p: p.name)
        if series_dirs:
            series_dirs.insert(0, series_dirs.pop())
        for series_dir in series_dirs:
            image_path = series_dir / "CT.nii.gz"
            if not image_path.is_file():
                continue
            phase_id = 0 if series_dir.name == "preliver" else _phase_id(series_dir.name)
            if phase_id in phase_images:
                raise ValueError(f"{patient_dir.name}: duplicate CT phase {phase_id}")
            phase_images[phase_id] = image_path
        cases.append(HCCCase(patient_dir.name, phase_images, segment_paths))
    return cases


def assemble_hcc_label(case: HCCCase) -> sitk.Image:
    masks = [sitk.ReadImage(str(path)) for path in case.segment_paths]
    reference = masks[0]
    for segment, mask in zip(HCC_SEGMENTS[1:], masks[1:]):
        if not _same_geometry(reference, mask):
            raise ValueError(f"{case.case_id}: {segment} mask geometry differs from liver mask")

    label_array = np.zeros_like(sitk.GetArrayViewFromImage(reference), dtype=np.uint8)
    for index, mask in enumerate(masks):
        mask_array = sitk.GetArrayViewFromImage(mask)
        label_array += (index + 1) * mask_array
    label = sitk.GetImageFromArray(label_array)
    label.CopyInformation(reference)
    return label


def compute_hcc_collapse(label: sitk.Image, image: sitk.Image) -> int:
    label_origin = label.GetOrigin()[-1]
    label_spacing = label.GetSpacing()[-1]
    label_slices = label.GetSize()[-1]
    image_origin = image.GetOrigin()[-1]
    image_spacing = image.GetSpacing()[-1]
    image_slices = image.GetSize()[-1]
    start = max(label_origin, image_origin)
    end = min(
        label_origin + label_spacing * label_slices,
        image_origin + image_spacing * image_slices,
    )
    return 0 if start >= end else int((end - start) / label_spacing)


def preprocess_hcc_image(image_path: str | Path, label: sitk.Image) -> sitk.Image | None:
    image = sitk.ReadImage(str(image_path))
    if compute_hcc_collapse(label, image) < label.GetSize()[-1] * 0.3:
        return None

    reference = _make_hcc_label_direction_positive(label)
    if _same_geometry(image, reference):
        return image
    return _resample_to_reference(reference, image)


def prepare_hcc_intermediate_dataset(
    normalized_root: str | Path, output_dir: str | Path
) -> dict[str, int]:
    cases = build_hcc_cases(normalized_root)
    train_ids, test_ids = legacy_train_test_split([case.case_id for case in cases])
    cases_by_id = {case.case_id: case for case in cases}
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent))
    try:
        for split, case_ids in (("Tr", train_ids), ("Ts", test_ids)):
            image_dir = staging / f"images{split}"
            label_dir = staging / f"labels{split}"
            image_dir.mkdir()
            label_dir.mkdir()
            for case_id in case_ids:
                case = cases_by_id[case_id]
                label = assemble_hcc_label(case)
                for phase_id, image_path in case.phase_images.items():
                    image = preprocess_hcc_image(image_path, label)
                    if image is not None:
                        sitk.WriteImage(
                            image,
                            str(image_dir / f"{case_id}_{phase_id:04d}.nii.gz"),
                        )
                sitk.WriteImage(label, str(label_dir / f"{case_id}.nii.gz"))

        metadata = {
            "channel_names": {"0": "CT"},
            "labels": HCC_INTERMEDIATE_LABELS,
            "numTraining": len(train_ids),
            "file_ending": ".nii.gz",
            "overwrite_image_reader_writer": "SimpleITKIO",
        }
        (staging / "dataset.json").write_text(
            json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
        )
        staging.rename(output_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return {"Tr": len(train_ids), "Ts": len(test_ids)}


def build_hcc_phase2_cases(
    intermediate_dir: str | Path,
) -> tuple[list[ExportCase], list[ExportCase], dict[str, list[str]]]:
    return build_phase_cases(intermediate_dir, phase_id=2)


def prepare_hcc_phase2_dataset(
    intermediate_dir: str | Path, output_dir: str | Path
) -> tuple[dict[str, int], dict[str, list[str]]]:
    train_cases, test_cases, excluded = build_hcc_phase2_cases(intermediate_dir)
    counts = export_dataset(
        output_dir,
        train_cases=train_cases,
        test_cases=test_cases,
        label_names={"background": 0, "liver": 1, "cancer": 2},
        label_map={3: 1, 4: 0},
    )
    return counts, excluded


def _phase_id(series_name: str) -> int:
    try:
        return int(series_name.rsplit("_", maxsplit=1)[1])
    except (IndexError, ValueError) as error:
        raise ValueError(f"Unsupported HCC CT series name: {series_name}") from error


def _make_hcc_label_direction_positive(label: sitk.Image) -> sitk.Image:
    direction = label.GetDirection()
    if direction[0] != -1.0 and direction[4] != -1.0:
        return label
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


def _resample_to_reference(reference: sitk.Image, image: sitk.Image) -> sitk.Image:
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference)
    resampler.SetTransform(sitk.AffineTransform(reference.GetDimension()))
    resampler.SetDefaultPixelValue(0)
    resampler.SetInterpolator(_legacy_interpolator(image))
    return resampler.Execute(image)


def _legacy_interpolator(image: sitk.Image) -> int:
    if image.GetPixelID() in (sitk.sitkInt8, sitk.sitkInt16):
        return sitk.sitkNearestNeighbor
    return sitk.sitkLinear


def _same_geometry(first: sitk.Image, second: sitk.Image) -> bool:
    return (
        first.GetSize() == second.GetSize()
        and first.GetOrigin() == second.GetOrigin()
        and first.GetSpacing() == second.GetSpacing()
        and first.GetDirection() == second.GetDirection()
    )
