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


WAW_LABELS = {"background": 0, "liver": 1, "cancer": 2}
WAW_INTERMEDIATE_LABELS = {"background": 0, "liver": 1, "cancer": 2}


@dataclass(frozen=True)
class WAWCase:
    case_id: str
    tumor_dir: Path
    organ_dir: Path
    ct_dir: Path


def build_waw_cases(waw_root: str | Path) -> list[WAWCase]:
    waw_root = Path(waw_root)
    tumor_root = waw_root / "tumor_masks_wawtace_v1_08_05_2024"
    organ_root = waw_root / "organ_masks_wawtace_09_05_24"
    cases_by_id: dict[str, WAWCase] = {}
    for index in range(1, 5):
        ct_root = waw_root / f"ct_scans_{index}_4_wawtace_09_05_24"
        if not ct_root.is_dir():
            continue
        for ct_dir in ct_root.iterdir():
            if ct_dir.is_dir():
                case_id = ct_dir.name
                cases_by_id[case_id] = WAWCase(
                    case_id,
                    tumor_root / case_id,
                    organ_root / case_id,
                    ct_dir,
                )
    cases = sorted(cases_by_id.values(), key=lambda case: int(case.case_id))
    if not cases:
        raise FileNotFoundError(f"No WAW CT cases found under {waw_root}")
    for case in cases:
        if not case.tumor_dir.is_dir() or not case.organ_dir.is_dir():
            raise FileNotFoundError(f"{case.case_id}: missing WAW tumor or organ masks")
    return cases


def split_waw_cases(
    cases: list[WAWCase], ratio: float = 0.7, seed: int = 42
) -> tuple[list[WAWCase], list[WAWCase]]:
    by_id = {case.case_id: case for case in cases}
    train_ids, test_ids = legacy_train_test_split(list(by_id), ratio=ratio, seed=seed)
    return (
        sorted((by_id[case_id] for case_id in train_ids), key=lambda case: int(case.case_id)),
        sorted((by_id[case_id] for case_id in test_ids), key=lambda case: int(case.case_id)),
    )


def merge_waw_label(
    organ_mask_path: str | Path, tumor_mask_paths: list[str | Path]
) -> sitk.Image:
    organ_mask = sitk.ReadImage(str(organ_mask_path))
    organ_array = sitk.GetArrayFromImage(organ_mask)
    organ_array[organ_array != 5] = 0
    organ_array[organ_array == 5] = 1
    liver = sitk.GetImageFromArray(organ_array)
    liver.CopyInformation(organ_mask)

    import nrrd

    tumor_masks = []
    for path in tumor_mask_paths:
        tumor_array, header = nrrd.read(str(path))
        tumor = sitk.GetImageFromArray(tumor_array.transpose(2, 1, 0))
        tumor.SetDirection(np.sign(header["space directions"].flatten()).tolist())
        tumor.SetOrigin(tuple(header["space origin"]))
        tumor.SetSpacing(tuple(np.abs(header["space directions"].diagonal())))
        tumor_masks.append(tumor)

    needs_resampling = any(not _same_geometry(liver, mask, tolerance=1e-3) for mask in tumor_masks)
    combined_tumor = np.zeros_like(sitk.GetArrayViewFromImage(liver))
    for mask in tumor_masks:
        aligned = _resample_to_reference(liver, mask) if needs_resampling else mask
        combined_tumor = np.maximum(sitk.GetArrayFromImage(aligned), combined_tumor)

    liver_array = sitk.GetArrayFromImage(liver)
    liver_array[combined_tumor > 0] = 2
    result = sitk.GetImageFromArray(liver_array)
    result.CopyInformation(liver)
    return result


def convert_waw_case(case: WAWCase) -> tuple[sitk.Image, dict[int, sitk.Image]]:
    tumor_paths = sorted(case.tumor_dir.glob("*.nrrd"))
    if not tumor_paths:
        raise FileNotFoundError(f"{case.case_id}: no WAW tumor masks found")
    tumor_phases = sorted({path.name.split("_")[1] for path in tumor_paths})
    if len(tumor_phases) != 1:
        raise ValueError(f"{case.case_id}: expected one tumor reference phase, got {tumor_phases}")
    tumor_phase = tumor_phases[0]
    organ_paths = sorted(
        path for path in case.organ_dir.glob("*.nii.gz")
        if path.name.split("_")[1] == tumor_phase
    )
    if not organ_paths:
        raise FileNotFoundError(f"{case.case_id}: organ mask for phase {tumor_phase} is missing")
    label = merge_waw_label(organ_paths[0], tumor_paths)

    phase_images: dict[int, sitk.Image] = {}
    for image_path in sorted(case.ct_dir.glob("*.nii.gz"), key=lambda path: path.name.split("_")[1]):
        phase = int(image_path.name.split("_")[1])
        image = sitk.ReadImage(str(image_path))
        if phase == int(tumor_phase):
            fixed = _resample_to_reference(label, image) if not _same_geometry(label, image, 1e-3) else image
            phase_images[phase] = fixed
        else:
            phase_images[phase] = image
    if int(tumor_phase) not in phase_images:
        raise FileNotFoundError(f"{case.case_id}: phase {tumor_phase} CT is missing")

    fixed = phase_images[int(tumor_phase)]
    # Legacy registration estimated transforms but exported resamples without them.
    aligned = {
        phase: _resample_to_reference(fixed, image)
        for phase, image in phase_images.items()
        if phase != int(tumor_phase)
    }
    aligned[int(tumor_phase)] = fixed
    return label, aligned


def prepare_waw_intermediate_dataset(
    waw_root: str | Path, output_dir: str | Path
) -> dict[str, int]:
    train_cases, test_cases = split_waw_cases(build_waw_cases(waw_root))
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(f"Output directory already exists: {output_dir}")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.tmp-", dir=output_dir.parent))
    try:
        for split, cases in (("Tr", train_cases), ("Ts", test_cases)):
            image_dir = staging / f"images{split}"
            label_dir = staging / f"labels{split}"
            image_dir.mkdir()
            label_dir.mkdir()
            for case in cases:
                label, phase_images = convert_waw_case(case)
                for phase, image in phase_images.items():
                    sitk.WriteImage(
                        image,
                        str(image_dir / f"waw_{int(case.case_id):03d}_{phase:04d}.nii.gz"),
                    )
                sitk.WriteImage(
                    label, str(label_dir / f"waw_{int(case.case_id):03d}.nii.gz")
                )
        metadata = {
            "channel_names": {"2": "CT"},
            "labels": WAW_INTERMEDIATE_LABELS,
            "numTraining": len(train_cases),
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
    return {"Tr": len(train_cases), "Ts": len(test_cases)}


def build_waw_phase2_cases(
    intermediate_dir: str | Path,
) -> tuple[list[ExportCase], list[ExportCase], dict[str, list[str]]]:
    return build_phase_cases(intermediate_dir, phase_id=2)


def prepare_waw_phase2_dataset(
    intermediate_dir: str | Path, output_dir: str | Path
) -> tuple[dict[str, int], dict[str, list[str]]]:
    train_cases, test_cases, excluded = build_waw_phase2_cases(intermediate_dir)
    counts = export_dataset(
        output_dir,
        train_cases=train_cases,
        test_cases=test_cases,
        label_names=WAW_LABELS,
        label_map={3: 1, 4: 0},
    )
    return counts, excluded


def _resample_to_reference(
    reference: sitk.Image, image: sitk.Image, tolerance: float = 0.0
) -> sitk.Image:
    resampler = sitk.ResampleImageFilter()
    resampler.SetReferenceImage(reference)
    resampler.SetTransform(sitk.AffineTransform(reference.GetDimension()))
    resampler.SetDefaultPixelValue(0)
    resampler.SetInterpolator(_legacy_interpolator(image))
    return resampler.Execute(image)


def _legacy_interpolator(image: sitk.Image) -> int:
    return (
        sitk.sitkNearestNeighbor
        if image.GetPixelID() in (sitk.sitkInt8, sitk.sitkInt16)
        else sitk.sitkLinear
    )


def _same_geometry(first: sitk.Image, second: sitk.Image, tolerance: float = 0.0) -> bool:
    if first.GetSize() != second.GetSize():
        return False
    for left, right in (
        (first.GetOrigin(), second.GetOrigin()),
        (first.GetSpacing(), second.GetSpacing()),
        (first.GetDirection(), second.GetDirection()),
    ):
        if any(abs(a - b) > tolerance for a, b in zip(left, right)):
            return False
    return True
