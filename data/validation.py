from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import SimpleITK as sitk


def validate_dataset(
    dataset_dir: str | Path,
    *,
    allowed_label_values: set[int] | None = None,
) -> dict[str, int]:
    """Check nnU-Net case pairing, phase naming, labels, and spatial geometry."""
    dataset_dir = Path(dataset_dir)
    metadata_path = dataset_dir / "dataset.json"
    if not metadata_path.is_file():
        raise ValueError(f"Missing dataset.json in {dataset_dir}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    channel_names = metadata.get("channel_names", {})
    if len(channel_names) != 1:
        raise ValueError("Expected exactly one input channel")
    channel_id = next(iter(channel_names))
    if not channel_id.isdigit():
        raise ValueError(f"Invalid channel ID: {channel_id}")

    counts: dict[str, int] = {}
    for split in ("Tr", "Ts"):
        image_dir = dataset_dir / f"images{split}"
        label_dir = dataset_dir / f"labels{split}"
        images = sorted(image_dir.glob("*.nii.gz")) if image_dir.is_dir() else []
        labels = sorted(label_dir.glob("*.nii.gz")) if label_dir.is_dir() else []
        image_cases = {}
        for image_path in images:
            suffix = f"_{int(channel_id):04d}.nii.gz"
            if not image_path.name.endswith(suffix):
                raise ValueError(f"Unexpected channel suffix: {image_path.name}")
            case_id = image_path.name[: -len(suffix)]
            if case_id in image_cases:
                raise ValueError(f"Duplicate {split} image case: {case_id}")
            image_cases[case_id] = image_path
        label_cases = {path.name[:-7]: path for path in labels}
        if len(label_cases) != len(labels):
            raise ValueError(f"Duplicate {split} label case IDs")
        if image_cases.keys() != label_cases.keys():
            missing_labels = sorted(image_cases.keys() - label_cases.keys())
            missing_images = sorted(label_cases.keys() - image_cases.keys())
            raise ValueError(
                f"{split} image/label case mismatch; missing labels={missing_labels}, "
                f"missing images={missing_images}"
            )

        for case_id, image_path in image_cases.items():
            image_geometry = read_image_geometry(image_path)
            label_path = label_cases[case_id]
            label_geometry = read_image_geometry(label_path)
            _validate_geometry(case_id, image_geometry, label_geometry)
            if allowed_label_values is not None:
                label = sitk.ReadImage(str(label_path))
                values = set(np.unique(sitk.GetArrayViewFromImage(label)).tolist())
                unexpected = values - allowed_label_values
                if unexpected:
                    raise ValueError(
                        f"{case_id}: unexpected label values {sorted(unexpected)}"
                    )
        counts[split] = len(images)

    if metadata.get("numTraining") != counts["Tr"]:
        raise ValueError(
            f"dataset.json numTraining={metadata.get('numTraining')} "
            f"does not match imagesTr count={counts['Tr']}"
        )
    return counts


def label_values_from_metadata(labels: dict[str, object]) -> set[int]:
    if all(str(key).isdigit() for key in labels):
        return {int(key) for key in labels}
    values = set()
    for value in labels.values():
        if not isinstance(value, int):
            raise ValueError(f"Invalid label value in dataset.json: {value!r}")
        values.add(value)
    return values


def read_image_geometry(path: str | Path) -> tuple[tuple, tuple, tuple, tuple]:
    reader = sitk.ImageFileReader()
    reader.SetFileName(str(path))
    reader.ReadImageInformation()
    return (
        reader.GetSize(),
        reader.GetOrigin(),
        reader.GetSpacing(),
        reader.GetDirection(),
    )


def _validate_geometry(case_id: str, image: tuple, label: tuple) -> None:
    if image[0] != label[0]:
        raise ValueError(f"{case_id}: image/label size geometry mismatch")
    for name, image_values, label_values in zip(
        ("origin", "spacing", "direction"), image[1:], label[1:]
    ):
        if not np.allclose(image_values, label_values, atol=1e-6, rtol=0.0):
            raise ValueError(f"{case_id}: image/label {name} geometry mismatch")
