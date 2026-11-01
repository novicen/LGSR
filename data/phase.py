from __future__ import annotations

from pathlib import Path

from data.export import ExportCase


def build_phase_cases(
    dataset_dir: str | Path, phase_id: int = 2
) -> tuple[list[ExportCase], list[ExportCase], dict[str, list[str]]]:
    """Pair one image channel with labels, retaining only complete cases."""
    dataset_dir = Path(dataset_dir)
    channel_suffix = f"_{phase_id:04d}.nii.gz"
    result: dict[str, list[ExportCase]] = {}
    excluded: dict[str, list[str]] = {}
    for split in ("Tr", "Ts"):
        image_dir = dataset_dir / f"images{split}"
        label_dir = dataset_dir / f"labels{split}"
        labels = {path.name[:-7]: path for path in label_dir.glob("*.nii.gz")}
        images: dict[str, Path] = {}
        for image_path in sorted(image_dir.glob(f"*{channel_suffix}")):
            case_id = image_path.name[: -len(channel_suffix)]
            if case_id in images:
                raise ValueError(f"{case_id}: duplicate phase-{phase_id} images")
            images[case_id] = image_path

        matched = sorted(images.keys() & labels.keys())
        excluded[split] = sorted(images.keys() ^ labels.keys())
        result[split] = [
            ExportCase(case_id, images[case_id], labels[case_id]) for case_id in matched
        ]
    return result["Tr"], result["Ts"], excluded
