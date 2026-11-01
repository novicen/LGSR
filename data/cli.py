from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from data.datasets.hcc_tace import (
    prepare_hcc_intermediate_dataset,
    prepare_hcc_phase2_dataset,
)
from data.datasets.task03_liver import prepare_task03_dataset
from data.datasets.waw_tace import (
    prepare_waw_intermediate_dataset,
    prepare_waw_phase2_dataset,
)
from data.folds import generate_nnunet_folds
from data.validation import label_values_from_metadata, validate_dataset


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Prepare LGSR nnU-Net datasets.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    hcc_intermediate = subparsers.add_parser("hcc-intermediate")
    hcc_intermediate.add_argument("--normalized-dir", type=Path, required=True)
    hcc_intermediate.add_argument("--output-dir", type=Path, required=True)

    hcc_phase2 = subparsers.add_parser("hcc-phase2")
    hcc_phase2.add_argument("--intermediate-dir", type=Path, required=True)
    hcc_phase2.add_argument("--output-dir", type=Path, required=True)

    waw_intermediate = subparsers.add_parser("waw-intermediate")
    waw_intermediate.add_argument("--waw-root", type=Path, required=True)
    waw_intermediate.add_argument("--output-dir", type=Path, required=True)

    waw_phase2 = subparsers.add_parser("waw-phase2")
    waw_phase2.add_argument("--intermediate-dir", type=Path, required=True)
    waw_phase2.add_argument("--output-dir", type=Path, required=True)

    task03 = subparsers.add_parser("task03")
    task03.add_argument("--source-dir", type=Path, required=True)
    task03.add_argument("--output-dir", type=Path, required=True)

    folds = subparsers.add_parser("folds")
    folds.add_argument("--dataset-dir", type=Path, required=True)
    folds.add_argument("--overwrite", action="store_true")

    validate = subparsers.add_parser("validate")
    validate.add_argument("--dataset-dir", type=Path, required=True)
    validate.add_argument("--check-label-values", action="store_true")

    args = parser.parse_args(argv)
    if args.command == "hcc-intermediate":
        result = prepare_hcc_intermediate_dataset(args.normalized_dir, args.output_dir)
    elif args.command == "hcc-phase2":
        result = prepare_hcc_phase2_dataset(args.intermediate_dir, args.output_dir)
    elif args.command == "waw-intermediate":
        result = prepare_waw_intermediate_dataset(args.waw_root, args.output_dir)
    elif args.command == "waw-phase2":
        result = prepare_waw_phase2_dataset(args.intermediate_dir, args.output_dir)
    elif args.command == "task03":
        result = prepare_task03_dataset(args.source_dir, args.output_dir)
    elif args.command == "folds":
        output_path = args.dataset_dir / "splits_final.json"
        if output_path.exists() and not args.overwrite:
            raise FileExistsError(f"{output_path} exists; pass --overwrite to replace it")
        case_ids = _training_case_ids(args.dataset_dir)
        result = generate_nnunet_folds(case_ids)
        output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        result = {"folds": len(result), "path": str(output_path)}
    else:
        metadata_path = args.dataset_dir / "dataset.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        result = validate_dataset(
            args.dataset_dir,
            allowed_label_values=(
                label_values_from_metadata(metadata.get("labels", {}))
                if args.check_label_values
                else None
            ),
        )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def _training_case_ids(dataset_dir: Path) -> list[str]:
    suffix = "_0002.nii.gz"
    image_paths = sorted((dataset_dir / "imagesTr").glob(f"*{suffix}"))
    case_ids = [path.name[: -len(suffix)] for path in image_paths]
    if not case_ids:
        raise ValueError(f"No phase-2 training images in {dataset_dir / 'imagesTr'}")
    if len(set(case_ids)) != len(case_ids):
        raise ValueError(f"Duplicate training case IDs in {dataset_dir / 'imagesTr'}")
    return case_ids


if __name__ == "__main__":
    raise SystemExit(main())
