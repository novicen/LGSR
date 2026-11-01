from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


RELEASE_ROOT = Path(__file__).resolve().parents[1]
for source_path in (RELEASE_ROOT / "src", RELEASE_ROOT / "vendor"):
    sys.path.insert(0, str(source_path))


def _required_paths() -> None:
    required = ("nnUNet_raw", "nnUNet_preprocessed", "nnUNet_results")
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        raise SystemExit("Set these nnU-Net paths before training: " + ", ".join(missing))


def _build_trainer(dataset, configuration, fold, plans_identifier, device):
    from batchgenerators.utilities.file_and_folder_operations import join, load_json
    from nnunetv2.paths import nnUNet_preprocessed
    from nnunetv2.training.nnUNetTrainer.nnUNetTrainer_lgsr import nnUNetTrainer_lgsr
    from nnunetv2.utilities.dataset_name_id_conversion import maybe_convert_to_dataset_name

    dataset_name = maybe_convert_to_dataset_name(dataset)
    preprocessed = join(nnUNet_preprocessed, dataset_name)
    plans = load_json(join(preprocessed, plans_identifier + ".json"))
    dataset_json = load_json(join(preprocessed, "dataset.json"))
    return nnUNetTrainer_lgsr(
        plans=plans,
        configuration=configuration,
        fold=fold,
        dataset_json=dataset_json,
        device=device,
    )


def main() -> None:
    _required_paths()
    parser = argparse.ArgumentParser(description="Train or validate the LGSR nnU-Net model.")
    parser.add_argument("dataset", help="nnU-Net dataset name or ID")
    parser.add_argument("configuration", help="For example 3d_fullres")
    parser.add_argument("fold", type=int)
    parser.add_argument("-p", "--plans", default="nnUNetPlans")
    parser.add_argument("--continue-training", action="store_true")
    parser.add_argument("--validation-only", action="store_true")
    parser.add_argument("--pretrained-weights")
    parser.add_argument("--save-probabilities", action="store_true")
    parser.add_argument("--disable-checkpointing", action="store_true")
    parser.add_argument("--device", choices=("cuda", "cpu", "mps"), default="cuda")
    args = parser.parse_args()
    if args.continue_training and args.pretrained_weights:
        parser.error("--continue-training and --pretrained-weights cannot be combined")

    import torch
    from batchgenerators.utilities.file_and_folder_operations import join, isfile
    from nnunetv2.run.load_pretrained_weights import load_pretrained_weights

    device = torch.device(args.device)
    if device.type == "cuda":
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    trainer = _build_trainer(args.dataset, args.configuration, args.fold, args.plans, device)
    trainer.disable_checkpointing = bool(args.disable_checkpointing)

    if args.continue_training or args.validation_only:
        candidates = (
            ("checkpoint_final.pth", "checkpoint_latest.pth", "checkpoint_best.pth")
            if args.continue_training
            else ("checkpoint_final.pth",)
        )
        checkpoint = next(
            (join(trainer.output_folder, name) for name in candidates if isfile(join(trainer.output_folder, name))),
            None,
        )
        if checkpoint is None:
            raise FileNotFoundError(f"No checkpoint found in {trainer.output_folder}")
        trainer.load_checkpoint(checkpoint)
    elif args.pretrained_weights:
        trainer.initialize()
        load_pretrained_weights(trainer.network, args.pretrained_weights, verbose=True)

    if not args.validation_only:
        trainer.run_training()
    trainer.perform_actual_validation(save_probabilities=args.save_probabilities)


if __name__ == "__main__":
    main()
