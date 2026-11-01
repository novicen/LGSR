from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch

RELEASE_ROOT = Path(__file__).resolve().parents[1]
for source_path in (RELEASE_ROOT / "src", RELEASE_ROOT / "vendor"):
    sys.path.insert(0, str(source_path))

from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LGSR nnU-Net inference.")
    parser.add_argument("-i", required=True, help="Input image folder in nnU-Net format.")
    parser.add_argument("-o", required=True, help="Output segmentation folder.")
    parser.add_argument("-m", required=True, help="LGSR model folder.")
    parser.add_argument("-f", nargs="+", default=[4], help="Fold ids.")
    parser.add_argument("-chk", default="checkpoint_final.pth")
    parser.add_argument("--save_probabilities", action="store_true")
    parser.add_argument("--continue_prediction", action="store_true")
    parser.add_argument("--disable_tta", action="store_true")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=not args.disable_tta,
        perform_everything_on_device=device.type == "cuda",
        device=device,
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=True,
    )
    folds = [int(v) if v != "all" else v for v in args.f]
    predictor.initialize_from_trained_model_folder(args.m, folds, checkpoint_name=args.chk)
    Path(args.o).mkdir(parents=True, exist_ok=True)
    predictor.predict_from_files(
        args.i,
        args.o,
        save_probabilities=args.save_probabilities,
        overwrite=not args.continue_prediction,
        num_processes_preprocessing=1,
        num_processes_segmentation_export=1,
    )


if __name__ == "__main__":
    main()
