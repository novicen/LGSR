#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "usage: $0 DATASET CONFIGURATION FOLD [extra nnUNet arguments...]" >&2
  exit 2
fi

release_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="$release_root/src:$release_root/vendor:${PYTHONPATH:-}"
: "${nnUNet_raw:?Set nnUNet_raw to the dataset root}"
: "${nnUNet_preprocessed:?Set nnUNet_preprocessed to the preprocessed root}"
: "${nnUNet_results:?Set nnUNet_results to the model output root}"
export nnUNet_compile="${nnUNet_compile:-f}"
export nnUNet_n_proc_DA="${nnUNet_n_proc_DA:-0}"

exec python "$release_root/scripts/train.py" "$@"
