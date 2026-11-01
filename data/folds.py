from __future__ import annotations

from collections.abc import Sequence

from sklearn.model_selection import KFold


def generate_nnunet_folds(
    case_ids: Sequence[str], n_splits: int = 5, seed: int = 12345
) -> list[dict[str, list[str]]]:
    ids = sorted(case_ids)
    if len(set(ids)) != len(ids):
        raise ValueError("Training case IDs must be unique")
    if n_splits < 2 or len(ids) < n_splits:
        raise ValueError(f"Need at least {n_splits} training cases for cross-validation")
    folds = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return [
        {
            "train": [ids[index] for index in train_indices],
            "val": [ids[index] for index in validation_indices],
        }
        for train_indices, validation_indices in folds.split(ids)
    ]
