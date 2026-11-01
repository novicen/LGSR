from __future__ import annotations

import random
from collections.abc import Sequence


def legacy_train_test_split(
    case_ids: Sequence[str], ratio: float = 0.7, seed: int = 42
) -> tuple[list[str], list[str]]:
    """Reproduce the project's random.sample split without changing caller state."""
    ids = list(case_ids)
    train_ids = random.Random(seed).sample(ids, int(len(ids) * ratio))
    train_set = set(train_ids)
    test_ids = [case_id for case_id in ids if case_id not in train_set]
    return train_ids, test_ids
