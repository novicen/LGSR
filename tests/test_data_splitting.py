from __future__ import annotations

import random

from data.splitting import legacy_train_test_split


def test_legacy_train_test_split_matches_python_random_sample_order():
    case_ids = [f"HCC_{case_id:03d}" for case_id in range(1, 11)]
    original = list(case_ids)

    train_ids, test_ids = legacy_train_test_split(case_ids, ratio=0.7, seed=42)

    assert train_ids == [
        "HCC_002", "HCC_001", "HCC_005", "HCC_010", "HCC_007", "HCC_006", "HCC_009"
    ]
    assert test_ids == ["HCC_003", "HCC_004", "HCC_008"]
    assert case_ids == original


def test_legacy_train_test_split_reseeds_for_each_dataset():
    first_ids = [f"first_{i}" for i in range(10)]
    second_ids = [f"second_{i}" for i in range(10)]

    first_train, _ = legacy_train_test_split(first_ids, ratio=0.7, seed=42)
    random.random()
    second_train, _ = legacy_train_test_split(second_ids, ratio=0.7, seed=42)

    assert [case.removeprefix("first_") for case in first_train] == [
        case.removeprefix("second_") for case in second_train
    ]


def test_legacy_train_test_split_uses_floor_for_train_count():
    train_ids, test_ids = legacy_train_test_split([f"case_{i}" for i in range(11)])

    assert len(train_ids) == 7
    assert len(test_ids) == 4
    assert set(train_ids).isdisjoint(test_ids)
