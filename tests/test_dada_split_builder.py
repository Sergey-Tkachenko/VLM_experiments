import random

from task_2_evaluation_suite.dada_split_builder import (
    allocate_counts_largest_remainder,
    stratified_split_by_class,
)


def test_allocate_counts_largest_remainder() -> None:
    ratios = {"train": 3.0, "val": 1.0, "test": 3.0}
    assert allocate_counts_largest_remainder(20, ratios) == {"train": 9, "val": 3, "test": 8}


def test_stratified_split_by_class_counts() -> None:
    ratios = {"train": 3.0, "val": 1.0, "test": 3.0}
    pairs = [("1", f"{idx:03d}") for idx in range(10)]
    pairs += [("2", f"{idx:03d}") for idx in range(5)]
    rng = random.Random(123)

    splits = stratified_split_by_class(pairs, ratios, rng)

    assert sum(len(items) for items in splits.values()) == 15
    assert len({pair for items in splits.values() for pair in items}) == 15

    counts_by_type = {"1": {"train": 0, "val": 0, "test": 0}, "2": {"train": 0, "val": 0, "test": 0}}
    for split_name, items in splits.items():
        for type_id, _ in items:
            counts_by_type[type_id][split_name] += 1

    assert counts_by_type["1"] == {"train": 4, "val": 2, "test": 4}
    assert counts_by_type["2"] == {"train": 2, "val": 1, "test": 2}
