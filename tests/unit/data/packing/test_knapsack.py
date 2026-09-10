import pytest

from nemo_rl.data.packing import (
    BalancedGreedyKnapsackPacker,
    GreedyKnapsackPacker,
    PackingAlgorithm,
    get_packer,
)


@pytest.mark.parametrize(
    ("algorithm", "packer_type"),
    [
        (PackingAlgorithm.GREEDY_KNAPSACK, GreedyKnapsackPacker),
        (PackingAlgorithm.BALANCED_GREEDY_KNAPSACK, BalancedGreedyKnapsackPacker),
    ],
)
def test_factory_builds_knapsack_packers(algorithm, packer_type) -> None:
    assert isinstance(get_packer(algorithm, 10), packer_type)
    assert isinstance(get_packer(algorithm.value, 10), packer_type)


def test_greedy_knapsack_takes_largest_remaining_item_that_fits() -> None:
    assert GreedyKnapsackPacker(10).pack([6, 5, 4, 3, 2]) == [
        [0, 2],
        [1, 3, 4],
    ]


def test_balanced_knapsack_spreads_equal_items_across_minimum_bins() -> None:
    packer = BalancedGreedyKnapsackPacker(8, balanced_knapsack_delta=0)

    assert packer.pack([4, 4, 4, 4]) == [[0, 2], [1, 3]]


@pytest.mark.parametrize(
    "packer",
    [GreedyKnapsackPacker(10), BalancedGreedyKnapsackPacker(10)],
)
def test_knapsack_packers_keep_common_interface_constraints(packer) -> None:
    packer.max_sequences_per_bin = 1
    assert packer.pack([4, 3, 2]) == [[0], [1], [2]]
    with pytest.raises(ValueError, match="exceeds bin capacity"):
        packer.pack([11])
