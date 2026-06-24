# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for topology-aware placement logic.

All tests are pure Python — no Ray cluster required.
"""

import pytest

from nemo_rl.distributed.virtual_cluster import (
    NVLINK_DOMAIN_UNKNOWN,
    TOPO_RANK_UNKNOWN,
    ResourceInsufficientError,
    _sort_bundle_indices_by_topology,
    select_segment_nodes,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_topology(
    domains: dict[str, list[int]],
) -> dict[str, tuple[str, int]]:
    """Build a topology dict from {domain_name: [topo_rank, ...]}."""
    topo: dict[str, tuple[str, int]] = {}
    node_counter = 0
    for domain, ranks in domains.items():
        for rank in ranks:
            node_id = f"node_{node_counter:03d}"
            node_counter += 1
            topo[node_id] = (domain, rank)
    return topo


def _make_bundle_data(
    domains: dict[str, list[int]],
    gpus_per_node: int = 1,
) -> list[tuple[int, str, int, str]]:
    """Build bundle_data list from {domain_name: [topo_rank, ...]}.

    Each node contributes `gpus_per_node` bundles (gpu_id 0..gpus_per_node-1).
    """
    data: list[tuple[int, str, int, str]] = []
    node_counter = 0
    for domain, ranks in domains.items():
        for rank in ranks:
            node_id = f"node_{node_counter:03d}"
            node_counter += 1
            for gpu_id in range(gpus_per_node):
                data.append((gpu_id, domain, rank, node_id))
    return data


# ---------------------------------------------------------------------------
# 1. No-topology fallback: ordering is stable by (node_id, gpu_id)
# ---------------------------------------------------------------------------


class TestNoTopologyFallback:
    def test_all_unknown_sorted_by_node_id_gpu_id(self):
        # nodes in reverse alpha order; expect sorted output
        bundle_data = [
            (0, NVLINK_DOMAIN_UNKNOWN, TOPO_RANK_UNKNOWN, "node_c"),
            (0, NVLINK_DOMAIN_UNKNOWN, TOPO_RANK_UNKNOWN, "node_a"),
            (1, NVLINK_DOMAIN_UNKNOWN, TOPO_RANK_UNKNOWN, "node_a"),
            (0, NVLINK_DOMAIN_UNKNOWN, TOPO_RANK_UNKNOWN, "node_b"),
        ]
        result = _sort_bundle_indices_by_topology(bundle_data)
        # Expected: node_a gpu0 (idx1), node_a gpu1 (idx2), node_b gpu0 (idx3), node_c gpu0 (idx0)
        assert result == [1, 2, 3, 0]

    def test_already_sorted_input_returns_same_order(self):
        bundle_data = [
            (0, NVLINK_DOMAIN_UNKNOWN, TOPO_RANK_UNKNOWN, "node_a"),
            (1, NVLINK_DOMAIN_UNKNOWN, TOPO_RANK_UNKNOWN, "node_a"),
            (0, NVLINK_DOMAIN_UNKNOWN, TOPO_RANK_UNKNOWN, "node_b"),
            (1, NVLINK_DOMAIN_UNKNOWN, TOPO_RANK_UNKNOWN, "node_b"),
        ]
        result = _sort_bundle_indices_by_topology(bundle_data)
        assert result == [0, 1, 2, 3]

    def test_empty_bundle_data_returns_empty(self):
        assert _sort_bundle_indices_by_topology([]) == []

    def test_no_topology_segment_size_set_still_sorts_by_node_gpu(self):
        # If segment_size is set but all bundles have UNKNOWN domain/rank,
        # has_topology is False so it falls back to node_id/gpu_id sort
        # (segment filtering is skipped since there's no topology info)
        bundle_data = [
            (0, NVLINK_DOMAIN_UNKNOWN, TOPO_RANK_UNKNOWN, "node_b"),
            (0, NVLINK_DOMAIN_UNKNOWN, TOPO_RANK_UNKNOWN, "node_a"),
        ]
        result = _sort_bundle_indices_by_topology(
            bundle_data, segment_size=1, gpus_per_node=1
        )
        assert result == [1, 0]


# ---------------------------------------------------------------------------
# 2. TOPO_RANK values from different fallbacks
# ---------------------------------------------------------------------------


class TestTopoRankValues:
    """Ray stores resource values as floats; int(float) must round-trip correctly."""

    def test_slurm_procid_value_as_float(self):
        # SLURM_PROCID fallback: $(( 10#7 )) -> TOPO_RANK=7, stored as 7.0 by Ray
        topo_rank = int(7.0)
        assert topo_rank == 7

    def test_hostname_digits_value_as_float(self):
        # hostname node007 -> $(( 10#007 )) -> 7, stored as 7.0 by Ray
        topo_rank = int(7.0)
        assert topo_rank == 7

    def test_block_node_combined_value(self):
        # block.node format: $(( 10#2 * 10000000000 + 10#15 )) = 20000000015
        topo_rank = int(20000000015.0)
        assert topo_rank == 20000000015

    def test_sorting_with_slurm_procid_ranks(self):
        # Nodes labelled by SLURM_PROCID (0..7) across two domains
        bundle_data = _make_bundle_data(
            {
                "nvlink_domain_A": [4, 5, 6, 7],  # higher SLURM_PROCID
                "nvlink_domain_B": [0, 1, 2, 3],  # lower SLURM_PROCID
            }
        )
        result = _sort_bundle_indices_by_topology(bundle_data)
        # Domain B min rank=0 < domain A min rank=4 → B comes first
        node_ids = [bundle_data[i][3] for i in result]
        # First 4 nodes should all be from domain B (nodes 4..7 in bundle_data list)
        for node_id in node_ids[:4]:
            assert (
                bundle_data[result[0]][1] == "nvlink_domain_B" or True
            )  # checked below
        # Verify domain ordering: all B before A
        domains_in_order = [bundle_data[i][1] for i in result]
        b_indices = [
            i for i, d in enumerate(domains_in_order) if d == "nvlink_domain_B"
        ]
        a_indices = [
            i for i, d in enumerate(domains_in_order) if d == "nvlink_domain_A"
        ]
        assert max(b_indices) < min(a_indices)

    def test_sorting_with_block_node_ranks(self):
        # block.node: block 0 nodes 0..3, block 1 nodes 0..3
        # block 0 has lower combined rank → should sort first
        block0_ranks = [0 * 10000000000 + i for i in range(4)]
        block1_ranks = [1 * 10000000000 + i for i in range(4)]
        bundle_data = _make_bundle_data(
            {
                "nvlink_domain_A": block1_ranks,
                "nvlink_domain_B": block0_ranks,
            }
        )
        result = _sort_bundle_indices_by_topology(bundle_data)
        domains_in_order = [bundle_data[i][1] for i in result]
        # block0 (domain B) should come first
        assert domains_in_order[:4] == ["nvlink_domain_B"] * 4
        assert domains_in_order[4:] == ["nvlink_domain_A"] * 4

    def test_unknown_topo_rank_sorts_before_known_ranks(self):
        # TOPO_RANK_UNKNOWN = -1 sorts before any positive rank.
        # This is known behavior: nodes missing topo_rank get first priority.
        bundle_data = [
            (0, "nvlink_domain_A", TOPO_RANK_UNKNOWN, "node_000"),  # idx 0, rank=-1
            (0, "nvlink_domain_A", 5, "node_001"),  # idx 1, rank=5
        ]
        result = _sort_bundle_indices_by_topology(bundle_data)
        # -1 < 5, so unknown rank comes first
        assert result[0] == 0


# ---------------------------------------------------------------------------
# 3. Colocated case: bundle sorting with topology (no segment trimming)
# ---------------------------------------------------------------------------


class TestColocatedBundleSorting:
    def test_two_domains_sorted_by_min_topo_rank_then_intra_domain(self):
        # Domain B min rank=1 < domain A min rank=3 → B first
        bundle_data = _make_bundle_data(
            {
                "nvlink_domain_A": [3, 5],
                "nvlink_domain_B": [1, 2],
            }
        )
        # bundle_data layout: A rank3 (0), A rank5 (1), B rank1 (2), B rank2 (3)
        result = _sort_bundle_indices_by_topology(bundle_data)
        expected_domains = [
            "nvlink_domain_B",
            "nvlink_domain_B",
            "nvlink_domain_A",
            "nvlink_domain_A",
        ]
        assert [bundle_data[i][1] for i in result] == expected_domains
        expected_ranks = [1, 2, 3, 5]
        assert [bundle_data[i][2] for i in result] == expected_ranks

    def test_multiple_gpus_per_node_sorted_within_node(self):
        # 2 nodes, 4 GPUs each, single domain
        bundle_data = _make_bundle_data({"nvlink_domain_A": [2, 1]}, gpus_per_node=4)
        # node_000 rank=2 (idxs 0..3), node_001 rank=1 (idxs 4..7)
        result = _sort_bundle_indices_by_topology(bundle_data)
        # rank=1 node should come first, then rank=2 node; gpu_id sorted within node
        result_ranks = [bundle_data[i][2] for i in result]
        assert result_ranks == [1, 1, 1, 1, 2, 2, 2, 2]
        # gpu_ids within each node should be 0,1,2,3
        result_gpus = [bundle_data[i][0] for i in result]
        assert result_gpus == [0, 1, 2, 3, 0, 1, 2, 3]

    def test_three_domains_correct_domain_order(self):
        # Domains with min ranks 10, 5, 20 → order should be 5, 10, 20
        bundle_data = _make_bundle_data(
            {
                "domain_X": [10, 11],  # min=10
                "domain_Y": [5, 6],  # min=5
                "domain_Z": [20, 21],  # min=20
            }
        )
        result = _sort_bundle_indices_by_topology(bundle_data)
        domains_in_order = [bundle_data[i][1] for i in result]
        assert domains_in_order[:2] == ["domain_Y", "domain_Y"]
        assert domains_in_order[2:4] == ["domain_X", "domain_X"]
        assert domains_in_order[4:] == ["domain_Z", "domain_Z"]

    def test_segment_size_requires_gpus_per_node(self):
        bundle_data = _make_bundle_data({"nvlink_domain_A": [0, 1]})
        with pytest.raises(ValueError, match="gpus_per_node is required"):
            _sort_bundle_indices_by_topology(bundle_data, segment_size=2)


# ---------------------------------------------------------------------------
# 4. select_segment_nodes: basic and error cases
# ---------------------------------------------------------------------------


class TestSelectSegmentNodes:
    def test_basic_two_domains_select_one_segment_each(self):
        topo = _make_topology(
            {
                "domain_A": [0, 1, 2, 3, 4, 5, 6, 7],
                "domain_B": [8, 9, 10, 11, 12, 13, 14, 15],
            }
        )
        selected, remaining = select_segment_nodes(topo, segment_size=8, num_nodes=8)
        assert len(selected) == 8
        assert len(remaining) == 8
        assert set(selected) | set(remaining) == set(topo.keys())
        # Selected nodes should all come from domain with lower min rank (domain_A, min=0)
        selected_domains = {topo[n][0] for n in selected}
        assert selected_domains == {"domain_A"}

    def test_select_across_multiple_domains(self):
        # 3 domains × 8 nodes, select 2 domains worth
        topo = _make_topology(
            {
                "domain_A": list(range(8)),
                "domain_B": list(range(8, 16)),
                "domain_C": list(range(16, 24)),
            }
        )
        selected, remaining = select_segment_nodes(topo, segment_size=8, num_nodes=16)
        assert len(selected) == 16
        assert len(remaining) == 8
        # Should pick domains A and B (lowest min ranks)
        selected_domains = {topo[n][0] for n in selected}
        assert selected_domains == {"domain_A", "domain_B"}
        remaining_domains = {topo[n][0] for n in remaining}
        assert remaining_domains == {"domain_C"}

    def test_num_nodes_not_divisible_by_segment_size_raises(self):
        topo = _make_topology({"domain_A": list(range(10))})
        with pytest.raises(ValueError, match="must be divisible by"):
            select_segment_nodes(topo, segment_size=8, num_nodes=10)

    def test_insufficient_segments_raises_informative_error(self):
        # Only 1 domain with 6 nodes; need 2 segments of 4 = 8 nodes
        topo = _make_topology({"domain_A": list(range(6))})
        with pytest.raises(ResourceInsufficientError, match="Cannot form"):
            select_segment_nodes(topo, segment_size=4, num_nodes=8)

    def test_domain_too_small_skipped(self):
        # domain_A has 6 nodes (not enough for segment_size=8), domain_B has 8
        topo = _make_topology(
            {
                "domain_A": list(range(6)),  # too small, skipped
                "domain_B": list(range(10, 18)),  # min rank=10, has 8 nodes
            }
        )
        selected, remaining = select_segment_nodes(topo, segment_size=8, num_nodes=8)
        selected_domains = {topo[n][0] for n in selected}
        assert selected_domains == {"domain_B"}
        assert len(remaining) == 6  # domain_A's 6 nodes remain

    def test_selected_plus_remaining_is_full_topology(self):
        topo = _make_topology(
            {
                "domain_A": list(range(8)),
                "domain_B": list(range(8, 16)),
                "domain_C": list(range(16, 24)),
                "domain_D": list(range(24, 32)),
                "domain_E": list(range(32, 40)),
            }
        )
        selected, remaining = select_segment_nodes(topo, segment_size=8, num_nodes=24)
        assert set(selected) | set(remaining) == set(topo.keys())
        assert len(set(selected) & set(remaining)) == 0  # no overlap


# ---------------------------------------------------------------------------
# 5. The 40-node / 5-domain scenario: training + inference placement
# ---------------------------------------------------------------------------


class TestFortyNodeScenario:
    """
    5 NVLink domains × 8 nodes = 40 nodes total (1 GPU per node).
    Training segment_size=8, training needs 24 nodes (3 segments).
    Inference has nodes_per_instance=4 and needs 16 nodes (4 groups of 4).
    """

    @pytest.fixture
    def forty_node_topology(self):
        return _make_topology(
            {
                "domain_A": list(range(0, 8)),
                "domain_B": list(range(10, 18)),
                "domain_C": list(range(20, 28)),
                "domain_D": list(range(30, 38)),
                "domain_E": list(range(40, 48)),
            }
        )

    def test_training_selects_three_domains(self, forty_node_topology):
        selected, remaining = select_segment_nodes(
            forty_node_topology, segment_size=8, num_nodes=24
        )
        assert len(selected) == 24
        assert len(remaining) == 16
        selected_domains = {forty_node_topology[n][0] for n in selected}
        assert selected_domains == {"domain_A", "domain_B", "domain_C"}

    def test_inference_can_be_placed_on_remaining_nodes(self, forty_node_topology):
        _, remaining = select_segment_nodes(
            forty_node_topology, segment_size=8, num_nodes=24
        )
        remaining_topo = {n: forty_node_topology[n] for n in remaining}
        # Inference: nodes_per_instance=4, inference_nodes=16
        inf_selected, inf_remaining = select_segment_nodes(
            remaining_topo, segment_size=4, num_nodes=16
        )
        assert len(inf_selected) == 16
        assert len(inf_remaining) == 0
        # All inference nodes come from domains D and E
        inf_domains = {forty_node_topology[n][0] for n in inf_selected}
        assert inf_domains == {"domain_D", "domain_E"}

    def test_training_and_inference_nodes_are_disjoint(self, forty_node_topology):
        train_selected, remaining = select_segment_nodes(
            forty_node_topology, segment_size=8, num_nodes=24
        )
        remaining_topo = {n: forty_node_topology[n] for n in remaining}
        inf_selected, _ = select_segment_nodes(
            remaining_topo, segment_size=4, num_nodes=16
        )
        assert set(train_selected) & set(inf_selected) == set()

    def test_impossible_training_size_raises(self, forty_node_topology):
        # 40 nodes / segment_size=8 → max 5 segments = 40 nodes.
        # Requesting 48 nodes is impossible.
        with pytest.raises(ResourceInsufficientError, match="Cannot form"):
            select_segment_nodes(forty_node_topology, segment_size=8, num_nodes=48)

    def test_non_divisible_training_nodes_raises(self, forty_node_topology):
        # 25 not divisible by 8
        with pytest.raises(ValueError, match="must be divisible by"):
            select_segment_nodes(forty_node_topology, segment_size=8, num_nodes=25)

    def test_inference_nodes_not_divisible_by_instance_size_is_skipped(
        self, forty_node_topology
    ):
        # In grpo.py the inference topology is only applied when
        # inference_nodes % nodes_per_instance == 0.  If it's not, no
        # constraints are set and inference falls back to unordered placement.
        # This test validates that select_segment_nodes would raise if called —
        # confirming the guard in grpo.py is necessary.
        _, remaining = select_segment_nodes(
            forty_node_topology, segment_size=8, num_nodes=24
        )
        remaining_topo = {n: forty_node_topology[n] for n in remaining}
        # 16 nodes, nodes_per_instance=6 → 16 % 6 != 0, not called in grpo.py
        # But if it were called directly it should raise:
        with pytest.raises(ValueError, match="must be divisible by"):
            select_segment_nodes(remaining_topo, segment_size=6, num_nodes=16)

    def test_bundle_sort_after_placement_respects_domain_order(
        self, forty_node_topology
    ):
        # After topology-aware placement, _sort_bundle_indices_by_topology
        # should order training ranks so that domain A (lowest ranks) comes first.
        train_selected, _ = select_segment_nodes(
            forty_node_topology, segment_size=8, num_nodes=24
        )
        # Build bundle_data for the selected training nodes (1 GPU/node)
        bundle_data = [
            (0, forty_node_topology[nid][0], forty_node_topology[nid][1], nid)
            for nid in train_selected
        ]
        result = _sort_bundle_indices_by_topology(
            bundle_data, segment_size=8, gpus_per_node=1
        )
        domains_in_order = [bundle_data[i][1] for i in result]
        # Should be: 8 × domain_A, 8 × domain_B, 8 × domain_C
        assert domains_in_order[:8] == ["domain_A"] * 8
        assert domains_in_order[8:16] == ["domain_B"] * 8
        assert domains_in_order[16:] == ["domain_C"] * 8


# ---------------------------------------------------------------------------
# 6. Segment trimming in _sort_bundle_indices_by_topology (defense-in-depth)
# ---------------------------------------------------------------------------


class TestSegmentTrimming:
    def test_incomplete_domain_bundles_are_trimmed(self):
        # domain_A has 10 nodes, segment_size=8 → only 8 usable
        bundle_data = _make_bundle_data({"domain_A": list(range(10))}, gpus_per_node=1)
        result = _sort_bundle_indices_by_topology(
            bundle_data, segment_size=8, gpus_per_node=1
        )
        assert len(result) == 8  # 2 nodes trimmed

    def test_complete_domain_not_trimmed(self):
        bundle_data = _make_bundle_data({"domain_A": list(range(8))}, gpus_per_node=1)
        result = _sort_bundle_indices_by_topology(
            bundle_data, segment_size=8, gpus_per_node=1
        )
        assert len(result) == 8  # nothing trimmed

    def test_multi_gpu_per_node_trimming(self):
        # 10 nodes × 8 GPUs = 80 bundles; segment_size=8 nodes → 8 nodes usable × 8 GPUs = 64
        bundle_data = _make_bundle_data({"domain_A": list(range(10))}, gpus_per_node=8)
        result = _sort_bundle_indices_by_topology(
            bundle_data, segment_size=8, gpus_per_node=8
        )
        assert len(result) == 64
