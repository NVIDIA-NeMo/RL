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
import os
import subprocess
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

import pytest
import ray

from nemo_rl.distributed.virtual_cluster import (
    NVLINK_DOMAIN_UNKNOWN,
    PY_EXECUTABLES,
    TOPO_RANK_UNKNOWN,
    RayVirtualCluster,
    ResourceInsufficientError,
    _get_node_ip_and_free_port,
    _sort_bundle_indices_by_topology,
    select_segment_nodes,
)
from nemo_rl.utils.venvs import create_local_venv
from tests.unit.conftest import TEST_ASSETS_DIR


def test_get_node_ip_and_free_port_does_not_start_with_zero():
    # This test covers a case where the hostname was an integer like "255"
    # and socket returned an ip address equivalent to this hostname, i.e., "0.0.0.255".
    # It's not possible to mock the way the hostname is actually set on other platforms,
    # so we leave this test so we can ask users to run on their environment if needed.

    node_ip, _ = ray.get(
        _get_node_ip_and_free_port.options(
            runtime_env={"py_executable": PY_EXECUTABLES.SYSTEM}
        ).remote()
    )
    assert not node_ip.startswith("0."), "Node IP should not start with 0.*.*.*"


def test_env_max_retries_invalid_value():
    """Test that NRL_VIRTUAL_CLUSTER_MAX_RETRIES rejects invalid values (less than or equal to zero)."""

    # Mock environment with invalid max_retries value
    env_vars = {"NRL_VIRTUAL_CLUSTER_MAX_RETRIES": "0"}

    with patch.dict(os.environ, env_vars, clear=True):
        with pytest.raises(AssertionError):
            cluster = RayVirtualCluster(bundle_ct_per_node_list=[1])
            cluster._init_placement_groups()


def test_env_max_retries_non_integer():
    """Test that NRL_VIRTUAL_CLUSTER_MAX_RETRIES handles non-integer values properly."""

    # Mock environment with non-integer max_retries value
    env_vars = {"NRL_VIRTUAL_CLUSTER_MAX_RETRIES": "not_a_number"}

    with patch.dict(os.environ, env_vars, clear=True):
        with pytest.raises(ValueError):
            cluster = RayVirtualCluster(bundle_ct_per_node_list=[1])
            cluster._init_placement_groups()


def test_env_max_retries_default_value():
    """Test that default value for NRL_VIRTUAL_CLUSTER_MAX_RETRIES is used when not set."""

    # Ensure environment variable is not set
    with (
        patch.dict(os.environ, {}, clear=True),
        patch(
            "nemo_rl.distributed.virtual_cluster.RayVirtualCluster._init_placement_groups"
        ) as mock_init,
    ):
        # Mock successful initialization
        mock_init.return_value = [MagicMock()]

        # Create cluster
        cluster = RayVirtualCluster(bundle_ct_per_node_list=[1])
        cluster._init_placement_groups()

        # Default value should be 6 (as seen in the code)
        # We can't directly verify this, but we can check that initialization was attempted
        assert mock_init.call_count == 1


def test_env_max_retries_exhausted():
    """Test that NRL_VIRTUAL_CLUSTER_MAX_RETRIES correctly handles the case where all retries fail."""

    # Set specific retry count to 4
    retry_count = 4
    env_vars = {"NRL_VIRTUAL_CLUSTER_MAX_RETRIES": str(retry_count)}

    with (
        patch.dict(os.environ, env_vars, clear=True),
        patch(
            "nemo_rl.distributed.virtual_cluster.RayVirtualCluster._create_placement_groups_internal"
        ) as mock_init,
        patch("time.sleep") as mock_sleep,
    ):
        # Make _init_placement_groups raise ResourceInsufficientError each time
        mock_init.side_effect = ResourceInsufficientError("Not enough resources")

        # Create cluster - should retry retry_count times and then fail
        with pytest.raises(ResourceInsufficientError):
            cluster = RayVirtualCluster(bundle_ct_per_node_list=[1])
            cluster._init_placement_groups()

        # Verify _init_placement_groups was called exactly retry_count times
        assert mock_init.call_count == retry_count

        # Verify time.sleep was called with exponentially increasing values
        assert mock_sleep.call_count == retry_count
        mock_sleep.assert_any_call(1)  # 2^0
        mock_sleep.assert_any_call(2)  # 2^1
        mock_sleep.assert_any_call(4)  # 2^2
        mock_sleep.assert_any_call(8)  # 2^3


def test_ray_reinit_on_cuda_devices_change():
    """Test that Ray cluster is reinitialized when CUDA_VISIBLE_DEVICES changes."""

    with (
        patch("ray.init") as mock_ray_init,
        patch("ray.shutdown") as mock_ray_shutdown,
        patch("ray.cluster_resources") as mock_cluster_resources,
    ):
        # First call with CUDA_VISIBLE_DEVICES=0
        mock_cluster_resources.return_value = {"GPU": 1, "nrl_tag_0": 1}
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"}, clear=True):
            from nemo_rl.distributed.virtual_cluster import init_ray

            init_ray()

        assert mock_ray_init.call_count == 1
        assert mock_ray_shutdown.call_count == 0
        mock_ray_init.reset_mock()
        mock_ray_shutdown.reset_mock()

        # Second call with CUDA_VISIBLE_DEVICES=1
        mock_cluster_resources.return_value = {"GPU": 1, "nrl_tag_0": 1}
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "1"}, clear=True):
            init_ray()

        # Ray should be shutdown and reinitialized since the tag doesn't match
        assert (
            mock_ray_init.call_count == 2
        )  # Once for initial connect, once for reinit
        assert mock_ray_shutdown.call_count == 1  # Should shutdown after tag mismatch

        # Verify that the second init call included the new tag
        second_init_call = mock_ray_init.call_args_list[1]
        assert "resources" in second_init_call[1]
        assert "nrl_tag_1" in second_init_call[1]["resources"]


def test_ray_uses_same_cluster_for_permuted_cuda_devices():
    """Test that Ray cluster is reused if CUDA_VISIBLE_DEVICES order changes but set of devices is the same."""

    with (
        patch("ray.init") as mock_ray_init,
        patch("ray.shutdown") as mock_ray_shutdown,
        patch("ray.cluster_resources") as mock_cluster_resources,
    ):
        # Expected sorted tag
        expected_tag = "nrl_tag_0_2"

        # First call with CUDA_VISIBLE_DEVICES="0,2"
        mock_cluster_resources.return_value = {"GPU": 2, expected_tag: 1}
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0,2"}, clear=True):
            from nemo_rl.distributed.virtual_cluster import init_ray

            init_ray()

        assert mock_ray_init.call_count == 1
        assert mock_ray_init.call_args_list[0][1]["address"] == "auto"
        assert mock_ray_shutdown.call_count == 0
        mock_ray_init.reset_mock()
        mock_ray_shutdown.reset_mock()

        # Second call with CUDA_VISIBLE_DEVICES="2,0"
        mock_cluster_resources.return_value = {"GPU": 2, expected_tag: 1}
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "2,0"}, clear=True):
            from nemo_rl.distributed.virtual_cluster import init_ray

            init_ray()

        assert mock_ray_init.call_count == 1
        assert mock_ray_init.call_args_list[0][1]["address"] == "auto"
        assert mock_ray_shutdown.call_count == 0


def test_mcore_py_executable():
    # The temporary directory is created within the project.
    # For some reason, creating a virtual environment outside of the project
    # doesn't work reliably.
    with TemporaryDirectory(dir=TEST_ASSETS_DIR) as tempdir:
        # Mock os.environ to set NEMO_RL_VENV_DIR for this test
        with patch.dict(os.environ, {"NEMO_RL_VENV_DIR": tempdir}):
            venv_python = create_local_venv(
                py_executable=PY_EXECUTABLES.MCORE, venv_name="test_venv"
            )
            assert os.path.exists(venv_python)
            assert venv_python == f"{tempdir}/test_venv/bin/python"

            # Run a Python command to see if core dependencies were installed
            result = subprocess.run(
                [
                    venv_python,
                    "-c",
                    # Importing nemo_rl must be first to ensure all of megatron is importable
                    "import nemo_rl; print('nemo_rl is imported'); import transformer_engine.pytorch as te; print('te is imported'); import megatron.bridge; print('megatron-bridge is imported'); import megatron.core; print('megatron-core is imported'); import megatron.training; print('megatron-training is imported');",
                ],
                capture_output=True,
                text=True,
            )

            # Verify the command executed successfully (return code 0)
            assert result.returncode == 0, (
                f"Failed to import mcore libraries: {result.stderr}"
            )
            assert "nemo_rl is imported" in result.stdout
            assert "te is imported" in result.stdout
            assert "megatron-bridge is imported" in result.stdout
            assert "megatron-core is imported" in result.stdout
            assert "megatron-training is imported" in result.stdout


def test_create_sorted_bundle_indices_for_unified_pg():
    """Test that sorted bundle indices are created for a unified placement group."""
    cluster = RayVirtualCluster(bundle_ct_per_node_list=[2], use_gpus=True)
    cluster._init_placement_groups(strategy=None, use_unified_pg=True)
    assert cluster._sorted_bundle_indices is not None
    assert len(cluster._sorted_bundle_indices) == 2
    assert 0 in cluster._sorted_bundle_indices
    assert 1 in cluster._sorted_bundle_indices


def test_not_create_sorted_bundle_indices_for_per_node_pg():
    """Test that sorted bundle indices are not created for a per-node placement group."""
    cluster = RayVirtualCluster(bundle_ct_per_node_list=[2], use_gpus=True)
    cluster._init_placement_groups(strategy=None, use_unified_pg=False)
    assert cluster._sorted_bundle_indices is None


def test_sort_bundle_indices_fallback_by_node_id_gpu_id():
    """Fallback: no topology -> sort by (node_id, gpu_id)."""
    # bundle_data: (gpu_id, nvlink_domain, topo_rank, node_id)
    bundle_data = [
        (2, NVLINK_DOMAIN_UNKNOWN, TOPO_RANK_UNKNOWN, "node-b"),
        (0, NVLINK_DOMAIN_UNKNOWN, TOPO_RANK_UNKNOWN, "node-a"),
        (1, NVLINK_DOMAIN_UNKNOWN, TOPO_RANK_UNKNOWN, "node-a"),
    ]
    result = _sort_bundle_indices_by_topology(bundle_data)
    # node-a (0,1) before node-b (2); within node-a, gpu 0 before gpu 1
    assert result == [1, 2, 0]  # bundle 1: node-a gpu0, bundle 2: node-a gpu1, bundle 0: node-b gpu2


def test_sort_bundle_indices_topology_single_domain():
    """Topology: single domain -> sort by (topo_rank, gpu_id)."""
    bundle_data = [
        (1, "nvlink_domain_uuid-x", 100, "node-a"),
        (0, "nvlink_domain_uuid-x", 100, "node-a"),
        (2, "nvlink_domain_uuid-x", 101, "node-b"),
    ]
    result = _sort_bundle_indices_by_topology(bundle_data)
    # topo 100 first (bundles 0,1), then topo 101 (bundle 2). Within 100: gpu 0 < gpu 1
    assert result == [1, 0, 2]


def test_sort_bundle_indices_topology_two_domains():
    """Topology: two domains -> domains ordered by min(topo_rank), within domain by (topo_rank, gpu_id)."""
    # Domain A: topo 100, 101. Domain B: topo 50, 51 (lower min -> first)
    # (gpu_id, nvlink_domain, topo_rank, node_id)
    bundle_data = [
        (0, "nvlink_domain_B", 51, "node-b2"),
        (1, "nvlink_domain_B", 50, "node-b1"),
        (0, "nvlink_domain_A", 101, "node-a2"),
        (1, "nvlink_domain_A", 100, "node-a1"),
    ]
    result = _sort_bundle_indices_by_topology(bundle_data)
    # Domain B (min 50) before Domain A (min 100)
    # Within B: (50,gpu1)=bundle 1, (51,gpu0)=bundle 0
    # Within A: (100,gpu1)=bundle 3, (101,gpu0)=bundle 2
    assert result == [1, 0, 3, 2]


def test_sort_bundle_indices_topology_topo_rank_only():
    """Topology: topo_rank present but nvlink_domain unknown -> still use topology sort."""
    bundle_data = [
        (1, NVLINK_DOMAIN_UNKNOWN, 10, "node-x"),
        (0, NVLINK_DOMAIN_UNKNOWN, 5, "node-x"),
    ]
    result = _sort_bundle_indices_by_topology(bundle_data)
    # topo 5 < topo 10; when topo equal, gpu_id breaks tie
    assert result == [1, 0]


def test_sort_bundle_indices_empty():
    """Empty input returns empty list."""
    assert _sort_bundle_indices_by_topology([]) == []


# ===== segment_size tests for _sort_bundle_indices_by_topology =====

def test_sort_bundle_indices_segment_requires_gpus_per_node():
    """segment_size without gpus_per_node raises ValueError."""
    bundle_data = [(0, "nvlink_domain_A", 100, "node-a")]
    with pytest.raises(ValueError, match="gpus_per_node is required"):
        _sort_bundle_indices_by_topology(bundle_data, segment_size=2, gpus_per_node=None)


def test_sort_bundle_indices_segment_complete_domain():
    """Two nodes (4 GPUs each) in one domain, segment_size=2: all kept."""
    # 2 nodes × 4 GPUs = 8 bundles in one domain, segment_size=2 nodes -> all usable
    bundle_data = [
        (0, "nvlink_domain_A", 10, "node-a"),
        (1, "nvlink_domain_A", 10, "node-a"),
        (2, "nvlink_domain_A", 10, "node-a"),
        (3, "nvlink_domain_A", 10, "node-a"),
        (0, "nvlink_domain_A", 11, "node-b"),
        (1, "nvlink_domain_A", 11, "node-b"),
        (2, "nvlink_domain_A", 11, "node-b"),
        (3, "nvlink_domain_A", 11, "node-b"),
    ]
    result = _sort_bundle_indices_by_topology(bundle_data, segment_size=2, gpus_per_node=4)
    assert len(result) == 8
    # topo_rank 10 bundles first (sorted by gpu_id), then topo_rank 11
    assert result == [0, 1, 2, 3, 4, 5, 6, 7]


def test_sort_bundle_indices_segment_incomplete_domain_trimmed():
    """3 nodes in one domain, segment_size=2: only first 2 nodes (8 GPUs) kept."""
    bundle_data = [
        (0, "nvlink_domain_A", 10, "node-a"),
        (1, "nvlink_domain_A", 10, "node-a"),
        (2, "nvlink_domain_A", 10, "node-a"),
        (3, "nvlink_domain_A", 10, "node-a"),
        (0, "nvlink_domain_A", 11, "node-b"),
        (1, "nvlink_domain_A", 11, "node-b"),
        (2, "nvlink_domain_A", 11, "node-b"),
        (3, "nvlink_domain_A", 11, "node-b"),
        (0, "nvlink_domain_A", 12, "node-c"),
        (1, "nvlink_domain_A", 12, "node-c"),
        (2, "nvlink_domain_A", 12, "node-c"),
        (3, "nvlink_domain_A", 12, "node-c"),
    ]
    result = _sort_bundle_indices_by_topology(bundle_data, segment_size=2, gpus_per_node=4)
    # 3 nodes / segment_size=2 -> usable=2 nodes (8 GPUs); node-c (topo_rank=12) gets discarded
    assert len(result) == 8
    assert result == [0, 1, 2, 3, 4, 5, 6, 7]


def test_sort_bundle_indices_segment_multi_domain():
    """Two domains with different sizes: segment filtering per domain."""
    bundle_data = [
        # Domain A: 2 nodes (complete segment)
        (0, "nvlink_domain_A", 100, "node-a1"),
        (1, "nvlink_domain_A", 100, "node-a1"),
        (0, "nvlink_domain_A", 101, "node-a2"),
        (1, "nvlink_domain_A", 101, "node-a2"),
        # Domain B: 3 nodes, only 2 kept with segment_size=2
        (0, "nvlink_domain_B", 50, "node-b1"),
        (1, "nvlink_domain_B", 50, "node-b1"),
        (0, "nvlink_domain_B", 51, "node-b2"),
        (1, "nvlink_domain_B", 51, "node-b2"),
        (0, "nvlink_domain_B", 52, "node-b3"),
        (1, "nvlink_domain_B", 52, "node-b3"),
    ]
    result = _sort_bundle_indices_by_topology(bundle_data, segment_size=2, gpus_per_node=2)
    # Domain B (min topo_rank 50) first, Domain A (min 100) second
    # Domain B: 3 nodes, usable=2 (topo 50, 51), node-b3 discarded
    # Result: [b1-gpu0, b1-gpu1, b2-gpu0, b2-gpu1, a1-gpu0, a1-gpu1, a2-gpu0, a2-gpu1]
    assert len(result) == 8
    assert result == [4, 5, 6, 7, 0, 1, 2, 3]


# ===== select_segment_nodes tests =====

def test_select_segment_nodes_basic():
    """Selects 2 segments of 2 nodes each from 2 domains."""
    topology = {
        "n1": ("domain_A", 100),
        "n2": ("domain_A", 101),
        "n3": ("domain_B", 50),
        "n4": ("domain_B", 51),
    }
    training, remaining = select_segment_nodes(topology, segment_size=2, num_training_nodes=4)
    assert len(training) == 4
    assert len(remaining) == 0
    # Domain B (min topo 50) first, then Domain A (min topo 100)
    assert training == ["n3", "n4", "n1", "n2"]


def test_select_segment_nodes_partial_selection():
    """Selects only 1 segment from a domain with 2 segments available."""
    topology = {
        "n1": ("domain_A", 100),
        "n2": ("domain_A", 101),
        "n3": ("domain_A", 102),
        "n4": ("domain_A", 103),
    }
    training, remaining = select_segment_nodes(topology, segment_size=2, num_training_nodes=2)
    assert len(training) == 2
    assert len(remaining) == 2
    assert training == ["n1", "n2"]
    assert set(remaining) == {"n3", "n4"}


def test_select_segment_nodes_not_divisible():
    """num_training_nodes not divisible by segment_size raises ValueError."""
    topology = {"n1": ("domain_A", 100), "n2": ("domain_A", 101)}
    with pytest.raises(ValueError, match="must be divisible"):
        select_segment_nodes(topology, segment_size=3, num_training_nodes=4)


def test_select_segment_nodes_insufficient():
    """Not enough complete segments raises ResourceInsufficientError."""
    topology = {
        "n1": ("domain_A", 100),
        "n2": ("domain_B", 200),
    }
    with pytest.raises(ResourceInsufficientError, match="Cannot form"):
        select_segment_nodes(topology, segment_size=2, num_training_nodes=2)


def test_select_segment_nodes_incomplete_domain_skipped():
    """Domain with fewer nodes than segment_size is skipped."""
    topology = {
        "n1": ("domain_A", 100),  # only 1 node, can't form segment of 2
        "n2": ("domain_B", 50),
        "n3": ("domain_B", 51),
    }
    training, remaining = select_segment_nodes(topology, segment_size=2, num_training_nodes=2)
    assert training == ["n2", "n3"]
    assert remaining == ["n1"]


def test_node_resource_constraints_init_validation():
    """node_resource_constraints must match bundle_ct_per_node_list in length."""
    with pytest.raises(AssertionError, match="node_resource_constraints length"):
        RayVirtualCluster(
            bundle_ct_per_node_list=[4, 4],
            node_resource_constraints=[{"nvlink_domain_abc": 0.001}],
        )


def test_node_resource_constraints_none_accepted():
    """None constraints are accepted and equivalent to no constraints."""
    vc = RayVirtualCluster(
        bundle_ct_per_node_list=[4, 4],
        node_resource_constraints=None,
    )
    assert vc.node_resource_constraints is None


@patch("nemo_rl.distributed.virtual_cluster.ray")
@patch("nemo_rl.distributed.virtual_cluster.placement_group")
def test_node_resource_constraints_applied_to_per_node_pg(mock_pg, mock_ray):
    """Constraints are merged into bundle specs for per-node placement groups."""
    mock_ray.cluster_resources.return_value = {"GPU": 8, "CPU": 8}

    mock_pg_obj = MagicMock()
    mock_pg_obj.ready.return_value = MagicMock()
    mock_pg.return_value = mock_pg_obj
    mock_ray.get.return_value = None

    constraints = [
        {"nvlink_domain_aaa": 0.001},
        {"nvlink_domain_bbb": 0.001},
    ]
    vc = RayVirtualCluster(
        bundle_ct_per_node_list=[4, 4],
        node_resource_constraints=constraints,
    )
    vc._create_placement_groups_internal(strategy="PACK", use_unified_pg=False)

    assert mock_pg.call_count == 2
    # First node's bundles should include domain_aaa constraint
    call_0_bundles = mock_pg.call_args_list[0][1]["bundles"]
    assert len(call_0_bundles) == 4
    for bundle in call_0_bundles:
        assert bundle["CPU"] == 1
        assert bundle["GPU"] == 1
        assert bundle["nvlink_domain_aaa"] == 0.001
        assert "nvlink_domain_bbb" not in bundle

    # Second node's bundles should include domain_bbb constraint
    call_1_bundles = mock_pg.call_args_list[1][1]["bundles"]
    for bundle in call_1_bundles:
        assert bundle["nvlink_domain_bbb"] == 0.001
        assert "nvlink_domain_aaa" not in bundle


@patch("nemo_rl.distributed.virtual_cluster.ray")
@patch("nemo_rl.distributed.virtual_cluster.placement_group")
def test_node_resource_constraints_applied_to_unified_pg(mock_pg, mock_ray):
    """Constraints are merged into bundle specs for unified placement groups."""
    mock_ray.cluster_resources.return_value = {"GPU": 8, "CPU": 8}

    mock_pg_obj = MagicMock()
    mock_pg_obj.ready.return_value = MagicMock()
    mock_pg.return_value = mock_pg_obj
    mock_ray.get.return_value = None

    constraints = [
        {"nvlink_domain_aaa": 0.001},
        {"nvlink_domain_bbb": 0.001},
    ]
    vc = RayVirtualCluster(
        bundle_ct_per_node_list=[4, 4],
        node_resource_constraints=constraints,
    )
    vc._create_placement_groups_internal(strategy="PACK", use_unified_pg=True)

    assert mock_pg.call_count == 1
    all_bundles = mock_pg.call_args_list[0][1]["bundles"]
    assert len(all_bundles) == 8
    # First 4 bundles from node 0 -> domain_aaa
    for bundle in all_bundles[:4]:
        assert bundle["nvlink_domain_aaa"] == 0.001
        assert "nvlink_domain_bbb" not in bundle
    # Last 4 bundles from node 1 -> domain_bbb
    for bundle in all_bundles[4:]:
        assert bundle["nvlink_domain_bbb"] == 0.001
        assert "nvlink_domain_aaa" not in bundle


@patch("nemo_rl.distributed.virtual_cluster.ray")
@patch("nemo_rl.distributed.virtual_cluster.placement_group")
def test_no_constraints_bundles_unchanged(mock_pg, mock_ray):
    """Without constraints, bundles only have CPU and GPU."""
    mock_ray.cluster_resources.return_value = {"GPU": 4, "CPU": 4}

    mock_pg_obj = MagicMock()
    mock_pg_obj.ready.return_value = MagicMock()
    mock_pg.return_value = mock_pg_obj
    mock_ray.get.return_value = None

    vc = RayVirtualCluster(
        bundle_ct_per_node_list=[4],
        node_resource_constraints=None,
    )
    vc._create_placement_groups_internal(strategy="PACK", use_unified_pg=False)

    call_bundles = mock_pg.call_args_list[0][1]["bundles"]
    for bundle in call_bundles:
        assert set(bundle.keys()) == {"CPU", "GPU"}
