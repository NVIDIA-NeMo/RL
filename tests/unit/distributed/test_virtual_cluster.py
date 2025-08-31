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
    PY_EXECUTABLES,
    RayVirtualCluster,
    ResourceInsufficientError,
    _get_node_ip_and_free_port,
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
                    "import nemo_rl; print('nemo_rl is imported'); import transformer_engine.pytorch as te; print('te is imported'); import nemo.tron; print('nemo-tron is imported'); import megatron.core; print('megatron-core is imported'); import megatron.training; print('megatron-training is imported');",
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
            assert "nemo-tron is imported" in result.stdout
            assert "megatron-core is imported" in result.stdout
            assert "megatron-training is imported" in result.stdout
