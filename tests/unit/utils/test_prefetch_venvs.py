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
from unittest.mock import patch

import pytest

from nemo_rl.utils.prefetch_venvs import prefetch_venvs


@pytest.fixture
def mock_registry():
    """Create a mock registry with various actor types."""
    return {
        "nemo_rl.models.generation.vllm.vllm_worker.VllmGenerationWorker": "uv run --group vllm",
        "nemo_rl.models.policy.workers.dtensor_policy_worker.DTensorPolicyWorker": "uv run --group vllm",
        "nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker": "uv run --group mcore",
        "nemo_rl.environments.math_environment.MathEnvironment": "python",
        "nemo_rl.environments.code_environment.CodeEnvironment": "python",
    }


class TestPrefetchVenvs:
    """Tests for the prefetch_venvs function."""

    def test_prefetch_venvs_no_filters(self, mock_registry):
        """Test that all uv-based venvs are prefetched when no filters are provided."""
        with (
            patch(
                "nemo_rl.utils.prefetch_venvs.ACTOR_ENVIRONMENT_REGISTRY", mock_registry
            ),
            patch("nemo_rl.utils.prefetch_venvs.create_local_venv") as mock_create_venv,
        ):
            mock_create_venv.return_value = "/path/to/venv/bin/python"

            prefetch_venvs(filters=None)

            assert mock_create_venv.call_count > 0

            # Verify the actors that were called
            call_args = [call[0] for call in mock_create_venv.call_args_list]
            actor_fqns = [args[1] for args in call_args]

            assert (
                "nemo_rl.models.generation.vllm.vllm_worker.VllmGenerationWorker"
                in actor_fqns
            )
            assert (
                "nemo_rl.models.policy.workers.dtensor_policy_worker.DTensorPolicyWorker"
                in actor_fqns
            )
            assert (
                "nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker"
                in actor_fqns
            )

    def test_prefetch_venvs_single_filter(self, mock_registry):
        """Test filtering with a single filter string."""
        with (
            patch(
                "nemo_rl.utils.prefetch_venvs.ACTOR_ENVIRONMENT_REGISTRY", mock_registry
            ),
            patch("nemo_rl.utils.prefetch_venvs.create_local_venv") as mock_create_venv,
        ):
            mock_create_venv.return_value = "/path/to/venv/bin/python"

            prefetch_venvs(filters=["vllm"])

            # Should only create venvs for actors containing "vllm" (1 actor)
            assert mock_create_venv.call_count == 1

            call_args = mock_create_venv.call_args[0]
            assert (
                call_args[1]
                == "nemo_rl.models.generation.vllm.vllm_worker.VllmGenerationWorker"
            )

    def test_prefetch_venvs_multiple_filters(self, mock_registry):
        """Test filtering with multiple filter strings (OR logic)."""
        with (
            patch(
                "nemo_rl.utils.prefetch_venvs.ACTOR_ENVIRONMENT_REGISTRY", mock_registry
            ),
            patch("nemo_rl.utils.prefetch_venvs.create_local_venv") as mock_create_venv,
        ):
            mock_create_venv.return_value = "/path/to/venv/bin/python"

            prefetch_venvs(filters=["vllm", "megatron"])

            # Should create venvs for actors containing "vllm" OR "megatron" (2 actors)
            assert mock_create_venv.call_count == 2

            call_args = [call[0] for call in mock_create_venv.call_args_list]
            actor_fqns = [args[1] for args in call_args]

            assert (
                "nemo_rl.models.generation.vllm.vllm_worker.VllmGenerationWorker"
                in actor_fqns
            )
            assert (
                "nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker"
                in actor_fqns
            )

    def test_prefetch_venvs_filter_no_match(self, mock_registry):
        """Test that no venvs are created when filter matches nothing."""
        with (
            patch(
                "nemo_rl.utils.prefetch_venvs.ACTOR_ENVIRONMENT_REGISTRY", mock_registry
            ),
            patch("nemo_rl.utils.prefetch_venvs.create_local_venv") as mock_create_venv,
        ):
            mock_create_venv.return_value = "/path/to/venv/bin/python"

            prefetch_venvs(filters=["nonexistent"])

            # Should not create any venvs
            assert mock_create_venv.call_count == 0

    def test_prefetch_venvs_skips_system_python(self, mock_registry):
        """Test that system python actors are skipped even if they match filters."""
        with (
            patch(
                "nemo_rl.utils.prefetch_venvs.ACTOR_ENVIRONMENT_REGISTRY", mock_registry
            ),
            patch("nemo_rl.utils.prefetch_venvs.create_local_venv") as mock_create_venv,
        ):
            mock_create_venv.return_value = "/path/to/venv/bin/python"

            # Filter for "environment" which matches system python actors
            prefetch_venvs(filters=["environment"])

            # Should not create any venvs since matching actors use system python
            assert mock_create_venv.call_count == 0

    def test_prefetch_venvs_partial_match(self, mock_registry):
        """Test that filter matches partial strings within FQN."""
        with (
            patch(
                "nemo_rl.utils.prefetch_venvs.ACTOR_ENVIRONMENT_REGISTRY", mock_registry
            ),
            patch("nemo_rl.utils.prefetch_venvs.create_local_venv") as mock_create_venv,
        ):
            mock_create_venv.return_value = "/path/to/venv/bin/python"

            # "policy" should match both dtensor_policy_worker and megatron_policy_worker
            prefetch_venvs(filters=["policy"])

            assert mock_create_venv.call_count == 2

            call_args = [call[0] for call in mock_create_venv.call_args_list]
            actor_fqns = [args[1] for args in call_args]

            assert (
                "nemo_rl.models.policy.workers.dtensor_policy_worker.DTensorPolicyWorker"
                in actor_fqns
            )
            assert (
                "nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker"
                in actor_fqns
            )

    def test_prefetch_venvs_empty_filter_list(self, mock_registry):
        """Test that empty filter list is treated as no filtering (falsy)."""
        with (
            patch(
                "nemo_rl.utils.prefetch_venvs.ACTOR_ENVIRONMENT_REGISTRY", mock_registry
            ),
            patch("nemo_rl.utils.prefetch_venvs.create_local_venv") as mock_create_venv,
        ):
            mock_create_venv.return_value = "/path/to/venv/bin/python"

            # Empty list should be falsy and prefetch all
            prefetch_venvs(filters=[])

            # Should create venvs for all uv-based actors (3 total)
            assert mock_create_venv.call_count == 3

    def test_prefetch_venvs_continues_on_error(self, mock_registry):
        """Test that prefetching continues even if one venv creation fails."""
        with (
            patch(
                "nemo_rl.utils.prefetch_venvs.ACTOR_ENVIRONMENT_REGISTRY", mock_registry
            ),
            patch("nemo_rl.utils.prefetch_venvs.create_local_venv") as mock_create_venv,
        ):
            # First call raises, subsequent calls succeed
            mock_create_venv.side_effect = [
                Exception("Test error"),
                "/path/to/venv/bin/python",
                "/path/to/venv/bin/python",
            ]

            # Should not raise, should continue with other venvs
            prefetch_venvs(filters=None)

            # All 3 uv-based actors should have been attempted
            assert mock_create_venv.call_count == 3

    def test_prefetch_venvs_case_sensitive_filter(self, mock_registry):
        """Test that filters are case-sensitive."""
        with (
            patch(
                "nemo_rl.utils.prefetch_venvs.ACTOR_ENVIRONMENT_REGISTRY", mock_registry
            ),
            patch("nemo_rl.utils.prefetch_venvs.create_local_venv") as mock_create_venv,
        ):
            mock_create_venv.return_value = "/path/to/venv/bin/python"

            # "VLLM" (uppercase) should not match "vllm" (lowercase)
            prefetch_venvs(filters=["VLLM"])

            assert mock_create_venv.call_count == 0
