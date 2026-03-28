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

from unittest.mock import MagicMock

import pytest

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.generation.vllm.routing_policy import (
    CacheAwareRoutingPolicy,
    ExternalRouter,
    RoundRobinRoutingPolicy,
    RoutingDecision,
    RoutingPolicyConfig,
    create_routing_policy,
)

DUMMY_DATA = BatchedDataDict({"input_ids": [[1, 2, 3]]})
DP_LEADER_INDICES = [0, 2, 4]  # 3 DP shards with leader workers at indices 0, 2, 4


class StubExternalRouter(ExternalRouter):
    """Concrete ExternalRouter for testing the factory and subclass check."""

    def __init__(self, dp_leader_worker_indices, **kwargs):
        self.dp_leader_worker_indices = dp_leader_worker_indices
        self.kwargs = kwargs

    def route(self, data):
        return RoutingDecision(dp_shard_idx=0, worker_idx=0)

    def prefill_complete(self, request_id):
        pass

    def generation_complete(self, request_id):
        pass

    def weights_updated(self):
        pass


class TestRoundRobinRoutingPolicy:
    def test_round_robin_distribution(self):
        policy = RoundRobinRoutingPolicy(DP_LEADER_INDICES)
        results = [
            policy.select_worker(data=DUMMY_DATA, request_id=f"req_{i}")
            for i in range(6)
        ]
        assert [r.dp_shard_idx for r in results] == [0, 1, 2, 0, 1, 2]

    def test_wraps_around(self):
        policy = RoundRobinRoutingPolicy(DP_LEADER_INDICES)
        for i in range(5):
            policy.select_worker(data=DUMMY_DATA, request_id=f"req_{i}")
        # After 5 calls (0,1,2,0,1), next should be 2
        result = policy.select_worker(data=DUMMY_DATA, request_id="req_5")
        assert result.dp_shard_idx == 2

    def test_returns_correct_worker_idx(self):
        policy = RoundRobinRoutingPolicy(DP_LEADER_INDICES)
        results = [
            policy.select_worker(data=DUMMY_DATA, request_id=f"req_{i}")
            for i in range(3)
        ]
        assert [r.worker_idx for r in results] == [0, 2, 4]

    def test_lifecycle_callbacks_are_noop(self):
        policy = RoundRobinRoutingPolicy(DP_LEADER_INDICES)
        # These should not raise
        policy.on_prefill_complete("req_1")
        policy.on_generation_complete("req_1")
        policy.on_weights_updated()


class TestCacheAwareRoutingPolicy:
    def _make_policy(self, route_return=None):
        mock_router = MagicMock()
        if route_return is not None:
            mock_router.route.return_value = route_return
        else:
            mock_router.route.return_value = RoutingDecision(dp_shard_idx=0, worker_idx=0)
        return CacheAwareRoutingPolicy(mock_router, DP_LEADER_INDICES), mock_router

    def test_delegates_to_external_router(self):
        decision = RoutingDecision(dp_shard_idx=1, worker_idx=2)
        policy, mock_router = self._make_policy(route_return=decision)

        result = policy.select_worker(data=DUMMY_DATA, request_id="req_1")
        assert result.dp_shard_idx == 1
        assert result.worker_idx == 2
        mock_router.route.assert_called_once_with(DUMMY_DATA)

    def test_validates_dp_shard_idx_bounds(self):
        decision = RoutingDecision(dp_shard_idx=5, worker_idx=0)
        policy, _ = self._make_policy(route_return=decision)

        with pytest.raises(ValueError, match="dp_shard_idx=5"):
            policy.select_worker(data=DUMMY_DATA, request_id="req_1")

    def test_validates_worker_idx_bounds(self):
        decision = RoutingDecision(dp_shard_idx=0, worker_idx=100)
        policy, _ = self._make_policy(route_return=decision)

        with pytest.raises(ValueError, match="worker_idx=100"):
            policy.select_worker(data=DUMMY_DATA, request_id="req_1")

    def test_on_prefill_complete(self):
        policy, mock_router = self._make_policy()
        policy.on_prefill_complete("req_42")
        mock_router.prefill_complete.assert_called_once_with("req_42")

    def test_on_generation_complete(self):
        policy, mock_router = self._make_policy()
        policy.on_generation_complete("req_42")
        mock_router.generation_complete.assert_called_once_with("req_42")

    def test_on_weights_updated(self):
        policy, mock_router = self._make_policy()
        policy.on_weights_updated()
        mock_router.weights_updated.assert_called_once()


class TestCreateRoutingPolicy:
    def _create(self, config):
        return create_routing_policy(config, dp_leader_worker_indices=DP_LEADER_INDICES)

    def test_none_config_returns_round_robin(self):
        policy = self._create(None)
        assert isinstance(policy, RoundRobinRoutingPolicy)

    def test_round_robin_config(self):
        policy = self._create({"type": "round_robin"})
        assert isinstance(policy, RoundRobinRoutingPolicy)

    def test_empty_config_returns_round_robin(self):
        policy = self._create({})
        assert isinstance(policy, RoundRobinRoutingPolicy)

    def test_cache_aware_config(self):
        config: RoutingPolicyConfig = {
            "type": "cache_aware",
            "router_class": "tests.unit.models.generation.test_routing_policy.StubExternalRouter",
            "router_kwargs": {"name": "test_router"},
        }
        policy = self._create(config)
        assert isinstance(policy, CacheAwareRoutingPolicy)

    def test_cache_aware_invalid_class_raises(self):
        config: RoutingPolicyConfig = {
            "type": "cache_aware",
            "router_class": "nonexistent.module.ClassName",
        }
        with pytest.raises(ModuleNotFoundError):
            self._create(config)

    def test_cache_aware_non_subclass_raises(self):
        config: RoutingPolicyConfig = {
            "type": "cache_aware",
            "router_class": "unittest.mock.MagicMock",
        }
        with pytest.raises(TypeError, match="must be a subclass of"):
            self._create(config)

    def test_cache_aware_missing_router_class_raises(self):
        config: RoutingPolicyConfig = {"type": "cache_aware"}
        with pytest.raises(ValueError, match="requires 'router_class'"):
            self._create(config)

    def test_unknown_type_raises(self):
        config: RoutingPolicyConfig = {"type": "unknown_policy"}
        with pytest.raises(ValueError, match="Unknown routing policy type"):
            self._create(config)


class TestRoutingPolicyConfig:
    def test_round_robin_config_construction(self):
        config: RoutingPolicyConfig = {"type": "round_robin"}
        assert config["type"] == "round_robin"

    def test_cache_aware_config_construction(self):
        config: RoutingPolicyConfig = {
            "type": "cache_aware",
            "router_class": "my_module.MyRouter",
            "router_kwargs": {"key": "value"},
        }
        assert config["type"] == "cache_aware"
        assert config["router_class"] == "my_module.MyRouter"
        assert config["router_kwargs"] == {"key": "value"}

    def test_minimal_config(self):
        config: RoutingPolicyConfig = {}
        assert "type" not in config
