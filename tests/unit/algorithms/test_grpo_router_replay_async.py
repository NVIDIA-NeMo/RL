# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from nemo_rl.algorithms.grpo import (
    MasterConfig,
    _build_async_grpo_train_data,
    _default_grpo_save_state,
    async_grpo_train,
)
from nemo_rl.distributed.batched_data_dict import BatchedDataDict


@pytest.fixture(scope="session", autouse=True)
def init_ray_cluster():
    yield


@pytest.fixture(scope="session", autouse=True)
def ray_gpu_monitor():
    class _NoopGpuMonitor:
        def _collect_metrics(self):
            return {}

        def stop(self):
            pass

    yield _NoopGpuMonitor()


@pytest.fixture(scope="session", autouse=True)
def session_data(_unit_test_data):
    yield _unit_test_data


def _make_async_master_config(data_plane=None) -> MasterConfig:
    return MasterConfig.model_construct(
        **{
            "policy": {
                "router_replay": {"enabled": True},
                "generation": {
                    "backend": "vllm",
                    "vllm_cfg": {"async_engine": True},
                },
            },
            "loss_fn": SimpleNamespace(use_importance_sampling_correction=True),
            "data_plane": data_plane,
        }
    )


# Keep this focused on async no-TQ batch construction instead of full Ray orchestration.
@pytest.mark.parametrize(
    ("policy_config", "expect_routed_experts"),
    [
        ({"router_replay": {"enabled": True}}, True),
        ({"router_replay": {"enabled": False}}, False),
    ],
)
def test_build_async_grpo_train_data_preserves_routed_experts_for_r3(
    policy_config, expect_routed_experts
):
    routes = torch.arange(1 * 3 * 2 * 4, dtype=torch.int32).reshape(1, 3, 2, 4)
    flat_messages = BatchedDataDict(
        {
            "token_ids": torch.tensor([[1, 2, 3]]),
            "generation_logprobs": torch.zeros(1, 3),
            "token_loss_mask": torch.tensor([[0, 1, 1]]),
            "routed_experts": routes,
        }
    )
    input_lengths = torch.tensor([3])
    repeated_batch = BatchedDataDict({"loss_multiplier": torch.tensor([1.0])})

    train_data = _build_async_grpo_train_data(
        flat_messages,
        input_lengths,
        repeated_batch,
        policy_config,
    )

    assert torch.equal(train_data["input_ids"], flat_messages["token_ids"])
    assert torch.equal(train_data["input_lengths"], input_lengths)
    assert torch.equal(
        train_data["generation_logprobs"], flat_messages["generation_logprobs"]
    )
    assert torch.equal(train_data["token_mask"], flat_messages["token_loss_mask"])
    assert torch.equal(train_data["sample_mask"], repeated_batch["loss_multiplier"])

    if expect_routed_experts:
        assert torch.equal(train_data["routed_experts"], routes)
    else:
        assert "routed_experts" not in train_data


def test_async_grpo_r3_rejects_data_plane_until_async_tq_exists():
    master_config = _make_async_master_config(data_plane={"enabled": True})

    with pytest.raises(NotImplementedError, match="data_plane.enabled=false"):
        async_grpo_train(
            MagicMock(),
            MagicMock(),
            MagicMock(),
            None,
            MagicMock(),
            MagicMock(),
            {"math": MagicMock()},
            None,
            MagicMock(),
            MagicMock(),
            _default_grpo_save_state(),
            master_config,
        )
