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

import pytest
import torch

from nemo_rl.algorithms.grpo import _preserve_dsa_topk_replay_indices, setup
from nemo_rl.distributed.batched_data_dict import BatchedDataDict


def test_dsa_topk_replay_indices_are_preserved_for_train_and_logprob_inputs():
    indices = torch.arange(2 * 4 * 3 * 2, dtype=torch.int16).reshape(2, 4, 3, 2)
    flat_messages = BatchedDataDict({"dsa_topk_indices": indices})
    policy_config = {"dsa_topk_replay": {"enabled": True, "layer_ids": [0, 2, 6]}}
    train_data = BatchedDataDict()
    logprob_data = BatchedDataDict()

    _preserve_dsa_topk_replay_indices(train_data, flat_messages, policy_config)
    _preserve_dsa_topk_replay_indices(logprob_data, flat_messages, policy_config)

    assert train_data["dsa_topk_indices"] is indices
    assert logprob_data["dsa_topk_indices"] is indices

    disabled_target = BatchedDataDict()
    _preserve_dsa_topk_replay_indices(
        disabled_target,
        flat_messages,
        {"dsa_topk_replay": {"enabled": False}},
    )
    assert "dsa_topk_indices" not in disabled_target


@pytest.mark.parametrize(
    ("async_enabled", "data_plane_enabled", "error_match"),
    [
        (True, False, "synchronous GRPO only"),
        (False, True, "legacy in-process data path only"),
    ],
)
def test_setup_rejects_dsa_topk_replay_on_unsupported_transport(
    async_enabled: bool,
    data_plane_enabled: bool,
    error_match: str,
):
    master_config = SimpleNamespace(
        policy={
            "generation": {
                "backend": "vllm",
                "vllm_cfg": {
                    "async_engine": False,
                    "pipeline_parallel_size": 1,
                    "enforce_eager": True,
                },
                "vllm_kwargs": {},
            },
            "megatron_cfg": {
                "enabled": True,
                "cuda_graph_impl": "none",
                "model_overrides": {"dsa_indexer_loss_coeff": 0.0},
            },
            "dsa_topk_replay": {"enabled": True, "layer_ids": None},
        },
        grpo=SimpleNamespace(
            async_grpo=SimpleNamespace(enabled=async_enabled),
        ),
        data_plane={"enabled": data_plane_enabled},
        loss_fn=object(),
        env={},
        data={},
        logger=object(),
        cluster=object(),
        checkpointing={},
    )

    with pytest.raises(ValueError, match=error_match):
        setup(
            master_config,
            tokenizer=None,
            dataset={},
            val_dataset=None,
        )


def test_setup_rejects_dsa_topk_replay_with_dtensor_policy_before_mutation(
    monkeypatch,
):
    configure_calls = []
    monkeypatch.setattr(
        "nemo_rl.algorithms.grpo.configure_vllm_for_dsa_topk_replay",
        lambda _config: configure_calls.append(True),
    )
    master_config = SimpleNamespace(
        policy={
            "generation": {
                "backend": "vllm",
                "vllm_cfg": {
                    "async_engine": False,
                    "pipeline_parallel_size": 1,
                    "enforce_eager": True,
                },
                "vllm_kwargs": {},
            },
            "megatron_cfg": {"enabled": False},
            "dtensor_cfg": {"enabled": True},
            "dsa_topk_replay": {"enabled": True, "layer_ids": [0]},
        },
        grpo=SimpleNamespace(async_grpo=SimpleNamespace(enabled=False)),
        data_plane={"enabled": False},
        loss_fn=object(),
        env={},
        data={},
        logger=object(),
        cluster=object(),
        checkpointing={},
    )

    with pytest.raises(ValueError, match="requires the Megatron policy backend"):
        setup(
            master_config,
            tokenizer=None,
            dataset={},
            val_dataset=None,
        )

    assert configure_calls == []
