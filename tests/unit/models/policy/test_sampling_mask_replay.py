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

from __future__ import annotations

from copy import deepcopy
from typing import Any, Callable

import pytest
import torch

from nemo_rl.models.policy.sampling_mask_replay import (
    attach_sampling_mask_to_assistant_message,
    backfill_missing_sampling_masks,
    configure_vllm_for_sampling_mask_replay,
    sampling_mask_replay_enabled,
)


def _enabled_config() -> dict[str, Any]:
    return {
        "sampling_mask_replay": {"enabled": True},
        "generation": {
            "backend": "vllm",
            "temperature": 0.8,
            "top_p": 0.9,
            "top_k": 3,
            "vllm_kwargs": {},
            "vllm_cfg": {"expose_http_server": False},
        },
        "megatron_cfg": {"use_fused_linear_logprobs": False},
        "sequence_packing": {"enabled": False},
    }


def test_sampling_mask_replay_defaults_disabled_when_absent() -> None:
    assert sampling_mask_replay_enabled({}) is False


def test_configure_rejects_engine_flag_without_policy_feature() -> None:
    config = {
        "generation": {
            "vllm_kwargs": {"return_sampling_mask": True},
        }
    }

    with pytest.raises(ValueError, match="must not be set directly"):
        configure_vllm_for_sampling_mask_replay(config)


def test_configure_sampling_mask_replay_sets_required_vllm_options() -> None:
    config = _enabled_config()

    configure_vllm_for_sampling_mask_replay(config)

    generation = config["generation"]
    assert generation["vllm_kwargs"]["return_sampling_mask"] is True
    assert generation["vllm_cfg"]["logprobs_mode"] == "processed_logprobs"
    assert generation["vllm_cfg"]["env_vars"]["VLLM_USE_V2_MODEL_RUNNER"] == "1"


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (
            lambda c: c["generation"].update(backend="sglang"),
            "requires vLLM generation",
        ),
        (
            lambda c: c["generation"].update(temperature=0),
            "temperature > 0",
        ),
        (
            lambda c: c["generation"].update(top_k=-1),
            "finite positive integer generation.top_k",
        ),
        (
            lambda c: c["generation"]["vllm_kwargs"].update(
                speculative_config={"method": "mtp"}
            ),
            "does not support speculative decoding",
        ),
        (
            lambda c: c["generation"]["vllm_kwargs"].update(
                logits_processors=["custom"]
            ),
            "does not support custom logits processors",
        ),
        (
            lambda c: c["megatron_cfg"].update(use_fused_linear_logprobs=True),
            "does not support fused linear logprobs",
        ),
        (
            lambda c: c["sequence_packing"].update(enabled=True),
            "does not support policy.sequence_packing.enabled=true",
        ),
        (
            lambda c: c["generation"]["vllm_cfg"].update(expose_http_server=True),
            "does not support OpenAI-compatible HTTP chat rollouts",
        ),
        (
            lambda c: c["generation"]["vllm_cfg"].update(
                env_vars={"VLLM_USE_V2_MODEL_RUNNER": "0"}
            ),
            "requires vLLM Model Runner V2",
        ),
    ],
)
def test_configure_sampling_mask_replay_rejects_unsupported_configs(
    mutate: Callable[[dict[str, Any]], None],
    match: str,
) -> None:
    config = deepcopy(_enabled_config())
    mutate(config)

    with pytest.raises(ValueError, match=match):
        configure_vllm_for_sampling_mask_replay(config)


def test_configure_sampling_mask_replay_rejects_nemo_gym() -> None:
    with pytest.raises(ValueError, match="does not support NeMo-Gym"):
        configure_vllm_for_sampling_mask_replay(
            _enabled_config(),
            use_nemo_gym=True,
        )


def test_attach_and_backfill_sampling_mask_message_rows() -> None:
    assistant = {
        "role": "assistant",
        "content": "answer",
        "token_ids": torch.tensor([20, 21, 22]),
    }
    outputs = {
        "sampling_mask_token_ids": torch.tensor(
            [[[0, 0, 0], [0, 0, 0], [20, 7, 8], [21, 9, 0], [22, 0, 0]]],
            dtype=torch.int32,
        ),
        "sampling_mask_sizes": torch.tensor([[0, 0, 3, 2, 1]], dtype=torch.int32),
    }

    assert attach_sampling_mask_to_assistant_message(
        assistant,
        outputs,
        batch_index=0,
        input_length=2,
        total_length=5,
    )
    message_log = [
        {
            "role": "user",
            "content": "prompt",
            "token_ids": torch.tensor([10, 11]),
        },
        assistant,
        {
            "role": "user",
            "content": "environment",
            "token_ids": torch.tensor([30]),
        },
    ]

    backfill_missing_sampling_masks([message_log])

    assert torch.equal(
        assistant["sampling_mask_token_ids"],
        outputs["sampling_mask_token_ids"][0, 2:5],
    )
    assert message_log[0]["sampling_mask_token_ids"].shape == (2, 3)
    assert message_log[2]["sampling_mask_token_ids"].shape == (1, 3)
    assert message_log[0]["sampling_mask_sizes"].tolist() == [0, 0]
    assert message_log[2]["sampling_mask_sizes"].tolist() == [0]
    assert message_log[0]["sampling_mask_token_ids"].dtype == torch.int32


def test_backfill_uses_sibling_width_for_failed_completion() -> None:
    captured = [
        {
            "role": "assistant",
            "token_ids": torch.tensor([20]),
            "sampling_mask_token_ids": torch.tensor([[20, 7]], dtype=torch.int32),
            "sampling_mask_sizes": torch.tensor([2], dtype=torch.int32),
        }
    ]
    failed = [{"role": "user", "token_ids": torch.tensor([10, 11])}]

    backfill_missing_sampling_masks([captured, failed])

    assert failed[0]["sampling_mask_token_ids"].shape == (2, 2)
    assert torch.count_nonzero(failed[0]["sampling_mask_token_ids"]) == 0
    assert failed[0]["sampling_mask_sizes"].tolist() == [0, 0]


def test_attach_rejects_torn_generation_output_pair() -> None:
    with pytest.raises(RuntimeError, match="must be returned together"):
        attach_sampling_mask_to_assistant_message(
            {"token_ids": torch.tensor([20])},
            {"sampling_mask_token_ids": torch.tensor([[[20]]], dtype=torch.int32)},
            batch_index=0,
            input_length=0,
            total_length=1,
        )
