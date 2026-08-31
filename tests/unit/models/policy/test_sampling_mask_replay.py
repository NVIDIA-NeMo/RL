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

from typing import Any, cast

import pytest

from nemo_rl.models.policy import PolicyConfig
from nemo_rl.models.policy.sampling_mask_replay import (
    configure_vllm_for_sampling_mask_replay,
    sampling_mask_replay_enabled,
)


def _enabled_config() -> PolicyConfig:
    return cast(
        PolicyConfig,
        {
            "sampling_mask_replay": {"enabled": True},
            "generation": {
                "backend": "vllm",
                "temperature": 1.0,
                "top_k": 50,
                "vllm_kwargs": {},
                "vllm_cfg": {"env_vars": None},
            },
            "megatron_cfg": {"enabled": True},
            "sequence_packing": {"enabled": False},
        },
    )


def test_sampling_mask_replay_defaults_disabled() -> None:
    config = cast(PolicyConfig, {})

    configure_vllm_for_sampling_mask_replay(config)

    assert not sampling_mask_replay_enabled(config)
    assert "generation" not in config


def test_sampling_mask_replay_configures_vllm() -> None:
    config = _enabled_config()

    configure_vllm_for_sampling_mask_replay(config)

    assert sampling_mask_replay_enabled(config)
    generation = config["generation"]
    assert generation["vllm_kwargs"]["return_sampling_mask"] is True
    assert generation["vllm_cfg"]["logprobs_mode"] == "processed_logprobs"
    assert generation["vllm_cfg"]["env_vars"]["VLLM_USE_V2_MODEL_RUNNER"] == "1"


@pytest.mark.parametrize(
    ("path", "value", "match"),
    [
        (("generation", "backend"), "sglang", "vLLM generation"),
        (("megatron_cfg", "enabled"), False, "Megatron policy backend"),
        (("generation", "temperature"), 0.0, "temperature > 0"),
        (("generation", "top_k"), None, "top_k.*positive integer"),
        (("sequence_packing", "enabled"), True, "sequence packing"),
        (
            ("megatron_cfg", "use_fused_linear_logprobs"),
            True,
            "use_fused_linear_logprobs",
        ),
    ],
)
def test_sampling_mask_replay_rejects_unsupported_config(
    path: tuple[str, str], value: Any, match: str
) -> None:
    config = _enabled_config()
    section = cast(dict[str, Any], config[path[0]])
    section[path[1]] = value

    with pytest.raises(ValueError, match=match):
        configure_vllm_for_sampling_mask_replay(config)
