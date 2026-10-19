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

import pytest

from nemo_rl.models.generation.dynamo.megatron_engine_pool import (
    build_megatron_adapter_argv,
    engine_layout_argv,
    engine_policy_config,
)
from nemo_rl.models.generation.dynamo.role_plan import EngineSpec


def _argv(**overrides):
    kwargs = dict(
        role="decode",
        component="backend",
        model_name="Qwen/Qwen3-0.6B",
        namespace="nemo-rl-7",
        parent_event_host="10.0.0.5",
        parent_event_port=4321,
        coordinator_host="10.0.0.5",
        megatron_root="/opt/megatron-lm",
        engine_start_timeout_s=600,
        handoff_owner="owner-a",
        endpoint_types=["completions"],
        engine_argv=["--tensor-model-parallel-size", "2"],
    )
    kwargs.update(overrides)
    return build_megatron_adapter_argv(**kwargs)


def _flag(argv: list[str], flag: str) -> str:
    return argv[argv.index(flag) + 1]


def test_adapter_argv_is_external_mode_with_role_and_component() -> None:
    argv = _argv()
    assert _flag(argv, "--engine-launch-mode") == "external"
    assert _flag(argv, "--role") == "decode"
    assert _flag(argv, "--component") == "backend"
    assert _flag(argv, "--parent-event-host") == "10.0.0.5"
    assert _flag(argv, "--parent-event-port") == "4321"
    assert _flag(argv, "--coordinator-host") == "10.0.0.5"
    assert _flag(argv, "--namespace") == "nemo-rl-7"
    assert _flag(argv, "--model") == "Qwen/Qwen3-0.6B"
    assert _flag(argv, "--served-model-name") == "Qwen/Qwen3-0.6B"
    assert _flag(argv, "--endpoint-types") == "completions"
    assert _flag(argv, "--handoff-owner") == "owner-a"
    assert "--nproc-per-node" not in argv
    # The adapter's parser requires an engine argv after '--' even though
    # external mode never launches it.
    assert argv[argv.index("--") + 1 :] == ["--tensor-model-parallel-size", "2"]


def test_prefill_adapter_has_its_own_component_and_no_owner() -> None:
    argv = _argv(role="prefill", component="prefill")
    assert _flag(argv, "--component") == "prefill"
    assert "--handoff-owner" not in argv


@pytest.mark.parametrize(
    "extra",
    [
        ["--engine-launch-mode", "local"],
        ["--engine-launch-mode=local"],
        ["--nproc-per-node", "2"],
        ["--parent-event-port", "9"],
        ["--handoff-owner", "x"],
    ],
)
def test_user_args_may_not_take_over_launch_or_ownership(extra) -> None:
    with pytest.raises(ValueError, match="owns it"):
        _argv(extra_cli_args=extra)


def test_user_args_may_not_append_an_engine_command() -> None:
    with pytest.raises(ValueError, match="may not contain '--'"):
        _argv(extra_cli_args=["--", "--load", "/x"])
    with pytest.raises(ValueError, match="role must be"):
        _argv(role="aggregated")


def test_engine_policy_config_applies_role_layout_to_both_layout_sources() -> None:
    policy = {
        "model_name": "m",
        "megatron_cfg": {
            "enabled": True,
            "tensor_model_parallel_size": 2,
            "pipeline_model_parallel_size": 1,
            "context_parallel_size": 1,
        },
        "generation": {
            "backend": "dynamo",
            "mcore_generation_config": {
                "tensor_model_parallel_size": 2,
                "max_requests": 8,
                "enable_prefix_caching": True,
            },
        },
    }
    spec = EngineSpec(
        role="prefill",
        index=0,
        world_size=4,
        engine_cfg={
            "megatron_cfg": {"tensor_model_parallel_size": 4},
            "mcore_generation_config": {"max_requests": 16},
        },
    )
    engine = engine_policy_config(policy, spec)
    assert engine["megatron_cfg"]["tensor_model_parallel_size"] == 4
    assert engine["megatron_cfg"]["context_parallel_size"] == 1
    mcore = engine["generation"]["mcore_generation_config"]
    assert mcore["tensor_model_parallel_size"] == 4
    assert mcore["max_requests"] == 16
    assert mcore["enable_prefix_caching"] is True
    # The source config is untouched.
    assert policy["megatron_cfg"]["tensor_model_parallel_size"] == 2
    assert policy["generation"]["mcore_generation_config"]["max_requests"] == 8
    assert engine_layout_argv(engine["megatron_cfg"]) == [
        "--tensor-model-parallel-size",
        "4",
        "--pipeline-model-parallel-size",
        "1",
    ]
