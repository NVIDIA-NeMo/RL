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
from pathlib import Path

import pytest
import yaml

from nemo_rl.models.generation import configure_generation_config
from nemo_rl.models.generation.dynamo.config import DynamoCfg, dynamo_cfg
from nemo_rl.models.generation.dynamo.managed_runtime import frontend_argv


class _Tokenizer:
    pad_token_id = 0
    eos_token_id = 1


def _vllm_config(**overrides) -> dict:
    config = {
        "backend": "vllm",
        "model_name": "Qwen/Qwen3-0.6B",
        "stop_token_ids": None,
        "stop_strings": None,
        "vllm_cfg": {
            "async_engine": False,
            "max_model_len": 512,
            "tensor_parallel_size": 1,
            "expert_parallel_size": 1,
        },
    }
    config.update(overrides)
    return config


_EXEMPLAR = (
    Path(__file__).resolve().parents[4]
    / "examples/configs/grpo_math_1B_dynamo_single_controller.yaml"
)


@pytest.mark.parametrize("dynamo", [False, True])
def test_dynamo_true_alone_is_a_complete_config(dynamo: bool) -> None:
    config = configure_generation_config(_vllm_config(dynamo=dynamo), _Tokenizer())

    # Dynamo serves the AsyncLLM, so the switch selects the async engine.
    # Rollouts read token IDs only, so the engine skips its tokenizer as natively.
    assert config["vllm_cfg"]["async_engine"] is dynamo
    assert config["vllm_cfg"]["skip_tokenizer_init"] is True
    assert "dynamo_cfg" not in config
    # The exemplar spells out exactly the defaults.
    exemplar = yaml.safe_load(_EXEMPLAR.read_text())["policy"]["generation"]
    assert dynamo_cfg(config) == DynamoCfg() == dynamo_cfg(exemplar)


@pytest.mark.parametrize(
    ("config", "error"),
    [
        # A colocated engine sleeps while the policy trains.
        (
            _vllm_config(dynamo=True, colocated={"enabled": True}),
            "requires non-colocated generation",
        ),
        (_vllm_config(dynamo=True, backend="sglang"), "backend=vllm"),
        # dynamo_cfg alone would be ignored.
        (_vllm_config(dynamo_cfg={}), "dynamo is not true"),
        # Dynamo does not serve data parallelism inside vLLM yet.
        (
            _vllm_config(
                dynamo=True,
                vllm_cfg={
                    "async_engine": False,
                    "max_model_len": 512,
                    "tensor_parallel_size": 1,
                    "expert_parallel_size": 2,
                },
            ),
            "expert_parallel_size <= tensor_parallel_size",
        ),
        # The frontend serves the same engines and checks no bearer token.
        (
            _vllm_config(
                dynamo=True,
                vllm_cfg={
                    "async_engine": True,
                    "max_model_len": 512,
                    "tensor_parallel_size": 1,
                    "expert_parallel_size": 1,
                    "http_server_api_key_required": True,
                },
            ),
            "checks no bearer token",
        ),
        # Direct routing needs a worker ID that rollouts never carry.
        (
            _vllm_config(
                dynamo=True, dynamo_cfg={"frontend_args": {"router_mode": "direct"}}
            ),
            "router_mode",
        ),
    ],
)
def test_configs_dynamo_would_not_serve_as_written_fail_at_startup(
    config, error
) -> None:
    with pytest.raises(ValueError, match=error):
        configure_generation_config(config, _Tokenizer())


def test_frontend_argv_is_runtime_owned() -> None:
    cfg = DynamoCfg.model_validate(
        {"frontend_args": {"router_mode": "kv", "extra_cli_args": ["--foo", "1"]}}
    )
    argv = frontend_argv(host="0.0.0.0", port=3001, namespace="nemo-rl", dynamo_cfg=cfg)
    assert argv[argv.index("--router-mode") + 1] == "kv"
    assert argv[-2:] == ["--foo", "1"]

    clashing = DynamoCfg.model_validate(
        {"frontend_args": {"extra_cli_args": ["--http-port", "9"]}}
    )
    with pytest.raises(ValueError, match="--http-port is set by both"):
        frontend_argv(
            host="0.0.0.0", port=3001, namespace="nemo-rl", dynamo_cfg=clashing
        )
