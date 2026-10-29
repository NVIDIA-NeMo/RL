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
from pydantic import ValidationError

from nemo_rl.models.generation import configure_generation_config
from nemo_rl.models.generation.dynamo.arguments import (
    build_dynamo_frontend_argv,
    redact_argv,
    redact_environment,
)
from nemo_rl.models.generation.dynamo.config import DynamoCfg, dynamo_cfg


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


def test_dynamo_true_alone_is_a_complete_config() -> None:
    config = configure_generation_config(_vllm_config(dynamo=True), _Tokenizer())

    # Dynamo serves the AsyncLLM, so the switch selects the async engine.
    # Rollouts read token IDs only, so the engine skips its tokenizer as natively.
    assert config["vllm_cfg"]["async_engine"] is True
    assert config["vllm_cfg"]["skip_tokenizer_init"] is True
    assert dynamo_cfg(config) == DynamoCfg()
    assert dynamo_cfg(config).frontend_args.router_mode == "kv"


def test_without_dynamo_the_vllm_config_is_untouched() -> None:
    config = configure_generation_config(_vllm_config(), _Tokenizer())

    assert config["vllm_cfg"]["async_engine"] is False
    assert config["vllm_cfg"]["skip_tokenizer_init"] is True
    assert "dynamo_cfg" not in config


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
        # The async engine Dynamo serves runs no data parallelism inside vLLM.
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
    ],
)
def test_configs_dynamo_would_not_serve_as_written_fail_at_startup(
    config, error
) -> None:
    with pytest.raises(ValueError, match=error):
        configure_generation_config(config, _Tokenizer())


def test_dynamo_cfg_rejects_unknown_keys() -> None:
    with pytest.raises(ValidationError, match="worker_args"):
        dynamo_cfg({"dynamo_cfg": {"worker_args": {}}})


def test_frontend_argv_is_runtime_owned() -> None:
    cfg = DynamoCfg.model_validate(
        {"frontend_args": {"router_mode": "kv", "extra_cli_args": ["--foo", "1"]}}
    )
    argv = build_dynamo_frontend_argv(
        host="0.0.0.0", port=3001, namespace="nemo-rl", dynamo_cfg=cfg
    )
    assert argv[argv.index("--router-mode") + 1] == "kv"
    assert argv[-2:] == ["--foo", "1"]

    clashing = DynamoCfg.model_validate(
        {"frontend_args": {"extra_cli_args": ["--http-port", "9"]}}
    )
    with pytest.raises(ValueError, match="--http-port is set by both"):
        build_dynamo_frontend_argv(
            host="0.0.0.0", port=3001, namespace="nemo-rl", dynamo_cfg=clashing
        )


def test_redaction_hides_credentials() -> None:
    assert redact_argv(["frontend", "--api-key", "secret"])[2] == "<redacted>"
    assert redact_environment({"HF_TOKEN": "secret", "NCCL_DEBUG": "INFO"}) == {
        "HF_TOKEN": "<redacted>",
        "NCCL_DEBUG": "INFO",
    }
