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
from nemo_rl.models.generation.dynamo.config import (
    DynamoCfg,
    dynamo_cfg,
    dynamo_chat_template,
    dynamo_parsers,
    native_default_chat_template_kwargs,
)


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

    # Dynamo serves the AsyncLLM, so the switch selects the async engine, and it
    # serves OpenAI clients, which need the engine's tokenizer.
    assert config["vllm_cfg"]["async_engine"] is True
    assert config["vllm_cfg"]["skip_tokenizer_init"] is False
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
    with pytest.raises(ValidationError, match="extra_cli_args"):
        dynamo_cfg({"dynamo_cfg": {"worker_args": {"extra_cli_args": []}}})
    with pytest.raises(ValidationError, match="duplicates"):
        dynamo_cfg({"dynamo_cfg": {"worker_args": {"endpoint_types": ["chat"] * 2}}})


def test_frontend_argv_is_runtime_owned() -> None:
    cfg = DynamoCfg.model_validate(
        {"frontend_args": {"router_mode": "kv", "extra_cli_args": ["--foo", "1"]}}
    )
    argv = build_dynamo_frontend_argv(
        host="0.0.0.0", port=3001, namespace="nemo-rl", dynamo_cfg=cfg
    )
    assert argv[argv.index("--router-mode") + 1] == "kv"
    # Gym names a token-capture call's session; keep the session on one engine.
    assert argv[argv.index("--router-session-affinity-ttl-secs") + 1] == "1800"
    assert argv[-2:] == ["--foo", "1"]
    no_affinity = DynamoCfg.model_validate(
        {"frontend_args": {"session_affinity_ttl_secs": None}}
    )
    assert "--router-session-affinity-ttl-secs" not in build_dynamo_frontend_argv(
        host="0.0.0.0", port=3001, namespace="nemo-rl", dynamo_cfg=no_affinity
    )

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


def _serving(**serving_chat_kwargs) -> dict:
    return _vllm_config(
        dynamo=True,
        vllm_cfg={
            "async_engine": True,
            "max_model_len": 512,
            "tensor_parallel_size": 1,
            "expert_parallel_size": 1,
            "http_server_serving_chat_kwargs": serving_chat_kwargs,
        },
    )


def test_parsers_follow_the_native_server_under_dynamos_names() -> None:
    assert dynamo_parsers(_serving(tool_parser="hermes", reasoning_parser="qwen3")) == (
        "hermes",
        "qwen3",
    )
    # NeMo RL's nano_v3 vLLM plugin is Dynamo's nemotron_v3 parser.
    nano_v3 = _serving(reasoning_parser="nano_v3")
    nano_v3["vllm_cfg"]["reasoning_parser_plugin"] = "nano_v3_reasoning_parser.py"
    assert dynamo_parsers(nano_v3) == (None, "nemotron_v3")
    assert dynamo_parsers(_serving(tool_parser="openai")) == ("harmony", None)
    # dynamo_cfg names them explicitly.
    explicit = _serving(tool_parser="hermes")
    explicit["dynamo_cfg"] = {"worker_args": {"tool_call_parser": "qwen3_coder"}}
    assert dynamo_parsers(explicit) == ("qwen3_coder", None)


def test_a_parser_plugin_without_a_dynamo_parser_is_rejected_at_startup() -> None:
    config = _serving(tool_parser="my_parser")
    config["vllm_cfg"]["tool_parser_plugin"] = "my_parser.py"

    with pytest.raises(ValueError, match="worker_args.tool_call_parser"):
        configure_generation_config(config, _Tokenizer())


def test_chat_template_and_default_kwargs_follow_the_native_server() -> None:
    config = _serving(
        chat_template="templates/chat.jinja",
        default_chat_template_kwargs={"enable_thinking": False},
    )
    assert dynamo_chat_template(config) == "templates/chat.jinja"
    assert native_default_chat_template_kwargs(config) == {"enable_thinking": False}
    # The older spelling the native server also accepts.
    legacy = _serving(chat_template_kwargs={"enable_thinking": True})
    assert native_default_chat_template_kwargs(legacy) == {"enable_thinking": True}
    assert dynamo_chat_template(_serving()) is None


@pytest.mark.parametrize(
    ("serving_chat_kwargs", "error"),
    [
        ({"chat_template_content_format": "string"}, "no content-format setting"),
        ({"enable_auto_tools": False}, "always parses"),
    ],
)
def test_chat_settings_the_frontend_has_no_counterpart_for_are_rejected(
    serving_chat_kwargs, error
) -> None:
    with pytest.raises(ValueError, match=error):
        configure_generation_config(_serving(**serving_chat_kwargs), _Tokenizer())
    # The native defaults are what the frontend does anyway.
    configure_generation_config(
        _serving(chat_template_content_format="auto", enable_auto_tools=True),
        _Tokenizer(),
    )


def test_a_bearer_token_is_rejected_because_the_frontend_checks_none() -> None:
    config = _vllm_config(dynamo=True)
    config["vllm_cfg"]["http_server_api_key_required"] = True

    with pytest.raises(ValueError, match="checks no bearer token"):
        configure_generation_config(config, _Tokenizer())
