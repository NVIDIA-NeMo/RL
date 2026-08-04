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

from nemo_rl.models.generation.dynamo.arguments import (
    build_dynamo_frontend_argv,
    build_dynamo_vllm_argv,
    build_managed_worker_env,
    redact_argv,
    redact_environment,
    validate_managed_vllm_config,
)
from nemo_rl.models.generation.dynamo.config import (
    DynamoCfg,
    DynamoConfig,
    DynamoWorkerArgs,
)


def _config(**overrides) -> dict:
    config = {
        "backend": "dynamo",
        "model_name": "Qwen/Qwen3-0.6B",
        "dynamo_cfg": _dynamo_cfg(),
        "vllm_cfg": {
            "async_engine": True,
            "tensor_parallel_size": 2,
            "pipeline_parallel_size": 1,
            "expert_parallel_size": 2,
            "gpu_memory_utilization": 0.8,
            "max_model_len": 512,
            "precision": "bfloat16",
            "kv_cache_dtype": "auto",
            "load_format": "auto",
            "enforce_eager": False,
            "expose_http_server": False,
            "enable_vllm_metrics_logger": True,
            "vllm_metrics_logger_interval": 1.0,
            "env_vars": None,
        },
        "vllm_kwargs": {},
        "colocated": {"enabled": False},
    }
    config.update(overrides)
    return config


def _dynamo_cfg() -> dict:
    return {
        "engine": "vllm",
        "startup_timeout_s": 600,
        "request_timeout_s": 900,
        "metrics_include_prefixes": None,
        "metrics_exclude_prefixes": None,
        "worker_args": {
            "tool_call_parser": None,
            "reasoning_parser": None,
            "exclude_tools_when_tool_choice_none": True,
            "enable_structural_tag": False,
            "structural_tag_scope": "auto",
            "structural_tag_schema": "auto",
            "custom_jinja_template": None,
            "endpoint_types": ["chat", "completions"],
            "extra_cli_args": [],
        },
        "frontend_args": {
            "tokenizer": "default",
            "tokenizer_cache": False,
            "tokenizer_cache_bytes": 50 * 1024 * 1024,
            "router_mode": "kv",
            "router_reset_states": True,
            "extra_cli_args": [],
        },
    }


def _flag_value(argv: list[str], flag: str) -> str:
    return argv[argv.index(flag) + 1]


def test_config_derives_world_size_and_rejects_removed_public_fields() -> None:
    assert DynamoConfig.model_validate(_config()).engine_world_size == 2
    for field in ("engine_world_size", "namespace", "dynamo_python", "etcd_port"):
        with pytest.raises(ValidationError, match=field):
            DynamoConfig.model_validate(
                _config(dynamo_cfg={field: 1 if field.endswith("port") else "x"})
            )


@pytest.mark.parametrize(
    ("override", "match"),
    [
        ({"vllm_cfg": {}}, "nonempty"),
        ({"sglang_cfg": {"foo": 1}}, "sglang_cfg"),
        ({"trtllm_cfg": {"foo": 1}}, "trtllm_cfg"),
        ({"colocated": {"enabled": True}}, "must be false"),
        ({"refit_transport": "nccl_reshard"}, "must be null"),
        ({"quant_cfg": "nvfp4"}, "quant_cfg"),
        ({"vllm_kwargs": {"speculative_config": {"model": "draft"}}}, "draft"),
    ],
)
def test_config_rejects_unsupported_modes(override, match) -> None:
    if override.get("vllm_cfg") == {}:
        config = _config()
        config["vllm_cfg"] = {}
    else:
        config = _config(**override)
    with pytest.raises(ValidationError, match=match):
        DynamoConfig.model_validate(config)


@pytest.mark.parametrize(
    "vllm_cfg",
    [
        {"tensor_parallel_size": 2, "expert_parallel_size": 3},
        {"tensor_parallel_size": 1, "precision": "fp8"},
        {"tensor_parallel_size": 1, "kv_cache_dtype": "fp8"},
    ],
)
def test_config_rejects_unsupported_parallelism_and_precision(vllm_cfg) -> None:
    config = _config()
    config["vllm_cfg"].update(vllm_cfg)
    with pytest.raises(ValidationError):
        DynamoConfig.model_validate(config)


def test_worker_argv_translates_structured_fields_and_warns_unclassified() -> None:
    config = _dynamo_cfg()
    config["worker_args"].update(
        {"tool_call_parser": "qwen3_coder", "reasoning_parser": "nemotron_nano"}
    )
    cfg = DynamoCfg.model_validate(config)
    vllm_cfg = _config()["vllm_cfg"] | {"unclassified_field": 1}
    with pytest.warns(UserWarning, match="unclassified_field"):
        validate_managed_vllm_config(vllm_cfg)
    argv = build_dynamo_vllm_argv(
        model_name="model",
        namespace="nemo-rl-1",
        seed=7,
        vllm_cfg=vllm_cfg,
        vllm_kwargs={"max_num_seqs": 16},
        dynamo_cfg=cfg,
    )

    assert _flag_value(argv, "--model") == "model"
    assert _flag_value(argv, "--weight-transfer-config") == '{"backend":"nccl"}'
    assert _flag_value(argv, "--dyn-tool-call-parser") == "qwen3_coder"
    assert _flag_value(argv, "--dyn-reasoning-parser") == "nemotron_nano"
    assert _flag_value(argv, "--max-num-seqs") == "16"
    assert "--enable-expert-parallel" in argv


def test_worker_argv_rejects_replaced_and_managed_options() -> None:
    with pytest.raises(ValueError, match="worker_args.custom_jinja_template"):
        validate_managed_vllm_config(
            _config()["vllm_cfg"]
            | {"http_server_serving_chat_kwargs": {"tool_parser": "x"}}
        )
    config = _dynamo_cfg()
    config["worker_args"]["extra_cli_args"] = ["--model", "other"]
    with pytest.raises(ValueError, match="--model is set by both"):
        build_dynamo_vllm_argv(
            model_name="model",
            namespace="namespace",
            seed=0,
            vllm_cfg={},
            vllm_kwargs={},
            dynamo_cfg=DynamoCfg.model_validate(config),
        )


def test_frontend_argv_and_environment_are_runtime_owned() -> None:
    cfg = DynamoCfg.model_validate(_dynamo_cfg())
    argv = build_dynamo_frontend_argv(
        host="0.0.0.0", port=3001, namespace="nemo-rl", dynamo_cfg=cfg
    )
    assert _flag_value(argv, "--router-mode") == "kv"

    env = build_managed_worker_env(
        base_env={"DYN_NAMESPACE": "stale", "NCCL_DEBUG": "INFO"},
        configured_env={"NCCL_IB_DISABLE": "0"},
        manager_env={"DYN_NAMESPACE": "owned", "DYN_SYSTEM_PORT": "4000"},
    )
    assert env["DYN_NAMESPACE"] == "owned"
    assert env["DYN_SYSTEM_PORT"] == "4000"
    with pytest.raises(ValueError, match="VLLM_PORT"):
        build_managed_worker_env(
            base_env={},
            configured_env={"VLLM_PORT": "9999"},
            manager_env={"VLLM_PORT": "7000"},
        )


def test_every_worker_config_field_reaches_argv(monkeypatch) -> None:
    from nemo_rl.models.generation.dynamo import arguments

    sources: set[str] = set()
    original_add = arguments._ArgvBuilder.add

    def record_source(self, flag, value=None, *, source):
        sources.add(source)
        return original_add(self, flag, value, source=source)

    monkeypatch.setattr(arguments._ArgvBuilder, "add", record_source)
    config = _dynamo_cfg()
    config["worker_args"].update(
        {
            "tool_call_parser": "qwen3_coder",
            "reasoning_parser": "nemotron_nano",
            "custom_jinja_template": "template",
        }
    )
    cfg = DynamoCfg.model_validate(config)
    build_dynamo_vllm_argv(
        model_name="model",
        namespace="namespace",
        seed=0,
        vllm_cfg={"tensor_parallel_size": 1},
        vllm_kwargs={},
        dynamo_cfg=cfg,
    )

    configured_fields = {
        source.rsplit(".", 1)[-1]
        for source in sources
        if source.startswith("dynamo_cfg.worker_args.")
    }
    assert set(DynamoWorkerArgs.model_fields) - {"extra_cli_args"} <= configured_fields


def test_redaction_hides_credentials() -> None:
    assert redact_argv(["worker", "--api-key", "secret"])[2] == "<redacted>"
    assert redact_environment({"HF_TOKEN": "secret", "NCCL_DEBUG": "INFO"}) == {
        "HF_TOKEN": "<redacted>",
        "NCCL_DEBUG": "INFO",
    }
