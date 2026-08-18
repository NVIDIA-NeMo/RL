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

import warnings

import pytest
from pydantic import ValidationError

from nemo_rl.models.generation import configure_generation_config
from nemo_rl.models.generation.dynamo.arguments import (
    build_dynamo_frontend_argv,
    build_dynamo_vllm_argv,
    build_managed_worker_env,
    redact_argv,
    redact_environment,
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
        "control_timeout_s": 600,
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
    ("vllm_cfg", "match"),
    [
        (
            {"tensor_parallel_size": 2, "expert_parallel_size": 3},
            "expert_parallel_size",
        ),
        ({"precision": "fp8"}, "precision"),
        ({"kv_cache_dtype": "fp8"}, "kv_cache_dtype"),
    ],
)
def test_config_rejects_unsupported_parallelism_and_precision(vllm_cfg, match) -> None:
    config = _config()
    config["vllm_cfg"].update(vllm_cfg)
    with pytest.raises(ValidationError, match=match):
        DynamoConfig.model_validate(config)


def test_config_classifies_managed_logprobs_and_runtime_fields() -> None:
    config = _config()
    config["vllm_cfg"].update(
        {
            "logprobs_mode": "processed_logprobs",
            "cap_max_tokens_to_context": False,
            "use_deep_gemm": False,
            "num_first_layers_in_bf16": 0,
            "num_last_layers_in_bf16": 0,
        }
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        DynamoConfig.model_validate(config)
    assert caught == []

    config["vllm_cfg"]["logprobs_mode"] = "raw_logprobs"
    with pytest.raises(ValidationError, match="processed_logprobs"):
        DynamoConfig.model_validate(config)


@pytest.mark.parametrize("field", ["skip_tokenizer_init", "cap_max_tokens_to_context"])
def test_config_warns_only_for_active_unsupported_fields(field) -> None:
    config = _config()
    config["vllm_cfg"][field] = True

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        DynamoConfig.model_validate(config)
    assert [str(warning.message) for warning in caught] == [
        f"policy.generation.vllm_cfg.{field} is ignored by backend='dynamo'"
    ]


@pytest.mark.parametrize(
    "field",
    [
        "data_parallel_size",
        "prefill_context_parallel_size",
        "decode_context_parallel_size",
    ],
)
@pytest.mark.parametrize("source", ["vllm_cfg", "vllm_kwargs"])
def test_config_rejects_parallel_dimensions_outside_tp_pp(field, source) -> None:
    config = _config()
    config[source][field] = 2

    with pytest.raises(ValidationError, match=f"{source}.{field} must be 1"):
        DynamoConfig.model_validate(config)


def test_config_rejects_more_than_32_stop_strings() -> None:
    config = _config(stop_strings=[str(index) for index in range(33)])

    with pytest.raises(ValidationError, match="stop_strings supports at most 32"):
        DynamoConfig.model_validate(config)


def test_config_does_not_limit_stop_token_ids() -> None:
    config = _config(stop_token_ids=list(range(33)))

    assert DynamoConfig.model_validate(config).model_extra["stop_token_ids"] == list(
        range(33)
    )


def test_configure_generation_config_selects_dynamo_load_format() -> None:
    class Tokenizer:
        pad_token_id = 0
        eos_token_id = 1

    training = _config(stop_token_ids=None)
    evaluation = _config(stop_token_ids=None)
    del training["vllm_cfg"]["load_format"]
    del evaluation["vllm_cfg"]["load_format"]

    assert (
        configure_generation_config(training, Tokenizer())["vllm_cfg"]["load_format"]
        == "dummy"
    )
    assert (
        configure_generation_config(evaluation, Tokenizer(), is_eval=True)["vllm_cfg"][
            "load_format"
        ]
        == "auto"
    )


def test_worker_argv_translates_structured_fields_and_warns_unclassified() -> None:
    config = _dynamo_cfg()
    config["worker_args"].update(
        {"tool_call_parser": "qwen3_coder", "reasoning_parser": "nemotron_nano"}
    )
    cfg = DynamoCfg.model_validate(config)
    generation_config = _config()
    generation_config["vllm_cfg"]["unclassified_field"] = 1
    with pytest.warns(UserWarning, match="unclassified_field"):
        validated = DynamoConfig.model_validate(generation_config)
    vllm_cfg = validated.vllm_cfg.model_dump()
    argv = build_dynamo_vllm_argv(
        model_name="model",
        namespace="nemo-rl-1",
        seed=7,
        vllm_cfg=vllm_cfg,
        vllm_kwargs={"max_num_seqs": 16, "hf_overrides": {"rope_theta": 1e6}},
        dynamo_cfg=cfg,
    )

    assert _flag_value(argv, "--model") == "model"
    assert _flag_value(argv, "--weight-transfer-config") == '{"backend":"nccl"}'
    assert _flag_value(argv, "--dyn-tool-call-parser") == "qwen3_coder"
    assert _flag_value(argv, "--dyn-reasoning-parser") == "nemotron_nano"
    assert _flag_value(argv, "--max-num-seqs") == "16"
    assert _flag_value(argv, "--hf-overrides") == '{"rope_theta":1000000.0}'
    assert "--enable-expert-parallel" in argv


def test_worker_argv_rejects_replaced_and_managed_options() -> None:
    generation_config = _config()
    generation_config["vllm_cfg"]["http_server_serving_chat_kwargs"] = {
        "tool_parser": "x"
    }
    with pytest.raises(ValueError, match="worker_args.custom_jinja_template"):
        DynamoConfig.model_validate(generation_config)
    config = _dynamo_cfg()
    config["worker_args"]["extra_cli_args"] = ["--model", "other"]
    with pytest.raises(ValueError, match="--model is set by both"):
        build_dynamo_vllm_argv(
            model_name="model",
            namespace="namespace",
            seed=0,
            vllm_cfg=_config()["vllm_cfg"],
            vllm_kwargs={},
            dynamo_cfg=DynamoCfg.model_validate(config),
        )


def test_config_accepts_inherited_unused_sections() -> None:
    config = _config(
        mcore_generation_config={"some_shared_setting": True},
        refit_cfg={"some_shared_setting": True},
    )

    validated = DynamoConfig.model_validate(config)

    assert validated.model_extra["mcore_generation_config"] == {
        "some_shared_setting": True
    }
    assert validated.model_extra["refit_cfg"] == {"some_shared_setting": True}


def test_frontend_argv_and_environment_are_runtime_owned() -> None:
    cfg = DynamoCfg.model_validate(_dynamo_cfg())
    argv = build_dynamo_frontend_argv(
        host="0.0.0.0", port=3001, namespace="nemo-rl", dynamo_cfg=cfg
    )
    assert _flag_value(argv, "--router-mode") == "kv"

    env = build_managed_worker_env(
        base_env={
            "DYN_NAMESPACE": "stale",
            "ETCD_ENDPOINTS": "http://stale-etcd:2379",
            "ETCD_USERNAME": "stale-user",
            "NATS_SERVER": "nats://stale-nats:4222",
            "NATS_AUTH_TOKEN": "stale-token",
            "NCCL_DEBUG": "INFO",
        },
        configured_env={"NCCL_IB_DISABLE": "0"},
        manager_env={
            "DYN_NAMESPACE": "owned",
            "DYN_SYSTEM_PORT": "4000",
            "ETCD_ENDPOINTS": "http://managed-etcd:2379",
            "NATS_SERVER": "nats://managed-nats:4222",
        },
    )
    assert env["DYN_NAMESPACE"] == "owned"
    assert env["DYN_SYSTEM_PORT"] == "4000"
    assert env["ETCD_ENDPOINTS"] == "http://managed-etcd:2379"
    assert env["NATS_SERVER"] == "nats://managed-nats:4222"
    assert "ETCD_USERNAME" not in env
    assert "NATS_AUTH_TOKEN" not in env
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
        vllm_cfg=_config()["vllm_cfg"],
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
    assert redact_argv(
        ["worker", "--max-num-batched-tokens", "8192", "--stop-token-ids", "1,2"]
    ) == ["worker", "--max-num-batched-tokens", "8192", "--stop-token-ids", "1,2"]
    assert redact_environment({"HF_TOKEN": "secret", "NCCL_DEBUG": "INFO"}) == {
        "HF_TOKEN": "<redacted>",
        "NCCL_DEBUG": "INFO",
    }
