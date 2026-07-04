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
)
from nemo_rl.models.generation.dynamo.config import DynamoCfg


def _managed_cfg(**overrides) -> DynamoCfg:
    return DynamoCfg.model_validate(
        {
            "deployment": "ray",
            "engine_world_size": 1,
            "frontend_port": 0,
            **overrides,
        }
    )


def _flag_value(argv: list[str], flag: str) -> str:
    return argv[argv.index(flag) + 1]


def test_ray_deployment_rejects_external_frontend_fields() -> None:
    with pytest.raises(ValidationError, match="owns its frontend"):
        _managed_cfg(frontend_url="http://example.test/v1")
    with pytest.raises(ValidationError, match="owns its frontend"):
        _managed_cfg(dgd_name="my-dgd")


def test_ray_deployment_uses_managed_runtime_defaults() -> None:
    cfg = DynamoCfg.model_validate({"deployment": "ray", "engine_world_size": 1})
    assert cfg.frontend_port == 0
    assert cfg.dynamo_python == "/opt/dynamo_venv/bin/python"


def test_qwen_worker_argv_has_managed_refit_flags_without_parser_defaults() -> None:
    argv = build_dynamo_vllm_argv(
        model_name="Qwen/Qwen2.5-1.5B",
        namespace="nemo-rl-123",
        seed=7,
        vllm_cfg={
            "tensor_parallel_size": 1,
            "pipeline_parallel_size": 1,
            "expert_parallel_size": 1,
            "gpu_memory_utilization": 0.6,
            "max_model_len": 512,
            "precision": "bfloat16",
            "async_engine": False,
            "http_server_serving_chat_kwargs": {"tool_parser": "nemotron_json"},
        },
        vllm_kwargs={},
        dynamo_cfg=_managed_cfg(),
    )

    assert _flag_value(argv, "--model") == "Qwen/Qwen2.5-1.5B"
    assert _flag_value(argv, "--served-model-name") == "Qwen/Qwen2.5-1.5B"
    assert _flag_value(argv, "--weight-transfer-config") == '{"backend":"nccl"}'
    assert _flag_value(argv, "--event-plane") == "nats"
    assert "--enable-rl" in argv
    assert "--dyn-tool-call-parser" not in argv
    assert "--dyn-reasoning-parser" not in argv
    assert "nemotron_json" not in argv
    assert "--async-engine" not in argv


def test_nemotron_worker_argv_maps_dynamo_parsers_and_advanced_vllm_values() -> None:
    cfg = _managed_cfg(
        engine_world_size=2,
        worker_args={
            "tool_call_parser": "nemotron_deci",
            "reasoning_parser": "nemotron_nano",
        },
    )
    argv = build_dynamo_vllm_argv(
        model_name="nvidia/NVIDIA-Nemotron-Nano-9B-v2",
        namespace="nemo-rl-456",
        seed=0,
        vllm_cfg={
            "tensor_parallel_size": 2,
            "pipeline_parallel_size": 1,
            "expert_parallel_size": 1,
            "max_model_len": 8192,
            "enforce_eager": True,
        },
        vllm_kwargs={
            "compilation_config": {"backend": "eager"},
            "mamba_ssm_cache_dtype": "float32",
            "max_num_batched_tokens": 16384,
            "some_disabled_feature": False,
            "ignored_value": None,
        },
        dynamo_cfg=cfg,
    )

    assert _flag_value(argv, "--dyn-tool-call-parser") == "nemotron_deci"
    assert _flag_value(argv, "--dyn-reasoning-parser") == "nemotron_nano"
    assert _flag_value(argv, "--compilation-config") == '{"backend":"eager"}'
    assert _flag_value(argv, "--mamba-ssm-cache-dtype") == "float32"
    assert _flag_value(argv, "--max-num-batched-tokens") == "16384"
    assert "--enforce-eager" in argv
    assert "--exclude-tools-when-tool-choice-none" in argv
    assert "--no-dyn-enable-structural-tag" in argv
    assert "--no-some-disabled-feature" in argv
    assert "--ignored-value" not in argv


def test_nemotron_nano_v35_swe_tp4_worker_argv() -> None:
    model = "/models/nemotron-nano-v3.5"
    cfg = _managed_cfg(
        engine_world_size=4,
        worker_args={
            "tool_call_parser": "qwen3_coder",
            "reasoning_parser": "nemotron_nano",
            "custom_jinja_template": f"{model}/chat_template.jinja",
        },
    )
    argv = build_dynamo_vllm_argv(
        model_name=model,
        namespace="nemo-rl-swe-r1",
        seed=0,
        vllm_cfg={
            "precision": "bfloat16",
            "kv_cache_dtype": "auto",
            "tensor_parallel_size": 4,
            "pipeline_parallel_size": 1,
            "expert_parallel_size": 1,
            "gpu_memory_utilization": 0.85,
            "max_model_len": 196608,
            "enforce_eager": False,
            "load_format": "auto",
            # These NeMo-RL HTTP-serving fields must never reach Dynamo.
            "async_engine": False,
            "expose_http_server": True,
            "reasoning_parser_plugin": "legacy-plugin.py",
            "http_server_serving_chat_kwargs": {"tool_parser": "nemotron_json"},
        },
        vllm_kwargs={
            "attention_backend": "FLASH_ATTN",
            "moe_backend": "triton",
            "mamba_ssm_cache_dtype": "float32",
            "compilation_config": {
                "cudagraph_capture_sizes": [1, 2, 4, 8, 16, 32, 64],
                "pass_config": {"fuse_allreduce_rms": False},
            },
        },
        dynamo_cfg=cfg,
    )

    assert _flag_value(argv, "--tensor-parallel-size") == "4"
    assert _flag_value(argv, "--pipeline-parallel-size") == "1"
    assert _flag_value(argv, "--dyn-tool-call-parser") == "qwen3_coder"
    assert _flag_value(argv, "--dyn-reasoning-parser") == "nemotron_nano"
    assert _flag_value(argv, "--custom-jinja-template") == (
        f"{model}/chat_template.jinja"
    )
    assert _flag_value(argv, "--attention-backend") == "FLASH_ATTN"
    assert _flag_value(argv, "--moe-backend") == "triton"
    assert _flag_value(argv, "--mamba-ssm-cache-dtype") == "float32"
    assert _flag_value(argv, "--compilation-config") == (
        '{"cudagraph_capture_sizes":[1,2,4,8,16,32,64],'
        '"pass_config":{"fuse_allreduce_rms":false}}'
    )
    assert "--no-enforce-eager" in argv
    assert "--async-engine" not in argv
    assert "--expose-http-server" not in argv
    assert "--reasoning-parser-plugin" not in argv
    assert "--http-server-serving-chat-kwargs" not in argv
    assert "legacy-plugin.py" not in argv
    assert "nemotron_json" not in argv


def test_worker_argv_rejects_duplicate_and_reserved_options() -> None:
    with pytest.raises(
        ValueError, match="structured engine option --tensor-parallel-size"
    ):
        build_dynamo_vllm_argv(
            model_name="model",
            namespace="namespace",
            seed=0,
            vllm_cfg={"tensor_parallel_size": 1},
            vllm_kwargs={"tensor_parallel_size": 2},
            dynamo_cfg=_managed_cfg(),
        )

    with pytest.raises(ValueError, match="may not override managed option --model"):
        build_dynamo_vllm_argv(
            model_name="model",
            namespace="namespace",
            seed=0,
            vllm_cfg={"tensor_parallel_size": 1},
            vllm_kwargs={},
            dynamo_cfg=_managed_cfg(
                worker_args={"extra_cli_args": ["--model", "other-model"]}
            ),
        )

    with pytest.raises(ValueError, match="structured Dynamo option"):
        build_dynamo_vllm_argv(
            model_name="model",
            namespace="namespace",
            seed=0,
            vllm_cfg={"tensor_parallel_size": 1},
            vllm_kwargs={},
            dynamo_cfg=_managed_cfg(
                worker_args={"extra_cli_args": ["--dyn-tool-call-parser", "hermes"]}
            ),
        )

    with pytest.raises(
        ValueError, match="structured engine option --tensor-parallel-size"
    ):
        build_dynamo_vllm_argv(
            model_name="model",
            namespace="namespace",
            seed=0,
            vllm_cfg={},
            vllm_kwargs={},
            dynamo_cfg=_managed_cfg(
                worker_args={"extra_cli_args": ["--tensor-parallel-size", "2"]}
            ),
        )

    with pytest.raises(ValueError, match="may not override managed option --endpoint"):
        build_dynamo_vllm_argv(
            model_name="model",
            namespace="namespace",
            seed=0,
            vllm_cfg={},
            vllm_kwargs={"endpoint": "dyn://other.backend.generate"},
            dynamo_cfg=_managed_cfg(),
        )

    with pytest.raises(ValueError, match="set by both"):
        build_dynamo_vllm_argv(
            model_name="model",
            namespace="namespace",
            seed=0,
            vllm_cfg={},
            vllm_kwargs={"max_num_seqs": 4},
            dynamo_cfg=_managed_cfg(
                worker_args={"extra_cli_args": ["--max_num_seqs", "8"]}
            ),
        )

    with pytest.raises(ValueError, match="invalid positional argument"):
        build_dynamo_vllm_argv(
            model_name="model",
            namespace="namespace",
            seed=0,
            vllm_cfg={},
            vllm_kwargs={},
            dynamo_cfg=_managed_cfg(
                worker_args={"extra_cli_args": ["--some-option", "value", "stray"]}
            ),
        )


def test_frontend_argv_is_structured_and_rejects_raw_overrides() -> None:
    cfg = _managed_cfg(
        frontend_args={"router_mode": "round-robin", "router_reset_states": True}
    )
    argv = build_dynamo_frontend_argv(
        host="0.0.0.0", port=1234, namespace="nemo-rl", dynamo_cfg=cfg
    )
    assert _flag_value(argv, "--router-mode") == "round-robin"
    assert "--router-reset-states" in argv

    cfg = _managed_cfg(frontend_args={"extra_cli_args": ["--http-port", "9999"]})
    with pytest.raises(ValueError, match="managed option --http-port"):
        build_dynamo_frontend_argv(
            host="0.0.0.0", port=1234, namespace="nemo-rl", dynamo_cfg=cfg
        )


def test_managed_environment_scrubs_semantic_dynamo_values() -> None:
    env = build_managed_worker_env(
        base_env={
            "PATH": "/usr/bin",
            "DYN_TOOL_CALL_PARSER": "wrong",
            "DYN_NAMESPACE": "wrong",
            "NCCL_DEBUG": "WARN",
        },
        configured_env={"NCCL_IB_DISABLE": "0"},
        manager_env={"DYN_NAMESPACE": "expected", "DYN_SYSTEM_PORT": "29000"},
    )
    assert env["DYN_NAMESPACE"] == "expected"
    assert env["DYN_SYSTEM_PORT"] == "29000"
    assert "DYN_TOOL_CALL_PARSER" not in env
    assert env["NCCL_DEBUG"] == "WARN"

    with pytest.raises(ValueError, match="DYN_REASONING_PARSER"):
        build_managed_worker_env(
            base_env={},
            configured_env={"DYN_REASONING_PARSER": "qwen3"},
            manager_env={},
        )


def test_redact_argv_hides_credential_values() -> None:
    assert redact_argv(["worker", "--api-key", "secret-value", "--model", "m"]) == [
        "worker",
        "--api-key",
        "<redacted>",
        "--model",
        "m",
    ]
    assert redact_environment({"HF_TOKEN": "secret-value", "NCCL_DEBUG": "INFO"}) == {
        "HF_TOKEN": "<redacted>",
        "NCCL_DEBUG": "INFO",
    }
    assert redact_argv(["worker", "--api-key=secret-value"]) == [
        "worker",
        "--api-key=<redacted>",
    ]
