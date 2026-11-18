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
import json
import subprocess
import sys
import typing

import pytest

from nemo_rl.models.generation import configure_generation_config
from nemo_rl.models.generation.dynamo.config import (
    DynamoCfg,
    DynamoFrontendArgs,
    dynamo_cfg,
    trace_export_env,
)
from nemo_rl.models.generation.dynamo.managed_runtime import (
    ChatProcessorOptions,
    frontend_argv,
)


class _Tokenizer:
    pad_token_id = 0
    eos_token_id = 1


def _vllm_config(**overrides) -> dict:
    config = {
        "backend": "vllm",
        "model_name": "Qwen/Qwen3-0.6B",
        "top_p": 1.0,
        "top_k": None,
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


@pytest.mark.parametrize("dynamo", [False, True])
def test_dynamo_true_alone_is_a_complete_config(dynamo: bool) -> None:
    config = configure_generation_config(_vllm_config(dynamo=dynamo), _Tokenizer())

    # Dynamo serves the AsyncLLM, so the switch selects the async engine.
    # Rollouts read token IDs only, so the engine skips its tokenizer as natively;
    # NeMo-Gym runs load it through expose_http_server, which Gym requires.
    assert config["vllm_cfg"]["async_engine"] is dynamo
    assert config["vllm_cfg"]["skip_tokenizer_init"] is True
    assert "dynamo_cfg" not in config
    assert dynamo_cfg(config) == DynamoCfg()


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
        # Gym's calls go through the Dynamo frontend, not vLLM's HTTP frontends.
        (
            _vllm_config(
                dynamo=True,
                vllm_cfg={
                    "async_engine": True,
                    "expose_http_server": True,
                    "max_model_len": 512,
                    "tensor_parallel_size": 1,
                    "expert_parallel_size": 1,
                    "http_api_server_count": 2,
                },
            ),
            "would serve none of them",
        ),
        # Direct routing needs a worker ID that rollouts never carry.
        (
            _vllm_config(
                dynamo=True, dynamo_cfg={"frontend_args": {"router_mode": "direct"}}
            ),
            "router_mode",
        ),
        # NeMo-Gym's chat calls are NeMo RL's chat engine's, never Dynamo's.
        (
            _vllm_config(
                dynamo=True,
                dynamo_cfg={
                    "frontend_args": {"extra_cli_args": ["--dyn-chat-processor=vllm"]}
                },
            ),
            "may not set --dyn-chat-processor",
        ),
        # Its settings are the managed runtime's.
        (
            _vllm_config(
                dynamo=True,
                dynamo_cfg={
                    "frontend_args": {
                        "extra_cli_args": ["--dyn-chat-processor-config", "{}"]
                    }
                },
            ),
            "may not set --dyn-chat-processor-config",
        ),
    ],
)
def test_configs_dynamo_would_not_serve_as_written_fail_at_startup(
    config, error
) -> None:
    with pytest.raises(ValueError, match=error):
        configure_generation_config(config, _Tokenizer())


def _frontend_argv(cfg: DynamoCfg) -> list[str]:
    return frontend_argv(
        host="0.0.0.0",
        port=3001,
        namespace="nemo-rl",
        dynamo_cfg=cfg,
        chat_options=ChatProcessorOptions(
            render_workers=cfg.frontend_args.render_workers,
            ray_address="10.0.0.1:6379",
            capture_file="/chat/capture.pkl",
            chat_settings_file=None,
        ),
    )


def test_frontend_argv_is_runtime_owned() -> None:
    cfg = DynamoCfg(
        frontend_args=DynamoFrontendArgs(
            router_mode="kv",
            extra_cli_args=["--foo", "1"],
            session_affinity_max_entries=1024,
        )
    )
    argv = _frontend_argv(cfg)
    assert argv[argv.index("--router-mode") + 1] == "kv"
    # Gym names a token-capture call's session; keep the session on one engine.
    assert argv[argv.index("--router-session-affinity-ttl-secs") + 1] == "300"
    assert argv[argv.index("--router-session-affinity-max-entries") + 1] == "1024"
    assert argv[-2:] == ["--foo", "1"]
    # One frontend has no replicas to sync with; several share in-flight load.
    assert "--router-replica-sync" not in argv
    replicas = DynamoCfg(frontend_args=DynamoFrontendArgs(num_frontends=2))
    assert "--router-replica-sync" in _frontend_argv(replicas)
    no_affinity = DynamoCfg.model_validate(
        {"frontend_args": {"session_affinity_ttl_secs": None}}
    )
    assert "--router-session-affinity-ttl-secs" not in _frontend_argv(no_affinity)

    clashing = DynamoCfg(
        frontend_args=DynamoFrontendArgs(extra_cli_args=["--http-port", "9"])
    )
    with pytest.raises(ValueError, match="--http-port is set by both"):
        _frontend_argv(clashing)


# Dynamo's own parser, in a fresh interpreter: importing dynamo.frontend.main
# replaces the process's logging.
_PARSE = """
import json, sys
import dynamo.frontend.main as frontend
for argv in json.loads(sys.stdin.read()):
    sys.argv = ["dynamo.frontend", *argv]
    frontend.parse_args()
"""


@pytest.mark.vllm
def test_dynamos_parser_accepts_the_frontend_arguments_for_every_router_mode() -> None:
    modes = typing.get_args(DynamoFrontendArgs.model_fields["router_mode"].annotation)
    argvs = [
        _frontend_argv(
            DynamoCfg(
                frontend_args=DynamoFrontendArgs(router_mode=mode, num_frontends=2)
            )
        )
        for mode in modes
    ]
    parsed = subprocess.run(
        [sys.executable, "-c", _PARSE],
        input=json.dumps(argvs),
        capture_output=True,
        text=True,
    )
    assert parsed.returncode == 0, parsed.stderr[-400:]


def test_dynamo_carries_trace_context_only_when_the_engines_trace_natively(
    monkeypatch,
) -> None:
    monkeypatch.setenv("OTEL_EXPORTER_OTLP_ENDPOINT", "http://collector:4317")
    monkeypatch.setenv("NEMO_RL_OTEL_ENABLED", "true")
    monkeypatch.setenv("NEMO_RL_OTEL_VLLM_NATIVE_TRACING", "false")
    assert trace_export_env() == {}

    # vLLM's own tracing is on: the frontends and engines carry the trace to it.
    monkeypatch.setenv("NEMO_RL_OTEL_VLLM_NATIVE_TRACING", "true")
    assert trace_export_env() == {"OTEL_EXPORT_ENABLED": "1"}

    # vLLM skips its tracing without a collector, so there is nothing to join.
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_ENDPOINT")
    monkeypatch.delenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", raising=False)
    assert trace_export_env() == {}
