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

"""Deterministic argument and environment construction for managed Dynamo."""

import json
from collections.abc import Mapping, Sequence
from typing import Any

from nemo_rl.models.generation.dynamo.config import DynamoCfg

_STANDARD_VLLM_FLAGS: dict[str, str] = {
    "tensor_parallel_size": "--tensor-parallel-size",
    "pipeline_parallel_size": "--pipeline-parallel-size",
    "gpu_memory_utilization": "--gpu-memory-utilization",
    "max_model_len": "--max-model-len",
    "kv_cache_dtype": "--kv-cache-dtype",
    "load_format": "--load-format",
    "precision": "--dtype",
    "enforce_eager": "--enforce-eager",
}

_MANAGED_FLAGS = {
    "--component",
    "--model",
    "--model-name",
    "--model-path",
    "--served-model-name",
    "--namespace",
    "--endpoint",
    "--discovery-backend",
    "--request-plane",
    "--event-plane",
    "--enable-rl",
    "--weight-transfer-config",
    "--trust-remote-code",
    "--seed",
    "--http-host",
    "--http-port",
    "--namespace-prefix",
    "--router-mode",
}

_STRUCTURED_ENGINE_FLAGS = set(_STANDARD_VLLM_FLAGS.values()) | {
    "--enable-expert-parallel",
}

_STRUCTURED_DYNAMO_FLAGS = {
    "--dyn-tool-call-parser",
    "--dyn-reasoning-parser",
    "--exclude-tools-when-tool-choice-none",
    "--dyn-enable-structural-tag",
    "--dyn-structural-tag-scope",
    "--dyn-structural-tag-schema",
    "--custom-jinja-template",
    "--endpoint-types",
    "--router-reset-states",
}

_RESERVED_ENV_KEYS = {
    "CUDA_VISIBLE_DEVICES",
    "DYNAMO_VLLM_PYTHON",
    "ETCD_ENDPOINTS",
    "NATS_SERVER",
    "DYN_DISCOVERY_BACKEND",
    "DYN_EVENT_PLANE",
    "DYN_NAMESPACE",
    "DYN_REQUEST_PLANE",
    "DYN_SYSTEM_PORT",
}


def _normalise_flag(flag: str) -> str:
    flag = flag.split("=", 1)[0].replace("_", "-")
    if flag.startswith("--no-"):
        return "--" + flag[5:]
    return flag


def _flag_for_key(key: str) -> str:
    return "--" + key.replace("_", "-")


def _serialise_value(value: Any) -> str:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, separators=(",", ":"), sort_keys=True)
    return str(value)


class _ArgvBuilder:
    def __init__(self) -> None:
        self.argv: list[str] = []
        self.sources: dict[str, str] = {}

    def add(self, flag: str, value: Any = None, *, source: str) -> None:
        normalised = _normalise_flag(flag)
        prior = self.sources.get(normalised)
        if prior is not None:
            raise ValueError(
                f"Dynamo worker option {normalised} is set by both {prior} and {source}."
            )
        self.sources[normalised] = source
        if isinstance(value, bool):
            self.argv.append(flag if value else f"--no-{flag.removeprefix('--')}")
        else:
            self.argv.append(flag)
            if value is not None:
                self.argv.append(_serialise_value(value))

    def add_raw(self, args: Sequence[str], *, source: str) -> None:
        if not args:
            return
        current_flag: str | None = None
        current_has_value = False
        for token in args:
            if token.startswith("--"):
                current_flag = token
                current_has_value = "=" in token
                normalised = _normalise_flag(token)
                if normalised in _MANAGED_FLAGS:
                    raise ValueError(
                        f"{source} may not override managed option {normalised}."
                    )
                if normalised in _STRUCTURED_DYNAMO_FLAGS:
                    raise ValueError(
                        f"{source} may not set structured Dynamo option {normalised}."
                    )
                if normalised in _STRUCTURED_ENGINE_FLAGS:
                    raise ValueError(
                        f"{source} may not set structured engine option {normalised}."
                    )
                if normalised in self.sources:
                    raise ValueError(
                        f"Dynamo worker option {normalised} is set by both "
                        f"{self.sources[normalised]} and {source}."
                    )
                self.sources[normalised] = source
            elif current_flag is None or current_has_value:
                raise ValueError(
                    f"{source} contains invalid positional argument {token!r}; "
                    "values must immediately follow one --option."
                )
            else:
                current_has_value = True
        self.argv.extend(args)


def build_dynamo_vllm_argv(
    *,
    model_name: str,
    namespace: str,
    seed: int,
    vllm_cfg: Mapping[str, Any],
    vllm_kwargs: Mapping[str, Any],
    dynamo_cfg: DynamoCfg,
) -> list[str]:
    """Build a conflict-free ``python -m dynamo.vllm`` argument list."""
    builder = _ArgvBuilder()
    builder.add("--model", model_name, source="managed runtime")
    builder.add("--served-model-name", model_name, source="managed runtime")
    builder.add("--namespace", namespace, source="managed runtime")
    builder.add("--discovery-backend", "etcd", source="managed runtime")
    builder.add("--request-plane", "tcp", source="managed runtime")
    builder.add("--event-plane", "nats", source="managed runtime")
    builder.add("--enable-rl", source="managed runtime")
    builder.add(
        "--weight-transfer-config",
        {"backend": "nccl"},
        source="managed runtime",
    )
    builder.add("--trust-remote-code", source="managed runtime")
    builder.add("--seed", seed, source="managed runtime")

    for key, flag in _STANDARD_VLLM_FLAGS.items():
        value = vllm_cfg.get(key)
        if value is not None:
            builder.add(flag, value, source=f"vllm_cfg.{key}")

    if int(vllm_cfg.get("expert_parallel_size", 1)) > 1:
        builder.add(
            "--enable-expert-parallel",
            source="vllm_cfg.expert_parallel_size",
        )

    for key, value in vllm_kwargs.items():
        if value is None:
            continue
        flag = _flag_for_key(key)
        normalised = _normalise_flag(flag)
        source = f"vllm_kwargs.{key}"
        if normalised in _MANAGED_FLAGS:
            raise ValueError(f"{source} may not override managed option {normalised}.")
        if normalised in _STRUCTURED_DYNAMO_FLAGS:
            raise ValueError(
                f"{source} may not set structured Dynamo option {normalised}."
            )
        if normalised in _STRUCTURED_ENGINE_FLAGS:
            raise ValueError(
                f"{source} may not set structured engine option {normalised}."
            )
        builder.add(flag, value, source=source)

    worker_args = dynamo_cfg.worker_args
    if worker_args.tool_call_parser is not None:
        builder.add(
            "--dyn-tool-call-parser",
            worker_args.tool_call_parser,
            source="dynamo_cfg.worker_args.tool_call_parser",
        )
    if worker_args.reasoning_parser is not None:
        builder.add(
            "--dyn-reasoning-parser",
            worker_args.reasoning_parser,
            source="dynamo_cfg.worker_args.reasoning_parser",
        )
    builder.add(
        "--exclude-tools-when-tool-choice-none",
        worker_args.exclude_tools_when_tool_choice_none,
        source="dynamo_cfg.worker_args.exclude_tools_when_tool_choice_none",
    )
    builder.add(
        "--dyn-enable-structural-tag",
        worker_args.enable_structural_tag,
        source="dynamo_cfg.worker_args.enable_structural_tag",
    )
    builder.add(
        "--dyn-structural-tag-scope",
        worker_args.structural_tag_scope,
        source="dynamo_cfg.worker_args.structural_tag_scope",
    )
    builder.add(
        "--dyn-structural-tag-schema",
        worker_args.structural_tag_schema,
        source="dynamo_cfg.worker_args.structural_tag_schema",
    )
    if worker_args.custom_jinja_template is not None:
        builder.add(
            "--custom-jinja-template",
            worker_args.custom_jinja_template,
            source="dynamo_cfg.worker_args.custom_jinja_template",
        )
    builder.add(
        "--endpoint-types",
        ",".join(worker_args.endpoint_types),
        source="dynamo_cfg.worker_args.endpoint_types",
    )
    builder.add_raw(
        worker_args.extra_cli_args,
        source="dynamo_cfg.worker_args.extra_cli_args",
    )
    return builder.argv


def build_dynamo_frontend_argv(
    *,
    host: str,
    port: int,
    namespace: str,
    dynamo_cfg: DynamoCfg,
) -> list[str]:
    """Build the managed ``dynamo.frontend`` argument list."""
    builder = _ArgvBuilder()
    builder.add("--http-host", host, source="managed runtime")
    builder.add("--http-port", port, source="managed runtime")
    builder.add("--namespace-prefix", namespace, source="managed runtime")
    builder.add("--discovery-backend", "etcd", source="managed runtime")
    builder.add("--request-plane", "tcp", source="managed runtime")
    builder.add("--event-plane", "nats", source="managed runtime")
    builder.add(
        "--router-mode",
        dynamo_cfg.frontend_args.router_mode,
        source="dynamo_cfg.frontend_args.router_mode",
    )
    builder.add(
        "--router-reset-states",
        dynamo_cfg.frontend_args.router_reset_states,
        source="dynamo_cfg.frontend_args.router_reset_states",
    )
    builder.add_raw(
        dynamo_cfg.frontend_args.extra_cli_args,
        source="dynamo_cfg.frontend_args.extra_cli_args",
    )
    return builder.argv


def validate_managed_vllm_config(vllm_cfg: Mapping[str, Any]) -> None:
    """Validate settings whose meaning differs between NeMo-RL and Dynamo."""
    tp = int(vllm_cfg.get("tensor_parallel_size", 1))
    pp = int(vllm_cfg.get("pipeline_parallel_size", 1))
    ep = int(vllm_cfg.get("expert_parallel_size", 1))
    if tp <= 0 or pp <= 0 or ep <= 0:
        raise ValueError(
            "tensor_parallel_size, pipeline_parallel_size, and "
            "expert_parallel_size must be positive."
        )


def build_managed_worker_env(
    *,
    base_env: Mapping[str, str],
    configured_env: Mapping[str, str],
    manager_env: Mapping[str, str],
) -> dict[str, str]:
    """Return a reproducible worker environment with semantic overrides blocked."""
    forbidden = sorted(
        key
        for key in configured_env
        if key.startswith("DYN_") or key in _RESERVED_ENV_KEYS
    )
    if forbidden:
        raise ValueError(
            "vllm_cfg.env_vars may not override managed Dynamo settings: "
            + ", ".join(forbidden)
        )

    env = {
        key: value
        for key, value in base_env.items()
        if not key.startswith("DYN_") and key not in _RESERVED_ENV_KEYS
    }
    env.update(configured_env)
    env.update(manager_env)
    return env


def redact_argv(argv: Sequence[str]) -> list[str]:
    """Redact values following credential-looking options for safe logging."""
    redacted = list(argv)
    sensitive_fragments = ("token", "password", "secret", "api-key", "apikey")
    for idx, token in enumerate(redacted):
        is_sensitive = token.startswith("--") and any(
            part in token.lower() for part in sensitive_fragments
        )
        if is_sensitive and "=" in token:
            redacted[idx] = token.split("=", 1)[0] + "=<redacted>"
        elif is_sensitive and idx + 1 < len(redacted):
            if not redacted[idx + 1].startswith("--"):
                redacted[idx + 1] = "<redacted>"
    return redacted


def redact_environment(env: Mapping[str, str]) -> dict[str, str]:
    """Redact credential-like environment values before logging."""
    sensitive_fragments = ("token", "password", "secret", "api_key", "apikey")
    return {
        key: "<redacted>"
        if any(part in key.lower() for part in sensitive_fragments)
        else value
        for key, value in env.items()
    }
