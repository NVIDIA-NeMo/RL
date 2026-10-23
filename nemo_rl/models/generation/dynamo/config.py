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

"""Validated configuration for the managed Dynamo generation backend."""

import warnings
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    BeforeValidator,
    PositiveFloat,
    PositiveInt,
    model_validator,
)

DYNAMO_VLLM_FLAGS: dict[str, str] = {
    "tensor_parallel_size": "--tensor-parallel-size",
    "pipeline_parallel_size": "--pipeline-parallel-size",
    "gpu_memory_utilization": "--gpu-memory-utilization",
    "max_model_len": "--max-model-len",
    "kv_cache_dtype": "--kv-cache-dtype",
    "load_format": "--load-format",
    "precision": "--dtype",
    "enforce_eager": "--enforce-eager",
}

_VLLM_CFG_STRUCTURAL = {
    "env_vars",
    "expert_parallel_size",
}

_VLLM_CFG_MOVED = {
    "http_server_serving_chat_kwargs": (
        "dynamo_cfg.worker_args.custom_jinja_template and tool_call_parser"
    ),
    "reasoning_parser_plugin": "dynamo_cfg.worker_args.reasoning_parser",
    "tool_parser_plugin": "dynamo_cfg.worker_args.tool_call_parser",
}

_VLLM_CFG_UNSUPPORTED = {
    # Applied inside NeMo RL's in-process vLLM worker; it cannot configure the
    # managed ``dynamo.vllm`` subprocess.
    "cap_max_tokens_to_context",
    # Enforced by NeMo RL's own HTTP server; the managed Dynamo frontend checks
    # no bearer token, so a truthy value is rejected below rather than warned.
    "http_server_api_key_required",
    "skip_tokenizer_init",
}

# Read by the vLLM backend's FP8 builder when precision='fp8'; the Dynamo worker
# actor turns its result into engine arguments, and NeMo RL's worker extension
# applies the matching patches inside each engine.
_VLLM_CFG_FP8 = {
    "is_mx",
    "num_first_layers_in_bf16",
    "num_last_layers_in_bf16",
    "pow2_activation_scaling_factors",
    "pow2_weight_scaling_factors",
    "quantization_ignore_patterns",
    "quantization_ignored_layer_kws",
    "refit_prequantize",
    "use_deep_gemm",
}

# Applied by the Dynamo worker actor as a vLLM source patch plus the engine
# environment variable that enables it, as the vLLM backend does.
_VLLM_CFG_ENGINE_PATCHES = {
    "fp32_lm_head",
}

# Read by NeMo RL's worker extension inside each engine at refit time.
_VLLM_CFG_REFIT = {
    "refit_with_reload_api",
}

_VLLM_CFG_MANAGED_RUNTIME = {
    "enable_vllm_metrics_logger",
    "expose_http_server",
    "logprobs_mode",
    "vllm_metrics_logger_interval",
}

_VLLM_CFG_INAPPLICABLE = {
    "async_engine",
    "enable_return_routed_experts",
    "http_refit_api_key_env_var",
    "http_refit_server_port",
    "use_tqdm",
    "zmq_refit_server_port",
}

_VLLM_SINGLE_RANK_ONLY_FIELDS = {
    "data_parallel_size",
    "decode_context_parallel_size",
    "prefill_context_parallel_size",
}

# Top-level ``policy.generation`` keys outside the declared fields that are accounted
# for; any other active key would be silently dropped, so it is warned about.
_GENERATION_CLASSIFIED = {
    # Read by Dynamo generation, or validated in DynamoConfig.
    "model_name",
    "max_new_tokens",
    "temperature",
    "top_p",
    "top_k",
    "stop_token_ids",
    "stop_strings",
    "_pad_token_id",
    "colocated",
    "refit_transport",
    "quant_cfg",
    "real_quant",
    "sglang_cfg",
    "trtllm_cfg",
    # Read by the algorithm, rollouts, or NeMo-Gym rather than the engine.
    "val_temperature",
    "val_top_p",
    "val_top_k",
    "port_range_low",
    "port_range_high",
    "use_async_rollouts",
    "_debug_payload_metrics",
    "_draft_weights_from_refit",
    "_mtp_weights_from_refit",
    "_draft_full_refit",
    # Shared-YAML sections for other backends and refit paths.
    "mcore_generation_config",
    "refit_cfg",
    "real_quant_export_cpu_offload",
    "real_quant_ignore",
}


def _is_active(value: Any) -> bool:
    """Whether a setting is on; a block with an ``enabled`` switch counts only then."""
    if isinstance(value, BaseModel):
        value = value.model_dump()
    if isinstance(value, dict) and "enabled" in value:
        return bool(value["enabled"])
    return bool(value)


class DynamoWorkerArgs(BaseModel, extra="forbid"):
    """Structured arguments passed to every managed ``dynamo.vllm`` worker."""

    tool_call_parser: str | None
    reasoning_parser: str | None
    exclude_tools_when_tool_choice_none: bool
    enable_structural_tag: bool
    structural_tag_scope: Literal["auto", "always"]
    structural_tag_schema: Literal["auto", "strict"]
    custom_jinja_template: str | None
    endpoint_types: list[Literal["chat", "completions"]]
    extra_cli_args: list[str]

    @model_validator(mode="after")
    def _validate_endpoint_types(self) -> "DynamoWorkerArgs":
        if not self.endpoint_types:
            raise ValueError("endpoint_types must contain at least one endpoint")
        if len(self.endpoint_types) != len(set(self.endpoint_types)):
            raise ValueError("endpoint_types must not contain duplicates")
        return self


class DynamoFrontendArgs(BaseModel, extra="forbid"):
    """Structured arguments passed to the managed Dynamo frontend."""

    tokenizer: Literal["default", "fastokens"]
    tokenizer_cache: bool
    tokenizer_cache_bytes: PositiveInt
    router_mode: Literal[
        "round-robin",
        "random",
        "power-of-two",
        "kv",
        "direct",
        "least-loaded",
        "device-aware-weighted",
    ]
    # Router session affinity: idle TTL in seconds for keeping one session on one
    # engine, or None to disable it. Token capture keys sessions by rollout, and the
    # engine splices each turn's staged prefix after routing, so without affinity
    # the KV router loses track of where a rollout's prefix is cached. The binding
    # mode (hard by default) is Dynamo's --router-session-affinity-mode.
    session_affinity_ttl_secs: PositiveInt | None = None
    extra_cli_args: list[str]


class DynamoCfg(BaseModel, extra="forbid"):
    """Driver-owned Dynamo service and worker-fleet configuration."""

    engine: Literal["vllm"]
    startup_timeout_s: PositiveFloat
    request_timeout_s: PositiveFloat
    control_timeout_s: PositiveFloat
    worker_args: DynamoWorkerArgs
    frontend_args: DynamoFrontendArgs
    metrics_include_prefixes: list[str] | None
    metrics_exclude_prefixes: list[str] | None


class DynamoVllmConfig(BaseModel, extra="allow"):
    """Known vLLM settings consumed by ``dynamo.vllm``.

    Additional fields remain visible in ``model_extra`` so argument construction
    can warn rather than silently dropping an inherited vLLM setting.
    """

    async_engine: bool
    tensor_parallel_size: PositiveInt
    pipeline_parallel_size: PositiveInt
    expert_parallel_size: PositiveInt
    gpu_memory_utilization: float
    max_model_len: PositiveInt
    kv_cache_dtype: str
    load_format: str
    precision: str
    enforce_eager: bool
    expose_http_server: bool
    enable_vllm_metrics_logger: bool
    vllm_metrics_logger_interval: PositiveFloat
    env_vars: dict[str, str] | None

    @model_validator(mode="after")
    def _validate_parallelism_and_precision(self) -> "DynamoVllmConfig":
        if self.expert_parallel_size not in (1, self.tensor_parallel_size):
            raise ValueError(
                "backend='dynamo' requires expert_parallel_size to be 1 or "
                "equal tensor_parallel_size"
            )
        if self.precision not in {"bfloat16", "fp8"}:
            raise ValueError(
                f"policy.generation.vllm_cfg.precision={self.precision!r} is not "
                "supported by backend='dynamo'; use 'bfloat16' or 'fp8'"
            )
        # FP8 KV-cache scales are refit only with synchronous rollouts, and managed
        # Dynamo always rolls out asynchronously.
        if self.kv_cache_dtype != "auto":
            raise ValueError(
                f"policy.generation.vllm_cfg.kv_cache_dtype={self.kv_cache_dtype!r} "
                "is not supported by backend='dynamo'; use 'auto'"
            )
        extra = self.model_extra or {}
        if extra.get("http_server_api_key_required"):
            raise ValueError(
                "policy.generation.vllm_cfg.http_server_api_key_required is not "
                "supported by backend='dynamo'; the managed Dynamo frontend checks "
                "no bearer token, so the exposed server would stay open"
            )
        if extra.get("refit_prequantize"):
            raise ValueError(
                "policy.generation.vllm_cfg.refit_prequantize is not supported by "
                "backend='dynamo'; managed Dynamo requests no trainer-side MXFP8 "
                "pre-quantization"
            )
        if self.precision != "fp8":
            active_fp8_fields = sorted(key for key in _VLLM_CFG_FP8 if extra.get(key))
            if active_fp8_fields:
                raise ValueError(
                    f"policy.generation.vllm_cfg.{active_fp8_fields} require "
                    "precision='fp8' when backend='dynamo'"
                )
        logprobs_mode = extra.get("logprobs_mode")
        if logprobs_mode not in (None, "processed_logprobs"):
            raise ValueError(
                "policy.generation.vllm_cfg.logprobs_mode must be "
                "'processed_logprobs' when backend='dynamo'; the managed "
                "--enable-rl option selects processed rollout log probabilities"
            )

        configured_fields = self.model_fields_set | set(extra)
        for key, replacement in _VLLM_CFG_MOVED.items():
            if extra.get(key) is not None:
                raise ValueError(
                    f"policy.generation.vllm_cfg.{key} is not read by the "
                    f"Dynamo backend; set {replacement} instead"
                )
        for key in sorted(_VLLM_CFG_UNSUPPORTED & configured_fields):
            if not extra.get(key):
                continue
            warnings.warn(
                f"policy.generation.vllm_cfg.{key} is ignored by backend='dynamo'",
                stacklevel=2,
            )
        classified = (
            set(DYNAMO_VLLM_FLAGS)
            | _VLLM_CFG_STRUCTURAL
            | set(_VLLM_CFG_MOVED)
            | _VLLM_CFG_UNSUPPORTED
            | _VLLM_CFG_MANAGED_RUNTIME
            | _VLLM_CFG_REFIT
            | _VLLM_CFG_ENGINE_PATCHES
            | _VLLM_CFG_FP8
            | _VLLM_CFG_INAPPLICABLE
            | _VLLM_SINGLE_RANK_ONLY_FIELDS
        )
        unclassified = {
            key for key in configured_fields if getattr(self, key, None) is not None
        } - classified
        if unclassified:
            warnings.warn(
                "vllm_cfg keys ignored by backend='dynamo': "
                f"{sorted(unclassified)}. Add them to DYNAMO_VLLM_FLAGS or an "
                "explicit not-forwarded classification.",
                stacklevel=2,
            )
        return self


def _require_nonempty_vllm_config(value: Any) -> Any:
    if not isinstance(value, dict) or not value:
        raise ValueError(
            "policy.generation.vllm_cfg must be a nonempty mapping when "
            "backend='dynamo'"
        )
    return value


class DynamoConfig(BaseModel, extra="allow"):
    """Validated boundary for ``policy.generation.backend=dynamo``."""

    backend: Literal["dynamo"]
    dynamo_cfg: DynamoCfg
    vllm_cfg: Annotated[
        DynamoVllmConfig, BeforeValidator(_require_nonempty_vllm_config)
    ]
    vllm_kwargs: dict[str, Any]

    @property
    def engine_world_size(self) -> int:
        """Return the derived ranks in each single-node vLLM engine."""
        return self.vllm_cfg.tensor_parallel_size * self.vllm_cfg.pipeline_parallel_size

    @model_validator(mode="after")
    def _validate_backend_boundary(self) -> "DynamoConfig":
        extra = self.model_extra or {}
        # Shared GRPO YAML inheritance supplies mcore_generation_config and
        # refit_cfg. Managed Dynamo uses vLLM arguments and the collective
        # synchronizer instead, so these two sections are intentionally ignored.
        for backend_cfg in ("sglang_cfg", "trtllm_cfg"):
            if extra.get(backend_cfg):
                raise ValueError(
                    f"policy.generation.{backend_cfg} is not valid when "
                    "backend='dynamo'; Dynamo manages vLLM only"
                )
        colocated = extra.get("colocated")
        if isinstance(colocated, dict) and colocated.get("enabled"):
            raise ValueError(
                "policy.generation.colocated.enabled must be false when "
                "backend='dynamo'"
            )
        refit_transport = extra.get("refit_transport")
        if refit_transport not in (None, "nccl_reshard"):
            raise ValueError(
                f"policy.generation.refit_transport={refit_transport!r} is not "
                "supported when backend='dynamo'; use null (NCCL collective refit) "
                "or 'nccl_reshard'"
            )
        for quantization_field in ("quant_cfg", "real_quant"):
            if extra.get(quantization_field):
                raise ValueError(
                    f"policy.generation.{quantization_field} is not supported "
                    "when backend='dynamo'"
                )
        if (self.vllm_cfg.model_extra or {}).get("speculative_config"):
            raise ValueError(
                "policy.generation.vllm_cfg.speculative_config is not read by the "
                "Dynamo backend; set policy.generation.vllm_kwargs.speculative_config"
            )
        if self.vllm_kwargs.get("quantization") is not None:
            raise ValueError(
                "policy.generation.vllm_kwargs.quantization is not supported "
                "when backend='dynamo'"
            )
        if self.vllm_kwargs.get("worker_extension_cls") is not None:
            raise ValueError(
                "policy.generation.vllm_kwargs.worker_extension_cls is not supported "
                "when backend='dynamo'; managed engines always load NeMo RL's worker "
                "extension, which every refit relies on"
            )
        vllm_extra = self.vllm_cfg.model_extra or {}
        if vllm_extra.get("reset_encoder_cache_after_weight_update"):
            raise ValueError(
                "policy.generation.vllm_cfg.reset_encoder_cache_after_weight_update "
                "is not supported when backend='dynamo'; Dynamo exposes no route to "
                "reset the multimodal encoder cache after a refit"
            )
        for field in sorted(_VLLM_SINGLE_RANK_ONLY_FIELDS):
            for source, value in (
                ("vllm_cfg", vllm_extra.get(field)),
                ("vllm_kwargs", self.vllm_kwargs.get(field)),
            ):
                if value is not None and int(value) != 1:
                    raise ValueError(
                        f"policy.generation.{source}.{field} must be 1 when "
                        "backend='dynamo'; managed refit rank geometry is TP × PP"
                    )
        stop_strings = extra.get("stop_strings")
        if stop_strings is not None and len(stop_strings) > 32:
            raise ValueError(
                "policy.generation.stop_strings supports at most 32 values when "
                "backend='dynamo'"
            )
        ignored = sorted(
            key
            for key, value in extra.items()
            if key not in _GENERATION_CLASSIFIED and _is_active(value)
        )
        if ignored:
            warnings.warn(
                f"policy.generation keys ignored by backend='dynamo': {ignored}",
                stacklevel=2,
            )
        return self
