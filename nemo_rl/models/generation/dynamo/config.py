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

# Must match vLLM's packed weight-transfer defaults exactly. The Dynamo
# producer and consumer recompute chunk boundaries without negotiation. Verify
# these values whenever Dynamo's pinned vLLM changes.
VLLM_PACKED_BUFFER_SIZE_BYTES = 1024**3
VLLM_PACKED_NUM_BUFFERS = 2

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
    "is_mx",
    "num_first_layers_in_bf16",
    "num_last_layers_in_bf16",
    "skip_tokenizer_init",
    "use_deep_gemm",
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
    router_reset_states: bool
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
        if self.precision.lower() not in {
            "bf16",
            "bfloat16",
        }:
            raise ValueError(
                f"policy.generation.vllm_cfg.precision={self.precision!r} is not "
                "supported by backend='dynamo'; managed weight refit currently "
                "supports BF16 generation only"
            )
        if self.kv_cache_dtype != "auto":
            raise ValueError(
                f"policy.generation.vllm_cfg.kv_cache_dtype={self.kv_cache_dtype!r} "
                "is not supported by backend='dynamo'; use 'auto'"
            )
        extra = self.model_extra or {}
        if extra.get("is_mx"):
            raise ValueError(
                "policy.generation.vllm_cfg.is_mx is not supported by "
                "backend='dynamo'; use backend='vllm' for MXFP8 generation"
            )
        if (
            int(extra.get("num_first_layers_in_bf16") or 0) != 0
            or int(extra.get("num_last_layers_in_bf16") or 0) != 0
        ):
            raise ValueError(
                "mixed BF16/FP8 generation is not supported by backend='dynamo'; "
                "use backend='vllm'"
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
        if extra.get("refit_transport") is not None:
            raise ValueError(
                "policy.generation.refit_transport must be null when "
                "backend='dynamo'; managed Dynamo supports NCCL collective refit only"
            )
        for quantization_field in ("quant_cfg", "real_quant"):
            if extra.get(quantization_field):
                raise ValueError(
                    f"policy.generation.{quantization_field} is not supported "
                    "when backend='dynamo'"
                )
        speculative_config = self.vllm_kwargs.get("speculative_config") or (
            self.vllm_cfg.model_extra or {}
        ).get("speculative_config")
        if speculative_config:
            raise ValueError(
                "policy.generation.vllm_kwargs.speculative_config is not "
                "supported by backend='dynamo' because draft weights are not "
                "refit after step 0"
            )
        if self.vllm_kwargs.get("quantization") is not None:
            raise ValueError(
                "policy.generation.vllm_kwargs.quantization is not supported "
                "when backend='dynamo'"
            )
        vllm_extra = self.vllm_cfg.model_extra or {}
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
        return self
