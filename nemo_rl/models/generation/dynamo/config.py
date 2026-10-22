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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    BeforeValidator,
    Field,
    PositiveFloat,
    PositiveInt,
    model_validator,
)

from nemo_rl.models.generation.interfaces import DisaggConfig

# What one managed engine serves. Under disaggregation every engine is a full
# TP x PP replica of the policy either way; only the request phase differs.
DynamoWorkerRole = Literal["aggregated", "prefill", "decode"]

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
    extra_cli_args: list[str]


class EngineLayout(BaseModel, extra="forbid"):
    """Tensor and pipeline parallel sizes of one prefill or decode worker.

    Both are required, so an entry reads as the worker's whole layout; every
    other parallel setting (expert parallelism, precision) still comes from the
    shared inference layout. The KV handoff reshards the cache between any two
    workers, so every worker may span a different number of GPUs.
    """

    tensor_model_parallel_size: PositiveInt
    pipeline_model_parallel_size: PositiveInt


class DynamoCfg(BaseModel, extra="forbid"):
    """Driver-owned Dynamo service and worker-fleet configuration.

    ``engine`` selects what serves behind the frontend: Dynamo-managed ``vllm``
    workers, or NeMo RL's own ``megatron`` inference engines (prefill/decode
    only; each engine is a dedicated Megatron inference policy, and Megatron's
    Dynamo adapter runs next to it). ``megatron_root`` is the Megatron-LM tree
    that adapter imports; None resolves to the Megatron-LM nested in the
    Megatron-Bridge submodule.
    """

    engine: Literal["vllm", "megatron"]
    megatron_root: str | None = None
    # Per-worker layouts for Megatron engines: entry k is the k-th prefill (or
    # decode) worker of every replica, so a non-empty list has exactly
    # disaggregation.num_prefill_engines (num_decode_engines) entries. Empty
    # keeps the shared layout for every worker of that role. Only Megatron
    # engines read these; vLLM engines reject a non-empty list.
    prefill_engines: list[EngineLayout] = Field(default_factory=list)
    decode_engines: list[EngineLayout] = Field(default_factory=list)
    startup_timeout_s: PositiveFloat
    request_timeout_s: PositiveFloat
    control_timeout_s: PositiveFloat
    worker_args: DynamoWorkerArgs
    frontend_args: DynamoFrontendArgs
    metrics_include_prefixes: list[str] | None
    metrics_exclude_prefixes: list[str] | None


@dataclass(frozen=True)
class DynamoWorkerLayout:
    """One Dynamo worker's role, the GPUs it spans, and its layout overrides.

    ``overrides`` holds the layout keys this worker sets on top of the shared
    inference layout; empty when it keeps the shared one.
    """

    role: DynamoWorkerRole
    width: int
    overrides: Mapping[str, int]


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


def _vllm_config_or_none(value: Any) -> Any:
    """Megatron engines carry no vLLM config; vLLM engines need a nonempty one."""
    if value is None:
        return None
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
        DynamoVllmConfig | None, BeforeValidator(_vllm_config_or_none)
    ] = None
    vllm_kwargs: dict[str, Any]
    disaggregation: DisaggConfig = Field(default_factory=DisaggConfig)

    @property
    def is_megatron_engine(self) -> bool:
        return self.dynamo_cfg.engine == "megatron"

    def layout_overrides(
        self, role: DynamoWorkerRole | None, index: int = 0
    ) -> dict[str, int]:
        """Layout keys one worker sets on top of the shared layout.

        ``dynamo_cfg.prefill_engines[index]`` or ``decode_engines[index]``,
        where ``index`` counts the workers of ``role`` within one replica. Empty
        for aggregated workers, for ``None`` (the shared layout), and for a role
        whose list is empty.
        """
        if role == "prefill":
            layouts = self.dynamo_cfg.prefill_engines
        elif role == "decode":
            layouts = self.dynamo_cfg.decode_engines
        else:
            return {}
        if not layouts:
            return {}
        return layouts[index].model_dump()

    def _has_layout_overrides(self) -> bool:
        return bool(self.dynamo_cfg.prefill_engines or self.dynamo_cfg.decode_engines)

    def engine_world_size(
        self,
        policy_config: Mapping[str, Any] | None = None,
        role: DynamoWorkerRole | None = None,
        index: int = 0,
    ) -> int:
        """Ranks one Dynamo worker spans.

        vLLM workers are TP x PP of ``vllm_cfg``, the same for every worker.
        Megatron workers are dedicated inference policies, so their width is
        whatever layout those policies are built with (``policy.megatron_cfg``
        with the ``mcore_generation_config`` overrides applied, then the
        worker's entry in ``prefill_engines`` / ``decode_engines``), which needs
        ``policy_config``. This is the only place that computes a worker width.

        Args:
            policy_config: The policy config; required for Megatron workers.
            role: Whose layout to size. ``None`` sizes the shared layout.
            index: Which worker of ``role`` within one replica.

        Raises:
            ValueError: For Megatron workers when ``policy_config`` is missing
                or the worker's expert-parallel grid does not fit inside it.
        """
        if not self.is_megatron_engine:
            assert self.vllm_cfg is not None, "validated: vLLM engines carry vllm_cfg"
            return (
                self.vllm_cfg.tensor_parallel_size
                * self.vllm_cfg.pipeline_parallel_size
            )
        if policy_config is None:
            raise ValueError(
                "Megatron engines behind Dynamo need the policy config to size "
                "their engines; pass policy_config"
            )
        # Deferred: the Megatron config module pulls in the policy config
        # types, which this module must stay importable without.
        from nemo_rl.models.generation.megatron.config import (
            merged_inference_megatron_cfg,
        )

        # The same merged layout the worker is built with (its
        # context_parallel_size is pinned to 1 by the merge).
        layout = merged_inference_megatron_cfg(
            policy_config,  # type: ignore[arg-type]
            overrides=self.layout_overrides(role, index),
        )
        tp = layout["tensor_model_parallel_size"]
        pp = layout["pipeline_model_parallel_size"]
        width = int(tp * pp * layout["context_parallel_size"])
        # MCore lays the expert grid (ETP x EP x PP) over the same ranks and
        # requires the world size to be divisible by it
        # (megatron/core/parallel_state.py, initialize_model_parallel). One
        # worker is sized by the dense layout, so an expert grid wider than
        # that cannot be built; refuse it here instead of at model build.
        etp = layout["expert_tensor_parallel_size"]
        ep = layout["expert_model_parallel_size"]
        if width % (etp * ep * pp) != 0:
            who = f"{role} engine {index}" if role else "engines"
            layout_key = (
                f"policy.generation.dynamo_cfg.{role}_engines[{index}] or "
                "policy.generation.mcore_generation_config"
                if role in ("prefill", "decode")
                else "policy.generation.mcore_generation_config"
            )
            raise ValueError(
                f"Megatron {who} behind Dynamo spans {width} ranks "
                f"(tensor_model_parallel_size={tp} x "
                f"pipeline_model_parallel_size={pp}), which is not divisible "
                f"by expert_tensor_parallel_size={etp} x "
                f"expert_model_parallel_size={ep} x "
                f"pipeline_model_parallel_size={pp}. Set {layout_key} so that "
                "tensor_model_parallel_size is a multiple of "
                "expert_tensor_parallel_size x expert_model_parallel_size."
            )
        return width

    def replica_layout(
        self, policy_config: Mapping[str, Any] | None = None
    ) -> list[DynamoWorkerLayout]:
        """The workers of one replica, in launch order, each with its width.

        Prefill workers first, then decode, as TRT-LLM disaggregation lays
        them out; one aggregated worker without disaggregation. What
        :meth:`replica_width` and :meth:`worker_layouts` consume.
        """
        layout = self.disaggregation
        if not layout.enabled:
            slots: list[tuple[DynamoWorkerRole, int]] = [("aggregated", 0)]
        else:
            slots = [("prefill", i) for i in range(layout.num_prefill_engines)] + [
                ("decode", i) for i in range(layout.num_decode_engines)
            ]
        return [
            DynamoWorkerLayout(
                role=role,
                width=self.engine_world_size(policy_config, role, index),
                overrides=self.layout_overrides(role, index),
            )
            for role, index in slots
        ]

    def replica_width(self, replica_layout: Sequence[DynamoWorkerLayout]) -> int:
        """GPUs one rollout replica spans: one worker, or one P/D group.

        The Dynamo frontend routes across the whole fleet, so a replica only
        fixes the P:D ratio and keeps each group's workers next to each other.

        Args:
            replica_layout: One replica's workers, from :meth:`replica_layout`.
        """
        return sum(worker.width for worker in replica_layout)

    def worker_layouts(
        self,
        inference_world_size: int,
        replica_layout: Sequence[DynamoWorkerLayout],
    ) -> list[DynamoWorkerLayout]:
        """Every worker of the fleet, in launch order, for this many GPUs.

        ``replica_layout`` tiled until it covers ``inference_world_size``.
        Launch order is also refit rank order.

        Args:
            inference_world_size: GPUs the whole fleet spans.
            replica_layout: One replica's workers, from :meth:`replica_layout`.
        """
        replica_width = self.replica_width(replica_layout)
        if inference_world_size % replica_width != 0:
            raise ValueError(
                f"Managed Dynamo has {inference_world_size} inference GPUs, which "
                f"is not a multiple of one replica's {replica_width} GPUs "
                "(the sum of the replica's prefill and decode engine widths)."
            )
        return list(replica_layout) * (inference_world_size // replica_width)

    @model_validator(mode="after")
    def _validate_backend_boundary(self) -> "DynamoConfig":
        extra = self.model_extra or {}
        if self.is_megatron_engine:
            self._validate_megatron_engines(extra)
            return self
        if self.vllm_cfg is None:
            raise ValueError(
                "policy.generation.vllm_cfg is required when dynamo_cfg.engine='vllm'"
            )
        if self._has_layout_overrides():
            raise ValueError(
                "policy.generation.dynamo_cfg.prefill_engines/decode_engines set "
                "per-worker layouts, which are implemented for "
                "dynamo_cfg.engine=megatron only; the managed vLLM fleet runs "
                "every worker at vllm_cfg.tensor_parallel_size x "
                "pipeline_parallel_size. Leave both lists empty."
            )
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
        vllm_cfg = self.vllm_cfg
        if (vllm_cfg.model_extra or {}).get("speculative_config"):
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
        vllm_extra = vllm_cfg.model_extra or {}
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
        if self.disaggregation.enabled:
            self._validate_disaggregation()
        return self

    def _validate_megatron_engines(self, extra: dict[str, Any]) -> None:
        """What a prefill/decode fleet of Megatron engines can and cannot carry."""
        if not self.disaggregation.enabled:
            raise ValueError(
                "dynamo_cfg.engine='megatron' requires "
                "policy.generation.disaggregation.enabled=true; aggregated Megatron "
                "engines behind Dynamo are not supported (use backend='megatron')"
            )
        if self.vllm_cfg is not None:
            raise ValueError(
                "policy.generation.vllm_cfg must be null when "
                "dynamo_cfg.engine='megatron'; the engines are Megatron inference "
                "policies and read mcore_generation_config"
            )
        if self.vllm_kwargs:
            raise ValueError(
                "policy.generation.vllm_kwargs must be empty when "
                "dynamo_cfg.engine='megatron'"
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
                "dynamo_cfg.engine='megatron'; the engines refit through the "
                "collective transport"
            )
        for quantization_field in ("quant_cfg", "real_quant"):
            if extra.get(quantization_field):
                raise ValueError(
                    f"policy.generation.{quantization_field} is not supported "
                    "when backend='dynamo'"
                )
        mcore = extra.get("mcore_generation_config")
        if not isinstance(mcore, dict):
            raise ValueError(
                "policy.generation.mcore_generation_config is required when "
                "dynamo_cfg.engine='megatron'"
            )
        if not mcore.get("enable_prefix_caching"):
            raise ValueError(
                "mcore_generation_config.enable_prefix_caching must be true for "
                "Megatron prefill/decode engines: decode admission reuses the "
                "prefix-cache allocator"
            )
        if mcore.get("kv_cache_management_mode") != "persist":
            raise ValueError(
                "mcore_generation_config.kv_cache_management_mode must be "
                "'persist' for Megatron prefill/decode engines; offload and "
                "recompute invalidate the registered KV transfer buffers"
            )
        if mcore.get("expose_http_server"):
            raise ValueError(
                "mcore_generation_config.expose_http_server must be false when "
                "dynamo_cfg.engine='megatron'; the Dynamo frontend is the HTTP "
                "surface"
            )
        self._validate_disaggregation()
        for role, layouts, count in (
            (
                "prefill",
                self.dynamo_cfg.prefill_engines,
                self.disaggregation.num_prefill_engines,
            ),
            (
                "decode",
                self.dynamo_cfg.decode_engines,
                self.disaggregation.num_decode_engines,
            ),
        ):
            if layouts and len(layouts) != count:
                raise ValueError(
                    f"policy.generation.dynamo_cfg.{role}_engines has "
                    f"{len(layouts)} entries but "
                    f"policy.generation.disaggregation.num_{role}_engines is "
                    f"{count}; give one entry per {role} engine of a replica, "
                    "or leave the list empty to use the shared layout"
                )

    def _validate_disaggregation(self) -> None:
        layout = self.disaggregation
        if layout.num_prefill_engines < 1 or layout.num_decode_engines < 1:
            raise ValueError(
                "policy.generation.disaggregation needs at least one prefill and "
                "one decode engine per replica when backend='dynamo'"
            )
        # One managed frontend routes the whole fleet and tokenizes itself.
        if layout.num_frontend_workers != 1 or layout.frontend_tokenize:
            raise ValueError(
                "policy.generation.disaggregation.num_frontend_workers must be 1 "
                "and frontend_tokenize false when backend='dynamo'; the managed "
                "Dynamo frontend routes every replica"
            )
        speculative_config = self.vllm_kwargs.get("speculative_config")
        if (
            speculative_config is not None
            and speculative_config.get("num_speculative_tokens") != 0
        ):
            raise ValueError(
                "policy.generation.vllm_kwargs.speculative_config is not supported "
                "with policy.generation.disaggregation when backend='dynamo'"
            )
