# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

from typing import Any, Literal, NotRequired, TypedDict, cast

from pydantic import (
    BaseModel,
    Field,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
    model_validator,
)

from nemo_rl.models.generation.interfaces import GenerationConfig


class SglangQuantizationConfig(TypedDict, total=False):
    """SGLang weight-precision config.

    ``scheme="bf16"`` (or omitting the block) means BF16 rollout/refit. Set
    ``scheme="mxfp8"`` to boot SGLang from an MXFP8 HF checkpoint and to send
    MXFP8 HF tensors during online refit.
    """

    scheme: Literal["bf16", "mxfp8"]
    weight_block_size: list[int]
    scale_fmt: str
    modules_to_not_convert: list[str]
    extra_high_precision_layers_hf: list[str]
    num_layers_at_start_in_bf16: int
    num_layers_at_end_in_bf16: int
    converted_model_path: str
    cache_root: str


class SGLangServerConfig(TypedDict):
    # When True, sets SGLang `enable_memory_saver=True` so weights/KV can be released
    # during training and re-acquired before generation.
    needs_offload: bool
    # When True, keeps a CPU-side copy of weights for the memory saver to restore from.
    # For testing/debug; see SGLang `enable_weights_cpu_backup`.
    cpu_weight_backup: bool
    # Per-engine concurrency cap; multiplied by num_gpus / num_gpus_per_engine to set
    # the global router concurrency limit.
    sglang_server_concurrency: int
    # How to handle in-flight requests on pause: "retract" (preempt) or "kill". Required in YAML.
    pause_generation_mode: str
    # Total number of GPUs allocated to inference across all engines.
    num_gpus: int
    # Optional compatibility hint. The effective mode is derived from
    # colocated.enabled: "ipc" when colocated and "broadcast" otherwise.
    weight_transfer_mode: NotRequired[str]
    # GPUs per SGLang engine
    # num_gpus_per_engine = tp_size * pp_size; set ep, dp-attn are not orthgonal to those
    # nodes_per_engine: max(1, num_gpus_per_engine // num_gpus_per_node)
    # node_rank_0_engine: all_engines[:: nodes_per_engine]
    # num_gpu_per_engine_local = min(num_gpus_per_engine, num_gpus_per_node)
    # num_engines = num_gpus // num_gpu_per_engine_local
    # num_gpus_per_engine = nodes_per_engine * num_gpu_per_engine_local
    num_gpus_per_engine: int


class SGLangRouterConfig(TypedDict):
    # When True, reuse an externally-launched router; ``sglang_router_ip`` and
    # ``sglang_router_port`` must both be set. When False/absent, NeMo-RL spawns
    # and owns a ``RouterActor``.
    use_external_router: NotRequired[bool]
    # External router endpoint; required iff ``use_external_router`` is True.
    sglang_router_ip: NotRequired[str]
    sglang_router_port: NotRequired[int]
    # Router load-balancing policy (e.g. "round_robin", "cache_aware").
    router_policy: NotRequired[str]
    # When True, fan generate requests out across Ray workers instead of a single httpx client.
    use_distributed_post: NotRequired[bool]
    # Per-request timeout (seconds) the router applies before giving up on a backend.
    sglang_router_request_timeout_secs: NotRequired[int]


class SglangSpecificArgs(TypedDict):
    """SGLang-specific configuration arguments.

    Most fields below map directly to SGLang's ServerArgs
    Please Check: https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/server_args.py
    """

    # Nested server/router configs. Kept under ``sglang_cfg`` so YAML and call
    # sites have a single sglang namespace instead of three sibling fields.
    sglang_server_config: SGLangServerConfig

    # Fault tolerance (RolloutHealthMonitor). Off by default; when enabled, a
    # daemon thread health-checks each engine and restarts hung/dead actors.
    use_fault_tolerance: NotRequired[bool]
    rollout_health_check_interval: NotRequired[int]
    rollout_health_check_timeout: NotRequired[int]
    rollout_health_check_first_wait: NotRequired[int]
    # Weight precision and (when scheme=mxfp8) offline-conversion knobs.
    quantization: NotRequired[SglangQuantizationConfig]
    sglang_router_config: SGLangRouterConfig

    # Path to model weights (local folder or HF repo id).
    model_path: NotRequired[str]
    # Random seed for the engine; if None, SGLang picks one.
    random_seed: int
    # Skip tokenizer init; callers must pass `input_ids` in generate requests.
    skip_tokenizer_init: NotRequired[bool]
    # Disable CUDA graphs entirely (use eager execution).
    disable_cuda_graph: NotRequired[bool]
    # Disable RadixAttention prefix caching.
    disable_radix_cache: NotRequired[bool]
    # Skip CUDA graphs only for batches that need padding; use them otherwise.
    disable_cuda_graph_padding: NotRequired[bool]
    # Disable piecewise CUDA graph for extend/prefill.
    # Enabling piecewise CUDA graph (i.e. setting this to False) currently crashes with
    # "illegal memory access", likely due to torch 2.10 + sglang incompatibility.
    # Defaulted to True (disabled) in sglang_worker.py until the upstream sglang fork is updated.
    disable_piecewise_cuda_graph: NotRequired[bool]
    # Enable NCCL NVLS for prefill-heavy requests when available.
    enable_nccl_nvls: NotRequired[bool]
    # Disable on-disk cache for the outlines grammar backend (avoids FS-related crashes).
    disable_outlines_disk_cache: NotRequired[bool]
    # Disable the custom all-reduce kernel and fall back to NCCL.
    disable_custom_all_reduce: NotRequired[bool]
    # Disable the overlap scheduler (CPU scheduler overlapped with GPU worker).
    disable_overlap_schedule: NotRequired[bool]
    # Allow mixing prefill and decode tokens in the same batch under chunked prefill.
    enable_mixed_chunk: NotRequired[bool]
    # Use data parallelism for attention + tensor parallelism for FFN. dp_size must equal tp_size.
    enable_dp_attention: NotRequired[bool]
    # Legacy MoE flags; superseded by `moe_a2a_backend` in newer SGLang. Kept for back-compat.
    enable_deepep_moe: NotRequired[bool]
    enable_ep_moe: NotRequired[bool]
    # Compile the model with torch.compile (experimental).
    enable_torch_compile: NotRequired[bool]
    # Maximum batch size when using torch.compile.
    torch_compile_max_bs: NotRequired[int]
    # Upper bound on CUDA-graph capture batch sizes; None = let SGLang pick.
    cuda_graph_max_bs: NotRequired[int | None]
    # Explicit list of batch sizes to capture CUDA graphs for; None = auto.
    cuda_graph_bs: NotRequired[list[int] | None]
    # torchao quantization config string, e.g. "int8wo", "fp8wo" (experimental).
    torchao_config: NotRequired[str]
    # [Deprecated] Use SGLANG_SPEC_NAN_DETECTION=1 / SGLANG_SPEC_OOB_DETECTION=1 instead.
    enable_nan_detection: NotRequired[bool]
    # Verify GPU P2P access at startup instead of assuming it works.
    enable_p2p_check: NotRequired[bool]
    # Cast intermediate Triton attention results to fp32 to avoid fp16 overflow.
    triton_attention_reduce_in_fp32: NotRequired[bool]
    # Number of KV splits in the Triton flash-decoding kernel; larger helps long-context.
    triton_attention_num_kv_splits: NotRequired[int]
    # Run multiple decode steps per schedule pass; reduces overhead at the cost of TTFT.
    num_continuous_decode_steps: NotRequired[int]
    # Truncate over-length requests instead of erroring.
    allow_auto_truncate: NotRequired[bool]
    # Attention backend to use (e.g. "fa3", "triton", "flashinfer", "trtllm_mha"); None = auto.
    attention_backend: NotRequired[str | None]
    # Enable multimodal serving for the model (no-op if model is text-only).
    enable_multimodal: NotRequired[bool]
    # Sampling kernel backend (e.g. "flashinfer", "pytorch"); None = auto.
    sampling_backend: NotRequired[str | None]
    # Maximum context length; None = take from model config.json.
    context_length: NotRequired[int | None]
    # Fraction of GPU memory used for static allocation (weights + KV pool). Lower if OOM.
    mem_fraction_static: NotRequired[float | None]
    # Cap on concurrently running requests; None = auto.
    max_running_requests: NotRequired[int | None]
    # Token budget per chunked-prefill chunk; -1 disables chunked prefill.
    chunked_prefill_size: NotRequired[int | None]
    # Token budget per prefill batch (max with model context length).
    max_prefill_tokens: NotRequired[int]
    # Request scheduling policy ("fcfs", "lpm", etc.).
    schedule_policy: NotRequired[str]
    # Conservativeness multiplier; raise if requests are being retracted often.
    schedule_conservativeness: NotRequired[float]
    # Reserve this many GB of host RAM for CPU offloading.
    cpu_offload_gb: NotRequired[int]
    # Model weight/activation dtype (e.g. "auto", "bfloat16", "float16").
    dtype: NotRequired[str]
    # KV cache dtype (e.g. "auto", "fp8_e4m3", "fp8_e5m2", "bfloat16").
    kv_cache_dtype: NotRequired[str]
    # Tensor parallelism size. Must satisfy tp_size == num_gpus_per_engine // pp_size.
    # Current pp_size set to 1, therefore, tp_size == num_gpus_per_engine
    tp_size: int
    # Data parallelism size; only used when enable_dp_attention=True.
    dp_size: int
    # Pipeline parallelism size.
    # PP > 1 does not support in current version, therefore, pp_size == 1
    pp_size: int
    # Expert parallelism size (MoE).
    ep_size: int
    # --- LoRA ---
    # Enable LoRA support; auto-set when `lora_paths` is provided.
    enable_lora: NotRequired[bool | None]
    # Cap on LoRA adapter rank; inferred from `lora_paths` if None.
    max_lora_rank: NotRequired[int | None]
    # Union of target module names where LoRA is applied.
    lora_target_modules: NotRequired[list[str] | None]
    # LoRA adapters to load: "<PATH>" | "<NAME>=<PATH>" | JSON {lora_name, lora_path, pinned}.
    lora_paths: NotRequired[list[str] | None]
    # Cap on adapters resident in CPU memory (>= max_loras_per_batch).
    max_loaded_loras: NotRequired[int]
    # Cap on distinct adapters per running batch (includes base-only requests).
    max_loras_per_batch: NotRequired[int]
    # Multi-LoRA kernel backend (e.g. "csgmv").
    lora_backend: NotRequired[str]
    # --- Logging ---
    # Logger level for all SGLang loggers.
    log_level: NotRequired[str]
    # HTTP server log level; reuses `log_level` if None.
    log_level_http: NotRequired[str | None]
    # Log per-request metadata/inputs/outputs (verbosity controlled by `log_requests_level`).
    log_requests: NotRequired[bool]
    # 0=metadata, 1=+sampling params, 2=+partial IO, 3=full IO.
    log_requests_level: NotRequired[int]
    # Print wall-clock time for instrumented internal marks.
    show_time_cost: NotRequired[bool]
    # Export Prometheus-style metrics from the server.
    enable_metrics: NotRequired[bool]
    # Decode-iteration interval at which throughput logs and Prometheus metrics are emitted.
    decode_log_interval: NotRequired[int]
    # --- Extra weight-loader args (passed via `model_loader_extra_config`) ---
    # Use threaded safetensors loading (default True in upstream).
    enable_multithread_load: NotRequired[bool]
    # Use the fast-load path when supported by the loader.
    enable_fast_load: NotRequired[bool]
    # --- Server warmup ---
    # Skip the post-startup warmup pass.
    skip_server_warmup: bool


class SGLangConfig(GenerationConfig):
    """Configuration for SGLang runtime."""

    sglang_cfg: SglangSpecificArgs
    sglang_kwargs: NotRequired[dict[str, Any]]


class SGLangQuantizationRuntimeConfig(BaseModel, extra="allow"):
    """Normalized SGLang rollout/refit precision settings."""

    scheme: Literal["bf16", "mxfp8"] = "bf16"
    weight_block_size: list[int] | None = None
    scale_fmt: str | None = None
    modules_to_not_convert: list[str] = Field(default_factory=list)
    extra_high_precision_layers_hf: list[str] = Field(default_factory=list)
    num_layers_at_start_in_bf16: NonNegativeInt = 0
    num_layers_at_end_in_bf16: NonNegativeInt = 0
    converted_model_path: str | None = None
    cache_root: str | None = None


class SGLangServerRuntimeConfig(BaseModel, extra="allow"):
    """NeMo-RL-owned SGLang server topology and lifecycle settings."""

    needs_offload: bool
    cpu_weight_backup: bool
    sglang_server_concurrency: PositiveInt
    pause_generation_mode: str = Field(min_length=1)
    num_gpus: PositiveInt
    num_gpus_per_engine: PositiveInt
    weight_transfer_mode: Literal["ipc", "broadcast"] | None = None


class SGLangRouterRuntimeConfig(BaseModel, extra="allow"):
    """SGLang router settings normalized before actor startup."""

    use_external_router: bool = False
    sglang_router_ip: str | None = None
    sglang_router_port: PositiveInt | None = None
    router_policy: str | None = None
    use_distributed_post: bool = False
    sglang_router_request_timeout_secs: PositiveInt | None = None

    @model_validator(mode="after")
    def validate_external_router_endpoint(self) -> "SGLangRouterRuntimeConfig":
        if self.use_external_router and (
            self.sglang_router_ip is None or self.sglang_router_port is None
        ):
            raise ValueError(
                "sglang_router_ip and sglang_router_port are required when "
                "use_external_router is true."
            )
        return self


class SGLangRuntimeConfig(BaseModel, extra="allow"):
    """Validated SGLang settings consumed by NeMo-RL.

    Unknown fields are preserved because the remaining keys are passed through
    to SGLang's evolving ``ServerArgs`` API.
    """

    sglang_server_config: SGLangServerRuntimeConfig
    sglang_router_config: SGLangRouterRuntimeConfig = Field(
        default_factory=SGLangRouterRuntimeConfig
    )
    model_path: str = Field(min_length=1)
    random_seed: int
    skip_server_warmup: bool
    tp_size: PositiveInt
    dp_size: PositiveInt
    pp_size: PositiveInt
    ep_size: PositiveInt
    context_length: PositiveInt | None = None
    use_fault_tolerance: bool = False
    rollout_health_check_interval: PositiveFloat | None = None
    rollout_health_check_timeout: PositiveFloat | None = None
    rollout_health_check_first_wait: float | None = Field(default=None, ge=0)
    quantization: SGLangQuantizationRuntimeConfig = Field(
        default_factory=SGLangQuantizationRuntimeConfig
    )

    @model_validator(mode="after")
    def validate_runtime_invariants(self) -> "SGLangRuntimeConfig":
        server = self.sglang_server_config
        if self.pp_size != 1:
            raise ValueError(
                "SGLang pipeline parallelism is not supported; pp_size must be 1."
            )
        expected_gpus_per_engine = self.tp_size * self.pp_size
        if server.num_gpus_per_engine != expected_gpus_per_engine:
            raise ValueError(
                "sglang_server_config.num_gpus_per_engine must equal "
                f"tp_size * pp_size ({expected_gpus_per_engine}), got "
                f"{server.num_gpus_per_engine}."
            )
        if server.num_gpus % server.num_gpus_per_engine != 0:
            raise ValueError(
                "sglang_server_config.num_gpus must be divisible by "
                "sglang_server_config.num_gpus_per_engine."
            )

        if self.use_fault_tolerance:
            missing = [
                field
                for field in (
                    "rollout_health_check_interval",
                    "rollout_health_check_timeout",
                    "rollout_health_check_first_wait",
                )
                if getattr(self, field) is None
            ]
            if missing:
                raise ValueError(
                    "SGLang fault tolerance requires explicit values for "
                    + ", ".join(missing)
                    + "."
                )
        return self


def normalize_sglang_config(
    config: SGLangConfig,
    *,
    colocated_inference: bool | None = None,
) -> SGLangRuntimeConfig:
    """Validate SGLang settings and materialize centralized defaults.

    ``weight_transfer_mode`` is not an independent runtime choice: refit
    dispatch derives IPC versus NCCL broadcast from the placement topology.
    A supplied value is accepted as a compatibility assertion and rejected
    when it conflicts with ``colocated_inference``.

    Args:
        config: Full generation configuration. Its ``sglang_cfg`` block is
            replaced with the normalized, dict-shaped representation.
        colocated_inference: Whether generation shares trainer GPUs. ``None``
            skips transfer-mode resolution for topology-agnostic callers.

    Returns:
        The validated runtime model.
    """
    normalized = SGLangRuntimeConfig.model_validate(config["sglang_cfg"])
    if colocated_inference is not None:
        derived_mode: Literal["ipc", "broadcast"] = (
            "ipc" if colocated_inference else "broadcast"
        )
        configured_mode = normalized.sglang_server_config.weight_transfer_mode
        if configured_mode is not None and configured_mode != derived_mode:
            raise ValueError(
                "sglang_server_config.weight_transfer_mode="
                f"{configured_mode!r} conflicts with colocated.enabled="
                f"{colocated_inference}; the topology requires {derived_mode!r}."
            )
        normalized.sglang_server_config.weight_transfer_mode = derived_mode

    config["sglang_cfg"] = cast(
        SglangSpecificArgs,
        normalized.model_dump(mode="python"),
    )
    return normalized
