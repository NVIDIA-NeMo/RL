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

from typing import Any, Literal, NotRequired, TypedDict

from pydantic import (
    BaseModel,
    NonNegativeFloat,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
)

from nemo_rl.models.generation import disagg_layout
from nemo_rl.models.generation.disagg_layout import Role
from nemo_rl.models.generation.interfaces import DisaggConfig, GenerationConfig


class SglangQuantizationConfig(TypedDict):
    """SGLang weight precision. Only BF16 on this layer."""

    scheme: Literal["bf16"]


SUPPORTED_SGLANG_QUANTIZATION_SCHEMES = frozenset({"bf16"})


def get_sglang_quantization_scheme(quantization_config: dict[str, Any]) -> str:
    """Return the configured SGLang weight precision.

    The block must declare a supported ``scheme`` so omissions and typos fail.
    """
    scheme = quantization_config["scheme"]
    if scheme not in SUPPORTED_SGLANG_QUANTIZATION_SCHEMES:
        supported = ", ".join(sorted(SUPPORTED_SGLANG_QUANTIZATION_SCHEMES))
        raise ValueError(
            f"SGLang quantization.scheme must be one of {{{supported}}}, got {scheme!r}."
        )
    return scheme


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
    # How to handle in-flight requests on pause, as accepted by SGLang's
    # /pause_generation: "abort" (drop them), "retract" (requeue, KV cache may
    # be flushed) or "in_place" (keep them and their KV cache). Weight refit
    # rejects in_place because cached entries would outlive their source weights.
    pause_generation_mode: Literal["abort", "retract", "in_place"]
    # Total number of GPUs allocated to inference across all engines.
    num_gpus: NotRequired[int]
    # GPUs per SGLang engine
    # num_gpus_per_engine = tp_size * pp_size; set ep, dp-attn are not orthgonal to those
    # nodes_per_engine: max(1, num_gpus_per_engine // num_gpus_per_node)
    # node_rank_0_engine: all_engines[:: nodes_per_engine]
    # num_gpu_per_engine_local = min(num_gpus_per_engine, num_gpus_per_node)
    # num_engines = num_gpus // num_gpu_per_engine_local
    # num_gpus_per_engine = nodes_per_engine * num_gpu_per_engine_local
    num_gpus_per_engine: NotRequired[int]


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
    # Managed-router total attempts per request (including the first); positive.
    # Omitted values retain the pinned router's default of 5.
    retry_max_retries: NotRequired[int]
    # Consecutive worker failures before its circuit opens; positive.
    # Omitted values retain the pinned router's default of 10.
    cb_failure_threshold: NotRequired[int]


class SGLangOutlinesCachePolicy(BaseModel, extra="allow", strict=True):
    """Disable the legacy disk cache unless trusted-cache use is requested."""

    disable_outlines_disk_cache: bool = True


class SGLangHttpClientConfig(BaseModel, extra="allow", strict=True):
    """NeMo-RL HTTP settings, including when the router is externally managed.

    ``max_retries`` counts total POST attempts, including the first attempt.
    Distributed dispatch and its local fallback each use this same budget.
    """

    max_retries: PositiveInt = 3


class SGLangFaultToleranceConfig(
    BaseModel, extra="allow", strict=True, allow_inf_nan=False
):
    """Serving-health and refit-time recovery settings for SGLang engines.

    Durations are seconds. The first-wait grace may be zero; probe intervals
    and timeouts must be positive. The restart budget applies per logical
    engine over the generation object's lifetime; zero disables restarts.
    """

    use_fault_tolerance: bool = False
    rollout_health_check_interval: PositiveFloat = 60.0
    rollout_health_check_timeout: PositiveFloat = 60.0
    rollout_health_check_first_wait: NonNegativeFloat = 60.0
    rollout_max_restart_attempts: NonNegativeInt = 3


class SGLangRoleEngineArgs(TypedDict, total=False):
    """Keys a ``sglang_cfg.{prefill,decode}_engine`` block may set.

    Only the knobs that may legitimately differ between a prefill engine and a
    decode engine of the same replica: the engine's width (``tp_size`` with
    ``ep_size``), its memory and scheduling budgets, and its CUDA-graph
    capture. The model, dtype, context length and the server/router/fault
    tolerance blocks are shared by the whole replica and are rejected here.
    """

    tp_size: int
    ep_size: int
    mem_fraction_static: float | None
    max_running_requests: int | None
    chunked_prefill_size: int | None
    max_prefill_tokens: int
    schedule_policy: str
    schedule_conservativeness: float
    num_continuous_decode_steps: int
    disable_cuda_graph: bool
    disable_cuda_graph_padding: bool
    cuda_graph_backend_decode: str
    cuda_graph_backend_prefill: str
    cuda_graph_max_bs_decode: int | None
    cuda_graph_max_bs_prefill: int | None
    cuda_graph_bs_decode: list[int] | None
    cuda_graph_bs_prefill: list[int] | None


SGLangRouterPolicy = Literal["random", "round_robin", "cache_aware", "power_of_two"]


class SGLangDisaggConfig(BaseModel, extra="forbid", strict=True):
    """How SGLang implements ``generation.disaggregation``.

    The sglang router's PD mode is the disagg server: it pairs every request
    with one prefill and one decode engine and relays the decode output. These
    are the knobs of that implementation; the layout itself (engines per
    replica) is the backend-agnostic ``generation.disaggregation`` block.
    """

    # "replica": one router per replica, so a request's prefill and decode
    # engines always sit in the same replica and the KV cache never leaves
    # the replica's placement domain; NeMo RL spreads prompt groups across the
    # routers. "global": one router pools every engine of the fleet and pairs
    # freely, which balances decode load fleet-wide at the price of
    # cross-replica (often cross-node) KV transfers.
    router_scope: Literal["replica", "global"] = "replica"
    # SGLang's prefill->decode KV transport (``--disaggregation-transfer-backend``).
    # mooncake is SGLang's own default and does not go through UCX; nixl is the
    # transport the vLLM path uses. fake moves no KV and only exercises the
    # control plane.
    transfer_backend: Literal["mooncake", "nixl", "mooncake_tcp", "fake"] = "mooncake"
    # ``--disaggregation-ib-device``: comma-separated IB devices for the
    # transport; None lets the transport pick.
    ib_device: str | None = None
    # Per-role router policies (``--prefill-policy`` / ``--decode-policy``).
    # cache_aware keeps the samples of one prompt on the prefill engine that
    # holds their prefix. power_of_two is refused: sglang-router 0.3.2 checks
    # it against the workers configured at startup, and NeMo RL registers its
    # engines after the router is up.
    prefill_policy: SGLangRouterPolicy = "cache_aware"
    decode_policy: SGLangRouterPolicy = "round_robin"
    # ``--disaggregation-decode-enable-radix-cache``: SGLang marks it
    # experimental; off, decode engines use the chunk cache.
    decode_enable_radix_cache: bool = False


class SglangSpecificArgs(TypedDict):
    """SGLang-specific configuration arguments.

    Most fields below map directly to SGLang's ServerArgs
    Please Check: https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/server_args.py
    """

    # Nested server/router configs. Kept under ``sglang_cfg`` so YAML and call
    # sites have a single sglang namespace instead of three sibling fields.
    sglang_server_config: SGLangServerConfig
    sglang_router_config: SGLangRouterConfig
    sglang_http_client_config: NotRequired[SGLangHttpClientConfig]
    sglang_fault_tolerance_config: SGLangFaultToleranceConfig

    # Prefill/decode disaggregation (generation.disaggregation.enabled); read
    # only then, inert otherwise. A role block names only what differs from the
    # shared values in this block for that role's engines, limited to
    # SGLangRoleEngineArgs; resolve_sglang_role_overrides replaces each with
    # the role's complete engine config, which is all a worker reads.
    prefill_engine: NotRequired[SGLangRoleEngineArgs]
    decode_engine: NotRequired[SGLangRoleEngineArgs]
    # How SGLang implements the layout: KV transport and router policies.
    disagg: NotRequired[SGLangDisaggConfig]

    # Weight precision for rollout/refit.
    quantization: SglangQuantizationConfig

    # Path to model weights (local folder or HF repo id).
    model_path: NotRequired[str]
    # Random seed for the engine; if None, SGLang picks one.
    random_seed: NotRequired[int]
    # Skip tokenizer init; callers must pass `input_ids` in generate requests.
    skip_tokenizer_init: NotRequired[bool]
    # Disable CUDA graphs entirely (use eager execution).
    disable_cuda_graph: NotRequired[bool]
    # Disable RadixAttention prefix caching.
    disable_radix_cache: NotRequired[bool]
    # Skip CUDA graphs only for batches that need padding; use them otherwise.
    disable_cuda_graph_padding: NotRequired[bool]
    # Per-phase CUDA graph backend. Prefill capture is off entirely: "tc_piecewise"
    # crashed with "illegal memory access" on torch 2.10, and "breakable" (upstream's
    # CUDA default) is rejected outright under memory saver, which
    # sglang_server_config.needs_offload turns on. Replaces the pre-v0.5.16
    # `disable_piecewise_cuda_graph` boolean.
    cuda_graph_backend_decode: NotRequired[str]
    cuda_graph_backend_prefill: NotRequired[str]
    # Enable NCCL NVLS for prefill-heavy requests when available.
    enable_nccl_nvls: NotRequired[bool]
    # Defaults to True. Set False to opt into the legacy pickle-based Outlines
    # disk cache only when its files are controlled by trusted writers.
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
    # Split per phase in v0.5.16 (was `cuda_graph_max_bs`).
    cuda_graph_max_bs_decode: NotRequired[int | None]
    cuda_graph_max_bs_prefill: NotRequired[int | None]
    # Explicit list of batch sizes to capture CUDA graphs for; None = auto.
    # Split per phase in v0.5.16 (was `cuda_graph_bs`).
    cuda_graph_bs_decode: NotRequired[list[int] | None]
    cuda_graph_bs_prefill: NotRequired[list[int] | None]
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
    tp_size: NotRequired[int]
    # Data parallelism size; only used when enable_dp_attention=True.
    dp_size: NotRequired[int]
    # Pipeline parallelism size.
    # PP > 1 does not support in current version, therefore, pp_size == 1
    pp_size: NotRequired[int]
    # Expert parallelism size (MoE).
    ep_size: NotRequired[int]
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
    skip_server_warmup: NotRequired[bool]


class SGLangConfig(GenerationConfig):
    """Configuration for SGLang runtime."""

    sglang_cfg: SglangSpecificArgs
    sglang_kwargs: NotRequired[dict[str, Any]]


# ---------------------------------------------------------------------------
# Prefill/decode disaggregation
# ---------------------------------------------------------------------------
# The role-override machinery is shared with the TRT-LLM and vLLM backends
# (nemo_rl/models/generation/disagg_layout.py); these bind it to the SGLang
# block names.
SGLANG_ROLE_ENGINE_ARG_KEYS = frozenset(
    SGLangRoleEngineArgs.__required_keys__ | SGLangRoleEngineArgs.__optional_keys__
)
SGLANG_NON_ENGINE_KEYS = frozenset({"prefill_engine", "decode_engine", "disagg"})
# ``sglang_kwargs`` is the shared helper's kwargs channel. The SGLang worker
# builds ServerArgs from the typed keys only, so the merged block is inert; a
# role block may not set it (it is outside SGLANG_ROLE_ENGINE_ARG_KEYS).
_SGLANG_KWARGS_KEY = "sglang_kwargs"
# Under disaggregation the sglang-miles scheduler refuses ``retract`` ("not
# supported in PD disaggregation mode yet") and NeMo RL already refuses
# ``in_place`` for refit, which leaves ``abort``. SGLang refits run between
# steps on idle engines, so abort has nothing to drop.
SGLANG_DISAGG_PAUSE_MODE = "abort"


def resolve_sglang_disagg_layout(config: SGLangConfig) -> DisaggConfig:
    """``generation.disaggregation`` as its schema; see :func:`disagg_layout.resolve_disagg_layout`."""
    return disagg_layout.resolve_disagg_layout(config)


def resolve_sglang_disagg_config(config: SGLangConfig) -> SGLangDisaggConfig:
    """``sglang_cfg.disagg`` as its schema, defaults filled in; absent is all-defaults."""
    raw = config["sglang_cfg"].get("disagg")
    if isinstance(raw, SGLangDisaggConfig):
        return raw
    return SGLangDisaggConfig.model_validate(disagg_layout.as_dict(raw))


def resolve_sglang_role_overrides(config: SGLangConfig) -> SGLangConfig:
    """Resolve ``sglang_cfg.{prefill,decode}_engine`` into complete engine configs.

    See :func:`disagg_layout.resolve_role_overrides` for the rule. A role may
    only name the :class:`SGLangRoleEngineArgs` keys. Returns ``config`` itself
    without disaggregation.
    """
    return disagg_layout.resolve_role_overrides(
        config,
        backend_cfg_key="sglang_cfg",
        allowed_keys=SGLANG_ROLE_ENGINE_ARG_KEYS,
        excluded_keys=SGLANG_NON_ENGINE_KEYS,
        kwargs_key=_SGLANG_KWARGS_KEY,
    )


def sglang_role_engine_config(config: SGLangConfig, role: Role) -> dict[str, Any]:
    """The whole ``sglang_cfg`` block of one engine of ``role``; ``config`` must be role-resolved."""
    return disagg_layout.role_engine_config(config, role, backend_cfg_key="sglang_cfg")


def sglang_engine_width(engine_cfg: dict[str, Any]) -> int:
    """GPUs one SGLang engine spans: ``tp_size * pp_size`` (DP-attention and EP subdivide them)."""
    return int(engine_cfg["tp_size"]) * int(engine_cfg["pp_size"])


def sglang_replica_width(config: SGLangConfig) -> int:
    """GPUs one rollout replica spans.

    Without disaggregation a replica is one engine of
    ``sglang_server_config.num_gpus_per_engine`` GPUs. Under it, a replica is
    its prefill engines plus its decode engines, the unit to keep inside one
    NVLink domain because the KV cache crosses from prefill to decode on every
    request. Accepts a raw or a role-resolved config.
    """
    layout = resolve_sglang_disagg_layout(config)
    sglang_cfg = config["sglang_cfg"]
    if not layout.enabled:
        return int(sglang_cfg["sglang_server_config"]["num_gpus_per_engine"])
    resolved = resolve_sglang_role_overrides(config)
    return layout.num_prefill_engines * sglang_engine_width(
        sglang_role_engine_config(resolved, "prefill")
    ) + layout.num_decode_engines * sglang_engine_width(
        sglang_role_engine_config(resolved, "decode")
    )


def sglang_disagg_server_kwargs(
    role: Role, pd_cfg: SGLangDisaggConfig, bootstrap_port: int | None
) -> dict[str, Any]:
    """The ``ServerArgs`` keys that make an engine a prefill or a decode engine.

    The bootstrap server (the prefill side of the KV handshake) binds
    ``bootstrap_port`` on the prefill engine's host, so every prefill engine
    needs its own; a decode engine learns its partner's port from each
    request, which the router stamps on.
    """
    kwargs: dict[str, Any] = {
        "disaggregation_mode": role,
        "disaggregation_transfer_backend": pd_cfg.transfer_backend,
    }
    if pd_cfg.ib_device is not None:
        kwargs["disaggregation_ib_device"] = pd_cfg.ib_device
    if role == "prefill":
        assert bootstrap_port is not None, "a prefill engine needs a bootstrap port"
        kwargs["disaggregation_bootstrap_port"] = int(bootstrap_port)
    else:
        kwargs["disaggregation_decode_enable_radix_cache"] = (
            pd_cfg.decode_enable_radix_cache
        )
    return kwargs


def validate_sglang_disaggregation(config: SGLangConfig, *, use_nemo_gym: bool) -> None:
    """Refuse a disaggregated SGLang config that cannot run as written.

    Silent without disaggregation. With it, every check here would otherwise
    surface only after the fleet is up: SGLang's own ``retract`` refusal, the
    router's startup validation, or a rollout path that never reaches the
    router.
    """
    layout = resolve_sglang_disagg_layout(config)
    if not layout.enabled:
        return
    sglang_cfg = config["sglang_cfg"]
    pd_cfg = resolve_sglang_disagg_config(config)
    # Resolving also rejects a role block that names a shared key.
    resolve_sglang_role_overrides(config)

    if use_nemo_gym:
        raise ValueError(
            "policy.generation.disaggregation with backend='sglang' runs on "
            "SGLang's native /generate path, which the NeMo-Gym rollout does not "
            "use; set env.should_use_nemo_gym=false."
        )
    if config["colocated"]["enabled"]:
        raise ValueError(
            "policy.generation.disaggregation with backend='sglang' requires "
            "colocated.enabled=false: colocated prefill/decode engines are not "
            "supported yet."
        )
    defaults = DisaggConfig()
    frontend_keys = ("num_frontend_workers", "frontend_tokenize", "frontend_base_port")
    changed = [
        key for key in frontend_keys if getattr(layout, key) != getattr(defaults, key)
    ]
    if changed:
        raise ValueError(
            f"policy.generation.disaggregation.{changed} only apply to the TRT-LLM "
            "and vLLM frontends; the sglang router is the frontend of the SGLang "
            "path and takes no frontend settings. Leave them at their defaults."
        )
    pause_mode = sglang_cfg["sglang_server_config"]["pause_generation_mode"]
    if pause_mode != SGLANG_DISAGG_PAUSE_MODE:
        raise ValueError(
            "sglang_cfg.sglang_server_config.pause_generation_mode must be "
            f"'{SGLANG_DISAGG_PAUSE_MODE}' under disaggregation, got {pause_mode!r}: "
            "SGLang refuses 'retract' in PD mode and NeMo RL refuses 'in_place' "
            "for refit."
        )
    if sglang_cfg["sglang_router_config"].get("use_external_router"):
        raise ValueError(
            "sglang_cfg.sglang_router_config.use_external_router is not supported "
            "under disaggregation: NeMo RL starts the PD routers itself, one per "
            "replica or one for the fleet (sglang_cfg.disagg.router_scope)."
        )
    if int(sglang_cfg["pp_size"]) != 1:
        raise ValueError(
            "sglang_cfg.pp_size must be 1 under disaggregation; SGLang's PD path "
            "runs tensor-parallel engines only."
        )
    for role in ("prefill", "decode"):
        policy = getattr(pd_cfg, f"{role}_policy")
        if policy == "power_of_two":
            raise ValueError(
                f"sglang_cfg.disagg.{role}_policy='power_of_two' cannot start: "
                "sglang-router 0.3.2 validates it against the workers configured "
                "at startup (at least two of the role), and NeMo RL registers its "
                "engines after the router is up. Use round_robin or cache_aware."
            )
