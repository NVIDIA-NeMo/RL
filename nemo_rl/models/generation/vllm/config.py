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

import warnings
from collections.abc import Mapping
from typing import Annotated, Any, Literal, NotRequired, TypedDict, cast, get_args

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
)

from nemo_rl.models.generation import disagg_layout
from nemo_rl.models.generation.interfaces import (
    DisaggConfig,
    GenerationConfig,
    get_num_routed_experts,
)
from nemo_rl.models.generation.vllm.kv_connector_profiles import (
    ConnectorProfile,
    connector_profile,
)
from nemo_rl.models.generation.vllm.quantization.nvfp4_pertoken_config import (
    MCORE_DEFAULT_NUM_LAYERS_AT_END_IN_BF16,
    MCORE_DEFAULT_NUM_LAYERS_AT_START_IN_BF16,
    NvFp4PerTokenRolloutConfig,
    resolve_boundary_ignore_patterns,
)

VllmRefitTransportName = Literal["s3", "zmq"]
VllmRefitSelector = Literal["vllm_s3_sparse", "vllm_zmq_sparse", "nixl", "nccl_reshard"]
VLLM_SPARSE_REFIT_TRANSPORTS = frozenset({"vllm_s3_sparse", "vllm_zmq_sparse"})
REFITTABLE_FP8_KV_CACHE_DTYPES = frozenset({"fp8", "fp8_e4m3"})


# TODO(rohitrango): Move model-specific video fields behind ProcessorInterface.
class VllmVideoConfig(BaseModel):
    """Video sampling contract shared by policy preprocessing and vLLM."""

    model_config = ConfigDict(extra="forbid")

    sampling_style: Literal["nemotron_vl"]
    num_frames: PositiveInt
    temporal_patch_size: PositiveInt


class VllmEngineArgs(BaseModel, extra="forbid"):
    """One role's engine overrides: ``vllm_cfg.{prefill,decode}_engine``.

    Deliberately narrow: everything else in ``vllm_cfg`` describes the model or
    the engine's contract with the rest of the system, and two engines of one
    replica differing on those would not serve one coherent policy. All
    optional -- a role block names only what differs. ``vllm_kwargs`` is
    deep-merged over ``generation.vllm_kwargs``, which is how prefill and
    decode get different engine tuning (``max_num_seqs``,
    ``max_num_batched_tokens``, ``enable_chunked_prefill``, ...).
    """

    tensor_parallel_size: int | None = None
    gpu_memory_utilization: float | None = None
    vllm_kwargs: dict[str, Any] | None = None


VLLM_ENGINE_ARG_KEYS = frozenset(VllmEngineArgs.model_fields)
# Blocks of vllm_cfg that are *about* the replica's engines, never settings of
# one engine, so they never flow into a resolved role config.
VLLM_NON_ENGINE_KEYS = frozenset(
    {"prefill_engine", "decode_engine", "disagg_frontend", "kv_transfer"}
)


class VllmKvConnectorExtraConfig(BaseModel, extra="allow"):
    """``kv_connector_extra_config`` with the NixlConnector keys typed.

    Passed through to vLLM, so unknown keys are allowed (``model_extra``) for
    other connectors; the typed ones get centralized documentation. Only the
    keys a recipe sets are forwarded (``exclude_none``), so vLLM's own defaults
    stay in force for the rest.
    """

    # NIXL transport plugins, e.g. ["UCX"] or ["UCX", "GDS"].
    backends: list[str] | None = None
    # Seconds a prefill engine keeps a finished request's blocks for the decode
    # engine to read; extended by heartbeats while the request queues there.
    kv_lease_duration: int | None = None
    # bidirectional_kv_xfer only: minimum tokens worth pulling back from the
    # decode engine instead of recomputing them on the prefill engine.
    kv_recompute_threshold: int | None = None
    # bidirectional_kv_xfer only: seconds the decode engine pins a finished
    # conversation's blocks for the next turn's pull. Not renewed.
    decoder_kv_blocks_ttl: int | None = None
    # Disable the P/D compatibility-hash check at handshake. Keep on.
    enforce_handshake_compat: bool | None = None
    # Written by NeMo RL from disagg_frontend.bidirectional_kv_xfer; a recipe
    # setting it directly is rejected so the two cannot disagree.
    bidirectional_kv_xfer: bool | None = None


class VllmKvTransferConfig(BaseModel, extra="forbid"):
    """``vllm_cfg.kv_transfer``: the ``KVTransferConfig`` every engine is built with.

    ``kv_role`` is not a field: it follows from the engine's role (prefill ->
    producer, decode -> consumer) and the connector profile. ``extra="forbid"``
    because any other ``KVTransferConfig`` key would pass validation and then
    silently do nothing.
    """

    kv_connector: str = "NixlConnector"
    # Module path for a connector that is not built into vLLM. Such a connector
    # also names the handshake protocol it speaks via
    # disagg_frontend.connector_profile.
    kv_connector_module_path: str | None = None
    kv_connector_extra_config: VllmKvConnectorExtraConfig = Field(
        default_factory=VllmKvConnectorExtraConfig
    )
    # "fail" ends a request whose KV load failed; "recompute" prefills it on the
    # decode engine instead, which is what disaggregation exists to avoid.
    kv_load_failure_policy: Literal["fail", "recompute"] = "fail"
    # Where the connector stages KV. "cuda" is direct GPU-to-GPU; "cpu" is for
    # platforms without device buffers.
    kv_buffer_device: str = "cuda"


class VllmDisaggFrontendConfig(BaseModel, extra="forbid"):
    """``vllm_cfg.disagg_frontend``: what the replica's frontends do.

    Not a :class:`DisaggConfig` subclass: the layout is a different thing, and
    inheriting would let this block declare layout keys nothing here reads.
    """

    # Which prefill engine serves a turn. conversation keeps a trajectory on
    # the engine holding its prefix; kv_cache_aware routes by the block hashes
    # the engines publish (needs disaggregation.frontend_tokenize, since the
    # frontend must hash the prompt's blocks itself); round_robin is stateless.
    prefill_router: Literal["conversation", "round_robin", "kv_cache_aware"] = (
        "conversation"
    )
    # Which decode engine serves a turn. A decode engine receives KV freshly on
    # every turn, so the stateless choices are the default; conversation keeps
    # the turns of a trajectory on one decode engine so its own blocks from
    # the previous turn are skipped on transfer.
    decode_router: Literal["load_balancing", "round_robin", "conversation"] = (
        "load_balancing"
    )
    # Let the prefill engine pull the previous turn's KV (the tokens the decode
    # engine generated) back from the decode engine instead of recomputing it.
    # The frontend then keeps per-conversation state, and the decode engine pins
    # a finished conversation's blocks until the next turn or the TTL. Off by
    # default: that pinning competes with new requests for the decode engine's
    # KV pool, and the trade-off depends on the workload.
    bidirectional_kv_xfer: bool = False
    # bidirectional_kv_xfer: seconds the frontend remembers a conversation's
    # decode-side blocks. Keep below the decode engine's decoder_kv_blocks_ttl
    # so the frontend never offers blocks the engine already freed.
    conversation_cache_ttl_s: PositiveFloat = 450.0
    # Relay prefill->decode prompt token ids as one base64 int32 string instead
    # of a JSON int array (the TRT-LLM gen_tokids_ctxbytes counterpart).
    decode_tokids_b64: bool = False
    # Send only the last message on the decode leg; the relayed token ids carry
    # the whole prefix (the TRT-LLM gen_strip_message_history counterpart).
    decode_strip_message_history: bool = False
    # Per-leg HTTP timeout. A turn can prefill tens of thousands of tokens and
    # decode for minutes behind other requests.
    request_timeout_s: PositiveInt = 1800
    # Conversations the conversation router remembers per frontend (LRU).
    prefill_router_table_size: PositiveInt = 100_000
    # Handshake protocol of a custom connector; built-in connectors need none.
    connector_profile: str | None = None


class VllmSpecificArgs(TypedDict):
    tensor_parallel_size: int
    pipeline_parallel_size: int
    expert_parallel_size: int
    gpu_memory_utilization: float
    max_model_len: int
    # Additional arguments for vLLM inserted by nemo rl based on the context of when vllm is used
    skip_tokenizer_init: bool
    async_engine: bool
    # Optional video contract. When present, NeMo RL registers its TorchCodec
    # loader and uses these exact sampling values on both sides of GRPO.
    video: NotRequired[VllmVideoConfig]
    load_format: NotRequired[str]
    precision: NotRequired[str]
    # Whether vLLM returns logprobs before or after generation-time logit
    # processors. RL policy recomputation uses raw model logits, so recipes
    # with generation-time processors should request ``raw_logprobs`` when
    # comparing generation and policy logprobs.
    logprobs_mode: NotRequired[Literal["processed_logprobs", "raw_logprobs"]]
    # Nemotron-H only: compute vLLM Nemotron-H logits with an fp32 LM head.
    # Pair this with policy.megatron_cfg.fp32_lm_head when using a Megatron
    # trainer.
    fp32_lm_head: NotRequired[bool]
    # Cap each request's generated tokens so the training prompt plus response
    # fits within max_model_len. This is needed when multimodal processing makes
    # the training prompt longer than its text-only representation.
    cap_max_tokens_to_context: NotRequired[bool]
    # Use ModelOpt MXFP8 quantization when precision is fp8.
    is_mx: NotRequired[bool]
    # With is_mx, quantize weights to MXFP8 on the trainer during refit and
    # stream E4M3 data plus scales (~47% smaller payload) instead of BF16;
    # the vLLM worker then skips its per-refit re-quantization. Requires the
    # Megatron policy backend.
    refit_prequantize: NotRequired[bool]
    # Deprecated in 0.8. Use quantization_ignore_patterns instead.
    quantization_ignored_layer_kws: NotRequired[list[str]]
    # MXFP8 exclusion patterns forwarded through vLLM's quantization config.
    # Supports exact names, substrings, and fnmatch wildcards.
    quantization_ignore_patterns: NotRequired[list[str]]
    kv_cache_dtype: Literal["auto", "fp8", "fp8_e4m3", "fp8_ds_mla"]
    enforce_eager: NotRequired[bool]
    enable_return_routed_experts: NotRequired[bool]
    # Collect vLLM request, cache, and cumulative token counters in a model-owner
    # background thread for performance diagnostics.
    enable_vllm_metrics_logger: NotRequired[bool]
    # Sampling cadence for the optional vLLM metrics logger.
    vllm_metrics_logger_interval: NotRequired[float]
    # Whether to show a tqdm progress bar during generation. Defaults to vLLM's own default (True) when absent. Only applies when async_engine is False.
    use_tqdm: NotRequired[bool]
    # By default, NeMo RL only has a Python handle to the vllm.LLM generation engine. The expose_http_server flag here will expose that generation engine as an HTTP server.
    # Exposing vLLM as a server is useful in instances where the multi-turn rollout is performed with utilities outside of NeMo RL, but the user still wants to take advantage of the refit logic in NeMo RL that keeps the policy and generation up to date.
    # Currently it will expose the /tokenize and /v1/chat/completions endpoints. Later on we may expose /v1/completions or /v1/responses.
    expose_http_server: NotRequired[bool]
    # Require `Authorization: Bearer <key>` on the exposed server's OpenAI routes
    # (/v1/* and /tokenize). VllmGeneration generates one random key per job, hands
    # it to the workers, and exposes it as `http_server_api_key` for the NeMo-Gym
    # model server. Needs expose_http_server; refused at setup without it.
    # Absent or false: the routes accept unauthenticated requests.
    http_server_api_key_required: NotRequired[bool]
    # The generated key. Filled by VllmGeneration in the copy of the config the
    # workers receive; never set in a YAML.
    http_server_api_key: NotRequired[str]
    # Environment variable containing the internal refit API key.
    http_refit_api_key_env_var: NotRequired[str | None]
    # Invalidate weight-dependent multimodal encoder outputs after a successful
    # async refit. Enable only when generation is quiesced during weight updates.
    reset_encoder_cache_after_weight_update: NotRequired[bool]
    # Fixed internal refit endpoint port for stable Kubernetes targetPorts.
    http_refit_server_port: NotRequired[int | None]
    # Fixed ZeroMQ relay port for stable Kubernetes targetPorts.
    zmq_refit_server_port: NotRequired[int | None]
    # These kwargs are passed to the vllm.LLM HTTP server Chat Completions endpoint config. Typically this will include things like tool parser, chat template, etc
    http_server_serving_chat_kwargs: NotRequired[dict[str, Any]]
    # Miscellaneous top level vLLM HTTP server arguments.
    # A filepath that can be imported to register a vLLM tool parser
    tool_parser_plugin: NotRequired[str]
    # Extra environment variables forwarded to every vLLM worker process. Useful
    # for per-recipe knobs (e.g. forcing a specific fused-MoE backend) without
    # affecting other test cases.
    env_vars: NotRequired[dict[str, str]]
    # Opt into vLLM's native reload_weights API for refit. The default stays
    # False so existing IPC/NCCL refit behavior keeps using NeMo-RL's legacy
    # loader path.
    refit_with_reload_api: NotRequired[bool]
    # A filepath that can be imported to register a vLLM reasoning parser
    reasoning_parser_plugin: NotRequired[str]

    # Prefill/decode disaggregation (generation.disaggregation.enabled). Read
    # only then, inert otherwise.
    #
    # Per-role engine overrides: a role names only what differs from the shared
    # values above (VllmEngineArgs); keys outside that set are rejected at
    # startup. resolve_vllm_role_overrides replaces each with that role's
    # complete engine config, which is all anything downstream reads.
    prefill_engine: NotRequired[VllmEngineArgs]
    decode_engine: NotRequired[VllmEngineArgs]
    # How the frontends in front of a replica route and relay: the vLLM
    # counterpart of trtllm_cfg.disagg_server.
    disagg_frontend: NotRequired[VllmDisaggFrontendConfig]
    # The KV connector every engine of a replica is built with: the vLLM
    # counterpart of trtllm_kwargs.cache_transceiver_config.
    kv_transfer: NotRequired[VllmKvTransferConfig]


def vllm_nemotron_h_fp32_lm_head_enabled(
    vllm_cfg: VllmSpecificArgs | dict[str, Any],
) -> bool:
    """Return whether vLLM should run Nemotron-H logits with an fp32 head."""
    return bool(vllm_cfg.get("fp32_lm_head"))


class VllmDeltaCompressionConfig(BaseModel, extra="allow"):
    encoding: Literal["xor", "overwrite"] = "xor"
    sparse_bucket_size_bytes: PositiveInt = 512 * 1024**2
    export_chunk_bytes: dict[str, PositiveInt] = Field(
        default_factory=lambda: {"s3": 64 * 1024**2, "zmq": 256 * 1024**2}
    )
    zstd_threads: dict[str, NonNegativeInt] = Field(
        default_factory=lambda: {"s3": 0, "zmq": 0}
    )


class VllmRefitStorageConfig(BaseModel, extra="allow"):
    s3_bucket: str | None = None
    s3_region: str = "us-east-1"
    s3_prefix: str = "nemo-rl-refit"
    staging_dir: str = "/dev/shm"


class VllmRefitBaselineConfig(BaseModel, extra="allow"):
    in_memory: bool = False
    mmap_dir: str | None = None


class VllmRefitTuningConfig(BaseModel, extra="allow"):
    encode_workers: dict[str, PositiveInt] = Field(
        default_factory=lambda: {"s3": 8, "zmq": 8}
    )
    transfer_workers: dict[str, PositiveInt] = Field(
        default_factory=lambda: {"s3": 32, "zmq": 4}
    )
    zmq_retries: NonNegativeInt = 3
    zmq_relay_payload_workers: PositiveInt = 16
    zmq_relay_forward_workers: PositiveInt = 8
    apply_queue_depth: PositiveInt = 32
    apply_batch_size: PositiveInt = 8
    partition_workers: PositiveInt = 8


class VllmSparseRefitConfig(BaseModel, extra="allow"):
    delta_compression: VllmDeltaCompressionConfig = Field(
        default_factory=VllmDeltaCompressionConfig
    )
    storage: VllmRefitStorageConfig = Field(default_factory=VllmRefitStorageConfig)
    baseline: VllmRefitBaselineConfig = Field(default_factory=VllmRefitBaselineConfig)
    tuning: VllmRefitTuningConfig = Field(default_factory=VllmRefitTuningConfig)
    verify_samples_per_payload: NonNegativeInt = 0
    request_timeout_s: PositiveFloat = 600.0


class VllmNixlRefitConfig(BaseModel, extra="forbid"):
    update_weights_bucket_memory_ratio: Annotated[float, Field(gt=0, lt=1)] = 0.05
    device: str = "cuda"
    backend_name: str = "UCX"
    backend_init_params: dict[str, Any] | None = None
    release_after_refit: bool = False
    shard_expert_weights: bool = False


class VllmCheckpointEnginePluginConfig(BaseModel, extra="allow"):
    update_weights_bucket_memory_ratio: Annotated[float, Field(gt=0, lt=1)] = 0.05
    release_after_refit: bool = False


class VllmRefitConfig(BaseModel, extra="allow"):
    sparse: VllmSparseRefitConfig = Field(default_factory=VllmSparseRefitConfig)
    nixl: VllmNixlRefitConfig = Field(default_factory=VllmNixlRefitConfig)


class VllmConfig(GenerationConfig):
    vllm_cfg: VllmSpecificArgs
    vllm_kwargs: NotRequired[dict[str, Any]]
    # Per-token NVFP4 W4A4 rollout (TE-training flow; no ModelOpt training).
    # Mutually exclusive with quant_cfg/real_quant below. Defaults and validation
    # live in NvFp4PerTokenRolloutConfig.
    nvfp4_pertoken_rollout: NotRequired[NvFp4PerTokenRolloutConfig]
    # Null uses the topology default (IPC colocated, NCCL non-colocated).
    # Built-ins select sparse delta over S3/ZeroMQ or NIXL.
    # A custom checkpoint engine may use a ``module:ClassName`` selector.
    refit_transport: NotRequired[VllmRefitSelector | str | None]
    refit_cfg: NotRequired[VllmRefitConfig | None]

    # quantization config
    quant_cfg: NotRequired[str | None]
    # When set with ``quant_cfg``, initialize rollout vLLM with real ModelOpt
    # NVFP4 kernels and stream packed quantized weights instead of fake-quant
    # modules. This is intended for ModelOpt NVFP4 rollout experiments.
    real_quant: NotRequired[bool]
    # CPU offload remains the default. Disabling it is supported only for
    # colocated CUDA-IPC refit, where packed export tensors can stay on GPU.
    real_quant_export_cpu_offload: NotRequired[bool]
    real_quant_ignore: NotRequired[list[str]]

    # FQN of a worker extension class to use instead of the resolved default
    # generation worker. Must be a subclass of the resolved worker and cannot
    # be combined with quant_cfg. Its runtime environment must already be in
    # ACTOR_ENVIRONMENT_REGISTRY.
    worker_extension_cls_fqn: NotRequired[str | None]


def validate_vllm_quantization_config(config: VllmConfig) -> None:
    """Reject quantization options that would otherwise be silently ignored."""
    vllm_cfg = config.get("vllm_cfg")
    if vllm_cfg is None:
        return
    refit_prequantize = vllm_cfg.get("refit_prequantize")
    if refit_prequantize is not None and not isinstance(refit_prequantize, bool):
        raise ValueError(
            "policy.generation.vllm_cfg.refit_prequantize must be a boolean."
        )
    if refit_prequantize and not (
        vllm_cfg.get("precision") == "fp8" and vllm_cfg.get("is_mx") is True
    ):
        raise ValueError(
            "policy.generation.vllm_cfg.refit_prequantize requires "
            "precision='fp8' and is_mx=true."
        )
    if refit_prequantize and config.get("refit_transport") == "nccl_reshard":
        raise ValueError(
            "policy.generation.vllm_cfg.refit_prequantize is not supported with "
            "nccl_reshard; that transport owns its weight-format conversion."
        )
    if refit_prequantize and vllm_cfg.get("refit_with_reload_api"):
        raise ValueError(
            "policy.generation.vllm_cfg.refit_prequantize is not supported with "
            "refit_with_reload_api; prequantized refit uses receiver scale names "
            "that are not accepted by vLLM's native reload path."
        )


def resolve_vllm_video_config(config: VllmConfig) -> VllmVideoConfig | None:
    """Validate and return the optional vLLM video sampling contract."""
    vllm_config = cast(dict[str, Any], config["vllm_cfg"])
    if "video_loader" in vllm_config:
        raise ValueError(
            "policy.generation.vllm_cfg.video_loader is not supported; "
            "use policy.generation.vllm_cfg.video"
        )

    raw_video_config = vllm_config.get("video")
    if raw_video_config is None:
        return None
    return VllmVideoConfig.model_validate(raw_video_config)


def materialize_vllm_video_config(
    policy_config: dict[str, Any], data_config: dict[str, Any]
) -> None:
    """Apply one video contract to tokenizer, data, and vLLM request config."""
    generation_config = policy_config["generation"]
    if generation_config["backend"] != "vllm":
        return

    video_config = resolve_vllm_video_config(generation_config)
    if video_config is None:
        return

    # Keep the normalized value dict-shaped for OmegaConf/Ray serialization.
    generation_config["vllm_cfg"]["video"] = video_config.model_dump()

    tokenizer_video_config = policy_config["tokenizer"].setdefault("video", {})
    tokenizer_video_config["num_frames"] = video_config.num_frames

    # TODO(rohitrango): Let ProcessorInterface materialize model-specific data keys.
    data_defaults = data_config.setdefault("default", {})
    data_defaults.update(
        {
            "num_frames": video_config.num_frames,
            "video_sampling_style": video_config.sampling_style,
            "video_temporal_patch_size": video_config.temporal_patch_size,
        }
    )

    vllm_kwargs = generation_config.get("vllm_kwargs")
    if vllm_kwargs is None:
        raise ValueError(
            "policy.generation.vllm_kwargs is required when vllm_cfg.video is set"
        )
    limit_mm_per_prompt = vllm_kwargs.get("limit_mm_per_prompt")
    if not isinstance(limit_mm_per_prompt, dict):
        raise ValueError(
            "policy.generation.vllm_kwargs.limit_mm_per_prompt must configure video"
        )
    video_limit = limit_mm_per_prompt.get("video")
    if not isinstance(video_limit, dict):
        raise ValueError(
            "policy.generation.vllm_kwargs.limit_mm_per_prompt.video must be a mapping"
        )
    video_limit["num_frames"] = video_config.num_frames

    media_io_kwargs = vllm_kwargs.setdefault("media_io_kwargs", {})
    if not isinstance(media_io_kwargs, dict):
        raise ValueError(
            "policy.generation.vllm_kwargs.media_io_kwargs must be a mapping"
        )
    video_media_io_kwargs = media_io_kwargs.setdefault("video", {})
    if not isinstance(video_media_io_kwargs, dict):
        raise ValueError(
            "policy.generation.vllm_kwargs.media_io_kwargs.video must be a mapping"
        )
    # VideoMediaIO otherwise defaults to 32 independently of the policy-side
    # frame count. Materializing the value here makes a mismatch impossible.
    video_media_io_kwargs["num_frames"] = video_config.num_frames


# Entry points with end-to-end coverage for the per-token NVFP4 rollout.
# Every other entry point shares the same Megatron training worker and vLLM
# refit path, so the mode is expected to work there, but is unvalidated.
NVFP4_PERTOKEN_VALIDATED_ENTRY_POINTS = frozenset({"grpo"})


def parse_nvfp4_pertoken_rollout(
    config: VllmConfig,
) -> NvFp4PerTokenRolloutConfig | None:
    """Parse the optional per-token rollout block under its strict schema."""
    raw = config.get("nvfp4_pertoken_rollout")
    if raw is None:
        return None
    parsed = NvFp4PerTokenRolloutConfig.model_validate(raw)
    return parsed if parsed.enabled else None


def normalize_nvfp4_pertoken_policy_config(
    policy_config: Mapping[str, Any],
    *,
    entry_point: str = "grpo",
) -> None:
    """Derive rollout BF16 exclusions from the policy's Megatron boundary.

    This runs on the driver before generation workers are created so trainer
    and rollout actors receive one normalized precision contract.

    ``entry_point`` names the calling algorithm so an unvalidated one warns
    instead of appearing supported; see
    ``NVFP4_PERTOKEN_VALIDATED_ENTRY_POINTS``.
    """
    generation_config = policy_config.get("generation") or {}
    if generation_config.get("backend") != "vllm":
        return
    rollout = parse_nvfp4_pertoken_rollout(cast(VllmConfig, generation_config))
    if rollout is None:
        return
    if entry_point not in NVFP4_PERTOKEN_VALIDATED_ENTRY_POINTS:
        warnings.warn(
            "generation.nvfp4_pertoken_rollout has end-to-end coverage on "
            f"{sorted(NVFP4_PERTOKEN_VALIDATED_ENTRY_POINTS)} only. {entry_point} "
            "reuses the same Megatron training worker and vLLM refit path, so the "
            "mode is expected to work, but it has not been validated there.",
            UserWarning,
            stacklevel=2,
        )

    from transformers import AutoConfig

    model_name = policy_config.get("model_name")
    if not model_name:
        raise ValueError(
            "NVFP4 per-token boundary resolution requires policy.model_name"
        )
    overrides = policy_config.get("hf_config_overrides") or {}
    hf_config = AutoConfig.from_pretrained(
        model_name, trust_remote_code=True, **overrides
    )
    text_config = getattr(hf_config, "text_config", None)
    num_hidden_layers = overrides.get("num_hidden_layers")
    if num_hidden_layers is None:
        num_hidden_layers = getattr(hf_config, "num_hidden_layers", None)
    if num_hidden_layers is None and text_config is not None:
        num_hidden_layers = getattr(text_config, "num_hidden_layers", None)
    if num_hidden_layers is None:
        raise ValueError(
            "NVFP4 per-token boundary resolution requires num_hidden_layers "
            "in the HF config"
        )

    megatron_cfg = policy_config.get("megatron_cfg") or {}
    first_last_layers_bf16 = bool(megatron_cfg.get("first_last_layers_bf16", False))
    num_start = int(
        megatron_cfg.get(
            "num_layers_at_start_in_bf16",
            MCORE_DEFAULT_NUM_LAYERS_AT_START_IN_BF16,
        )
        or 0
    )
    num_end = int(
        megatron_cfg.get(
            "num_layers_at_end_in_bf16",
            MCORE_DEFAULT_NUM_LAYERS_AT_END_IN_BF16,
        )
        or 0
    )
    raw_rollout = generation_config.setdefault("nvfp4_pertoken_rollout", {})
    # A user-written value is only cross-checked; the Megatron boundary below
    # is what the rollout ends up using.
    expected_ignore = (
        rollout.additional_ignore if "additional_ignore" in raw_rollout else None
    )
    resolved = resolve_boundary_ignore_patterns(
        num_hidden_layers=int(num_hidden_layers),
        first_last_layers_bf16=first_last_layers_bf16,
        num_layers_at_start_in_bf16=num_start,
        num_layers_at_end_in_bf16=num_end,
        expected_additional_ignore=expected_ignore,
    )
    raw_rollout["additional_ignore"] = resolved
    print(
        "[nvfp4_pertoken] derived rollout BF16 decoder layers from Megatron: "
        f"{resolved}",
        flush=True,
    )


def validate_nvfp4_pertoken_generation(
    config: VllmConfig, *, is_eval: bool
) -> NvFp4PerTokenRolloutConfig | None:
    """Reject topology and rollout combinations unsupported by per-token NVFP4."""
    rollout = parse_nvfp4_pertoken_rollout(config)
    if rollout is None:
        return None
    if is_eval:
        raise ValueError(
            "generation.nvfp4_pertoken_rollout does not support standalone evaluation"
        )
    if config.get("quant_cfg") is not None or config.get("real_quant"):
        raise ValueError(
            "generation.nvfp4_pertoken_rollout is mutually exclusive with "
            "generation.quant_cfg and generation.real_quant"
        )
    if config.get("refit_transport") is not None:
        raise ValueError(
            "generation.nvfp4_pertoken_rollout requires refit_transport=null"
        )
    colocated = config.get("colocated") or {}
    vllm_cfg = config["vllm_cfg"]
    if not colocated.get("enabled"):
        raise ValueError("generation.nvfp4_pertoken_rollout requires colocated rollout")
    if vllm_cfg["pipeline_parallel_size"] > 1:
        raise ValueError(
            "generation.nvfp4_pertoken_rollout does not support vLLM PP>1 yet: "
            "a stage that owns quantized layers rejects another stage's expert "
            "weights during refit. Set generation.vllm_cfg.pipeline_parallel_size=1."
        )
    if vllm_cfg.get("expert_parallel_size") != 1:
        raise ValueError("generation.nvfp4_pertoken_rollout requires vLLM EP=1")
    if vllm_cfg.get("kv_cache_dtype") != "auto":
        raise ValueError(
            "generation.nvfp4_pertoken_rollout requires kv_cache_dtype=auto"
        )
    if vllm_cfg.get("precision") != "bfloat16":
        raise ValueError(
            "generation.nvfp4_pertoken_rollout requires vLLM precision=bfloat16"
        )
    speculative_config = (config.get("vllm_kwargs") or {}).get("speculative_config")
    if speculative_config and not (
        isinstance(speculative_config, dict)
        and speculative_config.get("num_speculative_tokens") == 0
    ):
        raise ValueError(
            "generation.nvfp4_pertoken_rollout does not support speculative decoding"
        )
    return rollout


def validate_nvfp4_pertoken_model(hf_config: Any) -> None:
    """Preflight the model before vLLM performs constructed-model validation."""
    if get_num_routed_experts(hf_config) is None:
        architectures = getattr(hf_config, "architectures", []) or []
        raise ValueError(
            "generation.nvfp4_pertoken_rollout requires a model with routed "
            f"experts; no routed-expert count was found for {architectures or 'the HF config'}."
        )


def normalize_vllm_refit_config(config: VllmConfig) -> VllmRefitConfig | None:
    """Validate the selected refit transport and resolve its scoped defaults."""
    validate_vllm_quantization_config(config)
    rollout = parse_nvfp4_pertoken_rollout(config)
    if cast(dict[str, Any], config).get("checkpoint_engine") is not None:
        raise ValueError(
            "policy.generation.checkpoint_engine was replaced by "
            "policy.generation.refit_transport='nixl' and "
            "policy.generation.refit_cfg.nixl."
        )
    transport = config.get("refit_transport")
    if rollout is not None and transport is not None:
        raise ValueError(
            "generation.nvfp4_pertoken_rollout requires refit_transport=null"
        )
    if transport is None:
        return None
    if transport == "nccl_reshard":
        # nccl_reshard doesn't takes refit_cfg.
        return None
    if transport not in get_args(VllmRefitSelector) and ":" not in transport:
        raise ValueError(
            f"Unknown vLLM refit transport {transport!r}: expected null, "
            "'nccl_reshard', 'vllm_s3_sparse', 'vllm_zmq_sparse', 'nixl', or a "
            "'module:ClassName' checkpoint-engine path."
        )
    # The encoder-cache reset is implemented only on the collective/IPC and
    # nccl_reshard async refit paths (both returned above). Fail loudly rather
    # than let other transports silently keep stale multimodal encoder outputs
    # across weight updates. Some callers re-validate partial generation
    # configs (e.g. worker-side NIXL setup), so vllm_cfg may be absent here.
    vllm_cfg = config.get("vllm_cfg")
    if vllm_cfg and vllm_cfg.get("reset_encoder_cache_after_weight_update"):
        raise ValueError(
            "vllm_cfg.reset_encoder_cache_after_weight_update is not supported "
            f"with refit_transport={transport!r}: this transport's refit path "
            "does not reset the multimodal encoder cache, so stale multimodal "
            "embeddings would silently survive weight updates. Supported "
            "transports: null (collective/IPC) and 'nccl_reshard'."
        )
    refit_config = VllmRefitConfig.model_validate(config.get("refit_cfg") or {})
    if ":" in transport:
        plugin_config = (refit_config.model_extra or {}).get(transport)
        if plugin_config is None:
            raise ValueError(
                f"Custom checkpoint-engine transport {transport!r} requires "
                f"policy.generation.refit_cfg[{transport!r}]."
            )
        VllmCheckpointEnginePluginConfig.model_validate(plugin_config)
    config["refit_cfg"] = refit_config
    return refit_config


# --------------------------------------------------------------------------- #
#  Prefill/decode disaggregation
# --------------------------------------------------------------------------- #

VllmRole = Literal["prefill", "decode"]


def resolve_vllm_disagg_layout(config: VllmConfig) -> DisaggConfig:
    """``generation.disaggregation`` as its schema (absent == disabled)."""
    return disagg_layout.resolve_disagg_layout(config)


def resolve_vllm_role_overrides(config: VllmConfig) -> VllmConfig:
    """Resolve ``vllm_cfg.{prefill,decode}_engine`` into complete engine configs.

    See :func:`disagg_layout.resolve_role_overrides`. A role may only name the
    :class:`VllmEngineArgs` keys; its ``vllm_kwargs`` is deep-merged over
    ``generation.vllm_kwargs``. Returns ``config`` itself without
    disaggregation, where the role blocks are inert.
    """
    return cast(
        VllmConfig,
        disagg_layout.resolve_role_overrides(
            config,
            backend_cfg_key="vllm_cfg",
            allowed_keys=VLLM_ENGINE_ARG_KEYS,
            excluded_keys=VLLM_NON_ENGINE_KEYS,
            kwargs_key="vllm_kwargs",
        ),
    )


def vllm_role_engine_config(config: VllmConfig, role: VllmRole) -> dict[str, Any]:
    """The whole engine config of ``role``. ``config`` must be role-resolved."""
    return disagg_layout.role_engine_config(config, role, backend_cfg_key="vllm_cfg")


def vllm_replica_width(config: VllmConfig) -> int:
    """GPUs one rollout replica spans (one engine's TP x PP without disaggregation)."""
    return disagg_layout.replica_width(config, backend_cfg_key="vllm_cfg")


def resolve_vllm_disagg_frontend_config(config: VllmConfig) -> VllmDisaggFrontendConfig:
    """``vllm_cfg.disagg_frontend`` as its schema."""
    raw = disagg_layout.as_dict(config["vllm_cfg"].get("disagg_frontend"))
    # Caught ahead of extra="forbid" only to point at where the key belongs.
    misplaced = sorted(set(raw) & set(DisaggConfig.model_fields))
    if misplaced:
        raise ValueError(
            f"{misplaced} set in vllm_cfg.disagg_frontend, which only carries the "
            "frontends' routing and relay settings. The disaggregation layout "
            "belongs in generation.disaggregation."
        )
    return VllmDisaggFrontendConfig.model_validate(raw)


def resolve_vllm_kv_transfer_config(config: VllmConfig) -> VllmKvTransferConfig:
    """``vllm_cfg.kv_transfer`` as its schema."""
    raw = disagg_layout.as_dict(config["vllm_cfg"].get("kv_transfer"))
    resolved = VllmKvTransferConfig.model_validate(raw)
    if resolved.kv_connector_extra_config.bidirectional_kv_xfer is not None:
        raise ValueError(
            "vllm_cfg.kv_transfer.kv_connector_extra_config.bidirectional_kv_xfer "
            "is written by NeMo RL from vllm_cfg.disagg_frontend.bidirectional_kv_xfer; "
            "set that instead so the frontend and the engines cannot disagree."
        )
    return resolved


def resolve_vllm_connector_profile(config: VllmConfig) -> ConnectorProfile:
    """The handshake profile of the configured KV connector."""
    return connector_profile(
        resolve_vllm_kv_transfer_config(config).kv_connector,
        resolve_vllm_disagg_frontend_config(config).connector_profile,
    )


def vllm_kv_transfer_config_for_role(
    config: VllmConfig, role: VllmRole, *, engine_id: str
) -> dict[str, Any]:
    """The ``KVTransferConfig`` kwargs of one engine of ``role``.

    ``kv_role`` comes from the connector profile; ``bidirectional_kv_xfer`` is
    copied from the frontend block onto both roles' extra config, which is
    where vLLM reads it.
    """
    kv_transfer = resolve_vllm_kv_transfer_config(config)
    frontend = resolve_vllm_disagg_frontend_config(config)
    profile = connector_profile(kv_transfer.kv_connector, frontend.connector_profile)
    extra = kv_transfer.kv_connector_extra_config.model_dump(exclude_none=True)
    if frontend.bidirectional_kv_xfer:
        extra["bidirectional_kv_xfer"] = True
    kwargs: dict[str, Any] = {
        "kv_connector": kv_transfer.kv_connector,
        "kv_role": profile.producer_role
        if role == "prefill"
        else profile.consumer_role,
        "engine_id": engine_id,
        "kv_connector_extra_config": extra,
        "kv_load_failure_policy": kv_transfer.kv_load_failure_policy,
        "kv_buffer_device": kv_transfer.kv_buffer_device,
    }
    if kv_transfer.kv_connector_module_path is not None:
        kwargs["kv_connector_module_path"] = kv_transfer.kv_connector_module_path
    return kwargs


def validate_vllm_disaggregation_for_refit(
    config: VllmConfig, *, in_flight_weight_updates: bool
) -> None:
    """Reject disaggregation settings that cannot survive an in-flight refit.

    An in-flight weight update pauses every engine with vLLM's "keep" mode,
    which completes only once the engine is idle -- and a finished request
    whose blocks the KV connector still holds counts as work. Under
    ``bidirectional_kv_xfer`` every decode engine pins each finished turn's
    blocks for ``decoder_kv_blocks_ttl`` (vLLM default 480 s) so the next
    turn's prefill can pull them back, and vLLM offers no call to release them
    early, so the pause would block for up to that TTL on every update.
    """
    if not in_flight_weight_updates:
        return
    layout = resolve_vllm_disagg_layout(config)
    if not layout.enabled:
        return
    frontend = resolve_vllm_disagg_frontend_config(config)
    if frontend.bidirectional_kv_xfer:
        raise ValueError(
            "policy.generation.vllm_cfg.disagg_frontend.bidirectional_kv_xfer=true "
            "cannot be combined with grpo.async_grpo.in_flight_weight_updates=true: "
            "vLLM completes the in-flight pause only once an engine holds no "
            "connector-pinned blocks, and bidirectional KV transfer pins every "
            "finished turn's blocks on the decode engines for decoder_kv_blocks_ttl. "
            "Turn one of the two off."
        )


# Model types whose Mamba/SSM layers vLLM's NixlConnector can hand over only
# between engines of equal TP (the recurrent state has no head-splitting
# remap yet). Checked by name and by the HF config attributes hybrid models
# carry, so a new hybrid model is caught even if its name is not listed.
_HYBRID_SSM_MODEL_TYPES = frozenset(
    {
        "nemotron_h",
        "nemotron_h_omni",
        "mamba",
        "mamba2",
        "jamba",
        "bamba",
        "zamba2",
        "falcon_h1",
        "granitemoehybrid",
        "qwen3_next",
        "lfm2",
        "plamo2",
    }
)
_HYBRID_SSM_CONFIG_ATTRS = (
    "hybrid_override_pattern",
    "layers_block_type",
    "mamba_d_state",
)


def hybrid_ssm_requires_homogeneous_tp(hf_config: Any) -> bool:
    """Whether ``hf_config`` describes a model with Mamba/SSM layers."""
    text_config = getattr(hf_config, "text_config", None) or hf_config
    model_type = str(getattr(text_config, "model_type", "") or "").lower()
    if model_type in _HYBRID_SSM_MODEL_TYPES:
        return True
    return any(
        getattr(text_config, attr, None) is not None
        for attr in _HYBRID_SSM_CONFIG_ATTRS
    )


def validate_vllm_disaggregation(
    config: VllmConfig, *, hf_config: Any | None = None
) -> None:
    """Refuse at setup the disaggregation configs that would run wrong or hang.

    ``config`` must be role-resolved. ``hf_config`` enables the hybrid-SSM TP
    check when the caller has it; the driver passes it, pure-config callers
    may not.
    """
    layout = resolve_vllm_disagg_layout(config)
    if not layout.enabled:
        return
    vllm_cfg = cast(dict[str, Any], config["vllm_cfg"])

    if config["colocated"]["enabled"]:
        raise ValueError(
            "PD disaggregation requires non-colocated generation: colocated mode "
            "sleeps the engines between rollouts, which drops the KV cache the "
            "connector hands between prefill and decode. Set colocated.enabled=false "
            "or generation.disaggregation.enabled=false."
        )
    if not vllm_cfg.get("async_engine") or not vllm_cfg.get("expose_http_server"):
        raise ValueError(
            "PD disaggregation needs vllm_cfg.async_engine=true and "
            "vllm_cfg.expose_http_server=true: the frontends reach every engine "
            "over HTTP. The NeMo-Gym entrypoint sets both."
        )
    if int(vllm_cfg["pipeline_parallel_size"]) != 1:
        raise ValueError(
            "PD disaggregation on the vLLM backend supports pipeline_parallel_size=1 "
            f"only, got {vllm_cfg['pipeline_parallel_size']}."
        )
    if config.get("refit_transport") is not None:
        raise ValueError(
            "PD disaggregation is validated with the default collective refit only; "
            f"set generation.refit_transport=null (got {config['refit_transport']!r})."
        )
    for role in disagg_layout.ROLES:
        engine_cfg = vllm_role_engine_config(config, role)
        tp = int(engine_cfg["tensor_parallel_size"])
        ep = int(engine_cfg["expert_parallel_size"])
        if ep > tp:
            raise ValueError(
                f"vllm_cfg.{role}_engine: expert_parallel_size ({ep}) exceeds its "
                f"tensor_parallel_size ({tp}). EP > TP means data parallelism inside "
                "one vLLM engine, which the async (HTTP) path, and so PD "
                "disaggregation, does not support."
            )
        if "kv_transfer_config" in (engine_cfg.get("vllm_kwargs") or {}):
            raise ValueError(
                "vllm_kwargs.kv_transfer_config is owned by NeMo RL under PD "
                "disaggregation (built from vllm_cfg.kv_transfer and the engine's "
                "role); remove it from generation.vllm_kwargs and from "
                f"vllm_cfg.{role}_engine.vllm_kwargs."
            )
    frontend = resolve_vllm_disagg_frontend_config(config)
    profile = resolve_vllm_connector_profile(config)
    if frontend.bidirectional_kv_xfer and not profile.supports_bidirectional:
        raise ValueError(
            f"vllm_cfg.disagg_frontend.bidirectional_kv_xfer is not supported by "
            f"the {profile.name} connector profile."
        )
    if frontend.prefill_router == "kv_cache_aware" and not layout.frontend_tokenize:
        raise ValueError(
            "vllm_cfg.disagg_frontend.prefill_router=kv_cache_aware needs "
            "generation.disaggregation.frontend_tokenize=true: the frontend hashes "
            "the prompt's blocks itself to match them against the engines' events."
        )
    if hf_config is not None and hybrid_ssm_requires_homogeneous_tp(hf_config):
        tps = {
            role: int(vllm_role_engine_config(config, role)["tensor_parallel_size"])
            for role in disagg_layout.ROLES
        }
        if tps["prefill"] != tps["decode"]:
            raise ValueError(
                "This model has Mamba/SSM layers, whose recurrent state vLLM's KV "
                "connector can only hand over between engines of equal tensor "
                f"parallelism; got prefill TP {tps['prefill']} and decode TP "
                f"{tps['decode']}. Give both roles the same tensor_parallel_size."
            )
