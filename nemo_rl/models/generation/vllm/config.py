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

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Annotated, Any, Literal, NotRequired, TypedDict, cast, get_args

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    NonNegativeInt,
    PositiveFloat,
    PositiveInt,
)

from nemo_rl.models.generation.interfaces import GenerationConfig

VllmRefitTransportName = Literal["s3", "zmq"]
VllmRefitSelector = Literal["vllm_s3_sparse", "vllm_zmq_sparse", "nixl", "nccl_reshard"]
VLLM_SPARSE_REFIT_TRANSPORTS = frozenset({"vllm_s3_sparse", "vllm_zmq_sparse"})
LoraRefitMode = Literal["native", "merged"]

NATIVE_LORA_ADAPTER_ID = 1
NATIVE_LORA_ADAPTER_NAME = "nemo-rl-online"
NATIVE_LORA_ADAPTER_PATH = "/nemo-rl/in-memory-online-lora"
NATIVE_LORA_CONFIG_KEY = "nemo_rl_native_lora"


@dataclass(frozen=True)
class NativeLoraRefitSettings:
    """Internal native-LoRA settings forwarded to each vLLM worker."""

    rank: int
    alpha: int


def native_lora_refit_settings(vllm_config: Any) -> NativeLoraRefitSettings | None:
    """Read native-LoRA settings from a realized vLLM worker config."""
    additional_config = getattr(vllm_config, "additional_config", None)
    if not isinstance(additional_config, dict):
        return None
    raw_settings = additional_config.get(NATIVE_LORA_CONFIG_KEY)
    if raw_settings is None:
        return None
    return NativeLoraRefitSettings(
        rank=int(raw_settings["rank"]),
        alpha=int(raw_settings["alpha"]),
    )


# TODO(rohitrango): Move model-specific video fields behind ProcessorInterface.
class VllmVideoConfig(BaseModel):
    """Video sampling contract shared by policy preprocessing and vLLM."""

    model_config = ConfigDict(extra="forbid")

    sampling_style: Literal["nemotron_vl"]
    num_frames: PositiveInt
    temporal_patch_size: PositiveInt


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
    # Cap each request's generated tokens so the training prompt plus response
    # fits within max_model_len. This is needed when multimodal processing makes
    # the training prompt longer than its text-only representation.
    cap_max_tokens_to_context: NotRequired[bool]
    # Use ModelOpt MXFP8 quantization when precision is fp8.
    is_mx: NotRequired[bool]
    # Deprecated in 0.8. Use quantization_ignore_patterns instead.
    quantization_ignored_layer_kws: NotRequired[list[str]]
    # MXFP8 exclusion patterns forwarded through vLLM's quantization config.
    # Supports exact names, substrings, and fnmatch wildcards.
    quantization_ignore_patterns: NotRequired[list[str]]
    kv_cache_dtype: Literal["auto", "fp8", "fp8_e4m3"]
    enforce_eager: NotRequired[bool]
    enable_return_routed_experts: NotRequired[bool]
    # Whether to show a tqdm progress bar during generation. Defaults to vLLM's own default (True) when absent. Only applies when async_engine is False.
    use_tqdm: NotRequired[bool]
    # By default, NeMo RL only has a Python handle to the vllm.LLM generation engine. The expose_http_server flag here will expose that generation engine as an HTTP server.
    # Exposing vLLM as a server is useful in instances where the multi-turn rollout is performed with utilities outside of NeMo RL, but the user still wants to take advantage of the refit logic in NeMo RL that keeps the policy and generation up to date.
    # Currently it will expose the /tokenize and /v1/chat/completions endpoints. Later on we may expose /v1/completions or /v1/responses.
    expose_http_server: NotRequired[bool]
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
    # A filepath that can be imported to register a vLLM reasoning parser
    reasoning_parser_plugin: NotRequired[str]


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
    # Null uses the topology default (IPC colocated, NCCL non-colocated).
    # Built-ins select sparse delta over S3/ZeroMQ or NIXL.
    # A custom checkpoint engine may use a ``module:ClassName`` selector.
    refit_transport: NotRequired[VllmRefitSelector | str | None]
    refit_cfg: NotRequired[VllmRefitConfig | None]
    # LoRA policies use vLLM's factorized adapter runtime by default. Set to
    # ``merged`` to opt into materializing and refitting full ``W + BA`` weights.
    lora_refit_mode: NotRequired[LoraRefitMode]

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


def resolve_vllm_video_config(config: VllmConfig) -> VllmVideoConfig | None:
    """Validate and return the optional vLLM video sampling contract."""
    raw_video_config = config["vllm_cfg"].get("video")
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


def normalize_vllm_refit_config(config: VllmConfig) -> VllmRefitConfig | None:
    """Validate the selected refit transport and resolve its scoped defaults."""
    if cast(dict[str, Any], config).get("checkpoint_engine") is not None:
        raise ValueError(
            "policy.generation.checkpoint_engine was replaced by "
            "policy.generation.refit_transport='nixl' and "
            "policy.generation.refit_cfg.nixl."
        )
    transport = config.get("refit_transport")
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
            "does not reset the multimodal encoder cache, so stale vision "
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


def configure_vllm_lora_refit(policy_config: Mapping[str, Any]) -> None:
    """Validate and materialize vLLM's LoRA refit representation.

    When LoRA training is enabled, an omitted mode resolves to native refit;
    merged full-weight materialization must be selected explicitly.

    Args:
        policy_config: Complete policy config containing training and generation
            backend settings.

    Raises:
        ValueError: If native refit is selected for an unsupported policy,
            transport, engine mode, precision, or vLLM LoRA configuration.
    """
    generation_config = cast(VllmConfig, policy_config["generation"])

    dtensor_config = policy_config.get("dtensor_cfg") or {"enabled": False}
    dtensor_lora_config = (
        dtensor_config.get("lora_cfg") if dtensor_config["enabled"] else None
    )
    dtensor_lora_enabled = bool(
        dtensor_lora_config is not None and dtensor_lora_config["enabled"]
    )

    megatron_config = policy_config.get("megatron_cfg")
    megatron_lora_config = (
        megatron_config.get("peft")
        if megatron_config is not None and megatron_config["enabled"]
        else None
    )
    megatron_lora_enabled = bool(
        megatron_lora_config is not None and megatron_lora_config["enabled"]
    )

    if not dtensor_lora_enabled and not megatron_lora_enabled:
        return

    mode = generation_config.setdefault("lora_refit_mode", "native")
    if mode not in ("native", "merged"):
        raise ValueError(
            "LoRA training with vLLM requires "
            "policy.generation.lora_refit_mode to be 'native' or 'merged'."
        )
    if mode == "merged":
        return

    if generation_config["backend"] != "vllm":
        raise ValueError("Native LoRA refit is currently supported only by vLLM.")
    if not dtensor_lora_enabled or megatron_lora_enabled:
        raise ValueError(
            "Native LoRA refit currently requires the DTensor v2 LoRA policy. "
            "Set policy.generation.lora_refit_mode=merged for Megatron LoRA."
        )
    if not dtensor_config.get("_v2"):
        raise ValueError("Native LoRA refit requires policy.dtensor_cfg._v2=true.")
    if dtensor_lora_config.get("use_dora"):
        raise ValueError(
            "Native LoRA refit does not support use_dora=true because vLLM "
            "cannot apply the additional trainable magnitude vector."
        )
    if dtensor_lora_config.get("moe_rank_scaling"):
        raise ValueError(
            "Native LoRA refit does not support moe_rank_scaling=true because "
            "vLLM currently accepts one adapter rank and scaling factor."
        )

    automodel_kwargs = dtensor_config.get("automodel_kwargs")
    if automodel_kwargs is None or not automodel_kwargs.get("force_hf"):
        raise ValueError(
            "Native LoRA refit currently requires "
            "policy.dtensor_cfg.automodel_kwargs.force_hf=true so trainer LoRA "
            "tensor names follow the Hugging Face schema consumed by vLLM."
        )

    if generation_config.get("refit_transport") is not None:
        raise ValueError(
            "Native LoRA refit currently supports only the topology-default "
            "collective/non-colocated or CUDA-IPC/colocated transport. Set "
            "policy.generation.refit_transport=null, or opt into merged refit."
        )
    if generation_config["vllm_cfg"]["async_engine"]:
        raise ValueError(
            "Native LoRA refit does not yet support vllm_cfg.async_engine=true."
        )
    if generation_config["vllm_cfg"].get("expose_http_server"):
        raise ValueError(
            "Native LoRA refit does not yet support vLLM's exposed HTTP server "
            "because every request must carry the in-memory LoRARequest."
        )
    if generation_config.get("real_quant") or generation_config.get("quant_cfg"):
        raise ValueError(
            "Native LoRA refit does not yet support quantized rollout models."
        )

    vllm_kwargs = generation_config.setdefault("vllm_kwargs", {})
    speculative_config = vllm_kwargs.get("speculative_config")
    speculative_decoding_enabled = speculative_config is not None and not (
        isinstance(speculative_config, Mapping)
        and int(speculative_config.get("num_speculative_tokens", 0)) == 0
    )
    if speculative_decoding_enabled:
        raise ValueError("Native LoRA refit does not yet support speculative decoding.")
    if vllm_kwargs.get("enable_lora") is False:
        raise ValueError(
            "Native LoRA refit requires policy.generation.vllm_kwargs.enable_lora=true."
        )

    assert dtensor_lora_config is not None
    rank = int(dtensor_lora_config["dim"])
    alpha = int(dtensor_lora_config["alpha"])
    max_lora_rank = vllm_kwargs.get("max_lora_rank")
    if max_lora_rank is not None and int(max_lora_rank) < rank:
        raise ValueError(
            f"vllm_kwargs.max_lora_rank={max_lora_rank} is smaller than the "
            f"trainer LoRA rank {rank}."
        )

    precision = policy_config["precision"]
    lora_dtype_by_precision = {
        "bfloat16": "bfloat16",
        "bf16": "bfloat16",
        "float16": "float16",
        "fp16": "float16",
    }
    lora_dtype = lora_dtype_by_precision.get(precision)
    if lora_dtype is None:
        raise ValueError(
            "Native LoRA refit currently supports bfloat16 or float16 policy "
            f"precision, got {precision!r}."
        )
    rollout_precision = generation_config["vllm_cfg"].get("precision")
    if (
        not isinstance(rollout_precision, str)
        or lora_dtype_by_precision.get(rollout_precision) != lora_dtype
    ):
        raise ValueError(
            "Native LoRA refit requires matching policy and vLLM precision, got "
            f"policy={precision!r} and vLLM={rollout_precision!r}."
        )
    configured_lora_dtype = vllm_kwargs.get("lora_dtype")
    if configured_lora_dtype not in (None, "auto", lora_dtype):
        raise ValueError(
            f"vllm_kwargs.lora_dtype={configured_lora_dtype!r} does not match "
            f"policy precision {precision!r}."
        )

    vllm_kwargs["enable_lora"] = True
    vllm_kwargs.setdefault("max_lora_rank", rank)
    max_loras = int(vllm_kwargs.setdefault("max_loras", 1))
    if max_loras < 1:
        raise ValueError("vllm_kwargs.max_loras must be at least 1 for native LoRA.")
    max_cpu_loras = int(vllm_kwargs.setdefault("max_cpu_loras", max_loras))
    if max_cpu_loras < max_loras:
        raise ValueError(
            "vllm_kwargs.max_cpu_loras must be greater than or equal to "
            "vllm_kwargs.max_loras."
        )
    vllm_kwargs.setdefault("lora_dtype", lora_dtype)
    additional_config = dict(vllm_kwargs.get("additional_config") or {})
    additional_config[NATIVE_LORA_CONFIG_KEY] = {"rank": rank, "alpha": alpha}
    vllm_kwargs["additional_config"] = additional_config

    generation_config["vllm_cfg"]["load_format"] = "auto"
