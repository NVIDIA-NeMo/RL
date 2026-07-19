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

from pydantic import BaseModel, Field, NonNegativeInt, PositiveFloat, PositiveInt

from nemo_rl.models.generation.interfaces import GenerationConfig

VllmRefitTransportName = Literal["s3", "zmq"]


class VllmSpecificArgs(TypedDict):
    tensor_parallel_size: int
    pipeline_parallel_size: int
    expert_parallel_size: int
    gpu_memory_utilization: float
    max_model_len: int
    # Additional arguments for vLLM inserted by nemo rl based on the context of when vllm is used
    skip_tokenizer_init: bool
    async_engine: bool
    load_format: NotRequired[str]
    precision: NotRequired[str]
    # Use ModelOpt MXFP8 quantization when precision is fp8.
    is_mx: NotRequired[bool]
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


class VllmRefitConfig(BaseModel, extra="allow"):
    delta_compression: VllmDeltaCompressionConfig = Field(
        default_factory=VllmDeltaCompressionConfig
    )
    storage: VllmRefitStorageConfig = Field(default_factory=VllmRefitStorageConfig)
    baseline: VllmRefitBaselineConfig = Field(default_factory=VllmRefitBaselineConfig)
    tuning: VllmRefitTuningConfig = Field(default_factory=VllmRefitTuningConfig)
    verify_samples_per_payload: NonNegativeInt = 0
    request_timeout_s: PositiveFloat = 600.0


class VllmConfig(GenerationConfig):
    vllm_cfg: VllmSpecificArgs
    vllm_kwargs: NotRequired[dict[str, Any]]
    # Null uses NCCL; remote sparse refit supports S3 or ZeroMQ value planes.
    refit_transport: NotRequired[Literal["vllm_s3_sparse", "vllm_zmq_sparse"] | None]
    refit_cfg: NotRequired[VllmRefitConfig | None]

    # quantization config
    quant_cfg: NotRequired[str | None]
    # When set with ``quant_cfg``, initialize rollout vLLM with real ModelOpt
    # NVFP4 kernels and stream packed quantized weights instead of fake-quant
    # modules. This is intended for ModelOpt NVFP4 rollout experiments.
    real_quant: NotRequired[bool]
    real_quant_ignore: NotRequired[list[str]]


def normalize_vllm_refit_config(config: VllmConfig) -> VllmRefitConfig | None:
    """Resolve sparse-refit defaults into the generation config."""
    if config.get("refit_transport") is None:
        return None
    refit_config = VllmRefitConfig.model_validate(config.get("refit_cfg") or {})
    config["refit_cfg"] = refit_config
    return refit_config
