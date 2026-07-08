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

from nemo_rl.models.generation.interfaces import GenerationConfig


class StreamingToolCallConfig(TypedDict):
    """Configuration for resumable prefill during a running tool call.

    Attributes:
        enabled: Whether streaming tool-call prefill endpoints are enabled.
            The recommended default is false until the full environment path is
            configured.
        tokenizer_only: Whether to repeatedly tokenize cumulative partial tool
            output without creating a streaming prefill request. This is an
            instrumentation mode for isolating tokenizer effects, not a
            stateful incremental encoder; the recommended default is false.
        max_sessions: Maximum number of concurrent prefill sessions per vLLM
            replica. The recommended default is 256.
        session_ttl_seconds: Idle lifetime of a prefill session before cleanup.
            The recommended default is 900 seconds.
        stability_margin_tokens: Number of tokens held behind the proven common
            prefix to protect tokenizer boundary changes. The recommended
            default is 8.
        min_chunk_chars: Minimum new shell-output characters before requesting
            another tokenization. The recommended default is 256.
        snapshot_poll_interval_seconds: Interval between runtime reads of the
            current shell-output snapshot. This should not be lower than the
            producer's snapshot cadence. The recommended default is 0.1
            seconds; use 0.05 seconds to match OpenHands' current producer
            cadence when measuring admission coverage.
        flush_interval_seconds: Maximum interval between eligible partial-output
            tokenizations. The recommended default is 0.25 seconds.
        request_timeout_seconds: Maximum duration of one streaming prefill HTTP
            request before the action falls back to normal execution. The
            recommended default is 60 seconds.
    """

    enabled: bool
    tokenizer_only: bool
    max_sessions: int
    session_ttl_seconds: float
    stability_margin_tokens: int
    min_chunk_chars: int
    snapshot_poll_interval_seconds: float
    flush_interval_seconds: float
    request_timeout_seconds: float


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
    streaming_tool_call: NotRequired[StreamingToolCallConfig]


class VllmConfig(GenerationConfig):
    vllm_cfg: VllmSpecificArgs
    vllm_kwargs: NotRequired[dict[str, Any]]

    # quantization config
    quant_cfg: NotRequired[str | None]
    # When set with ``quant_cfg``, initialize rollout vLLM with real ModelOpt
    # NVFP4 kernels and stream packed quantized weights instead of fake-quant
    # modules. This is intended for ModelOpt NVFP4 rollout experiments.
    real_quant: NotRequired[bool]
    real_quant_ignore: NotRequired[list[str]]
