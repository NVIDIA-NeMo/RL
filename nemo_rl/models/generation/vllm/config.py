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
            instrumentation mode for isolating tokenizer effects. The
            recommended default is false.
        exact_incremental_tokenizer: Whether tokenizer-only mode keeps an exact
            backend session and re-encodes only the mutable prompt tail between
            full checkpoints. The recommended default is false.
        final_only_incremental_tokenizer: Whether exact tokenizer-only mode
            starts from an empty tool output while the command runs and then
            incrementally encodes only the authoritative final output. This
            removes the snapshot admission requirement. The recommended
            default is false.
        final_only_prefill: Whether final-only mode submits the exact
            empty-tool-output prompt to one speculative vLLM prefill request
            while the command runs. The request is closed after its stable
            prefix is cached; the authoritative final request remains
            unchanged. The recommended default is false.
        final_only_prefill_completion_grace_seconds: Maximum time to wait for
            an in-flight final-only prefill after its command finishes. Zero
            minimizes post-command overhead; a small positive value trades
            tail latency for higher admission. The recommended default is 0.0.
        prefix_seeded_start: Whether a final-only prefill session reuses the
            prior model call's authoritative tokens and encodes only the short
            chat-template suffix after its final EOS. Structural checks fall
            back to full tokenization. The recommended default is false while
            this optimization is being validated.
        prefill_after_admission: Whether a successfully admitted final-only
            prefill session remains open and prefills bucketed command-output
            snapshots while the command is still running. The authoritative
            final generation request remains unchanged. The recommended
            default is false while admission overhead is being measured.
        stable_first_snapshot_prefill: Whether deferred admission proves a
            stable token boundary in its first nonempty tool-output snapshot
            by comparing it with an empty-output rendering. This can prefill
            useful tool-output tokens one snapshot earlier. Structural checks
            fall back to the authoritative model prefix. The recommended
            default is false while this optimization is being validated.
        background_prefill_completion: Whether the first continuation prefill
            returns after engine enqueue and completes under server ownership
            while the tool keeps running. Final close reports completed and
            cancelled work without adding a completion grace. The recommended
            default is false while this optimization is being validated.
        background_prefill_priority: vLLM scheduler priority assigned only to
            background prefill requests. Foreground requests use priority zero,
            and larger values run later. Background completion automatically
            enables vLLM's priority scheduler. The recommended default is 1.
        stop_after_first_prefill_page: Whether continuation streaming stops
            requesting shell snapshots after vLLM schedules the first
            cache-page-aligned background prefill. The authoritative final
            tokenizer and generation requests remain unchanged. This bounds
            control-plane overhead while preserving the first reusable APC
            page. The recommended default is false while this optimization is
            being validated.
        same_request_final_decode: Whether a settled background prefill session
            appends the authoritative final prompt suffix and performs final
            generation in that same vLLM request. Incompatible, missing, or
            in-flight sessions fail open to the unchanged foreground request.
            The recommended default is false while this optimization is being
            validated.
        compact_request_context: Whether sequence-zero tokenizer requests
            register their immutable chat context so later sequence-numbered
            requests send only cumulative tool output. Both HTTP hops rebuild
            the exact request from bounded, expiring state. The recommended
            default is false while this optimization is being validated.
        incremental_tokenizer_checkpoint_interval: Number of ordered partial
            snapshots between full authoritative tokenizer checkpoints. A
            valid session may finalize from its incrementally constructed
            tokens without an additional checkpoint. The recommended default
            is 8.
        counterfactual_full_tokenizer_timing: Whether a checkpoint-free final
            incremental result is also passed through the authoritative full
            tokenizer for diagnostic timing and exact-token comparison. This
            duplicates work and must remain false in performance runs.
        max_sessions: Maximum number of concurrent prefill sessions per vLLM
            replica. The recommended default is 256.
        session_ttl_seconds: Idle lifetime of a prefill session before cleanup.
            The recommended default is 900 seconds.
        stability_margin_tokens: Number of tokens held behind the proven common
            prefix to protect tokenizer boundary changes. The recommended
            default is 8.
        min_chunk_chars: Minimum new shell-output characters before requesting
            another tokenization after the first growing snapshot. The
            recommended default is 512 because the first request captures most
            reusable history and later requests only add tool-output tokens.
        initial_chunk_chars: Minimum shell-output characters before the first
            growing prefill request. The recommended default is 256 so the
            typical SWE candidate crosses the first profitable cache-block
            boundary without sacrificing longer-output admission.
        snapshot_poll_interval_seconds: Interval between runtime reads of the
            current shell-output snapshot. This should not be lower than the
            producer's snapshot cadence. The recommended default is 0.1
            seconds; use 0.05 seconds to match OpenHands' current producer
            cadence when measuring admission coverage.
        snapshot_long_poll_timeout_seconds: Maximum server-side wait for a
            shell snapshot to reach the next character target. Long polling
            wakes immediately when the target is reached and avoids periodic
            HTTP polling. The recommended default is 1.0 second.
        event_driven_snapshot_wait: Whether shell-output updates wake matching
            snapshot requests immediately. Disable only for polling-baseline
            experiments or as a fail-open compatibility fallback.
        flush_interval_seconds: Maximum interval between eligible partial-output
            tokenizations. The recommended default is 0.25 seconds.
        request_timeout_seconds: Maximum duration of one streaming prefill HTTP
            request before the action falls back to normal execution. The
            recommended default is 60 seconds.
    """

    enabled: bool
    tokenizer_only: bool
    exact_incremental_tokenizer: NotRequired[bool]
    final_only_incremental_tokenizer: NotRequired[bool]
    final_only_prefill: NotRequired[bool]
    final_only_prefill_completion_grace_seconds: float
    prefix_seeded_start: NotRequired[bool]
    prefill_after_admission: NotRequired[bool]
    stable_first_snapshot_prefill: NotRequired[bool]
    background_prefill_completion: NotRequired[bool]
    background_prefill_priority: int
    stop_after_first_prefill_page: bool
    same_request_final_decode: NotRequired[bool]
    compact_request_context: NotRequired[bool]
    incremental_tokenizer_checkpoint_interval: NotRequired[int]
    counterfactual_full_tokenizer_timing: NotRequired[bool]
    max_sessions: int
    session_ttl_seconds: float
    stability_margin_tokens: int
    min_chunk_chars: int
    initial_chunk_chars: int
    snapshot_poll_interval_seconds: float
    snapshot_long_poll_timeout_seconds: float
    event_driven_snapshot_wait: NotRequired[bool]
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
