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

import asyncio
import copy
import gc
import logging
import threading
import time
import uuid
import warnings
from collections import OrderedDict
from collections.abc import Callable, Coroutine
from dataclasses import asdict
from typing import Any, AsyncGenerator, Optional, TypeVar, cast

import ray
import torch
import uvicorn
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import (
    DEFAULT_GENERATION_PORT_RANGE_HIGH,
    DEFAULT_GENERATION_PORT_RANGE_LOW,
    _get_free_port_local,
    _get_node_ip_local,
)
from nemo_rl.distributed.worker_group_utils import get_nsight_config_if_pattern_matches
from nemo_rl.models.generation.interfaces import (
    GenerationDatumSpec,
    GenerationOutputSpec,
    verify_right_padding,
)
from nemo_rl.models.generation.vllm.incremental_tokenizer import (
    ExactIncrementalTokenizerSessionManager,
    IncrementalTokenizerError,
    IncrementalTokenizerPrefixSeedError,
    IncrementalTokenizerStablePrefixError,
)
from nemo_rl.models.generation.vllm.cache_utils import writeback_vllm_cache
from nemo_rl.models.generation.vllm.streaming_tool_call import (
    StreamingToolCallError,
    StreamingToolCallFinalizationUnavailableError,
    StreamingToolCallPrefillManager,
    StreamingToolCallPrefixMismatchError,
    StreamingToolCallPromptTooLongError,
    StreamingToolCallSessionClosedError,
    StreamingToolCallSessionNotFoundError,
)
from nemo_rl.models.generation.vllm.utils import (
    attach_routed_experts_to_chat_response_choices,
    format_prompt_for_vllm_generation,
    model_dump_chat_response_with_routed_experts,
    pad_and_align_routed_expert_indices,
)
from nemo_rl.models.generation.vllm.vllm_worker import BaseVllmGenerationWorker

LOGGER = logging.getLogger(__name__)


StreamingToolCallManagerResult = TypeVar("StreamingToolCallManagerResult")


def _configure_background_prefill_scheduling(
    llm_kwargs: dict[str, Any],
    streaming_tool_call_config: dict[str, Any] | None,
) -> int:
    """Protect foreground generation with a lower-priority prefill class."""
    if streaming_tool_call_config is None or not streaming_tool_call_config.get(
        "background_prefill_completion"
    ):
        return 0

    background_prefill_priority = streaming_tool_call_config[
        "background_prefill_priority"
    ]
    if (
        isinstance(background_prefill_priority, bool)
        or not isinstance(background_prefill_priority, int)
        or background_prefill_priority <= 0
    ):
        raise ValueError(
            "background_prefill_priority must be a positive integer so foreground "
            "requests keep priority zero"
        )
    scheduling_policy = llm_kwargs.setdefault("scheduling_policy", "priority")
    if scheduling_policy != "priority":
        raise ValueError(
            "background prefill completion requires vLLM scheduling_policy='priority'"
        )
    return background_prefill_priority


class StreamingToolCallPrefillRequest(BaseModel):
    """Tokenized candidate prompt for a streaming prefill session."""

    session_id: str
    sequence_no: int
    prompt_token_ids: list[int]


class StreamingToolCallCloseRequest(BaseModel):
    """Authoritative final prompt for a streaming prefill session."""

    session_id: str
    final_prompt_token_ids: list[int]


class StreamingToolCallAbortRequest(BaseModel):
    """Request to abort a streaming prefill session."""

    session_id: str


def _materialize_message_tool_calls(messages: list[dict[str, Any]]) -> None:
    """Replace lazy tool-call iterators with lists in-place."""
    for message in messages:
        if message.get("tool_calls"):
            message["tool_calls"] = list(message["tool_calls"])


def _replace_prefix_tokens(
    tokenizer,
    model_prefix_token_ids: list[int],
    template_prefix_token_ids: list[int],
    template_token_ids: list[int],
) -> list[int]:
    """This is a subroutine used inside the vLLM Chat Completion server.

    This function is for fixing up the chat template-tokenized messages history
    to match the model output tokenization up to the last assistant turn,
    in order to preserve the monotonic tokens property for optimized multi-turn
    training.

    Some environments (namely NeMo-Gym) require an OpenAI compatible server
    endpoint rather than an inference engine handle. This is fine for the most
    part, but it may cause issues when the environment is used as a part of
    training.

    RL training frameworks train models on token IDs, but the OpenAI compatible
    server communicates in what is basically de-tokenized text. When multiple
    model calls are made to the OpenAI compatible server in a single trajectory,
    model generations in previous model calls may be re-tokenized to something
    that is different than what was generated. This is not too big of an issue
    (that we know of) at inference time, but the log probs the model produces
    are different enough for the differently re-tokenized generation result that
    it causes the training to be off policy. Off policy isn't necessarily a bad
    thing in isolation, but this source of off-policyness may cause unexpected
    issues if not properly accounted for. It also mis-aligns the token ID
    sequences across model calls, which feels very strange during training.

    There are real cases where the model output string _does not match_ the chat
    template tokenization of the parsed model output. A concrete example is
    inconsistent whitespace tokens around tool call special tokens.

    TODO When NeMo RL supports training image generation models, we want to
    revisit and possibly update this function. This issue occurs when the model
    generates tokens that are de-tokenized into text or images, and then
    re-tokenized into tokens. So if there is a situation like that with images
    and image tokenization is non-unique, then we will need to uppdate this
    function.

    Example (turn-by-turn, concise; eos_token_id = 2):
        Turn 1:
            - prefill_T1 (template prefill) = [11,12,13,40,41]
            - model output = [220,17,2]  # decodes to " 4" + EOS
            - model_prefix_token_ids = prefill_T1 + model output
              => [11,12,13,40,41,220,17,2]

        Turn 2 (template retokenizes prior assistant text differently):
            - template_prefix_token_ids = [11,12,13,40,41,1001,2]  # 1001 decodes to " 4"
            - template_token_ids = [11,12,13,40,41,1001,2,21,22,40,41]

        _replace_prefix_tokens keeps the exact prior model tokens up to EOS and
        resumes from the template after that EOS:
            output => [11,12,13,40,41,220,17,2,21,22,40,41]
    """
    if not model_prefix_token_ids:
        return template_token_ids

    eos_token_id = tokenizer.eos_token_id
    assert eos_token_id is not None, "Your tokenizer must have an EOS token ID!"

    model_cut_end = len(model_prefix_token_ids)
    if model_prefix_token_ids:
        # We are not always guaranteed that the model outputs an EOS token as the stop criteria of the previous model call e.g. when the model reaches max_tokens.
        # And since chat templates will always add one for us, we just cut the model input to right before the EOS token ID (if applicable)
        if model_prefix_token_ids[-1] == eos_token_id:
            model_cut_end -= 1

    # Assert here to prepare for the logic below
    assert len(template_token_ids) > len(
        template_prefix_token_ids
    ), f"""Found possibly non-monotonically increasing trajectory!
Template prefix token IDs (everything before the final assistant message): {template_prefix_token_ids}

Template token IDs (everything that was sent to the model endpoint): {template_token_ids}

Template prefix repr (detokenized): {repr(tokenizer.decode(template_prefix_token_ids))}

Template repr (detokenized): {repr(tokenizer.decode(template_token_ids))}
"""

    # We take everything starting with the EOS token ID.
    template_cut_start = -1
    for pos in reversed(range(len(template_prefix_token_ids))):
        if template_token_ids[pos] == eos_token_id:
            template_cut_start = pos
            break

    # This should never be the case, but
    assert (
        template_cut_start >= 0
    ), f"""No EOS token ID found in the chat-templated messages!
Template prefix token IDs (everything before the final assistant message): {template_prefix_token_ids}

Template token IDs (everything that was sent to the model endpoint): {template_token_ids}

Template prefix repr (detokenized): {repr(tokenizer.decode(template_prefix_token_ids))}

Template repr (detokenized): {repr(tokenizer.decode(template_token_ids))}"""

    return (
        model_prefix_token_ids[:model_cut_end] + template_token_ids[template_cut_start:]
    )


class VllmAsyncGenerationWorkerImpl(BaseVllmGenerationWorker):
    def __init__(
        self,
        config,
        bundle_indices=None,
        fraction_of_gpus: float = 1.0,
        seed=None,
        extra_env_vars: Optional[list[str]] = None,
        defer_model_load: bool = False,
    ):
        """Initialize an async vLLM worker.

        When defer_model_load=True, only stores config and reserves a port for
        the HTTP server (if expose_http_server is enabled). Call load_model()
        later to perform the heavy model loading. This enables overlapping vLLM
        model loading with NeMo Gym init.

        Args:
            config: Configuration dictionary for the policy
            bundle_indices: List of local bundle indices within a node for parallelism.
            fraction_of_gpus: Fraction of GPUs to use for this worker
            seed: Random seed for initialization
            extra_env_vars: Additional environment variable names to forward into
                          the vLLM worker subprocess.
            defer_model_load: If True, skip model loading and only reserve port
        """
        # Deferred-loading state. Always initialized so every instance has a
        # consistent set of attributes regardless of init path.
        self._reserved_socket = None
        self._reserved_port = None
        self._reserved_node_ip = None
        self._deferred_bundle_indices = None
        self._deferred_seed = None

        # Defaults for HTTP server state; overwritten by _create_engine()
        # when the worker is a model owner and the model is actually loaded.
        self.server_thread = None
        self.base_url = None
        self.http_server = None

        super().__init__(
            config,
            bundle_indices,
            fraction_of_gpus,
            seed,
            extra_env_vars,
            defer_model_load,
        )

        if not self.is_model_owner or not defer_model_load:
            return

        self._deferred_bundle_indices = bundle_indices
        self._deferred_seed = seed

        if self.cfg["vllm_cfg"].get("expose_http_server"):
            self._reserve_port()

        self.llm = None
        self.vllm_device_ids = None

    def _return_routed_experts_enabled(self) -> bool:
        engine_args = getattr(self, "llm_async_engine_args", None)
        if bool(getattr(engine_args, "enable_return_routed_experts", False)):
            return True
        return bool(
            self.cfg.get("vllm_kwargs", {}).get("enable_return_routed_experts", False)
        )

    def _reserve_port(self) -> None:
        """Bind and listen on a TCP socket to reserve a free port from the OS.

        The socket is held open in LISTENING state and later passed directly to
        uvicorn via the ``sockets=`` parameter in ``server.serve()``. The socket
        is never closed and re-opened, so there is zero gap where another process
        could steal the port.
        """
        import socket

        from nemo_rl.distributed.virtual_cluster import _get_node_ip_local

        self._reserved_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._reserved_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._reserved_socket.bind(("", 0))
        self._reserved_socket.listen(128)
        self._reserved_socket.setblocking(False)
        self._reserved_port = self._reserved_socket.getsockname()[1]
        self._reserved_node_ip = _get_node_ip_local()
        print(
            f"Reserved port {self._reserved_port} on {self._reserved_node_ip} "
            f"for vLLM HTTP server"
        )

    def load_model(self) -> None:
        """Load the vLLM model and create the engine.

        Called after a deferred init to perform the heavy model loading.
        """
        if not self.is_model_owner:
            return
        self._load_model(self._deferred_bundle_indices, self._deferred_seed)

    def _create_engine(self, llm_kwargs: dict[str, Any]) -> None:
        from vllm.config import CompilationConfig
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.v1.engine.async_llm import AsyncLLM
        from vllm.v1.metrics.loggers import PrometheusStatLogger

        streaming_tool_call_config = self.cfg["vllm_cfg"].get("streaming_tool_call")
        if streaming_tool_call_config and streaming_tool_call_config.get(
            "same_request_final_decode", False
        ):
            required_patches = {
                "streaming_session_max_tokens",
                "streaming_session_output_state",
                "streaming_session_priority",
            }
            available_patches = {
                name
                for name, available in getattr(
                    self, "vllm_patch_capabilities", {}
                ).items()
                if available
            }
            missing_patches = required_patches - available_patches
            if missing_patches:
                raise RuntimeError(
                    "same-request final decode requires compatible vLLM streaming "
                    f"session patches; missing {sorted(missing_patches)}"
                )
        background_prefill_priority = _configure_background_prefill_scheduling(
            llm_kwargs, streaming_tool_call_config
        )

        # Workaround: convert compilation_config dict to CompilationConfig object
        # since AsyncEngineArgs doesn't handle the dict-to-pydantic conversion.
        if llm_kwargs.get("compilation_config", None):
            compilation_config = dict(llm_kwargs["compilation_config"])
            # use_inductor was removed in vLLM v0.12+ (https://github.com/vllm-project/vllm/pull/29323)
            # and replaced by the `backend` field: use_inductor=True -> backend="" (inductor),
            # use_inductor=False -> backend="eager".
            if "use_inductor" in compilation_config:
                use_inductor = compilation_config.pop("use_inductor")
                if "backend" not in compilation_config:
                    compilation_config["backend"] = "" if use_inductor else "eager"
                warnings.warn(
                    "compilation_config.use_inductor is deprecated in vLLM v0.12+. "
                    "Use compilation_config.backend instead: "
                    "use_inductor=True -> backend='inductor', "
                    "use_inductor=False -> backend='eager'.",
                    DeprecationWarning,
                    stacklevel=1,
                )
            llm_kwargs["compilation_config"] = CompilationConfig(**compilation_config)

        self.llm_async_engine_args = AsyncEngineArgs(**llm_kwargs)
        self.stat_loggers = (
            [PrometheusStatLogger]
            if self.cfg["vllm_cfg"].get("enable_vllm_metrics_logger", False)
            else []
        )
        self.llm = AsyncLLM.from_engine_args(
            self.llm_async_engine_args, stat_loggers=self.stat_loggers
        )

        self.streaming_tool_call_manager = None
        if (
            streaming_tool_call_config is not None
            and streaming_tool_call_config["enabled"]
        ):
            from vllm import TokensPrompt
            from vllm.engine.protocol import StreamingInput
            from vllm.sampling_params import RequestOutputKind, SamplingParams

            prefill_sampling_params = SamplingParams(
                temperature=0,
                top_p=1,
                top_k=1,
                max_tokens=1,
                output_kind=RequestOutputKind.DELTA,
            )

            def generate_prefill(input_stream, request_id):
                return self.llm.generate(
                    prompt=input_stream,
                    sampling_params=prefill_sampling_params,
                    request_id=request_id,
                    priority=background_prefill_priority,
                )

            def make_streaming_input(token_ids):
                return StreamingInput(
                    prompt=TokensPrompt(prompt_token_ids=token_ids),
                    sampling_params=prefill_sampling_params,
                )

            def make_final_streaming_input(token_ids, sampling_params):
                return StreamingInput(
                    prompt=TokensPrompt(prompt_token_ids=token_ids),
                    sampling_params=sampling_params,
                    priority=0,
                )

            def count_output_tokens(output):
                return sum(
                    len(completion_output.token_ids)
                    for completion_output in output.outputs
                )

            self.streaming_tool_call_manager = StreamingToolCallPrefillManager(
                generate=generate_prefill,
                make_streaming_input=make_streaming_input,
                make_final_streaming_input=make_final_streaming_input,
                count_output_tokens=count_output_tokens,
                max_sessions=streaming_tool_call_config["max_sessions"],
                session_ttl_seconds=streaming_tool_call_config["session_ttl_seconds"],
                stability_margin_tokens=streaming_tool_call_config[
                    "stability_margin_tokens"
                ],
                # Each submitted streaming chunk produces one dummy decode token.
                max_prompt_tokens=self.llm_async_engine_args.max_model_len - 1,
                require_cache_page_crossing=not streaming_tool_call_config.get(
                    "same_request_final_decode", False
                ),
            )

        self.server_thread, self.base_url, self.http_server = None, None, None
        if self.cfg["vllm_cfg"].get("expose_http_server"):
            # Must run after AsyncLLM.from_engine_args and before
            # _setup_vllm_server spawns the uvicorn thread.
            self._install_engine_input_socket_lock()
            self.server_thread, self.base_url, self.http_server = (
                self._setup_vllm_server()
            )

        # vLLM Metrics Logger
        # Metrics logger only enabled for per-actor, model-owner only
        self._vllm_metrics_lock = threading.Lock()
        if self.cfg["vllm_cfg"].get("enable_vllm_metrics_logger", False):
            self._start_vllm_metrics_logger()

    def _install_engine_input_socket_lock(self) -> None:
        """Serialise sends on AsyncMPClient.input_socket across OS threads
        to prevent race conditions that block the vLLM engine (e.g. during
        in flight weight updates in async grpo).
        """
        shadow_sock = self.llm.engine_core.input_socket._shadow_sock

        lock = threading.Lock()
        original_send_multipart = shadow_sock.send_multipart

        def locked_send_multipart(*args: Any, **kwargs: Any) -> Any:
            with lock:
                return original_send_multipart(*args, **kwargs)

        # Replace the bound method on this socket instance only; other zmq
        # sockets in the process are unaffected.
        shadow_sock.send_multipart = locked_send_multipart  # type: ignore[assignment]

    def _start_vllm_metrics_logger(self) -> None:
        """Start a background thread that periodically collects vLLM logger metrics.

        Controlled by vllm_metrics_logger_interval (default: 0.5) in vllm_cfg.
        Runs only on the model-owner actor.
        """
        from vllm.v1.metrics.reader import Gauge, Counter, get_metrics_snapshot

        assert self.cfg["vllm_cfg"].get("async_engine", False), (
            "vLLM metrics logger is only supported with async engine enabled"
        )
        # Run only on the model-owner actor
        if not getattr(self, "is_model_owner", False):
            return

        assert "vllm_metrics_logger_interval" in self.cfg["vllm_cfg"], (
            "vllm_metrics_logger_interval must be set in vllm_cfg if enable_vllm_metrics_logger is True"
        )
        interval_s = self.cfg["vllm_cfg"]["vllm_metrics_logger_interval"]
        assert interval_s > 0, (
            f"vllm_metrics_logger_interval must be a positive float, got {interval_s}"
        )

        # Lazy import inside thread target to avoid import overhead if disabled
        stop_event = threading.Event()
        self._vllm_metrics_logger_stop_event = stop_event

        self._reset_vllm_logger_metrics()

        def _logger_loop():
            # Delay a little to let engine settle
            time.sleep(min(2.0, interval_s))
            while True:
                try:
                    for metric in get_metrics_snapshot():
                        with self._vllm_metrics_lock:
                            if isinstance(metric, Gauge):
                                self._record_vllm_gauge_metric(
                                    metric.name, metric.value
                                )
                            elif isinstance(metric, Counter):
                                self._record_vllm_counter_metric(
                                    metric.name, metric.value
                                )
                except Exception:
                    print(
                        "⚠️[vLLM Metric Logger] Exception in vLLM metrics logger",
                        flush=True,
                    )
                    pass
                time.sleep(interval_s)

        t = threading.Thread(
            target=_logger_loop, name="vllm-metrics-logger", daemon=True
        )
        t.start()
        self._vllm_metrics_logger_thread = t
        print(
            "📋[vLLM Metric Logger] vLLM metrics logger thread started",
            flush=True,
        )

    def _reset_vllm_logger_metrics(self) -> None:
        self.inflight_batch_sizes: list[int] = []
        self.num_pending_samples: list[int] = []
        self.kv_cache_usage_perc: list[float] = []
        self.generation_tokens: list[int] = []
        self.streaming_tool_call_dummy_tokens: list[int] = []
        self.streaming_tool_call_prefill_tokens: list[int] = []
        self.streaming_tool_call_prompt_too_long_rejections: list[int] = []
        self.prefix_cache_queries: list[int] = []
        self.prefix_cache_hits: list[int] = []

    def _record_vllm_gauge_metric(
        self, metric_name: str, metric_value: int | float
    ) -> None:
        if metric_name == "vllm:num_requests_running":
            self.inflight_batch_sizes.append(int(metric_value))
        elif metric_name == "vllm:num_requests_waiting":
            self.num_pending_samples.append(int(metric_value))
        elif metric_name == "vllm:kv_cache_usage_perc":
            self.kv_cache_usage_perc.append(float(metric_value))

    def _record_vllm_counter_metric(
        self, metric_name: str, metric_value: int | float
    ) -> None:
        if metric_name == "vllm:generation_tokens":
            manager = self.streaming_tool_call_manager
            dummy_tokens = manager.total_dummy_tokens if manager is not None else 0
            prefill_tokens = manager.total_prefill_tokens if manager is not None else 0
            prompt_too_long_rejections = (
                manager.total_prompt_too_long_rejections if manager is not None else 0
            )
            self.generation_tokens.append(max(0, int(metric_value) - dummy_tokens))
            self.streaming_tool_call_dummy_tokens.append(dummy_tokens)
            self.streaming_tool_call_prefill_tokens.append(prefill_tokens)
            self.streaming_tool_call_prompt_too_long_rejections.append(
                prompt_too_long_rejections
            )
        elif metric_name == "vllm:prefix_cache_queries":
            self.prefix_cache_queries.append(int(metric_value))
        elif metric_name == "vllm:prefix_cache_hits":
            self.prefix_cache_hits.append(int(metric_value))

    def get_vllm_logger_metrics(self) -> dict[str, Any]:
        if not self.cfg["vllm_cfg"].get("enable_vllm_metrics_logger", False):
            return {}

        with self._vllm_metrics_lock:
            metric = {
                "inflight_batch_sizes": copy.deepcopy(self.inflight_batch_sizes),
                "num_pending_samples": copy.deepcopy(self.num_pending_samples),
                "kv_cache_usage_perc": copy.deepcopy(self.kv_cache_usage_perc),
                "generation_tokens": copy.deepcopy(self.generation_tokens),
                "streaming_tool_call_dummy_tokens": copy.deepcopy(
                    self.streaming_tool_call_dummy_tokens
                ),
                "streaming_tool_call_prefill_tokens": copy.deepcopy(
                    self.streaming_tool_call_prefill_tokens
                ),
                "streaming_tool_call_prompt_too_long_rejections": copy.deepcopy(
                    self.streaming_tool_call_prompt_too_long_rejections
                ),
                "prefix_cache_queries": copy.deepcopy(self.prefix_cache_queries),
                "prefix_cache_hits": copy.deepcopy(self.prefix_cache_hits),
            }
        return metric

    def clear_vllm_logger_metrics(self) -> None:
        if not self.cfg["vllm_cfg"].get("enable_vllm_metrics_logger", False):
            return

        with self._vllm_metrics_lock:
            self._reset_vllm_logger_metrics()

    async def post_init_async(self):
        if self.llm is not None:
            await self.llm.collective_rpc("bind_numa", args=tuple())
            manager = self.streaming_tool_call_manager
            if manager is not None:
                block_sizes = await self.llm.collective_rpc(
                    "report_kv_cache_block_size", args=tuple()
                )
                unique_block_sizes = {int(block_size) for block_size in block_sizes}
                if len(unique_block_sizes) != 1:
                    raise RuntimeError(
                        "vLLM workers reported inconsistent KV cache block sizes: "
                        f"{sorted(unique_block_sizes)}"
                    )
                cache_page_size_tokens = unique_block_sizes.pop()
                manager.set_cache_page_size_tokens(cache_page_size_tokens)
                LOGGER.info(
                    "Streaming tool-call background prefill uses a %d-token APC page",
                    cache_page_size_tokens,
                )
        self.vllm_device_ids = await self.report_device_id_async()
        if self._mtp_load_from_disk:
            await self.llm.collective_rpc(
                "load_mtp_weights_from_disk", args=(self.model_name,)
            )

    async def get_reserved_url(self) -> Optional[str]:
        """Return the URL from the reserved socket, available before model loading."""
        if self._reserved_socket is not None:
            return f"http://{self._reserved_node_ip}:{self._reserved_port}/v1"
        return None

    async def report_dp_openai_server_base_url(self) -> Optional[str]:
        return self.base_url

    # ruff: noqa
    def _setup_vllm_openai_api_server(self, app: FastAPI) -> FastAPI:
        from copy import deepcopy
        from logging import Filter as LoggingFilter
        from logging import LogRecord
        from typing import ClassVar, List, Optional, Union

        from fastapi import Request
        from fastapi.responses import JSONResponse, StreamingResponse
        from vllm.entrypoints.openai.chat_completion.protocol import (
            ChatCompletionRequest,
            ChatCompletionResponse,
        )
        from vllm.entrypoints.openai.chat_completion.serving import (
            OpenAIServingChat,
        )
        from vllm.entrypoints.openai.engine.protocol import (
            ErrorResponse,
            RequestResponseMetadata,
        )
        from vllm.entrypoints.openai.models.protocol import BaseModelPath
        from vllm.entrypoints.openai.models.serving import OpenAIServingModels
        from vllm.entrypoints.utils import get_max_tokens
        from vllm.entrypoints.serve.tokenize.protocol import (
            TokenizeChatRequest,
            TokenizeCompletionRequest,
            TokenizeResponse,
        )
        from vllm.entrypoints.serve.render.serving import (
            OpenAIServingRender,
        )
        from vllm.entrypoints.serve.tokenize.serving import (
            OpenAIServingTokenization,
        )
        from vllm.exceptions import VLLMValidationError
        from vllm.reasoning.abs_reasoning_parsers import ReasoningParserManager
        from vllm.renderers import merge_kwargs
        from vllm.tool_parsers.abstract_tool_parser import ToolParserManager
        from vllm.utils.mistral import is_mistral_tokenizer, is_mistral_tool_parser
        from vllm.v1.engine.async_llm import logger as vllm_async_llm_logger
        from vllm.sampling_params import RequestOutputKind

        maybe_tool_parser_plugin = self.cfg["vllm_cfg"].get("tool_parser_plugin")
        if maybe_tool_parser_plugin:
            ToolParserManager.import_tool_parser(maybe_tool_parser_plugin)

        maybe_reasoning_parser_plugin = self.cfg["vllm_cfg"].get(
            "reasoning_parser_plugin"
        )
        if maybe_reasoning_parser_plugin:
            ReasoningParserManager.import_reasoning_parser(
                maybe_reasoning_parser_plugin
            )

        engine_client = self.llm
        model_config = self.llm_async_engine_args.create_model_config()
        base_model_paths = [
            BaseModelPath(
                name=model_config.served_model_name, model_path=model_config.model
            ),
            BaseModelPath(name=model_config.model, model_path=model_config.model),
        ]

        openai_serving_models_kwargs = dict(
            engine_client=engine_client,
            base_model_paths=base_model_paths,
            lora_modules=None,
        )
        openai_serving_models = OpenAIServingModels(**openai_serving_models_kwargs)

        class NeMoRLOpenAIChatRequestMixin:
            def model_post_init(self, context):
                # NeMo-Gym specific processing. This is just how NeMo-Gym returns the extra token information.
                if self.required_prefix_token_ids is None:
                    for message in reversed(self.messages):
                        if "prompt_token_ids" in message:
                            self.required_prefix_token_ids = (
                                message["prompt_token_ids"]
                                + message["generation_token_ids"]
                            )
                            break

                return super().model_post_init(context)

        class NeMoRLOpenAIServingMixin:
            @staticmethod
            def _set_max_tokens(request, max_tokens: int) -> None:
                """Set the request's max output tokens.

                Mutates the request in place. Handles both max_completion_tokens (newer OpenAI API)
                and max_tokens (deprecated but still supported by vLLM).
                """
                if request.max_completion_tokens is not None:
                    request.max_completion_tokens = max_tokens
                elif request.max_tokens is not None:
                    request.max_tokens = max_tokens

            def _clamp_max_tokens(
                self, request, request_max_tokens: int, prompt_token_ids: list[int]
            ) -> None:
                """Clamp the request's max output tokens so that input + output <= max_model_len."""
                remaining = self.model_config.max_model_len - len(prompt_token_ids)
                if remaining <= 0:
                    raise ValueError(
                        f"Prompt length ({len(prompt_token_ids)}) fills or exceeds "
                        f"max_model_len ({self.model_config.max_model_len}). "
                        f"No room for output tokens."
                    )
                max_tokens = min(request_max_tokens, remaining)
                self._set_max_tokens(request, max_tokens)

            async def _preprocess_chat_with_prompt_token_ids(
                self,
                request,
                messages,
                default_template,
                default_template_content_format,
                default_template_kwargs,
                tool_dicts,
                tool_parser,
                reasoning_parser,
                *,
                skip_mm_cache: bool,
            ):
                """Render chat metadata while reusing authoritative prompt tokens."""
                renderer = self.renderer
                mm_config = self.model_config.multimodal_config
                merged_template_kwargs = merge_kwargs(
                    default_template_kwargs,
                    dict(
                        tools=tool_dicts,
                        tokenize=is_mistral_tokenizer(renderer.tokenizer),
                    ),
                )
                tok_params = request.build_tok_params(self.model_config)
                chat_params = request.build_chat_params(
                    default_template, default_template_content_format
                ).with_defaults(
                    merged_template_kwargs,
                    default_media_io_kwargs=(
                        mm_config.media_io_kwargs if mm_config else None
                    ),
                    default_mm_processor_kwargs=getattr(
                        request, "mm_processor_kwargs", None
                    ),
                )

                conversation, prompt = await renderer.render_messages_async(
                    messages, chat_params
                )
                prompt["prompt_token_ids"] = list(
                    request.required_full_prompt_token_ids
                )
                tokenized_prompt = await renderer.tokenize_prompt_async(
                    prompt, tok_params
                )
                renderer._apply_prompt_extras(
                    [tokenized_prompt],
                    {
                        key: value
                        for key in ("mm_processor_kwargs", "cache_salt")
                        if (value := getattr(request, key, None)) is not None
                    },
                )
                engine_prompt = await renderer.process_for_engine_async(
                    tokenized_prompt,
                    time.time(),
                    skip_mm_cache=skip_mm_cache,
                )

                if reasoning_parser is not None:
                    tokenizer = renderer.get_tokenizer()
                    request = reasoning_parser(
                        tokenizer,
                        model_config=self.model_config,
                        chat_template_kwargs=chat_params.chat_template_kwargs,
                    ).adjust_request(request=request)

                if tool_parser is not None:
                    tool_choice = getattr(request, "tool_choice", "none")
                    tokenizer = renderer.get_tokenizer()
                    is_mistral_grammar_eligible = (
                        is_mistral_tool_parser(tool_parser)
                        and is_mistral_tokenizer(tokenizer)
                        and tokenizer.supports_grammar
                    )
                    if tool_choice != "none" or is_mistral_grammar_eligible:
                        tool_parser(tokenizer, request.tools).adjust_request(
                            request=request
                        )

                return conversation, [engine_prompt]

            # vLLM 0.20 moved chat preprocessing from
            # OpenAIServing._preprocess_chat to OpenAIServingRender.preprocess_chat,
            # so this override now applies via the render subclass.
            async def preprocess_chat(
                self,
                request,
                messages,
                default_template,
                default_template_content_format,
                default_template_kwargs,
                tool_dicts=None,
                tool_parser=None,
                reasoning_parser=None,
                *,
                skip_mm_cache: bool = False,
            ):
                # Materialize the message tool calls so we can deepcopy below.
                _materialize_message_tool_calls(messages)

                messages_for_replace_prefix_tokens = deepcopy(messages)

                # Temporarily set to 1 so vLLM's pre-tokenization length check passes;
                # the actual value will be set through _clamp_max_tokens later.
                actual_request_max_tokens = None
                if isinstance(request, NeMoRLChatCompletionRequest):
                    actual_request_max_tokens = (
                        request.max_completion_tokens
                        if request.max_completion_tokens is not None
                        else request.max_tokens
                    )
                    # If max_completion_tokens or max_tokens is not set, we don't need to do _clamp_max_tokens.
                    # So we don't need to set the request's max output tokens to 1 here.
                    if actual_request_max_tokens is not None:
                        self._set_max_tokens(request, 1)

                if (
                    isinstance(request, NeMoRLChatCompletionRequest)
                    and request.required_full_prompt_token_ids is not None
                ):
                    res = await self._preprocess_chat_with_prompt_token_ids(
                        request=request,
                        messages=messages,
                        default_template=default_template,
                        default_template_content_format=(
                            default_template_content_format
                        ),
                        default_template_kwargs=default_template_kwargs,
                        tool_dicts=tool_dicts,
                        tool_parser=tool_parser,
                        reasoning_parser=reasoning_parser,
                        skip_mm_cache=skip_mm_cache,
                    )
                    if actual_request_max_tokens is not None:
                        self._clamp_max_tokens(
                            request,
                            actual_request_max_tokens,
                            request.required_full_prompt_token_ids,
                        )
                    return res

                try:
                    res = await super().preprocess_chat(
                        request=request,
                        messages=messages,
                        default_template=default_template,
                        default_template_content_format=default_template_content_format,
                        default_template_kwargs=default_template_kwargs,
                        tool_dicts=tool_dicts,
                        tool_parser=tool_parser,
                        reasoning_parser=reasoning_parser,
                        skip_mm_cache=skip_mm_cache,
                    )
                except (ValueError, VLLMValidationError) as e:
                    if "maximum context length" in str(e):
                        import logging

                        logging.getLogger(__name__).warning(
                            "Prompt exceeds max_model_len: %s", e
                        )
                    raise

                if (
                    not hasattr(request, "required_prefix_token_ids")
                    or request.required_prefix_token_ids is None
                ):
                    # Clamp the request's max output tokens so that input + output <= max_model_len.
                    if actual_request_max_tokens is not None:
                        self._clamp_max_tokens(
                            request,
                            actual_request_max_tokens,
                            res[1][0]["prompt_token_ids"],
                        )
                    return res

                last_assistant_message_idx = None
                for i in reversed(range(len(messages_for_replace_prefix_tokens))):
                    if messages_for_replace_prefix_tokens[i]["role"] == "assistant":
                        last_assistant_message_idx = i
                        break

                if last_assistant_message_idx is None:
                    messages_to_last_assistant_message = (
                        messages_for_replace_prefix_tokens
                    )
                else:
                    messages_to_last_assistant_message = (
                        messages_for_replace_prefix_tokens[
                            : last_assistant_message_idx + 1
                        ]
                    )

                modified_request = request.model_copy(
                    update={"add_generation_prompt": False}
                )

                corresponding_res = await super().preprocess_chat(
                    request=modified_request,
                    messages=messages_to_last_assistant_message,
                    default_template=default_template,
                    default_template_content_format=default_template_content_format,
                    default_template_kwargs=default_template_kwargs,
                    tool_dicts=tool_dicts,
                    tool_parser=tool_parser,
                    reasoning_parser=reasoning_parser,
                    skip_mm_cache=skip_mm_cache,
                )
                actual_corresponding_token_ids = corresponding_res[1][0][
                    "prompt_token_ids"
                ]

                engine_prompt = res[1][0]

                final_prompt_token_ids = _replace_prefix_tokens(
                    tokenizer=self.renderer.tokenizer,
                    model_prefix_token_ids=request.required_prefix_token_ids,
                    template_prefix_token_ids=actual_corresponding_token_ids,
                    template_token_ids=engine_prompt["prompt_token_ids"],
                )

                engine_prompt["prompt_token_ids"] = final_prompt_token_ids

                # Clamp after prefix replacement since the prompt length may have changed.
                if actual_request_max_tokens is not None:
                    self._clamp_max_tokens(
                        request,
                        actual_request_max_tokens,
                        final_prompt_token_ids,
                    )

                return res

        ########################################
        # /v1/chat/completions endpoint
        ########################################

        # This MRO is necessary i.e. NeMoRLOpenAIChatRequestMixin > ChatCompletionRequest
        class NeMoRLChatCompletionRequest(
            NeMoRLOpenAIChatRequestMixin, ChatCompletionRequest
        ):
            required_prefix_token_ids: Optional[List[int]] = None
            required_full_prompt_token_ids: Optional[List[int]] = None
            streaming_tool_call_session_id: Optional[str] = None

        # vLLM 0.20 routes both /v1/chat/completions and /tokenize through
        # OpenAIServingRender.preprocess_chat, so the prefix-token override
        # belongs on the render subclass.
        worker_self = self

        class NeMoRLOpenAIServingChatMixin:
            async def create_chat_completion_from_streaming_tool_call(
                self,
                request,
                raw_request,
                *,
                manager,
                session_id,
            ):
                """Format one same-request final decode through vLLM OpenAI APIs."""
                if (
                    request.stream
                    or request.use_beam_search
                    or request.n
                    not in (
                        None,
                        1,
                    )
                ):
                    raise StreamingToolCallFinalizationUnavailableError(
                        "same-request final decode requires non-streaming n=1 sampling"
                    )

                tokenizer = self.renderer.tokenizer
                assert tokenizer is not None
                chat_template_kwargs = self._effective_chat_template_kwargs(request)
                reasoning_parser = None
                if self.reasoning_parser_cls:
                    reasoning_parser = self.reasoning_parser_cls(
                        tokenizer,
                        chat_template_kwargs=chat_template_kwargs,
                    )

                rendered = await self.render_chat_request(request)
                if isinstance(rendered, ErrorResponse):
                    return rendered
                conversation, engine_inputs = rendered
                if len(engine_inputs) != 1:
                    raise StreamingToolCallFinalizationUnavailableError(
                        "same-request final decode requires exactly one rendered prompt"
                    )
                engine_input = engine_inputs[0]
                prompt_token_ids = self._extract_prompt_components(
                    engine_input
                ).token_ids
                if prompt_token_ids is None:
                    raise StreamingToolCallFinalizationUnavailableError(
                        "same-request final decode requires prompt token IDs"
                    )

                lora_request = self._maybe_get_adapters(
                    request,
                    supports_default_mm_loras=True,
                )
                if lora_request is not None:
                    raise StreamingToolCallFinalizationUnavailableError(
                        "same-request final decode does not support LoRA adapters"
                    )
                if self._get_data_parallel_rank(raw_request) is not None:
                    raise StreamingToolCallFinalizationUnavailableError(
                        "same-request final decode does not support routed DP ranks"
                    )

                max_tokens = get_max_tokens(
                    self.model_config.max_model_len,
                    (
                        request.max_completion_tokens
                        if request.max_completion_tokens is not None
                        else request.max_tokens
                    ),
                    self._extract_prompt_len(engine_input),
                    self.default_sampling_params,
                    self.override_max_tokens,
                )
                sampling_params = request.to_sampling_params(
                    max_tokens,
                    self.default_sampling_params,
                )
                if sampling_params.stop or sampling_params.structured_outputs:
                    raise StreamingToolCallFinalizationUnavailableError(
                        "same-request final decode does not support stop strings or "
                        "structured output"
                    )
                sampling_params.output_kind = RequestOutputKind.DELTA

                result_generator = await manager.finalize(
                    session_id=session_id,
                    final_prompt_token_ids=list(prompt_token_ids),
                    final_sampling_params=sampling_params,
                )

                async def aggregate_final_outputs():
                    final_output = None
                    async for output in result_generator:
                        if not output.outputs:
                            continue
                        if final_output is None:
                            final_output = output
                        else:
                            final_output.add(output, aggregate=True)
                            final_output.num_cached_tokens = output.num_cached_tokens
                            final_output.metrics = output.metrics
                            final_output.kv_transfer_params = output.kv_transfer_params
                    if final_output is not None:
                        final_output.prompt_token_ids = list(prompt_token_ids)
                        yield final_output

                request_id = (
                    f"chatcmpl-{self._base_request_id(raw_request, request.request_id)}"
                )
                request_metadata = RequestResponseMetadata(request_id=request_id)
                if raw_request:
                    raw_request.state.request_metadata = request_metadata
                model_name = self.models.model_name(lora_request)
                response = await self.chat_completion_full_generator(
                    request,
                    aggregate_final_outputs(),
                    request_id,
                    model_name,
                    conversation,
                    tokenizer,
                    request_metadata,
                    reasoning_parser,
                )
                if isinstance(response, ErrorResponse) and response.error.code >= 500:
                    raise StreamingToolCallSessionClosedError(
                        "same-request final decode produced no usable output"
                    )
                return response

            async def chat_completion_full_generator(
                self,
                request,
                result_generator,
                *args,
                **kwargs,
            ):
                final_res = None

                async def capture_result_generator():
                    nonlocal final_res
                    async for res in result_generator:
                        final_res = res
                        yield res

                response = await super().chat_completion_full_generator(
                    request,
                    capture_result_generator(),
                    *args,
                    **kwargs,
                )
                if (
                    not worker_self._return_routed_experts_enabled()
                    or not isinstance(response, ChatCompletionResponse)
                    or final_res is None
                ):
                    return response

                return attach_routed_experts_to_chat_response_choices(
                    response,
                    final_res,
                    device=torch.device("cpu"),
                    logger=LOGGER,
                    routed_experts_dtype=worker_self.routed_experts_dtype,
                )

        class NeMoRLOpenAIServingChat(NeMoRLOpenAIServingChatMixin, OpenAIServingChat):
            pass

        class NeMoRLOpenAIServingRender(NeMoRLOpenAIServingMixin, OpenAIServingRender):
            pass

        serving_chat_default_kwargs = dict(
            response_role="assistant",
            request_logger=None,
            chat_template=None,
            chat_template_content_format="auto",
            enable_auto_tools=True,
        )
        serving_chat_kwargs = serving_chat_default_kwargs | self.cfg["vllm_cfg"].get(
            "http_server_serving_chat_kwargs", dict()
        )
        openai_serving_render = NeMoRLOpenAIServingRender(
            model_config=engine_client.model_config,
            renderer=engine_client.renderer,
            model_registry=openai_serving_models.registry,
            request_logger=serving_chat_kwargs["request_logger"],
            chat_template=serving_chat_kwargs["chat_template"],
            chat_template_content_format=serving_chat_kwargs[
                "chat_template_content_format"
            ],
            enable_auto_tools=serving_chat_kwargs["enable_auto_tools"],
        )
        serving_chat_kwargs.update(
            dict(
                engine_client=engine_client,
                models=openai_serving_models,
                openai_serving_render=openai_serving_render,
                return_tokens_as_token_ids=True,
            )
        )
        openai_serving_chat = NeMoRLOpenAIServingChat(**serving_chat_kwargs)

        generation_config = self.cfg

        # The create_chat_completion and tokenize methods are taken from vllm/entrypoints/openai/api_server.py
        @app.post("/v1/chat/completions")
        async def create_chat_completion(
            request: NeMoRLChatCompletionRequest, raw_request: Request
        ):
            # This needs to match the behavior in nemo_rl/models/generation/vllm/vllm_worker.py::BaseVllmGenerationWorker::_build_sampling_params
            # Right now we explicitly assert set this to -1.
            assert request.top_k in (None, -1), (
                f"Top k sampling parameter must be unset, empty, or -1. Got `{request.top_k}`"
            )
            request.top_k = -1

            # The request sampling params need to exactly match those as are set in NeMo RL.
            # If they do not match, the inference will be off policy and destroy training stability.
            assert request.temperature == generation_config["temperature"]
            assert request.top_p == generation_config["top_p"]

            same_request_status = None
            same_request_session_id = request.streaming_tool_call_session_id
            same_request_enabled = bool(
                same_request_session_id
                and streaming_config
                and streaming_config.get("same_request_final_decode", False)
            )
            try:
                if same_request_enabled:
                    assert same_request_session_id is not None
                    manager = get_streaming_tool_call_manager()
                    fallback_request = request.model_copy(deep=True)
                    try:
                        generator = await openai_serving_chat.create_chat_completion_from_streaming_tool_call(
                            request,
                            raw_request,
                            manager=manager,
                            session_id=same_request_session_id,
                        )
                    except StreamingToolCallError as error:
                        await manager.abort(session_id=same_request_session_id)
                        request = fallback_request
                        request.streaming_tool_call_session_id = None
                        if isinstance(
                            error,
                            StreamingToolCallSessionNotFoundError,
                        ):
                            same_request_status = "fallback_missing"
                        elif isinstance(
                            error,
                            (
                                StreamingToolCallFinalizationUnavailableError,
                                StreamingToolCallPrefixMismatchError,
                            ),
                        ):
                            same_request_status = "fallback_incompatible"
                        elif isinstance(
                            error,
                            StreamingToolCallSessionClosedError,
                        ):
                            same_request_status = "fallback_engine_error"
                        else:
                            same_request_status = "fallback_error"
                        LOGGER.warning(
                            "Same-request final decode fell back (%s): %s",
                            same_request_status,
                            error,
                        )
                        generator = await openai_serving_chat.create_chat_completion(
                            request,
                            raw_request,
                        )
                    except Exception as error:
                        await manager.abort(session_id=same_request_session_id)
                        request = fallback_request
                        request.streaming_tool_call_session_id = None
                        same_request_status = "fallback_error"
                        LOGGER.exception(
                            "Same-request final decode failed open after an "
                            "unexpected integration error: %s",
                            error,
                        )
                        generator = await openai_serving_chat.create_chat_completion(
                            request,
                            raw_request,
                        )
                    else:
                        same_request_status = "used"
                        if isinstance(generator, ErrorResponse):
                            await manager.abort(session_id=same_request_session_id)
                else:
                    generator = await openai_serving_chat.create_chat_completion(
                        request, raw_request
                    )
            except VLLMValidationError as e:
                # vLLM 0.20 raises VLLMValidationError for prompts exceeding
                # max_model_len during tokenization, instead of returning an
                # ErrorResponse. Convert to HTTP 400 so the Gym proxy can
                # detect context-length overflow and handle it gracefully.
                return JSONResponse(
                    content={
                        "error": {
                            "message": str(e),
                            "type": "invalid_request_error",
                            "code": 400,
                        }
                    },
                    status_code=400,
                )

            if isinstance(generator, ErrorResponse):
                return JSONResponse(
                    content=generator.model_dump(), status_code=generator.error.code
                )

            elif isinstance(generator, ChatCompletionResponse):
                response_content = model_dump_chat_response_with_routed_experts(
                    generator
                )
                if same_request_status is not None:
                    response_content["streaming_tool_call_same_request_status"] = (
                        same_request_status
                    )
                return JSONResponse(content=response_content)

            return StreamingResponse(content=generator, media_type="text/event-stream")

        ########################################
        # /tokenize endpoint
        ########################################

        # This MRO is necessary i.e. NeMoRLOpenAIChatRequestMixin > TokenizeRequest
        class NeMoRLTokenizeChatRequest(
            NeMoRLOpenAIChatRequestMixin, TokenizeChatRequest
        ):
            # OpenAIBaseModel caches field names on the class. Reset the
            # inherited cache so this subclass's extension is not logged as an
            # ignored request field on every tokenizer call.
            field_names: ClassVar[set[str] | None] = None
            required_prefix_token_ids: Optional[List[int]] = None

        class NeMoRLIncrementalTokenizeChatRequest(NeMoRLTokenizeChatRequest):
            # NeMoRLTokenizeChatRequest can populate its own cache before the
            # incremental endpoint receives a request, so this subclass needs
            # an independent cache as well.
            field_names: ClassVar[set[str] | None] = None
            session_id: str
            sequence_no: int
            final: bool = False
            return_tokens: bool = False
            prefill: bool = False
            prefill_continuation: bool = False
            prefill_from_required_prefix: bool = False
            finalize_from_required_prefix: bool = False
            compact_context: bool = False
            max_contexts: Optional[int] = None
            context_ttl_seconds: Optional[float] = None

        class NeMoRLCompactIncrementalTokenizeRequest(BaseModel):
            session_id: str
            sequence_no: int
            tool_output: str
            final: bool = False
            prefill: bool = False
            prefill_continuation: bool = False
            prefill_from_required_prefix: bool = False
            finalize_from_required_prefix: bool = False

        NeMoRLTokenizeRequest = Union[
            TokenizeCompletionRequest, NeMoRLTokenizeChatRequest
        ]

        # Tokenize path delegates to OpenAIServingRender.preprocess_chat in
        # vLLM 0.20, where the prefix-token override lives.
        class NeMoRLOpenAIServingTokenization(OpenAIServingTokenization):
            pass

        serving_tokenization_kwargs = dict(
            request_logger=serving_chat_kwargs["request_logger"],
            chat_template=serving_chat_kwargs["chat_template"],
            chat_template_content_format=serving_chat_kwargs[
                "chat_template_content_format"
            ],
            engine_client=serving_chat_kwargs["engine_client"],
            models=serving_chat_kwargs["models"],
            openai_serving_render=openai_serving_render,
        )
        openai_serving_tokenization = NeMoRLOpenAIServingTokenization(
            **serving_tokenization_kwargs
        )

        streaming_config = self.cfg["vllm_cfg"].get("streaming_tool_call")
        incremental_tokenizer_manager = None
        if streaming_config and streaming_config.get("exact_incremental_tokenizer"):
            incremental_tokenizer_manager = ExactIncrementalTokenizerSessionManager(
                tokenizer=openai_serving_tokenization.renderer.get_tokenizer(),
                max_sessions=streaming_config["max_sessions"],
                session_ttl_seconds=streaming_config["session_ttl_seconds"],
                checkpoint_interval=streaming_config[
                    "incremental_tokenizer_checkpoint_interval"
                ],
            )
        compact_request_context_enabled = bool(
            streaming_config and streaming_config.get("compact_request_context")
        )
        incremental_request_contexts: OrderedDict[
            str, tuple[NeMoRLIncrementalTokenizeChatRequest, float, float]
        ] = OrderedDict()

        def expire_incremental_request_contexts() -> None:
            now = time.time()
            expired_session_ids = [
                session_id
                for session_id, (_, _, expires_at) in (
                    incremental_request_contexts.items()
                )
                if expires_at <= now
            ]
            for session_id in expired_session_ids:
                incremental_request_contexts.pop(session_id, None)

        def store_incremental_request_context(
            request: NeMoRLIncrementalTokenizeChatRequest,
        ) -> None:
            max_contexts = request.max_contexts
            context_ttl_seconds = request.context_ttl_seconds
            if (
                max_contexts is None
                or max_contexts <= 0
                or context_ttl_seconds is None
                or context_ttl_seconds <= 0
            ):
                raise HTTPException(
                    status_code=422,
                    detail=(
                        "compact sequence-zero requests require positive context limits"
                    ),
                )
            expire_incremental_request_contexts()
            incremental_request_contexts.pop(request.session_id, None)
            incremental_request_contexts[request.session_id] = (
                request,
                context_ttl_seconds,
                time.time() + context_ttl_seconds,
            )
            while len(incremental_request_contexts) > max_contexts:
                incremental_request_contexts.popitem(last=False)

        def get_incremental_request_context(
            session_id: str,
        ) -> NeMoRLIncrementalTokenizeChatRequest:
            expire_incremental_request_contexts()
            context_entry = incremental_request_contexts.get(session_id)
            if context_entry is None:
                raise HTTPException(
                    status_code=409,
                    detail="compact tokenization context is missing or expired",
                )
            request_template, context_ttl_seconds, _ = context_entry
            incremental_request_contexts[session_id] = (
                request_template,
                context_ttl_seconds,
                time.time() + context_ttl_seconds,
            )
            incremental_request_contexts.move_to_end(session_id)
            return request_template

        @app.post("/tokenize")
        async def tokenize(request: NeMoRLTokenizeRequest, raw_request: Request):
            generator = await openai_serving_tokenization.create_tokenize(
                request, raw_request
            )

            if isinstance(generator, ErrorResponse):
                return JSONResponse(
                    content=generator.model_dump(), status_code=generator.error.code
                )
            elif isinstance(generator, TokenizeResponse):
                return JSONResponse(content=generator.model_dump())

        async def get_authoritative_token_ids(
            request: NeMoRLIncrementalTokenizeChatRequest,
            raw_request: Request,
        ) -> list[int] | JSONResponse:
            generator = await openai_serving_tokenization.create_tokenize(
                request, raw_request
            )
            if isinstance(generator, ErrorResponse):
                return JSONResponse(
                    content=generator.model_dump(),
                    status_code=generator.error.code,
                )
            return list(generator.tokens)

        async def render_incremental_prompt(
            request: NeMoRLIncrementalTokenizeChatRequest,
        ) -> str:
            tool_dicts = (
                None
                if request.tools is None
                else [tool.model_dump() for tool in request.tools]
            )
            merged_template_kwargs = merge_kwargs(
                None,
                dict(
                    tools=tool_dicts,
                    tokenize=is_mistral_tokenizer(
                        openai_serving_tokenization.renderer.tokenizer
                    ),
                ),
            )
            chat_params = request.build_chat_params(
                openai_serving_tokenization.chat_template,
                openai_serving_tokenization.chat_template_content_format,
            ).with_defaults(merged_template_kwargs)
            (
                _,
                prompt,
            ) = await openai_serving_tokenization.renderer.render_messages_async(
                request.messages,
                chat_params,
            )
            rendered_prompt = prompt.get("prompt")
            if not isinstance(rendered_prompt, str):
                raise HTTPException(
                    status_code=422,
                    detail="incremental tokenization requires a text prompt",
                )
            return rendered_prompt

        async def render_incremental_prefix_prompt(
            request: NeMoRLIncrementalTokenizeChatRequest,
        ) -> str:
            last_assistant_message_idx = next(
                (
                    index
                    for index in reversed(range(len(request.messages)))
                    if request.messages[index].get("role") == "assistant"
                ),
                None,
            )
            if last_assistant_message_idx is None:
                raise IncrementalTokenizerPrefixSeedError(
                    "the incremental prompt has no assistant token prefix"
                )
            prefix_request = request.model_copy(
                update={
                    "messages": deepcopy(
                        request.messages[: last_assistant_message_idx + 1]
                    ),
                    "add_generation_prompt": False,
                }
            )
            return await render_incremental_prompt(prefix_request)

        async def render_incremental_empty_tool_prompt(
            request: NeMoRLIncrementalTokenizeChatRequest,
        ) -> str:
            messages = deepcopy(request.messages)
            last_tool_message_idx = next(
                (
                    index
                    for index in reversed(range(len(messages)))
                    if messages[index].get("role") == "tool"
                ),
                None,
            )
            if last_tool_message_idx is None:
                raise IncrementalTokenizerStablePrefixError(
                    "the incremental prompt has no trailing tool output"
                )
            messages[last_tool_message_idx] = {
                **messages[last_tool_message_idx],
                "content": "",
            }
            empty_tool_request = request.model_copy(update={"messages": messages})
            return await render_incremental_prompt(empty_tool_request)

        @app.post("/incremental_tokenize")
        async def incremental_tokenize(
            request: NeMoRLIncrementalTokenizeChatRequest,
            raw_request: Request,
        ):
            request_handler_start = time.perf_counter()
            if request.compact_context:
                if not compact_request_context_enabled:
                    raise HTTPException(
                        status_code=503,
                        detail="compact request context is not enabled",
                    )
                if request.sequence_no != 0:
                    raise HTTPException(
                        status_code=422,
                        detail=("full compact-context requests require sequence zero"),
                    )
                if not request.final and (
                    request.max_contexts is None
                    or request.max_contexts <= 0
                    or request.context_ttl_seconds is None
                    or request.context_ttl_seconds <= 0
                ):
                    raise HTTPException(
                        status_code=422,
                        detail=(
                            "compact sequence-zero requests require positive "
                            "context limits"
                        ),
                    )
            if request.prefill_continuation and not request.prefill:
                raise HTTPException(
                    status_code=422,
                    detail="prefill continuation requires prefill",
                )
            if request.prefill_from_required_prefix and (
                not request.prefill
                or not request.prefill_continuation
                or request.sequence_no != 0
                or request.final
            ):
                raise HTTPException(
                    status_code=422,
                    detail=(
                        "prefill from required prefix requires a non-final "
                        "sequence-zero prefill continuation"
                    ),
                )
            if request.finalize_from_required_prefix and (
                not request.final or request.sequence_no != 0 or request.prefill
            ):
                raise HTTPException(
                    status_code=422,
                    detail=(
                        "finalization from required prefix requires a final "
                        "sequence-zero request without prefill"
                    ),
                )
            if (
                request.prefill_from_required_prefix
                or request.finalize_from_required_prefix
            ) and request.required_prefix_token_ids is None:
                raise HTTPException(
                    status_code=422,
                    detail="an authoritative required token prefix is absent",
                )
            if (
                request.prefill
                and not request.prefill_continuation
                and (request.sequence_no != 0 or request.final)
            ):
                raise HTTPException(
                    status_code=422,
                    detail="prefill requires a non-final sequence-zero request",
                )
            if request.prefill and request.sequence_no == 0 and request.final:
                raise HTTPException(
                    status_code=422,
                    detail="prefill cannot start with a final request",
                )
            manager = incremental_tokenizer_manager
            if manager is None:
                raise HTTPException(
                    status_code=503,
                    detail="exact incremental tokenization is not enabled",
                )
            prefix_seed_attempts = 0
            prefix_seed_successes = 0
            prefix_seed_fallbacks = 0
            prefix_seed_seconds = 0.0
            prefix_seed_fallback_reason = None
            stable_first_snapshot_prefill_attempts = 0
            stable_first_snapshot_prefill_successes = 0
            stable_first_snapshot_prefill_fallbacks = 0
            stable_first_snapshot_prefill_seconds = 0.0
            stable_first_snapshot_prefill_stable_tokens = 0
            stable_first_snapshot_prefill_committable_tokens = 0
            stable_first_snapshot_prefill_dynamic_tokens = 0
            stable_first_snapshot_prefill_fallback_reason = None
            materialize_seconds = 0.0
            render_seconds = 0.0
            prefix_render_seconds = 0.0
            incremental_tokenizer_seconds = 0.0
            authoritative_tokenizer_seconds = 0.0
            prefill_initial_candidate_token_ids = (
                request.required_prefix_token_ids
                if request.prefill_from_required_prefix
                else None
            )
            try:
                # Pydantic parses tool calls into lazy ValidatorIterator values.
                # Both render passes need the same materialized values, and the
                # prefix pass deep-copies them into its shortened request.
                materialize_start = time.perf_counter()
                _materialize_message_tool_calls(request.messages)
                materialize_seconds = time.perf_counter() - materialize_start
                render_start = time.perf_counter()
                rendered_prompt = await render_incremental_prompt(request)
                render_seconds = time.perf_counter() - render_start
                authoritative_token_ids = None
                if request.sequence_no == 0:
                    prefix_seed_enabled = bool(
                        (request.prefill or request.finalize_from_required_prefix)
                        and streaming_config
                        and streaming_config.get("prefix_seeded_start")
                    )
                    if prefix_seed_enabled:
                        prefix_seed_attempts = 1
                        prefix_seed_start = time.perf_counter()
                        if request.required_prefix_token_ids is None:
                            prefix_seed_fallbacks = 1
                            prefix_seed_fallback_reason = (
                                "authoritative prefix tokens are absent"
                            )
                        else:
                            try:
                                prefix_render_start = time.perf_counter()
                                template_prefix_prompt = (
                                    await render_incremental_prefix_prompt(request)
                                )
                                prefix_render_seconds = (
                                    time.perf_counter() - prefix_render_start
                                )
                                incremental_tokenizer_start = time.perf_counter()
                                (
                                    result,
                                    authoritative_token_ids,
                                ) = manager.start_from_authoritative_prefix(
                                    session_id=request.session_id,
                                    sequence_no=request.sequence_no,
                                    rendered_prompt=rendered_prompt,
                                    template_prefix_prompt=template_prefix_prompt,
                                    authoritative_prefix_token_ids=(
                                        request.required_prefix_token_ids
                                    ),
                                )
                                incremental_tokenizer_seconds += (
                                    time.perf_counter() - incremental_tokenizer_start
                                )
                            except IncrementalTokenizerPrefixSeedError as error:
                                prefix_seed_fallbacks = 1
                                prefix_seed_fallback_reason = str(error)
                            else:
                                prefix_seed_successes = 1
                                stable_first_snapshot_prefill_enabled = bool(
                                    request.prefill_from_required_prefix
                                    and streaming_config
                                    and streaming_config.get(
                                        "stable_first_snapshot_prefill"
                                    )
                                )
                                if stable_first_snapshot_prefill_enabled:
                                    stable_first_snapshot_prefill_attempts = 1
                                    stable_first_snapshot_prefill_start = (
                                        time.perf_counter()
                                    )
                                    try:
                                        empty_tool_prompt = (
                                            await render_incremental_empty_tool_prompt(
                                                request
                                            )
                                        )
                                        stable_prefix_token_ids = manager.stable_prefix_token_ids_before_alternate_suffix(
                                            session_id=request.session_id,
                                            alternate_rendered_prompt=(
                                                empty_tool_prompt
                                            ),
                                        )
                                        stability_margin_tokens = streaming_config[
                                            "stability_margin_tokens"
                                        ]
                                        candidate_token_count = min(
                                            len(authoritative_token_ids),
                                            len(stable_prefix_token_ids)
                                            + stability_margin_tokens,
                                        )
                                        prefill_initial_candidate_token_ids = (
                                            authoritative_token_ids[
                                                :candidate_token_count
                                            ]
                                        )
                                    except (
                                        IncrementalTokenizerStablePrefixError
                                    ) as error:
                                        stable_first_snapshot_prefill_fallbacks = 1
                                        stable_first_snapshot_prefill_fallback_reason = str(
                                            error
                                        )
                                    else:
                                        stable_first_snapshot_prefill_successes = 1
                                        stable_first_snapshot_prefill_stable_tokens = (
                                            len(stable_prefix_token_ids)
                                        )
                                        stable_first_snapshot_prefill_committable_tokens = max(
                                            0,
                                            candidate_token_count
                                            - stability_margin_tokens,
                                        )
                                        stable_first_snapshot_prefill_dynamic_tokens = max(
                                            0,
                                            stable_first_snapshot_prefill_committable_tokens
                                            - len(request.required_prefix_token_ids),
                                        )
                                    finally:
                                        stable_first_snapshot_prefill_seconds = (
                                            time.perf_counter()
                                            - stable_first_snapshot_prefill_start
                                        )
                        prefix_seed_seconds = time.perf_counter() - prefix_seed_start

                    if authoritative_token_ids is None:
                        authoritative_tokenizer_start = time.perf_counter()
                        authoritative_token_ids = await get_authoritative_token_ids(
                            request,
                            raw_request,
                        )
                        authoritative_tokenizer_seconds += (
                            time.perf_counter() - authoritative_tokenizer_start
                        )
                        if isinstance(authoritative_token_ids, JSONResponse):
                            return authoritative_token_ids
                        incremental_tokenizer_start = time.perf_counter()
                        result = manager.start(
                            session_id=request.session_id,
                            sequence_no=request.sequence_no,
                            rendered_prompt=rendered_prompt,
                            authoritative_token_ids=authoritative_token_ids,
                        )
                        incremental_tokenizer_seconds += (
                            time.perf_counter() - incremental_tokenizer_start
                        )
                    if request.finalize_from_required_prefix:
                        incremental_tokenizer_start = time.perf_counter()
                        result = manager.finalize_current(request.session_id)
                        incremental_tokenizer_seconds += (
                            time.perf_counter() - incremental_tokenizer_start
                        )
                else:
                    checkpoint_due = manager.requires_checkpoint(
                        session_id=request.session_id,
                        sequence_no=request.sequence_no,
                    )
                    if checkpoint_due:
                        authoritative_tokenizer_start = time.perf_counter()
                        authoritative_token_ids = await get_authoritative_token_ids(
                            request,
                            raw_request,
                        )
                        authoritative_tokenizer_seconds += (
                            time.perf_counter() - authoritative_tokenizer_start
                        )
                        if isinstance(authoritative_token_ids, JSONResponse):
                            return authoritative_token_ids
                    incremental_tokenizer_start = time.perf_counter()
                    if request.final:
                        result = manager.finalize(
                            session_id=request.session_id,
                            sequence_no=request.sequence_no,
                            rendered_prompt=rendered_prompt,
                            authoritative_token_ids=authoritative_token_ids,
                        )
                    else:
                        result = manager.append(
                            session_id=request.session_id,
                            sequence_no=request.sequence_no,
                            rendered_prompt=rendered_prompt,
                            authoritative_token_ids=authoritative_token_ids,
                        )
                    incremental_tokenizer_seconds += (
                        time.perf_counter() - incremental_tokenizer_start
                    )
            except IncrementalTokenizerError as error:
                raise HTTPException(status_code=409, detail=str(error)) from error

            counterfactual_requests = 0
            counterfactual_seconds = 0.0
            counterfactual_tokens = 0
            counterfactual_mismatches = 0
            counterfactual_failures = 0
            if (
                request.final
                and authoritative_token_ids is None
                and streaming_config
                and streaming_config.get("counterfactual_full_tokenizer_timing", False)
            ):
                counterfactual_requests = 1
                counterfactual_start = time.perf_counter()
                counterfactual_token_ids = await get_authoritative_token_ids(
                    request,
                    raw_request,
                )
                counterfactual_seconds = time.perf_counter() - counterfactual_start
                if isinstance(counterfactual_token_ids, JSONResponse):
                    counterfactual_failures = 1
                else:
                    counterfactual_tokens = len(counterfactual_token_ids)
                    counterfactual_mismatches = int(
                        result.tokens != counterfactual_token_ids
                    )

            prefill_sessions_started = 0
            prefill_requests = 0
            prefill_tokens = 0
            prefill_completed_chunks = 0
            prefill_dummy_tokens = 0
            prefill_prefix_matched = False
            prefill_failures = 0
            prefill_prompt_too_long_rejections = 0
            prefill_committed_tokens = 0
            prefill_dynamic_tokens = 0
            prefill_effective_requests = 0
            prefill_seconds = 0.0
            prefill_background_scheduled_chunks = 0
            prefill_background_scheduled_tokens = 0
            prefill_background_completed_chunks = 0
            prefill_background_completed_tokens = 0
            prefill_background_completed_dummy_tokens = 0
            prefill_background_effective_chunks = 0
            prefill_background_dynamic_tokens = 0
            prefill_background_cancelled_chunks = 0
            prefill_background_cancelled_tokens = 0
            prefill_background_failed_chunks = 0
            prefill_background_failed_tokens = 0
            prefill_background_enqueue_seconds = 0.0
            prefill_background_completion_seconds = 0.0
            prefill_same_request_session_sealed = 0
            if request.prefill:
                prefill_manager = get_streaming_tool_call_manager()
                background_prefill_completion = bool(
                    streaming_config
                    and streaming_config.get("background_prefill_completion")
                )
                prefill_start = time.perf_counter()
                try:
                    if request.final:
                        assert result.tokens is not None
                        if streaming_config and streaming_config.get(
                            "same_request_final_decode", False
                        ):
                            prefill_close_result = await prefill_manager.seal(
                                session_id=request.session_id,
                                final_prompt_token_ids=result.tokens,
                            )
                            prefill_same_request_session_sealed = 1
                        else:
                            prefill_close_result = await prefill_manager.close(
                                session_id=request.session_id,
                                final_prompt_token_ids=result.tokens,
                            )
                    else:
                        prefill_prompt_token_ids = authoritative_token_ids
                        if prefill_prompt_token_ids is None:
                            prefill_prompt_token_ids = manager.current_token_ids(
                                request.session_id
                            )
                        if request.sequence_no == 0:
                            if request.prefill_continuation:
                                if background_prefill_completion:
                                    prefill_background_start_result = (
                                        await prefill_manager.start_background(
                                            session_id=request.session_id,
                                            prompt_token_ids=(prefill_prompt_token_ids),
                                            sequence_no=0,
                                            initial_candidate_token_ids=(
                                                prefill_initial_candidate_token_ids
                                            ),
                                            dynamic_token_baseline=len(
                                                request.required_prefix_token_ids or []
                                            ),
                                        )
                                    )
                                else:
                                    prefill_append_result = await prefill_manager.start(
                                        session_id=request.session_id,
                                        prompt_token_ids=prefill_prompt_token_ids,
                                        sequence_no=0,
                                        initial_candidate_token_ids=(
                                            prefill_initial_candidate_token_ids
                                        ),
                                    )
                            else:
                                prefill_prime_result = await prefill_manager.prime(
                                    session_id=request.session_id,
                                    prompt_token_ids=prefill_prompt_token_ids,
                                )
                        else:
                            prefill_append_result = await prefill_manager.append(
                                session_id=request.session_id,
                                prompt_token_ids=prefill_prompt_token_ids,
                                sequence_no=request.sequence_no,
                            )
                except (
                    IncrementalTokenizerError,
                    StreamingToolCallError,
                ) as error:
                    prefill_failures = 1
                    prefill_prompt_too_long_rejections = int(
                        isinstance(error, StreamingToolCallPromptTooLongError)
                    )
                    await prefill_manager.abort(session_id=request.session_id)
                else:
                    if request.final:
                        prefill_prefix_matched = prefill_close_result.prefix_matched
                        prefill_committed_tokens = prefill_close_result.committed_tokens
                        prefill_background_completed_chunks = (
                            prefill_close_result.background_completed_chunks
                        )
                        prefill_background_completed_tokens = (
                            prefill_close_result.background_completed_tokens
                        )
                        prefill_background_completed_dummy_tokens = (
                            prefill_close_result.background_completed_dummy_tokens
                        )
                        prefill_background_effective_chunks = (
                            prefill_close_result.background_effective_chunks
                        )
                        prefill_background_dynamic_tokens = (
                            prefill_close_result.background_dynamic_tokens
                        )
                        prefill_background_cancelled_chunks = (
                            prefill_close_result.background_cancelled_chunks
                        )
                        prefill_background_cancelled_tokens = (
                            prefill_close_result.background_cancelled_tokens
                        )
                        prefill_background_failed_chunks = (
                            prefill_close_result.background_failed_chunks
                        )
                        prefill_background_failed_tokens = (
                            prefill_close_result.background_failed_tokens
                        )
                        prefill_background_completion_seconds = (
                            prefill_close_result.background_completion_seconds
                        )
                        # Background work is intentionally unreported by the
                        # start response. Settle it exactly once at close.
                        prefill_requests = prefill_background_completed_chunks
                        prefill_tokens = prefill_background_completed_tokens
                        prefill_completed_chunks = prefill_background_completed_chunks
                        prefill_dummy_tokens = prefill_background_completed_dummy_tokens
                        prefill_dynamic_tokens = prefill_background_dynamic_tokens
                        prefill_effective_requests = prefill_background_effective_chunks
                        prefill_failures = int(prefill_background_failed_chunks > 0)
                    elif request.sequence_no == 0 and not request.prefill_continuation:
                        prefill_sessions_started = 1
                        prefill_requests = int(prefill_prime_result.chunk_tokens > 0)
                        prefill_tokens = prefill_prime_result.chunk_tokens
                        prefill_completed_chunks = prefill_prime_result.completed_chunks
                        prefill_dummy_tokens = prefill_prime_result.dummy_tokens
                        prefill_prefix_matched = prefill_prime_result.prefix_matched
                    elif request.sequence_no == 0 and background_prefill_completion:
                        prefill_sessions_started = 1
                        prefill_background_scheduled_chunks = (
                            prefill_background_start_result.scheduled_chunks
                        )
                        prefill_background_scheduled_tokens = (
                            prefill_background_start_result.scheduled_tokens
                        )
                        prefill_background_enqueue_seconds = (
                            prefill_background_start_result.enqueue_seconds
                        )
                    else:
                        prefill_sessions_started = int(request.sequence_no == 0)
                        prefill_requests = int(prefill_append_result.chunk_tokens > 0)
                        prefill_tokens = prefill_append_result.chunk_tokens
                        prefill_completed_chunks = prefill_requests
                        prefill_dummy_tokens = prefill_requests
                    if not request.final and not (
                        request.sequence_no == 0 and background_prefill_completion
                    ):
                        if (
                            request.sequence_no == 0
                            and not request.prefill_continuation
                        ):
                            prefill_committed_tokens = (
                                prefill_prime_result.committed_tokens
                            )
                            prefill_chunk_tokens = prefill_prime_result.chunk_tokens
                        else:
                            prefill_committed_tokens = (
                                prefill_append_result.committed_tokens
                            )
                            prefill_chunk_tokens = prefill_append_result.chunk_tokens
                        if request.required_prefix_token_ids is not None:
                            required_prefix_tokens = len(
                                request.required_prefix_token_ids
                            )
                            previous_committed_tokens = (
                                prefill_committed_tokens - prefill_chunk_tokens
                            )
                            prefill_dynamic_tokens = max(
                                0,
                                prefill_committed_tokens - required_prefix_tokens,
                            ) - max(
                                0,
                                previous_committed_tokens - required_prefix_tokens,
                            )
                            prefill_effective_requests = int(prefill_dynamic_tokens > 0)
                finally:
                    prefill_seconds = time.perf_counter() - prefill_start

            compact_context_registered = False
            compact_context_registration_seconds = 0.0
            if request.compact_context and not request.final:
                compact_context_registration_start = time.perf_counter()
                store_incremental_request_context(request)
                compact_context_registration_seconds = (
                    time.perf_counter() - compact_context_registration_start
                )
                compact_context_registered = True
            if request.final:
                incremental_request_contexts.pop(request.session_id, None)

            response = asdict(result)
            if request.return_tokens and authoritative_token_ids is not None:
                response["tokens"] = authoritative_token_ids
            response.update(
                {
                    "counterfactual_full_tokenizer_requests": (counterfactual_requests),
                    "counterfactual_full_tokenizer_seconds": (counterfactual_seconds),
                    "counterfactual_full_tokenizer_tokens": (counterfactual_tokens),
                    "counterfactual_full_tokenizer_mismatches": (
                        counterfactual_mismatches
                    ),
                    "counterfactual_full_tokenizer_failures": (counterfactual_failures),
                    "prefill_sessions_started": prefill_sessions_started,
                    "prefill_requests": prefill_requests,
                    "prefill_control_plane_requests": 0,
                    "prefill_tokens": prefill_tokens,
                    "prefill_completed_chunks": prefill_completed_chunks,
                    "prefill_dummy_tokens": prefill_dummy_tokens,
                    "prefill_prefix_matched": prefill_prefix_matched,
                    "prefill_failures": prefill_failures,
                    "prefill_prompt_too_long_rejections": (
                        prefill_prompt_too_long_rejections
                    ),
                    "prefill_committed_tokens": prefill_committed_tokens,
                    "prefill_dynamic_tokens": prefill_dynamic_tokens,
                    "prefill_effective_requests": prefill_effective_requests,
                    "prefill_seconds": prefill_seconds,
                    "prefill_background_scheduled_chunks": (
                        prefill_background_scheduled_chunks
                    ),
                    "prefill_background_scheduled_tokens": (
                        prefill_background_scheduled_tokens
                    ),
                    "prefill_background_completed_chunks": (
                        prefill_background_completed_chunks
                    ),
                    "prefill_background_completed_tokens": (
                        prefill_background_completed_tokens
                    ),
                    "prefill_background_completed_dummy_tokens": (
                        prefill_background_completed_dummy_tokens
                    ),
                    "prefill_background_effective_chunks": (
                        prefill_background_effective_chunks
                    ),
                    "prefill_background_dynamic_tokens": (
                        prefill_background_dynamic_tokens
                    ),
                    "prefill_background_cancelled_chunks": (
                        prefill_background_cancelled_chunks
                    ),
                    "prefill_background_cancelled_tokens": (
                        prefill_background_cancelled_tokens
                    ),
                    "prefill_background_failed_chunks": (
                        prefill_background_failed_chunks
                    ),
                    "prefill_background_failed_tokens": (
                        prefill_background_failed_tokens
                    ),
                    "prefill_background_enqueue_seconds": (
                        prefill_background_enqueue_seconds
                    ),
                    "prefill_background_completion_seconds": (
                        prefill_background_completion_seconds
                    ),
                    "prefill_same_request_session_sealed": (
                        prefill_same_request_session_sealed
                    ),
                    "prefix_seed_attempts": prefix_seed_attempts,
                    "prefix_seed_successes": prefix_seed_successes,
                    "prefix_seed_fallbacks": prefix_seed_fallbacks,
                    "prefix_seed_seconds": prefix_seed_seconds,
                    "prefix_seed_fallback_reason": prefix_seed_fallback_reason,
                    "stable_first_snapshot_prefill_attempts": (
                        stable_first_snapshot_prefill_attempts
                    ),
                    "stable_first_snapshot_prefill_successes": (
                        stable_first_snapshot_prefill_successes
                    ),
                    "stable_first_snapshot_prefill_fallbacks": (
                        stable_first_snapshot_prefill_fallbacks
                    ),
                    "stable_first_snapshot_prefill_seconds": (
                        stable_first_snapshot_prefill_seconds
                    ),
                    "stable_first_snapshot_prefill_stable_tokens": (
                        stable_first_snapshot_prefill_stable_tokens
                    ),
                    "stable_first_snapshot_prefill_committable_tokens": (
                        stable_first_snapshot_prefill_committable_tokens
                    ),
                    "stable_first_snapshot_prefill_dynamic_tokens": (
                        stable_first_snapshot_prefill_dynamic_tokens
                    ),
                    "stable_first_snapshot_prefill_fallback_reason": (
                        stable_first_snapshot_prefill_fallback_reason
                    ),
                    "deferred_prefill_admissions": int(
                        request.prefill_from_required_prefix
                        and prefill_sessions_started > 0
                        and prefill_failures == 0
                    ),
                    "final_prefix_tokenizations": int(
                        request.finalize_from_required_prefix
                        and result.tokens is not None
                    ),
                    "server_materialize_seconds": materialize_seconds,
                    "server_render_seconds": render_seconds,
                    "server_prefix_render_seconds": prefix_render_seconds,
                    "server_incremental_tokenizer_seconds": (
                        incremental_tokenizer_seconds
                    ),
                    "server_authoritative_tokenizer_seconds": (
                        authoritative_tokenizer_seconds
                    ),
                    "server_compact_context_registrations": int(
                        compact_context_registered
                    ),
                    "server_compact_context_hits": 0,
                    "server_compact_context_rebuild_seconds": 0.0,
                    "server_compact_context_registration_seconds": (
                        compact_context_registration_seconds
                    ),
                    "server_request_handler_seconds": (
                        time.perf_counter() - request_handler_start
                    ),
                }
            )
            return response

        @app.post("/incremental_tokenize/compact")
        async def compact_incremental_tokenize(
            request: NeMoRLCompactIncrementalTokenizeRequest,
            raw_request: Request,
        ):
            compact_context_rebuild_start = time.perf_counter()
            if not compact_request_context_enabled:
                raise HTTPException(
                    status_code=503,
                    detail="compact request context is not enabled",
                )
            if request.sequence_no <= 0:
                raise HTTPException(
                    status_code=422,
                    detail="compact tool-output requests require a positive sequence",
                )
            request_template = get_incremental_request_context(request.session_id)
            if (
                not request_template.messages
                or request_template.messages[-1].get("role") != "tool"
            ):
                incremental_request_contexts.pop(request.session_id, None)
                raise HTTPException(
                    status_code=409,
                    detail=(
                        "compact tokenization context has no trailing tool message"
                    ),
                )
            messages = list(request_template.messages)
            messages[-1] = {
                **messages[-1],
                "content": request.tool_output,
            }
            reconstructed_request = request_template.model_copy(
                update={
                    "messages": messages,
                    "sequence_no": request.sequence_no,
                    "final": request.final,
                    "prefill": request.prefill,
                    "prefill_continuation": request.prefill_continuation,
                    "prefill_from_required_prefix": (
                        request.prefill_from_required_prefix
                    ),
                    "finalize_from_required_prefix": (
                        request.finalize_from_required_prefix
                    ),
                    "compact_context": False,
                    "max_contexts": None,
                    "context_ttl_seconds": None,
                }
            )
            compact_context_rebuild_seconds = (
                time.perf_counter() - compact_context_rebuild_start
            )
            result = await incremental_tokenize(
                reconstructed_request,
                raw_request,
            )
            if isinstance(result, dict):
                result["server_compact_context_hits"] = 1
                result["server_compact_context_rebuild_seconds"] = (
                    compact_context_rebuild_seconds
                )
            return result

        @app.post("/incremental_tokenize/abort")
        async def abort_incremental_tokenize(
            request: StreamingToolCallAbortRequest,
        ):
            incremental_request_contexts.pop(request.session_id, None)
            manager = incremental_tokenizer_manager
            tokenizer_aborted = (
                manager.abort(request.session_id) if manager is not None else False
            )
            prefill_manager = getattr(self, "streaming_tool_call_manager", None)
            prefill_aborted = False
            if prefill_manager is not None:
                prefill_manager.bind_to_current_loop()
                prefill_aborted = await prefill_manager.abort(
                    session_id=request.session_id
                )
            return {"aborted": tokenizer_aborted or prefill_aborted}

        ########################################
        # Streaming tool-call prefill endpoints
        ########################################

        def get_streaming_tool_call_manager() -> StreamingToolCallPrefillManager:
            manager = getattr(self, "streaming_tool_call_manager", None)
            if manager is None:
                raise HTTPException(
                    status_code=503,
                    detail="streaming tool-call prefill is not enabled",
                )
            manager.bind_to_current_loop()
            return manager

        @app.on_event("startup")
        async def bind_streaming_tool_call_manager() -> None:
            manager = getattr(self, "streaming_tool_call_manager", None)
            if manager is not None:
                manager.bind_to_current_loop()

        @app.post("/v1/streaming_tool_call/start")
        async def start_streaming_tool_call(
            request: StreamingToolCallPrefillRequest,
        ):
            manager = get_streaming_tool_call_manager()
            try:
                result = await manager.start(
                    session_id=request.session_id,
                    prompt_token_ids=request.prompt_token_ids,
                    sequence_no=request.sequence_no,
                )
            except StreamingToolCallPromptTooLongError as error:
                await manager.abort(session_id=request.session_id)
                raise HTTPException(status_code=409, detail=str(error)) from error
            except StreamingToolCallError as error:
                raise HTTPException(status_code=409, detail=str(error)) from error
            return asdict(result)

        @app.post("/v1/streaming_tool_call/append")
        async def append_streaming_tool_call(
            request: StreamingToolCallPrefillRequest,
        ):
            manager = get_streaming_tool_call_manager()
            try:
                result = await manager.append(
                    session_id=request.session_id,
                    prompt_token_ids=request.prompt_token_ids,
                    sequence_no=request.sequence_no,
                )
            except StreamingToolCallPromptTooLongError as error:
                await manager.abort(session_id=request.session_id)
                raise HTTPException(status_code=409, detail=str(error)) from error
            except StreamingToolCallSessionNotFoundError as error:
                raise HTTPException(status_code=404, detail=str(error)) from error
            except StreamingToolCallError as error:
                raise HTTPException(status_code=409, detail=str(error)) from error
            return asdict(result)

        @app.post("/v1/streaming_tool_call/close")
        async def close_streaming_tool_call(request: StreamingToolCallCloseRequest):
            manager = get_streaming_tool_call_manager()
            try:
                result = await manager.close(
                    session_id=request.session_id,
                    final_prompt_token_ids=request.final_prompt_token_ids,
                )
            except StreamingToolCallSessionNotFoundError as error:
                raise HTTPException(status_code=404, detail=str(error)) from error
            return asdict(result)

        @app.post("/v1/streaming_tool_call/abort")
        async def abort_streaming_tool_call(request: StreamingToolCallAbortRequest):
            manager = get_streaming_tool_call_manager()
            return {"aborted": await manager.abort(session_id=request.session_id)}

        ########################################
        # Logging
        ########################################
        print(
            "Adding a vLLM logging filter so that the logs aren't spammed with not useful messages like `Added request ...`. This is to help errors pop up better and filter out noise."
        )

        class CleanLoggingFilter(LoggingFilter):
            def filter(self, record: LogRecord) -> bool:
                msg = record.getMessage()

                # vLLM does not accept `strict` tool definitions and reporting it to the user is not useful either.
                return (
                    "Added request" not in msg
                    and "The following fields were present in the request but ignored: {'strict'}"
                    not in msg
                )

        vllm_async_llm_logger.addFilter(CleanLoggingFilter())

        from logging import getLogger as _getLogger

        _getLogger("vllm.entrypoints.openai.engine.protocol").addFilter(
            CleanLoggingFilter()
        )

        # Suppress the noisy vLLM traceback when a prompt exceeds max_model_len.
        # This is expected during multi-turn rollouts; we log a clean one-line
        # warning from _preprocess_chat instead.
        class MaxContextLengthFilter(LoggingFilter):
            def filter(self, record: LogRecord) -> bool:
                if record.exc_info and record.exc_info[1]:
                    if "maximum context length" in str(record.exc_info[1]):
                        return False
                return True

        _getLogger("vllm.entrypoints.openai.serving_chat").addFilter(
            MaxContextLengthFilter()
        )

        return app

    def _setup_vllm_server(self) -> "tuple[threading.Thread, str, uvicorn.Server]":
        import threading
        from logging import Filter as LoggingFilter
        from logging import LogRecord, getLogger

        import uvicorn
        from fastapi import FastAPI

        # We initialize the FastAPI app here in case we want to do some generic configuration before the subsequent server inits
        # e.g. last-run middleware.
        app = FastAPI()

        app = self._setup_vllm_openai_api_server(app)

        ########################################
        # Server spinup
        ########################################

        if self._reserved_socket is not None:
            # Use the socket reserved during __init__ (deferred model load path).
            # Pass it directly to uvicorn via sockets= — zero gap, the socket is
            # never closed and re-opened, so no other process can steal the port.
            node_ip = self._reserved_node_ip
            free_port = self._reserved_port
            reserved_sock = self._reserved_socket
            self._reserved_socket = None  # Transfer ownership to uvicorn
        else:
            node_ip = _get_node_ip_local()
            port_range_low = self.cfg.get(
                "port_range_low", DEFAULT_GENERATION_PORT_RANGE_LOW
            )
            port_range_high = self.cfg.get(
                "port_range_high", DEFAULT_GENERATION_PORT_RANGE_HIGH
            )
            free_port = _get_free_port_local(port_range_low, port_range_high)
            reserved_sock = None

        base_url = f"http://{node_ip}:{free_port}/v1"
        print(f"Starting server on {base_url}")

        config = uvicorn.Config(
            app,
            host="0.0.0.0",
            port=free_port,
            timeout_keep_alive=120,  # Keep connections alive longer (default is 5s), fix for this error: Hit an exception while making a request (try 1): <class 'aiohttp.client_exceptions.ClientOSError'>: [Errno 104] Connection reset by peer
        )
        server = uvicorn.Server(config=config)

        print(
            "Adding a uvicorn logging filter so that the logs aren't spammed with 200 OK messages. This is to help errors pop up better and filter out noise."
        )

        class No200Filter(LoggingFilter):
            def filter(self, record: LogRecord) -> bool:
                msg = record.getMessage()
                return not msg.strip().endswith("200")

        uvicorn_logger = getLogger("uvicorn.access")
        uvicorn_logger.addFilter(No200Filter())

        if reserved_sock is not None:
            # Hand the pre-bound listening socket directly to uvicorn's asyncio
            # server via server.serve(sockets=). No close-and-rebind needed.
            import asyncio

            def _run_with_socket():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(server.serve(sockets=[reserved_sock]))

            thread = threading.Thread(target=_run_with_socket, daemon=True)
        else:
            thread = threading.Thread(target=server.run, daemon=True)
        thread.start()

        if self.streaming_tool_call_manager is not None:
            startup_deadline = time.monotonic() + 30
            while not server.started:
                if not thread.is_alive():
                    raise RuntimeError("vLLM HTTP server exited during startup")
                if time.monotonic() >= startup_deadline:
                    server.should_exit = True
                    raise TimeoutError(
                        "vLLM HTTP server did not start within 30 seconds"
                    )
                time.sleep(0.01)

        return thread, base_url, server

    async def init_collective_async(
        self,
        rank_prefix: int,
        ip: str,
        port: int,
        world_size: int,
        train_world_size: int,
    ) -> None:
        await self.llm.collective_rpc(
            "init_collective",
            args=(
                rank_prefix,
                ip,
                port,
                world_size,
                train_world_size,
            ),
        )

    async def generate_async(
        self,
        data: BatchedDataDict[GenerationDatumSpec],
        greedy: bool = False,
    ) -> AsyncGenerator[tuple[int, BatchedDataDict[GenerationOutputSpec]], None]:
        """Generate a batch of data using vLLM's AsyncLLMEngine, yielding results as they are ready.

        Args:
            data: BatchedDataDict with input_ids and input_lengths
            greedy: Whether to use greedy decoding instead of sampling

        Yields:
            Tuple of (original_index, BatchedDataDict conforming to GenerationOutputSpec for the single sequence)
        """
        if not self.cfg["vllm_cfg"]["async_engine"]:
            raise RuntimeError(
                "generate_async can only be used when async_engine is enabled in vLLM config."
            )

        # Handle empty input case
        if len(data["input_ids"]) == 0:
            return

        verify_right_padding(data, pad_value=self.cfg["_pad_token_id"])

        input_ids_batch = data["input_ids"]
        input_lengths_batch = data["input_lengths"]
        batch_size = input_ids_batch.shape[0]

        # Ensure generate_async only receives single samples (batch_size = 1)
        assert batch_size == 1, (
            f"generate_async is restricted to handle only single samples, "
            f"but received batch_size={batch_size}. Please handle batching outside this method."
        )

        batch_specific_stop_strings_list = data.get(
            "stop_strings", [[] for _ in range(batch_size)]
        )

        # Create tasks for each sample in the batch
        async def process_single_sample(sample_idx):
            """Process a single sample and return the result."""
            current_input_actual_length = input_lengths_batch[sample_idx].item()
            prompt = format_prompt_for_vllm_generation(data, sample_idx)

            per_sample_stop_strings = None
            if batch_specific_stop_strings_list and sample_idx < len(
                batch_specific_stop_strings_list
            ):
                per_sample_stop_strings = batch_specific_stop_strings_list[sample_idx]

            final_stop_strings_for_sample = self._merge_stop_strings(
                [per_sample_stop_strings] if per_sample_stop_strings else None
            )

            remaining_ctx = (
                self.cfg["vllm_cfg"]["max_model_len"] - current_input_actual_length
            )
            allowed_new_tokens = max(0, min(self.cfg["max_new_tokens"], remaining_ctx))

            # Handle case where no tokens can be generated due to length constraints
            if allowed_new_tokens == 0:
                # Access the input data directly from the function parameters
                input_ids_single_row = input_ids_batch[sample_idx]

                # Create output tensors with just the input (no generated tokens)
                output_ids_single_item_batched = input_ids_single_row[
                    :current_input_actual_length
                ].unsqueeze(0)

                logprobs_single_item = torch.zeros(
                    (1, current_input_actual_length),
                    dtype=torch.float32,
                    device=input_ids_single_row.device,
                )

                generation_lengths_tensor = torch.tensor(
                    [0], dtype=torch.long, device=input_ids_single_row.device
                )

                unpadded_sequence_lengths_tensor = torch.tensor(
                    [current_input_actual_length],
                    dtype=torch.long,
                    device=input_ids_single_row.device,
                )

                # Not truncated since no generation was attempted (length constraint)
                truncated_tensor = torch.tensor(
                    [False], dtype=torch.bool, device=input_ids_single_row.device
                )

                result_batch = BatchedDataDict[GenerationOutputSpec](
                    {
                        "output_ids": output_ids_single_item_batched,
                        "logprobs": logprobs_single_item,
                        "generation_lengths": generation_lengths_tensor,
                        "unpadded_sequence_lengths": unpadded_sequence_lengths_tensor,
                        "truncated": truncated_tensor,
                    }
                )

                return (sample_idx, result_batch)

            sampling_params_for_request = self._build_sampling_params(
                greedy=greedy,
                stop_strings=final_stop_strings_for_sample,
                max_new_tokens=allowed_new_tokens,
            )

            request_id = str(uuid.uuid4())

            # Generate using vLLM async engine
            vllm_request_generator = self.llm.generate(
                prompt=prompt,
                sampling_params=sampling_params_for_request,
                request_id=request_id,
            )

            # Get the final result from the generator
            final_request_output = None
            async for req_output in vllm_request_generator:
                final_request_output = req_output

            if final_request_output is None:
                raise RuntimeError(f"No output received for request {request_id}")

            # Process the output
            generation_details = final_request_output.outputs[0]
            generated_token_ids = list(generation_details.token_ids)
            num_generated_tokens = len(generated_token_ids)
            return_routed_experts = self._return_routed_experts_enabled()

            original_input_ids_single_row = input_ids_batch[sample_idx]
            final_output_tensor_len = current_input_actual_length + num_generated_tokens

            # Create output_ids tensor for this single item
            output_ids_single_item = torch.full(
                (final_output_tensor_len,),
                self.cfg["_pad_token_id"],
                dtype=original_input_ids_single_row.dtype,
                device=original_input_ids_single_row.device,
            )
            # Copy original input (up to its actual length)
            output_ids_single_item[:current_input_actual_length] = (
                original_input_ids_single_row[:current_input_actual_length]
            )
            # Add generated tokens after the actual input
            output_ids_single_item[
                current_input_actual_length : current_input_actual_length
                + num_generated_tokens
            ] = torch.tensor(
                generated_token_ids,
                dtype=original_input_ids_single_row.dtype,
                device=original_input_ids_single_row.device,
            )

            # Reshape to (1, seq_len) for BatchedDataDict
            output_ids_single_item_batched = output_ids_single_item.unsqueeze(0)

            # Create logprobs tensor for this single item
            logprobs_single_item = torch.zeros(
                (1, final_output_tensor_len),
                dtype=torch.float32,
                device=original_input_ids_single_row.device,
            )
            if hasattr(generation_details, "logprobs") and generation_details.logprobs:
                for idx, logprob_dict_per_token in enumerate(
                    generation_details.logprobs
                ):
                    if logprob_dict_per_token and idx < len(generated_token_ids):
                        token_id_at_idx = generated_token_ids[idx]
                        if token_id_at_idx in logprob_dict_per_token:
                            logprob_value = logprob_dict_per_token[
                                token_id_at_idx
                            ].logprob
                            position_in_output_tensor = (
                                current_input_actual_length + idx
                            )
                            if position_in_output_tensor < final_output_tensor_len:
                                logprobs_single_item[0, position_in_output_tensor] = (
                                    logprob_value
                                )

            # Generation lengths
            generation_lengths_tensor = torch.tensor(
                [num_generated_tokens],
                dtype=torch.long,
                device=original_input_ids_single_row.device,
            )

            # Unpadded sequence lengths (actual_input + actual_generated)
            unpadded_total_length = current_input_actual_length + num_generated_tokens
            unpadded_sequence_lengths_tensor = torch.tensor(
                [unpadded_total_length],
                dtype=torch.long,
                device=original_input_ids_single_row.device,
            )

            # Check if response was truncated (hit max_tokens length limit)
            is_truncated = generation_details.finish_reason == "length"
            truncated_tensor = torch.tensor(
                [is_truncated],
                dtype=torch.bool,
                device=original_input_ids_single_row.device,
            )

            result_dict = {
                "output_ids": output_ids_single_item_batched,
                "logprobs": logprobs_single_item,
                "generation_lengths": generation_lengths_tensor,
                "unpadded_sequence_lengths": unpadded_sequence_lengths_tensor,
                "truncated": truncated_tensor,
            }
            routed_experts, r3_stats = pad_and_align_routed_expert_indices(
                final_request_output,
                generation_details,
                valid_length=unpadded_total_length,
                padded_length=final_output_tensor_len,
                device=original_input_ids_single_row.device,
                require_complete_routed_experts=return_routed_experts,
                return_stats=True,
                routed_experts_dtype=self.routed_experts_dtype,
            )
            if return_routed_experts and routed_experts is None:
                raise RuntimeError(
                    "vLLM was asked to return routed experts but the generation output "
                    "did not include routed_experts."
                )
            if return_routed_experts:
                if r3_stats["missing_routes"] > 0:
                    LOGGER.warning(
                        "R3 router replay fallback: vLLM returned incomplete "
                        "routed_experts for sample_idx=%d, missing_token_routes=%d, "
                        "actual_routes=%d, expected_routes=%d. Megatron will use its "
                        "own router for those missing token routes.",
                        sample_idx,
                        r3_stats["missing_routes"],
                        r3_stats["actual_routes"],
                        r3_stats["expected_routes"],
                    )
                result_dict["r3_routed_experts_missing_routes"] = torch.tensor(
                    [r3_stats["missing_routes"]],
                    dtype=torch.long,
                    device=original_input_ids_single_row.device,
                )
                result_dict["r3_routed_experts_expected_routes"] = torch.tensor(
                    [r3_stats["expected_routes"]],
                    dtype=torch.long,
                    device=original_input_ids_single_row.device,
                )
                result_dict["r3_routed_experts_actual_routes"] = torch.tensor(
                    [r3_stats["actual_routes"]],
                    dtype=torch.long,
                    device=original_input_ids_single_row.device,
                )
            if routed_experts is not None:
                result_dict["routed_experts"] = routed_experts.unsqueeze(0)

            result_batch = BatchedDataDict[GenerationOutputSpec](result_dict)

            return (sample_idx, result_batch)

        # Create tasks for all samples and yield results as they complete
        sample_tasks = [
            asyncio.create_task(process_single_sample(i)) for i in range(batch_size)
        ]

        # Yield results as they become available
        for completed_task in asyncio.as_completed(sample_tasks):
            try:
                result = await completed_task
                yield result
            except Exception as e:
                # Cancel remaining tasks
                for task in sample_tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*sample_tasks, return_exceptions=True)
                raise e

    async def generate_text_async(
        self, data: BatchedDataDict[GenerationDatumSpec], greedy: bool = False
    ) -> AsyncGenerator[tuple[int, BatchedDataDict[GenerationOutputSpec]], None]:
        """Generate text responses asynchronously, yielding results as they are ready.

        Args:
            data: BatchedDataDict containing prompts with text strings
            greedy: Whether to use greedy decoding instead of sampling

        Yields:
            Tuple of (original_index, BatchedDataDict containing single text response)
        """
        if not self.cfg["vllm_cfg"]["async_engine"]:
            raise RuntimeError(
                "generate_text_async can only be used when async_engine is enabled in vLLM config."
            )

        # Handle empty input case
        if len(data["prompts"]) == 0:
            return

        prompts = data["prompts"]
        batch_size = len(prompts)

        # Extract stop_strings if provided, else use default from config
        batch_stop_strings: list[list[str] | None] = data.get(
            "stop_strings", [self.cfg.get("stop_strings")] * batch_size
        )

        # Create tasks for each prompt
        async def process_single_prompt(prompt_idx):
            """Process a single prompt and return the result."""
            prompt = prompts[prompt_idx]

            # Get stop strings for this specific prompt
            per_prompt_stop_strings = None
            if batch_stop_strings and prompt_idx < len(batch_stop_strings):
                per_prompt_stop_strings = batch_stop_strings[prompt_idx]

            # Merge stop strings
            final_stop_strings = self._merge_stop_strings(
                [per_prompt_stop_strings] if per_prompt_stop_strings else None
            )

            # Create sampling parameters
            top_k = self.cfg["top_k"] if self.cfg["top_k"] is not None else -1
            sampling_params = self.SamplingParams(
                temperature=self.cfg["temperature"] if not greedy else 0,
                top_p=self.cfg["top_p"],
                top_k=top_k if not greedy else 1,
                max_tokens=self.cfg["max_new_tokens"],
                stop_token_ids=self.cfg["stop_token_ids"],
                stop=final_stop_strings,
                include_stop_str_in_output=True,  # returning stop strings like hf
            )

            request_id = str(uuid.uuid4())

            # Generate using vLLM async engine
            vllm_request_generator = self.llm.generate(
                prompt=prompt,
                sampling_params=sampling_params,
                request_id=request_id,
            )

            # Get the final result from the generator
            final_request_output = None
            async for req_output in vllm_request_generator:
                final_request_output = req_output

            if final_request_output is None:
                raise RuntimeError(f"No output received for request {request_id}")

            # Extract the generated text
            generated_text = final_request_output.outputs[0].text

            # Create result in BatchedDataDict format
            result_batch = BatchedDataDict[GenerationOutputSpec](
                {"texts": [generated_text]}
            )

            return (prompt_idx, result_batch)

        # Create tasks for all prompts and yield results as they complete
        prompt_tasks = [
            asyncio.create_task(process_single_prompt(i)) for i in range(batch_size)
        ]

        # Yield results as they become available
        for completed_task in asyncio.as_completed(prompt_tasks):
            try:
                result = await completed_task
                yield result
            except Exception as e:
                # Cancel remaining tasks
                for task in prompt_tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*prompt_tasks, return_exceptions=True)
                raise e

    async def report_device_id_async(self) -> list[str]:
        """Async version of report_device_id."""
        assert self.llm is not None, (
            "Attempting to report device id with either an uninitialized vLLM or non-model-owner"
        )

        if not self.cfg["vllm_cfg"]["async_engine"]:
            raise RuntimeError(
                "report_device_id_async can only be used with async_engine=True. Use report_device_id instead."
            )

        result_or_coro = await self.llm.collective_rpc("report_device_id", args=tuple())

        if asyncio.iscoroutine(result_or_coro):
            list_of_worker_results = await result_or_coro
        else:
            list_of_worker_results = result_or_coro

        return cast(list[str], list_of_worker_results)

    async def prepare_refit_info_async(self, state_dict_info: dict[str, Any]) -> None:
        """Async version of prepare_refit_info."""
        await self.llm.collective_rpc("prepare_refit_info", args=(state_dict_info,))

    async def _run_streaming_tool_call_manager_operation(
        self,
        operation: Callable[
            [StreamingToolCallPrefillManager],
            Coroutine[Any, Any, StreamingToolCallManagerResult],
        ],
    ) -> StreamingToolCallManagerResult:
        """Run a manager operation on the HTTP server's event loop."""
        manager = self.streaming_tool_call_manager
        assert manager is not None
        owner_loop = manager.event_loop
        if owner_loop is None or owner_loop.is_closed():
            raise RuntimeError("streaming tool-call HTTP event loop is not available")

        if owner_loop is asyncio.get_running_loop():
            return await operation(manager)

        operation_coro = operation(manager)
        try:
            concurrent_future = asyncio.run_coroutine_threadsafe(
                operation_coro, owner_loop
            )
        except Exception:
            operation_coro.close()
            raise
        return await asyncio.wrap_future(concurrent_future)

    async def _invalidate_streaming_tool_call_sessions(self) -> int:
        manager = getattr(self, "streaming_tool_call_manager", None)
        if manager is None:
            return 0
        return await self._run_streaming_tool_call_manager_operation(
            lambda active_manager: active_manager.invalidate_all()
        )

    async def _pause_streaming_tool_call_sessions(self) -> int:
        manager = getattr(self, "streaming_tool_call_manager", None)
        if manager is None:
            return 0
        return await self._run_streaming_tool_call_manager_operation(
            lambda active_manager: active_manager.pause_and_invalidate()
        )

    async def _resume_streaming_tool_call_sessions(self) -> None:
        manager = getattr(self, "streaming_tool_call_manager", None)
        if manager is not None:
            await self._run_streaming_tool_call_manager_operation(
                lambda active_manager: active_manager.resume()
            )

    async def update_weights_via_ipc_zmq_async(
        self,
    ) -> bool:
        """Async version of update_weights_via_ipc_zmq."""
        try:
            assert self.llm is not None, (
                "Attempting to update weights with either an uninitialized vLLM or non-model-owner"
            )

            if not self.cfg["vllm_cfg"]["async_engine"]:
                raise RuntimeError(
                    "update_weights_via_ipc_zmq_async can only be used with async_engine=True. Use update_weights_via_ipc_zmq instead."
                )

            await self._pause_streaming_tool_call_sessions()

            # TODO: switch to update_weights_from_local_ipc_handles for better performance once collectively report_device_id is supported in asyncLLM initialization
            try:
                result_or_coro = await self.llm.collective_rpc(
                    "update_weights_via_ipc_zmq", args=tuple()
                )

                if asyncio.iscoroutine(result_or_coro):
                    worker_results = await result_or_coro
                else:
                    worker_results = result_or_coro
            finally:
                await self._resume_streaming_tool_call_sessions()

            worker_result = worker_results[0]

            if not worker_result:
                print(
                    f"Error: Worker failed to update weights. Result: {worker_result}"
                )
                return False
            return True
        except Exception as e:
            print(f"Exception during collective_rpc for weight update: {e}")
            import traceback

            traceback.print_exc()
            return False

    async def update_weights_from_collective_async(self) -> bool:
        """Async version of update_weights_from_collective."""
        try:
            assert self.llm is not None, (
                "Attempting to update weights with either an uninitialized vLLM or non-model-owner"
            )

            if not self.cfg["vllm_cfg"]["async_engine"]:
                raise RuntimeError(
                    "update_weights_from_collective_async can only be used with async_engine=True. Use update_weights_from_collective instead."
                )

            await self._pause_streaming_tool_call_sessions()

            try:
                result_or_coro = await self.llm.collective_rpc(
                    "update_weights_from_collective", args=tuple()
                )

                if asyncio.iscoroutine(result_or_coro):
                    worker_results = await result_or_coro
                else:
                    worker_results = result_or_coro
            finally:
                await self._resume_streaming_tool_call_sessions()

            worker_result = worker_results[0]

            if not worker_result:
                print(
                    f"Error: Worker failed to update weights. Result: {worker_result}"
                )
                return False
            return True
        except Exception as e:
            print(f"Exception during collective_rpc for weight update: {e}")
            import traceback

            traceback.print_exc()
            return False

    async def reset_prefix_cache_async(self):
        """Async version of reset_prefix_cache."""
        assert self.llm is not None, (
            "Attempting to reset prefix cache with either an uninitialized vLLM or non-model-owner"
        )

        if not self.cfg["vllm_cfg"]["async_engine"]:
            raise RuntimeError(
                "reset_prefix_cache_async can only be used with async_engine=True. Use reset_prefix_cache instead."
            )

        await self._pause_streaming_tool_call_sessions()
        try:
            await self.llm.reset_prefix_cache()
        finally:
            await self._resume_streaming_tool_call_sessions()
        gc.collect()
        torch.cuda.empty_cache()

    async def sleep_async(self):
        """Async version of sleep."""
        assert self.llm is not None, (
            "Attempting to sleep with either an uninitialized vLLM or non-model-owner"
        )

        if not self.cfg["vllm_cfg"]["async_engine"]:
            raise RuntimeError(
                "sleep_async can only be used with async_engine=True. Use sleep instead."
            )

        await self._pause_streaming_tool_call_sessions()
        try:
            # Reset the prefix cache to ensure that prefix cache is not reused after weights are updated
            await self.llm.reset_prefix_cache()
            # Reset the multimodal processor cache (sender side) so it stays in
            # sync with the receiver cache that vLLM clears internally during
            # sleep.  Without this, the sender thinks images are already cached on
            # the receiver and sends data=None, causing an assertion error.
            if hasattr(self.llm, "reset_mm_cache"):
                await self.llm.reset_mm_cache()
            await self.llm.sleep(level=1)
        except Exception:
            await self._resume_streaming_tool_call_sessions()
            raise

        gc.collect()
        torch.cuda.empty_cache()

    async def wake_up_async(self, **kwargs):
        """Async version of wake_up."""
        assert self.llm is not None, (
            "Attempting to wake up with either an uninitialized vLLM or non-model-owner"
        )

        if not self.cfg["vllm_cfg"]["async_engine"]:
            raise RuntimeError(
                "wake_up_async can only be used with async_engine=True. Use wake_up instead."
            )

        tags = kwargs.get("tags")

        wake_up_args = {}
        if tags is not None:
            wake_up_args["tags"] = tags

        await self.llm.wake_up(**wake_up_args)
        await self._resume_streaming_tool_call_sessions()

    async def shutdown(self) -> bool:
        """Clean up vLLM resources."""
        try:
            await self._pause_streaming_tool_call_sessions()
            if self.llm is not None:
                # Clean up extension resources (e.g., ZMQ sockets)
                await self.llm.collective_rpc("cleanup", args=tuple())
                try:
                    self.llm.shutdown()
                except Exception as e_stop:
                    print(f"Error calling shutdown_background_loop: {e_stop}")

                # Explicitly delete the engine. This may trigger its __del__ method.
                del self.llm

            self.llm = None
            self.tokenizer = None

            # Force garbage collection
            gc.collect()
            torch.cuda.empty_cache()

            server_thread = getattr(self, "server_thread", None)
            if server_thread is not None:
                from uvicorn import Server

                self.http_server: Server

                self.http_server.should_exit = True
                server_thread.join()

            return True
        except Exception as e:
            print(f"Error during vLLM shutdown: {e}")
            return False
        finally:
            writeback_vllm_cache()


@ray.remote(
    runtime_env={**get_nsight_config_if_pattern_matches("vllm_async_generation_worker")}
)  # pragma: no cover
class VllmAsyncGenerationWorker(VllmAsyncGenerationWorkerImpl):
    pass
