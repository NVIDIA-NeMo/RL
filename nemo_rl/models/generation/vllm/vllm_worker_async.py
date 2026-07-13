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
import hashlib
import json
import logging
import os
import threading
import time
import uuid
import warnings
from typing import Any, AsyncGenerator, Optional, cast

import ray
import torch
import uvicorn
from fastapi import FastAPI

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
from nemo_rl.models.generation.vllm.utils import (
    attach_routed_experts_to_chat_response_choices,
    format_prompt_for_vllm_generation,
    model_dump_chat_response_with_routed_experts,
    pad_and_align_routed_expert_indices,
)
from nemo_rl.models.generation.vllm.vllm_worker import BaseVllmGenerationWorker

LOGGER = logging.getLogger(__name__)
_TRACE_WRITE_LOCK = threading.Lock()


def _json_hash(value: Any) -> str | None:
    try:
        payload = json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            default=str,
        ).encode("utf-8")
    except Exception:
        return None
    return hashlib.sha256(payload).hexdigest()


def _token_ids_hash(token_ids: list[int] | None) -> str | None:
    if token_ids is None:
        return None
    return _json_hash(token_ids)


def _token_id_from_value(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        if value.startswith("token_id:"):
            value = value.removeprefix("token_id:")
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _as_int_token_list(value: Any) -> list[int] | None:
    if not isinstance(value, list):
        return None
    token_ids = []
    for item in value:
        token_id = _token_id_from_value(item)
        if token_id is None:
            return None
        token_ids.append(token_id)
    return token_ids


def _find_first_token_list(value: Any, key: str) -> list[int] | None:
    if isinstance(value, dict):
        tokens = _as_int_token_list(value.get(key))
        if tokens is not None:
            return tokens
        for child in value.values():
            tokens = _find_first_token_list(child, key)
            if tokens is not None:
                return tokens
    elif isinstance(value, list):
        for child in value:
            tokens = _find_first_token_list(child, key)
            if tokens is not None:
                return tokens
    return None


def _model_dump(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return value
    try:
        if hasattr(value, "model_dump"):
            return value.model_dump(mode="json")
    except TypeError:
        try:
            return value.model_dump()
        except Exception:
            return None
    except Exception:
        return None
    return None


def _extract_sse_json_dict(chunk: Any) -> dict[str, Any] | None:
    if isinstance(chunk, dict):
        return chunk

    if isinstance(chunk, bytes):
        text = chunk.decode("utf-8", errors="ignore")
    elif isinstance(chunk, str):
        text = chunk
    else:
        return None

    for line in reversed(text.splitlines()):
        stripped = line.strip()
        if not stripped.startswith("data:"):
            continue
        payload = stripped.removeprefix("data:").strip()
        if not payload or payload == "[DONE]":
            continue
        try:
            parsed = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed

    return None


def _generation_token_ids_from_logprobs(
    response_dump: dict[str, Any] | None,
) -> list[int] | None:
    choices = response_dump.get("choices") if response_dump else None
    if not isinstance(choices, list):
        return None

    token_ids = []
    saw_logprob_content = False
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        logprobs = choice.get("logprobs")
        if not isinstance(logprobs, dict):
            continue
        content = logprobs.get("content")
        if not isinstance(content, list):
            continue
        saw_logprob_content = True
        for item in content:
            if not isinstance(item, dict):
                continue
            token_id = _token_id_from_value(item.get("token"))
            if token_id is not None:
                token_ids.append(token_id)

    return token_ids if saw_logprob_content else None


def _generation_token_ids_from_choices(
    response_dump: dict[str, Any] | None,
) -> list[int] | None:
    choices = response_dump.get("choices") if response_dump else None
    if not isinstance(choices, list):
        return None

    token_ids = []
    saw_token_ids = False
    for choice in choices:
        if not isinstance(choice, dict):
            continue
        choice_token_ids = _as_int_token_list(choice.get("token_ids"))
        if choice_token_ids is None:
            continue
        saw_token_ids = True
        token_ids.extend(choice_token_ids)

    return token_ids if saw_token_ids else None


def _generation_token_ids_from_response_dump(
    response_dump: dict[str, Any] | None,
) -> list[int] | None:
    token_ids = _generation_token_ids_from_choices(response_dump)
    if token_ids is None:
        token_ids = _find_first_token_list(response_dump, "generation_token_ids")
    if token_ids is None:
        token_ids = _generation_token_ids_from_logprobs(response_dump)
    return token_ids


def _usage_from_response(
    response_dump: dict[str, Any] | None,
) -> dict[str, int | None]:
    usage = response_dump.get("usage") if response_dump else None
    if not isinstance(usage, dict):
        return {
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
        }
    return {
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "total_tokens": usage.get("total_tokens"),
    }


def _coerce_int(value: Any) -> int | None:
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return None
    return None


def _snapshot_vllm_trace_metrics() -> dict[str, float | int | None]:
    metrics: dict[str, float | int | None] = {
        "num_requests_running": None,
        "num_requests_waiting": None,
        "kv_cache_usage_perc": None,
        "generation_tokens": None,
    }
    try:
        from vllm.v1.metrics.reader import get_metrics_snapshot

        name_map = {
            "vllm:num_requests_running": "num_requests_running",
            "vllm:num_requests_waiting": "num_requests_waiting",
            "vllm:kv_cache_usage_perc": "kv_cache_usage_perc",
            "vllm:generation_tokens": "generation_tokens",
        }
        for metric in get_metrics_snapshot():
            name = getattr(metric, "name", None)
            target = name_map.get(name)
            if target is None:
                continue
            value = getattr(metric, "value", None)
            if value is None:
                values = getattr(metric, "values", None)
                if isinstance(values, list) and values:
                    value = values[-1]
            if value is not None:
                metrics[target] = value
    except Exception:
        metrics["snapshot_error"] = 1
    return metrics


def _sharded_trace_path(path: str) -> str:
    """Per-process shard of the configured trace path.

    Many vLLM server processes across nodes append to this file; O_APPEND is
    not atomic on NFS, so a shared file gets torn/interleaved lines (~6% of
    rows observed). Each process writes its own shard; concatenate
    `<base>.shard-*.jsonl` after the run to produce the configured file.
    """
    import socket

    base, ext = os.path.splitext(path)
    return f"{base}.shard-{socket.gethostname()}-{os.getpid()}{ext}"


def _append_trace_jsonl(path: str, record: dict[str, Any]) -> None:
    path = _sharded_trace_path(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    line = json.dumps(record, separators=(",", ":"), default=str) + "\n"
    encoded = line.encode("utf-8")
    with _TRACE_WRITE_LOCK:
        fd = os.open(path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o644)
        try:
            os.write(fd, encoded)
        finally:
            os.close(fd)


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

        self.inflight_batch_sizes: list[int] = []
        self.num_pending_samples: list[int] = []
        self.kv_cache_usage_perc: list[float] = []
        self.generation_tokens: list[int] = []

        def _logger_loop():
            # Delay a little to let engine settle
            time.sleep(min(2.0, interval_s))
            while True:
                try:
                    for m in get_metrics_snapshot():
                        with self._vllm_metrics_lock:
                            if isinstance(m, Gauge):
                                # Log the vllm inflight batch sizes
                                if m.name == "vllm:num_requests_running":
                                    self.inflight_batch_sizes.append(int(m.value))
                                # Log the vllm pending number of requests in the queue
                                elif m.name == "vllm:num_requests_waiting":
                                    self.num_pending_samples.append(int(m.value))
                                # Log the vllm kv cache usage
                                elif m.name == "vllm:kv_cache_usage_perc":
                                    self.kv_cache_usage_perc.append(float(m.value))
                            elif isinstance(m, Counter):
                                if m.name == "vllm:generation_tokens":
                                    self.generation_tokens.append(int(m.value))
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

    def get_vllm_logger_metrics(self) -> dict[str, Any]:
        if not self.cfg["vllm_cfg"].get("enable_vllm_metrics_logger", False):
            return {}

        with self._vllm_metrics_lock:
            metric = {
                "inflight_batch_sizes": copy.deepcopy(self.inflight_batch_sizes),
                "num_pending_samples": copy.deepcopy(self.num_pending_samples),
                "kv_cache_usage_perc": copy.deepcopy(self.kv_cache_usage_perc),
                "generation_tokens": copy.deepcopy(self.generation_tokens),
            }
        return metric

    def clear_vllm_logger_metrics(self) -> None:
        if not self.cfg["vllm_cfg"].get("enable_vllm_metrics_logger", False):
            return

        with self._vllm_metrics_lock:
            self.inflight_batch_sizes = []
            self.num_pending_samples = []
            self.kv_cache_usage_perc = []
            self.generation_tokens = []

    async def post_init_async(self):
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

    def _mocker_request_server_trace_jsonl_path(self) -> str | None:
        vllm_cfg = self.cfg.get("vllm_cfg", {})
        path = vllm_cfg.get("mocker_request_server_trace_jsonl")
        if path:
            return str(path)

        path = self.cfg.get("mocker_request_server_trace_jsonl")
        if path:
            return str(path)

        return None

    def _build_request_trace_base_record(
        self,
        raw_request,
        request: Any,
        *,
        arrival_unix_ms: float,
        arrival_monotonic_ms: float,
    ) -> dict[str, Any]:
        request_dump = _model_dump(request)
        client_host = getattr(getattr(raw_request, "client", None), "host", None)
        client_port = getattr(getattr(raw_request, "client", None), "port", None)
        trace_id = raw_request.headers.get("x-request-id") or str(uuid.uuid4())
        arrival_metrics = _snapshot_vllm_trace_metrics()
        max_completion_tokens = getattr(request, "max_completion_tokens", None)
        deprecated_max_tokens = getattr(request, "max_tokens", None)
        effective_max_tokens = (
            max_completion_tokens
            if max_completion_tokens is not None
            else deprecated_max_tokens
        )
        return {
            "trace_schema_version": 1,
            "event": "vllm_chat_completion",
            "trace_id": trace_id,
            "request_id": trace_id,
            "worker_id": getattr(self, "trace_worker_id", None),
            "chosen_worker": getattr(self, "trace_worker_id", None),
            "worker_seed": getattr(self, "trace_worker_seed", None),
            "bundle_indices": getattr(self, "trace_bundle_indices", None),
            "base_url": getattr(self, "base_url", None),
            "node_ip": _get_node_ip_local(),
            "client_host": client_host,
            "client_port": client_port,
            "http_path": str(getattr(raw_request, "url", "")),
            "request_header_id": raw_request.headers.get("x-request-id"),
            "request_hash": _json_hash(request_dump),
            "model": getattr(request, "model", None),
            "max_tokens": effective_max_tokens,
            "max_completion_tokens": max_completion_tokens,
            "temperature": getattr(request, "temperature", None),
            "top_p": getattr(request, "top_p", None),
            "stream": bool(getattr(request, "stream", False)),
            "arrival_timestamp_ms": arrival_unix_ms,
            "arrival_monotonic_ms": arrival_monotonic_ms,
            "arrival_metrics": arrival_metrics,
            "num_requests_running_at_arrival": arrival_metrics.get(
                "num_requests_running"
            ),
            "num_requests_waiting_at_arrival": arrival_metrics.get(
                "num_requests_waiting"
            ),
            "kv_cache_usage_perc_at_arrival": arrival_metrics.get(
                "kv_cache_usage_perc"
            ),
        }

    def _finish_request_trace_record(
        self,
        record: dict[str, Any],
        *,
        completion_unix_ms: float,
        completion_monotonic_ms: float,
        response: Any = None,
        error: Any = None,
        status_code: int | None = None,
        streaming: bool = False,
        prompt_token_ids: list[int] | None = None,
        generation_token_ids: list[int] | None = None,
    ) -> dict[str, Any]:
        response_dump = _model_dump(response)
        if prompt_token_ids is None:
            prompt_token_ids = _find_first_token_list(
                response_dump, "prompt_token_ids"
            )
        if generation_token_ids is None:
            generation_token_ids = _generation_token_ids_from_response_dump(
                response_dump
            )
        usage = _usage_from_response(response_dump)
        response_id = response_dump.get("id") if response_dump else None
        completion_metrics = _snapshot_vllm_trace_metrics()
        prompt_tokens = _coerce_int(usage["prompt_tokens"])
        if prompt_tokens is None and prompt_token_ids is not None:
            prompt_tokens = len(prompt_token_ids)
        completion_tokens = _coerce_int(usage["completion_tokens"])
        if completion_tokens is None and generation_token_ids is not None:
            completion_tokens = len(generation_token_ids)
        total_tokens = _coerce_int(usage["total_tokens"])
        if (
            total_tokens is None
            and prompt_tokens is not None
            and completion_tokens is not None
        ):
            total_tokens = prompt_tokens + completion_tokens
        prompt_token_hash = _token_ids_hash(prompt_token_ids)
        generation_token_hash = _token_ids_hash(generation_token_ids)
        record.update(
            {
                "completion_timestamp_ms": completion_unix_ms,
                "completion_monotonic_ms": completion_monotonic_ms,
                "duration_ms": completion_monotonic_ms
                - record["arrival_monotonic_ms"],
                "completion_metrics": completion_metrics,
                "num_requests_running_at_completion": completion_metrics.get(
                    "num_requests_running"
                ),
                "num_requests_waiting_at_completion": completion_metrics.get(
                    "num_requests_waiting"
                ),
                "kv_cache_usage_perc_at_completion": completion_metrics.get(
                    "kv_cache_usage_perc"
                ),
                "status_code": status_code,
                "streaming": streaming,
                "response_id": response_id,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "output_tokens": completion_tokens,
                "input_length": prompt_tokens,
                "output_length": completion_tokens,
                "total_tokens": total_tokens,
                "extracted_prompt_tokens": (
                    len(prompt_token_ids)
                    if prompt_token_ids is not None
                    else None
                ),
                "extracted_generation_tokens": (
                    len(generation_token_ids)
                    if generation_token_ids is not None
                    else None
                ),
                "prompt_token_ids": prompt_token_ids,
                "generation_token_ids": generation_token_ids,
                "prompt_token_hash": prompt_token_hash,
                "generation_token_hash": generation_token_hash,
                "trace_join_key": (
                    f"{prompt_token_hash}:{generation_token_hash}"
                    if prompt_token_hash and generation_token_hash
                    else None
                ),
                "error": str(error) if error is not None else None,
            }
        )
        return record

    # ruff: noqa
    def _setup_vllm_openai_api_server(self, app: FastAPI) -> FastAPI:
        from copy import deepcopy
        from logging import Filter as LoggingFilter
        from logging import LogRecord
        from typing import List, Optional, Union

        from fastapi import Request
        from fastapi.responses import JSONResponse, StreamingResponse
        from vllm.entrypoints.openai.chat_completion.protocol import (
            ChatCompletionRequest,
            ChatCompletionResponse,
        )
        from vllm.entrypoints.openai.chat_completion.serving import (
            OpenAIServingChat,
        )
        from vllm.entrypoints.openai.engine.protocol import ErrorResponse
        from vllm.entrypoints.openai.models.protocol import BaseModelPath
        from vllm.entrypoints.openai.models.serving import OpenAIServingModels
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
        from vllm.tool_parsers.abstract_tool_parser import ToolParserManager
        from vllm.v1.engine.async_llm import logger as vllm_async_llm_logger

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

        trace_enabled = bool(self._mocker_request_server_trace_jsonl_path())
        trace_prompt_token_ids_by_request: dict[int, list[int]] = {}

        def _record_trace_prompt_token_ids(request, engine_prompts) -> None:
            if not trace_enabled or not engine_prompts:
                return
            if not (
                hasattr(request, "max_tokens")
                or hasattr(request, "max_completion_tokens")
            ):
                return
            engine_prompt = engine_prompts[0]
            if not isinstance(engine_prompt, dict):
                return
            prompt_token_ids = _as_int_token_list(
                engine_prompt.get("prompt_token_ids")
            )
            if prompt_token_ids is not None:
                trace_prompt_token_ids_by_request[id(request)] = prompt_token_ids

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
                for message in messages:
                    if message.get("tool_calls"):
                        message["tool_calls"] = list(message["tool_calls"])

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

                _record_trace_prompt_token_ids(request, res[1])

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
                _record_trace_prompt_token_ids(request, [engine_prompt])

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

        # vLLM 0.20 routes both /v1/chat/completions and /tokenize through
        # OpenAIServingRender.preprocess_chat, so the prefix-token override
        # belongs on the render subclass.
        worker_self = self

        class NeMoRLOpenAIServingChatMixin:
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

            trace_request_key = id(request)
            trace_path = self._mocker_request_server_trace_jsonl_path()
            trace_record = None
            if trace_path:
                trace_record = self._build_request_trace_base_record(
                    raw_request,
                    request,
                    arrival_unix_ms=time.time() * 1000.0,
                    arrival_monotonic_ms=time.monotonic() * 1000.0,
                )

            def pop_trace_prompt_token_ids() -> list[int] | None:
                return trace_prompt_token_ids_by_request.pop(
                    trace_request_key, None
                )

            try:
                generator = await openai_serving_chat.create_chat_completion(
                    request, raw_request
                )
            except asyncio.CancelledError:
                if trace_path and trace_record is not None:
                    _append_trace_jsonl(
                        trace_path,
                        self._finish_request_trace_record(
                            trace_record,
                            completion_unix_ms=time.time() * 1000.0,
                            completion_monotonic_ms=time.monotonic() * 1000.0,
                            error="request cancelled",
                            status_code=499,
                            streaming=bool(getattr(request, "stream", False)),
                            prompt_token_ids=pop_trace_prompt_token_ids(),
                        ),
                    )
                raise
            except VLLMValidationError as e:
                if trace_path and trace_record is not None:
                    _append_trace_jsonl(
                        trace_path,
                        self._finish_request_trace_record(
                            trace_record,
                            completion_unix_ms=time.time() * 1000.0,
                            completion_monotonic_ms=time.monotonic() * 1000.0,
                            error=e,
                            status_code=400,
                            streaming=bool(getattr(request, "stream", False)),
                            prompt_token_ids=pop_trace_prompt_token_ids(),
                        ),
                    )
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
            except Exception as e:
                if trace_path and trace_record is not None:
                    _append_trace_jsonl(
                        trace_path,
                        self._finish_request_trace_record(
                            trace_record,
                            completion_unix_ms=time.time() * 1000.0,
                            completion_monotonic_ms=time.monotonic() * 1000.0,
                            error=e,
                            status_code=500,
                            streaming=bool(getattr(request, "stream", False)),
                            prompt_token_ids=pop_trace_prompt_token_ids(),
                        ),
                    )
                raise

            if isinstance(generator, ErrorResponse):
                if trace_path and trace_record is not None:
                    _append_trace_jsonl(
                        trace_path,
                        self._finish_request_trace_record(
                            trace_record,
                            completion_unix_ms=time.time() * 1000.0,
                            completion_monotonic_ms=time.monotonic() * 1000.0,
                            response=generator,
                            error=getattr(
                                getattr(generator, "error", None),
                                "message",
                                None,
                            ),
                            status_code=generator.error.code,
                            streaming=False,
                            prompt_token_ids=pop_trace_prompt_token_ids(),
                        ),
                    )
                return JSONResponse(
                    content=generator.model_dump(), status_code=generator.error.code
                )

            elif isinstance(generator, ChatCompletionResponse):
                response_payload = model_dump_chat_response_with_routed_experts(
                    generator
                )
                if trace_path and trace_record is not None:
                    _append_trace_jsonl(
                        trace_path,
                        self._finish_request_trace_record(
                            trace_record,
                            completion_unix_ms=time.time() * 1000.0,
                            completion_monotonic_ms=time.monotonic() * 1000.0,
                            response=response_payload,
                            status_code=200,
                            streaming=False,
                            prompt_token_ids=pop_trace_prompt_token_ids(),
                        ),
                    )
                return JSONResponse(content=response_payload)

            if not trace_path or trace_record is None:
                return StreamingResponse(
                    content=generator, media_type="text/event-stream"
                )

            async def traced_stream():
                last_payload: dict[str, Any] | None = None
                stream_prompt_token_ids: list[int] | None = None
                stream_generation_token_ids: list[int] = []
                saw_stream_generation_token_ids = False
                stream_error: Exception | None = None
                try:
                    if hasattr(generator, "__aiter__"):
                        async for chunk in generator:
                            payload = _extract_sse_json_dict(chunk)
                            if payload is not None:
                                last_payload = payload
                                payload_prompt_token_ids = _find_first_token_list(
                                    payload, "prompt_token_ids"
                                )
                                if payload_prompt_token_ids is not None:
                                    stream_prompt_token_ids = payload_prompt_token_ids
                                payload_generation_token_ids = (
                                    _generation_token_ids_from_response_dump(payload)
                                )
                                if payload_generation_token_ids is not None:
                                    saw_stream_generation_token_ids = True
                                    stream_generation_token_ids.extend(
                                        payload_generation_token_ids
                                    )
                            yield chunk
                    else:
                        for chunk in generator:
                            payload = _extract_sse_json_dict(chunk)
                            if payload is not None:
                                last_payload = payload
                                payload_prompt_token_ids = _find_first_token_list(
                                    payload, "prompt_token_ids"
                                )
                                if payload_prompt_token_ids is not None:
                                    stream_prompt_token_ids = payload_prompt_token_ids
                                payload_generation_token_ids = (
                                    _generation_token_ids_from_response_dump(payload)
                                )
                                if payload_generation_token_ids is not None:
                                    saw_stream_generation_token_ids = True
                                    stream_generation_token_ids.extend(
                                        payload_generation_token_ids
                                    )
                            yield chunk
                except Exception as e:
                    stream_error = e
                    raise
                finally:
                    prompt_token_ids = pop_trace_prompt_token_ids()
                    if prompt_token_ids is None:
                        prompt_token_ids = stream_prompt_token_ids
                    _append_trace_jsonl(
                        trace_path,
                        self._finish_request_trace_record(
                            trace_record,
                            completion_unix_ms=time.time() * 1000.0,
                            completion_monotonic_ms=time.monotonic() * 1000.0,
                            response=last_payload,
                            error=stream_error,
                            status_code=(
                                500 if stream_error is not None else 200
                            ),
                            streaming=True,
                            prompt_token_ids=prompt_token_ids,
                            generation_token_ids=(
                                stream_generation_token_ids
                                if saw_stream_generation_token_ids
                                else None
                            ),
                        ),
                    )

            return StreamingResponse(
                content=traced_stream(), media_type="text/event-stream"
            )

        ########################################
        # /tokenize endpoint
        ########################################

        # This MRO is necessary i.e. NeMoRLOpenAIChatRequestMixin > TokenizeRequest
        class NeMoRLTokenizeChatRequest(
            NeMoRLOpenAIChatRequestMixin, TokenizeChatRequest
        ):
            required_prefix_token_ids: Optional[List[int]] = None

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

            # TODO: switch to update_weights_from_local_ipc_handles for better performance once collectively report_device_id is supported in asyncLLM initialization
            result_or_coro = await self.llm.collective_rpc(
                "update_weights_via_ipc_zmq", args=tuple()
            )

            if asyncio.iscoroutine(result_or_coro):
                worker_results = await result_or_coro
            else:
                worker_results = result_or_coro

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

            result_or_coro = await self.llm.collective_rpc(
                "update_weights_from_collective", args=tuple()
            )

            if asyncio.iscoroutine(result_or_coro):
                worker_results = await result_or_coro
            else:
                worker_results = result_or_coro

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

        await self.llm.reset_prefix_cache()
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

        # Reset the prefix cache to ensure that prefix cache is not reused after weights are updated
        await self.llm.reset_prefix_cache()
        # Reset the multimodal processor cache (sender side) so it stays in
        # sync with the receiver cache that vLLM clears internally during
        # sleep.  Without this, the sender thinks images are already cached on
        # the receiver and sends data=None, causing an assertion error.
        if hasattr(self.llm, "reset_mm_cache"):
            await self.llm.reset_mm_cache()
        await self.llm.sleep(level=1)

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

    async def shutdown(self) -> bool:
        """Clean up vLLM resources."""
        try:
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

            if self.server_thread is not None:
                from threading import Thread

                from uvicorn import Server

                self.http_server: Server
                self.server_thread: Thread

                self.http_server.should_exit = True
                self.server_thread.join()

            return True
        except Exception as e:
            print(f"Error during vLLM shutdown: {e}")
            return False


@ray.remote(
    runtime_env={**get_nsight_config_if_pattern_matches("vllm_async_generation_worker")}
)  # pragma: no cover
class VllmAsyncGenerationWorker(VllmAsyncGenerationWorkerImpl):
    pass
