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


"""HTTP serving and token capture for a vLLM engine client."""

import asyncio
import contextvars
import copy
import logging
import secrets
import socket
import threading
from collections.abc import Awaitable, Callable, Coroutine
from typing import Any, AsyncGenerator, Literal, Optional

import torch
import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse

from nemo_rl.data.captured_media import (
    CapturedMedia,
    CapturedMediaItem,
    MediaCaptureRejected,
    MediaColumnSpec,
    capture_processed_media,
    verified_media_items,
    verify_media_geometry,
)
from nemo_rl.data_plane.interfaces import DataPlaneConfig, LocalDataPlaneConfig
from nemo_rl.data_plane.tq_token_sink import MediaMetadataIntegrityError
from nemo_rl.distributed.virtual_cluster import (
    DEFAULT_GENERATION_PORT_RANGE_HIGH,
    DEFAULT_GENERATION_PORT_RANGE_LOW,
    _get_free_port_local,
    _get_node_ip_local,
)
from nemo_rl.experience.route_assembly import boundary_route_rows
from nemo_rl.models.generation.capture_context import decide_capture_input
from nemo_rl.models.generation.openai_server_utils import (
    PrefixSplice,
    splice_prefix_tokens,
)
from nemo_rl.models.generation.prefix_recovery import (
    CaptureRegistry,
    NoPrefixRecovery,
    PrefixRecovery,
    PrefixRecoveryParts,
)
from nemo_rl.models.generation.prefix_recovery.restore import (
    _build_restored_prefix_terminal_output,
    _classify_restored_prefix_terminal,
    _full_generator_signature,
    _RestoredPrefixTerminal,
    mark_restored_terminal,
    plan_restore,
    restore_reply_usage,
    restored_terminal,
    wrap_parser_for_restore,
)
from nemo_rl.models.generation.prefix_recovery.state import _RequestCaptureState
from nemo_rl.models.generation.score_centering import validate_score_centering_request
from nemo_rl.models.generation.vllm.captured_chat_logprobs import (
    CapturedChatLogprobsMixin,
)
from nemo_rl.models.generation.vllm.config import VllmConfig
from nemo_rl.models.generation.vllm.generation_cut_route import (
    register_generation_cut_route,
)
from nemo_rl.models.generation.vllm.progress_reader import VllmProgressReader
from nemo_rl.models.generation.vllm.utils import (
    attach_routed_experts_to_chat_response_choices,
    attach_sampler_topk_logprobs_to_chat_response_choices,
    attach_token_information_to_chat_response_choices,
    check_sampling_param,
    model_dump_chat_response_with_dynamic_message_fields,
    remap_multimodal_placeholders,
)
from nemo_rl.utils.routed_experts_codec import routed_experts_tensor_metadata

LOGGER = logging.getLogger(__name__)

# The captured model call the current HTTP request serves. Set when capture
# begins the call, in the request's task, and read by the engine client when the
# same task submits the call to the engine, which links the engine request ID to
# the captured call.
_CAPTURED_MODEL_CALL: contextvars.ContextVar[Optional[str]] = contextvars.ContextVar(
    "nemo_rl_captured_model_call", default=None
)


# Paths the bearer-token check protects: the OpenAI-compatible routes this module
# registers, which NeMo-Gym's model server calls. The sparse-refit routes the
# receiver registers on the same app carry their own header token
# (vllm_sparse_refit.setup_api_server) and are not covered.
HTTP_SERVER_PROTECTED_PATH_PREFIXES = ("/v1/", "/tokenize")


def install_http_server_api_key_check(app: FastAPI, api_key: str) -> None:
    """Refuse OpenAI-route requests that do not carry ``Authorization: Bearer <api_key>``.

    Registers an HTTP middleware on ``app``. For a request whose path starts with one
    of ``HTTP_SERVER_PROTECTED_PATH_PREFIXES``, the ``Authorization`` header is
    compared in constant time with ``Bearer <api_key>``; a missing or different
    value is answered with HTTP 401 in OpenAI's error envelope and never reaches the
    route. Requests to other paths pass through unchanged.
    """
    expected = f"Bearer {api_key}".encode()

    @app.middleware("http")
    async def _require_bearer_token(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        if request.url.path.startswith(HTTP_SERVER_PROTECTED_PATH_PREFIXES):
            supplied = (request.headers.get("Authorization") or "").encode()
            if not secrets.compare_digest(supplied, expected):
                return JSONResponse(
                    status_code=401,
                    content={
                        "error": {
                            "message": "missing or invalid bearer token",
                            "type": "authentication_error",
                            "code": 401,
                        }
                    },
                )
        return await call_next(request)


class _AsyncLLMHTTPClient:
    """Keep HTTP generation on the loop that owns AsyncLLM request state.

    The engine-client surface is explicit. Do not add a ``__getattr__`` fallback.
    Add each new member here and decide whether it must run on the engine loop.
    """

    def __init__(
        self,
        engine_client: Any,
        engine_loop: asyncio.AbstractEventLoop,
        *,
        flat_logprobs: bool,
    ):
        self._engine_client = engine_client
        self._engine_loop = engine_loop
        self._flat_logprobs = flat_logprobs
        # Engine request ID -> captured model call ID, while the request runs.
        self._captured_requests: dict[str, str] = {}
        # Called with each request ID once its consumer is done with it.
        self._request_done_listeners: list[Callable[[str], None]] = []
        self.model_config = engine_client.model_config
        self.renderer = engine_client.renderer
        self.input_processor = engine_client.input_processor
        self.vllm_config = engine_client.vllm_config

    async def _run_on_engine_loop(
        self, operation: Callable[[], Coroutine[Any, Any, Any]]
    ) -> Any:
        if asyncio.get_running_loop() is self._engine_loop:
            return await operation()

        future = asyncio.run_coroutine_threadsafe(operation(), self._engine_loop)
        try:
            return await asyncio.wrap_future(future)
        except asyncio.CancelledError:
            future.cancel()
            raise

    def generate(
        self,
        prompt: Any,
        sampling_params: Any,
        request_id: str,
        **kwargs: Any,
    ) -> AsyncGenerator[Any, None]:
        # Flat logprobs avoid per-token Logprob objects that cause long GC pauses.
        # Not for prompt logprobs: vLLM's -inf clamp edits copies of flat storage.
        if (
            self._flat_logprobs
            and sampling_params.logprobs is not None
            and sampling_params.prompt_logprobs is None
        ):
            sampling_params.flat_logprobs = True
        return self._generate(prompt, sampling_params, request_id, kwargs)

    async def _generate(
        self,
        prompt: Any,
        sampling_params: Any,
        request_id: str,
        kwargs: dict[str, Any],
    ) -> AsyncGenerator[Any, None]:
        terminal = restored_terminal()
        if terminal is not None:
            yield _build_restored_prefix_terminal_output(request_id, terminal)
            return

        iterator = None
        completed = False
        model_call_id = _CAPTURED_MODEL_CALL.get()
        if model_call_id is not None:
            self._captured_requests[request_id] = model_call_id

        async def next_output() -> Any:
            nonlocal iterator
            if iterator is None:
                iterator = self._engine_client.generate(
                    prompt, sampling_params, request_id, **kwargs
                )
            return await anext(iterator)

        try:
            while True:
                try:
                    yield await self._run_on_engine_loop(next_output)
                except StopAsyncIteration:
                    completed = True
                    return
        finally:
            try:
                self._captured_requests.pop(request_id, None)
                if not completed:
                    try:
                        await self._run_on_engine_loop(
                            lambda: self._engine_client.abort(request_id)
                        )
                    except Exception:
                        LOGGER.exception("Failed to abort vLLM request %s", request_id)
            finally:
                for listener in self._request_done_listeners:
                    listener(request_id)

    # Members a VllmProgressReader uses; they read this client's own state.
    @property
    def output_processor(self) -> Any:
        return self._engine_client.output_processor

    def captured_request_ids(self) -> dict[str, str]:
        """Engine request ID to captured model call ID, for running requests."""
        # One C-level copy under the GIL; the HTTP thread may add entries.
        return dict(self._captured_requests)

    def is_captured(self, request_id: str) -> bool:
        return request_id in self._captured_requests

    def add_request_done_listener(self, listener: Callable[[str], None]) -> None:
        """Call ``listener(request_id)`` once a request's consumer is done with it."""
        self._request_done_listeners.append(listener)

    # These members only read engine status or immutable configuration. Running
    # them on the engine loop added a cross-thread wait to each HTTP request.
    @property
    def errored(self) -> bool:
        return self._engine_client.errored

    @property
    def dead_error(self) -> BaseException:
        return self._engine_client.dead_error

    def check_admission(self, n: int = 1, request_id: str | None = None) -> None:
        """Queue-limit admission check vLLM >= 0.29 runs before every response.

        ``OpenAIServing._preflight`` calls this (vllm-project/vllm#49445,
        ``max_num_queued_reqs`` / ``max_num_queued_tokens``); without it every
        chat completion 500s with ``'_AsyncLLMHTTPClient' object has no
        attribute 'check_admission'``. It only reads scheduler config and
        unfinished-request counters, so it stays off the engine loop like the
        other status reads above. Raises vLLM's HTTP-mapped overflow errors.
        """
        self._engine_client.check_admission(n, request_id=request_id)

    async def is_tracing_enabled(self) -> bool:
        return await self._engine_client.is_tracing_enabled()

    async def captured_call_progress(self) -> list[list[Any]]:
        """Return ``[model_call_id, generated_tokens]`` for each running captured call.

        Reads vLLM's output-processor request state, which has no public API, on
        the engine loop. Returns nothing if this vLLM keeps that state elsewhere.
        """

        async def read() -> list[list[Any]]:
            states = getattr(
                getattr(self._engine_client, "output_processor", None),
                "request_states",
                None,
            )
            if states is None:
                return []
            progress = []
            for state in list(states.values()):
                model_call_id = self._captured_requests.get(state.external_req_id)
                if model_call_id is not None and state.detokenizer is not None:
                    progress.append(
                        [model_call_id, state.detokenizer.num_output_tokens()]
                    )
            return progress

        return await self._run_on_engine_loop(read)


class VllmHTTPFrontend:
    """Serve OpenAI requests and host token capture for one vLLM engine client."""

    def __init__(
        self,
        cfg: VllmConfig,
        *,
        engine_client: Any,
        engine_args: Any,
        engine_loop: asyncio.AbstractEventLoop,
        routed_experts_dtype: torch.dtype,
        listener: socket.socket | None = None,
        node_ip: str | None = None,
        register_routes: Callable[[FastAPI], None] | None = None,
    ) -> None:
        """Initialize a frontend attached to an engine and its owning event loop."""
        self.cfg = cfg
        self.llm = engine_client
        self.llm_async_engine_args = engine_args
        self.routed_experts_dtype = routed_experts_dtype
        self._register_routes = register_routes
        self._reserved_socket = listener
        self._reserved_node_ip = node_ip
        self.server_thread: threading.Thread | None = None
        self.base_url: str | None = None
        self.http_server: uvicorn.Server | None = None
        self._http_engine_client = _AsyncLLMHTTPClient(
            engine_client,
            engine_loop,
            flat_logprobs=bool(cfg["vllm_cfg"].get("flat_logprobs")),
        )
        # Ledger-authoritative token capture. The weight version is stamped
        # per model call at begin_call time and rotated by the
        # set_rollout_weight_version fan-out from the SC's _sync_weights.
        self.token_capture = None
        self._rollout_weight_version = 0
        # In-flight captured calls, by request and by Gym's model-call ID.
        self._capture_registry = CaptureRegistry()
        self._capture_sink: Any | None = None
        # Generation-prefix cuts and restores; a no-op unless setup_token_capture
        # enables them.
        self._prefix_recovery: PrefixRecovery | NoPrefixRecovery = NoPrefixRecovery()
        # Built once per engine: it wraps vLLM's output processor.
        self._progress_reader: Any | None = None
        self._capture_media = False
        self._capture_patch_size: int | None = None
        self._capture_image_token_id: int | None = None
        # TQTokenSource installed by setup_token_capture. The media path reads
        # parent-chain rows through it directly; prefix resolution goes through
        # the shared ChainPrefixCache below.
        self._staging_source: Any | None = None
        self._capture_dp_client: Any | None = None
        # Resolved staging-chain prefixes, shared implementation with the Megatron
        # preparer. Installed by setup_token_capture; fetch runs on executor threads.
        # Deferred import: tq_token_sink pulls in the data-plane stack.
        from nemo_rl.data_plane.tq_token_sink import ChainPrefixCache

        self._chain_prefix = ChainPrefixCache()
        # policy.generation.top_logprobs: sampler top-k log-probs staged next to each
        # captured call's delta; 0 leaves captured calls as they are.
        self._capture_top_logprobs = 0

    def _return_routed_experts_enabled(self) -> bool:
        engine_args = getattr(self, "llm_async_engine_args", None)
        if bool(getattr(engine_args, "enable_return_routed_experts", False)):
            return True
        return bool(
            self.cfg.get("vllm_kwargs", {}).get("enable_return_routed_experts", False)
        )

    def _capture_admission(self, request: Any) -> Any | None:
        """Parse the ledger's ``ng_capture`` context into a ``CaptureAdmission``.

        Returns None unless capture is installed and the request carries the
        context. The dict itself is never mutated: the admission is the typed,
        read-only contract that the prefix resolution and ``begin_call`` share.
        """
        context = getattr(request, "ng_capture", None)
        if self.token_capture is None or not context:
            return None
        # Deferred: nemo_gym is an optional extra absent in non-gym runs.
        from nemo_gym.token_id_capture.staging.records import CaptureAdmission

        return CaptureAdmission.model_validate(context)

    def install_token_capture(self, capture: Any) -> None:
        """Gym's ``install_capture`` seam (the ``CaptureHost`` contract)."""
        self.token_capture = capture

    async def setup_token_capture(
        self,
        dp_cfg: DataPlaneConfig | LocalDataPlaneConfig,
        staging_partition: str,
        *,
        capture_media: bool = False,
        top_logprobs: int = 0,
        generation_prefix_cuts_enabled: bool = False,
        generation_cut_control_token: str | None = None,
        generation_cut_control_timeout_s: float | None = None,
    ) -> MediaColumnSpec | None:
        """Install ledger-authoritative capture on this frontend.

        Returns:
            The pinned media column spec, or None for text-only capture.
        """
        # Deferred: nemo_gym is an optional extra absent in non-gym runs.
        from nemo_gym.token_id_capture.adapters.vllm import VLLMCaptureAdapter
        from nemo_gym.token_id_capture.staging import install_capture

        from nemo_rl.data_plane import build_data_plane_client
        from nemo_rl.data_plane.tq_token_sink import TQTokenSink, TQTokenSource

        # Own no segment: a save would otherwise wait on this actor, whose event
        # loop is busy serving rollouts. Capture writes land in other segments.
        dp_client = build_data_plane_client(dp_cfg, bootstrap=False, segment_size=0)
        self._capture_dp_client = dp_client
        # The Omni processor emits pixels in the engine's model dtype; the
        # sink pins its media column to it so text-call sentinels never
        # introduce a second dtype (TQ keeps one dtype per field).
        pixel_dtype = self.llm.model_config.dtype if capture_media else None
        sink = TQTokenSink(
            dp_client,
            staging_partition=staging_partition,
            capture_media=capture_media,
            media_pixel_dtype=pixel_dtype,
        )
        self._capture_top_logprobs = int(top_logprobs)
        if generation_prefix_cuts_enabled and not generation_cut_control_token:
            raise ValueError(
                "generation-prefix cuts require a non-empty control bearer token"
            )
        media_spec = None
        if capture_media:
            # Omni-only: a new processor family must also change setup.py (driver
            # checks), captured_media.py (_processed_omni_tensors, pack_images,
            # capture_processed_media), tq_token_sink.py (MEDIA_TENSOR_COLUMNS,
            # validate_media_tensors, fetch_media) and rollout_reassembler.py
            # (_concat_media, _trainer_media).
            # Optional engine/Gym capabilities are checked only on VLM workers.
            from vllm.model_executor.models.nano_nemotron_vl import (
                NanoNemotronVLProcessingInfo,
            )

            info = self.llm.renderer.get_mm_processor().info
            if (
                not isinstance(info, NanoNemotronVLProcessingInfo)
                or not info.is_dynamic_tiler
            ):
                raise ValueError(
                    "Media capture requires vLLM's Omni dynamic-resolution processor"
                )
            if info.get_video_pruning_rate():
                raise ValueError(
                    "Omni media capture does not support video token pruning"
                )
            context_ids = info.get_hf_processor()._img_context_token_ids
            if len(context_ids) != 1:
                raise ValueError(
                    "Omni media capture requires one image-context token ID"
                )
            self._capture_image_token_id = int(context_ids[0])
            self._capture_patch_size = int(info.get_hf_config().patch_size)
            media_spec = MediaColumnSpec(
                pixel_dtype=self.llm.model_config.dtype,
                patch_size=self._capture_patch_size,
            )
        self._capture_media = capture_media
        self._capture_sink = sink
        source = TQTokenSource(
            dp_client, staging_partition=staging_partition, capture_media=capture_media
        )
        self._staging_source = source
        self._chain_prefix.install(source)
        install_capture(
            self,
            sink=sink,
            weight_version_fn=lambda: self._rollout_weight_version,
            adapter=VLLMCaptureAdapter(),
        )
        self._prefix_recovery.shutdown()
        self._prefix_recovery = NoPrefixRecovery()
        if generation_prefix_cuts_enabled:
            if self._progress_reader is None:
                self._progress_reader = VllmProgressReader(self._http_engine_client)
            self._prefix_recovery = PrefixRecovery(
                PrefixRecoveryParts(
                    capture=self.token_capture,
                    sink=sink,
                    staging_source=source,
                    registry=self._capture_registry,
                    # Cuts must land on character boundaries.
                    tokenizer=self.llm.renderer.get_tokenizer(),
                    weight_version=lambda: self._rollout_weight_version,
                    control_token=generation_cut_control_token,
                    control_timeout_s=generation_cut_control_timeout_s,
                    progress_reader=self._progress_reader,
                )
            )
        return media_spec

    def get_data_plane_snapshot(self) -> dict[str, Any] | None:
        """The capture client's data-plane counters, ``None`` when it has none."""
        from nemo_rl.data_plane.observability import is_metrics_client

        if not is_metrics_client(self._capture_dp_client):
            return None
        return self._capture_dp_client.snapshot(reset_step_window=True)

    async def set_rollout_weight_version(self, version: int) -> None:
        """Rotate the weight version stamped on subsequent captured calls."""
        self._rollout_weight_version = int(version)

    async def captured_call_progress(self) -> Optional[dict[str, Any]]:
        """Report how far each running captured call got on the current weights.

        Called while generation is paused for a refit, so the counts are where
        each call switches to the new weights. ``None`` on ranks that serve no
        captured calls.
        """
        if self.token_capture is None:
            return None
        return {
            "weight_version": self._rollout_weight_version,
            "calls": await self._http_engine_client.captured_call_progress(),
        }

    def _begin_request_capture(
        self,
        request: Any,
        prompt_token_ids: list[int],
        *,
        admission: Any | None = None,
        prefix_token_ids: list[int] | None = None,
        media: CapturedMedia | None = None,
        generation_cut: Any | None = None,
        resumed_generation_token_ids: list[int] | None = None,
    ) -> None:
        """Admit one ledger-forwarded call into the capture layer.

        Called from preprocess_chat once the exact engine prompt is known
        (post-splice in token-in mode, full render in text mode). No-op
        unless capture is installed and the request carries the ledger's
        ``ng_capture`` context.

        ``prefix_token_ids`` is the prefix resolved by
        :meth:`_resolve_admission_prefix`. Gym's ``begin_call`` requires it for
        a ``staging_chain`` admission and checks its length against
        ``prev_len``; violations raise ``CaptureError`` before any state is kept.
        """
        capture = self.token_capture
        if capture is None or getattr(request, "ng_measure_only", False):
            return
        if admission is None:
            admission = self._capture_admission(request)
            if admission is None:
                return
        if self._capture_top_logprobs:
            validate_score_centering_request(
                request, num_candidates=self._capture_top_logprobs
            )
        call = capture.begin_call(
            admission,
            prefix_token_ids=prefix_token_ids,
            generation_cut=generation_cut,
            generation_cut_staging_keys=(
                admission.generation_cut.staging_keys
                if admission.generation_cut is not None
                else None
            ),
            stream=bool(getattr(request, "stream", False)),
        )
        _CAPTURED_MODEL_CALL.set(call.model_call_id)
        state = _RequestCaptureState(
            call,
            prompt_token_ids,
            media=media,
            resumed_generation_token_ids=resumed_generation_token_ids or (),
            effective_output_limit=(
                admission.generation_cut.effective_output_limit
                if admission.generation_cut is not None
                else None
            ),
            staging_keys=(
                admission.generation_cut.staging_keys
                if admission.generation_cut is not None
                else ()
            ),
        )
        self._capture_registry.add(request, state)
        # Counted once Gym's begin_call has accepted a restored cut.
        self._prefix_recovery.on_begin(admission)

    def _capture_request_media(
        self,
        engine_prompt: dict[str, Any],
        *,
        admission: Any | None,
        splice: PrefixSplice | None = None,
        verify_retained_media: bool = False,
    ) -> CapturedMedia | None:
        """Run off-loop: resolve retained geometry and snapshot processed pixels."""
        if admission is None:
            return None
        if not self._capture_media:
            if engine_prompt.get("mm_placeholders") or engine_prompt.get("mm_kwargs"):
                raise MediaCaptureRejected(
                    "Multimodal token capture requires media capture setup"
                )
            return None
        retained: tuple[CapturedMediaItem, ...] = ()
        retained_tensors = () if verify_retained_media else None
        if admission.parent_call_id is not None:
            # Optional Gym dependency: this method only runs on captured calls.
            # Gym owns the media_spans key its adapter copies into the extras.
            from nemo_gym.token_id_capture.adapters.vllm import MEDIA_SPANS_FIELD
            from nemo_gym.token_id_capture.staging.digest import compute_chain_hash
            from nemo_gym.token_id_capture.staging.records import staging_key

            source = self._staging_source
            if source is None:
                raise RuntimeError("Media capture staging source is not initialized")
            try:
                if admission.staging_chain:
                    calls = source.fetch_for_finalization(
                        list(admission.staging_chain),
                        include_route_fragments=verify_retained_media,
                    )
                else:
                    # Inline token admissions still have receipt-owned parent keys.
                    calls, visited = [], set()
                    parent = admission.parent_call_id
                    while parent is not None:
                        if parent in visited:
                            raise MediaCaptureRejected("Cycle in retained media chain")
                        visited.add(parent)
                        call = source.fetch_for_finalization(
                            [staging_key(admission.capture_key, parent)],
                            include_route_fragments=verify_retained_media,
                        )[0]
                        calls.append(call)
                        parent = call.snapshot.parent_call_id
                    calls.reverse()
            except (KeyError, MediaMetadataIntegrityError) as error:
                raise MediaCaptureRejected(
                    f"Invalid retained media metadata: {error}"
                ) from error
            parent, length, chain_hash = None, 0, None
            for call in calls:
                snapshot = call.snapshot
                if (
                    snapshot.capture_key != admission.capture_key
                    or snapshot.parent_call_id != parent
                    or snapshot.prev_len != length
                    or snapshot.chain_hash
                    != compute_chain_hash(chain_hash, snapshot.token_ids_delta)
                ):
                    raise MediaCaptureRejected("Invalid retained media call chain")
                parent, length, chain_hash = (
                    snapshot.model_call_id,
                    snapshot.cum_len,
                    snapshot.chain_hash,
                )
            if (parent, length, chain_hash) != (
                admission.parent_call_id,
                admission.prev_len,
                admission.parent_chain_hash,
            ):
                raise MediaCaptureRejected(
                    "Retained image chain does not match capture admission"
                )
            descriptors = {}
            for call in calls:
                if verify_retained_media:
                    items = verified_media_items(call)
                else:
                    extras = call.extras or {}
                    if MEDIA_SPANS_FIELD not in extras:
                        raise MediaCaptureRejected(
                            "Retained vLLM media spans are missing"
                        )
                    items = tuple(
                        CapturedMediaItem.from_dict(value)
                        for value in extras[MEDIA_SPANS_FIELD]
                    )
                    for item in items:
                        item.verify_tokens(
                            call.snapshot.token_ids_delta, origin=call.snapshot.prev_len
                        )
                descriptors[call.snapshot.model_call_id] = items
            retained = tuple(item for items in descriptors.values() for item in items)
            if verify_retained_media:
                media_calls = [call for call in calls if call.media_present]
                retained_tensors = (
                    tuple(source.fetch_media(media_calls)) if media_calls else ()
                )
                for call, tensors in zip(media_calls, retained_tensors, strict=True):
                    verify_media_geometry(
                        descriptors[call.snapshot.model_call_id], tensors
                    )
        return capture_processed_media(
            engine_prompt,
            prev_len=admission.prev_len,
            retained=retained,
            splice=splice,
            image_token_id=self._capture_image_token_id,
            patch_size=self._capture_patch_size,
            retained_tensors=retained_tensors,
        )

    def _fetch_chain_prefix(self, staging_chain: list[str]) -> list[int]:
        """Resolve a staging chain through the shared, cached TQ read."""
        return self._chain_prefix.fetch(staging_chain)

    def _resolve_admission_prefix(self, admission: Any) -> list[int]:
        """Resolve a ``CaptureAdmission`` to the flat prefix the engine prompt starts with."""
        # Deferred import, matching setup_token_capture.
        from nemo_rl.data_plane.tq_token_sink import resolve_admission_prefix

        return resolve_admission_prefix(admission, self._chain_prefix)

    def _enter_request_prefix(self, request: Any, prefix_token_ids: list[int]) -> None:
        """Attach the resolved prefix to the request through the capture adapter.

        ``VLLMCaptureAdapter.enter_prefix`` writes the engine-native field
        (``required_prefix_token_ids``) into a payload; the same fields are
        applied to the pydantic request so the existing prefix-splice branch
        of preprocess_chat handles staged and inline prefixes alike.
        """
        adapter = self.token_capture.adapter
        for field_name, value in adapter.enter_prefix({}, prefix_token_ids).items():
            setattr(request, field_name, value)

    @staticmethod
    def _delta_align_routed_experts(
        payload: dict[str, Any], *, prev_len: int, prompt_len: int, generated_len: int
    ) -> torch.Tensor | None:
        """Slice native routes for the delta plus boundary row and bind metadata."""
        choices = payload.get("choices") or []
        if len(choices) != 1 or not isinstance(choices[0], dict):
            return None
        choice = dict(choices[0])
        message = dict(choice.get("message") or {})
        routed = message.get("routed_experts")
        if routed is None:
            return None
        try:
            if not isinstance(routed, torch.Tensor):
                raise TypeError("captured routed_experts must be a native tensor")
            expected_full_len = prompt_len + generated_len
            if routed.dim() != 3 or routed.shape[0] != expected_full_len:
                raise ValueError(
                    f"route shape {tuple(routed.shape)} does not match engine sequence "
                    f"length {expected_full_len}"
                )
            start = prev_len - boundary_route_rows(prev_len)
            # Keep this as a view: capture uses a local client, and SimpleStorage
            # serializes only the slice's bytes. Message routes sent through
            # Ray/pickle use slice_routed_experts() to copy only their token span.
            routed = routed[start:]
            message["routed_experts"] = routed_experts_tensor_metadata(routed)
        except (IndexError, TypeError, ValueError) as error:
            LOGGER.warning(
                "dropping invalid routed_experts from staged capture: %s", error
            )
            message.pop("routed_experts", None)
            routed = None
        choice["message"] = message
        payload["choices"] = [choice]
        return routed

    def _finish_request_capture(self, request: Any, content: dict) -> dict:
        """Stage one canonical terminal row and retire its prefix chunks.

        Tokens, routes, and the call's new media tensors go to TQ in one write
        (routes and media ride as opaque attachments beside the record), so
        ``staged`` coords vouch for all and any failure is ``capture_failed`` at call
        time. Token ids, logprobs, and routes are stripped after staging, so
        the worker->gate hop carries the completion and coords only.
        Sampler tensors ride as opaque attachments in that same staging write.
        """
        sampler_topk = None
        for choice in content.get("choices") or []:
            message = choice.get("message")
            if isinstance(message, dict) and (
                "sampler_topk_token_ids" in message
                or "sampler_topk_logprobs" in message
            ):
                sampler_topk = (
                    message.pop("sampler_topk_token_ids", None),
                    message.pop("sampler_topk_logprobs", None),
                )
        state = self._capture_registry.get(request)
        if state is None:
            return content
        with state.lifecycle() as owned:
            # A failed call is retried from its task or its previous boundary,
            # never from these rows, so they go whether or not its row stages.
            obsolete_staging_keys = owned.begin_terminal()
            content = self._finish_request_capture_with_lifecycle_owned(
                state, request, content, sampler_topk=sampler_topk
            )
        self._prefix_recovery.retire(obsolete_staging_keys)
        return content

    def _abort_request_capture(
        self, request: Any, *, reason: str, blocking: bool = True
    ) -> bool:
        """Drop the in-flight capture state for a request that errored.

        A generation cut or a terminal write holds the call's lifecycle lock
        across a TQ round trip. With ``blocking=False`` the abort does not wait
        for it and returns False instead; it returns True once the call needs
        no abort.
        """
        state = self._capture_registry.get(request)
        if state is None:
            return True
        obsolete_staging_keys: tuple[str, ...] = ()
        with state.lifecycle() if blocking else state.try_lifecycle() as owned:
            if owned is None:
                return False
            if not owned.terminal_started:
                staging_keys = owned.begin_terminal()
                popped = self._capture_registry.pop(request)
                if popped is state and self.token_capture is not None:
                    coords = self.token_capture.fail_call(state.call, reason=reason)
                    self._prefix_recovery.remember_completed(
                        state.call.model_call_id, coords, 0
                    )
                    obsolete_staging_keys = staging_keys
        self._prefix_recovery.retire(obsolete_staging_keys)
        return True

    def _finish_request_capture_with_lifecycle_owned(
        self,
        state: _RequestCaptureState,
        request: Any,
        content: dict,
        *,
        sampler_topk: tuple[Any, Any] | None = None,
    ) -> dict:
        """Stage a terminal call while its lifecycle is owned."""
        call, prompt_token_ids = state.call, state.prompt_token_ids
        payload = dict(content)
        # vLLM's OpenAI response carries no prompt ids; the adapter reads the
        # preprocess-time engine prompt off the payload (see
        # nemo_gym.token_id_capture.adapters.vllm.extract_prompt_ids).
        payload["prompt_token_ids"] = prompt_token_ids
        if state.media is not None:
            # Optional Gym dependency: only captured media calls reach here.
            from nemo_gym.token_id_capture.adapters.vllm import MEDIA_SPANS_FIELD

            # Placeholder metadata (offsets, token hashes, sizes) rides the
            # digest-covered extras; the pixels themselves are attachments.
            payload[MEDIA_SPANS_FIELD] = [item.to_dict() for item in state.media.items]
        adapter = self.token_capture.adapter
        routed_experts = None
        generated_token_ids: list[int] = []
        if adapter is not None:
            try:
                generated_token_ids, _ = adapter.extract_generation(payload)
            except Exception:  # capture core will report the authoritative failure
                generated_token_ids = []
            routed_experts = self._delta_align_routed_experts(
                payload,
                prev_len=call.admission.prev_len,
                prompt_len=len(prompt_token_ids),
                generated_len=len(generated_token_ids),
            )
        attachments = dict(state.media.tensors or {}) if state.media is not None else {}
        if routed_experts is not None:
            attachments["routed_experts"] = routed_experts
        if sampler_topk is not None:
            attachments.update(
                sampler_topk_token_ids=sampler_topk[0],
                sampler_topk_logprobs=sampler_topk[1],
            )
        if self._capture_top_logprobs > 0 and sampler_topk is None:
            coords = self.token_capture.fail_call(
                call, reason="missing sampler top-k tensors"
            )
        else:
            coords = self.token_capture.complete_call_from_response(
                call,
                payload,
                attachments=attachments or None,
            )
        total_generation_token_count = len(generated_token_ids)
        if call.generation_cut is not None:
            prefix_tokens = sum(
                mask == 1.0 for mask in call.generation_cut.token_mask_delta
            )
            total_generation_token_count += prefix_tokens
            LOGGER.info(
                "generation prefix completed: rollout_id=%s model_call_id=%s "
                "source_model_call_id=%s prefix_tokens=%d tail_tokens=%d "
                "total_generation_tokens=%d",
                call.capture_key,
                call.model_call_id,
                call.generation_cut.model_call_id,
                prefix_tokens,
                len(generated_token_ids),
                total_generation_token_count,
            )
        progress = state.snapshot()
        effective_output_limit = progress.effective_output_limit
        terminal_finish_reason = progress.terminal_finish_reason
        terminal_stop_reason = progress.terminal_stop_reason
        if terminal_finish_reason is None:
            choices = content.get("choices") or []
            finish_reason = choices[0].get("finish_reason") if choices else None
            if finish_reason in ("stop", "length"):
                terminal_finish_reason = finish_reason
                terminal_stop_reason = choices[0].get("stop_reason")
        self._prefix_recovery.remember_completed(
            call.model_call_id,
            coords,
            total_generation_token_count,
            effective_output_limit=effective_output_limit,
            terminal_finish_reason=terminal_finish_reason,
            terminal_stop_reason=terminal_stop_reason,
        )
        self._capture_registry.pop(request)
        for choice in content.get("choices") or []:
            choice.pop("logprobs", None)
            choice.pop("routed_experts", None)
            # Token arrays and delta-aligned routes were staged to TQ above;
            # remove the serializer's message fields before the worker->gate hop.
            message = choice.get("message")
            if isinstance(message, dict):
                for field in (
                    "prompt_token_ids",
                    "generation_token_ids",
                    "generation_log_probs",
                    "routed_experts",
                ):
                    message.pop(field, None)
        content["ng_commit_coords"] = coords.model_dump()
        return content

    def _context_refusal_headers(self, request: Any) -> dict[str, str]:
        """Acknowledge a validation refusal after draining capture state, before staging."""
        admission = self._capture_admission(request)
        if admission is None:
            return {}
        # Deferred: Gym is optional outside framework-owned capture.
        from nemo_gym.token_id_capture.protocols import CAPTURE_REFUSAL_HEADER

        return {CAPTURE_REFUSAL_HEADER: admission.model_call_id}

    def _setup_vllm_openai_api_server(self, app: FastAPI) -> FastAPI:
        frontend_self = self
        from copy import deepcopy
        from logging import Filter as LoggingFilter
        from logging import LogRecord
        from typing import List, Optional, Union

        from fastapi import Request
        from fastapi.responses import JSONResponse, StreamingResponse
        from pydantic import PrivateAttr
        from vllm.entrypoints.chat_utils import load_chat_template
        from vllm.entrypoints.openai.chat_completion.protocol import (
            ChatCompletionRequest,
            ChatCompletionResponse,
        )
        from vllm.entrypoints.openai.chat_completion.serving import (
            OpenAIServingChat,
        )

        full_generator_signature = _full_generator_signature(
            OpenAIServingChat.chat_completion_full_generator
        )

        from vllm.entrypoints.openai.models.protocol import BaseModelPath
        from vllm.entrypoints.openai.models.serving import OpenAIServingModels

        # vLLM 0.29 moved this out of the openai package (vllm-project/vllm#54492).
        from vllm.entrypoints.serve.engine.protocol import ErrorResponse
        from vllm.entrypoints.serve.tokenize.protocol import (
            TokenizeChatRequest,
            TokenizeCompletionRequest,
            TokenizeResponse,
        )
        from vllm.entrypoints.serve.tokenize.serving import (
            ServingTokenization,
        )
        from vllm.entrypoints.serve.utils.api_utils import with_cancellation
        from vllm.exceptions import VLLMValidationError
        from vllm.reasoning.abs_reasoning_parsers import ReasoningParserManager
        from vllm.renderers.hf import HfRenderer
        from vllm.renderers.online_renderer import OnlineRenderer
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

        engine_client = self._http_engine_client
        model_config = self.llm_async_engine_args.create_model_config()
        # vLLM adds the generation config's EOS to every request's stop set
        # regardless of the configured stops, so a turn can end on either.
        model_stop_token_ids = set(self.cfg.get("stop_token_ids") or ())
        generation_eos_token_ids = model_config.try_get_generation_config().get(
            "eos_token_id"
        )
        if isinstance(generation_eos_token_ids, int):
            generation_eos_token_ids = [generation_eos_token_ids]
        model_stop_token_ids |= set(generation_eos_token_ids or ())
        base_model_paths = [
            BaseModelPath(
                name=model_config.served_model_name, model_path=model_config.model
            ),
            BaseModelPath(name=model_config.model, model_path=model_config.model),
        ]

        openai_serving_models_kwargs: dict[str, Any] = dict(
            engine_client=engine_client,
            base_model_paths=base_model_paths,
            lora_modules=None,
        )
        openai_serving_models = OpenAIServingModels(**openai_serving_models_kwargs)

        class NeMoRLOpenAIChatRequestMixin:
            required_prefix_token_ids: list[int] | None

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
            def _set_max_tokens(self, request: Any, max_tokens: int) -> None:
                """Set the request's max output tokens.

                Mutates the request in place. Handles both max_completion_tokens (newer OpenAI API)
                and max_tokens (deprecated but still supported by vLLM). A request
                that set neither gets max_tokens: otherwise vLLM would fall back to
                the generation config's budget, e.g. for a restored call whose
                remaining budget is smaller.
                """
                if request.max_completion_tokens is not None:
                    request.max_completion_tokens = max_tokens
                else:
                    request.max_tokens = max_tokens

            def _clamp_max_tokens(
                self, request, request_max_tokens: int, prompt_token_ids: list[int]
            ) -> None:
                """Clamp the request's max output tokens so that input + output <= max_model_len."""
                remaining = self.model_config.max_model_len - len(prompt_token_ids)
                if remaining <= 0:
                    # preserve the literal "context length" in this message to match Gym's overflow handling
                    message = (
                        f"Prompt length ({len(prompt_token_ids)}) fills or exceeds "
                        f"this model's maximum context length ({self.model_config.max_model_len}). "
                        f"No room for output tokens."
                    )
                    LOGGER.warning("Prompt exceeds max_model_len: %s", message)
                    raise VLLMValidationError(
                        message,
                        parameter="input_tokens",
                        value=len(prompt_token_ids),
                    )
                max_tokens = min(request_max_tokens, remaining)
                self._set_max_tokens(request, max_tokens)

            # vLLM 0.25 moved chat preprocessing to
            # OnlineRenderer.preprocess_chat (tool_parser/reasoning_parser were
            # folded into a single `parser`), so this override now applies via
            # the renderer subclass.
            async def preprocess_chat(
                self,
                request,
                messages,
                default_template,
                default_template_content_format,
                default_template_kwargs,
                tool_dicts=None,
                parser=None,
                *,
                skip_mm_cache: bool = False,
            ):
                for message in messages:
                    if message.get("tool_calls"):
                        message["tool_calls"] = list(message["tool_calls"])

                messages_for_replace_prefix_tokens = deepcopy(messages)
                # #4124: processor-only cache reads retain concrete pixels even
                # when the engine's sender cache would return references.
                if (
                    frontend_self._capture_media
                    and frontend_self._capture_admission(request) is not None
                ):
                    skip_mm_cache = True

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
                        parser=parser,
                        skip_mm_cache=skip_mm_cache,
                    )
                except (ValueError, VLLMValidationError) as e:
                    if "maximum context length" in str(e):
                        import logging

                        logging.getLogger(__name__).warning(
                            "Prompt exceeds max_model_len: %s", e
                        )
                    raise

                # Token capture: build the admission once, before branching,
                # and resolve its prefix from it (staging_chain -> cached TQ
                # read, inline ids, or nothing for a text root). The
                # ``ng_capture`` dict is never mutated. Off-loop: the chain
                # fetch is a blocking TQ read, and Gym's staging protocol
                # requires the serving host to move blocking staging I/O off
                # its event loop explicitly. The adapter then attaches the
                # prefix to the request, so the inline-prefix branch below is
                # the single splice path for staged and inline prefixes.
                admission = frontend_self._capture_admission(request)
                serving_admission = admission
                verify_retained_media = False
                if admission is not None and (
                    admission.mode == "candidate"
                    or admission.request_replay is not None
                ):
                    # Inline assistant token fields cannot override RL's decision.
                    request.required_prefix_token_ids = None
                    decision = decide_capture_input(
                        admission,
                        messages=messages_for_replace_prefix_tokens,
                        # HF/string parsing maps null and empty tool content to
                        # identical template inputs. Structured content and
                        # other renderers cannot authorize that equivalence.
                        allow_empty_tool_content=(
                            type(self.renderer) is HfRenderer
                            and bool(res[0])
                            and all(
                                isinstance(item.get("content"), str) for item in res[0]
                            )
                        ),
                    )
                    admission = decision.storage
                    serving_admission = decision.serving
                    verify_retained_media = decision.verify_retained_media
                capture_prefix_token_ids: list[int] | None = None
                generation_cut = None
                resumed_generation_token_ids: list[int] = []
                restored_request_output_tokens: int | None = None
                if serving_admission is not None:
                    capture_prefix_token_ids = await asyncio.to_thread(
                        frontend_self._resolve_admission_prefix, serving_admission
                    )
                    resolved_admission, generation_cut = await asyncio.to_thread(
                        frontend_self._prefix_recovery.resolve_restore,
                        serving_admission,
                        capture_prefix_token_ids,
                    )
                    # An unusable prefix comes back without its cut, so the
                    # call regenerates; storage and serving share one admission here.
                    if serving_admission is admission:
                        admission = resolved_admission
                    serving_admission = resolved_admission
                    engine_prefix_token_ids = list(capture_prefix_token_ids)
                    if generation_cut is not None:
                        plan = plan_restore(
                            prefix_token_ids=capture_prefix_token_ids,
                            saved=generation_cut,
                            continuation=serving_admission.generation_cut,
                            min_tokens=getattr(request, "min_tokens", None),
                        )
                        restored_request_output_tokens = plan.output_limit
                        engine_prefix_token_ids = plan.engine_prefix_token_ids
                        resumed_generation_token_ids = list(
                            plan.resumed_generation_token_ids
                        )
                        request._restored_generation_token_ids = (
                            plan.resumed_generation_token_ids
                        )
                        request._original_min_tokens = getattr(
                            request, "min_tokens", None
                        )
                        if plan.max_tokens is not None:
                            actual_request_max_tokens = plan.max_tokens
                        if plan.min_tokens is not None:
                            request.min_tokens = plan.min_tokens
                    if engine_prefix_token_ids:
                        frontend_self._enter_request_prefix(
                            request, engine_prefix_token_ids
                        )

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
                    # Token capture, text mode: the full render is the exact
                    # engine prompt.
                    media = await asyncio.to_thread(
                        frontend_self._capture_request_media,
                        res[1][0],
                        admission=admission,
                        verify_retained_media=verify_retained_media,
                    )
                    frontend_self._begin_request_capture(
                        request,
                        res[1][0]["prompt_token_ids"],
                        admission=admission,
                        prefix_token_ids=capture_prefix_token_ids,
                        media=media,
                        generation_cut=generation_cut,
                        resumed_generation_token_ids=resumed_generation_token_ids,
                    )
                    return res

                model_prefix_token_ids = list(request.required_prefix_token_ids)

                # Token-in splice path — shared by staging_chain and direct prefix.
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
                    parser=parser,
                    skip_mm_cache=skip_mm_cache,
                )
                actual_corresponding_token_ids = corresponding_res[1][0][
                    "prompt_token_ids"
                ]

                engine_prompt = res[1][0]

                splice = None
                if generation_cut is not None:
                    # The durable prefix is the exact engine prompt at the cut;
                    # rendering/splicing it again could add or drop tokens.
                    # truncate_prompt_tokens already applied when the original
                    # call's chat was tokenized, and vLLM does not truncate
                    # this list again (test_vllm_progress_read_layout.py).
                    final_prompt_token_ids = model_prefix_token_ids
                else:
                    splice = splice_prefix_tokens(
                        tokenizer=self.renderer.tokenizer,
                        model_prefix_token_ids=model_prefix_token_ids,
                        template_prefix_token_ids=actual_corresponding_token_ids,
                        template_token_ids=engine_prompt["prompt_token_ids"],
                        # The template's own EOS must stay countable even when
                        # the configured stops don't include it.
                        eos_token_id=model_stop_token_ids
                        | {self.renderer.tokenizer.eos_token_id},
                    )
                    final_prompt_token_ids = splice.token_ids

                restored_prefix_terminal = None
                if generation_cut is not None:
                    try:
                        restored_prefix_terminal = _classify_restored_prefix_terminal(
                            prompt_token_ids=final_prompt_token_ids,
                            prefix_token_count=(
                                admission.generation_cut.prefix_token_count
                            ),
                            requested_output_tokens=restored_request_output_tokens,
                            model_max_tokens=self.model_config.max_model_len,
                            terminal_finish_reason=(
                                admission.generation_cut.terminal_finish_reason
                            ),
                            terminal_stop_reason=(
                                admission.generation_cut.terminal_stop_reason
                            ),
                        )
                    except ValueError as error:
                        raise VLLMValidationError(
                            str(error),
                            parameter="generation_prefix",
                            value=admission.generation_cut.prefix_token_count,
                        ) from error

                # Reject overflows before media validation can mask the length
                # error. A restored terminal prefix generates nothing more.
                if (
                    restored_prefix_terminal is None
                    and actual_request_max_tokens is not None
                ):
                    self._clamp_max_tokens(
                        request,
                        actual_request_max_tokens,
                        final_prompt_token_ids,
                    )

                # Generation-prefix cuts are text-only (setup refuses them with
                # media capture), so a restored prompt has no media to capture.
                media = None
                if generation_cut is None:
                    media = await asyncio.to_thread(
                        frontend_self._capture_request_media,
                        engine_prompt,
                        admission=serving_admission,
                        splice=splice,
                        verify_retained_media=verify_retained_media,
                    )
                # Must precede the reassignment below: the remap reads the
                # current value of prompt_token_ids as the template-coordinate
                # sequence the placeholders were computed against. Reordering
                # these two statements silently degrades to a no-op.
                # Successful media capture already remapped these ranges.
                if media is None and (
                    mm_placeholders := engine_prompt.get("mm_placeholders")
                ):
                    engine_prompt["mm_placeholders"] = remap_multimodal_placeholders(
                        template_token_ids=engine_prompt["prompt_token_ids"],
                        final_token_ids=final_prompt_token_ids,
                        mm_placeholders=mm_placeholders,
                        splice=splice,
                    )

                engine_prompt["prompt_token_ids"] = final_prompt_token_ids

                # Token capture, token-in mode: the spliced prompt is the
                # exact engine prompt; begin_call re-checks the prefix it
                # was spliced from against the admission.
                frontend_self._begin_request_capture(
                    request,
                    final_prompt_token_ids,
                    admission=admission,
                    prefix_token_ids=(
                        capture_prefix_token_ids
                        if admission is not None and admission.mode == "token_in"
                        else None
                    ),
                    media=media,
                    generation_cut=generation_cut,
                    resumed_generation_token_ids=resumed_generation_token_ids,
                )

                if restored_prefix_terminal is not None:
                    # The engine client answers it without decoding; one
                    # token is the smallest budget vLLM accepts, so a missed
                    # marker costs one token, not the whole call.
                    request.min_tokens = 0
                    self._set_max_tokens(request, 1)
                    mark_restored_terminal(restored_prefix_terminal)
                    if len(final_prompt_token_ids) == self.model_config.max_model_len:
                        engine_prompt["prompt_token_ids"] = final_prompt_token_ids[:-1]
                    request._restored_prefix_terminal = restored_prefix_terminal
                    LOGGER.info(
                        "generation prefix already terminal: "
                        "rollout_id=%s model_call_id=%s prefix_tokens=%d reason=%s",
                        admission.capture_key,
                        admission.model_call_id,
                        restored_prefix_terminal.prefix_token_count,
                        restored_prefix_terminal.reason,
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
            # Ledger-authoritative token capture: the call identity the ledger
            # attaches (capture_key, call_id, parent_call_id, prev_len, mode).
            ng_capture: Optional[dict[str, Any]] = None
            _restored_prefix_terminal: _RestoredPrefixTerminal | None = PrivateAttr(
                default=None
            )
            _restored_generation_token_ids: tuple[int, ...] = PrivateAttr(default=())
            # The client's min_tokens before it is reduced by a restored prefix;
            # replaying the prefix's stop handling needs the original window.
            _original_min_tokens: int | None = PrivateAttr(default=None)

            def to_sampling_params(self, *args, **kwargs):
                sampling_params = super().to_sampling_params(*args, **kwargs)
                if self.ng_capture is not None:
                    frontend_self._prefix_recovery.on_sampling_params(
                        self, sampling_params
                    )
                return sampling_params

        # vLLM 0.25 routes both /v1/chat/completions and /tokenize through
        # OnlineRenderer.preprocess_chat, so the prefix-token override
        # belongs on the renderer subclass.
        frontend_self = self

        register_generation_cut_route(app, lambda: frontend_self._prefix_recovery)

        class NeMoRLOpenAIServingChatMixin:
            async def chat_completion_full_generator(
                self,
                request,
                result_generator,
                *args,
                **kwargs,
            ):
                return_as_token_id = (
                    request.return_tokens_as_token_ids
                    if request.return_tokens_as_token_ids is not None
                    else self.return_tokens_as_token_ids
                )
                if (
                    request.logprobs
                    and return_as_token_id
                    and request.top_logprobs is None
                ):
                    raise VLLMValidationError(
                        "`top_logprobs` must be set when requesting token "
                        "information from the NeMo-RL chat endpoint.",
                        parameter="top_logprobs",
                    )

                final_res = None
                capture_routes = request in frontend_self._capture_registry
                cancelled = False

                def _without_routes(res):
                    # vLLM encodes choice-level routes in its response. Keep
                    # engine arrays for our native capture, and hide them only
                    # in the superclass's shallow view.
                    res = copy.copy(res)
                    res.outputs = [copy.copy(output) for output in res.outputs]
                    for output in res.outputs:
                        output.routed_experts = None
                    return res

                async def capture_result_generator():
                    nonlocal final_res, cancelled
                    try:
                        async for res in result_generator:
                            # Recorded before asking for the next output, which
                            # releases vLLM's finished state for this request.
                            frontend_self._prefix_recovery.observe(request, res)
                            final_res = res
                        if final_res is None:
                            return
                        frontend_self._prefix_recovery.restore_reply_text(
                            request,
                            final_res,
                            tokenizer=self.renderer.tokenizer,
                        )
                        yield (
                            _without_routes(final_res) if capture_routes else final_res
                        )
                    except asyncio.CancelledError:
                        cancelled = True
                        raise

                # Bound by name: vLLM passes request_metadata by position and
                # parser by keyword, and either may move.
                bound = full_generator_signature.bind_partial(
                    self, request, capture_result_generator(), *args, **kwargs
                )
                if bound.arguments.get("parser") is not None:
                    bound.arguments["parser"] = wrap_parser_for_restore(
                        bound.arguments["parser"],
                        request._restored_generation_token_ids,
                    )

                # Capture stages the raw arrays below and strips OpenAI logprobs
                # after the terminal PUT. Skip only their object construction;
                # changing request.logprobs would also change engine sampling.
                bypass_logprobs = bool(
                    not request.stream
                    and request.logprobs
                    and return_as_token_id
                    and request in frontend_self._capture_registry
                )
                with self._bypass_chat_logprobs(enabled=bypass_logprobs):
                    response = await super().chat_completion_full_generator(
                        *bound.args[1:], **bound.kwargs
                    )
                if cancelled:
                    # vLLM's chat_completion_full_generator catches the
                    # CancelledError that with_cancellation throws into the
                    # handler task on a client disconnect and returns
                    # ErrorResponse("Client disconnected") instead of
                    # re-raising. Nobody reads that response (the wrapper has
                    # already returned), and letting it through would make the
                    # route label the capture "error_response". Re-raise so the
                    # task finishes cancelled and the route sees the disconnect.
                    raise asyncio.CancelledError()
                if (
                    not isinstance(response, ChatCompletionResponse)
                    or final_res is None
                ):
                    return response

                restored_prefix_terminal = request._restored_prefix_terminal
                if not isinstance(restored_prefix_terminal, _RestoredPrefixTerminal):
                    restored_prefix_terminal = None
                if restore_reply_usage(
                    response,
                    restored_token_count=len(request._restored_generation_token_ids),
                    terminal=restored_prefix_terminal,
                ):
                    request_metadata = bound.arguments.get("request_metadata")
                    if request_metadata is not None:
                        request_metadata.final_usage_info = response.usage

                if request.logprobs and return_as_token_id:
                    response = attach_token_information_to_chat_response_choices(
                        response,
                        final_res,
                    )
                    if request.ng_capture and frontend_self._capture_top_logprobs:
                        # Extraction scales with completion length and K.
                        response = await asyncio.to_thread(
                            attach_sampler_topk_logprobs_to_chat_response_choices,
                            response,
                            final_res,
                            top_k=frontend_self._capture_top_logprobs,
                        )

                if frontend_self._return_routed_experts_enabled():
                    response = attach_routed_experts_to_chat_response_choices(
                        response,
                        final_res,
                        device=torch.device("cpu"),
                        logger=LOGGER,
                        routed_experts_dtype=frontend_self.routed_experts_dtype,
                        encode_for_wire=not capture_routes,
                    )

                return response

        class NeMoRLOpenAIServingChat(
            NeMoRLOpenAIServingChatMixin,
            CapturedChatLogprobsMixin,
            OpenAIServingChat,
        ):
            pass

        class NeMoRLOnlineRenderer(NeMoRLOpenAIServingMixin, OnlineRenderer):
            pass

        serving_chat_default_kwargs: dict[str, Any] = dict(
            response_role="assistant",
            request_logger=None,
            chat_template=None,
            chat_template_content_format="auto",
            enable_auto_tools=True,
        )
        serving_chat_kwargs: dict[str, Any] = serving_chat_default_kwargs | self.cfg[
            "vllm_cfg"
        ].get("http_server_serving_chat_kwargs", dict())
        # The embedded server is constructed directly instead of through
        # vLLM's CLI, where chat-template file paths are normally loaded.
        # OnlineRenderer expects literal Jinja content; passing a path makes
        # Transformers render the path itself and drops multimodal
        # placeholders such as <image>.
        configured_chat_template = serving_chat_kwargs.get("chat_template")
        if configured_chat_template is not None:
            serving_chat_kwargs["chat_template"] = load_chat_template(
                configured_chat_template
            )
        # Recipes may name the parameter either way: ``default_chat_template_kwargs``
        # is vLLM's own spelling, ``chat_template_kwargs`` is accepted for recipes
        # written against the older name. Normalize onto the native key rather
        # than popping it: OnlineRenderer, OpenAIServingChat and ServingTokenization
        # each keep their *own* copy and read it independently -- the chat serving
        # builds its reasoning parser from it, and the tokenize path passes its own
        # into preprocess_chat -- so the renderer's copy does not reach either.
        # vLLM's api_server hands the same value to all three for that reason.
        #
        # Popped separately, not `A or B`: short-circuiting on a truthy A would
        # leave B in the bag and OpenAIServingChat(**kwargs) would reject it.
        _legacy_chat_template_kwargs = serving_chat_kwargs.pop(
            "chat_template_kwargs", None
        )
        if serving_chat_kwargs.get("default_chat_template_kwargs") is None:
            serving_chat_kwargs["default_chat_template_kwargs"] = (
                _legacy_chat_template_kwargs
            )
        default_chat_template_kwargs: dict[str, Any] = (
            serving_chat_kwargs["default_chat_template_kwargs"] or {}
        )
        online_renderer = NeMoRLOnlineRenderer(
            model_config=engine_client.model_config,
            renderer=engine_client.renderer,
            request_logger=serving_chat_kwargs["request_logger"],
            chat_template=serving_chat_kwargs["chat_template"],
            chat_template_content_format=serving_chat_kwargs[
                "chat_template_content_format"
            ],
            enable_auto_tools=serving_chat_kwargs["enable_auto_tools"],
            # Keep the renderer's parser consistent with any parser overrides
            # passed to OpenAIServingChat via http_server_serving_chat_kwargs.
            tool_parser=serving_chat_kwargs.get("tool_parser"),
            reasoning_parser=serving_chat_kwargs.get("reasoning_parser"),
            # vLLM merges these into every render, with request-supplied keys
            # winning (preprocess_chat's default_template_kwargs). The renderer
            # is shared by /v1/chat/completions and /tokenize, so setting it
            # here keeps the two endpoints rendering identical prompts.
            default_chat_template_kwargs=default_chat_template_kwargs,
        )
        serving_chat_kwargs.update(
            engine_client=engine_client,
            models=openai_serving_models,
            online_renderer=online_renderer,
            return_tokens_as_token_ids=True,
        )
        openai_serving_chat = NeMoRLOpenAIServingChat(**serving_chat_kwargs)

        generation_config = self.cfg

        async def abort_request_capture(request: Any, reason: str) -> None:
            """Fail a request's capture without blocking the event loop.

            An uncontended abort finishes here, so it never queues behind
            executor work. Only one that must wait out a generation cut or a
            terminal write holding the call's lifecycle lock moves off the loop.
            """
            if frontend_self._abort_request_capture(
                request, reason=reason, blocking=False
            ):
                return
            # Shielded: these paths often run after the handler was cancelled,
            # and a second cancel must not drop the abort before it starts,
            # leaving the call neither completed nor failed.
            await asyncio.shield(
                asyncio.to_thread(
                    frontend_self._abort_request_capture, request, reason=reason
                )
            )

        # The create_chat_completion and tokenize methods are taken from vllm/entrypoints/openai/api_server.py
        @app.post("/v1/chat/completions")
        # Like vLLM's own routes: cancel the handler (and with it the engine
        # request) when the client disconnects, instead of generating for nobody.
        @with_cancellation
        async def create_chat_completion(
            request: NeMoRLChatCompletionRequest, raw_request: Request
        ):
            # Accept train and validation profiles; reject off-policy requests.
            # vLLM fills an unset top_k/min_p/repetition_penalty from
            # generation_config.json when vllm_kwargs sets generation_config="auto",
            # so pin top_k to the matched profile's (as
            # vllm_worker.py::BaseVllmGenerationWorker::_build_sampling_params does)
            # and disable min_p and the repetition penalty. vLLM applies these and
            # the presence/frequency penalties and logit bias before sampling; the
            # trainer replays none of them, so a request may not set them.
            top_k = check_sampling_param(
                temperature=request.temperature,
                top_p=request.top_p,
                top_k=request.top_k,
                generation_config=generation_config,
            )
            assert request.min_p in (None, 0.0), (
                f"request min_p={request.min_p} is not supported; leave it unset"
            )
            assert request.repetition_penalty in (None, 1.0), (
                f"request repetition_penalty={request.repetition_penalty} is not "
                "supported; leave it unset"
            )
            assert request.presence_penalty in (None, 0.0), (
                f"request presence_penalty={request.presence_penalty} is not "
                "supported; leave it unset"
            )
            assert request.frequency_penalty in (None, 0.0), (
                f"request frequency_penalty={request.frequency_penalty} is not "
                "supported; leave it unset"
            )
            assert not request.logit_bias, "request logit_bias is not supported"
            request.top_k = top_k
            request.min_p = 0.0
            request.repetition_penalty = 1.0

            try:
                generator = await openai_serving_chat.create_chat_completion(
                    request, raw_request
                )
            except VLLMValidationError as e:
                # vLLM raises VLLMValidationError for prompts exceeding
                # max_model_len during tokenization, instead of returning an
                # ErrorResponse. Convert to HTTP 400 so the Gym proxy can
                # detect context-length overflow and handle it gracefully.
                await abort_request_capture(request, "context_length")
                return JSONResponse(
                    content={
                        "error": {
                            "message": str(e),
                            "type": "invalid_request_error",
                            "param": e.parameter,
                            "code": 400,
                        }
                    },
                    status_code=400,
                    headers=frontend_self._context_refusal_headers(request),
                )
            except MediaCaptureRejected as e:
                # Raised inside preprocess_chat before begin_call, so no capture
                # state exists yet and the abort below is a no-op kept for
                # symmetry. Return a 400 carrying a stable code so Gym and the
                # worker log can distinguish a retained-media re-tile from a
                # real engine error (which stays a 500 below).
                await abort_request_capture(request, e.code)
                LOGGER.warning(
                    "Rejected captured call before inference (%s): %s", e.code, e
                )
                return JSONResponse(
                    content={
                        "error": {
                            "message": str(e),
                            "type": "invalid_request_error",
                            "param": "messages",
                            "code": e.code,
                        }
                    },
                    status_code=400,
                )
            except ValueError as e:
                # vLLM's get_max_tokens raises a plain ValueError ("Input length
                # (N) exceeds model's maximum context length (M).") after
                # preprocess_chat, e.g. when a token-in splice grows the prompt
                # past max_model_len on a request with no max_tokens to clamp.
                # Convert only that overflow to the same 400 as above; any other
                # ValueError is a server bug and stays a 500. Must follow
                # MediaCaptureRejected, which is also a ValueError.
                if "maximum context length" not in str(e):
                    # Re-raising here bypasses the BaseException handler below.
                    await abort_request_capture(request, "engine_error")
                    raise
                LOGGER.warning("Prompt exceeds max_model_len: %s", e)
                await abort_request_capture(request, "context_length")
                return JSONResponse(
                    content={
                        "error": {
                            "message": str(e),
                            "type": "invalid_request_error",
                            "param": "input_tokens",
                            "code": 400,
                        }
                    },
                    status_code=400,
                    # Only vLLM's known pre-generation get_max_tokens refusal
                    # can settle capture; an arbitrary ValueError cannot.
                    headers=(
                        frontend_self._context_refusal_headers(request)
                        if str(e).startswith("Input length (")
                        and "exceeds model's maximum context length" in str(e)
                        else None
                    ),
                )
            except asyncio.CancelledError:
                # Client disconnected (see with_cancellation above). A cancel
                # before generation starts propagates here directly; one that
                # lands mid-generation is swallowed by vLLM and re-raised by
                # NeMoRLOpenAIServingChatMixin.chat_completion_full_generator.
                await abort_request_capture(request, "client_disconnect")
                raise
            except BaseException:
                await abort_request_capture(request, "engine_error")
                raise

            if isinstance(generator, ErrorResponse):
                await abort_request_capture(request, "error_response")
                return JSONResponse(
                    content=generator.model_dump(), status_code=generator.error.code
                )

            elif isinstance(generator, ChatCompletionResponse):
                content = model_dump_chat_response_with_dynamic_message_fields(
                    generator
                )
                # Token capture: stage the delta and ride the coords on the
                # response; strips logprobs/ids (no-op when capture is off).
                # Off-loop: the sink write inside complete_call is a blocking
                # TQ round trip (see the staging protocol's serving-host rule).
                try:
                    content = await asyncio.to_thread(
                        frontend_self._finish_request_capture, request, content
                    )
                except asyncio.CancelledError:
                    # Queued executor work may never start after cancellation.
                    # If staging already claimed the capture, abort is a no-op.
                    await abort_request_capture(request, "client_disconnect")
                    raise
                return JSONResponse(content=content)

            await abort_request_capture(request, "streaming_response")
            return StreamingResponse(content=generator, media_type="text/event-stream")

        ########################################
        # /tokenize endpoint
        ########################################

        # This MRO is necessary i.e. NeMoRLOpenAIChatRequestMixin > TokenizeRequest
        class NeMoRLTokenizeChatRequest(
            NeMoRLOpenAIChatRequestMixin, TokenizeChatRequest
        ):
            required_prefix_token_ids: Optional[List[int]] = None
            ng_capture: Optional[dict[str, Any]] = None
            ng_measure_only: Literal[True] = True

        NeMoRLTokenizeRequest = Union[
            TokenizeCompletionRequest, NeMoRLTokenizeChatRequest
        ]

        # Tokenize path delegates to OnlineRenderer.preprocess_chat,
        # where the prefix-token override lives.
        class NeMoRLServingTokenization(ServingTokenization):
            pass

        serving_tokenization_kwargs: dict[str, Any] = dict(
            request_logger=serving_chat_kwargs["request_logger"],
            chat_template=serving_chat_kwargs["chat_template"],
            chat_template_content_format=serving_chat_kwargs[
                "chat_template_content_format"
            ],
            models=serving_chat_kwargs["models"],
            online_renderer=online_renderer,
            # ServingTokenization reads its own copy in preprocess_chat rather
            # than the renderer's, so /tokenize would otherwise render with {}
            # and diverge from /v1/chat/completions under multi-turn.
            default_chat_template_kwargs=default_chat_template_kwargs,
        )
        openai_serving_tokenization = NeMoRLServingTokenization(
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
                if (
                    isinstance(request, NeMoRLTokenizeChatRequest)
                    and request.ng_capture is not None
                ):
                    return JSONResponse(
                        content={
                            "prompt_token_count": generator.count,
                            "ng_context_measured": True,
                        }
                    )
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

        _getLogger("vllm.entrypoints.openai.chat_completion.serving").addFilter(
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

        @app.get("/health")
        async def health() -> dict[str, str]:
            return {"status": "ok"}

        app = self._setup_vllm_openai_api_server(app)
        if self._register_routes is not None:
            self._register_routes(app)
        # Present only when the driver generated a key (vllm_cfg.http_server_api_key_required).
        http_server_api_key = self.cfg["vllm_cfg"].get("http_server_api_key")
        if http_server_api_key:
            install_http_server_api_key_check(app, http_server_api_key)

        ########################################
        # Server spinup
        ########################################

        if self._reserved_socket is not None:
            # Use the socket reserved during __init__ (deferred model load path).
            # Pass it directly to uvicorn via sockets= — zero gap, the socket is
            # never closed and re-opened, so no other process can steal the port.
            node_ip = self._reserved_node_ip
            reserved_sock = self._reserved_socket
            free_port = reserved_sock.getsockname()[1]
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
            sockets = [reserved_sock]

            def _run_with_socket() -> None:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(server.serve(sockets=sockets))

            thread = threading.Thread(target=_run_with_socket, daemon=True)
        else:
            thread = threading.Thread(target=server.run, daemon=True)
        thread.start()

        return thread, base_url, server

    async def flush_obsolete_generation_cut_rows_async(self, epoch: int) -> bool:
        """Delete the chunk rows of calls that ended before snapshot ``epoch``.

        Returns once the delete is done: TQ must not clear rows while the
        snapshot is saved.
        """
        await self._prefix_recovery.flush(epoch)
        return True

    def get_generation_prefix_counts(self) -> dict[str, int]:
        """Generation-prefix restore and progress-read counts; see PrefixRecovery."""
        return self._prefix_recovery.counts()

    def start(self) -> None:
        """Start serving requests on the frontend's listener."""
        self.server_thread, self.base_url, self.http_server = self._setup_vllm_server()

    async def stop(self, *, join_timeout: float | None = None) -> None:
        """Stop prefix recovery and join the HTTP server thread.

        Args:
            join_timeout: Maximum seconds to wait for the server; None waits until it exits.
        """
        # Waiting for a wedged TQ write here would make actor shutdown hang.
        self._prefix_recovery.shutdown()
        if self.http_server is not None and self.server_thread is not None:
            self.http_server.should_exit = True
            await asyncio.to_thread(self.server_thread.join, join_timeout)
            self.server_thread = None
