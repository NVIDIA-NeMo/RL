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

from __future__ import annotations

import asyncio
import contextvars
import functools
import hashlib
import logging
import threading
import time
from collections.abc import Callable, Generator
from dataclasses import dataclass, field, replace
from typing import Any, Literal, cast

import numpy as np
import torch

from nemo_rl.experience.route_assembly import (
    ROUTE_MISSING_SENTINEL,
    verify_route_fragment_integrity,
)
from nemo_rl.telemetry.rollout_checkpoint import checkpoint_phase, checkpoint_phase_sync
from nemo_rl.utils.prefix_cut_diagnostics import (
    prefix_cut_count,
    prefix_cut_timing,
    prefix_cut_wait,
)
from nemo_rl.utils.routed_experts_codec import (
    ROUTED_EXPERTS_STAGING_ATTACHMENT,
    RoutedExpertsStagingPayload,
    encode_routed_experts,
)

LOGGER = logging.getLogger(__name__)

# Completion evidence only bridges the short race between a terminal response
# and Gym freezing its admitted-call inventory. Time-based retention avoids
# evicting fresh evidence merely because a large deployment completed more
# than 100k calls, while still bounding idle-job memory growth.
_COMPLETED_CAPTURE_RETENTION_S = 60.0 * 60.0

# First-row extras identify the route-digest layout. Version 2 commits to the
# ordered chunk digests instead of re-encoding all cumulative route bytes.
_CUT_DIGEST_LAYOUT = "nrl_generation_cut_digest_layout"
_CUT_CHUNKS_DIGEST = "nrl_generation_cut_chunks_sha256"

# Bound owner-loop work and the number of live-route requests per lookup.
# Transport batching retains its independent row/token limits.
_ROUTE_SNAPSHOT_GROUP_MAX_CALLS = 64


@dataclass(frozen=True)
class _RouteSnapshotRequest:
    """A transaction paused after freezing, before reading engine routes."""

    engine_request_id: str
    token_positions: int


@dataclass(frozen=True)
class _EngineRouteSnapshot:
    """Local route references or a per-request error; never sent over the wire."""

    routes: Any | None = None
    error: Exception | None = None


def _cut_chunk_digest(digests: list[str]) -> str:
    """Commit to an ordered sequence of fixed-width, route-covered digests."""
    digest = hashlib.sha256(b"nrl-generation-cut-chunks-v2\0")
    for value in digests:
        raw = bytes.fromhex(value)
        if len(raw) != 32:
            raise ValueError("generation-cut chunk digest must be SHA-256")
        digest.update(raw)
    return digest.hexdigest()


def _final_route_placeholder(
    layers: int, topk: int, dtype: torch.dtype
) -> torch.Tensor:
    """The route row staged for a prefix's last token, which no step has routed.

    A cut builds it when staging and a restore rebuilds it when joining
    chunks; both must use this helper, or every resumed call fails the
    digest check. It holds ``ROUTE_MISSING_SENTINEL``, as the final row of an
    uncut call's routes does.
    """
    return torch.full((1, layers, topk), ROUTE_MISSING_SENTINEL, dtype=dtype)


def _prefix_routes_sha256(
    delta_routes: torch.Tensor | None, *, generated_prefix_tokens: int
) -> str:
    """Hash the routes a resumed call keeps for its restored generated prefix.

    ``delta_routes`` holds one row per prefix delta token. The kept rows are
    the generated prefix tokens' except the last, whose route the resumed
    call's own prefill computes. Restore and completion both log this, so the
    staged row can be checked against the routes restored from TQ.

    Returns:
        A SHA-256 hex digest over the kept rows as int32, or ``"none"`` when
        the call carries no routes.
    """
    if delta_routes is None:
        return "none"
    end = int(delta_routes.shape[0])
    kept = delta_routes[end - generated_prefix_tokens : end - 1]
    return hashlib.sha256(
        kept.to(torch.int32).contiguous().numpy().tobytes()
    ).hexdigest()


@dataclass
class _RequestCaptureState:
    """Append-only generated tokens for one in-flight captured request.

    ``observe`` appends each step's new tokens exactly once. A checkpoint
    freezes the current length under ``lock`` and performs its blocking TQ
    write after releasing it, so decoding keeps appending meanwhile.

    Each successful cut stages only the tokens past ``staged_token_count``.
    The ordered TQ keys remain in Gym lineage; the full list is kept locally to
    compute the cumulative digest advertised by the latest cut.
    """

    call: Any
    prompt_token_ids: list[int]
    # Router replay: a cut reads the engine's routes once, after it freezes
    # the token count; see ``frozen_cut_routed_experts``.
    capture_routed_experts: bool = False
    routed_experts_dtype: torch.dtype = torch.int16
    restored_routed_experts: torch.Tensor | None = None
    # The engine's ID for this request, from its first output.
    engine_request_id: str | None = None
    # Full-sequence routes from the request's final output.
    terminal_routed_experts: torch.Tensor | None = None
    # Backend-owned opaque state that must survive until the terminal row is
    # staged. vLLM uses this for its processed-media snapshot; the first
    # generation-prefix row carries its tensor digests across recovery.
    media: Any | None = None
    resumed_generation_token_ids: list[int] = field(default_factory=list)
    effective_output_limit: int | None = None
    terminal_finish_reason: Literal["stop", "length"] | None = None
    terminal_stop_reason: str | int | None = None
    observation_error: str | None = None
    # Set when the request's sampling settings would not carry across a cut,
    # so the call regenerates after a restore instead of resuming its prefix.
    uncuttable_reason: str | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)
    # Serializes lifecycle-changing TQ operations for this call. Token
    # observation deliberately uses ``lock`` instead so decoding can continue
    # while a detached generation chunk is staged.
    lifecycle_lock: threading.Lock = field(default_factory=threading.Lock)
    terminal_started: bool = False
    generated_token_ids: list[int] = field(default_factory=list)
    generated_logprobs: list[float] = field(default_factory=list)
    # Tokens already written to TQ by sealed cuts.
    staged_token_count: int = 0
    generation_cut_staging_keys: list[str] = field(default_factory=list)
    # None retains the legacy cumulative-route digest when extending an old
    # checkpoint. Only sealed rows enter this list; observed route cache state
    # below is independent of durability and survives a failed cut.
    generation_cut_chunk_digests: list[str] | None = field(default_factory=list)
    engine_route_piece_count: int = 0
    engine_route_rows: int = 0
    engine_route_buffer: np.ndarray | None = None
    engine_route_last_piece: Any | None = None
    # The cut in progress: its checkpoint and the token count it froze.
    frozen_cut: tuple[str, int] | None = None
    # Names each frozen chunk, so a rolled-back cut never reuses a TQ key.
    next_chunk_sequence: int = 0

    @property
    def observed_generation_token_count(self) -> int:
        return len(self.generated_token_ids)

    def observe(
        self,
        generated_token_ids: list[int],
        generated_logprobs: list[float],
    ) -> None:
        """Append the tokens one vLLM step added."""
        if len(generated_token_ids) != len(generated_logprobs):
            self.observation_error = (
                "generated token IDs and log probabilities must have equal lengths"
            )
            return
        self.generated_token_ids.extend(generated_token_ids)
        self.generated_logprobs.extend(generated_logprobs)

    def _cumulative_route_count(self, generated_token_count: int) -> int:
        """Rows in the delta-aligned routes of the first generated tokens.

        One row per delta token: the prompt tokens past ``prev_len`` and the
        generated ones, the last being the final-position placeholder.
        """
        if self.restored_routed_experts is not None:
            return int(self.restored_routed_experts.shape[0]) + generated_token_count
        return (
            len(self.prompt_token_ids)
            - self.call.admission.prev_len
            + generated_token_count
        )

    def freeze_for_checkpoint(
        self, checkpoint_id: str
    ) -> tuple[str, int, list[int], list[float], list[int], list[float]]:
        """Freeze the current length; return the unstaged chunk and the prefix."""
        if self.frozen_cut is not None:
            raise RuntimeError(
                "cannot start a generation cut while another cut is in progress"
            )
        frozen = len(self.generated_token_ids)
        sequence = self.next_chunk_sequence
        self.next_chunk_sequence += 1
        self.frozen_cut = (checkpoint_id, frozen)
        staged = self.staged_token_count
        return (
            f"active/{checkpoint_id}/{sequence}",
            sequence,
            self.generated_token_ids[staged:frozen],
            self.generated_logprobs[staged:frozen],
            self.generated_token_ids[:frozen],
            self.generated_logprobs[:frozen],
        )

    def absorb_engine_route_pieces(self, pieces: list[Any]) -> torch.Tensor:
        """Join only newly published pieces into an amortized-growth CPU cache.

        Called by the serial cut executor for this physical request. Published
        pieces must be immutable and append-only. Cache progress is observed
        engine state, not a successful cut, so rollback does not rewind it.
        """
        if not pieces:
            raise ValueError("vLLM has not routed any token of the request yet")
        covered = self.engine_route_piece_count
        if len(pieces) < covered:
            raise ValueError("vLLM route-piece list became shorter")
        if covered and pieces[covered - 1] is not self.engine_route_last_piece:
            raise ValueError("vLLM replaced a published route piece")
        if len(pieces) > covered:
            with prefix_cut_timing("route_history_concat"):
                arrays = [np.asarray(piece) for piece in pieces[covered:]]
                shape = tuple(arrays[0].shape[1:])
                if (
                    any(
                        array.ndim != 3
                        or tuple(array.shape[1:]) != shape
                        or array.dtype.kind not in "iu"
                        for array in arrays
                    )
                    or not shape
                    or min(shape) < 1
                ):
                    raise ValueError(
                        "vLLM route chunks must share shape [*, layers, topk]"
                    )
                buffer = self.engine_route_buffer
                if buffer is not None and tuple(buffer.shape[1:]) != shape:
                    raise ValueError("vLLM route model dimensions changed between cuts")
                added = np.concatenate(arrays, axis=0)
                rows = self.engine_route_rows + len(added)
                if buffer is None or rows > len(buffer):
                    capacity = max(rows, 2 * len(buffer) if buffer is not None else 1)
                    dtype = (
                        torch.empty((), dtype=self.routed_experts_dtype).numpy().dtype
                    )
                    grown = np.empty((capacity, *shape), dtype=dtype)
                    if buffer is not None:
                        grown[: self.engine_route_rows] = buffer[
                            : self.engine_route_rows
                        ]
                    buffer = grown
                buffer[self.engine_route_rows : rows] = added
                # Publish cache metadata only after validation and copying succeed.
                self.engine_route_buffer = buffer
                self.engine_route_rows = rows
                self.engine_route_piece_count = len(pieces)
                self.engine_route_last_piece = pieces[-1]
            prefix_cut_count("route_history_chunks", len(arrays))
            prefix_cut_count("route_history_bytes", added.nbytes)
        assert self.engine_route_buffer is not None
        return torch.from_numpy(self.engine_route_buffer[: self.engine_route_rows])

    def frozen_cut_routed_experts(
        self, engine_routed_experts: Any | None
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Return the routes of the cut in progress: its chunk's, then the prefix's.

        ``engine_routed_experts`` are the request's full-sequence routes, read
        from the engine after the freeze; ``None`` once the request has left
        the engine. Rows decoding added after the freeze are dropped. vLLM only
        appends routes, so earlier rows never change and every cut stages rows
        that line up with what earlier cuts staged. The final output's routes
        are used when the request has finished.

        The chunk has one row per chunk token; the prefix's cumulative routes
        end with a final placeholder row.

        Raises:
            ValueError: the routes are malformed, or do not cover every
                frozen token.
        """
        if self.frozen_cut is None:
            raise RuntimeError("no generation cut is in progress")
        if not self.capture_routed_experts:
            return None, None
        frozen = self.frozen_cut[1]
        staged = self.staged_token_count
        restored = self.restored_routed_experts
        if frozen == 0:
            # Nothing decoded since the restore: the prefix is the restored one.
            return None, None if restored is None else restored.clone()
        raw = self.terminal_routed_experts
        if raw is None:
            raw = engine_routed_experts
        if raw is None:
            raise ValueError(
                "the request left vLLM before its routes were read for the cut"
            )
        raw = torch.as_tensor(raw, dtype=self.routed_experts_dtype).cpu()
        if raw.dim() != 3 or raw.shape[1] < 1 or raw.shape[2] < 1:
            raise ValueError("routed experts must have shape [tokens, layers, topk]")
        layers, topk = int(raw.shape[1]), int(raw.shape[2])
        # Rows parent lineage or the restored cut already own.
        skip = self.call.admission.prev_len
        if restored is not None:
            if tuple(restored.shape[1:]) != (layers, topk):
                raise ValueError("restored routed experts changed model dimensions")
            skip += int(restored.shape[0]) - 1
        # The last frozen token is not routed until a later step prefills it.
        needed = self._cumulative_route_count(frozen) - 1
        if restored is not None:
            needed -= int(restored.shape[0]) - 1
        routes = raw[skip : skip + needed]
        if routes.shape[0] < needed:
            # vLLM can omit route rows (prefix caching with chunked prefill).
            # Padding would shift later rows onto the wrong tokens, and a later
            # cut would see different rows than this one staged.
            raise ValueError(
                "vLLM returned fewer routed experts than the frozen tokens need: "
                f"routes={routes.shape[0]} needed={needed}"
            )
        final = _final_route_placeholder(layers, topk, self.routed_experts_dtype)
        parts = [routes, final] if restored is None else [restored[:-1], routes, final]
        cumulative = torch.cat(parts, dim=0).contiguous()
        if frozen == staged:
            return None, cumulative
        if staged == 0 and restored is None:
            return cumulative, cumulative
        # The previous placeholder row becomes this chunk's first route.
        start = self._cumulative_route_count(staged) - 1
        return cumulative[start:-1], cumulative

    def seal_frozen_cut(self, checkpoint_id: str) -> None:
        """Count a successfully staged cut's tokens as written to TQ."""
        self.staged_token_count = self._require_frozen_cut(checkpoint_id)
        self.frozen_cut = None

    def rollback_frozen_cut(self, checkpoint_id: str) -> None:
        """Abandon a failed cut; its tokens stay unstaged for the next one."""
        self._require_frozen_cut(checkpoint_id)
        self.frozen_cut = None

    def _require_frozen_cut(self, checkpoint_id: str) -> int:
        if self.frozen_cut is None or self.frozen_cut[0] != checkpoint_id:
            raise RuntimeError(
                f"generation cut {checkpoint_id!r} does not own the frozen cut"
            )
        return self.frozen_cut[1]


@dataclass(frozen=True)
class _CompletedCaptureState:
    """Time-bounded terminal evidence retained across the response/cut race."""

    coords: Any
    generation_token_count: int
    effective_output_limit: int | None
    terminal_finish_reason: Literal["stop", "length"] | None
    terminal_stop_reason: str | int | None
    completed_at_monotonic: float


@dataclass(frozen=True)
class _ResolvedGenerationCut:
    """A verified restored prefix, with its call's media and router-replay routes."""

    snapshot: Any
    media: Any | None = None
    routed_experts: torch.Tensor | None = None
    chunk_digests: tuple[str, ...] | None = None


def _remaining_generation_limits_after_prefix(
    *,
    max_tokens: int | None,
    min_tokens: int | None,
    prefix_token_count: int,
) -> tuple[int | None, int | None]:
    """Return output limits for the suffix after restoring generated tokens."""
    if prefix_token_count < 0:
        raise ValueError("prefix_token_count must be non-negative")
    return (
        None if max_tokens is None else max_tokens - prefix_token_count,
        None if min_tokens is None else max(0, min_tokens - prefix_token_count),
    )


# One character spans at most four UTF-8 bytes, hence at most four byte-level
# tokens; decoding a slightly longer tail is enough to see whether it is whole.
_PREFIX_BOUNDARY_TAIL_TOKENS = 8


def _ends_inside_character(tokenizer: Any, generated_token_ids: list[int]) -> bool:
    """Whether the generated text stops partway through a multi-byte character.

    vLLM detokenizes a resumed call's tail with a stream primed on the prompt,
    which ends with the restored prefix. A prefix ending inside a character
    makes that stream emit the whole decoded prompt as the tail's first text.
    """
    tail = generated_token_ids[-_PREFIX_BOUNDARY_TAIL_TOKENS:]
    return tokenizer.decode(tail, skip_special_tokens=False).endswith("\ufffd")


class _TokenCaptureSnapshotGate:
    """Fence terminal writes and background prefix CLEARs across a TQ snapshot.

    A closed gate waits for active permit holders and blocks new ones until
    ``reopen``. Each fence takes an epoch before its close is queued.
    ``reopen`` releases every epoch issued so far, so a close that runs after
    the driver already timed out and released its fence is a no-op instead of
    a stuck gate.
    """

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._open = True
        self._active = 0
        self._epoch = 0
        self._released_epoch = 0

    def enter(self) -> None:
        with self._condition:
            while not self._open:
                self._condition.wait()
            self._active += 1

    def exit(self) -> None:
        with self._condition:
            self._active -= 1
            if self._active == 0:
                self._condition.notify_all()

    def begin_epoch(self) -> int:
        with self._condition:
            self._epoch += 1
            return self._epoch

    def close_and_wait(self, epoch: int) -> None:
        with self._condition:
            if epoch <= self._released_epoch:
                return
            self._open = False
            while self._active and self._released_epoch < epoch:
                self._condition.wait()

    def reopen(self) -> None:
        with self._condition:
            self._released_epoch = self._epoch
            self._open = True
            self._condition.notify_all()


class GenerationCutCaptureMixin:
    """Registry, checkpoint-cut, and resume logic shared by generation backends.

    Methods use the capture-host state initialized by the vLLM worker and
    ``_RequestCaptureState``; nothing here touches a generation engine.
    """

    # Initialized by the vLLM worker; declared so type checkers can see the
    # insertion-ordered dicts this mixin evicts from and the route dtype.
    _completed_capture_calls: dict[str, _CompletedCaptureState]
    _generation_cut_receipts: dict[tuple[str, str], Any]
    _generation_cut_tokenizer: Any
    _restored_generation_prefix_calls: int
    _restored_generation_prefix_tokens: int
    routed_experts_dtype: torch.dtype

    def _generation_cut_record_payload(
        self, state: _RequestCaptureState
    ) -> dict[str, Any] | None:
        """Return the backend's extras for a call's first generation-prefix row.

        Text-only hosts have none; the vLLM worker overrides this with media
        metadata.
        """
        del state
        return None

    def _return_routed_experts_enabled(self) -> bool:
        """Whether calls carry router-replay routes; the host implements this."""
        raise NotImplementedError

    # Set by the host's ``setup_token_capture`` before its capture sink, so a
    # cut, which first requires the sink, always finds them.
    _generation_prefix_batch_max_rows: int
    _generation_prefix_batch_max_tokens: int

    def _snapshot_engine_routed_experts(
        self, engine_request_id: str, *, timeout_s: float | None
    ) -> Any | None:
        """Return an active request's full-sequence routes from the engine.

        ``None`` once the request has left the engine. Backends that serve
        router replay override this.
        """
        del engine_request_id, timeout_s
        raise NotImplementedError("this backend cannot read routes for a cut")

    def _snapshot_engine_routed_experts_many(
        self, engine_request_ids: list[str], *, deadline: float | None
    ) -> dict[str, _EngineRouteSnapshot]:
        """Compatibility path for backends with only a single-request lookup.

        vLLM overrides this with one owner-loop handoff. Keep the absolute
        deadline here so a fallback never gives each request a fresh timeout.
        """
        results = {}
        for request_id in engine_request_ids:
            try:
                timeout_s = (
                    None if deadline is None else max(deadline - time.monotonic(), 0.0)
                )
                if timeout_s is not None and timeout_s <= 0:
                    raise TimeoutError("generation route snapshot deadline passed")
                results[request_id] = _EngineRouteSnapshot(
                    routes=self._snapshot_engine_routed_experts(
                        request_id, timeout_s=timeout_s
                    )
                )
            except (TimeoutError, TypeError, ValueError) as error:
                results[request_id] = _EngineRouteSnapshot(error=error)
        return results

    def _pop_request_capture(self, request: Any) -> _RequestCaptureState | None:
        with self._capture_registry_lock:
            state = self._capture_calls.pop(id(request), None)
            if state is not None:
                self._capture_calls_by_model_call_id.pop(state.call.model_call_id, None)
            return state

    def _get_request_capture(self, request: Any) -> _RequestCaptureState | None:
        with self._capture_registry_lock:
            return self._capture_calls.get(id(request))

    def _remember_completed_capture(
        self,
        model_call_id: str,
        coords: Any,
        generation_token_count: int,
        *,
        effective_output_limit: int | None = None,
        terminal_finish_reason: Literal["stop", "length"] | None = None,
        terminal_stop_reason: str | int | None = None,
    ) -> None:
        now = time.monotonic()
        with self._capture_registry_lock:
            # Dict insertion order is completion order. Remove only entries old
            # enough that the response/inventory race has certainly elapsed;
            # never discard fresh evidence because the deployment crossed an
            # arbitrary call-count threshold.
            while self._completed_capture_calls:
                oldest_id = next(iter(self._completed_capture_calls))
                oldest = self._completed_capture_calls[oldest_id]
                if (
                    now - oldest.completed_at_monotonic
                    <= _COMPLETED_CAPTURE_RETENTION_S
                ):
                    break
                self._completed_capture_calls.pop(oldest_id)
            self._completed_capture_calls.pop(model_call_id, None)
            self._completed_capture_calls[model_call_id] = _CompletedCaptureState(
                coords=coords,
                generation_token_count=generation_token_count,
                effective_output_limit=effective_output_limit,
                terminal_finish_reason=terminal_finish_reason,
                terminal_stop_reason=terminal_stop_reason,
                completed_at_monotonic=now,
            )

    def _fetch_generation_cut_chunks(self, continuation: Any) -> list[Any]:
        """Fetch the staged chunks named by a ``GenerationCutContinuation``, in order."""
        chunks = self._fetch_generation_cut_rows(list(continuation.staging_keys))
        if not chunks:
            raise RuntimeError("generation-cut continuation has no staged chunks")
        return chunks

    @checkpoint_phase_sync("prefix_restore_fetch", detail=True)
    def _fetch_generation_cut_rows(self, staging_keys: list[str]) -> list[Any]:
        """Fetch cut rows with their extras and routes, in the requested order.

        Also the fetch callback of the worker's ``PrefixReadBatcher``, which
        coalesces the keys of concurrent restores into one call.
        """
        if self._staging_source is None:
            raise RuntimeError(
                "_staging_source not initialized; call setup_token_capture() first"
            )
        # The finalization read also returns each row's extras, which carry a
        # media call's spans and tensor digests, and its route fragment when
        # router replay is on; plain ``fetch`` returns neither.
        chunks = self._staging_source.fetch_for_finalization(
            staging_keys,
            include_route_fragments=self._return_routed_experts_enabled(),
        )
        if len(chunks) != len(staging_keys):
            raise RuntimeError("generation-cut fetch did not return every staged chunk")
        return chunks

    def _generation_cut_routes(
        self, chunks: list[Any], snapshots: list[Any]
    ) -> torch.Tensor | None:
        """Verify and join the route fragments carried by cut chunks.

        The first chunk stores a normal full, delta-aligned route tensor whose
        last row is the unused final-position placeholder. Every later chunk
        stores one route per chunk token: its first row replaces that
        placeholder, and a new placeholder is appended at the new boundary.
        """
        fragments = [chunk.fragment for chunk in chunks]
        if all(fragment is None for fragment in fragments):
            if self._return_routed_experts_enabled():
                raise RuntimeError(
                    "generation-cut checkpoint has no router-replay payload"
                )
            return None
        if any(fragment is None for fragment in fragments):
            raise RuntimeError("generation-cut route fragments are incomplete")

        routes: torch.Tensor | None = None
        for chunk, snapshot, fragment in zip(chunks, snapshots, fragments, strict=True):
            assert fragment is not None
            if chunk.routed_len != snapshot.delta_len:
                raise RuntimeError(
                    "generation-cut route length does not match token delta: "
                    f"routes={chunk.routed_len} tokens={snapshot.delta_len}"
                )
            if not verify_route_fragment_integrity(
                fragment,
                extras_digest_version=snapshot.extras_digest_version,
                expected_extras_digest=snapshot.extras_digest,
            ):
                raise RuntimeError(
                    "generation-cut route fragment failed integrity check"
                )
            fragment_routes = fragment.routes.cpu().contiguous()
            if fragment_routes.dim() != 3:
                raise RuntimeError("generation-cut routes must have rank 3")
            if routes is None:
                routes = fragment_routes
                continue
            if tuple(fragment_routes.shape[1:]) != tuple(routes.shape[1:]):
                raise RuntimeError("generation-cut route model dimensions changed")
            final = _final_route_placeholder(
                int(fragment_routes.shape[1]),
                int(fragment_routes.shape[2]),
                fragment_routes.dtype,
            )
            routes = torch.cat((routes[:-1], fragment_routes, final), dim=0)

        assert routes is not None
        expected_len = sum(snapshot.delta_len for snapshot in snapshots)
        if routes.shape[0] != expected_len:
            raise RuntimeError(
                "rebuilt generation-cut routes do not match cumulative token delta"
            )
        return routes

    @checkpoint_phase_sync("prefix_restore_reconstruct", detail=True)
    def _rebuild_generation_cut_snapshot(
        self, admission: Any, prefix_token_ids: list[int], chunks: list[Any]
    ) -> _ResolvedGenerationCut:
        """Rebuild and verify the cumulative staged prefix from its fetched chunks."""
        snapshots = [chunk.snapshot for chunk in chunks]
        # The source pops each chunk's route payload out of its decoded extras,
        # so a router-replay chunk with nothing else decodes as ``{}``.
        extras_by_chunk = [chunk.extras for chunk in chunks]
        if any(extras for extras in extras_by_chunk[1:]):
            raise RuntimeError(
                "only the first generation-cut chunk may carry call extras"
            )
        call_extras = extras_by_chunk[0]
        layout = (call_extras or {}).get(_CUT_DIGEST_LAYOUT)
        if layout is not None and (type(layout) is not int or layout != 2):
            raise RuntimeError("unsupported generation-cut digest layout")
        continuation = admission.generation_cut
        weight_versions = [snapshot.weight_version for snapshot in snapshots]
        if weight_versions != sorted(weight_versions):
            raise RuntimeError(
                "generation-cut chunk policy versions are not monotonically non-decreasing"
            )
        current_weight_version = self._rollout_weight_version
        if weight_versions[-1] > current_weight_version:
            raise RuntimeError(
                "generation-cut prefix contains policy version "
                f"{weight_versions[-1]}, newer than the current rollout version "
                f"{current_weight_version}"
            )
        token_ids_delta = [
            token_id for snapshot in snapshots for token_id in snapshot.token_ids_delta
        ]
        token_mask_delta = [
            mask for snapshot in snapshots for mask in snapshot.token_mask_delta
        ]
        generation_logprobs_delta = [
            logprob
            for snapshot in snapshots
            for logprob in snapshot.generation_log_probs_delta
        ]
        if any(
            mask == 0.0
            for snapshot in snapshots[1:]
            for mask in snapshot.token_mask_delta
        ):
            raise RuntimeError(
                "only the first generation-cut chunk may contain prompt tokens"
            )

        # Deferred: nemo_gym is an optional extra absent in non-Gym runs.
        from nemo_gym.token_id_capture.staging.digest import (
            EXTRAS_DIGEST_VERSION,
            STAGING_DIGEST_VERSION,
            compute_chain_hash,
            compute_extras_digest,
            compute_staging_digest,
            hash_token_ids,
        )
        from nemo_gym.token_id_capture.staging.records import StagedCallBaseSnapshot

        delta_len = len(token_ids_delta)
        cum_len = admission.prev_len + delta_len
        # A cut chain may span retries performed after later weight updates.
        # Its token-level behavior logprobs remain attached to their original
        # chunks. The cumulative snapshot uses the oldest contributing version
        # so downstream replay-buffer staleness checks remain conservative.
        weight_version = weight_versions[0]
        routed_experts = self._generation_cut_routes(chunks, snapshots)
        chunk_digests = None
        extras = call_extras
        if layout == 2:
            if routed_experts is None:
                raise RuntimeError("route-digest layout requires route fragments")
            chunk_digests = tuple(snapshot.digest for snapshot in snapshots)
            extras = {
                **(call_extras or {}),
                _CUT_CHUNKS_DIGEST: _cut_chunk_digest(list(chunk_digests)),
            }
        elif routed_experts is not None:
            # Backward-compatible reader for active-prefix and canonical
            # terminal-completion rows written before layout 2.
            extras = {
                **(call_extras or {}),
                "routed_experts": encode_routed_experts(routed_experts),
            }
        extras_digest = compute_extras_digest(extras)
        chain_hash = compute_chain_hash(admission.parent_chain_hash, token_ids_delta)
        cumulative_hash = hash_token_ids(prefix_token_ids + token_ids_delta)
        digest = compute_staging_digest(
            schema_version=admission.schema_version,
            digest_version=STAGING_DIGEST_VERSION,
            extras_digest_version=EXTRAS_DIGEST_VERSION,
            rollout_id=continuation.source_capture_key,
            model_call_id=continuation.source_model_call_id,
            parent_call_id=admission.parent_call_id,
            mode=admission.mode,
            prev_len=admission.prev_len,
            delta_len=delta_len,
            cum_len=cum_len,
            weight_version=weight_version,
            token_ids_delta=token_ids_delta,
            token_mask_delta=token_mask_delta,
            generation_log_probs_delta=generation_logprobs_delta,
            extras_digest=extras_digest,
            chain_hash=chain_hash,
            cumulative_hash=cumulative_hash,
        )
        # Gym's begin_call checks this snapshot's digest and generated-token
        # count against the admission before the call is served.
        snapshot = StagedCallBaseSnapshot(
            rollout_id=continuation.source_capture_key,
            model_call_id=continuation.source_model_call_id,
            parent_call_id=admission.parent_call_id,
            mode=admission.mode,
            prev_len=admission.prev_len,
            delta_len=delta_len,
            cum_len=cum_len,
            weight_version=weight_version,
            digest=digest,
            token_ids_delta=token_ids_delta,
            token_mask_delta=token_mask_delta,
            generation_log_probs_delta=generation_logprobs_delta,
            extras_digest=extras_digest,
            chain_hash=chain_hash,
            cumulative_hash=cumulative_hash,
        )
        # Lets a consumer confirm the exact generated tokens it later receives
        # (e.g. a trained row) without access to the staging store.
        generated_prefix_ids = [
            token_id
            for token_id, mask in zip(token_ids_delta, token_mask_delta, strict=True)
            if mask == 1.0
        ]
        prefix_ids_sha256 = hashlib.sha256(
            ",".join(map(str, generated_prefix_ids)).encode()
        ).hexdigest()
        LOGGER.info(
            "generation prefix restored: rollout_id=%s model_call_id=%s "
            "source_model_call_id=%s prefix_tokens=%d prefix_digest=%s "
            "weight_version_span=[%d,%d] prefix_ids_sha256=%s "
            "prefix_routes_sha256=%s",
            admission.rollout_id,
            admission.model_call_id,
            continuation.source_model_call_id,
            continuation.prefix_token_count,
            continuation.prefix_digest,
            weight_versions[0],
            weight_versions[-1],
            prefix_ids_sha256,
            _prefix_routes_sha256(
                routed_experts, generated_prefix_tokens=len(generated_prefix_ids)
            ),
        )
        return _ResolvedGenerationCut(
            snapshot=snapshot,
            routed_experts=routed_experts,
            chunk_digests=chunk_digests,
        )

    def _resolve_generation_cut(
        self,
        admission: Any,
        prefix_token_ids: list[int],
        snapshots: list[Any] | None = None,
    ) -> _ResolvedGenerationCut | None:
        """Fetch and rebuild the cumulative staged prefix named by an admission."""
        continuation = admission.generation_cut
        if continuation is None:
            return None
        if snapshots is None:
            snapshots = self._fetch_generation_cut_chunks(continuation)
        return self._rebuild_generation_cut_snapshot(
            admission, prefix_token_ids, snapshots
        )

    def _restore_generation_cut_media(self, chunks: list[Any]) -> Any | None:
        """Rebuild a restored call's media from its fetched prefix chunks.

        Text-only hosts have none; the vLLM worker overrides this.
        """
        del chunks
        return None

    def _resolve_usable_generation_cut(
        self,
        admission: Any,
        prefix_token_ids: list[int],
        chunks: list[Any] | None = None,
    ) -> tuple[Any, _ResolvedGenerationCut | None]:
        """Resolve an admission's restored prefix, or drop it so the call regenerates.

        Gym has already consumed the restored cut when it sends this call, and
        generating the whole response from the turn boundary is always valid,
        so a prefix that cannot be fetched, whose media cannot be restored,
        or that does not match its admission costs time, not the rollout.
        Returns the admission to serve and the verified prefix with its media,
        or the admission without its cut and None. ``chunks`` are the cut's
        rows when the caller already read them; otherwise they are read here.
        """
        continuation = admission.generation_cut
        if continuation is None:
            return admission, None
        try:
            # One fetch serves both the media and the prefix.
            if chunks is None:
                chunks = self._fetch_generation_cut_chunks(continuation)
            media = self._restore_generation_cut_media(chunks)
            resolved = GenerationCutCaptureMixin._resolve_generation_cut(
                self, admission, prefix_token_ids, chunks
            )
            snapshot = resolved.snapshot
            # Checked here, before the prefix is spliced into the prompt; Gym's
            # begin_call repeats it and would otherwise fail the call.
            if snapshot.digest != continuation.prefix_digest:
                raise RuntimeError("rebuilt prefix digest does not match the admission")
            generated = sum(mask == 1.0 for mask in snapshot.token_mask_delta)
            if generated != continuation.prefix_token_count:
                raise RuntimeError(
                    f"rebuilt prefix has {generated} generated tokens, "
                    f"admission expects {continuation.prefix_token_count}"
                )
            return admission, replace(resolved, media=media)
        except Exception:  # noqa: BLE001 - any unusable prefix regenerates
            LOGGER.warning(
                "cannot restore the generation prefix of model call %s; "
                "regenerating it from its turn boundary",
                admission.model_call_id,
                exc_info=True,
            )
        # The regenerated call will not extend these rows, so they are retired
        # here instead of by its terminal write.
        self._generation_cut_cleanup_executor.submit(
            GenerationCutCaptureMixin._clear_generation_cut_rows_after_snapshot_fence,
            self,
            admission.model_call_id,
            tuple(continuation.staging_keys),
        )
        return admission.model_copy(update={"generation_cut": None}), None

    def get_generation_prefix_restore_counts(self) -> dict[str, int]:
        """Calls resumed from a restored generation prefix since this process started.

        Returns:
            ``restored_calls`` and their total ``restored_prefix_tokens``.
        """
        with self._capture_registry_lock:
            return {
                "restored_calls": self._restored_generation_prefix_calls,
                "restored_prefix_tokens": self._restored_generation_prefix_tokens,
            }

    def _abort_request_capture(
        self, request: Any, *, reason: str, blocking: bool = True
    ) -> bool:
        """Drop the in-flight capture state for a request that errored.

        A generation cut or a terminal write holds the call's lifecycle lock
        across a TQ round trip. With ``blocking=False`` the abort does not wait
        for it and returns False instead; it returns True once the call needs
        no abort.
        """
        state = self._get_request_capture(request)
        if state is None:
            return True
        if not state.lifecycle_lock.acquire(blocking=blocking):
            return False
        obsolete_staging_keys: tuple[str, ...] = ()
        try:
            if not state.terminal_started:
                state.terminal_started = True
                popped = self._pop_request_capture(request)
                if popped is state and self.token_capture is not None:
                    coords = self.token_capture.fail_call(state.call, reason=reason)
                    self._remember_completed_capture(
                        state.call.model_call_id, coords, 0
                    )
                    obsolete_staging_keys = tuple(state.generation_cut_staging_keys)
        finally:
            state.lifecycle_lock.release()
        if obsolete_staging_keys:
            # Off the caller's thread: the clear may wait at the snapshot gate,
            # and the fast abort runs on the serving event loop.
            self._generation_cut_cleanup_executor.submit(
                GenerationCutCaptureMixin._clear_generation_cut_rows_after_snapshot_fence,
                self,
                state.call.model_call_id,
                obsolete_staging_keys,
            )
        return True

    def _clear_generation_cut_rows_after_snapshot_fence(
        self, model_call_id: str, staging_keys: tuple[str, ...]
    ) -> None:
        """Delete an ended call's chunk rows outside any cut-to-snapshot window."""
        gate = self._token_capture_snapshot_gate
        gate.enter()
        try:
            GenerationCutCaptureMixin._clear_generation_cut_rows(
                self, model_call_id, staging_keys
            )
        finally:
            gate.exit()

    def _clear_generation_cut_rows(
        self, model_call_id: str, staging_keys: tuple[str, ...]
    ) -> None:
        """Delete a call's chunk rows; the caller holds a snapshot-gate permit.

        Once the call has ended no live reader needs them, and a committed
        snapshot that names them saved its own copy of TransferQueue, so only a
        snapshot still in progress must be waited out.
        """
        sink = self._capture_sink
        if not staging_keys or sink is None:
            return
        try:
            sink.clear(list(staging_keys))
        except Exception:  # noqa: BLE001 - the call has already ended
            LOGGER.exception(
                "failed to clear obsolete generation chunks for model call %s",
                model_call_id,
            )

    async def _run_generation_cut_control(
        self,
        operation: Callable[..., Any],
        *args: Any,
        submitted_at: float | None = None,
    ) -> Any:
        """Run cut control independently of blocked terminal capture writes."""
        with checkpoint_phase("prefix_control"):
            context = contextvars.copy_context()
            if submitted_at is None:
                submitted_at = time.perf_counter()
            completed_at: float | None = None

            def execute() -> Any:
                def run() -> Any:
                    nonlocal completed_at
                    prefix_cut_wait(
                        "control_queue_wait", time.perf_counter() - submitted_at
                    )
                    try:
                        with prefix_cut_timing("control_execution"):
                            return operation(*args)
                    finally:
                        completed_at = time.perf_counter()

                return context.run(run)

            try:
                return await asyncio.get_running_loop().run_in_executor(
                    self._generation_cut_control_executor,
                    cast(Callable[..., Any], execute),
                )
            finally:
                if completed_at is not None:
                    prefix_cut_wait(
                        "control_result_delivery", time.perf_counter() - completed_at
                    )

    async def _checkpoint_generation_cut_control(
        self, inventory: Any, *, deadline: float | None
    ) -> Any:
        """Admit independent inventories without abandoning cancelled cut threads."""
        with self._capture_registry_lock:
            cached = self._generation_cut_receipts.get(
                (inventory.checkpoint_id, inventory.inventory_digest)
            )
        if cached is not None:
            return cached
        submitted_at = time.perf_counter()

        async def execute() -> Any:
            prefix_cut_wait(
                "control_admission_wait", time.perf_counter() - submitted_at
            )
            return await self._run_generation_cut_control(
                functools.partial(
                    self._checkpoint_generation_cut, inventory, deadline=deadline
                ),
                submitted_at=submitted_at,
            )

        return await self._generation_cut_control.run(
            checkpoint_id=inventory.checkpoint_id,
            inventory_digest=inventory.inventory_digest,
            call_ids=frozenset(
                prefix.model_call_id for prefix in inventory.active_prefixes
            ),
            deadline=deadline,
            operation=execute,
        )

    def _completed_generation_cut_ack(self, prefix: Any, checkpoint_id: str) -> Any:
        """Describe a call whose terminal path won the cut/abort race."""
        from nemo_gym._checkpoint.generation_cut import (
            GenerationCutPrefixAck,
        )
        from nemo_gym.episode_types import EpisodeId

        with self._capture_registry_lock:
            completed = self._completed_capture_calls.get(prefix.model_call_id)
        expected_capture_key = EpisodeId(
            rollout_id=prefix.rollout_id, attempt=prefix.attempt
        ).capture_key
        if (
            completed is not None
            and completed.coords.rollout_id != expected_capture_key
        ):
            raise RuntimeError(
                "generation-prefix inventory identity does not match the "
                f"completed call: model_call_id={prefix.model_call_id!r}"
            )
        if (
            completed is not None
            and completed.coords.disposition == "staged"
            and completed.effective_output_limit is not None
            and completed.terminal_finish_reason is not None
        ):
            return GenerationCutPrefixAck(
                **prefix.model_dump(mode="json"),
                disposition="durable_prefix",
                cut_kind="terminal_completion",
                frozen_buffer_id=f"terminal/{checkpoint_id}",
                staging_keys=(completed.coords.staging_key,),
                prefix_token_count=completed.generation_token_count,
                prefix_digest=completed.coords.digest,
                effective_output_limit=completed.effective_output_limit,
                terminal_finish_reason=completed.terminal_finish_reason,
                terminal_stop_reason=completed.terminal_stop_reason,
            )
        # Evidence for this completion aged out; restart it from its prior boundary.
        return GenerationCutPrefixAck.failure(prefix)

    def _frozen_cut_engine_request_id(self, state: _RequestCaptureState) -> str | None:
        """Select a frozen call's live lookup without holding its token lock."""
        with state.lock:
            needs_engine_routes = (
                state.capture_routed_experts
                and state.frozen_cut is not None
                and state.frozen_cut[1] > 0
                and state.terminal_routed_experts is None
            )
            engine_request_id = state.engine_request_id
        if not needs_engine_routes:
            return None
        if engine_request_id is None:
            raise ValueError("no vLLM output has named the request yet")
        return engine_request_id

    def _frozen_cut_engine_routed_experts(
        self, state: _RequestCaptureState, *, deadline: float | None
    ) -> Any | None:
        """Read engine routes after freezing, without blocking token observation."""
        engine_request_id = self._frozen_cut_engine_request_id(state)
        if engine_request_id is None:
            return None
        timeout_s = None if deadline is None else max(deadline - time.monotonic(), 0.0)
        return self._snapshot_engine_routed_experts(
            engine_request_id, timeout_s=timeout_s
        )

    def _generation_cut_transaction(
        self,
        prefix: Any,
        state: _RequestCaptureState,
        checkpoint_id: str,
        *,
        deadline: float | None = None,
        batch_route_lookup: bool = False,
    ) -> Generator[Any, Any, Any]:
        """Keep a live call frozen until transport and acknowledgement validate.

        With batched lookups it first yields a ``_RouteSnapshotRequest`` and
        receives an ``_EngineRouteSnapshot``. It then yields
        ``(record, sequence, route_payload)``, receives the staged key, then yields its
        validated acknowledgement; the caller resumes it once more to seal.
        A call with nothing to stage returns its acknowledgement (or ``None``
        if abort won the race) without yielding a record. Closing before the
        seal rolls the frozen cut back.
        """
        from nemo_gym._checkpoint.generation_cut import (
            GenerationCutPrefixAck,
        )
        from nemo_gym.episode_types import EpisodeId

        capture = self.token_capture
        sink = self._capture_sink
        if capture is None or sink is None:
            raise RuntimeError("generation-prefix cuts require token capture setup")

        # Abort and terminal completion use the same lifecycle lock. Once the
        # cut owns it, the call cannot be removed until its detached buffer is
        # staged or rolled back. If abort won first, use its terminal evidence.
        lock_started_at = time.perf_counter()
        with state.lifecycle_lock:
            prefix_cut_wait(
                "call_lifecycle_lock_wait", time.perf_counter() - lock_started_at
            )
            with self._capture_registry_lock:
                current = self._capture_calls_by_model_call_id.get(prefix.model_call_id)
            if current is not state or state.terminal_started:
                return None

            expected_capture_key = EpisodeId(
                rollout_id=prefix.rollout_id, attempt=prefix.attempt
            ).capture_key
            if state.call.rollout_id != expected_capture_key:
                raise RuntimeError(
                    "generation-prefix inventory identity does not match the "
                    f"active call: model_call_id={prefix.model_call_id!r}, "
                    f"inventory_rollout_id={prefix.rollout_id!r}, "
                    f"worker_rollout_id={state.call.rollout_id!r}"
                )

            with prefix_cut_timing("capture_freeze"), state.lock:
                uncuttable_reason = state.uncuttable_reason
                observation_error = state.observation_error
                effective_output_limit = state.effective_output_limit
                terminal_finish_reason = state.terminal_finish_reason
                terminal_stop_reason = state.terminal_stop_reason
                if (
                    uncuttable_reason is None
                    and observation_error is None
                    and effective_output_limit is not None
                ):
                    (
                        frozen_buffer_id,
                        chunk_sequence,
                        chunk_token_ids,
                        chunk_logprobs,
                        generated_token_ids,
                        generated_logprobs,
                    ) = state.freeze_for_checkpoint(checkpoint_id)
            if uncuttable_reason is not None:
                # Expected for this request; the worker warned when it was set.
                LOGGER.debug(
                    "not cutting model call %s: %s",
                    prefix.model_call_id,
                    uncuttable_reason,
                )
                return GenerationCutPrefixAck(
                    **prefix.model_dump(mode="json"),
                    disposition="durable_failure",
                )
            if observation_error is not None:
                # A dropped delta leaves the buffer short of what vLLM decoded.
                # Only this call regenerates; the rest of the inventory is cut.
                LOGGER.warning(
                    "cannot cut model call %s: %s",
                    prefix.model_call_id,
                    observation_error,
                )
                return GenerationCutPrefixAck(
                    **prefix.model_dump(mode="json"),
                    disposition="durable_failure",
                )
            if effective_output_limit is None:
                # Checked before freezing, so the call's buffer is untouched.
                LOGGER.warning(
                    "cannot cut model call %s: vLLM did not resolve its "
                    "effective output limit",
                    prefix.model_call_id,
                )
                return GenerationCutPrefixAck(
                    **prefix.model_dump(mode="json"),
                    disposition="durable_failure",
                )
            inherited_prefix_token_count = (
                state.call.admission.generation_cut.prefix_token_count
                if state.call.admission.generation_cut is not None
                else 0
            )
            total_generation_token_count = inherited_prefix_token_count + len(
                generated_token_ids
            )
            with prefix_cut_timing("utf8_boundary_check"):
                cannot_cut_boundary = total_generation_token_count == 0 or (
                    generated_token_ids
                    and _ends_inside_character(
                        self._generation_cut_tokenizer,
                        state.resumed_generation_token_ids + generated_token_ids,
                    )
                )
            if cannot_cut_boundary:
                with state.lock:
                    state.rollback_frozen_cut(checkpoint_id)
                return GenerationCutPrefixAck(
                    **prefix.model_dump(mode="json"),
                    disposition="durable_failure",
                )
            try:
                engine_routed_experts = None
                if batch_route_lookup:
                    request_id = self._frozen_cut_engine_request_id(state)
                    if request_id is not None:
                        positions = len(chunk_token_ids)
                        if not state.generation_cut_staging_keys:
                            positions += (
                                len(state.prompt_token_ids)
                                - state.call.admission.prev_len
                            )
                        try:
                            snapshot = yield _RouteSnapshotRequest(
                                request_id, positions
                            )
                            if not isinstance(snapshot, _EngineRouteSnapshot):
                                raise TypeError(
                                    "cut requires an engine route snapshot result"
                                )
                            if snapshot.error is not None:
                                raise snapshot.error
                            engine_routed_experts = snapshot.routes
                        except BaseException:
                            # Includes GeneratorExit when a queued group is
                            # closed before lookup/resume. No TQ row exists yet.
                            with state.lock:
                                state.rollback_frozen_cut(checkpoint_id)
                            raise
                with prefix_cut_timing("route_fetch_total"):
                    if not batch_route_lookup:
                        engine_routed_experts = self._frozen_cut_engine_routed_experts(
                            state, deadline=deadline
                        )
                    if isinstance(engine_routed_experts, list):
                        # Only the cut executor mutates this cache. Do the
                        # potentially large copy without blocking HTTP token
                        # observation on state.lock.
                        engine_routed_experts = state.absorb_engine_route_pieces(
                            engine_routed_experts
                        )
                with state.lock, prefix_cut_timing("route_prefix_assembly"):
                    (
                        chunk_routed_experts,
                        cumulative_routed_experts,
                    ) = state.frozen_cut_routed_experts(engine_routed_experts)
            except (TimeoutError, TypeError, ValueError) as error:
                # Only this call regenerates; the rest of the inventory is cut,
                # and the next checkpoint tries this call again.
                with state.lock:
                    if state.frozen_cut is not None:
                        state.rollback_frozen_cut(checkpoint_id)
                LOGGER.warning(
                    "cannot cut model call %s: routed experts: %s: %s",
                    prefix.model_call_id,
                    type(error).__name__,
                    error,
                )
                return GenerationCutPrefixAck(
                    **prefix.model_dump(mode="json"),
                    disposition="durable_failure",
                )

            staged_key = None
            try:
                cumulative_extras = self._generation_cut_record_payload(state)
                chunk_record = None
                route_payload = None
                chunk_digests = state.generation_cut_chunk_digests
                new_layout = state.capture_routed_experts and chunk_digests is not None
                if new_layout:
                    cumulative_extras = {
                        **(cumulative_extras or {}),
                        _CUT_DIGEST_LAYOUT: 2,
                    }
                    candidate_digests = list(chunk_digests)
                    if len(candidate_digests) != len(state.generation_cut_staging_keys):
                        raise RuntimeError(
                            "generation-cut chunk digests and keys differ"
                        )
                    if chunk_token_ids:
                        if chunk_routed_experts is None:
                            raise ValueError(
                                "router-replay cut has no new route fragment"
                            )
                        with prefix_cut_timing("delta_route_encode"):
                            route_payload = RoutedExpertsStagingPayload(
                                chunk_routed_experts
                            )
                            chunk_extras = {"routed_experts": route_payload.encoded}
                        with prefix_cut_timing("delta_record_build_and_digest"):
                            if not state.generation_cut_staging_keys:
                                # The first persisted row carries prompt/call
                                # extras and full routes, but is distinct from
                                # the small cumulative digest record below.
                                chunk_record = capture.build_prefix_record(
                                    state.call,
                                    prompt_token_ids=state.prompt_token_ids,
                                    generated_token_ids=generated_token_ids,
                                    generated_logprobs=generated_logprobs,
                                    extras={**cumulative_extras, **chunk_extras},
                                )
                            else:
                                chunk_record = capture.build_generation_chunk_record(
                                    state.call,
                                    generated_token_ids=chunk_token_ids,
                                    generated_logprobs=chunk_logprobs,
                                    extras=chunk_extras,
                                )
                        candidate_digests.append(chunk_record.digest)
                    cumulative_extras[_CUT_CHUNKS_DIGEST] = _cut_chunk_digest(
                        candidate_digests
                    )
                if cumulative_routed_experts is not None:
                    route_bytes = (
                        cumulative_routed_experts.numel()
                        * cumulative_routed_experts.element_size()
                    )
                    prefix_cut_count("cumulative_route_bytes", route_bytes)
                    prefix_cut_count(
                        "max_cumulative_route_bytes", route_bytes, maximum=True
                    )
                    if not new_layout:
                        with prefix_cut_timing("cumulative_route_encode"):
                            if not state.generation_cut_staging_keys:
                                route_payload = RoutedExpertsStagingPayload(
                                    cumulative_routed_experts
                                )
                                encoded_routes = route_payload.encoded
                            else:
                                encoded_routes = encode_routed_experts(
                                    cumulative_routed_experts
                                )
                        cumulative_extras = {
                            **(cumulative_extras or {}),
                            "routed_experts": encoded_routes,
                        }
                prefix_cut_count(
                    "cumulative_generation_tokens", len(generated_token_ids)
                )
                prefix_cut_count("new_chunk_tokens", len(chunk_token_ids))
                with prefix_cut_timing("cumulative_record_build_and_digest"):
                    cumulative_record = capture.build_prefix_record(
                        state.call,
                        prompt_token_ids=state.prompt_token_ids,
                        generated_token_ids=generated_token_ids,
                        generated_logprobs=generated_logprobs,
                        extras=cumulative_extras,
                    )
                if chunk_token_ids:
                    if chunk_routed_experts is not None:
                        prefix_cut_count(
                            "delta_route_bytes",
                            chunk_routed_experts.numel()
                            * chunk_routed_experts.element_size(),
                        )
                    if new_layout:
                        assert chunk_record is not None
                    elif not state.generation_cut_staging_keys:
                        chunk_record = cumulative_record
                    else:
                        chunk_extras = None
                        if chunk_routed_experts is not None:
                            with prefix_cut_timing("delta_route_encode"):
                                route_payload = RoutedExpertsStagingPayload(
                                    chunk_routed_experts
                                )
                                chunk_extras = {"routed_experts": route_payload.encoded}
                        with prefix_cut_timing("delta_record_build_and_digest"):
                            chunk_record = capture.build_generation_chunk_record(
                                state.call,
                                generated_token_ids=chunk_token_ids,
                                generated_logprobs=chunk_logprobs,
                                extras=chunk_extras,
                            )
                    staged_key = yield chunk_record, chunk_sequence, route_payload
                with prefix_cut_timing("acknowledgement_build"), state.lock:
                    candidate_staging_keys = tuple(state.generation_cut_staging_keys)
                    if staged_key is not None:
                        candidate_staging_keys += (staged_key,)
                    acknowledgement = GenerationCutPrefixAck(
                        **prefix.model_dump(mode="json"),
                        disposition="durable_prefix",
                        cut_kind="active_prefix",
                        frozen_buffer_id=frozen_buffer_id,
                        staging_keys=candidate_staging_keys,
                        prefix_token_count=total_generation_token_count,
                        prefix_digest=cumulative_record.digest,
                        effective_output_limit=effective_output_limit,
                        terminal_finish_reason=terminal_finish_reason,
                        terminal_stop_reason=terminal_stop_reason,
                    )
                if chunk_token_ids:
                    # Validate every acknowledgement in a batch before sealing
                    # any frozen cut into its live request state. A failed
                    # acknowledgement leaves the cut available for rollback.
                    yield acknowledgement
                with prefix_cut_timing("cut_seal"), state.lock:
                    state.seal_frozen_cut(checkpoint_id)
                    if staged_key is not None:
                        state.generation_cut_staging_keys.append(staged_key)
                        if new_layout:
                            assert (
                                chunk_record is not None and chunk_digests is not None
                            )
                            chunk_digests.append(chunk_record.digest)
            except BaseException:
                # Generator.close() injects GeneratorExit when another row in
                # the batch fails. Roll back without hiding the root failure.
                with state.lock:
                    frozen = state.frozen_cut
                    if frozen is not None and frozen[0] == checkpoint_id:
                        state.rollback_frozen_cut(checkpoint_id)
                if staged_key is not None:
                    try:
                        sink.clear([staged_key])
                    except Exception:  # noqa: BLE001 — preserve the cut failure
                        LOGGER.exception(
                            "Failed to clear rejected generation-prefix row %s",
                            staged_key,
                        )
                raise
            return acknowledgement

    def _checkpoint_generation_cut_batched(
        self, inventory: Any, *, deadline: float | None = None
    ) -> list[Any]:
        """Batch active prefixes while retaining each call's lifecycle lock.

        Past ``deadline`` no further TQ write starts: rows frozen for the next
        batch roll back and, with every call not yet reached, are acknowledged
        as failures. ``max_rows=1`` writes each call in its own put.

        Gym acknowledges cuts per call, so a call that cannot be cut (a failed
        freeze, write, acknowledgement or seal) is acknowledged as a failure
        and regenerates after a restore; the other calls keep their cuts. A
        failed batch put is retried one row per put, so one bad row fails only
        its own call.
        """
        from nemo_gym._checkpoint.generation_cut import GenerationCutPrefixAck

        max_rows = self._generation_prefix_batch_max_rows
        max_tokens = self._generation_prefix_batch_max_tokens
        call_ids = [prefix.model_call_id for prefix in inventory.active_prefixes]
        if len(set(call_ids)) != len(call_ids):
            raise ValueError(
                "generation-cut inventory contains duplicate model-call identities"
            )

        sink = self._capture_sink
        acknowledgements: list[Any] = []
        pending: list[
            tuple[
                Generator[Any, Any, Any],
                Any,
                int,
                Any,
                RoutedExpertsStagingPayload | None,
            ]
        ] = []
        pending_tokens = 0
        route_pending: list[
            tuple[Generator[Any, Any, Any], Any, _RouteSnapshotRequest]
        ] = []
        route_pending_tokens = 0
        route_group_limit = min(max_rows, _ROUTE_SNAPSHOT_GROUP_MAX_CALLS)
        uncut: list[Any] = []

        def past_deadline() -> bool:
            return deadline is not None and time.monotonic() >= deadline

        def release_pending() -> None:
            nonlocal pending_tokens
            for transaction, _, _, prefix, _ in pending:
                transaction.close()
                uncut.append(prefix)
            pending.clear()
            pending_tokens = 0

        def release_route_pending() -> None:
            nonlocal route_pending_tokens
            for transaction, prefix, _ in route_pending:
                transaction.close()
                uncut.append(prefix)
            route_pending.clear()
            route_pending_tokens = 0

        def fail(prefix: Any) -> None:
            LOGGER.exception(
                "cannot cut model call %s; it regenerates after a restore",
                prefix.model_call_id,
            )
            acknowledgements.append(GenerationCutPrefixAck.failure(prefix))

        def stage(
            batch: list[tuple[Any, Any, int, Any, RoutedExpertsStagingPayload | None]],
        ) -> list[str | None]:
            """Write a batch; if its put fails, write each row alone."""
            try:
                return sink.stage_generation_prefix_batch(
                    [record for _, record, _, _, _ in batch],
                    checkpoint_id=inventory.checkpoint_id,
                    chunk_sequences=[sequence for _, _, sequence, _, _ in batch],
                    attachments=[
                        {ROUTED_EXPERTS_STAGING_ATTACHMENT: routes}
                        if routes is not None
                        else None
                        for _, _, _, _, routes in batch
                    ],
                )
            except Exception:
                # The sink has already discarded the batch's keys.
                if len(batch) == 1:
                    fail(batch[0][3])
                    return [None]
                LOGGER.warning(
                    "generation-prefix batch of %d rows failed; writing each alone",
                    len(batch),
                    exc_info=True,
                )
            keys: list[str | None] = []
            for _, record, sequence, prefix, routes in batch:
                if past_deadline():
                    uncut.append(prefix)
                    keys.append(None)
                    continue
                try:
                    keys += sink.stage_generation_prefix_batch(
                        [record],
                        checkpoint_id=inventory.checkpoint_id,
                        chunk_sequences=[sequence],
                        attachments=[
                            {ROUTED_EXPERTS_STAGING_ATTACHMENT: routes}
                            if routes is not None
                            else None
                        ],
                    )
                except Exception:
                    fail(prefix)
                    keys.append(None)
            return keys

        def flush() -> None:
            nonlocal pending_tokens
            if not pending:
                return
            if past_deadline():
                release_pending()
                return
            batch = list(pending)
            pending.clear()
            pending_tokens = 0
            keys: list[str | None] = []
            handed = 0
            try:
                prefix_cut_count("batches", 1)
                prefix_cut_count("batch_rows", len(batch))
                prefix_cut_count(
                    "batch_token_positions",
                    sum(len(record.token_ids_delta) for _, record, _, _, _ in batch),
                )
                prefix_cut_count("max_batch_rows", len(batch), maximum=True)
                with (
                    prefix_cut_timing("batch_stage_total"),
                    checkpoint_phase(
                        "prefix_batch_put",
                        attributes={"rl.checkpoint.rows": len(batch)},
                    ),
                ):
                    keys = stage(batch)
                ready = []
                for (transaction, _, _, prefix, _), key in zip(
                    batch, keys, strict=True
                ):
                    handed += 1
                    if key is None:
                        # Closing rolls the frozen cut back.
                        transaction.close()
                        continue
                    try:
                        # A failure here rolls the call back and clears its row.
                        with prefix_cut_timing("transaction_acknowledge"):
                            ready.append((transaction, prefix, transaction.send(key)))
                    except Exception:
                        fail(prefix)
                # Every acknowledgement in the batch is built before any call
                # seals its frozen cut into its live request state.
                for transaction, prefix, acknowledgement in ready:
                    try:
                        with prefix_cut_timing(
                            "transaction_seal", expected_stop_iteration=True
                        ):
                            transaction.send(None)
                    except StopIteration:
                        acknowledgements.append(acknowledgement)
                    except Exception:
                        fail(prefix)
                    else:
                        transaction.close()
                        acknowledgements.append(GenerationCutPrefixAck.failure(prefix))
            finally:
                # Only an interruption leaves written rows that no transaction
                # received; nothing may advertise them.
                orphaned = [key for key in keys[handed:] if key is not None]
                if orphaned:
                    try:
                        sink.clear(orphaned)
                    except Exception:  # noqa: BLE001 — preserve the interruption
                        LOGGER.exception(
                            "Failed to clear interrupted generation-prefix batch"
                        )
                with prefix_cut_timing("transaction_release"):
                    for transaction, _, _, _, _ in batch:
                        transaction.close()

        def queue_record(
            transaction: Generator[Any, Any, Any],
            prefix: Any,
            record: Any,
            sequence: int,
            routes: RoutedExpertsStagingPayload | None,
        ) -> None:
            nonlocal pending_tokens
            tokens = len(record.token_ids_delta)
            if pending and pending_tokens + tokens > max_tokens:
                flush()
            pending.append((transaction, record, sequence, prefix, routes))
            pending_tokens += tokens
            if len(pending) >= max_rows or pending_tokens >= max_tokens:
                # An oversized single row is written alone, never split.
                flush()

        def resolve_route_pending() -> None:
            """Fetch a frozen group's routes, then transfer records to TQ batching."""
            nonlocal route_pending_tokens
            if not route_pending:
                return
            if past_deadline():
                release_route_pending()
                return
            group = list(route_pending)
            route_pending.clear()
            route_pending_tokens = 0
            handed = 0
            try:
                request_ids = list(
                    dict.fromkeys(request.engine_request_id for _, _, request in group)
                )
                try:
                    snapshots = self._snapshot_engine_routed_experts_many(
                        request_ids, deadline=deadline
                    )
                except Exception as error:
                    # A transport/owner failure affects this bounded group.
                    # Never retry each request with a fresh control timeout.
                    LOGGER.exception("generation route snapshot group failed")
                    snapshots = {
                        request_id: _EngineRouteSnapshot(error=error)
                        for request_id in request_ids
                    }
                for transaction, prefix, request in group:
                    handed += 1
                    transferred = False
                    try:
                        if past_deadline():
                            uncut.append(prefix)
                            continue
                        with prefix_cut_timing(
                            "transaction_route_resume", expected_stop_iteration=True
                        ):
                            record, sequence, routes = transaction.send(
                                snapshots[request.engine_request_id]
                            )
                        queue_record(transaction, prefix, record, sequence, routes)
                        transferred = True
                    except StopIteration as finished:
                        acknowledgements.append(finished.value)
                    except Exception:
                        fail(prefix)
                    finally:
                        if not transferred:
                            transaction.close()
            finally:
                # Transactions handed to pending belong to flush(); only this
                # group's not-yet-resumed transactions remain ours to close.
                for transaction, _, _ in group[handed:]:
                    transaction.close()

        try:
            for index, prefix in enumerate(inventory.active_prefixes):
                if past_deadline():
                    release_route_pending()
                    release_pending()
                    uncut.extend(inventory.active_prefixes[index:])
                    break
                with prefix_cut_timing("registry_lookup"), self._capture_registry_lock:
                    state = self._capture_calls_by_model_call_id.get(
                        prefix.model_call_id
                    )
                try:
                    if state is None:
                        acknowledgements.append(
                            self._completed_generation_cut_ack(
                                prefix, inventory.checkpoint_id
                            )
                        )
                        continue
                    transaction = self._generation_cut_transaction(
                        prefix,
                        state,
                        inventory.checkpoint_id,
                        deadline=deadline,
                        batch_route_lookup=True,
                    )
                    with prefix_cut_timing(
                        "transaction_prepare_total", expected_stop_iteration=True
                    ):
                        prepared = next(transaction)
                except StopIteration as finished:
                    acknowledgement = finished.value
                    if acknowledgement is None:
                        acknowledgement = self._completed_generation_cut_ack(
                            prefix, inventory.checkpoint_id
                        )
                    acknowledgements.append(acknowledgement)
                    continue
                except Exception:
                    fail(prefix)
                    if state is not None:
                        # The transaction rolls back only failures from its
                        # route snapshot on; release a cut it froze earlier.
                        with state.lock:
                            frozen = state.frozen_cut
                            if (
                                frozen is not None
                                and frozen[0] == inventory.checkpoint_id
                            ):
                                state.rollback_frozen_cut(inventory.checkpoint_id)
                    continue
                if isinstance(prepared, _RouteSnapshotRequest):
                    # Check before adding this call. It is already frozen, so
                    # close it too if resolving the previous group interrupts.
                    try:
                        if (
                            route_pending
                            and route_pending_tokens + prepared.token_positions
                            > max_tokens
                        ):
                            resolve_route_pending()
                    except BaseException:
                        transaction.close()
                        raise
                    route_pending.append((transaction, prefix, prepared))
                    route_pending_tokens += prepared.token_positions
                    if (
                        len(route_pending) >= route_group_limit
                        or route_pending_tokens >= max_tokens
                    ):
                        resolve_route_pending()
                else:
                    try:
                        record, sequence, routes = prepared
                        queue_record(transaction, prefix, record, sequence, routes)
                    except BaseException:
                        transaction.close()
                        raise
            resolve_route_pending()
            flush()
        finally:
            for transaction, _, _ in route_pending:
                transaction.close()
            for transaction, _, _, _, _ in pending:
                transaction.close()
        if uncut:
            prefix_cut_count("deadline_skipped_calls", len(uncut))
            LOGGER.warning(
                "generation cut passed its control deadline: checkpoint_id=%s "
                "uncut=%d of %d",
                inventory.checkpoint_id,
                len(uncut),
                len(inventory.active_prefixes),
            )
            acknowledgements.extend(
                GenerationCutPrefixAck.failure(prefix) for prefix in uncut
            )
        by_ticket = {ack.ticket_id: ack for ack in acknowledgements}
        return [by_ticket[prefix.ticket_id] for prefix in inventory.active_prefixes]

    @checkpoint_phase_sync("prefix_cut")
    def _checkpoint_generation_cut(
        self, inventory: Any, *, deadline: float | None = None
    ) -> Any:
        """Stage a stable prefix for every call named by Gym's frozen inventory.

        ``deadline`` is a ``time.monotonic()`` bound. Gym stops waiting for the
        receipt at its own control deadline, so calls not reached by then are
        acknowledged as failures (they regenerate after restore) instead of
        holding the single cut-control thread for the next checkpoint.
        """
        from nemo_gym._checkpoint.generation_cut import GenerationCutReceipt

        capture = self.token_capture
        sink = self._capture_sink
        if capture is None or sink is None:
            raise RuntimeError("generation-prefix cuts require token capture setup")
        receipt_key = (inventory.checkpoint_id, inventory.inventory_digest)
        with self._capture_registry_lock:
            cached_receipt = self._generation_cut_receipts.get(receipt_key)
        if cached_receipt is not None:
            return cached_receipt
        acknowledgements = self._checkpoint_generation_cut_batched(
            inventory, deadline=deadline
        )
        receipt = GenerationCutReceipt(
            checkpoint_id=inventory.checkpoint_id,
            cut_id=f"worker-{inventory.inventory_digest}",
            inventory_digest=inventory.inventory_digest,
            inventory=inventory,
            backend_snapshot_id=f"tq-{inventory.inventory_digest}",
            prefixes=tuple(acknowledgements),
        )
        with self._capture_registry_lock:
            self._generation_cut_receipts[receipt_key] = receipt
            if len(self._generation_cut_receipts) > 256:
                self._generation_cut_receipts.pop(
                    next(iter(self._generation_cut_receipts))
                )
        return receipt
