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
import hashlib
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from typing import Any, Literal

import torch

from nemo_rl.experience.route_assembly import (
    ROUTE_MISSING_SENTINEL,
    verify_route_fragment_integrity,
)
from nemo_rl.utils.routed_experts_codec import encode_routed_experts

LOGGER = logging.getLogger(__name__)

# Completion evidence only bridges the short race between a terminal response
# and Gym freezing its admitted-call inventory. Time-based retention avoids
# evicting fresh evidence merely because a large deployment completed more
# than 100k calls, while still bounding idle-job memory growth.
_COMPLETED_CAPTURE_RETENTION_S = 60.0 * 60.0


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
    """Drain active terminal writes and block new ones across a TQ snapshot.

    Each fence takes an epoch before its close is queued. ``reopen`` releases
    every epoch issued so far, so a close that runs after the driver already
    timed out and released its fence is a no-op instead of a stuck gate.
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

    def _snapshot_engine_routed_experts(
        self, engine_request_id: str, *, timeout_s: float | None
    ) -> Any | None:
        """Return an active request's full-sequence routes from the engine.

        ``None`` once the request has left the engine. Backends that serve
        router replay override this.
        """
        del engine_request_id, timeout_s
        raise NotImplementedError("this backend cannot read routes for a cut")

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
        if self._staging_source is None:
            raise RuntimeError(
                "_staging_source not initialized; call setup_token_capture() first"
            )
        staging_keys = list(continuation.staging_keys)
        # The finalization read also returns each row's extras, which carry a
        # media call's spans and tensor digests, and its route fragment when
        # router replay is on; plain ``fetch`` returns neither.
        chunks = self._staging_source.fetch_for_finalization(
            staging_keys,
            include_route_fragments=self._return_routed_experts_enabled(),
        )
        if len(chunks) != len(staging_keys):
            raise RuntimeError("generation-cut fetch did not return every staged chunk")
        if not chunks:
            raise RuntimeError("generation-cut continuation has no staged chunks")
        return chunks

    def _generation_cut_route_extras(
        self, chunks: list[Any], snapshots: list[Any]
    ) -> tuple[dict[str, Any] | None, torch.Tensor | None]:
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
            return None, None
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
        return {"routed_experts": encode_routed_experts(routes)}, routes

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
        route_extras, routed_experts = self._generation_cut_route_extras(
            chunks, snapshots
        )
        extras = (
            call_extras
            if route_extras is None
            else {**(call_extras or {}), **route_extras}
        )
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
        return _ResolvedGenerationCut(snapshot=snapshot, routed_experts=routed_experts)

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
        self, admission: Any, prefix_token_ids: list[int]
    ) -> tuple[Any, _ResolvedGenerationCut | None]:
        """Resolve an admission's restored prefix, or drop it so the call regenerates.

        Gym has already consumed the restored cut when it sends this call, and
        generating the whole response from the turn boundary is always valid,
        so a prefix that cannot be fetched, whose media cannot be restored,
        or that does not match its admission costs time, not the rollout.
        Returns the admission to serve and the verified prefix with its media,
        or the admission without its cut and None.
        """
        continuation = admission.generation_cut
        if continuation is None:
            return admission, None
        try:
            # One fetch serves both the media and the prefix.
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
    ) -> Any:
        """Run cut control independently of blocked terminal capture writes."""
        return await asyncio.get_running_loop().run_in_executor(
            self._generation_cut_control_executor,
            operation,
            *args,
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
        if completed is None:
            unusable_reason = "its terminal result is no longer retained"
        elif completed.coords.disposition != "staged":
            unusable_reason = (
                f"its terminal write ended as {completed.coords.disposition!r}"
            )
        elif completed.effective_output_limit is None:
            unusable_reason = "vLLM did not resolve its effective output limit"
        elif completed.terminal_finish_reason is None:
            unusable_reason = "it finished without a stop or length finish reason"
        else:
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
        # The call regenerates from its previous boundary after a restore.
        LOGGER.warning(
            "cannot cut completed model call %s: %s",
            prefix.model_call_id,
            unusable_reason,
        )
        return GenerationCutPrefixAck.failure(prefix)

    def _frozen_cut_engine_routed_experts(
        self, state: _RequestCaptureState, *, deadline: float | None
    ) -> Any | None:
        """Read the engine's routes for a frozen cut that needs them.

        Runs after the freeze, without ``state.lock``, so decoding continues.

        Raises:
            TimeoutError: the engine did not answer before ``deadline``.
            ValueError: the engine cannot name the request's routes.
        """
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
        timeout_s = None if deadline is None else max(deadline - time.monotonic(), 0.0)
        return self._snapshot_engine_routed_experts(
            engine_request_id, timeout_s=timeout_s
        )

    def _checkpoint_active_generation_cut(
        self,
        prefix: Any,
        state: _RequestCaptureState,
        checkpoint_id: str,
        *,
        deadline: float | None = None,
    ) -> Any | None:
        """Cut one live call, or return ``None`` if abort won the race."""
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
        with state.lifecycle_lock:
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

            with state.lock:
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
            unusable_prefix = True
            if total_generation_token_count == 0:
                # Expected for a call still prefilling; nothing to keep yet.
                LOGGER.debug(
                    "not cutting model call %s: it has no generated tokens yet",
                    prefix.model_call_id,
                )
            elif generated_token_ids and _ends_inside_character(
                self._generation_cut_tokenizer,
                state.resumed_generation_token_ids + generated_token_ids,
            ):
                LOGGER.warning(
                    "cannot cut model call %s: its %d generated tokens end inside "
                    "a multi-byte character",
                    prefix.model_call_id,
                    total_generation_token_count,
                )
            else:
                unusable_prefix = False
            if unusable_prefix:
                with state.lock:
                    state.rollback_frozen_cut(checkpoint_id)
                return GenerationCutPrefixAck(
                    **prefix.model_dump(mode="json"),
                    disposition="durable_failure",
                )
            try:
                engine_routed_experts = self._frozen_cut_engine_routed_experts(
                    state, deadline=deadline
                )
                with state.lock:
                    (
                        chunk_routed_experts,
                        cumulative_routed_experts,
                    ) = state.frozen_cut_routed_experts(engine_routed_experts)
            except (TimeoutError, TypeError, ValueError) as error:
                # Only this call regenerates; the rest of the inventory is cut,
                # and the next checkpoint tries this call again.
                with state.lock:
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
                if cumulative_routed_experts is not None:
                    cumulative_extras = {
                        **(cumulative_extras or {}),
                        "routed_experts": encode_routed_experts(
                            cumulative_routed_experts
                        ),
                    }
                cumulative_record = capture.build_prefix_record(
                    state.call,
                    prompt_token_ids=state.prompt_token_ids,
                    generated_token_ids=generated_token_ids,
                    generated_logprobs=generated_logprobs,
                    extras=cumulative_extras,
                )
                if chunk_token_ids:
                    chunk_record = (
                        cumulative_record
                        if not state.generation_cut_staging_keys
                        else capture.build_generation_chunk_record(
                            state.call,
                            generated_token_ids=chunk_token_ids,
                            generated_logprobs=chunk_logprobs,
                            extras=(
                                None
                                if chunk_routed_experts is None
                                else {
                                    "routed_experts": encode_routed_experts(
                                        chunk_routed_experts
                                    )
                                }
                            ),
                        )
                    )
                    result = sink.stage_generation_prefix(
                        chunk_record,
                        checkpoint_id=checkpoint_id,
                        chunk_sequence=chunk_sequence,
                    )
                    staged_key = result.staging_key
                    if not result.ok:
                        raise RuntimeError(
                            "generation-prefix staging failed for "
                            f"{prefix.model_call_id!r}: {result.error}"
                        )
                    if staged_key is None:
                        raise RuntimeError(
                            "generation-prefix staging returned no staging key for "
                            f"{prefix.model_call_id!r}"
                        )
                with state.lock:
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
                    # Validate the complete acknowledgement before adopting the
                    # staged chunk. A failed acknowledgement must leave the
                    # frozen buffer available for rollback.
                    state.seal_frozen_cut(checkpoint_id)
                    if staged_key is not None:
                        state.generation_cut_staging_keys.append(staged_key)
            except Exception:
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

    def _checkpoint_generation_cut(
        self, inventory: Any, *, deadline: float | None = None
    ) -> Any:
        """Stage a stable prefix for every call named by Gym's frozen inventory.

        ``deadline`` is a ``time.monotonic()`` bound. Gym stops waiting for the
        receipt at its own control deadline, so calls not reached by then are
        acknowledged as failures (they regenerate after restore) instead of
        holding the single cut-control thread for the next checkpoint.
        """
        from nemo_gym._checkpoint.generation_cut import (
            GenerationCutPrefixAck,
            GenerationCutReceipt,
        )

        capture = self.token_capture
        sink = self._capture_sink
        if capture is None or sink is None:
            raise RuntimeError("generation-prefix cuts require token capture setup")
        receipt_key = (inventory.checkpoint_id, inventory.inventory_digest)
        with self._capture_registry_lock:
            cached_receipt = self._generation_cut_receipts.get(receipt_key)
        if cached_receipt is not None:
            return cached_receipt
        acknowledgements = []
        for index, prefix in enumerate(inventory.active_prefixes):
            if deadline is not None and time.monotonic() >= deadline:
                remaining = inventory.active_prefixes[index:]
                LOGGER.warning(
                    "generation cut passed its control deadline: checkpoint_id=%s "
                    "uncut=%d of %d model_call_ids=%s",
                    inventory.checkpoint_id,
                    len(remaining),
                    len(inventory.active_prefixes),
                    [uncut.model_call_id for uncut in remaining],
                )
                acknowledgements.extend(
                    GenerationCutPrefixAck.failure(uncut) for uncut in remaining
                )
                break
            with self._capture_registry_lock:
                state = self._capture_calls_by_model_call_id.get(prefix.model_call_id)
            try:
                acknowledgement = (
                    self._checkpoint_active_generation_cut(
                        prefix, state, inventory.checkpoint_id, deadline=deadline
                    )
                    if state is not None
                    else None
                )
                if acknowledgement is None:
                    acknowledgement = self._completed_generation_cut_ack(
                        prefix, inventory.checkpoint_id
                    )
            # Gym acknowledges cuts per call, so one call's failure (a TQ write,
            # an identity mismatch) regenerates that call alone after a restore
            # instead of failing the receipt for every call on this worker.
            except Exception:  # noqa: BLE001 - see above
                LOGGER.exception(
                    "cannot cut model call %s; it regenerates after a restore",
                    prefix.model_call_id,
                )
                if state is not None:
                    with state.lock:
                        frozen = state.frozen_cut
                        if frozen is not None and frozen[0] == inventory.checkpoint_id:
                            state.rollback_frozen_cut(inventory.checkpoint_id)
                acknowledgement = GenerationCutPrefixAck.failure(prefix)
            acknowledgements.append(acknowledgement)
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
