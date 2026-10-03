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
from collections.abc import Callable, Generator
from dataclasses import dataclass, field
from typing import Any, Literal

LOGGER = logging.getLogger(__name__)

# Completion evidence only bridges the short race between a terminal response
# and Gym freezing its admitted-call inventory. Time-based retention avoids
# evicting fresh evidence merely because a large deployment completed more
# than 100k calls, while still bounding idle-job memory growth.
_COMPLETED_CAPTURE_RETENTION_S = 60.0 * 60.0


@dataclass(frozen=True)
class GenerationPrefixBatchLimits:
    """Backend-neutral limits for one worker's prefix checkpoint writes."""

    max_rows: int
    max_tokens: int

    def __post_init__(self) -> None:
        if self.max_rows < 1 or self.max_tokens < 1:
            raise ValueError(
                "generation-prefix batch row and token limits must be positive"
            )


@dataclass
class _RequestCaptureBuffer:
    """One append-only generation segment owned by an in-flight request."""

    sequence: int
    generated_token_ids: list[int] = field(default_factory=list)
    generated_logprobs: list[float] = field(default_factory=list)


@dataclass
class _FrozenRequestCaptureBuffer:
    """An active buffer detached for a checkpoint write."""

    checkpoint_id: str
    buffer: _RequestCaptureBuffer


@dataclass
class _RequestCaptureState:
    """Append-only token buffers for one in-flight captured request.

    vLLM publishes delta progress. ``observe`` appends each delta to the current
    active buffer exactly once. A checkpoint swaps the active buffer under
    ``lock`` and performs its blocking TQ write after releasing the lock, so
    later observations can continue in a fresh buffer.

    Each successful cut stages only the detached buffer. The ordered TQ keys
    remain in Gym lineage, while the sealed token arrays are retained locally
    only to compute the cumulative digest advertised by the latest cut.
    """

    call: Any
    prompt_token_ids: list[int]
    # Backend-owned opaque state that must survive until the terminal row is
    # staged. vLLM uses this for its processed-media snapshot; prefix recovery
    # is currently setup-guarded to text-only, but ordinary terminal capture
    # still shares this lifecycle container.
    media: Any | None = None
    resumed_generation_token_ids: list[int] = field(default_factory=list)
    effective_output_limit: int | None = None
    terminal_finish_reason: Literal["stop", "length"] | None = None
    terminal_stop_reason: str | int | None = None
    observation_error: str | None = None
    lock: threading.Lock = field(default_factory=threading.Lock)
    # Serializes lifecycle-changing TQ operations for this call. Token
    # observation deliberately uses ``lock`` instead so decoding can continue
    # while a detached generation chunk is staged.
    lifecycle_lock: threading.Lock = field(default_factory=threading.Lock)
    terminal_started: bool = False
    sealed_generated_token_ids: list[int] = field(default_factory=list)
    sealed_generated_logprobs: list[float] = field(default_factory=list)
    generation_cut_staging_keys: list[str] = field(default_factory=list)
    active_buffer: _RequestCaptureBuffer = field(
        default_factory=lambda: _RequestCaptureBuffer(sequence=0)
    )
    frozen_buffer: _FrozenRequestCaptureBuffer | None = None
    observed_generation_token_count: int = 0
    next_buffer_sequence: int = 1

    def observe(
        self,
        generated_token_ids: list[int],
        generated_logprobs: list[float],
    ) -> None:
        """Append one immutable vLLM output delta."""
        if len(generated_token_ids) != len(generated_logprobs):
            self.observation_error = (
                "generated token IDs and log probabilities must have equal lengths"
            )
            return
        self.active_buffer.generated_token_ids.extend(generated_token_ids)
        self.active_buffer.generated_logprobs.extend(generated_logprobs)
        self.observed_generation_token_count += len(generated_token_ids)

    def freeze_for_checkpoint(
        self, checkpoint_id: str
    ) -> tuple[str, int, list[int], list[float], list[int], list[float]]:
        """Swap the active buffer and return its delta plus the stable prefix."""
        if self.frozen_buffer is not None:
            raise RuntimeError(
                "cannot start a generation cut while another cut is in progress"
            )
        buffer = self.active_buffer
        self.active_buffer = _RequestCaptureBuffer(sequence=self.next_buffer_sequence)
        self.next_buffer_sequence += 1
        self.frozen_buffer = _FrozenRequestCaptureBuffer(
            checkpoint_id=checkpoint_id,
            buffer=buffer,
        )
        return (
            f"active/{checkpoint_id}/{buffer.sequence}",
            buffer.sequence,
            list(buffer.generated_token_ids),
            list(buffer.generated_logprobs),
            self.sealed_generated_token_ids + buffer.generated_token_ids,
            self.sealed_generated_logprobs + buffer.generated_logprobs,
        )

    def seal_frozen_buffer(self, checkpoint_id: str) -> None:
        """Adopt a successfully staged frozen buffer into the live prefix."""
        frozen = self._require_frozen_buffer(checkpoint_id)
        self.sealed_generated_token_ids.extend(frozen.generated_token_ids)
        self.sealed_generated_logprobs.extend(frozen.generated_logprobs)
        self.frozen_buffer = None

    def rollback_frozen_buffer(self, checkpoint_id: str) -> None:
        """Restore a failed cut ahead of progress collected after its swap."""
        frozen = self._require_frozen_buffer(checkpoint_id)
        self.active_buffer.generated_token_ids[:0] = frozen.generated_token_ids
        self.active_buffer.generated_logprobs[:0] = frozen.generated_logprobs
        self.frozen_buffer = None

    def _require_frozen_buffer(self, checkpoint_id: str) -> _RequestCaptureBuffer:
        frozen = self.frozen_buffer
        if frozen is None or frozen.checkpoint_id != checkpoint_id:
            raise RuntimeError(
                f"generation cut {checkpoint_id!r} does not own the frozen buffer"
            )
        return frozen.buffer


@dataclass(frozen=True)
class _CompletedCaptureState:
    """Time-bounded terminal evidence retained across the response/cut race."""

    coords: Any
    generation_token_count: int
    effective_output_limit: int | None
    terminal_finish_reason: Literal["stop", "length"] | None
    terminal_stop_reason: str | int | None
    completed_at_monotonic: float


def _remaining_generation_limits_after_prefix(
    *,
    max_tokens: int | None,
    min_tokens: int | None,
    generation_token_count: int,
) -> tuple[int | None, int | None]:
    """Return output limits for the suffix after restoring generated tokens."""
    if generation_token_count < 0:
        raise ValueError("generation_token_count must be non-negative")
    return (
        None if max_tokens is None else max_tokens - generation_token_count,
        None if min_tokens is None else max(0, min_tokens - generation_token_count),
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
    # insertion-ordered dicts this mixin evicts from.
    _completed_capture_calls: dict[str, _CompletedCaptureState]
    _generation_cut_receipts: dict[tuple[str, str], Any]
    _generation_cut_tokenizer: Any
    _generation_prefix_batch_limits: GenerationPrefixBatchLimits | None = None

    def _configure_generation_prefix_batching(
        self, *, max_rows: int, max_tokens: int
    ) -> None:
        """Install validated per-worker limits for cut-time TQ writes."""
        self._generation_prefix_batch_limits = GenerationPrefixBatchLimits(
            max_rows=max_rows,
            max_tokens=max_tokens,
        )

    def _require_generation_prefix_batch_limits(
        self,
    ) -> GenerationPrefixBatchLimits:
        limits = self._generation_prefix_batch_limits
        if limits is None:
            raise RuntimeError(
                "generation-prefix batch limits require token capture setup"
            )
        return limits

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
        snapshots = self._staging_source.fetch(staging_keys)
        if len(snapshots) != len(staging_keys):
            raise RuntimeError("generation-cut fetch did not return every staged chunk")
        if not snapshots:
            raise RuntimeError("generation-cut continuation has no staged chunks")
        return snapshots

    def _rebuild_generation_cut_snapshot(
        self, admission: Any, prefix_token_ids: list[int], snapshots: list[Any]
    ) -> Any:
        """Rebuild and verify the cumulative staged prefix from its fetched chunks."""
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
        extras_digest = compute_extras_digest(None)
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
        generation_token_count = sum(mask == 1.0 for mask in token_mask_delta)
        if generation_token_count != continuation.generation_token_count:
            message = (
                "generation-cut token count mismatch: "
                f"expected={continuation.generation_token_count} "
                f"actual={generation_token_count} "
                f"source_capture_key={continuation.source_capture_key!r} "
                f"source_model_call_id={continuation.source_model_call_id!r}"
            )
            LOGGER.error("%s", message)
            raise RuntimeError(message)
        if digest != continuation.digest:
            parent_chain_hash_prefix = (
                admission.parent_chain_hash[:12]
                if admission.parent_chain_hash is not None
                else None
            )
            message = (
                "generation-cut digest mismatch: "
                f"expected={continuation.digest} actual={digest} "
                f"source_capture_key={continuation.source_capture_key!r} "
                f"source_model_call_id={continuation.source_model_call_id!r} "
                f"parent_call_id={admission.parent_call_id!r} "
                f"mode={admission.mode!r} prev_len={admission.prev_len} "
                f"delta_len={delta_len} cum_len={cum_len} "
                f"weight_version={weight_version} "
                f"schema_version={admission.schema_version} "
                f"parent_chain_hash_prefix={parent_chain_hash_prefix!r} "
                f"chain_hash_prefix={chain_hash[:12]!r} "
                f"cumulative_hash_prefix={cumulative_hash[:12]!r}"
            )
            LOGGER.error("%s", message)
            raise RuntimeError(message)
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
            "weight_version_span=[%d,%d] prefix_ids_sha256=%s",
            admission.rollout_id,
            admission.model_call_id,
            continuation.source_model_call_id,
            continuation.generation_token_count,
            continuation.digest,
            weight_versions[0],
            weight_versions[-1],
            prefix_ids_sha256,
        )
        return snapshot

    def _resolve_generation_cut(
        self,
        admission: Any,
        prefix_token_ids: list[int],
        snapshots: list[Any] | None = None,
    ) -> Any | None:
        """Fetch and rebuild the cumulative staged prefix named by an admission."""
        continuation = admission.generation_cut
        if continuation is None:
            return None
        if snapshots is None:
            # Class-qualified so a host that borrows this method alone (tests bind
            # the worker's methods onto plain namespaces) needs nothing else.
            snapshots = GenerationCutCaptureMixin._fetch_generation_cut_chunks(
                self, continuation
            )
        return GenerationCutCaptureMixin._rebuild_generation_cut_snapshot(
            self, admission, prefix_token_ids, snapshots
        )

    def _abort_request_capture(self, request: Any, *, reason: str) -> None:
        """Drop the in-flight capture state for a request that errored."""
        state = self._get_request_capture(request)
        if state is None:
            return
        with state.lifecycle_lock:
            if state.terminal_started:
                return
            state.terminal_started = True
            popped = self._pop_request_capture(request)
            if popped is not state or self.token_capture is None:
                return
            coords = self.token_capture.fail_call(state.call, reason=reason)
            self._remember_completed_capture(state.call.model_call_id, coords, 0)

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
        from nemo_gym.token_id_capture.staging.records import staging_key

        with self._capture_registry_lock:
            completed = self._completed_capture_calls.get(prefix.model_call_id)
        expected_capture_key = (
            prefix.rollout_id
            if prefix.attempt_index == 0
            else f"{prefix.rollout_id}-a{prefix.attempt_index}"
        )
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
        # The in-memory cache is only an optimization. Its durable canonical
        # row remains authoritative if the race evidence aged out before a
        # delayed checkpoint inventory arrived.
        if completed is None and self._staging_source is not None:
            canonical_key = staging_key(expected_capture_key, prefix.model_call_id)
            try:
                snapshots = self._staging_source.fetch([canonical_key])
            except KeyError:
                snapshots = []
            if snapshots:
                if len(snapshots) != 1:
                    raise RuntimeError(
                        "terminal completion lookup returned an unexpected row count"
                    )
                snapshot = snapshots[0]
                if (
                    snapshot.rollout_id != expected_capture_key
                    or snapshot.model_call_id != prefix.model_call_id
                ):
                    raise RuntimeError(
                        "generation-prefix inventory identity does not match the "
                        f"durable terminal row: model_call_id={prefix.model_call_id!r}"
                    )
                # The canonical token row proves completion, but it predates
                # the recovery contract's effective budget and finish reason.
                # Without those fields we cannot reproduce the same terminal
                # API result, so restart this attempt from its prior boundary.
                LOGGER.warning(
                    "terminal completion row lacks recovery metadata: model_call_id=%s",
                    prefix.model_call_id,
                )
        return GenerationCutPrefixAck(
            **prefix.model_dump(mode="json"),
            disposition="durable_failure",
        )

    def _checkpoint_active_generation_cut(
        self,
        prefix: Any,
        state: _RequestCaptureState,
        checkpoint_id: str,
    ) -> Any | None:
        """Single-row compatibility path using the freeze/seal transaction."""
        transaction = self._generation_cut_transaction(prefix, state, checkpoint_id)
        try:
            try:
                record, sequence = next(transaction)
            except StopIteration as finished:
                return finished.value
            result = self._capture_sink.stage_generation_prefix(
                record,
                checkpoint_id=checkpoint_id,
                chunk_sequence=sequence,
            )
            acknowledgement = transaction.send(result)
            try:
                transaction.send(None)
            except StopIteration:
                return acknowledgement
            raise RuntimeError("generation-cut transaction did not seal")
        finally:
            transaction.close()

    def _generation_cut_transaction(
        self,
        prefix: Any,
        state: _RequestCaptureState,
        checkpoint_id: str,
    ) -> Generator[Any, Any, Any]:
        """Keep a live call frozen until transport and acknowledgement validate.

        The generator yields ``(record, sequence)``, receives a ``StageResult``,
        then yields its validated acknowledgement. The caller resumes it once
        more to seal. Closing before that point rolls the detached buffer back.
        """
        from nemo_gym._checkpoint.generation_cut import GenerationCutPrefixAck

        capture = self.token_capture
        sink = self._capture_sink
        if capture is None or sink is None:
            raise RuntimeError("generation-prefix cuts require token capture setup")

        with state.lifecycle_lock:
            with self._capture_registry_lock:
                current = self._capture_calls_by_model_call_id.get(prefix.model_call_id)
            if current is not state or state.terminal_started:
                return None

            expected_capture_key = (
                prefix.rollout_id
                if prefix.attempt_index == 0
                else f"{prefix.rollout_id}-a{prefix.attempt_index}"
            )
            if state.call.rollout_id != expected_capture_key:
                raise RuntimeError(
                    "generation-prefix inventory identity does not match the "
                    f"active call: model_call_id={prefix.model_call_id!r}, "
                    f"inventory_rollout_id={prefix.rollout_id!r}, "
                    f"worker_rollout_id={state.call.rollout_id!r}"
                )

            with state.lock:
                observation_error = state.observation_error
                effective_output_limit = state.effective_output_limit
                terminal_finish_reason = state.terminal_finish_reason
                terminal_stop_reason = state.terminal_stop_reason
                if observation_error is None:
                    (
                        frozen_buffer_id,
                        chunk_sequence,
                        chunk_token_ids,
                        chunk_logprobs,
                        generated_token_ids,
                        generated_logprobs,
                    ) = state.freeze_for_checkpoint(checkpoint_id)
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

            inherited_generation_token_count = (
                state.call.admission.generation_cut.generation_token_count
                if state.call.admission.generation_cut is not None
                else 0
            )
            total_generation_token_count = inherited_generation_token_count + len(
                generated_token_ids
            )
            if total_generation_token_count == 0 or (
                generated_token_ids
                and _ends_inside_character(
                    self._generation_cut_tokenizer,
                    state.resumed_generation_token_ids + generated_token_ids,
                )
            ):
                with state.lock:
                    state.rollback_frozen_buffer(checkpoint_id)
                return GenerationCutPrefixAck(
                    **prefix.model_dump(mode="json"),
                    disposition="durable_failure",
                )
            if effective_output_limit is None:
                with state.lock:
                    state.rollback_frozen_buffer(checkpoint_id)
                raise RuntimeError(
                    f"cannot cut model call {prefix.model_call_id!r}: "
                    "vLLM did not resolve its effective output limit"
                )

            staged_key = None
            try:
                cumulative_record = capture.build_prefix_record(
                    state.call,
                    prompt_token_ids=state.prompt_token_ids,
                    generated_token_ids=generated_token_ids,
                    generated_logprobs=generated_logprobs,
                )
                if chunk_token_ids:
                    chunk_record = (
                        cumulative_record
                        if not state.generation_cut_staging_keys
                        else capture.build_generation_chunk_record(
                            state.call,
                            generated_token_ids=chunk_token_ids,
                            generated_logprobs=chunk_logprobs,
                        )
                    )
                    result = yield chunk_record, chunk_sequence
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
                if chunk_token_ids:
                    # Validate every acknowledgement in a batch before sealing
                    # any detached buffer into its live request state.
                    yield acknowledgement
                with state.lock:
                    state.seal_frozen_buffer(checkpoint_id)
                    if staged_key is not None:
                        state.generation_cut_staging_keys.append(staged_key)
            except BaseException:
                # Generator.close() injects GeneratorExit when another row in
                # the batch fails. Roll back without hiding the root failure.
                with state.lock:
                    frozen = state.frozen_buffer
                    if frozen is not None and frozen.checkpoint_id == checkpoint_id:
                        state.rollback_frozen_buffer(checkpoint_id)
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
        as failures, matching the unbatched path's deadline contract.
        """
        from nemo_gym._checkpoint.generation_cut import GenerationCutPrefixAck

        from nemo_rl.data_plane.tq_token_sink import generation_cut_staging_key

        batch_limits = self._require_generation_prefix_batch_limits()
        call_ids = [prefix.model_call_id for prefix in inventory.active_prefixes]
        if len(set(call_ids)) != len(call_ids):
            raise ValueError(
                "generation-cut inventory contains duplicate model-call identities"
            )

        sink = self._capture_sink
        acknowledgements: list[Any] = []
        pending: list[tuple[Generator[Any, Any, Any], Any, int, Any]] = []
        pending_tokens = 0
        uncut: list[Any] = []

        def past_deadline() -> bool:
            return deadline is not None and time.monotonic() >= deadline

        def release_pending() -> None:
            nonlocal pending_tokens
            for transaction, _, _, prefix in pending:
                transaction.close()
                uncut.append(prefix)
            pending.clear()
            pending_tokens = 0

        def flush() -> None:
            nonlocal pending_tokens
            if not pending:
                return
            if past_deadline():
                release_pending()
                return
            keys = [
                generation_cut_staging_key(
                    inventory.checkpoint_id,
                    record.rollout_id,
                    record.model_call_id,
                    chunk_sequence=sequence,
                )
                for _, record, sequence, _ in pending
            ]
            sealed_keys: set[str] = set()
            try:
                results = sink.stage_generation_prefix_batch(
                    [record for _, record, _, _ in pending],
                    checkpoint_id=inventory.checkpoint_id,
                    chunk_sequences=[sequence for _, _, sequence, _ in pending],
                )
                if len(results) != len(pending):
                    raise RuntimeError(
                        "generation-prefix batch returned an incorrect result count"
                    )
                for result, key in zip(results, keys, strict=True):
                    if not result.ok or result.staging_key != key:
                        raise RuntimeError(
                            "generation-prefix batch staging failed: "
                            f"{result.error}; key={key}"
                        )
                batch_acks = [
                    transaction.send(result)
                    for (transaction, _, _, _), result in zip(
                        pending, results, strict=True
                    )
                ]
                for (transaction, _, _, _), key in zip(pending, keys, strict=True):
                    try:
                        transaction.send(None)
                    except StopIteration:
                        sealed_keys.add(key)
                    else:
                        raise RuntimeError("generation-cut transaction did not seal")
                acknowledgements.extend(batch_acks)
            except BaseException:
                # A TQ call can fail after accepting some rows. None of those
                # keys may be advertised unless its transaction sealed.
                rejected = [key for key in keys if key not in sealed_keys]
                if rejected:
                    try:
                        sink.clear(rejected)
                    except Exception:  # noqa: BLE001 — preserve the batch failure
                        LOGGER.exception(
                            "Failed to clear rejected generation-prefix batch"
                        )
                raise
            finally:
                for transaction, _, _, _ in pending:
                    transaction.close()
                pending.clear()
                pending_tokens = 0

        try:
            for index, prefix in enumerate(inventory.active_prefixes):
                if past_deadline():
                    release_pending()
                    uncut.extend(inventory.active_prefixes[index:])
                    break
                with self._capture_registry_lock:
                    state = self._capture_calls_by_model_call_id.get(
                        prefix.model_call_id
                    )
                if state is None:
                    acknowledgements.append(
                        self._completed_generation_cut_ack(
                            prefix, inventory.checkpoint_id
                        )
                    )
                    continue
                transaction = self._generation_cut_transaction(
                    prefix, state, inventory.checkpoint_id
                )
                try:
                    try:
                        record, sequence = next(transaction)
                    except StopIteration as finished:
                        acknowledgement = finished.value
                        if acknowledgement is None:
                            acknowledgement = self._completed_generation_cut_ack(
                                prefix, inventory.checkpoint_id
                            )
                        acknowledgements.append(acknowledgement)
                        continue
                    tokens = len(record.token_ids_delta)
                    if pending and pending_tokens + tokens > batch_limits.max_tokens:
                        flush()
                    pending.append((transaction, record, sequence, prefix))
                except BaseException:
                    transaction.close()
                    raise
                pending_tokens += tokens
                if (
                    len(pending) >= batch_limits.max_rows
                    or pending_tokens >= batch_limits.max_tokens
                ):
                    # An oversized single row is written alone, never split.
                    flush()
            flush()
        finally:
            for transaction, _, _, _ in pending:
                transaction.close()
        if uncut:
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
        single_prefixes = inventory.active_prefixes
        batch_limits = self._require_generation_prefix_batch_limits()
        if batch_limits.max_rows > 1:
            acknowledgements = self._checkpoint_generation_cut_batched(
                inventory, deadline=deadline
            )
            single_prefixes = ()
        for index, prefix in enumerate(single_prefixes):
            if deadline is not None and time.monotonic() >= deadline:
                remaining = single_prefixes[index:]
                LOGGER.warning(
                    "generation cut passed its control deadline: checkpoint_id=%s "
                    "uncut=%d of %d",
                    inventory.checkpoint_id,
                    len(remaining),
                    len(single_prefixes),
                )
                acknowledgements.extend(
                    GenerationCutPrefixAck.failure(uncut) for uncut in remaining
                )
                break
            with self._capture_registry_lock:
                state = self._capture_calls_by_model_call_id.get(prefix.model_call_id)
            if state is None:
                acknowledgements.append(
                    self._completed_generation_cut_ack(prefix, inventory.checkpoint_id)
                )
                continue
            acknowledgement = self._checkpoint_active_generation_cut(
                prefix, state, inventory.checkpoint_id
            )
            if acknowledgement is None:
                acknowledgement = self._completed_generation_cut_ack(
                    prefix, inventory.checkpoint_id
                )
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
