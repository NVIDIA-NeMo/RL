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
import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

LOGGER = logging.getLogger(__name__)

# Completion evidence only bridges the short race between a terminal response
# and Gym freezing its admitted-call inventory. Time-based retention avoids
# evicting fresh evidence merely because a large deployment completed more
# than 100k calls, while still bounding idle-job memory growth.
_COMPLETED_CAPTURE_RETENTION_S = 60.0 * 60.0


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
        self.observation_error = None

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


class _TokenCaptureSnapshotGate:
    """Drain active terminal writes and block new ones across a TQ snapshot."""

    def __init__(self) -> None:
        self._condition = threading.Condition()
        self._open = True
        self._active = 0

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

    def close_and_wait(self) -> None:
        with self._condition:
            self._open = False
            while self._active:
                self._condition.wait()

    def reopen(self) -> None:
        with self._condition:
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
        LOGGER.info(
            "generation prefix restored: rollout_id=%s model_call_id=%s "
            "source_model_call_id=%s prefix_tokens=%d prefix_digest=%s "
            "weight_version_span=[%d,%d]",
            admission.rollout_id,
            admission.model_call_id,
            continuation.source_model_call_id,
            continuation.generation_token_count,
            continuation.digest,
            weight_versions[0],
            weight_versions[-1],
        )
        return snapshot

    def _resolve_generation_cut(
        self, admission: Any, prefix_token_ids: list[int]
    ) -> Any | None:
        """Fetch and rebuild the cumulative staged prefix named by an admission."""
        continuation = admission.generation_cut
        if continuation is None:
            return None
        # Class-qualified so a host that borrows this method alone (tests bind the
        # worker's methods onto plain namespaces) needs nothing else.
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
        """Cut one live call, or return ``None`` if abort won the race."""
        from nemo_gym._checkpoint.generation_cut import (
            GenerationCutPrefixAck,
        )

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
                raise RuntimeError(
                    f"cannot cut model call {prefix.model_call_id!r}: "
                    f"{observation_error}"
                )
            inherited_generation_token_count = (
                state.call.admission.generation_cut.generation_token_count
                if state.call.admission.generation_cut is not None
                else 0
            )
            total_generation_token_count = inherited_generation_token_count + len(
                generated_token_ids
            )
            if total_generation_token_count == 0:
                with state.lock:
                    state.rollback_frozen_buffer(checkpoint_id)
                return GenerationCutPrefixAck(
                    **prefix.model_dump(mode="json"),
                    disposition="durable_failure",
                )
            if effective_output_limit is None:
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
                    state.seal_frozen_buffer(checkpoint_id)
                    if staged_key is not None:
                        state.generation_cut_staging_keys.append(staged_key)
            except Exception:
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

    def _checkpoint_generation_cut(self, inventory: Any) -> Any:
        """Stage a stable prefix for every call named by Gym's frozen inventory."""
        from nemo_gym._checkpoint.generation_cut import (
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
        for prefix in inventory.active_prefixes:
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
