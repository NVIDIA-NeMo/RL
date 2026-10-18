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
from dataclasses import dataclass, field
from typing import Any, Literal

LOGGER = logging.getLogger(__name__)

# Completion evidence only bridges the short race between a terminal response
# and Gym freezing its admitted-call inventory. Time-based retention avoids
# evicting fresh evidence merely because a large deployment completed more
# than 100k calls, while still bounding idle-job memory growth.
_COMPLETED_CAPTURE_RETENTION_S = 60.0 * 60.0


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
    # insertion-ordered dicts this mixin evicts from.
    _completed_capture_calls: dict[str, _CompletedCaptureState]
    _generation_cut_receipts: dict[tuple[str, str], Any]
    _generation_cut_tokenizer: Any
    _restored_generation_prefix_calls: int
    _restored_generation_prefix_tokens: int

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
            "weight_version_span=[%d,%d] prefix_ids_sha256=%s",
            admission.rollout_id,
            admission.model_call_id,
            continuation.source_model_call_id,
            continuation.prefix_token_count,
            continuation.prefix_digest,
            weight_versions[0],
            weight_versions[-1],
            prefix_ids_sha256,
        )
        return snapshot

    def _resolve_generation_cut(
        self, admission: Any, prefix_token_ids: list[int]
    ) -> Any | None:
        """Fetch and rebuild the cumulative staged prefix named by an admission."""
        continuation = admission.generation_cut
        if continuation is None:
            return None
        snapshots = self._fetch_generation_cut_chunks(continuation)
        return self._rebuild_generation_cut_snapshot(
            admission, prefix_token_ids, snapshots
        )

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
        try:
            if not state.terminal_started:
                state.terminal_started = True
                popped = self._pop_request_capture(request)
                if popped is state and self.token_capture is not None:
                    coords = self.token_capture.fail_call(state.call, reason=reason)
                    self._remember_completed_capture(
                        state.call.model_call_id, coords, 0
                    )
        finally:
            state.lifecycle_lock.release()
        return True

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
                if uncuttable_reason is None and observation_error is None:
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
            inherited_prefix_token_count = (
                state.call.admission.generation_cut.prefix_token_count
                if state.call.admission.generation_cut is not None
                else 0
            )
            total_generation_token_count = inherited_prefix_token_count + len(
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
                    state.rollback_frozen_cut(checkpoint_id)
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
                    "uncut=%d of %d",
                    inventory.checkpoint_id,
                    len(remaining),
                    len(inventory.active_prefixes),
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
