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

"""Cut live model calls at a checkpoint and resume them after a restore.

The worker builds one ``PrefixRecovery`` in ``setup_token_capture`` when
``target_level`` is ``prefix``, or a ``NoPrefixRecovery`` otherwise, and calls
it at fixed points: when a call begins and gets its sampling settings, when its
final output arrives and its terminal row is written, at every checkpoint cut
and snapshot flush. Only the ``ProgressReader`` it is given knows the engine.
"""

from __future__ import annotations

import asyncio
import functools
import hashlib
import json
import logging
import threading
import time
from collections import Counter
from collections.abc import Generator, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from typing import Any, Literal

import torch

from nemo_rl.data_plane.prefix_pipeline import (
    BatchReleased,
    CutSteps,
    PrefixWrite,
    PrefixWritePipeline,
    PrefixWritePipelineConfig,
)
from nemo_rl.data_plane.schema import ROUTED_EXPERTS_FIELD
from nemo_rl.models.generation.prefix_read_batcher import PrefixReadBatcher
from nemo_rl.models.generation.prefix_recovery.parts import PrefixRecoveryParts
from nemo_rl.models.generation.prefix_recovery.restore import (
    _generation_cut_blockers,
    restore_reply_text,
)
from nemo_rl.models.generation.prefix_recovery.routes import (
    _cut_chunk_routes,
    _join_chunk_routes,
    _prefix_routes_sha256,
)
from nemo_rl.models.generation.prefix_recovery.state import (
    _CompletedCaptureState,
    _RequestCaptureState,
)
from nemo_rl.utils.routed_experts_codec import routed_experts_tensor_metadata

LOGGER = logging.getLogger(__name__)

# Completion evidence only bridges the short race between a terminal response
# and Gym freezing its admitted-call inventory. Time-based retention avoids
# evicting fresh evidence merely because a large deployment completed more
# than 100k calls, while still bounding idle-job memory growth.
_COMPLETED_CAPTURE_RETENTION_S = 60.0 * 60.0

# Sampling settings already warned about, so each warns once per process.
_LOGGED_UNCUTTABLE_REASONS: set[str] = set()

# One character spans at most four UTF-8 bytes, hence at most four byte-level
# tokens; decoding a slightly longer tail is enough to see whether it is whole.
_PREFIX_BOUNDARY_TAIL_TOKENS = 8


def initial_generation_cut_counts() -> dict[str, int]:
    """Zeroed progress-read counters, so each is reported from the first step.

    How many reads of the engine's decoding progress, for how many calls, and
    the total microseconds spent waiting for and holding its output lock.
    """
    return {
        "progress_reads": 0,
        "progress_read_calls": 0,
        "progress_read_wait_us": 0,
        "progress_read_hold_us": 0,
    }


def _require_inventory_identity(
    prefix: Any, worker_capture_key: str, *, call_kind: str
) -> None:
    """Refuse a cut whose inventory entry names another episode than the call."""
    # Deferred: nemo_gym is an optional extra absent in non-Gym runs.
    from nemo_gym.episode_types import EpisodeId

    expected = EpisodeId(rollout_id=prefix.rollout_id, attempt=prefix.attempt)
    if worker_capture_key != expected.capture_key:
        raise RuntimeError(
            "generation-prefix inventory identity does not match the "
            f"{call_kind} call: model_call_id={prefix.model_call_id!r}, "
            f"inventory_rollout_id={prefix.rollout_id!r}, "
            f"worker_capture_key={worker_capture_key!r}"
        )


@dataclass(frozen=True)
class _ResolvedGenerationCut:
    """A verified restored prefix, with its call's media and router-replay routes."""

    snapshot: Any
    media: Any | None = None
    routed_experts: torch.Tensor | None = None


def _ends_inside_character(tokenizer: Any, generated_token_ids: list[int]) -> bool:
    """Whether the generated text stops partway through a multi-byte character.

    The engine detokenizes a resumed call's tail with a stream primed on the
    prompt, which ends with the restored prefix. A prefix ending inside a
    character makes that stream emit the whole decoded prompt as the tail's
    first text.
    """
    tail = generated_token_ids[-_PREFIX_BOUNDARY_TAIL_TOKENS:]
    return tokenizer.decode(tail, skip_special_tokens=False).endswith("\ufffd")


class PrefixRecovery:
    """Generation-prefix cuts and restores for one worker's captured calls."""

    def __init__(
        self,
        parts: PrefixRecoveryParts,
        *,
        pipeline_config: PrefixWritePipelineConfig | None = None,
    ) -> None:
        self._parts = parts
        self._registry = parts.registry
        # Guards this object's own state below; never held across TQ I/O.
        self._lock = threading.Lock()
        self._completed: dict[str, _CompletedCaptureState] = {}
        self._receipts: dict[tuple[str, str], Any] = {}
        # Calls this process resumed from a restored generation prefix, and
        # their prefix tokens; the driver logs the sums each step.
        self._restored_calls = 0
        self._restored_tokens = 0
        self._counts = initial_generation_cut_counts()
        # Chunk rows of ended calls, deleted when the next snapshot starts, and
        # the latest snapshot that deleted them.
        self._obsolete_keys: list[str] = []
        self._flushed_epoch = 0
        # A dedicated thread keeps completions' TQ writes from consuming the
        # threads needed to create the checkpoint cut.
        self._control_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="nrl-generation-cut-control"
        )
        # Obsolete chunk rows are deleted here, off the serving loop.
        self._cleanup_executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="nrl-generation-cut-cleanup"
        )
        # Restores arriving together share their chunk reads.
        self._reader = PrefixReadBatcher(
            self._fetch_rows, max_rows=parts.generation_prefix_batch_size
        )
        self._pipeline_config = pipeline_config
        self._pipeline = (
            PrefixWritePipeline(
                config=pipeline_config,
                sink=parts.sink,
                prepare=lambda inventory, deadline: self._checkpoint_steps(
                    inventory, deadline=deadline
                ),
            )
            if pipeline_config is not None and pipeline_config.enabled
            else None
        )

    @property
    def control_token(self) -> str | None:
        """The bearer token the cut route requires; None refuses every cut."""
        return self._parts.control_token

    # ---- A call's life --------------------------------------------------

    def on_begin(self, admission: Any) -> None:
        """Count a call Gym's ``begin_call`` accepted with a restored cut."""
        if admission.generation_cut is None:
            return
        with self._lock:
            self._restored_calls += 1
            self._restored_tokens += admission.generation_cut.prefix_token_count

    def on_sampling_params(self, request: Any, sampling_params: Any) -> None:
        """Record the engine's resolved output budget, and whether a cut can resume."""
        state = self._registry.get(request)
        if state is None:
            return
        effective_output_limit = int(sampling_params.max_tokens)
        if effective_output_limit <= 0:
            raise ValueError("effective output limit must be positive")
        state.set_effective_output_limit(effective_output_limit)
        # Such a call regenerates from its previous boundary after a restore,
        # which costs time but not correctness; see _generation_cut_blockers.
        blockers = _generation_cut_blockers(sampling_params, self._parts.tokenizer)
        if not blockers:
            return
        reason = f"{', '.join(blockers)} do not carry across a generation cut"
        state.mark_uncuttable(reason)
        if reason not in _LOGGED_UNCUTTABLE_REASONS:
            _LOGGED_UNCUTTABLE_REASONS.add(reason)
            LOGGER.warning(
                "model calls using %s are not cut; they regenerate after a restore",
                ", ".join(blockers),
            )

    def observe(self, request: Any, request_output: Any) -> None:
        """Record the tokens a finished output adds past the last cut."""
        state = self._registry.get(request)
        if state is None:
            return
        outputs = getattr(request_output, "outputs", None)
        if not outputs:
            return
        observed = state.recorded_token_count
        try:
            output = outputs[0]
            # Finished, so the engine appends to these lists no more.
            generation_token_ids = list(getattr(output, "token_ids", None) or ())[
                observed:
            ]
            generation_logprobs = self._parts.progress_reader.logprobs(
                generation_token_ids, getattr(output, "logprobs", None), start=observed
            )
            finish_reason = getattr(output, "finish_reason", None)
            stop_reason = getattr(output, "stop_reason", None)
            terminal_routes = None
            if state.routed_experts_dtype is not None and finish_reason is not None:
                route_parts = [
                    value
                    for value in (
                        getattr(request_output, "prompt_routed_experts", None),
                        getattr(output, "routed_experts", None),
                    )
                    if value is not None
                ]
                if route_parts:
                    terminal_routes = torch.cat(
                        [torch.as_tensor(value) for value in route_parts], dim=0
                    )
        except (RuntimeError, TypeError, ValueError) as error:
            state.record_error(f"{type(error).__name__}: {error}")
            return
        if terminal_routes is not None:
            state.record_terminal_routes(terminal_routes)
        state.record(
            generation_token_ids,
            generation_logprobs,
            start=observed,
            finish_reason=finish_reason,
            stop_reason=stop_reason,
        )

    def restore_reply_text(
        self, request: Any, request_output: Any, *, tokenizer: Any
    ) -> None:
        """Prepend a resumed call's restored text to its final output."""
        state = self._registry.get(request)
        if state is None or not state.resumed_generation_token_ids:
            return
        restore_reply_text(
            request,
            request_output,
            prefix_token_ids=state.resumed_generation_token_ids,
            engine_prompt_token_ids=state.prompt_token_ids,
            tokenizer=tokenizer,
        )

    def retire(self, staging_keys: Sequence[str]) -> None:
        """Queue an ended call's chunk rows for the next snapshot to delete.

        A snapshot in progress may still name them, so they wait until the
        next one starts, after this one is saved.
        """
        if not staging_keys:
            return
        with self._lock:
            self._obsolete_keys.extend(staging_keys)

    async def flush(self, epoch: int) -> None:
        """Delete the chunk rows retired before snapshot ``epoch`` started.

        Returns once the delete is done: TQ must not clear rows while the
        snapshot is saved.
        """
        await asyncio.get_running_loop().run_in_executor(
            self._cleanup_executor, self._flush_retired, epoch
        )

    def _flush_retired(self, epoch: int) -> None:
        # They belong to calls that ended before Gym froze this snapshot's
        # inventory, which names a finished call by its terminal row only, and
        # a saved snapshot keeps its own copy of TransferQueue.
        with self._lock:
            # A late flush from an older snapshot must not delete rows retired
            # since a newer one started; that snapshot may still name them.
            if epoch <= self._flushed_epoch:
                return
            self._flushed_epoch = epoch
            staging_keys = self._obsolete_keys
            self._obsolete_keys = []
        if not staging_keys:
            return
        try:
            self._parts.sink.clear(staging_keys)
        except Exception:  # noqa: BLE001 - the calls have already ended
            # A failed clear only leaves rows the next restore clears.
            LOGGER.exception(
                "failed to clear %d obsolete generation chunk rows",
                len(staging_keys),
            )

    def counts(self) -> dict[str, int]:
        """Restore and progress-read counts since this process started.

        Returns:
            ``restored_calls`` and their total ``restored_prefix_tokens``, plus
            ``progress_reads``, ``progress_read_calls`` and the microseconds
            they waited for and held the engine's output lock.
        """
        with self._lock:
            return {
                "restored_calls": self._restored_calls,
                "restored_prefix_tokens": self._restored_tokens,
                **self._counts,
            }

    def shutdown(self) -> None:
        if self._pipeline is not None:
            self._pipeline.close(wait=False)
        # Waiting for a wedged TQ write here would make actor shutdown hang.
        for executor in (self._control_executor, self._cleanup_executor):
            executor.shutdown(wait=False, cancel_futures=True)

    # ---- A checkpoint cut -----------------------------------------------

    async def run_cut(self, inventory: Any) -> Any:
        """Cut every call Gym's inventory names, off the serving loop."""
        # Measured from arrival so time queued behind a stale cut counts.
        timeout_s = self._parts.control_timeout_s
        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        if self._pipeline is not None:
            future = self._pipeline.submit(inventory, deadline=deadline)
            return await asyncio.shield(asyncio.wrap_future(future))
        return await asyncio.get_running_loop().run_in_executor(
            self._control_executor,
            functools.partial(self.checkpoint_cut, inventory, deadline=deadline),
        )

    def _record_progress(
        self, states: list[_RequestCaptureState], *, deadline: float | None
    ) -> dict[str, list[Any]]:
        """Record the tokens each call decoded since its last record.

        One read at an engine output boundary copies every call's new token
        IDs, so all calls are read at the same point; their logprobs are then
        read outside it, from positions that read saw published. A call the
        engine has finished but whose consumer has not yet taken the output is
        read whole, with its finish reason. One the engine does not hold (not
        yet submitted, or aborted) keeps what it has recorded.

        Returns:
            The route pieces the same read copied, by model call ID, for each
            router-replay call whose final output has not already brought its
            routes; a cut stages its routes from these.
        """
        pending = [
            (state, state.call.model_call_id, state.recorded_token_count)
            for state in states
        ]
        if not pending:
            return {}
        route_calls = frozenset(
            model_call_id
            for state, model_call_id, _ in pending
            if state.routed_experts_dtype is not None
            and state.snapshot().terminal_routed_experts is None
        )
        reader = self._parts.progress_reader
        try:
            read = reader.read(
                [(model_call_id, start) for _, model_call_id, start in pending],
                timeout_s=None if deadline is None else deadline - time.monotonic(),
                route_calls=route_calls,
            )
        except TimeoutError:
            LOGGER.warning(
                "generation cut could not read the engine's progress before its "
                "deadline; %d calls are cut at the tokens they have recorded",
                len(pending),
            )
            return {}
        with self._lock:
            self._counts["progress_reads"] += 1
            self._counts["progress_read_calls"] += len(pending)
            self._counts["progress_read_wait_us"] += round(read.wait_seconds * 1e6)
            self._counts["progress_read_hold_us"] += round(read.hold_seconds * 1e6)
        LOGGER.debug(
            "generation cut read %d calls: waited %.2f ms, held %.2f ms",
            len(pending),
            read.wait_seconds * 1e3,
            read.hold_seconds * 1e3,
        )
        route_pieces = {}
        for state, model_call_id, start in pending:
            progress = read.progress.get(model_call_id)
            if progress is None:
                continue
            if progress.route_pieces is not None:
                route_pieces[model_call_id] = progress.route_pieces
            try:
                generated_logprobs = reader.logprobs(
                    progress.token_ids, progress.logprobs, start=start
                )
            except (RuntimeError, TypeError, ValueError) as error:
                state.record_error(f"{type(error).__name__}: {error}")
                continue
            state.record(
                progress.token_ids,
                generated_logprobs,
                start=start,
                finish_reason=progress.finish_reason,
                stop_reason=progress.stop_reason,
            )
        return route_pieces

    @staticmethod
    def _failed_ack(prefix: Any, *, reason: str, uncut: Counter[str]) -> Any:
        """Acknowledge one uncut call and tally it under ``reason``."""
        from nemo_gym._checkpoint.generation_cut import GenerationCutPrefixAck

        uncut[reason] += 1
        return GenerationCutPrefixAck.failure(prefix)

    def _completed_ack(
        self, prefix: Any, checkpoint_id: str, uncut: Counter[str]
    ) -> Any:
        """Describe a call whose terminal path won the cut/abort race."""
        from nemo_gym._checkpoint.generation_cut import (
            GenerationCutPrefixAck,
        )

        with self._lock:
            completed = self._completed.get(prefix.model_call_id)
        if completed is not None:
            _require_inventory_identity(
                prefix, completed.coords.capture_key, call_kind="completed"
            )
        if completed is None:
            # Gym can name a call before its request reaches this worker, so
            # absence does not mean the call's finished result aged out.
            reason = "no_capture_or_completed_state"
            unusable_reason = "this worker has no record of it"
        elif completed.coords.disposition != "staged":
            reason = "completed_not_staged"
            unusable_reason = (
                f"its terminal write ended as {completed.coords.disposition!r}"
            )
        elif completed.effective_output_limit is None:
            reason = "completed_missing_output_limit"
            unusable_reason = "vLLM did not resolve its effective output limit"
        elif completed.terminal_finish_reason is None:
            reason = "completed_missing_finish_reason"
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
        return self._failed_ack(prefix, reason=reason, uncut=uncut)

    def _cut_transaction(
        self,
        prefix: Any,
        state: _RequestCaptureState,
        checkpoint_id: str,
        uncut: Counter[str],
        *,
        route_pieces: list[Any] | None = None,
    ) -> Generator[Any, Any, Any]:
        """Hold one live call's cut frozen until its row is written and acknowledged.

        Yields ``(record, chunk_sequence, attachments)``, receives the staged
        key, then yields the call's acknowledgement; resuming it once more
        seals the cut. A call with nothing new to stage returns its
        acknowledgement (or ``None`` if abort won the race) without yielding.
        Closing it before the seal rolls the cut back and clears its row.

        ``route_pieces`` are the call's router-replay route pieces, copied with
        its tokens by the cut's progress read.
        """
        from nemo_gym._checkpoint.generation_cut import (
            GenerationCutPrefixAck,
        )

        capture = self._parts.capture
        sink = self._parts.sink

        # Abort and terminal completion own the same lifecycle. Once the cut
        # owns it, the call cannot be removed until its frozen tokens are
        # staged or rolled back (leaving the block rolls back an unsealed cut).
        # If abort won first, use its terminal evidence.
        with state.lifecycle() as owned:
            current = self._registry.get_by_model_call_id(prefix.model_call_id)
            if current is not state or owned.terminal_started:
                return None

            _require_inventory_identity(
                prefix, state.call.capture_key, call_kind="active"
            )

            progress, frozen = owned.freeze(checkpoint_id)
            uncuttable_reason = progress.uncuttable_reason
            observation_error = progress.observation_error
            effective_output_limit = progress.effective_output_limit
            if uncuttable_reason is not None:
                # Expected for this request; the worker warned when it was set.
                LOGGER.debug(
                    "not cutting model call %s: %s",
                    prefix.model_call_id,
                    uncuttable_reason,
                )
                return self._failed_ack(
                    prefix, reason="uncuttable_request", uncut=uncut
                )
            if observation_error is not None:
                # Its recorded tokens or logprobs are inconsistent with vLLM's.
                # Only this call regenerates; the rest of the inventory is cut.
                LOGGER.warning(
                    "cannot cut model call %s: %s",
                    prefix.model_call_id,
                    observation_error,
                )
                return self._failed_ack(prefix, reason="observation_error", uncut=uncut)
            if effective_output_limit is None:
                # Checked before freezing, so the call's buffer is untouched.
                LOGGER.warning(
                    "cannot cut model call %s: vLLM did not resolve its "
                    "effective output limit",
                    prefix.model_call_id,
                )
                return self._failed_ack(
                    prefix, reason="missing_output_limit", uncut=uncut
                )
            assert frozen is not None  # freeze refuses only the cases above
            generated_token_ids = frozen.generated_token_ids
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
                self._parts.tokenizer,
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
                owned.rollback(checkpoint_id)
                return self._failed_ack(
                    prefix,
                    reason="zero_generated_tokens"
                    if total_generation_token_count == 0
                    else "incomplete_utf8_boundary",
                    uncut=uncut,
                )
            try:
                chunk_routes = self._chunk_routes(
                    owned, state, progress, frozen, route_pieces
                )
            except (TypeError, ValueError) as error:
                # Only this call regenerates; the rest of the inventory is cut,
                # and the next checkpoint tries this call again.
                owned.rollback(checkpoint_id)
                LOGGER.warning(
                    "cannot cut model call %s: routed experts: %s: %s",
                    prefix.model_call_id,
                    type(error).__name__,
                    error,
                )
                return self._failed_ack(
                    prefix, reason="routed_experts_unavailable", uncut=uncut
                )
            staged_key = None
            try:
                # The advertised prefix digest covers tokens and call extras,
                # not routes: each chunk's own digest covers its routes, and a
                # restore checks every chunk's routes against it.
                call_extras = self._parts.prefix_row_extras(state)
                cumulative_record = capture.build_prefix_record(
                    state.call,
                    prompt_token_ids=state.prompt_token_ids,
                    generated_token_ids=generated_token_ids,
                    generated_logprobs=frozen.generated_logprobs,
                    extras=call_extras,
                )
                if frozen.chunk_token_ids:
                    route_extras = None
                    attachments = None
                    if chunk_routes is not None:
                        route_extras = {
                            ROUTED_EXPERTS_FIELD: routed_experts_tensor_metadata(
                                chunk_routes
                            )
                        }
                        attachments = {ROUTED_EXPERTS_FIELD: chunk_routes}
                    if progress.staging_keys:
                        chunk_record = capture.build_generation_chunk_record(
                            state.call,
                            generated_token_ids=frozen.chunk_token_ids,
                            generated_logprobs=frozen.chunk_logprobs,
                            extras=route_extras,
                        )
                    elif route_extras is None:
                        chunk_record = cumulative_record
                    else:
                        # The first chunk is the whole prefix so far, with its
                        # routes beside the call's extras.
                        chunk_record = capture.build_prefix_record(
                            state.call,
                            prompt_token_ids=state.prompt_token_ids,
                            generated_token_ids=generated_token_ids,
                            generated_logprobs=frozen.generated_logprobs,
                            extras={**(call_extras or {}), **route_extras},
                        )
                    staged_key = yield chunk_record, frozen.chunk_sequence, attachments
                candidate_staging_keys = progress.staging_keys
                if staged_key is not None:
                    candidate_staging_keys += (staged_key,)
                acknowledgement = GenerationCutPrefixAck(
                    **prefix.model_dump(mode="json"),
                    disposition="durable_prefix",
                    cut_kind="active_prefix",
                    frozen_buffer_id=frozen.chunk_key,
                    staging_keys=candidate_staging_keys,
                    prefix_token_count=total_generation_token_count,
                    prefix_digest=cumulative_record.digest,
                    effective_output_limit=effective_output_limit,
                    terminal_finish_reason=progress.terminal_finish_reason,
                    terminal_stop_reason=progress.terminal_stop_reason,
                )
                if frozen.chunk_token_ids:
                    # Every acknowledgement in a batch is built before any of
                    # its calls seals; a failed one leaves its cut to roll back.
                    yield acknowledgement
                owned.seal(checkpoint_id, staged_key)
            except BaseException:
                # Closing the transaction early raises GeneratorExit here.
                # Leaving the lifecycle rolls the unsealed cut back.
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

    @staticmethod
    def _chunk_routes(
        owned: Any,
        state: _RequestCaptureState,
        progress: Any,
        frozen: Any,
        route_pieces: list[Any] | None,
    ) -> torch.Tensor | None:
        """The router-replay routes of the chunk a cut stages, or None.

        A finished call's own output carries its routes; otherwise the cut's
        progress read copied the engine's route pieces with its tokens.

        Raises:
            ValueError: the routes do not cover every token the cut stages.
        """
        dtype = state.routed_experts_dtype
        if dtype is None:
            return None
        frozen_count = len(frozen.generated_token_ids)
        staged = frozen_count - len(frozen.chunk_token_ids)
        engine_routes = progress.terminal_routed_experts
        if engine_routes is None and route_pieces is not None and frozen_count > staged:
            engine_routes = owned.absorb_engine_route_pieces(route_pieces)
        return _cut_chunk_routes(
            engine_routes,
            dtype=dtype,
            prev_len=state.call.admission.prev_len,
            prompt_len=len(state.prompt_token_ids),
            restored=state.restored_routed_experts,
            staged=staged,
            frozen=frozen_count,
        )

    def checkpoint_cut(self, inventory: Any, *, deadline: float | None = None) -> Any:
        """Execute a cut synchronously, keeping every lifecycle on this thread."""
        steps = self._checkpoint_steps(inventory, deadline=deadline)
        try:
            request = next(steps)
            while True:
                if isinstance(request, PrefixWrite):
                    request = steps.send(request.fallback())
                else:
                    request = next(steps)
        except StopIteration as finished:
            return finished.value
        finally:
            steps.close()

    def _checkpoint_steps(self, inventory: Any, *, deadline: float | None) -> CutSteps:
        """Prepare transactions, then seal their rows after durable write replies."""
        from nemo_gym._checkpoint.generation_cut import (
            GenerationCutReceipt,
        )

        receipt_key = (inventory.checkpoint_id, inventory.inventory_digest)
        with self._lock:
            cached_receipt = self._receipts.get(receipt_key)
        if cached_receipt is not None:
            return cached_receipt
        active_states = [
            state
            for prefix in inventory.active_prefixes
            if (state := self._registry.get_by_model_call_id(prefix.model_call_id))
            is not None
        ]
        route_pieces = self._record_progress(active_states, deadline=deadline)
        # Why each uncut call was left uncut; Gym counts them per checkpoint.
        uncut: Counter[str] = Counter()
        acknowledgements = yield from self._cut_batched(
            inventory, uncut, route_pieces=route_pieces, deadline=deadline
        )
        if uncut:
            LOGGER.info(
                "generation cut %s left %d of %d model calls uncut: %s",
                inventory.checkpoint_id,
                uncut.total(),
                len(inventory.active_prefixes),
                dict(sorted(uncut.items())),
            )
        receipt = GenerationCutReceipt(
            checkpoint_id=inventory.checkpoint_id,
            cut_id=f"worker-{inventory.inventory_digest}",
            inventory_digest=inventory.inventory_digest,
            inventory=inventory,
            backend_snapshot_id=f"tq-{inventory.inventory_digest}",
            prefixes=tuple(acknowledgements),
        )
        with self._lock:
            self._receipts[receipt_key] = receipt
            if len(self._receipts) > 256:
                self._receipts.pop(next(iter(self._receipts)))
        return receipt

    def _cut_batched(
        self,
        inventory: Any,
        uncut: Counter[str],
        *,
        route_pieces: dict[str, list[Any]],
        deadline: float | None,
    ) -> CutSteps:
        """Cut every call of the inventory, writing their rows in batches.

        Each call's cut stays frozen until its batch is written and every
        acknowledgement in the batch is built. A batch holds at most
        ``generation_prefix_batch_size`` rows and, unless one row alone is
        larger, ``generation_prefix_batch_max_tokens`` tokens.

        Gym acknowledges cuts per call, so a call that cannot be cut (a failed
        freeze, write, acknowledgement or seal) is acknowledged as a failure
        and regenerates after a restore; the other calls keep their cuts. A
        failed batch write is retried one row per write, so one bad row fails
        only its own call. Past ``deadline`` no further write starts, and the
        calls not yet written are acknowledged as failures.
        """
        max_rows = self._parts.generation_prefix_batch_size
        max_tokens = self._parts.generation_prefix_batch_max_tokens
        if self._pipeline_config is not None and self._pipeline_config.enabled:
            max_rows = min(max_rows, self._pipeline_config.max_batch_rows)
            max_tokens = min(max_tokens, self._pipeline_config.max_batch_tokens)
        checkpoint_id = inventory.checkpoint_id
        call_ids = [prefix.model_call_id for prefix in inventory.active_prefixes]
        if len(set(call_ids)) != len(call_ids):
            raise ValueError(
                "generation-cut inventory contains duplicate model-call identities"
            )
        sink = self._parts.sink
        acknowledgements: list[Any] = []
        pending: list[tuple[Generator[Any, Any, Any], Any, int, Any, Any]] = []
        pending_tokens = 0
        pending_bytes = 0
        config = self._pipeline_config
        byte_limit = (
            None
            if config is None or not config.enabled
            else config.max_batch_payload_bytes
        )
        late: list[Any] = []

        def past_deadline() -> bool:
            return deadline is not None and time.monotonic() >= deadline

        def fail(prefix: Any) -> None:
            LOGGER.exception(
                "cannot cut model call %s; it regenerates after a restore",
                prefix.model_call_id,
            )
            acknowledgements.append(
                self._failed_ack(prefix, reason="cut_error", uncut=uncut)
            )

        def release_pending() -> None:
            nonlocal pending_tokens, pending_bytes
            for transaction, _, _, _, prefix in pending:
                transaction.close()
                late.append(prefix)
            pending.clear()
            pending_tokens = 0
            pending_bytes = 0

        def stage(batch: list[tuple[Any, Any, int, Any, Any]]) -> list[str | None]:
            """Write a batch; if its write fails, write each row alone."""
            keys: list[str | None] = []
            if past_deadline():
                late.extend(prefix for _, _, _, _, prefix in batch)
                return [None] * len(batch)
            try:
                keys.extend(
                    sink.stage_generation_prefix_batch(
                        [record for _, record, _, _, _ in batch],
                        checkpoint_id=checkpoint_id,
                        chunk_sequences=[sequence for _, _, sequence, _, _ in batch],
                        attachments=[attachments for _, _, _, attachments, _ in batch],
                    )
                )
                return keys
            except Exception:
                # The sink has already discarded the batch's keys.
                if len(batch) == 1:
                    fail(batch[0][4])
                    return [None]
                LOGGER.warning(
                    "generation-prefix batch of %d rows failed; writing each alone",
                    len(batch),
                    exc_info=True,
                )
            keys = []
            for _, record, sequence, attachments, prefix in batch:
                if past_deadline():
                    late.append(prefix)
                    keys.append(None)
                    continue
                try:
                    keys += sink.stage_generation_prefix_batch(
                        [record],
                        checkpoint_id=checkpoint_id,
                        chunk_sequences=[sequence],
                        attachments=[attachments],
                    )
                except Exception:
                    fail(prefix)
                    keys.append(None)
            return keys

        def flush() -> CutSteps:
            nonlocal pending_tokens, pending_bytes
            if not pending:
                return
            if past_deadline():
                release_pending()
                return
            batch = list(pending)
            pending.clear()
            pending_tokens = 0
            size, pending_bytes = pending_bytes, 0
            keys: list[str | None] = []
            handed = 0
            try:
                keys = yield PrefixWrite(
                    checkpoint_id=checkpoint_id,
                    records=tuple(record for _, record, _, _, _ in batch),
                    sequences=tuple(sequence for _, _, sequence, _, _ in batch),
                    attachments=tuple(attached for _, _, _, attached, _ in batch),
                    payload_bytes=size,
                    fallback=functools.partial(stage, batch),
                )
                ready = []
                for (transaction, _, _, _, prefix), key in zip(
                    batch, keys, strict=True
                ):
                    handed += 1
                    if key is None:
                        transaction.close()
                        continue
                    try:
                        # A failure here rolls the call back and clears its row.
                        ready.append((transaction, prefix, transaction.send(key)))
                    except Exception:
                        fail(prefix)
                for transaction, prefix, acknowledgement in ready:
                    try:
                        transaction.send(None)
                    except StopIteration:
                        acknowledgements.append(acknowledgement)
                    except Exception:
                        fail(prefix)
                    else:
                        transaction.close()
                        acknowledgements.append(
                            self._failed_ack(prefix, reason="cut_error", uncut=uncut)
                        )
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
                for transaction, _, _, _, _ in batch:
                    transaction.close()
            yield BatchReleased()

        try:
            for index, prefix in enumerate(inventory.active_prefixes):
                if past_deadline():
                    release_pending()
                    late.extend(inventory.active_prefixes[index:])
                    break
                state = self._registry.get_by_model_call_id(prefix.model_call_id)
                try:
                    if state is None:
                        acknowledgements.append(
                            self._completed_ack(prefix, checkpoint_id, uncut)
                        )
                        continue
                    transaction = self._cut_transaction(
                        prefix,
                        state,
                        checkpoint_id,
                        uncut,
                        route_pieces=route_pieces.get(prefix.model_call_id),
                    )
                    record, sequence, attachments = next(transaction)
                except StopIteration as finished:
                    acknowledgement = finished.value
                    if acknowledgement is None:
                        acknowledgement = self._completed_ack(
                            prefix, checkpoint_id, uncut
                        )
                    acknowledgements.append(acknowledgement)
                    continue
                except Exception:
                    fail(prefix)
                    continue
                tokens = len(record.token_ids_delta)
                size = (
                    tokens * 16
                    + 512
                    + len(record.capture_key.encode("utf-8"))
                    + len(record.model_call_id.encode("utf-8"))
                )
                if byte_limit is not None and record.parent_call_id is not None:
                    size += len(record.parent_call_id.encode("utf-8"))
                if byte_limit is not None and record.extras is not None:
                    size += len(
                        json.dumps(record.extras, ensure_ascii=False).encode("utf-8")
                    )
                if attachments:
                    size += sum(
                        value.numel() * value.element_size()
                        for value in attachments.values()
                    )
                if byte_limit is not None and size > byte_limit:
                    transaction.close()
                    acknowledgements.append(
                        self._failed_ack(
                            prefix, reason="prefix_payload_limit", uncut=uncut
                        )
                    )
                    continue
                if pending and (
                    pending_tokens + tokens > max_tokens
                    or (byte_limit is not None and pending_bytes + size > byte_limit)
                ):
                    try:
                        yield from flush()
                    except BaseException:
                        transaction.close()
                        raise
                pending.append((transaction, record, sequence, attachments, prefix))
                pending_tokens += tokens
                pending_bytes += size
                if len(pending) >= max_rows or pending_tokens >= max_tokens:
                    yield from flush()
            yield from flush()
        finally:
            for transaction, _, _, _, _ in pending:
                transaction.close()
        if late:
            LOGGER.warning(
                "generation cut passed its control deadline: checkpoint_id=%s "
                "uncut=%d of %d model_call_ids=%s",
                checkpoint_id,
                len(late),
                len(inventory.active_prefixes),
                [prefix.model_call_id for prefix in late],
            )
            acknowledgements.extend(
                self._failed_ack(prefix, reason="control_deadline", uncut=uncut)
                for prefix in late
            )
        by_ticket = {ack.ticket_id: ack for ack in acknowledgements}
        return [by_ticket[prefix.ticket_id] for prefix in inventory.active_prefixes]

    def _fetch_chunks(self, continuation: Any) -> list[Any]:
        """Fetch the staged chunks named by a ``GenerationCutContinuation``, in order."""
        # Gym requires at least one staging key per continuation.
        chunks = self._fetch_rows(list(continuation.staging_keys))
        if not chunks:
            raise RuntimeError("generation-cut continuation has no staged chunks")
        return chunks

    def _fetch_rows(self, staging_keys: list[str]) -> list[Any]:
        """Fetch cut rows with their extras and routes, in the requested order.

        Also the fetch callback of the restore reader, which coalesces the
        keys of concurrent restores into one call.
        """
        # The finalization read also returns each row's extras, which carry a
        # media call's spans and tensor digests, and its route fragment when
        # router replay is on; plain ``fetch`` returns neither.
        chunks = self._parts.staging_source.fetch_for_finalization(
            staging_keys, include_route_fragments=self._parts.router_replay
        )
        if len(chunks) != len(staging_keys):
            raise RuntimeError("generation-cut fetch did not return every staged chunk")
        return chunks

    async def read_restore_chunks(self, admission: Any) -> list[Any] | None:
        """Read a restored call's chunks through the shared restore reader.

        Returns None when the admission restores no cut, or when the shared
        read failed; ``resolve_restore`` then reads the rows itself.
        """
        continuation = admission.generation_cut
        if continuation is None:
            return None
        try:
            return await self._reader.fetch(list(continuation.staging_keys))
        except Exception:  # noqa: BLE001 - resolve_restore retries the read
            LOGGER.debug(
                "shared prefix read failed for model call %s",
                admission.model_call_id,
                exc_info=True,
            )
            return None

    async def aclose(self) -> None:
        """Fail pending restore reads; call before ``shutdown``."""
        await self._reader.aclose()

    def _rebuild_prefix(
        self,
        admission: Any,
        prefix_token_ids: list[int],
        chunks: list[Any] | None = None,
    ) -> _ResolvedGenerationCut:
        """Rebuild the cumulative prefix from an admission's staged chunks.

        ``chunks`` are the already-fetched chunks, so a restore that also reads
        their media fetches once; they are fetched here when omitted. With
        router replay on, each chunk's routes are verified and joined too.
        """
        continuation = admission.generation_cut
        if chunks is None:
            chunks = self._fetch_chunks(continuation)
        snapshots = [chunk.snapshot for chunk in chunks]
        # A media call's spans and tensor digests ride the first chunk's extras,
        # and the advertised prefix digest covers them. The source pops each
        # chunk's route payload out of its decoded extras, so a router-replay
        # chunk with nothing else decodes as ``{}``.
        if any(chunk.extras for chunk in chunks[1:]):
            raise RuntimeError(
                "only the first generation-cut chunk may carry call extras"
            )
        # The cut digested no extras for a text call, so match it.
        extras = chunks[0].extras or None
        # Routes are checked per chunk and stay out of the prefix digest.
        routed_experts = _join_chunk_routes(chunks, required=self._parts.router_replay)
        weight_versions = [snapshot.weight_version for snapshot in snapshots]
        if weight_versions != sorted(weight_versions):
            raise RuntimeError(
                "generation-cut chunk policy versions are not monotonically non-decreasing"
            )
        current_weight_version = self._parts.weight_version()
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
            compute_chain_hash,
            compute_extras_digest,
            hash_token_ids,
        )
        from nemo_gym.token_id_capture.staging.records import StagedCallRecord

        delta_len = len(token_ids_delta)
        # A cut chain may span retries performed after later weight updates.
        # Its token-level behavior logprobs remain attached to their original
        # chunks. The cumulative snapshot uses the oldest contributing version
        # so downstream replay-buffer staleness checks remain conservative.
        # Gym's begin_call checks this snapshot's digest and generated-token
        # count against the admission before the call is served.
        snapshot = StagedCallRecord.from_components(
            capture_key=continuation.source_capture_key,
            model_call_id=continuation.source_model_call_id,
            parent_call_id=admission.parent_call_id,
            mode=admission.mode,
            prev_len=admission.prev_len,
            delta_len=delta_len,
            cum_len=admission.prev_len + delta_len,
            weight_version=weight_versions[0],
            token_ids_delta=token_ids_delta,
            token_mask_delta=token_mask_delta,
            generation_log_probs_delta=generation_logprobs_delta,
            extras=extras,
            extras_digest=compute_extras_digest(extras),
            chain_hash=compute_chain_hash(admission.parent_chain_hash, token_ids_delta),
            cumulative_hash=hash_token_ids(prefix_token_ids + token_ids_delta),
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
            admission.capture_key,
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

    def resolve_restore(
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
        Returns the admission to serve and the verified prefix with its media
        and routes, or the admission without its cut and None. ``chunks`` are
        the cut's rows when ``read_restore_chunks`` already read them;
        otherwise they are read here.
        """
        continuation = admission.generation_cut
        if continuation is None:
            return admission, None
        try:
            # One fetch serves both the media and the prefix.
            if chunks is None:
                chunks = self._fetch_chunks(continuation)
            media = self._parts.restore_media(chunks)
            resolved = self._rebuild_prefix(admission, prefix_token_ids, chunks)
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
        self.retire(continuation.staging_keys)
        return admission.model_copy(update={"generation_cut": None}), None

    def remember_completed(
        self,
        model_call_id: str,
        coords: Any,
        generation_token_count: int,
        *,
        effective_output_limit: int | None = None,
        terminal_finish_reason: Literal["stop", "length"] | None = None,
        terminal_stop_reason: str | int | None = None,
    ) -> None:
        """Keep a finished call's terminal row for a cut that still names it."""
        now = time.monotonic()
        with self._lock:
            # Dict insertion order is completion order. Remove only entries old
            # enough that the response/inventory race has certainly elapsed;
            # never discard fresh evidence because the deployment crossed an
            # arbitrary call-count threshold.
            while self._completed:
                oldest_id = next(iter(self._completed))
                oldest = self._completed[oldest_id]
                if (
                    now - oldest.completed_at_monotonic
                    <= _COMPLETED_CAPTURE_RETENTION_S
                ):
                    break
                self._completed.pop(oldest_id)
            self._completed.pop(model_call_id, None)
            self._completed[model_call_id] = _CompletedCaptureState(
                coords=coords,
                generation_token_count=generation_token_count,
                effective_output_limit=effective_output_limit,
                terminal_finish_reason=terminal_finish_reason,
                terminal_stop_reason=terminal_stop_reason,
                completed_at_monotonic=now,
            )


class NoPrefixRecovery:
    """What a worker without generation-prefix cuts does at each recovery point.

    Every hook is a no-op, so the worker never asks whether cuts are on.
    """

    control_token: str | None = None

    def resolve_restore(
        self,
        admission: Any,
        prefix_token_ids: list[int],
        chunks: list[Any] | None = None,
    ) -> tuple[Any, _ResolvedGenerationCut | None]:
        if admission.generation_cut is None:
            return admission, None
        # Gym only sends a restored cut to a worker that acknowledged one, so
        # this is a misconfiguration; regenerating the call is always valid.
        LOGGER.warning(
            "generation-prefix cuts are off; regenerating model call %s from "
            "its turn boundary",
            admission.model_call_id,
        )
        return admission.model_copy(update={"generation_cut": None}), None

    def on_begin(self, admission: Any) -> None:
        pass

    def on_sampling_params(self, request: Any, sampling_params: Any) -> None:
        pass

    def observe(self, request: Any, request_output: Any) -> None:
        pass

    def restore_reply_text(
        self, request: Any, request_output: Any, *, tokenizer: Any
    ) -> None:
        pass

    def remember_completed(
        self,
        model_call_id: str,
        coords: Any,
        generation_token_count: int,
        *,
        effective_output_limit: int | None = None,
        terminal_finish_reason: Literal["stop", "length"] | None = None,
        terminal_stop_reason: str | int | None = None,
    ) -> None:
        # Only a generation cut reads completed calls.
        pass

    def retire(self, staging_keys: Sequence[str]) -> None:
        pass

    async def flush(self, epoch: int) -> None:
        pass

    async def read_restore_chunks(self, admission: Any) -> list[Any] | None:
        return None

    async def aclose(self) -> None:
        pass

    def counts(self) -> dict[str, int]:
        return {"restored_calls": 0, "restored_prefix_tokens": 0}

    def shutdown(self) -> None:
        pass

    async def run_cut(self, inventory: Any) -> Any:
        raise RuntimeError("generation-prefix cuts are disabled on this worker")
