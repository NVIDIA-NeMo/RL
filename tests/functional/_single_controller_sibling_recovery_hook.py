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

"""Test-only SC entrypoint for deterministic sibling-level recovery.

The first process lets one sibling become ledger-sealed, then parks the next
completion before the ledger records it. Earlier-step completions wait for that
cut, guaranteeing that the step-1 checkpoint contains the selected partial
group. The second process records which generation indices are redispatched.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
from pathlib import Path
from typing import Any, cast

import ray

from examples import run_grpo_single_controller
from nemo_rl.data_plane import build_data_plane_client
from nemo_rl.data_plane.adapters.tq_mooncake_checkpoint import run_checkpoint_command
from nemo_rl.data_plane.tq_token_sink import TQTokenSource, _select_row
from nemo_rl.environments.gym_checkpoint import gym_capture_key
from nemo_rl.experience.rollout_reassembler import RolloutReassembler
from nemo_rl.experience import rollout_reassembler_actor as reassembler_actor_module
from nemo_rl.experience.rollout_reassembler_actor import assert_metadata_only
from nemo_rl.experience.rollout_manager import RolloutCompletionCallback, RolloutManager
from nemo_rl.experience.rollout_recovery import RecoveryGranularity
from nemo_rl.utils.venvs import make_actor_runtime_env
from tests.functional._turn_recovery_token_evidence import (
    build_rollout_token_evidence,
)


_SELECTION_ENV = "SC_GYM_TURN_RECOVERY_SELECTION"
_TOKEN_EVIDENCE_DIR_ENV = "SC_GYM_TURN_RECOVERY_TOKEN_EVIDENCE_DIR"


@ray.remote(
    num_cpus=1,
    num_gpus=0,
    max_restarts=0,
    max_task_retries=0,
)
class _TokenEvidenceRolloutReassemblerActor:
    """Production finalizer plus test-only proof of pre-cut token preservation."""

    def __init__(
        self,
        dp_config: Any,
        config: Any,
        *,
        selection_path: str,
        evidence_dir: str,
    ) -> None:
        self._dp_client = build_data_plane_client(dp_config, bootstrap=False)
        self._partition_id = config.partition_id
        self._staging = TQTokenSource(
            self._dp_client,
            staging_partition=config.staging_partition,
        )
        self._finalizer = RolloutReassembler(
            self._dp_client,
            partition_id=config.partition_id,
            staging_partition=config.staging_partition,
            pad_token_id=config.pad_token_id,
            router_replay_enabled=config.router_replay_enabled,
            defer_routed_experts_to_policy=config.defer_routed_experts_to_policy,
            max_seq_len=config.max_seq_len,
        )
        path = Path(selection_path)
        self._selection = json.loads(path.read_text()) if path.is_file() else None
        self._evidence_dir = Path(evidence_dir)

    def mooncake_checkpoint(self, body: dict[str, Any]) -> dict[str, Any] | None:
        return run_checkpoint_command(body)

    def check_dependencies(self) -> None:
        from nemo_gym.token_id_capture.staging.rebuild import (  # noqa: F401
            RebuildError,
            ReceiptVerificationError,
            verify_and_linearize,
        )
        from nemo_gym.token_id_capture.staging.records import (  # noqa: F401
            RolloutReceipt,
        )

    def _continued_rollouts(self, request: Any) -> list[dict[str, Any]]:
        selection = self._selection
        if (
            not isinstance(selection, dict)
            or selection.get("group_id") != request.group_id
        ):
            return []
        rows = selection.get("continued_rollouts")
        if not isinstance(rows, list) or not rows:
            raise AssertionError(
                "selected recovery group contains no continued-rollout token evidence"
            )
        return rows

    def finalize(self, request: Any) -> Any:
        assert_metadata_only(request)
        if not (
            len(request.rollout_ids)
            == len(request.canonical_sample_ids)
            == len(request.receipts)
            == len(request.rewards)
            == len(request.mask_sample)
        ):
            raise ValueError(
                "finalizer request rollout_ids, canonical_sample_ids, receipts, "
                "rewards, and mask_sample must be parallel"
            )
        continued_rollouts = self._continued_rollouts(request)
        staged_by_rollout: dict[str, list[dict[str, Any]]] = {}
        for descriptor in continued_rollouts:
            rollout_id = descriptor["rollout_id"]
            source_capture_key = descriptor["source_capture_key"]
            expected_segments = descriptor["pre_cut_segments"]
            staging_keys = [segment["staging_key"] for segment in expected_segments]
            fetched = self._staging.fetch_for_finalization(staging_keys)
            fetched_by_key = {row.staging_key: row for row in fetched}
            if set(fetched_by_key) != set(staging_keys):
                raise AssertionError(
                    "restored TQ rows do not match the phase-one storage references: "
                    f"rollout={rollout_id!r}"
                )
            segments: list[dict[str, Any]] = []
            for expected in expected_segments:
                row = fetched_by_key[expected["staging_key"]]
                snapshot = row.snapshot
                if snapshot.rollout_id != source_capture_key:
                    raise AssertionError(
                        "snapshot storage reference resolved to the wrong rollout: "
                        f"key={row.staging_key!r}, expected={source_capture_key!r}, "
                        f"actual={snapshot.rollout_id!r}"
                    )
                if (
                    snapshot.prev_len != expected["prev_len"]
                    or snapshot.cum_len != expected["cum_len"]
                ):
                    raise AssertionError(
                        "restored TQ row moved its pre-crash token boundaries: "
                        f"key={row.staging_key!r}"
                    )
                segments.append(
                    {
                        "staging_key": row.staging_key,
                        "prev_len": snapshot.prev_len,
                        "cum_len": snapshot.cum_len,
                        "token_ids_delta": list(snapshot.token_ids_delta),
                        "selection_staging_digest": expected["staging_digest"],
                        "restored_staging_digest": snapshot.digest,
                    }
                )
            staged_by_rollout[rollout_id] = segments

        result = self._finalizer.finalize_group(
            request.group_id,
            list(request.rollout_ids),
            list(request.receipts),
            list(request.rewards),
            mask_sample=list(request.mask_sample),
            fallback_weight_version=request.fallback_weight_version,
            prompt_idx=request.prompt_idx,
            loss_multiplier=request.loss_multiplier,
            canonical_sample_ids=list(request.canonical_sample_ids),
        )
        assert_metadata_only(result)
        if not continued_rollouts:
            return result
        if result.meta is None or result.dropped:
            raise AssertionError(
                f"selected recovery group {request.group_id!r} produced no canonical row"
            )

        canonical_rows = self._dp_client.get_samples(
            sample_ids=list(request.canonical_sample_ids),
            partition_id=self._partition_id,
            select_fields=["input_ids", "input_lengths"],
        )
        row_count = (
            int(canonical_rows.batch_size[0]) if canonical_rows.batch_size else 0
        )
        if row_count != len(request.canonical_sample_ids):
            raise AssertionError(
                "finalizer token-evidence fetch returned the wrong number of rows: "
                f"expected={len(request.canonical_sample_ids)}, actual={row_count}"
            )

        evidence_rows: list[dict[str, Any]] = []
        for descriptor in continued_rollouts:
            rollout_id = descriptor["rollout_id"]
            try:
                row_index = request.canonical_sample_ids.index(rollout_id)
            except ValueError as error:
                raise AssertionError(
                    f"continued rollout {rollout_id!r} has no final canonical row"
                ) from error
            row = _select_row(canonical_rows, row_index)
            input_length = int(row["input_lengths"].reshape(-1)[0].item())
            canonical_token_ids = [
                int(token_id)
                for token_id in row["input_ids"].reshape(-1).tolist()[:input_length]
            ]
            evidence_rows.append(
                build_rollout_token_evidence(
                    rollout_id=rollout_id,
                    canonical_token_ids=canonical_token_ids,
                    staged_segments=staged_by_rollout[rollout_id],
                )
            )

        self._evidence_dir.mkdir(parents=True, exist_ok=True)
        name = hashlib.sha256(request.group_id.encode("utf-8")).hexdigest()
        destination = self._evidence_dir / f"{name}.json"
        temporary = destination.with_suffix(".tmp")
        temporary.write_text(
            json.dumps(
                {"group_id": request.group_id, "rollouts": evidence_rows},
                sort_keys=True,
                indent=2,
            )
            + "\n"
        )
        os.replace(temporary, destination)
        return result


def _create_token_evidence_finalizers(
    dp_config: Any,
    config: Any,
    *,
    num_workers: int,
) -> list[Any]:
    """Create finalizers that additionally verify the restored canonical token row."""
    if num_workers <= 0:
        raise ValueError(f"num_reassembler_workers must be positive, got {num_workers}")
    selection_path = os.environ[_SELECTION_ENV]
    evidence_dir = os.environ[_TOKEN_EVIDENCE_DIR_ENV]
    runtime_env = make_actor_runtime_env(
        "nemo_rl.experience.rollout_reassembler_actor.RolloutReassemblerActor"
    )
    actors = [
        _TokenEvidenceRolloutReassemblerActor.options(runtime_env=runtime_env).remote(
            dp_config,
            config,
            selection_path=selection_path,
            evidence_dir=evidence_dir,
        )
        for _ in range(num_workers)
    ]
    try:
        ray.get([actor.check_dependencies.remote() for actor in actors])
    except ray.exceptions.RayError:
        for actor in actors:
            try:
                ray.kill(actor)
            except Exception as error:
                print(f"finalizer actor termination failed: {error}", flush=True)
        raise
    return actors


class _InstrumentedNemoGymRolloutImpl:
    """Delegate Gym rollouts while controlling one streamed-completion cut."""

    def __init__(
        self,
        delegate: Any,
        *,
        recovery_ledger: Any,
        events_path: Path,
        block_target_step: int | None,
    ) -> None:
        self._delegate = delegate
        self._recovery_ledger = recovery_ledger
        self._events_path = events_path
        self._block_target_step = block_target_step
        self._selected_group_id: str | None = None
        # Construct this lazily inside the Ray actor's event loop. The rollout
        # manager and this test wrapper are built driver-side and serialized into
        # that actor, so an eagerly created asyncio primitive could bind to the
        # wrong loop.
        self._selected_sibling_sealed: asyncio.Event | None = None

    def __getattr__(self, name: str) -> Any:
        delegate = self.__dict__.get("_delegate")
        if delegate is None:
            raise AttributeError(name)
        return getattr(delegate, name)

    def _append_event(self, event: str, **fields: Any) -> None:
        self._events_path.parent.mkdir(parents=True, exist_ok=True)
        with self._events_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps({"event": event, **fields}, sort_keys=True) + "\n")

    def _find_group(self, rollout_ids: list[str]) -> Any:
        rollout_id_set = set(rollout_ids)
        matches = [
            group
            for group in self._recovery_ledger.groups()
            if rollout_id_set.intersection(group.logical_rollout_ids)
        ]
        if len(matches) != 1:
            raise RuntimeError(
                "sibling recovery hook could not uniquely resolve rollout IDs "
                f"to one ledger group: ids={rollout_ids!r}, matches="
                f"{[group.group_id for group in matches]!r}"
            )
        return matches[0]

    def _sibling_sealed_event(self) -> asyncio.Event:
        if self._selected_sibling_sealed is None:
            self._selected_sibling_sealed = asyncio.Event()
        return self._selected_sibling_sealed

    def _record_forwarded_completion(
        self,
        *,
        completion: Any,
        fields: dict[str, Any],
    ) -> None:
        self._append_event(
            "completion_forwarded",
            **fields,
            reward=float(completion.reward),
        )

    async def run_rollout(
        self,
        input_sample: Any,
        *,
        rollout_ids: list[str] | None = None,
        attempt_indices: list[int] | None = None,
        generation_indices: list[int] | None = None,
        on_completion: RolloutCompletionCallback | None = None,
        recovery_granularity: RecoveryGranularity = RecoveryGranularity.SIBLING,
    ) -> Any:
        if (
            rollout_ids is None
            or attempt_indices is None
            or generation_indices is None
            or on_completion is None
        ):
            raise RuntimeError(
                "sibling recovery hook requires the token-capture rollout path"
            )
        if recovery_granularity is not RecoveryGranularity.SIBLING:
            raise RuntimeError(
                "sibling recovery hook requires sibling recovery granularity"
            )

        group = self._find_group(rollout_ids)
        indices = list(generation_indices)
        capture_rollout_ids = [
            gym_capture_key(rollout_id, attempt_index)
            for rollout_id, attempt_index in zip(
                rollout_ids, attempt_indices, strict=True
            )
        ]
        fields = {
            "group_id": group.group_id,
            "prompt_idx": int(input_sample["idx"]),
            "target_step": group.target_step,
            "generation_indices": indices,
            "rollout_ids": capture_rollout_ids,
        }
        self._append_event("dispatch", **fields)

        selected = False
        if (
            self._block_target_step is not None
            and group.target_step == self._block_target_step
            and self._selected_group_id is None
        ):
            # Selection occurs before the first await, so concurrent rollout tasks
            # cannot select two groups on this event loop.
            self._selected_group_id = group.group_id
            selected = True

        sealed_in_selected_call = False

        async def _instrumented_completion(
            generation_index: int, completion: Any
        ) -> None:
            nonlocal sealed_in_selected_call
            completion_fields = {
                **fields,
                "generation_index": generation_index,
                "rollout_id": capture_rollout_ids[generation_index],
            }
            if selected:
                if not sealed_in_selected_call:
                    await on_completion(generation_index, completion)
                    self._record_forwarded_completion(
                        completion=completion,
                        fields=completion_fields,
                    )
                    sealed_in_selected_call = True
                    self._append_event("sibling_sealed", **completion_fields)
                    self._sibling_sealed_event().set()
                    return

                self._append_event("blocked_before_ledger_seal", **completion_fields)
                print(
                    "sibling recovery functional hook: blocked group="
                    f"{group.group_id} generation_index={generation_index}",
                    flush=True,
                )
                await asyncio.Event().wait()

            # Do not allow the preceding train step to complete until the selected
            # lookahead group has one durable sibling. This removes checkpoint timing
            # from the test: the step-1 save cannot occur before the intended cut.
            if (
                self._block_target_step is not None
                and group.target_step is not None
                and group.target_step < self._block_target_step
            ):
                await self._sibling_sealed_event().wait()
            await on_completion(generation_index, completion)
            self._record_forwarded_completion(
                completion=completion,
                fields=completion_fields,
            )

        result = await self._delegate.run_rollout(
            input_sample,
            rollout_ids=rollout_ids,
            attempt_indices=attempt_indices,
            generation_indices=indices,
            on_completion=_instrumented_completion,
            recovery_granularity=recovery_granularity,
        )
        self._append_event("capture_complete", **fields)
        return result


class _InstrumentedRolloutManager:
    """Install the streamed-completion hook inside the controller actor.

    The rollout manager is created on the driver and serialized into Ray. Deferring
    installation until the first actor-side call ensures the inner hook receives the
    exact recovery ledger mutated by the live manager rather than a driver-side copy.
    """

    def __init__(
        self,
        delegate: RolloutManager,
        *,
        events_path: Path,
        block_target_step: int | None,
    ) -> None:
        self._delegate = delegate
        self._events_path = events_path
        self._block_target_step = block_target_step
        self._hook_installed = False

    def __getattr__(self, name: str) -> Any:
        delegate = self.__dict__.get("_delegate")
        if delegate is None:
            raise AttributeError(name)
        return getattr(delegate, name)

    @property
    def _tq_buffer(self) -> Any:
        return self._delegate._tq_buffer

    @_tq_buffer.setter
    def _tq_buffer(self, value: Any) -> None:
        self._delegate._tq_buffer = value

    def _install_hook(self) -> None:
        if self._hook_installed:
            return
        self._delegate._impl = _InstrumentedNemoGymRolloutImpl(
            self._delegate._impl,
            recovery_ledger=self._delegate.recovery_ledger,
            events_path=self._events_path,
            block_target_step=self._block_target_step,
        )
        self._hook_installed = True

    async def generate_for_finalization(
        self,
        input_sample: Any,
        *,
        target_step: int | None = None,
        inflight_registry: Any = None,
        lineage_group_id: str | None = None,
    ) -> Any:
        self._install_hook()
        return await self._delegate.generate_for_finalization(
            input_sample,
            target_step=target_step,
            inflight_registry=inflight_registry,
            lineage_group_id=lineage_group_id,
        )


reassembler_actor_module.create_rollout_reassembler_actors = (
    _create_token_evidence_finalizers
)
_original_setup_single_controller = run_grpo_single_controller.setup_single_controller


def _setup_with_sibling_recovery_hook(*args: Any, **kwargs: Any) -> Any:
    actor_args, timing_metrics = _original_setup_single_controller(*args, **kwargs)
    events_path = Path(os.environ["SC_SIBLING_RECOVERY_TEST_EVENTS"])
    raw_target_step = os.environ.get("SC_SIBLING_RECOVERY_BLOCK_TARGET_STEP")
    block_target_step = int(raw_target_step) if raw_target_step is not None else None
    actor_args.rollout_manager = cast(
        RolloutManager,
        _InstrumentedRolloutManager(
            actor_args.rollout_manager,
            events_path=events_path,
            block_target_step=block_target_step,
        ),
    )
    return actor_args, timing_metrics


run_grpo_single_controller.setup_single_controller = _setup_with_sibling_recovery_hook


if __name__ == "__main__":
    run_grpo_single_controller.main()
