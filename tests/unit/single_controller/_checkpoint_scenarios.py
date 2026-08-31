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

"""Scenario harness for the checkpoint no-data-loss property.

The property under test, in one sentence: **pausing a run to checkpoint and
resuming it must train on exactly the same prompt groups as never pausing at
all.** Concretely, every prompt group the dataloader has already handed out and
the trainer has not yet consumed must come back after a restore -- whether it
finished generating or not. If it does not come back, that prompt is lost for
good, because the dataloader cursor is saved where it stands and never rewinds.

Everything here runs on CPU. The real ``TQReplayBuffer``, the real samplers, the
real ``DataPlaneCheckpointBarrier`` and the real ``NoOpDataPlaneClient``
save/load are exercised. Only two things are stubbed, and neither is on the path
under test:

* ``record_to_train_batch`` -- the tensor converter, so a scenario can use empty
  prompt records instead of building real rollouts.
* the trainer/generation side -- absent entirely; this harness is the buffer and
  the samplers, which is where save/restore lives.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import torch

from nemo_rl.algorithms.async_utils import replay_buffer as _rb
from nemo_rl.algorithms.async_utils.replay_buffer import (
    DataPlaneCheckpointBarrier,
    TQReplayBuffer,
)
from nemo_rl.algorithms.async_utils.staleness_sampler import (
    InOrderSamplerConfig,
    ReadyFirstSamplerConfig,
    WeightFifoSamplerConfig,
    WindowedSamplerConfig,
    create_sampler,
)
from nemo_rl.data_plane.adapters.noop import NoOpDataPlaneClient
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.experience.interfaces import PromptGroupRecord
from nemo_rl.experience.rollout_recovery import RolloutRecoveryLedger

PARTITION = "rollout_data"
ROLLOUTS_PER_GROUP = 2  # rollouts_per_prompt_group
GROUPS_PER_STEP = 3  # prompt_groups per training step
CAPACITY = 64  # max_buffered_rollouts
_FIELDS = ["input_ids", "input_lengths", "total_reward"]

SAMPLERS = ("windowed", "ready_first", "weight_fifo", "in_order")


def sampler_config(name: str, lag: int):
    """Build the real discriminated sampler config for ``name``."""
    if name == "windowed":
        return WindowedSamplerConfig(max_staleness_versions=lag)
    if name == "ready_first":
        return ReadyFirstSamplerConfig(max_staleness_versions=lag)
    if name == "weight_fifo":
        return WeightFifoSamplerConfig(max_staleness_versions=lag)
    if name == "in_order":
        return InOrderSamplerConfig(max_lookahead_versions=lag)
    raise ValueError(f"unknown sampler {name!r}")


# ── scenario description ────────────────────────────────────────────────────


@dataclass(frozen=True)
class Group:
    """One prompt group at checkpoint time.

    Args:
        gid: Prompt-group number, matching the order the dataloader served it.
        done: How many of its ``ROLLOUTS_PER_GROUP`` rollouts have finished.
            ``ROLLOUTS_PER_GROUP`` means the group committed; anything less
            means it is still in flight.
        weight: Weight version the group was dispatched at.
        target: ``target_step`` stamp, used by the gated samplers.
        evicted: The sampler deliberately dropped it (too stale). An evicted
            group is an intentional discard, not data loss, so it is excluded
            from what a restore must return.
    """

    gid: int
    done: int
    weight: int = 0
    target: Optional[int] = None
    evicted: bool = False


@dataclass(frozen=True)
class Scenario:
    """A checkpoint taken mid-run.

    Args:
        name: Short label used in test ids.
        groups: Every prompt group the dataloader has handed out so far.
        cursor: The next prompt-group number the dataloader would serve. Every
            group below it has already been handed out and will never be
            handed out again after a restore.
        trained: Groups the trainer has already consumed. Not necessarily
            contiguous -- some samplers train whatever is ready and leave an
            earlier unfinished group behind.
        lag: How far generation may run ahead, in steps.
    """

    name: str
    groups: tuple[Group, ...]
    cursor: int
    trained: frozenset[int] = field(default_factory=frozenset)
    lag: int = 1

    def must_survive(self) -> set[str]:
        """Groups a restore has to return, or data is lost.

        Handed out, not trained, not deliberately evicted -- and below the
        cursor, so the dataloader will never produce them again. This is the
        no-data-loss bar enforced by the recovery matrix.
        """
        return {
            _gid(g.gid)
            for g in self.groups
            if not g.evicted and g.gid not in self.trained and g.gid < self.cursor
        }

    def committed_outstanding(self) -> set[str]:
        """Completed, unconsumed groups covered by #3480's TQ recovery."""
        return {
            _gid(g.gid)
            for g in self.groups
            if not g.evicted
            and g.gid not in self.trained
            and g.done == ROLLOUTS_PER_GROUP
        }

    def expected_stamps(self) -> dict[str, tuple[int | None, int]]:
        """Target-step and start-weight stamps every restored group must retain."""
        return {
            _gid(group.gid): (group.target, group.weight)
            for group in self.groups
            if not group.evicted and group.gid not in self.trained
        }


def _gid(n: int) -> str:
    return f"g{n:02d}"


@dataclass(frozen=True)
class Case:
    """One row of the test matrix: a scenario run under one sampler.

    Args:
        scenario: The buffer state at checkpoint time.
        sampler: Which sampler the run is configured with.
        why: Optional diagnostic context for an expected behavior gap.
    """

    scenario: Scenario
    sampler: str
    why: str = ""

    @property
    def id(self) -> str:
        return f"{self.sampler}::{self.scenario.name}"


# ── the round trip ──────────────────────────────────────────────────────────


def _record() -> PromptGroupRecord:
    return PromptGroupRecord(
        prompt_idx=0,
        prompt=[],
        extra_env_info=None,
        metadata={},
        completions=[],
        rollout_metrics={},
    )


def _stub_converter(record: PromptGroupRecord, *, pad_value_dict: Any):
    del record, pad_value_dict
    return BatchedDataDict[Any](
        {
            "input_ids": torch.ones((ROLLOUTS_PER_GROUP, 3), dtype=torch.long),
            "input_lengths": torch.full((ROLLOUTS_PER_GROUP,), 3, dtype=torch.long),
            "total_reward": torch.zeros(ROLLOUTS_PER_GROUP, dtype=torch.float32),
        }
    )


def patch_converter(monkeypatch) -> None:
    """Swap the tensor converter so scenarios can use empty prompt records."""
    monkeypatch.setattr(_rb, "record_to_train_batch", _stub_converter)


def _fresh_client(register: bool) -> NoOpDataPlaneClient:
    dp = NoOpDataPlaneClient()
    if register:
        dp.register_partition(
            partition_id=PARTITION,
            fields=list(_FIELDS),
            num_samples=CAPACITY * ROLLOUTS_PER_GROUP,
            consumer_tasks=["train"],
        )
    return dp


def _new_buffer(dp: NoOpDataPlaneClient) -> TQReplayBuffer:
    buf = TQReplayBuffer(
        dp,
        partition_id=PARTITION,
        pad_value_dict={"input_ids": 0},
        require_routed_experts=False,
    )
    buf.set_data_plane_checkpoint_barrier(DataPlaneCheckpointBarrier())
    return buf


async def _fill(buf: TQReplayBuffer, scenario: Scenario) -> None:
    """Recreate the scenario through the buffer's real reserve/commit path.

    Only groups still held at checkpoint time are added. A trained group is gone
    -- ``_finalize_selection`` removes it from the buffer as it hands it to the
    trainer -- and an evicted one was dropped by the staleness rule.
    """
    for g in scenario.groups:
        if g.evicted or g.gid in scenario.trained:
            continue
        gid = buf.reserve(
            weight_version=g.weight, target_step=g.target, group_id=_gid(g.gid)
        )
        if g.done == ROLLOUTS_PER_GROUP:
            await buf.commit(
                gid,
                _record(),
                start_weight_version=g.weight,
                end_weight_version=g.weight,
            )
        # done < ROLLOUTS_PER_GROUP: still generating, so no commit yet.


@dataclass
class RoundTrip:
    """What a save/restore cycle returned.

    ``recovered`` is deliberately *presence*, not readiness: a group counts as
    recovered if the restored buffer knows about it at all. That keeps the
    assertions independent of whether recovery regenerates the whole group or
    later resumes only missing siblings. A group could come back already
    committed, or as a reserved slot waiting to be finished -- either way the
    run has not lost the prompt, and either way these tests notice.
    ``ready`` and ``pending`` are reported separately for diagnosis only;
    nothing asserts on them. ``stamps`` records each restored group's
    ``target_step`` and start weight. ``selected`` and ``selected_count`` report
    the optional restore-then-select result used to verify each sampler's
    recovery key at multiple gate lags.
    """

    recovered: set[str]
    ready: set[str]
    pending: set[str]
    saved_sidecar: bool
    rows_before: set[str]
    rows_after: set[str]
    stamps: dict[str, tuple[int | None, int]]
    selected: set[str]
    selected_count: int


async def _round_trip(
    scenario: Scenario,
    sampler_name: str,
    tmp_path: Path,
    *,
    select_current_train_weight: int | None = None,
) -> RoundTrip:
    dp_a = _fresh_client(register=True)
    buf_a = _new_buffer(dp_a)
    await _fill(buf_a, scenario)
    sampler_a = create_sampler(buf_a, sampler_config(sampler_name, scenario.lag))

    # Mirrors SingleControllerActor._save_checkpoint: the sidecar is written
    # only when the sampler says it can restore one.
    sidecar = (
        buf_a.metadata_state_dict(saved_capacity=CAPACITY)
        if sampler_a.supports_buffer_checkpoint
        else None
    )
    recovery_ledger_a = RolloutRecoveryLedger()
    async with buf_a.data_plane_checkpoint_barrier.mutation() as cut:
        for group in scenario.groups:
            if (
                group.evicted
                or group.gid in scenario.trained
                or group.done == ROLLOUTS_PER_GROUP
            ):
                continue
            recovery_ledger_a.reserve_group(
                cut,
                group_id=_gid(group.gid),
                admission_id=f"batch-{group.target}",
                prompt_id=str(group.gid),
                prompt_payload={"idx": group.gid, "message_log": []},
                expected_generations=ROLLOUTS_PER_GROUP,
                target_step=group.target,
                start_weight_version=group.weight,
                admitted=True,
            )
    recovery_sidecar = recovery_ledger_a.state_dict()
    rows_before = set(dp_a.list_sample_ids(PARTITION))
    dp_a.save_checkpoint(tmp_path / "data_plane")

    # ---- restart: brand new process, nothing carried over in memory ----
    dp_b = _fresh_client(register=False)  # load_checkpoint demands a clean client
    dp_b.load_checkpoint(tmp_path / "data_plane")
    buf_b = _new_buffer(dp_b)
    sampler_b = create_sampler(buf_b, sampler_config(sampler_name, scenario.lag))

    # Mirrors SingleControllerActor._maybe_restore_replay_buffer.
    if sidecar is not None and sampler_b.supports_buffer_checkpoint:
        await buf_b.load_state_dict(
            sidecar,
            max_groups=CAPACITY,
            expected_partition_id=PARTITION,
            expected_group_size=ROLLOUTS_PER_GROUP,
            expected_manifest_digest=sidecar["manifest_digest"],
        )
        recovery_ledger_b = RolloutRecoveryLedger()
        async with buf_b.data_plane_checkpoint_barrier.mutation() as cut:
            recovery_ledger_b.load_state_dict(cut, recovery_sidecar)
            recovery_ledger_b.discard_canonical_groups(cut, set(buf_b._group_ids))
        for group in recovery_ledger_b.groups():
            group_id = buf_b.reserve(
                weight_version=group.start_weight_version,
                target_step=group.target_step,
                group_id=group.group_id,
            )
            await buf_b.commit(
                group_id,
                _record(),
                start_weight_version=group.start_weight_version,
                end_weight_version=group.start_weight_version,
            )
            async with buf_b.data_plane_checkpoint_barrier.mutation() as cut:
                recovery_ledger_b.discard_group(cut, group_id)

    ready = {
        gid for gid, is_ready in zip(buf_b._group_ids, buf_b.ready_list) if is_ready
    }
    recovered = set(buf_b._group_ids)
    pending = recovered - ready
    rows_after = set(dp_b.list_sample_ids(PARTITION))
    stamps = {
        group_id: (
            buf_b.target_step_list[index],
            buf_b.start_weight_list[index],
        )
        for index, group_id in enumerate(buf_b._group_ids)
    }
    selected: set[str] = set()
    selected_count = 0
    if select_current_train_weight is not None:
        selected_meta, selected_count = await sampler_b.select(
            current_train_weight=select_current_train_weight,
            min_prompt_groups=GROUPS_PER_STEP,
            max_prompt_groups=GROUPS_PER_STEP,
        )
        if selected_meta is not None:
            selected = {
                sample_id.rpartition("_g")[0] for sample_id in selected_meta.sample_ids
            }
    return RoundTrip(
        recovered=recovered,
        ready=ready,
        pending=pending,
        saved_sidecar=sidecar is not None,
        rows_before=rows_before,
        rows_after=rows_after,
        stamps=stamps,
        selected=selected,
        selected_count=selected_count,
    )


def round_trip(
    scenario: Scenario,
    sampler_name: str,
    tmp_path: Path,
    *,
    select_current_train_weight: int | None = None,
) -> RoundTrip:
    """Save the scenario, restore it into a fresh buffer, report what came back."""
    return asyncio.run(
        _round_trip(
            scenario,
            sampler_name,
            tmp_path,
            select_current_train_weight=select_current_train_weight,
        )
    )


def assert_no_data_loss(
    scenario: Scenario, sampler_name: str, tmp_path: Path
) -> RoundTrip:
    """Fail if the restore dropped any group the run still needs.

    The full bar: everything handed out and not yet trained comes back.
    """
    result = round_trip(scenario, sampler_name, tmp_path)
    missing = sorted(scenario.must_survive() - result.recovered)
    assert not missing, (
        f"{sampler_name}/{scenario.name}: restore dropped {missing}. "
        f"These groups were handed out by the dataloader, never trained, and sit "
        f"below the saved cursor ({scenario.cursor}), so nothing will produce them "
        f"again. recovered={sorted(result.recovered)}"
    )
    return result


def assert_completed_groups_survive(
    scenario: Scenario, sampler_name: str, tmp_path: Path
) -> RoundTrip:
    """Fail if #3480 loses a completed, unconsumed prompt group."""
    result = round_trip(scenario, sampler_name, tmp_path)
    lost = sorted(scenario.committed_outstanding() - result.recovered)
    assert not lost, (
        f"{sampler_name}/{scenario.name}: restore dropped completed groups {lost}. "
        "Their rows and replay index should both be present in the #3480 "
        f"checkpoint. recovered={sorted(result.recovered)}"
    )
    return result


# ── the scenarios ───────────────────────────────────────────────────────────
# Numbering follows the worked example: groups 09-11 are trained, 12+ are not.

S_ALL_COMPLETE = Scenario(
    name="lag1-next-step-complete",
    groups=(
        Group(9, 2, weight=0),
        Group(10, 2, weight=0),
        Group(11, 2, weight=0),
        Group(12, 2, weight=1, target=5),
        Group(13, 2, weight=1, target=5),
        Group(14, 2, weight=1, target=5),
    ),
    cursor=15,
    trained=frozenset({9, 10, 11}),
    lag=1,
)

S_ZERO_LAG_ALL_COMPLETE = Scenario(
    name="lag0-current-step-complete",
    groups=(
        Group(9, 2, weight=4, target=4),
        Group(10, 2, weight=4, target=4),
        Group(11, 2, weight=4, target=4),
        Group(12, 2, weight=5, target=5),
        Group(13, 2, weight=5, target=5),
        Group(14, 2, weight=5, target=5),
    ),
    cursor=15,
    trained=frozenset({9, 10, 11}),
    lag=0,
)

S_PARTIAL = Scenario(
    name="lag1-next-step-partly-generated",
    groups=(
        Group(9, 2, weight=0),
        Group(10, 2, weight=0),
        Group(11, 2, weight=0),
        Group(12, 1, weight=1, target=5),  # one rollout still running
        Group(13, 2, weight=1, target=5),
        Group(14, 0, weight=1, target=5),  # not started
    ),
    cursor=15,
    trained=frozenset({9, 10, 11}),
    lag=1,
)

S_LAG2 = Scenario(
    name="lag2-two-batches-in-flight",
    groups=(
        Group(9, 2, weight=0),
        Group(10, 2, weight=0),
        Group(11, 2, weight=0),
        Group(12, 1, weight=1, target=5),
        Group(13, 2, weight=1, target=5),
        Group(14, 0, weight=1, target=5),
        Group(15, 2, weight=2, target=6),
        Group(16, 1, weight=2, target=6),
        Group(17, 0, weight=2, target=6),
    ),
    cursor=18,
    trained=frozenset({9, 10, 11}),
    lag=2,
)

S_EVICTED = Scenario(
    name="lag1-with-an-evicted-group",
    groups=(
        Group(9, 2, weight=0),
        Group(10, 2, weight=0, evicted=True),  # dropped on purpose: too stale
        Group(11, 2, weight=0),
        Group(12, 1, weight=1, target=5),
        Group(13, 2, weight=1, target=5),
        Group(14, 0, weight=1, target=5),
    ),
    cursor=15,
    trained=frozenset({9, 11}),
    lag=1,
)

S_TRAINED_OUT_OF_ORDER = Scenario(
    name="trained-what-was-ready-leaving-a-hole",
    groups=(
        Group(9, 2, weight=0),
        Group(10, 2, weight=0),
        Group(11, 2, weight=0),
        Group(12, 1, weight=1, target=5),  # skipped: still generating
        Group(13, 2, weight=1, target=5),  # trained ahead of 12
        Group(14, 0, weight=1, target=5),
    ),
    cursor=15,
    trained=frozenset({9, 10, 11, 13}),
    lag=1,
)

S_STALE_ONLY = Scenario(
    name="one-group-far-outside-the-staleness-window",
    groups=(
        Group(9, ROLLOUTS_PER_GROUP, weight=0, target=1),
        Group(10, ROLLOUTS_PER_GROUP, weight=7, target=8),
    ),
    cursor=11,
    trained=frozenset(),
    lag=1,
)

# Everything fully generated -- the case this PR set out to recover.
FULLY_GENERATED = (S_ZERO_LAG_ALL_COMPLETE, S_ALL_COMPLETE, S_STALE_ONLY)
# At least one group still generating when the snapshot was taken.
WITH_IN_FLIGHT = (S_PARTIAL, S_LAG2, S_EVICTED, S_TRAINED_OUT_OF_ORDER)
ALL_SCENARIOS = FULLY_GENERATED + WITH_IN_FLIGHT
