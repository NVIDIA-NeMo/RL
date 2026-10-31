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

"""Gym's environment checkpoint: retire, prepare, commit, drain, validate and restore checks."""

from __future__ import annotations

import asyncio
import contextlib
import warnings
from collections.abc import AsyncIterator, Awaitable, Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import pytest

from nemo_rl.algorithms.async_utils.replay_buffer import (
    DataPlaneCheckpointBarrier,
    DataPlaneMutationCut,
)
from nemo_rl.algorithms.single_controller import (
    SingleControllerActor,
)
from nemo_rl.environments.checkpointing import build_environment_checkpoint
from nemo_rl.environments.checkpointing.interfaces import NoEnvironmentCheckpoint
from nemo_rl.environments.checkpointing.nemo_gym import (
    GymEnvironmentCheckpoint,
    _summarize_gym_episodes,
)
from nemo_rl.environments.gym_checkpoint_adapter import (
    GymCheckpointEpisode,
)
from nemo_rl.environments.gym_checkpoint_coordinator import (
    GymCheckpointCommitResult,
    GymCheckpointCoordinator,
    GymCheckpointNotReady,
    load_gym_checkpoint_manifest,
)
from nemo_rl.experience.rollout_manager import (
    RequestDeadlineRegistry,
    _Deadline,
)
from nemo_rl.experience.rollout_recovery import (
    PromptGroupRecoveryRecord,
    RecoveryTargetLevel,
    RolloutRecoveryLedger,
)
from tests.unit.environments.gym_checkpoint_fakes import FakeGymActor, gym_coordinator

_ASYNC_TEST_TIMEOUT_S = 10.0


async def _no_staging_keys() -> list[str]:
    return []


def _deadline_pausing_manager(
    registry: Optional[RequestDeadlineRegistry] = None,
) -> SimpleNamespace:
    """Rollout-manager stand-in for the deadline pause a Gym checkpoint takes."""
    registry = registry if registry is not None else RequestDeadlineRegistry()
    return SimpleNamespace(
        suspend_request_deadlines=registry.suspend,
        resume_request_deadlines=registry.resume,
    )


def _gym_checkpoint(
    coordinator: GymCheckpointCoordinator,
    ledger: Optional[RolloutRecoveryLedger] = None,
    *,
    manager: Optional[Any] = None,
    list_staging_keys: Callable[[], Awaitable[list[str]]] = _no_staging_keys,
    allow_restart_only_servers: bool = False,
) -> GymEnvironmentCheckpoint:
    """Gym's checkpoint over ``coordinator``, with a stand-in rollout manager."""
    return GymEnvironmentCheckpoint(
        coordinator,
        ledger=ledger if ledger is not None else RolloutRecoveryLedger(),
        rollout_manager=(
            manager if manager is not None else _deadline_pausing_manager()
        ),
        list_staging_keys=list_staging_keys,
        allow_restart_only_servers=allow_restart_only_servers,
    )


@contextlib.asynccontextmanager
async def _gym_commit(
    gym: GymEnvironmentCheckpoint, checkpoint_path: Path, *, checkpoint_id: str
) -> AsyncIterator[GymCheckpointCommitResult]:
    """Gym's commit for one snapshot, while Gym is parked."""
    async with gym.snapshot(checkpoint_path, checkpoint_id=checkpoint_id) as snapshot:
        yield snapshot.commit


def test_turn_checkpoint_drains_on_wire_reply_before_tq_cut(tmp_path: Path) -> None:
    async def exercise() -> None:
        ledger = RolloutRecoveryLedger()
        barrier = DataPlaneCheckpointBarrier()
        async with barrier.mutation() as cut:
            group = _dispatched_turn_group(ledger, cut, generations=1)

        actor = FakeGymActor(exported_episodes=())
        gym = _gym_checkpoint(gym_coordinator(actor), ledger)

        async def complete_on_wire_reply() -> None:
            while "commit" not in actor.operations():
                await asyncio.sleep(0)
            async with barrier.mutation("sibling_seals") as cut:
                ledger.mark_sibling_sealed(
                    cut,
                    group.group_id,
                    generation_index=0,
                    gate_rollout_id=group.gate_rollout_id(0),
                    receipt={
                        "capture_key": group.gate_rollout_id(0),
                        "manifest": [{"staging_key": "on-wire/call-0"}],
                    },
                    reward=1.0,
                    mask_sample=False,
                )

        completion = asyncio.create_task(complete_on_wire_reply())
        async with _gym_commit(
            gym,
            tmp_path,
            checkpoint_id="save-1",
        ) as gym_commit:
            assert gym_commit is not None
            assert gym_commit.manifest.instances == {"tools/replica-0": ()}
            async with barrier.checkpoint() as cut:
                await gym.validate_cut(cut, gym_commit, checkpoint_path=tmp_path)
                assert ledger.expected_staging_keys() == {"on-wire/call-0"}
                actor.mark("tq-cut")
        await asyncio.wait_for(completion, _ASYNC_TEST_TIMEOUT_S)

        assert actor.operations() == ["prepare", "renew", "commit", "tq-cut", "resume"]
        (commit_call,) = [call for call in actor.calls if call[0] == "commit"]
        assert commit_call[1] == ("save-1", str(tmp_path), (group.gym_episode(0),))

    asyncio.run(exercise())


def test_completion_callback_can_run_while_gym_commit_is_pending(
    tmp_path: Path,
) -> None:
    async def exercise() -> None:
        ledger = RolloutRecoveryLedger()
        barrier = DataPlaneCheckpointBarrier()
        async with barrier.mutation() as cut:
            group = _dispatched_turn_group(ledger, cut, generations=1)

        actor = FakeGymActor(exported_episodes=(), block_commit=True)
        gym = _gym_checkpoint(gym_coordinator(actor), ledger)

        async def checkpoint() -> GymCheckpointCommitResult:
            async with _gym_commit(
                gym,
                tmp_path,
                checkpoint_id="save-1",
            ) as gym_commit:
                assert gym_commit is not None
                return gym_commit

        checkpoint_task = asyncio.create_task(checkpoint())
        await asyncio.wait_for(
            actor.commit_started.wait(),
            timeout=_ASYNC_TEST_TIMEOUT_S,
        )

        async def complete_rollout() -> None:
            async with barrier.mutation("sibling_seals") as cut:
                ledger.mark_sibling_sealed(
                    cut,
                    group.group_id,
                    generation_index=0,
                    gate_rollout_id=group.gate_rollout_id(0),
                    receipt={
                        "capture_key": group.gate_rollout_id(0),
                        "manifest": [{"staging_key": "on-wire/call-0"}],
                    },
                    reward=1.0,
                    mask_sample=False,
                )

        await asyncio.wait_for(
            complete_rollout(),
            timeout=_ASYNC_TEST_TIMEOUT_S,
        )

        actor.release_commit.set()
        commit = await asyncio.wait_for(
            checkpoint_task,
            timeout=_ASYNC_TEST_TIMEOUT_S,
        )

        assert commit.manifest.instances == {"tools/replica-0": ()}
        assert ledger.expected_staging_keys() == {"on-wire/call-0"}
        assert actor.operations()[-1] == "resume"

    asyncio.run(exercise())


def test_turn_checkpoint_aborts_when_candidate_is_neither_exported_nor_drained(
    tmp_path: Path,
) -> None:
    async def exercise() -> None:
        ledger = RolloutRecoveryLedger()
        async with DataPlaneCheckpointBarrier().mutation() as cut:
            group = _dispatched_turn_group(ledger, cut, generations=1)

        actor = FakeGymActor(exported_episodes=())
        gym = _gym_checkpoint(gym_coordinator(actor, control_timeout_s=0.01), ledger)

        with pytest.raises(TimeoutError, match="neither|classify every candidate"):
            async with _gym_commit(
                gym,
                tmp_path,
                checkpoint_id="save-1",
            ):
                raise AssertionError("unreachable")

        assert actor.operations()[-1] == "resume"

    asyncio.run(exercise())


def test_turn_checkpoint_fails_fast_on_parked_session_without_an_owner(
    tmp_path: Path,
) -> None:
    """A session Gym parks but cannot export never drains; do not wait for it."""

    async def exercise() -> None:
        ledger = RolloutRecoveryLedger()
        async with DataPlaneCheckpointBarrier().mutation() as cut:
            group = _dispatched_turn_group(ledger, cut, generations=1)

        (stranded,) = ledger.gym_checkpoint_inventory(frozenset({"tools/replica-0"}))[
            "tools/replica-0"
        ]
        actor = FakeGymActor(exported_episodes=(), unowned_live_episodes=(stranded,))
        gym = _gym_checkpoint(gym_coordinator(actor, control_timeout_s=60.0), ledger)

        async def checkpoint() -> None:
            async with _gym_commit(
                gym,
                tmp_path,
                checkpoint_id="save-1",
            ):
                raise AssertionError("unreachable")

        with pytest.raises(RuntimeError, match="parked agent sessions"):
            await asyncio.wait_for(checkpoint(), timeout=5.0)

        assert actor.operations()[-1] == "resume"

    asyncio.run(exercise())


def _dispatched_turn_group(
    ledger: RolloutRecoveryLedger,
    cut: DataPlaneMutationCut,
    *,
    generations: int = 2,
) -> PromptGroupRecoveryRecord:
    """An admitted turn-level group, dispatched to ``tools/replica-0``."""
    group = ledger.reserve_group(
        cut,
        group_id="batch-7-prompt-0",
        admission_id="batch-7",
        prompt_id="70",
        prompt_payload={"idx": 70, "message_log": []},
        expected_generations=generations,
        target_step=7,
        start_weight_version=7,
        restore_level=RecoveryTargetLevel.TURN,
        admitted=True,
    )
    ledger.mark_group_dispatched(cut, group.group_id, gym_instance_id="tools/replica-0")
    return ledger.get_group(group.group_id)


def _gym_episodes(group: PromptGroupRecoveryRecord) -> set[GymCheckpointEpisode]:
    return {group.gym_episode(sibling.generation_index) for sibling in group.siblings}


def test_turn_checkpoint_retires_abandoned_attempts_gym_is_still_running(
    tmp_path: Path,
) -> None:
    """A cancelled group's /run keeps running in Gym; RL must retire it.

    Abandoning drops the attempts from RL's inventory, but nothing stops the
    episodes Gym already admitted. Unretired, Gym exports them at the next
    commit and every checkpoint fails until they finish on their own.
    """

    async def exercise() -> None:
        barrier = DataPlaneCheckpointBarrier()
        ledger = RolloutRecoveryLedger()
        async with barrier.mutation() as cut:
            group = _dispatched_turn_group(ledger, cut)
        abandoned = _gym_episodes(group)
        # The group task was cancelled after Ray accepted its submission.
        async with barrier.mutation() as cut:
            ledger.abandon_unsealed(cut, group.group_id)

        actor = FakeGymActor(live=abandoned)
        gym = _gym_checkpoint(gym_coordinator(actor), ledger)

        async with _gym_commit(
            gym,
            tmp_path,
            checkpoint_id="save-1",
        ) as gym_commit:
            assert gym_commit is not None
            assert gym_commit.manifest.instances == {"tools/replica-0": ()}

        # Retired, then forgotten, while Gym is idle, before prepare parks anything.
        assert actor.operations() == [
            "retire",
            "forget",
            "prepare",
            "renew",
            "commit",
            "resume",
        ]
        assert frozenset(actor.calls[0][1][1]) == frozenset(abandoned)
        assert actor.calls[1][1][1] == tuple(
            sorted(episode.rollout_id for episode in abandoned)
        )
        assert actor.live == set()
        assert ledger.gym_checkpoint_forgets({"tools/replica-0"}) == {
            "tools/replica-0": ()
        }

    asyncio.run(exercise())


def test_turn_checkpoint_retires_each_dropped_attempt_only_once(
    tmp_path: Path,
) -> None:
    """A retired attempt leaves the queue, so the next checkpoint skips it."""

    async def exercise() -> None:
        barrier = DataPlaneCheckpointBarrier()
        ledger = RolloutRecoveryLedger()
        async with barrier.mutation() as cut:
            group = _dispatched_turn_group(ledger, cut)
        async with barrier.mutation() as cut:
            ledger.abandon_unsealed(cut, group.group_id)

        actor = FakeGymActor(live=_gym_episodes(group))
        gym = _gym_checkpoint(gym_coordinator(actor), ledger)
        for checkpoint_id in ("save-1", "save-2"):
            (tmp_path / checkpoint_id).mkdir()
            async with _gym_commit(
                gym,
                tmp_path / checkpoint_id,
                checkpoint_id=checkpoint_id,
            ):
                pass

        assert ledger.gym_checkpoint_retirements({"tools/replica-0"}) == {
            "tools/replica-0": ()
        }
        retires = [call for call in actor.calls if call[0] == "retire"]
        assert [call[1][0] for call in retires] == ["save-1"]

    asyncio.run(exercise())


@pytest.mark.parametrize("forget_failures", [1, 2])
def test_turn_checkpoint_retires_again_and_forgets_a_rollout_gym_refused_to_forget(
    tmp_path: Path, forget_failures: int
) -> None:
    """A refused forget gets the retire again, then the forget, as Gym asks.

    A deployment that still refuses keeps the rollouts queued for the next
    snapshot: a mark costs one entry, so the snapshot itself goes ahead.
    """

    async def exercise() -> None:
        barrier = DataPlaneCheckpointBarrier()
        ledger = RolloutRecoveryLedger()
        async with barrier.mutation() as cut:
            group = _dispatched_turn_group(ledger, cut)
        abandoned = _gym_episodes(group)
        async with barrier.mutation() as cut:
            ledger.abandon_unsealed(cut, group.group_id)

        actor = FakeGymActor(
            live=abandoned,
            forget_errors=(RuntimeError("retire_incomplete"),) * forget_failures,
        )
        gym = _gym_checkpoint(gym_coordinator(actor), ledger)

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            async with _gym_commit(gym, tmp_path, checkpoint_id="save-1") as gym_commit:
                assert gym_commit is not None

        operations = actor.operations()
        assert operations[:4] == ["retire", "forget", "retire", "forget"]
        assert operations[4] == "prepare"
        assert frozenset(actor.calls[2][1][1]) == frozenset(abandoned)
        pending = ledger.gym_checkpoint_forgets({"tools/replica-0"})
        forgot_warnings = [w for w in caught if "could not forget" in str(w.message)]
        if forget_failures == 1:
            assert pending == {"tools/replica-0": ()}
            assert not forgot_warnings
        else:
            assert set(pending["tools/replica-0"]) == abandoned
            assert forgot_warnings

    asyncio.run(exercise())


def test_turn_checkpoint_retires_only_the_abandoned_attempts_of_a_retried_group(
    tmp_path: Path,
) -> None:
    """The retry's fresh attempts stay live and are committed normally."""

    async def exercise() -> None:
        barrier = DataPlaneCheckpointBarrier()
        ledger = RolloutRecoveryLedger()
        async with barrier.mutation() as cut:
            group = _dispatched_turn_group(ledger, cut)
        abandoned = _gym_episodes(group)
        async with barrier.mutation() as cut:
            ledger.abandon_unsealed(cut, group.group_id)
            ledger.prepare_incomplete_retry(cut, group.group_id)
            ledger.mark_group_dispatched(
                cut, group.group_id, gym_instance_id="tools/replica-0"
            )
        replacements = _gym_episodes(ledger.get_group(group.group_id))
        assert replacements.isdisjoint(abandoned)

        actor = FakeGymActor(live=abandoned | replacements)
        gym = _gym_checkpoint(gym_coordinator(actor), ledger)

        async with _gym_commit(
            gym,
            tmp_path,
            checkpoint_id="save-1",
        ) as gym_commit:
            assert gym_commit is not None
            assert set(gym_commit.manifest.instances["tools/replica-0"]) == (
                replacements
            )

        assert actor.calls[0][0] == "retire"
        assert frozenset(actor.calls[0][1][1]) == frozenset(abandoned)
        assert actor.live == replacements

    asyncio.run(exercise())


def test_turn_checkpoint_leaves_gym_restarts_out_and_records_them(
    tmp_path: Path,
) -> None:
    """Gym refuses a commit that names a restart; the snapshot records it instead.

    The restart is still running in Gym, so the drain must not wait for it and
    the cut must accept RL still owning it.
    """

    async def exercise() -> None:
        barrier = DataPlaneCheckpointBarrier()
        ledger = RolloutRecoveryLedger()
        async with barrier.mutation() as cut:
            group = _dispatched_turn_group(ledger, cut)
        saved, restart = sorted(_gym_episodes(group))
        actor = FakeGymActor(live={saved, restart}, restarts=(restart,))
        gym = _gym_checkpoint(gym_coordinator(actor), ledger)

        async with _gym_commit(
            gym,
            tmp_path,
            checkpoint_id="save-1",
        ) as gym_commit:
            assert gym_commit is not None
            async with barrier.checkpoint() as cut:
                await gym.validate_cut(cut, gym_commit, checkpoint_path=tmp_path)

        (commit,) = [call for call in actor.calls if call[0] == "commit"]
        assert commit[1][2] == (saved,)
        manifest = load_gym_checkpoint_manifest(tmp_path)
        assert manifest.instances == {"tools/replica-0": (saved,)}
        assert manifest.restarts == {"tools/replica-0": (restart,)}
        assert actor.operations()[-1] == "resume"

    asyncio.run(exercise())


def test_turn_checkpoint_rejects_missing_gym_staging_key(tmp_path: Path) -> None:
    async def exercise() -> None:
        ledger = RolloutRecoveryLedger()
        barrier = DataPlaneCheckpointBarrier()
        async with barrier.mutation() as cut:
            group = _dispatched_turn_group(ledger, cut, generations=1)

        episode = group.gym_episode(0)
        actor = FakeGymActor(staging_keys_by_episode={episode: ("missing-call-row",)})
        # TQ holds no staging rows, so the one Gym references is missing.
        gym = _gym_checkpoint(gym_coordinator(actor), ledger)

        with pytest.raises(RuntimeError, match="staging rows missing"):
            async with _gym_commit(
                gym,
                tmp_path,
                checkpoint_id="save-1",
            ) as gym_commit:
                assert gym_commit is not None
                async with barrier.checkpoint() as cut:
                    await gym.validate_cut(cut, gym_commit, checkpoint_path=tmp_path)

        assert actor.operations()[-1] == "resume"

    asyncio.run(exercise())


def test_turn_checkpoint_commits_owned_gym_episode_before_tq_cut(
    tmp_path: Path,
) -> None:
    async def exercise() -> None:
        ledger = RolloutRecoveryLedger()
        async with DataPlaneCheckpointBarrier().mutation() as cut:
            group = _dispatched_turn_group(ledger, cut, generations=1)

        actor = FakeGymActor()
        gym = _gym_checkpoint(gym_coordinator(actor), ledger)

        async with _gym_commit(
            gym,
            tmp_path,
            checkpoint_id="save-1",
        ):
            async with DataPlaneCheckpointBarrier().checkpoint():
                actor.mark("tq-cut")

        assert actor.operations() == ["prepare", "renew", "commit", "tq-cut", "resume"]
        (commit_call,) = [call for call in actor.calls if call[0] == "commit"]
        assert commit_call[1] == ("save-1", str(tmp_path), (group.gym_episode(0),))

    asyncio.run(exercise())


class TestGymCheckpointPausesRolloutDeadlines:
    """Parked rollouts cannot progress, so a checkpoint must not use up their budget."""

    def test_a_parked_rollout_outlives_a_checkpoint_longer_than_its_budget(
        self, tmp_path: Path
    ) -> None:
        registry = RequestDeadlineRegistry()
        gym = _gym_checkpoint(
            gym_coordinator(FakeGymActor(exported_episodes=())),
            manager=_deadline_pausing_manager(registry),
        )

        async def exercise() -> None:
            entered = asyncio.Event()
            checkpoint_done = asyncio.Event()

            async def parked_rollout() -> None:
                async with _Deadline(0.2, "NeMo-Gym prompt group", registry=registry):
                    entered.set()
                    await checkpoint_done.wait()
                    await asyncio.sleep(0.05)

            rollout = asyncio.create_task(parked_rollout())
            await entered.wait()
            async with _gym_commit(gym, tmp_path, checkpoint_id="save-1"):
                assert registry.suspended
                # Twice the rollout's whole budget.
                await asyncio.sleep(0.4)
            checkpoint_done.set()
            # Raises RolloutTimeout if the checkpoint was charged to the rollout.
            await rollout
            assert not registry.suspended

        asyncio.run(exercise())

    def test_ending_a_checkpoint_keeps_a_generation_stand_down_pause(
        self, tmp_path: Path
    ) -> None:
        registry = RequestDeadlineRegistry()
        gym = _gym_checkpoint(
            gym_coordinator(FakeGymActor(exported_episodes=())),
            manager=_deadline_pausing_manager(registry),
        )

        async def exercise() -> None:
            # The colocated engine is training; its own pause must outlast ours.
            registry.suspend()
            async with _gym_commit(gym, tmp_path, checkpoint_id="save-1"):
                pass
            assert registry.suspended
            registry.resume()
            assert not registry.suspended

        asyncio.run(exercise())

    def test_a_checkpoint_that_cannot_park_releases_its_pause(
        self, tmp_path: Path
    ) -> None:
        registry = RequestDeadlineRegistry()
        gym = _gym_checkpoint(
            gym_coordinator(FakeGymActor(prepared=False)),
            manager=_deadline_pausing_manager(registry),
        )

        async def exercise() -> None:
            with pytest.raises(GymCheckpointNotReady):
                async with _gym_commit(gym, tmp_path, checkpoint_id="save-1"):
                    raise AssertionError("unreachable")
            assert not registry.suspended

        asyncio.run(exercise())


class TestDrainScalesWithPendingRepliesNotTheLedger:
    """The drain waits on a few on-wire replies; its cost must not track every rollout.

    At hundreds of thousands of live rollouts, rescanning the whole ledger on
    each check blocks the event loop that delivers the very replies it waits
    for, with every rollout parked behind the checkpoint.
    """

    @staticmethod
    async def _dispatched_groups(
        ledger: RolloutRecoveryLedger,
        barrier: DataPlaneCheckpointBarrier,
        count: int,
    ) -> list[PromptGroupRecoveryRecord]:
        groups = []
        async with barrier.mutation() as cut:
            for index in range(count):
                group = ledger.reserve_group(
                    cut,
                    group_id=f"batch-7-prompt-{index}",
                    admission_id="batch-7",
                    prompt_id=str(index),
                    prompt_payload={"idx": index, "message_log": []},
                    expected_generations=1,
                    target_step=7,
                    start_weight_version=7,
                    restore_level=RecoveryTargetLevel.TURN,
                    admitted=True,
                )
                ledger.mark_group_dispatched(
                    cut, group.group_id, gym_instance_id="tools/replica-0"
                )
                groups.append(ledger.get_group(group.group_id))
        return groups

    @staticmethod
    async def _drain_setup(
        tmp_path: Path,
        *,
        rollouts: int,
        on_wire: int,
        control_timeout_s: float = 5.0,
    ) -> SimpleNamespace:
        """Commit a cut in which Gym exported all but the first ``on_wire`` episodes."""
        ledger = RolloutRecoveryLedger()
        barrier = DataPlaneCheckpointBarrier()
        groups = await TestDrainScalesWithPendingRepliesNotTheLedger._dispatched_groups(
            ledger, barrier, rollouts
        )
        episodes = [group.gym_episode(0) for group in groups]
        coordinator = gym_coordinator(
            FakeGymActor(exported_episodes=tuple(episodes[on_wire:])),
            control_timeout_s=control_timeout_s,
        )
        candidates = {"tools/replica-0": tuple(episodes)}
        commit = await coordinator.commit(
            "save-1", tmp_path, candidates, restarts={"tools/replica-0": ()}
        )
        gym = _gym_checkpoint(coordinator, ledger)
        return SimpleNamespace(
            ledger=ledger,
            barrier=barrier,
            groups=groups,
            drain=lambda: gym._drain_non_exported_candidates(candidates, commit),
        )

    @staticmethod
    async def _resolve(setup: SimpleNamespace, index: int, how: str) -> None:
        group = setup.groups[index]
        async with setup.barrier.mutation("sibling_seals") as cut:
            if how == "sealed":
                setup.ledger.mark_sibling_sealed(
                    cut,
                    group.group_id,
                    generation_index=0,
                    gate_rollout_id=group.gate_rollout_id(0),
                    receipt=None,
                    reward=1.0,
                    mask_sample=False,
                )
            else:
                setup.ledger.abandon_unsealed(cut, group.group_id)

    def test_waiting_never_rescans_every_rollout(self, tmp_path: Path) -> None:
        async def exercise() -> None:
            setup = await self._drain_setup(tmp_path, rollouts=50, on_wire=1)
            scans = 0
            full_scan = setup.ledger.gym_checkpoint_inventory

            def counting_scan(instance_ids):
                nonlocal scans
                scans += 1
                return full_scan(instance_ids)

            setup.ledger.gym_checkpoint_inventory = counting_scan
            drain = asyncio.create_task(setup.drain())
            # Long enough for a 50 ms poll to have rescanned several times.
            await asyncio.sleep(0.3)
            await self._resolve(setup, 0, "sealed")
            await asyncio.wait_for(drain, timeout=5.0)

            assert scans == 0

        asyncio.run(exercise())

    @pytest.mark.parametrize("how", ["sealed", "abandoned"])
    def test_it_returns_once_the_last_reply_resolves(
        self, tmp_path: Path, how: str
    ) -> None:
        async def exercise() -> None:
            setup = await self._drain_setup(tmp_path, rollouts=3, on_wire=2)
            drain = asyncio.create_task(setup.drain())
            await asyncio.sleep(0.1)
            await self._resolve(setup, 0, how)
            # Several polls' worth, so an early return would have happened.
            await asyncio.sleep(0.2)
            assert not drain.done(), "one on-wire reply is still outstanding"

            await self._resolve(setup, 1, how)
            await asyncio.wait_for(drain, timeout=1.0)

        asyncio.run(exercise())

    def test_a_timeout_names_the_unresolved_replies(self, tmp_path: Path) -> None:
        async def exercise() -> None:
            setup = await self._drain_setup(
                tmp_path, rollouts=2, on_wire=2, control_timeout_s=0.05
            )
            with pytest.raises(TimeoutError, match=r"tools/replica-0: 2 episode\(s\)"):
                await setup.drain()

        asyncio.run(exercise())

    def test_the_summary_counts_every_episode_but_names_only_a_few(self) -> None:
        episodes = [
            GymCheckpointEpisode(f"rollout-{index:02d}", 0) for index in range(12)
        ]

        summary = _summarize_gym_episodes(
            {"tools/replica-1": episodes, "tools/replica-0": ()}
        )

        assert summary.startswith("tools/replica-1: 12 episode(s), e.g. ")
        assert "rollout-04" in summary and "rollout-05" not in summary
        assert summary.endswith(" and 7 more")
        assert "replica-0" not in summary
        assert _summarize_gym_episodes({"tools/replica-0": ()}) == "none"


class TestARowDroppedDuringACheckpointDoesNotFailIt:
    """A broken stream can drop rows while Gym is prepared; the checkpoint must hold.

    The drop marks the rows abandoned at once, but only the next checkpoint
    retires them, so Gym still holds and exports them. Restore requires the
    saved Gym manifest to equal RL's inventory, so the saved checkpoint must
    leave the dropped episodes out, and they stay queued for that retire.
    """

    @staticmethod
    async def _drop(
        ledger: RolloutRecoveryLedger,
        barrier: DataPlaneCheckpointBarrier,
        group_id: str,
        generation_indices: list[int],
    ) -> None:
        # What _replace_unreturned_rows records before the rows are re-sent.
        async with barrier.mutation() as cut:
            ledger.replace_dropped_dispatch(
                cut, group_id, generation_indices=generation_indices
            )

    @staticmethod
    async def _save(
        gym: Any, barrier: DataPlaneCheckpointBarrier, tmp_path: Path
    ) -> None:
        async with _gym_commit(gym, tmp_path, checkpoint_id="save-1") as gym_commit:
            assert gym_commit is not None
            async with barrier.checkpoint() as cut:
                await gym.validate_cut(cut, gym_commit, checkpoint_path=tmp_path)

    @staticmethod
    def _assert_restorable(
        gym: Any,
        coordinator: GymCheckpointCoordinator,
        tmp_path: Path,
        *,
        kept: set[GymCheckpointEpisode],
    ) -> None:
        saved = load_gym_checkpoint_manifest(tmp_path).instances
        assert saved == gym.inventory()
        assert set(saved["tools/replica-0"]) == kept

    def test_a_drop_after_the_pre_prepare_retire(self, tmp_path: Path) -> None:
        """Gym exports the dropped row it still holds, outside RL's candidates."""

        async def exercise() -> None:
            barrier = DataPlaneCheckpointBarrier()
            ledger = RolloutRecoveryLedger()
            async with barrier.mutation() as cut:
                group = _dispatched_turn_group(ledger, cut)
            row_0, row_1 = (group.gym_episode(index) for index in (0, 1))

            coordinator = gym_coordinator(
                FakeGymActor(
                    live={row_0, row_1},
                    while_prepared=lambda: self._drop(
                        ledger, barrier, group.group_id, [1]
                    ),
                )
            )
            gym = _gym_checkpoint(coordinator, ledger)

            await asyncio.wait_for(
                self._save(gym, barrier, tmp_path), _ASYNC_TEST_TIMEOUT_S
            )

            self._assert_restorable(gym, coordinator, tmp_path, kept={row_0})
            assert ledger.gym_checkpoint_retirements({"tools/replica-0"}) == {
                "tools/replica-0": (row_1,)
            }

        asyncio.run(exercise())

    def test_a_drop_while_the_commit_drains(self, tmp_path: Path) -> None:
        """A failed on-wire reply ends the stream, dropping a row Gym exported."""

        async def exercise() -> None:
            barrier = DataPlaneCheckpointBarrier()
            ledger = RolloutRecoveryLedger()
            async with barrier.mutation() as cut:
                group = _dispatched_turn_group(ledger, cut)
            row_0, row_1 = (group.gym_episode(index) for index in (0, 1))
            # Row 0 is parked in Gym and exported; row 1's reply is on the wire.
            actor = FakeGymActor(live={row_0})
            coordinator = gym_coordinator(actor)
            gym = _gym_checkpoint(coordinator, ledger)

            async def fail_the_on_wire_reply() -> None:
                while "commit" not in actor.operations():
                    await asyncio.sleep(0)
                await self._drop(ledger, barrier, group.group_id, [0, 1])

            failing = asyncio.create_task(fail_the_on_wire_reply())
            await asyncio.wait_for(
                self._save(gym, barrier, tmp_path), _ASYNC_TEST_TIMEOUT_S
            )
            await asyncio.wait_for(failing, _ASYNC_TEST_TIMEOUT_S)

            self._assert_restorable(gym, coordinator, tmp_path, kept=set())
            assert ledger.gym_checkpoint_retirements({"tools/replica-0"}) == {
                "tools/replica-0": tuple(sorted((row_0, row_1)))
            }

        asyncio.run(exercise())

    def test_a_dropped_episode_takes_its_staging_keys_with_it(
        self, tmp_path: Path
    ) -> None:
        """The cut needs, and the saved manifest keeps, only the kept episodes' rows."""

        async def exercise() -> None:
            barrier = DataPlaneCheckpointBarrier()
            ledger = RolloutRecoveryLedger()
            async with barrier.mutation() as cut:
                group = _dispatched_turn_group(ledger, cut)
            row_0, row_1 = (group.gym_episode(index) for index in (0, 1))
            key_0 = f"{row_0.rollout_id}/call-0"
            key_1 = f"{row_1.rollout_id}/call-0"
            coordinator = gym_coordinator(
                FakeGymActor(
                    live={row_0, row_1},
                    while_prepared=lambda: self._drop(
                        ledger, barrier, group.group_id, [1]
                    ),
                    staging_keys_by_episode={row_0: (key_0,), row_1: (key_1,)},
                )
            )

            async def list_staging_keys() -> list[str]:
                # The dropped row's staging is gone by the time the cut looks.
                return [key_0]

            gym = _gym_checkpoint(
                coordinator, ledger, list_staging_keys=list_staging_keys
            )

            await asyncio.wait_for(
                self._save(gym, barrier, tmp_path), _ASYNC_TEST_TIMEOUT_S
            )

            saved = load_gym_checkpoint_manifest(tmp_path)
            assert set(saved.instances["tools/replica-0"]) == {row_0}
            assert saved.staging_keys == {"tools/replica-0": (key_0,)}

        asyncio.run(exercise())

    def test_only_a_queued_drop_may_differ_from_the_saved_manifest(
        self, tmp_path: Path
    ) -> None:
        """An exported episode RL dropped without queuing a retire still fails."""

        async def exercise() -> None:
            barrier = DataPlaneCheckpointBarrier()
            ledger = RolloutRecoveryLedger()
            async with barrier.mutation() as cut:
                group = _dispatched_turn_group(ledger, cut)
            row_0, row_1 = (group.gym_episode(index) for index in (0, 1))
            coordinator = gym_coordinator(FakeGymActor(live={row_0, row_1}))
            gym = _gym_checkpoint(coordinator, ledger)

            async with _gym_commit(gym, tmp_path, checkpoint_id="save-1") as gym_commit:
                assert gym_commit is not None
                await self._drop(ledger, barrier, group.group_id, [0])
                # Take the drop off the retire queue, as if RL never queued it.
                ledger.mark_gym_retired({"tools/replica-0": [row_0]})
                async with barrier.checkpoint() as cut:
                    with pytest.raises(RuntimeError, match="ownership changed"):
                        await gym.validate_cut(
                            cut, gym_commit, checkpoint_path=tmp_path
                        )

        asyncio.run(exercise())


def _restart_only_check(
    restart_only_servers: tuple[str, ...], *, allowed: bool
) -> tuple[Any, FakeGymActor]:
    actor = FakeGymActor(restart_only_servers=restart_only_servers)
    gym = _gym_checkpoint(gym_coordinator(actor), allow_restart_only_servers=allowed)
    return gym, actor


def test_turn_recovery_refuses_a_restart_only_gym_server_by_default() -> None:
    """Its rollouts would start over after a crash without anyone being told."""
    gym, actor = _restart_only_check(("tools_resources",), allowed=False)

    with pytest.raises(ValueError, match="restart-only Gym servers") as raised:
        asyncio.run(gym.check_startup())

    assert "tools/replica-0" in str(raised.value)
    assert "tools_resources" in str(raised.value)
    assert "allow_restart_only_gym_servers=true" in str(raised.value)
    assert actor.operations() == ["status"]


def test_an_allowed_restart_only_gym_server_is_named_in_a_warning() -> None:
    gym, _ = _restart_only_check(("tools_agent",), allowed=True)

    with pytest.warns(UserWarning, match="start over after a crash.*tools_agent"):
        asyncio.run(gym.check_startup())


def test_allowing_restart_only_gym_servers_warns_when_there_are_none() -> None:
    gym, _ = _restart_only_check((), allowed=True)

    with pytest.warns(UserWarning, match="no Gym server is restart-only"):
        asyncio.run(gym.check_startup())


def test_no_restart_only_gym_server_and_no_opt_in_passes_silently() -> None:
    gym, _ = _restart_only_check((), allowed=False)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        asyncio.run(gym.check_startup())


def test_without_turn_recovery_there_are_no_gym_servers_to_check() -> None:
    gym = object.__new__(SingleControllerActor.__ray_metadata__.modified_class)
    gym = NoEnvironmentCheckpoint()

    asyncio.run(gym.check_startup())


class _GateRecordingManager:
    def __init__(self) -> None:
        self.recovery_ledger = RolloutRecoveryLedger()
        self.gates: list[Any] = []

    def set_rollout_dispatch_admission_gate(self, gate: Any) -> None:
        self.gates.append(gate)


class _StagingClient:
    def __init__(self) -> None:
        self.partitions: list[str] = []

    def list_sample_ids(self, *, partition_id: str) -> list[str]:
        self.partitions.append(partition_id)
        return ["staged-0"]


def _builder_config(*, turn: bool) -> SimpleNamespace:
    return SimpleNamespace(
        rollout_recovery=SimpleNamespace(
            turn_checkpointing_enabled=turn,
            gym_prepare_timeout_s=30.0,
            allow_restart_only_gym_servers=True,
        ),
        token_capture=SimpleNamespace(
            staging_partition="rollout_staging", control_timeout_s=5.0
        ),
    )


def test_build_keeps_no_environment_state_without_turn_recovery() -> None:
    manager = _GateRecordingManager()

    built = build_environment_checkpoint(
        _builder_config(turn=False),  # type: ignore[arg-type]
        env_handles={"nemo_gym": FakeGymActor()},
        rollout_manager=manager,  # type: ignore[arg-type]
        dp_client=_StagingClient(),
    )

    assert isinstance(built, NoEnvironmentCheckpoint)
    assert manager.gates == []


def test_build_wires_gym_for_turn_recovery() -> None:
    """Gym closes dispatch through the gate it installs and reads staged rows."""
    manager = _GateRecordingManager()
    client = _StagingClient()

    built = build_environment_checkpoint(
        _builder_config(turn=True),  # type: ignore[arg-type]
        env_handles={"nemo_gym": FakeGymActor()},
        rollout_manager=manager,  # type: ignore[arg-type]
        dp_client=client,
    )

    assert isinstance(built, GymEnvironmentCheckpoint)
    assert built.holds_rollout_state
    assert manager.gates == [built.admission_gate]
    assert built._ledger is manager.recovery_ledger
    assert built._allow_restart_only_servers is True
    assert asyncio.run(built._list_staging_keys()) == ["staged-0"]
    assert client.partitions == ["rollout_staging"]
