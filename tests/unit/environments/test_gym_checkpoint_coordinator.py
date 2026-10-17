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
import pickle
import time
from pathlib import Path

import pytest
import ray.exceptions

from nemo_rl.environments.gym_checkpoint_adapter import (
    GymCheckpointCommitSummary,
    GymCheckpointEpisode,
    GymCheckpointUnavailable,
)
from nemo_rl.environments.gym_checkpoint_coordinator import (
    GYM_CHECKPOINT_SCHEMA_VERSION,
    GymCheckpointCoordinator,
    GymCheckpointManifest,
    GymCheckpointNotReady,
    GymCheckpointOperationError,
    is_transient_gym_checkpoint_failure,
    load_gym_checkpoint_manifest,
)
from nemo_rl.environments.nemo_gym import NemoGymShardSet
from tests.unit.environments.gym_checkpoint_fakes import FakeGymActor, gym_coordinator

pytestmark = pytest.mark.nemo_gym


_FakeGymActor = FakeGymActor
_NO_RESTARTS = {"tools/replica-0": ()}
_NO_RESTARTS_TWO = {"tools/replica-0": (), "tools/replica-1": ()}


def _coordinator(*actors: FakeGymActor) -> GymCheckpointCoordinator:
    return gym_coordinator(*actors)


@pytest.mark.parametrize(
    "error",
    [
        GymCheckpointOperationError(
            "prepare", {"tools/replica-0": TimeoutError("no reply")}
        ),
        GymCheckpointNotReady({"tools/replica-0": {"agent": ("rollout-1",)}}),
    ],
    ids=["operation-error", "not-ready"],
)
def test_checkpoint_errors_survive_a_ray_boundary(error: RuntimeError) -> None:
    """Pickled across Ray, an error keeps its type, fields, message and notes."""
    error.add_note("Gym checkpoint cleanup also failed: lost")

    restored = pickle.loads(pickle.dumps(error))

    assert type(restored) is type(error)
    assert str(restored) == str(error)
    assert restored.__notes__ == error.__notes__
    assert is_transient_gym_checkpoint_failure(restored)
    if isinstance(error, GymCheckpointOperationError):
        assert restored.operation == "prepare"
        assert set(restored.failures) == {"tools/replica-0"}
        assert isinstance(restored.failures["tools/replica-0"], TimeoutError)
    else:
        assert restored.blockers == error.blockers


def test_manifest_rejects_duplicate_episode_identity() -> None:
    episode = GymCheckpointEpisode("rollout-0", 2)

    with pytest.raises(ValueError, match="duplicate episodes"):
        GymCheckpointManifest(
            schema_version=GYM_CHECKPOINT_SCHEMA_VERSION,
            checkpoint_id="save-1",
            instances={"tools/replica-0": (episode, episode)},
            staging_keys={"tools/replica-0": ()},
            restarts={"tools/replica-0": ()},
        )


def test_coordinator_commits_and_restores_each_instance_inventory(
    tmp_path: Path,
) -> None:
    first = _FakeGymActor(
        staging_keys_by_episode={
            GymCheckpointEpisode("rollout-0", 0): ("rollout-0/call-0",)
        }
    )
    second = _FakeGymActor()
    coordinator = _coordinator(first, second)
    inventory = {
        "tools/replica-0": (GymCheckpointEpisode("rollout-0", 0),),
        "tools/replica-1": (GymCheckpointEpisode("rollout-1", 2),),
    }

    async def exercise() -> None:
        async with coordinator.prepared("save-1"):
            result = await coordinator.commit(
                "save-1", tmp_path, inventory, restarts=_NO_RESTARTS_TWO
            )
            assert result.manifest.instances == inventory
            assert set(result.instances) == set(inventory)
            assert result.instances["tools/replica-0"].exported_episodes == (
                GymCheckpointEpisode("rollout-0", 0),
            )
            assert result.manifest.staging_keys == {
                "tools/replica-0": ("rollout-0/call-0",),
                "tools/replica-1": (),
            }
            await coordinator.renew("save-1")
        manifest = load_gym_checkpoint_manifest(tmp_path)
        assert manifest.checkpoint_id == "save-1"
        assert manifest.instances == inventory
        assert manifest.staging_keys == {
            "tools/replica-0": ("rollout-0/call-0",),
            "tools/replica-1": (),
        }
        await coordinator.restore("restore-1", tmp_path, manifest)
        await coordinator.retire_restored("restore-1", manifest)
        await coordinator.resume("restore-1")

    asyncio.run(exercise())

    for replica, actor in enumerate((first, second)):
        operations = [call[0] for call in actor.calls]
        assert operations == [
            "prepare",
            "renew",
            "commit",
            "renew",
            "resume",
            "restore",
            "retire",
            "resume",
        ]
        commit_args = actor.calls[2][1]
        assert commit_args[0] == "save-1"
        assert commit_args[2] == inventory[f"tools/replica-{replica}"]
        retire_args = actor.calls[-2][1]
        assert retire_args[0] == "restore-1"
        assert retire_args[1] == tuple(
            episode.next_attempt() for episode in inventory[f"tools/replica-{replica}"]
        )
        assert actor.calls[5][2]["source_checkpoint_id"] == "save-1"


def test_coordinator_resumes_every_actor_when_prepare_is_not_ready() -> None:
    ready = _FakeGymActor()
    blocked = _FakeGymActor(prepared=False)
    coordinator = _coordinator(ready, blocked)

    async def exercise() -> None:
        with pytest.raises(GymCheckpointNotReady, match="rollout-1"):
            async with coordinator.prepared("save-1"):
                raise AssertionError("unreachable")

    asyncio.run(exercise())

    assert [call[0] for call in ready.calls] == ["prepare", "resume"]
    assert [call[0] for call in blocked.calls] == ["prepare", "resume"]


def test_coordinator_renews_lease_while_outer_checkpoint_is_slow() -> None:
    actor = _FakeGymActor()
    coordinator = GymCheckpointCoordinator(
        NemoGymShardSet(handles={"tools": [actor]}),
        control_timeout_s=0.03,
        prepare_timeout_s=0.03,
    )

    async def exercise() -> None:
        async with coordinator.prepared("save-1"):
            await asyncio.sleep(0.04)

    asyncio.run(exercise())

    operations = [call[0] for call in actor.calls]
    assert operations[0:2] == ["prepare", "renew"]
    assert operations.count("renew") >= 2
    assert operations[-1] == "resume"


def test_coordinator_rejects_inventory_for_the_wrong_actor_topology(
    tmp_path: Path,
) -> None:
    coordinator = _coordinator(_FakeGymActor())

    with pytest.raises(ValueError, match="live actor topology"):
        asyncio.run(
            coordinator.commit(
                "save-1",
                tmp_path,
                {"other/replica-0": ()},
                restarts=_NO_RESTARTS,
            )
        )


def test_coordinator_uses_environment_exported_subset_as_manifest(
    tmp_path: Path,
) -> None:
    coordinator = _coordinator(_FakeGymActor(exported_episodes=()))

    result = asyncio.run(
        coordinator.commit(
            "save-1",
            tmp_path,
            {"tools/replica-0": (GymCheckpointEpisode("rollout-on-wire", 0),)},
            restarts=_NO_RESTARTS,
        )
    )

    assert result.manifest.instances == {"tools/replica-0": ()}
    assert load_gym_checkpoint_manifest(tmp_path).instances == {"tools/replica-0": ()}


def test_manifest_keeps_only_the_staging_keys_of_episodes_gym_saved(
    tmp_path: Path,
) -> None:
    """A candidate whose reply is on the wire is not saved, so neither are its rows.

    Gym's model participant reports the staged rows of every candidate in scope.
    An on-wire candidate seals and its group can clean those rows up before the
    cut, so keeping its keys would fail the checkpoint, or a later restore,
    over rows nothing needs. Gym says whose each key is: a cut key names no
    rollout, and a row carried over from an earlier attempt serves both.
    """
    saved = (GymCheckpointEpisode("rollout-0", 0), GymCheckpointEpisode("rollout-1", 2))
    on_wire = GymCheckpointEpisode("rollout-on-wire", 0)
    coordinator = _coordinator(
        _FakeGymActor(
            exported_episodes=saved,
            staging_keys_by_episode={
                saved[0]: ("rollout-0/call-0",),
                saved[1]: ("__generation_cut__/save-1/call-1", "carried/call-0"),
                on_wire: ("carried/call-0", "rollout-on-wire/call-0"),
            },
        )
    )

    result = asyncio.run(
        coordinator.commit(
            "save-1",
            tmp_path,
            {"tools/replica-0": (*saved, on_wire)},
            restarts=_NO_RESTARTS,
        )
    )

    kept = {
        "tools/replica-0": (
            "__generation_cut__/save-1/call-1",
            "carried/call-0",
            "rollout-0/call-0",
        )
    }
    assert result.manifest.staging_keys == kept
    assert load_gym_checkpoint_manifest(tmp_path).staging_keys == kept


def test_a_summary_must_attribute_every_staging_key() -> None:
    """A key no episode owns could be neither kept nor dropped safely."""
    with pytest.raises(ValueError, match="staging_keys_by_episode"):
        GymCheckpointCommitSummary(
            exported_episodes=(),
            staging_keys=("orphan/call-0",),
        )


def test_coordinator_rejects_episode_outside_candidate_set(tmp_path: Path) -> None:
    coordinator = _coordinator(
        _FakeGymActor(
            exported_episodes=(GymCheckpointEpisode("unexpected-rollout", 0),)
        )
    )

    with pytest.raises(RuntimeError, match="outside the candidate set"):
        asyncio.run(
            coordinator.commit(
                "save-1",
                tmp_path,
                {"tools/replica-0": (GymCheckpointEpisode("rollout-on-wire", 0),)},
                restarts=_NO_RESTARTS,
            )
        )


def test_restore_failure_discards_every_actor_under_uncertain_outcome() -> None:
    restored = _FakeGymActor()
    failed = _FakeGymActor(restore_error=RuntimeError("restore failed"))
    coordinator = _coordinator(restored, failed)
    manifest = GymCheckpointManifest(
        schema_version=GYM_CHECKPOINT_SCHEMA_VERSION,
        checkpoint_id="save-1",
        instances={
            "tools/replica-0": (GymCheckpointEpisode("rollout-0", 0),),
            "tools/replica-1": (GymCheckpointEpisode("rollout-1", 0),),
        },
        staging_keys={
            "tools/replica-0": (),
            "tools/replica-1": (),
        },
        restarts=_NO_RESTARTS_TWO,
    )

    with pytest.raises(GymCheckpointOperationError, match="restore failed"):
        asyncio.run(coordinator.restore("restore-1", "/unused", manifest))

    assert [call[0] for call in restored.calls] == ["restore", "retire", "resume"]
    assert [call[0] for call in failed.calls] == ["restore", "retire", "resume"]


# Bound on how long a test waits for the coordinator before calling it hung.
_HANG_GUARD_S = 5.0


def _run_guarded(exercise) -> None:
    async def guarded() -> None:
        try:
            await asyncio.wait_for(exercise(), timeout=_HANG_GUARD_S)
        except TimeoutError:
            pytest.fail(
                f"Gym checkpoint control call still waiting after {_HANG_GUARD_S}s"
            )

    asyncio.run(guarded())


def _fast_coordinator(*actors: FakeGymActor) -> GymCheckpointCoordinator:
    return gym_coordinator(*actors, control_timeout_s=0.05, prepare_timeout_s=0.05)


def test_an_actor_that_never_answers_prepare_fails_the_checkpoint() -> None:
    """Gym only enforces deadline_ts once it runs the call; a stuck actor never does."""
    healthy = _FakeGymActor()
    stuck = _FakeGymActor(hang=frozenset({"prepare"}))
    coordinator = _fast_coordinator(healthy, stuck)

    async def exercise() -> None:
        with pytest.raises(GymCheckpointOperationError) as raised:
            async with coordinator.prepared("save-1"):
                raise AssertionError("unreachable")
        assert set(raised.value.failures) == {"tools/replica-1"}
        assert isinstance(raised.value.failures["tools/replica-1"], TimeoutError)

    _run_guarded(exercise)

    # Admission was closed on the healthy actor, so cleanup must still reopen it.
    assert [call[0] for call in healthy.calls] == ["prepare", "resume"]
    assert [call[0] for call in stuck.calls] == ["prepare", "resume"]


def test_a_renewal_that_never_answers_does_not_hang_the_checkpoint() -> None:
    actor = _FakeGymActor(hang=frozenset({"renew"}))
    coordinator = _fast_coordinator(actor)

    async def exercise() -> None:
        with pytest.raises(GymCheckpointOperationError) as raised:
            async with coordinator.prepared("save-1"):
                await asyncio.sleep(0.2)
        assert raised.value.operation == "renew"
        assert isinstance(raised.value.failures["tools/replica-0"], TimeoutError)

    _run_guarded(exercise)

    assert [call[0] for call in actor.calls][-1] == "resume"


def test_a_resume_that_never_answers_keeps_the_original_failure() -> None:
    """Cleanup retries once, then gives up: Gym's lease reopens admission."""
    actor = _FakeGymActor(prepared=False, hang=frozenset({"resume"}))
    coordinator = _fast_coordinator(actor)

    async def exercise() -> None:
        with pytest.raises(GymCheckpointNotReady) as raised:
            async with coordinator.prepared("save-1"):
                raise AssertionError("unreachable")
        assert any("cleanup also failed" in note for note in raised.value.__notes__)

    _run_guarded(exercise)

    assert [call[0] for call in actor.calls] == ["prepare", "resume", "resume"]


def _restore_manifest() -> GymCheckpointManifest:
    return GymCheckpointManifest(
        schema_version=GYM_CHECKPOINT_SCHEMA_VERSION,
        checkpoint_id="save-1",
        instances={"tools/replica-0": (GymCheckpointEpisode("rollout-0", 0),)},
        staging_keys={"tools/replica-0": ()},
        restarts=_NO_RESTARTS,
    )


@pytest.mark.parametrize(
    ("outcome", "operations"),
    [
        ("body-succeeds", ["restore", "resume"]),
        # The controller could not take over the replacements: retire them first.
        ("body-fails", ["restore", "retire", "resume"]),
        # The restore cleans up its own failure, and the body never runs.
        ("restore-fails", ["restore", "retire", "resume"]),
    ],
)
def test_restored_reopens_gym_exactly_once(outcome: str, operations: list[str]) -> None:
    actor = _FakeGymActor(
        restore_error=RuntimeError("bad state") if outcome == "restore-fails" else None
    )
    coordinator = _coordinator(actor)
    manifest = _restore_manifest()
    entered = []

    async def exercise() -> None:
        async with coordinator.restored("restore-1", "/unused", manifest):
            entered.append(True)
            if outcome == "body-fails":
                raise ValueError("ledger rejected the restore")

    if outcome == "body-succeeds":
        _run_guarded(exercise)
    else:
        with pytest.raises((ValueError, GymCheckpointOperationError)):
            _run_guarded(exercise)

    assert [call[0] for call in actor.calls] == operations
    assert entered == ([] if outcome == "restore-fails" else [True])
    retired = [call for call in actor.calls if call[0] == "retire"]
    for call in retired:
        assert call[1][1] == (GymCheckpointEpisode("rollout-0", 1),)


def test_a_lost_resume_is_retried_on_only_the_actor_that_lost_it() -> None:
    """A transient resume failure would otherwise leave Gym closed until its lease."""
    healthy = _FakeGymActor()
    lost = _FakeGymActor(resume_errors=(ConnectionError("reply lost"),))
    coordinator = _coordinator(healthy, lost)

    async def exercise() -> None:
        async with coordinator.prepared("save-1"):
            pass

    _run_guarded(exercise)

    assert [call[0] for call in healthy.calls].count("resume") == 1
    assert [call[0] for call in lost.calls].count("resume") == 2
    assert all(call[1] == ("save-1",) for call in lost.calls if call[0] == "resume")


def test_a_resume_lost_after_a_successful_body_is_reported_not_raised() -> None:
    """Every commit is written by then; Gym's lease reopens the lost deployment."""
    healthy = _FakeGymActor()
    lost = _FakeGymActor(resume_errors=(ConnectionError("reply lost"),) * 2)
    coordinator = _coordinator(healthy, lost)
    entered = []

    async def exercise() -> None:
        with pytest.warns(UserWarning, match=r"is saved, but resume failed on"):
            async with coordinator.prepared("save-1"):
                entered.append(True)

    _run_guarded(exercise)

    assert entered == [True]
    assert [call[0] for call in lost.calls].count("resume") == 2


def test_a_resume_that_breaks_an_invariant_is_not_retried() -> None:
    actor = _FakeGymActor(resume_errors=(ValueError("malformed reply"),))
    coordinator = _coordinator(actor)

    async def exercise() -> None:
        with pytest.raises(GymCheckpointOperationError) as raised:
            await coordinator.resume("save-1")
        assert isinstance(raised.value.failures["tools/replica-0"], ValueError)

    _run_guarded(exercise)

    assert [call[0] for call in actor.calls] == ["resume"]


@pytest.mark.parametrize("operation", ["commit", "restore", "retire"])
def test_every_control_operation_is_bounded(operation: str, tmp_path: Path) -> None:
    actor = _FakeGymActor(hang=frozenset({operation}))
    coordinator = _fast_coordinator(actor)
    episodes = {"tools/replica-0": (GymCheckpointEpisode("rollout-0", 0),)}
    manifest = GymCheckpointManifest(
        schema_version=GYM_CHECKPOINT_SCHEMA_VERSION,
        checkpoint_id="save-1",
        instances=episodes,
        staging_keys={"tools/replica-0": ()},
        restarts=_NO_RESTARTS,
    )
    calls = {
        "commit": lambda: coordinator.commit(
            "save-1", tmp_path, episodes, restarts=_NO_RESTARTS
        ),
        "restore": lambda: coordinator.restore("restore-1", tmp_path, manifest),
        "retire": lambda: coordinator.retire("save-1", episodes),
    }

    async def exercise() -> None:
        with pytest.raises(GymCheckpointOperationError) as raised:
            await calls[operation]()
        assert any(
            isinstance(failure, TimeoutError)
            for failure in raised.value.failures.values()
        )

    _run_guarded(exercise)


@pytest.mark.parametrize(
    "error",
    [
        GymCheckpointNotReady({"tools/replica-0": {"agent": ("rollout-1",)}}),
        GymCheckpointOperationError("prepare", {"tools/replica-0": TimeoutError()}),
        GymCheckpointOperationError(
            "commit",
            {"tools/replica-0": GymCheckpointUnavailable("HTTP 500 participant")},
        ),
        GymCheckpointOperationError(
            "renew", {"tools/replica-0": ray.exceptions.RayActorError()}
        ),
    ],
    ids=["not-ready", "unanswered", "control-plane", "dead-actor"],
)
def test_failures_that_leave_nothing_behind_are_transient(error) -> None:
    assert is_transient_gym_checkpoint_failure(error)


@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("Gym episode ownership changed after commit"),
        ValueError("inventory does not match the live actor topology"),
        # An actor-side validation failure, such as an unexpected participant set.
        GymCheckpointOperationError(
            "commit", {"tools/replica-0": RuntimeError("unexpected participant set")}
        ),
        # One broken instance is not excused by another that merely timed out.
        GymCheckpointOperationError(
            "commit",
            {
                "tools/replica-0": TimeoutError(),
                "tools/replica-1": TypeError("invalid commit summary"),
            },
        ),
    ],
    ids=["controller-invariant", "topology", "actor-invariant", "mixed"],
)
def test_broken_invariants_are_not_transient(error) -> None:
    assert not is_transient_gym_checkpoint_failure(error)


def test_prepare_gets_its_own_deadline_and_other_calls_keep_the_control_one() -> None:
    """Parking waits for agent steps; renew, commit and resume are plain RPCs."""
    actor = _FakeGymActor()
    coordinator = GymCheckpointCoordinator(
        NemoGymShardSet(handles={"tools": [actor]}),
        control_timeout_s=30.0,
        prepare_timeout_s=600.0,
    )

    async def exercise() -> float:
        started = time.time()
        async with coordinator.prepared("save-1"):
            pass
        return started

    started = asyncio.run(exercise())

    deadlines = {
        operation: kwargs["deadline_ts"] for operation, _, kwargs in actor.calls
    }
    assert set(deadlines) == {"prepare", "renew", "resume"}
    assert 590.0 < deadlines["prepare"] - started < 610.0
    assert 20.0 < deadlines["renew"] - started < 40.0
    assert 20.0 < deadlines["resume"] - started < 40.0
