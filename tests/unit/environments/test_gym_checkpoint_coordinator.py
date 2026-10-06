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
import json
from pathlib import Path
from typing import Any

import pytest
import ray.exceptions

from nemo_rl.environments.gym_checkpoint_adapter import (
    GymCheckpointCommitSummary,
    GymCheckpointEpisode,
    GymCheckpointParticipantCommitSummary,
    GymCheckpointParticipantManifest,
    GymCheckpointParticipantRecord,
    GymCheckpointPrepareSummary,
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

pytestmark = pytest.mark.nemo_gym


class _RemoteMethod:
    def __init__(self, function) -> None:
        self._function = function

    def remote(self, *args, **kwargs):
        return self._function(*args, **kwargs)


class _FakeGymActor:
    def __init__(
        self,
        *,
        prepared: bool = True,
        restore_error: BaseException | None = None,
        exported_episodes: tuple[GymCheckpointEpisode, ...] | None = None,
        staging_keys_by_episode: dict[GymCheckpointEpisode, tuple[str, ...]]
        | None = None,
        hang: frozenset[str] = frozenset(),
        participants: tuple[GymCheckpointParticipantCommitSummary, ...] = (),
    ) -> None:
        self.prepared = prepared
        self.participants = participants
        # Operations whose call is accepted but never answered, like an actor
        # whose event loop is stuck.
        self.hang = hang
        self.restore_error = restore_error
        self.exported_episodes = exported_episodes
        self.staging_keys_by_episode = staging_keys_by_episode or {}
        self.calls: list[tuple[str, tuple[Any, ...], dict[str, Any]]] = []
        self.checkpoint_prepare = _RemoteMethod(self._prepare)
        self.checkpoint_renew = _RemoteMethod(self._renew)
        self.checkpoint_retire = _RemoteMethod(self._retire)
        self.checkpoint_commit = _RemoteMethod(self._commit)
        self.checkpoint_restore = _RemoteMethod(self._restore)
        self.checkpoint_resume = _RemoteMethod(self._resume)

    async def _record(self, operation: str, args, kwargs) -> None:
        self.calls.append((operation, args, kwargs))
        if operation in self.hang:
            await asyncio.Event().wait()

    async def _prepare(self, *args, **kwargs) -> GymCheckpointPrepareSummary:
        await self._record("prepare", args, kwargs)
        return GymCheckpointPrepareSummary(
            prepared=self.prepared,
            blockers={} if self.prepared else {"agent": ("rollout-1",)},
        )

    async def _renew(self, *args, **kwargs) -> None:
        await self._record("renew", args, kwargs)

    async def _commit(self, *args, **kwargs) -> GymCheckpointCommitSummary:
        await self._record("commit", args, kwargs)
        episodes = args[2] if self.exported_episodes is None else self.exported_episodes
        return GymCheckpointCommitSummary(
            exported_episodes=episodes,
            staging_keys=tuple(
                sorted(
                    {k for keys in self.staging_keys_by_episode.values() for k in keys}
                )
            ),
            participants=self.participants,
            staging_keys_by_episode=self.staging_keys_by_episode,
        )

    async def _retire(self, *args, **kwargs) -> None:
        await self._record("retire", args, kwargs)

    async def _restore(self, *args, **kwargs) -> None:
        await self._record("restore", args, kwargs)
        if self.restore_error is not None:
            raise self.restore_error

    async def _resume(self, *args, **kwargs) -> None:
        await self._record("resume", args, kwargs)


def _coordinator(*actors: _FakeGymActor) -> GymCheckpointCoordinator:
    return GymCheckpointCoordinator(
        NemoGymShardSet(handles={"tools": list(actors)}),
        control_timeout_s=30.0,
    )


def test_manifest_rejects_duplicate_episode_identity() -> None:
    episode = GymCheckpointEpisode("rollout-0", 2)

    with pytest.raises(ValueError, match="duplicate episodes"):
        GymCheckpointManifest(
            schema_version=GYM_CHECKPOINT_SCHEMA_VERSION,
            checkpoint_id="save-1",
            instances={"tools/replica-0": (episode, episode)},
            staging_keys={"tools/replica-0": ()},
            participants={"tools/replica-0": ()},
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
            result = await coordinator.commit("save-1", tmp_path, inventory)
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
        coordinator.commit("save-1", tmp_path, {"tools/replica-0": (*saved, on_wire)})
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
            participants=(),
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
        participants={
            "tools/replica-0": (),
            "tools/replica-1": (),
        },
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


def _fast_coordinator(*actors: _FakeGymActor) -> GymCheckpointCoordinator:
    return GymCheckpointCoordinator(
        NemoGymShardSet(handles={"tools": list(actors)}),
        control_timeout_s=0.05,
    )


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
    """Cleanup is best effort: Gym's lease reopens admission if resume is lost."""
    actor = _FakeGymActor(prepared=False, hang=frozenset({"resume"}))
    coordinator = _fast_coordinator(actor)

    async def exercise() -> None:
        with pytest.raises(GymCheckpointNotReady) as raised:
            async with coordinator.prepared("save-1"):
                raise AssertionError("unreachable")
        assert any("cleanup also failed" in note for note in raised.value.__notes__)

    _run_guarded(exercise)

    assert [call[0] for call in actor.calls] == ["prepare", "resume"]


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
        participants={"tools/replica-0": ()},
    )
    calls = {
        "commit": lambda: coordinator.commit("save-1", tmp_path, episodes),
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


def _committed_participant(
    server_name: str, kind: str, digest: str
) -> GymCheckpointParticipantCommitSummary:
    return GymCheckpointParticipantCommitSummary(
        server_name=server_name,
        kind=kind,
        phase="committed",
        capture_keys=(),
        manifest=GymCheckpointParticipantManifest(
            schema_version=1,
            kind=kind,
            instance=f"{server_name}-state",
            checkpoint_id="save-1",
            records_file="records.jsonl",
            records_sha256=digest,
            record_count=0,
        ),
        staging_keys=(),
    )


def test_commit_records_each_instance_participant_topology(tmp_path: Path) -> None:
    """Restore needs it: a participant missing later must not be silently skipped."""
    first = _FakeGymActor(
        participants=(
            _committed_participant("tools-environment", "environment", "a" * 64),
            _committed_participant("tools-resources", "resources", "b" * 64),
        )
    )
    second = _FakeGymActor(
        participants=(_committed_participant("tools-env", "environment", "c" * 64),)
    )
    coordinator = _coordinator(first, second)
    inventory = {"tools/replica-0": (), "tools/replica-1": ()}

    result = asyncio.run(coordinator.commit("save-1", tmp_path, inventory))

    expected = {
        "tools/replica-0": (
            GymCheckpointParticipantRecord(
                "tools-environment", "environment", "tools-environment-state", "a" * 64
            ),
            GymCheckpointParticipantRecord(
                "tools-resources", "resources", "tools-resources-state", "b" * 64
            ),
        ),
        "tools/replica-1": (
            GymCheckpointParticipantRecord(
                "tools-env", "environment", "tools-env-state", "c" * 64
            ),
        ),
    }
    assert result.manifest.participants == expected
    assert load_gym_checkpoint_manifest(tmp_path).participants == expected


def test_restore_hands_each_instance_its_recorded_participants() -> None:
    first = _FakeGymActor()
    second = _FakeGymActor()
    coordinator = _coordinator(first, second)
    recorded = {
        "tools/replica-0": (
            GymCheckpointParticipantRecord("a-env", "environment", "a-env", "a" * 64),
        ),
        "tools/replica-1": (
            GymCheckpointParticipantRecord("b-env", "environment", "b-env", "b" * 64),
        ),
    }
    manifest = GymCheckpointManifest(
        schema_version=GYM_CHECKPOINT_SCHEMA_VERSION,
        checkpoint_id="save-1",
        instances={"tools/replica-0": (), "tools/replica-1": ()},
        staging_keys={"tools/replica-0": (), "tools/replica-1": ()},
        participants=recorded,
    )

    asyncio.run(coordinator.restore("restore-1", "/unused", manifest))

    for actor, instance_id in ((first, "tools/replica-0"), (second, "tools/replica-1")):
        (restore_call,) = [call for call in actor.calls if call[0] == "restore"]
        assert restore_call[2]["participants"] == recorded[instance_id]


def test_a_manifest_without_participant_topology_is_rejected(tmp_path: Path) -> None:
    manifest = GymCheckpointManifest(
        schema_version=GYM_CHECKPOINT_SCHEMA_VERSION,
        checkpoint_id="save-1",
        instances={"tools/replica-0": ()},
        staging_keys={"tools/replica-0": ()},
        participants={"tools/replica-0": ()},
    )
    raw = manifest.to_dict()
    del raw["participants"]
    (tmp_path / "gym_checkpoint.json").write_text(json.dumps(raw))

    with pytest.raises(ValueError, match="participants must be a mapping"):
        load_gym_checkpoint_manifest(tmp_path)


@pytest.mark.parametrize(
    ("participants", "match"),
    [
        ({}, "participant topology must match"),
        (
            {
                "tools/replica-0": (
                    GymCheckpointParticipantRecord("a", "environment", "a", "a" * 64),
                    GymCheckpointParticipantRecord("a", "model", "a2", "b" * 64),
                )
            },
            "duplicate participants",
        ),
        (
            {
                "tools/replica-0": (
                    GymCheckpointParticipantRecord("a", "judge", "a", "a" * 64),
                )
            },
            "unknown kind",
        ),
        (
            {
                "tools/replica-0": (
                    GymCheckpointParticipantRecord("a", "environment", "a", "nope"),
                )
            },
            "invalid records_sha256",
        ),
    ],
    ids=["missing-instance", "duplicate", "unknown-kind", "bad-digest"],
)
def test_manifest_rejects_a_malformed_participant_topology(
    participants, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        GymCheckpointManifest(
            schema_version=GYM_CHECKPOINT_SCHEMA_VERSION,
            checkpoint_id="save-1",
            instances={"tools/replica-0": ()},
            staging_keys={"tools/replica-0": ()},
            participants=participants,
        )
