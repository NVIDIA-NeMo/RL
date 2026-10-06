# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Preserve original CC actor replay guards under RL-owned selection."""

import asyncio
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
import torch

import nemo_rl.environments.nemo_gym as gym_environment
from nemo_rl.algorithms.single_controller_utils.config import RolloutRecoveryConfig
from nemo_rl.data_plane.tq_token_sink import TQTokenSink, TQTokenSource
from nemo_rl.environments.nemo_gym import GymTransportError, NemoGym
from nemo_rl.experience.interfaces import Completion
from nemo_rl.experience.rollout_manager import RolloutRetryPolicy, RolloutTimeouts
from nemo_rl.experience.rollout_reassembler import (
    ActionOutputFlags,
    RolloutReassembler,
    RolloutSelection,
)
from nemo_rl.experience.rollout_reassembler_actor import (
    RolloutReassemblerActor,
    assert_metadata_only,
)
from nemo_rl.experience.rollout_recovery import (
    RolloutRecoveryLedger,
    UnresolvedCaptureAcknowledgement,
)
from nemo_rl.models.generation.capture_context import decide_capture_input
from nemo_rl.utils.timer import Timer
from tests.unit.experience.test_logical_owner_finalization import (
    PublicationDataPlane,
    capture_segment,
    gym_harness,
)
from tests.unit.experience.test_rollout_generation_failures import (
    _FakeGymMethod,
    _gym_rows,
    _make_gym_impl,
)
from tests.unit.experience.test_rollout_manager import (
    _FakeCaptureBuffer,
    _make_capture_manager,
    _receipt_record,
)

pytestmark = pytest.mark.nemo_gym


def _env(harness):
    env = object.__new__(NemoGym.__ray_metadata__.modified_class)
    env._context_compaction = True
    env.cfg = {}

    async def control(method, path):
        assert method == "GET"
        capture_id = path.split("/")[-2]
        return await harness.ledger.manifest(capture_id)

    env._control = AsyncMock(side_effect=control)
    return env


def test_cc_row_stream_disables_ray_replay_and_row_redispatch():
    class Remote:
        def __init__(self):
            self.calls = 0
            self.call_options = None

        def options(self, **kwargs):
            self.call_options = kwargs
            return self

        async def remote(self, *_args, per_prompt=False):
            assert per_prompt is False
            self.calls += 1
            raise GymTransportError("mutating agent ran; stream response lost")
            yield  # Ray streaming method shape, deliberately no result after failure.

    remote = Remote()
    impl = _make_gym_impl(remote, row_attempts=5)
    impl._context_compaction = True
    with pytest.raises(GymTransportError, match="stream response lost"):
        asyncio.run(
            impl._run_rollouts(
                [{"_rowidx": 0, "agent_ref": {"name": "agent"}}], Timer(), "test"
            )
        )
    assert remote.calls == 1
    assert remote.call_options == {"num_returns": "streaming", "max_task_retries": 0}


@pytest.mark.parametrize("failure_type", [GymTransportError, ValueError, TimeoutError])
def test_cc_group_uses_ordinary_retry_budgets(failure_type):
    attempts = []

    async def execute_then_lose_response(_sample):
        attempts.append("mutation applied")
        raise failure_type("response lost")

    policy = RolloutRetryPolicy(
        max_infra_attempts=5,
        max_data_attempts=4,
        max_gym_row_attempts=3,
        backoff_base_s=0,
        max_backoff_s=0,
        max_skipped_prompts=10,
        max_consecutive_dropped_prompts=10,
    )
    buffer = _FakeCaptureBuffer()
    manager = _make_capture_manager(
        buffer, retry_policy=policy, on_run=execute_then_lose_response
    )
    manager._context_compaction = True
    if failure_type is ValueError:
        with pytest.raises(ValueError, match="response lost"):
            asyncio.run(manager.generate_for_finalization({"idx": 1}))
        expected_attempts = 4
    else:
        assert asyncio.run(manager.generate_for_finalization({"idx": 1})) is None
        expected_attempts = 5
    assert attempts == ["mutation applied"] * expected_attempts
    assert len(buffer.abort_calls) == expected_attempts
    assert buffer.commit_calls == []
    assert manager.stats.skipped == (0 if failure_type is ValueError else 1)


def test_cc_hung_stream_deadline_drops_group_and_cancels_collection():
    method = _FakeGymMethod(rows_to_yield=1, hang_after=True)
    impl = _make_gym_impl(method, timeouts=RolloutTimeouts(rollout_s=0.01))
    impl._context_compaction = True

    async def hung_stream(_sample):
        await impl._run_rollouts(_gym_rows(2), Timer(), "test")

    buffer = _FakeCaptureBuffer()
    manager = _make_capture_manager(
        buffer,
        on_run=hung_stream,
        retry_policy=RolloutRetryPolicy(
            max_infra_attempts=1,
            max_data_attempts=1,
            max_gym_row_attempts=1,
            max_consecutive_dropped_prompts=1,
        ),
    )
    manager._context_compaction = True
    assert asyncio.run(manager.generate_for_finalization({"idx": 1})) is None
    assert method.cancelled == 1
    assert len(buffer.abort_calls) == 1 and not buffer.commit_calls
    assert manager.stats.skipped == 1


def test_cc_seals_failed_owner_without_claiming_its_pending_write():
    buffer = _FakeCaptureBuffer()
    manager = _make_capture_manager(buffer)
    manager._context_compaction = True
    manager._execution_row_multiple = 1

    async def completed(
        _sample, *, rollout_ids, generation_indices, on_completion, **kwargs
    ):
        receipts = [
            {
                "rollout_id": rollout_ids[0],
                "manifest": [
                    {"model_call_id": "good", "staging_key": f"{rollout_ids[0]}/good"}
                ],
            },
            {
                "rollout_id": rollout_ids[1],
                "manifest": [],
                "capture_poisoned": True,
                "attempted_call_ids": ["failed"],
                "pending_call_ids": ["failed"],
            },
        ]
        record = _receipt_record(rollout_ids, receipts)
        for index, completion in enumerate(record.completions):
            completion.env_extras["ng_logical_selection"] = (
                RolloutSelection(("response",), (ActionOutputFlags(False, False),))
                if index == 0
                else RolloutSelection((), ())
            )
            await on_completion(index, completion)
        return record

    manager._impl.run_rollout = completed
    request = asyncio.run(manager.generate_for_finalization({"idx": 1}))
    assert request.receipts[1]["pending_call_ids"] == ["failed"]
    group = manager._recovery_ledger.get_group(request.group_id)
    assert group.siblings[0].current_attempt.staging_keys == [
        f"{request.rollout_ids[0]}/good"
    ]
    assert group.siblings[1].current_attempt.staging_keys == []
    assert not buffer.abort_calls


@pytest.mark.parametrize("cc", [False, True])
def test_actor_creation_preserves_foundation_no_replay_defaults(monkeypatch, cc):
    # The foundation now disables replay on the actor declaration for all runs.
    assert NemoGym._default_options["max_restarts"] == 0
    assert NemoGym._default_options["max_task_retries"] == 0
    options = MagicMock()
    actor = options.return_value.remote.return_value
    monkeypatch.setattr(
        gym_environment, "build_nemo_gym_config", lambda *_args, **_kwargs: {}
    )
    monkeypatch.setattr(
        gym_environment, "make_actor_runtime_env", lambda _: {"existing": "runtime"}
    )
    monkeypatch.setattr(gym_environment, "NemoGym", MagicMock(options=options))
    monkeypatch.setattr(gym_environment.ray, "get", lambda value: value)
    created = gym_environment.spinup_nemo_gym_actor(
        {"nemo_gym": {}},
        base_urls=[],
        model_name="model",
        tokenizer=None,
        enable_router_replay=False,
        use_fastokens=False,
        token_capture={"enabled": True, "context_compaction": cc},
    )
    assert created is actor
    expected = {"runtime_env": {"existing": "runtime"}}
    options.assert_called_once_with(**expected)
    actor._spinup.remote.assert_called_once()
    actor.set_tokenizer.remote.assert_called_once_with(None)


@pytest.fixture
def recovery_stack(monkeypatch, tmp_path):
    plane = PublicationDataPlane()
    source = TQTokenSource(plane, staging_partition="staged")
    harness = gym_harness.make_capture_harness(
        monkeypatch,
        tmp_path,
        sink=TQTokenSink(plane, staging_partition="staged"),
        fetch_prefix=source.fetch_prefix_token_ids,
        root_prompt=[10, 11],
        decide_input=decide_capture_input,
    )
    finalizer = RolloutReassembler(
        plane,
        partition_id="canonical",
        staging_partition="staged",
        pad_token_id=0,
        max_seq_len=1024,
    )
    try:
        yield harness, plane, finalizer
    finally:
        harness.client.close()


def _captured_result(harness, owner, count):
    chunks = [capture_segment(harness, owner, child=True) for _ in range(count)]
    manifest = chunks[-1]["manifest"]
    ids = tuple(identity for chunk in chunks for identity in chunk["ids"])
    return (
        dict(
            rollout_id=owner,
            manifest=[record.model_dump() for record in manifest.records],
            terminal_model_call_id=manifest.records[-1].model_call_id,
            terminal_selection="declared",
            attempted_call_ids=manifest.attempted_call_ids,
            pending_call_ids=manifest.pending_call_ids,
        ),
        RolloutSelection(ids, (ActionOutputFlags(False, False),) * len(ids)),
    )


@pytest.mark.parametrize("granularity", ["sibling", "prompt_group"])
@pytest.mark.parametrize(
    "interruption", ["runtime", "data_retry", "cold", "sealed_cold"]
)
def test_cc_retry_and_restart_reuse_durable_selection(
    recovery_stack, tmp_path: Path, granularity: str, interruption: str
):
    harness, plane, finalizer = recovery_stack
    policy = RolloutRetryPolicy(
        max_infra_attempts=2,
        max_data_attempts=2,
        max_gym_row_attempts=1,
        backoff_base_s=0,
        max_backoff_s=0,
    )

    def make_manager():
        manager = _make_capture_manager(
            _FakeCaptureBuffer(),
            retry_policy=policy,
            recovery_config=RolloutRecoveryConfig(default_granularity=granularity),
        )
        manager._context_compaction = True
        manager._execution_row_multiple = 1
        return manager

    manager = make_manager()
    calls, observed = [], {}

    async def run(
        _sample, *, rollout_ids, generation_indices, on_completion, recovery_granularity
    ):
        calls.append((list(rollout_ids), list(generation_indices)))
        for index in generation_indices:
            owner = rollout_ids[index]
            receipt, selection = await asyncio.to_thread(
                _captured_result, harness, owner, 2 if index == 0 else 1
            )
            observed[owner] = (receipt, selection)
            await on_completion(
                index,
                Completion(
                    message_log=[],
                    reward=float(index),
                    truncated=False,
                    env_extras={
                        "ng_receipt": receipt,
                        "ng_rollout_id": owner,
                        "ng_logical_selection": selection,
                    },
                ),
            )
            if len(calls) == 1 and (index == 0 or interruption == "sealed_cold"):
                if interruption == "runtime":
                    raise GymTransportError("lost stream")
                if interruption == "data_retry":
                    raise ValueError("invalid unfinished sibling")
                if interruption == "cold" or index == 1:
                    raise asyncio.CancelledError()

    async def exercise():
        nonlocal manager
        manager._impl.run_rollout = run
        if interruption in {"cold", "sealed_cold"}:
            with pytest.raises(asyncio.CancelledError):
                await manager.generate_for_finalization({"idx": 99})
            assert len(calls) == 1
            path = tmp_path / "recovery.pt"
            torch.save(manager.recovery_ledger.state_dict(), path)
            restored = RolloutRecoveryLedger.from_state_dict(
                torch.load(path, weights_only=True)
            )
            manager = make_manager()
            manager._recovery_ledger = restored
            async with manager._recovery_mutation() as cut:
                restored.prepare_for_restart(cut)
                restored.bind_runtime_prompt(
                    cut, restored.groups()[0].group_id, {"idx": 99}
                )
            manager._impl.run_rollout = run
            return await manager.generate_for_finalization(
                {"idx": 99}, lineage_group_id=restored.groups()[0].group_id
            )
        return await manager.generate_for_finalization({"idx": 99})

    request = asyncio.run(exercise())
    assert_metadata_only(request)
    if interruption == "sealed_cold":
        assert len(calls) == 1
        assert request.rollout_ids == tuple(calls[0][0])
    else:
        assert len(calls) == 2
        assert calls[1][1] == ([1] if granularity == "sibling" else [0, 1])
        assert (calls[0][0][0] == calls[1][0][0]) == (granularity == "sibling")
        assert calls[0][0][1] != calls[1][0][1]
    assert request.logical_selections == tuple(
        observed[owner][1] for owner in request.rollout_ids
    )
    assert request.receipts == tuple(
        observed[owner][0] for owner in request.rollout_ids
    )
    expected_keys = {
        entry["staging_key"]
        for receipt in request.receipts
        for entry in receipt["manifest"]
    }
    assert manager.recovery_ledger.expected_staging_keys() == expected_keys
    actor = object.__new__(RolloutReassemblerActor.__ray_metadata__.modified_class)
    actor._finalizer = finalizer
    result = actor.finalize(request)
    assert result.valid_row_count == result.total_row_count == 3
    assert [tag["logical_slot"] for tag in result.meta.tags] == [0, 0, 1]
    assert not any(
        part == "staged" and key in expected_keys for part, key in plane.rows
    )


@pytest.mark.parametrize("granularity", ["sibling", "prompt_group"])
@pytest.mark.parametrize("damage", ["foreign", "pending"])
def test_cc_unsafe_custody_never_reaches_cleanup(recovery_stack, granularity, damage):
    harness, plane, _ = recovery_stack
    buffer = _FakeCaptureBuffer()
    manager = _make_capture_manager(
        buffer, recovery_config=RolloutRecoveryConfig(default_granularity=granularity)
    )
    manager._context_compaction = True
    calls = []

    async def run(
        _sample, *, rollout_ids, generation_indices, on_completion, recovery_granularity
    ):
        calls.append(rollout_ids)
        for index in generation_indices:
            owner = rollout_ids[index]
            receipt, selection = await asyncio.to_thread(
                _captured_result, harness, owner, 1
            )
            if damage == "pending":
                receipt["pending_call_ids"] = ["unknown"]
            else:
                receipt["manifest"][0]["staging_key"] = "foreign/call"
            await on_completion(
                index,
                Completion(
                    message_log=[],
                    reward=1.0,
                    truncated=False,
                    env_extras={
                        "ng_receipt": receipt,
                        "ng_rollout_id": owner,
                        "ng_logical_selection": selection,
                    },
                ),
            )

    manager._impl.run_rollout = run
    expected = UnresolvedCaptureAcknowledgement if damage == "pending" else ValueError
    with pytest.raises(expected):
        asyncio.run(manager.generate_for_finalization({"idx": 99}))
    assert len(calls) == 1
    assert not manager.recovery_ledger.expected_staging_keys()
    assert plane.delete_count == 0
    if damage == "pending":
        assert buffer.abort_calls == []
        assert all(
            s.current_attempt.status.value == "dispatched"
            for s in manager.recovery_ledger.groups()[0].siblings
        )
