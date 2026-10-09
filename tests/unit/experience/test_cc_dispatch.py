# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Preserve original CC actor replay guards under RL-owned selection."""

import asyncio
from copy import deepcopy
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

import nemo_rl.environments.nemo_gym as gym_environment
from nemo_rl.algorithms.single_controller_utils.config import RolloutRecoveryConfig
from nemo_rl.environments.nemo_gym import GymTransportError, NemoGym
from nemo_rl.experience.rollout_manager import RolloutRetryPolicy, RolloutTimeouts
from nemo_rl.experience.rollout_reassembler import ActionOutputFlags, RolloutSelection
from nemo_rl.experience.rollout_recovery import RecoveryGranularity
from nemo_rl.utils.timer import Timer
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
def test_cc_group_retries_infrastructure_failures_until_budget_exhausted(
    failure_type, monkeypatch
):
    attempts = []
    sleep = AsyncMock()
    monkeypatch.setattr(asyncio, "sleep", sleep)

    async def execute_then_lose_response(_sample):
        attempts.append("mutation applied")
        raise failure_type("response lost")

    policy = RolloutRetryPolicy(
        max_infra_attempts=5,
        max_data_attempts=4,
        max_gym_row_attempts=3,
        backoff_base_s=0.25,
        max_backoff_s=1,
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
    else:
        assert asyncio.run(manager.generate_for_finalization({"idx": 1})) is None
    expected_attempts = 1 if failure_type is ValueError else policy.max_infra_attempts
    assert attempts == ["mutation applied"] * expected_attempts
    assert len(buffer.abort_calls) == expected_attempts
    assert [call.args[0] for call in sleep.await_args_list] == [
        policy.backoff_for(attempt) for attempt in range(1, expected_attempts)
    ]
    assert buffer.commit_calls == []
    assert manager.stats.skipped == (0 if failure_type is ValueError else 1)


@pytest.mark.parametrize(
    "granularity", [RecoveryGranularity.SIBLING, RecoveryGranularity.PROMPT_GROUP]
)
def test_cc_infra_retry_preserves_only_sealed_selections(granularity):
    """Retry only missing siblings; discard unsealed selections on a group restart."""
    first_selection = RolloutSelection(("first",), (ActionOutputFlags(False, False),))
    retried_selection = RolloutSelection(
        ("retried",), (ActionOutputFlags(False, False),)
    )
    second_selection = RolloutSelection(("second",), (ActionOutputFlags(False, False),))
    attempts = []
    policy = RolloutRetryPolicy(
        max_infra_attempts=2,
        max_data_attempts=1,
        max_gym_row_attempts=1,
        backoff_base_s=0,
        max_backoff_s=0,
        max_skipped_prompts=0,
        max_consecutive_dropped_prompts=1,
    )
    buffer = _FakeCaptureBuffer()
    manager = _make_capture_manager(
        buffer,
        retry_policy=policy,
        recovery_config=RolloutRecoveryConfig(default_granularity=granularity),
    )
    manager._context_compaction = True

    async def run_with_one_transient_failure(
        _sample, *, rollout_ids, generation_indices, on_completion, **_kwargs
    ):
        attempts.append(tuple(generation_indices))
        selected_ids = [rollout_ids[index] for index in generation_indices]
        record = _receipt_record(
            selected_ids,
            [
                {
                    "rollout_id": rollout_id,
                    "manifest": [{"staging_key": f"{rollout_id}/call"}],
                }
                for rollout_id in selected_ids
            ],
        )
        for index, completion in zip(generation_indices, record.completions):
            completion.env_extras["ng_logical_selection"] = (
                (first_selection if len(attempts) == 1 else retried_selection)
                if index == 0
                else second_selection
            )
            await on_completion(index, completion)
            if len(attempts) == 1:
                # Simulate a lost response after sibling 0 has been reported.
                raise GymTransportError("transient response loss")
        return record

    manager._impl.run_rollout = run_with_one_transient_failure
    request = asyncio.run(manager.generate_for_finalization({"idx": 1}))
    assert request is not None
    expected_first = (
        first_selection
        if granularity is RecoveryGranularity.SIBLING
        else retried_selection
    )
    assert request.logical_selections == (expected_first, second_selection)
    assert attempts == (
        [(0, 1), (1,)]
        if granularity is RecoveryGranularity.SIBLING
        else [(0, 1), (0, 1)]
    )
    assert len(buffer.abort_calls) == 1
    assert manager.stats.skipped == 0


@pytest.mark.parametrize("granularity", list(RecoveryGranularity))
def test_cc_retry_discards_selection_if_sealing_fails(monkeypatch, granularity):
    """A reported selection must not outlive its failed receipt seal."""
    manager = _make_capture_manager(
        _FakeCaptureBuffer(),
        retry_policy=RolloutRetryPolicy.single_attempt(
            max_infra_attempts=2, backoff_base_s=0, max_backoff_s=0
        ),
        recovery_config=RolloutRecoveryConfig(default_granularity=granularity),
    )
    manager._context_compaction = True
    attempts = []
    selections = [
        RolloutSelection((name,), (ActionOutputFlags(False, False),))
        for name in ("abandoned", "fresh")
    ]
    method_name = (
        "mark_sibling_sealed"
        if granularity is RecoveryGranularity.SIBLING
        else "mark_group_sealed"
    )
    original = getattr(manager._recovery_ledger, method_name)
    seals = []

    def fail_first_seal(*args, **kwargs):
        seals.append(1)
        if len(seals) == 1:
            raise GymTransportError("seal interrupted")
        return original(*args, **kwargs)

    monkeypatch.setattr(manager._recovery_ledger, method_name, fail_first_seal)

    async def run(_sample, *, rollout_ids, generation_indices, on_completion, **kw):
        attempts.append(tuple(generation_indices))
        selected_ids = [rollout_ids[i] for i in generation_indices]
        record = _receipt_record(selected_ids, [None] * len(selected_ids))
        for index, completion in zip(generation_indices, record.completions):
            completion.env_extras["ng_logical_selection"] = selections[
                len(attempts) - 1
            ]
            await on_completion(index, completion)
        return record

    manager._impl.run_rollout = run
    request = asyncio.run(manager.generate_for_finalization({"idx": 1}))
    assert attempts == [(0, 1), (0, 1)]
    assert request.logical_selections == (selections[1], selections[1])


@pytest.mark.parametrize("granularity", list(RecoveryGranularity))
def test_cc_retry_after_all_siblings_sealed_does_not_replay(granularity):
    manager = _make_capture_manager(
        _FakeCaptureBuffer(),
        retry_policy=RolloutRetryPolicy.single_attempt(
            max_infra_attempts=2, backoff_base_s=0, max_backoff_s=0
        ),
        recovery_config=RolloutRecoveryConfig(default_granularity=granularity),
    )
    manager._context_compaction = True
    selection = RolloutSelection(("accepted",), (ActionOutputFlags(False, False),))
    attempts = []

    async def run(_sample, *, rollout_ids, generation_indices, on_completion, **kw):
        attempts.append(tuple(generation_indices))
        record = _receipt_record(rollout_ids, [None] * len(rollout_ids))
        for index, completion in zip(generation_indices, record.completions):
            completion.env_extras["ng_logical_selection"] = selection
            await on_completion(index, completion)
        raise GymTransportError("stream failed after final completion")

    manager._impl.run_rollout = run
    request = asyncio.run(manager.generate_for_finalization({"idx": 1}))
    assert attempts == [(0, 1)]
    assert request.logical_selections == (selection, selection)
    assert manager.stats.skipped == 0


def test_cc_cancellation_cleans_up_without_retry():
    buffer = _FakeCaptureBuffer()
    attempts = []

    async def cancelled(_sample):
        attempts.append(1)
        raise asyncio.CancelledError()

    manager = _make_capture_manager(
        buffer,
        on_run=cancelled,
        retry_policy=RolloutRetryPolicy.single_attempt(max_infra_attempts=2),
    )
    manager._context_compaction = True
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(manager.generate_for_finalization({"idx": 1}))
    assert attempts == [1]
    assert len(buffer.abort_calls) == 1
    assert manager.stats.skipped == 0


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
                "manifest": [{"staging_key": f"{rollout_ids[0]}/good"}],
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


@pytest.mark.parametrize("mode", ["legacy", "capture", "cc"])
@pytest.mark.parametrize("top,nested", [(True, False), (False, True), (False, False)])
def test_validity_normalization_is_confined_to_cc(
    mode: str, top: bool, nested: bool
) -> None:
    """Ordinary streams preserve verifier output; CC projects its combined mask."""
    row = {
        "_rowidx": 0,
        "_ng_rollout_id": "owner",
        "agent_ref": {"name": "agent"},
        "responses_create_params": {"input": []},
    }
    raw = {
        "response": {"output": []},
        "mask_sample": top,
        "instance_config": {"mask_sample": nested},
    }

    class Helper:
        def run_examples(
            self, examples: list[dict], head_server_config: object, **kwargs: Any
        ) -> list[Any]:
            async def done() -> tuple[dict, dict]:
                return row, deepcopy(raw)

            return [done()]

    cls = NemoGym.__ray_metadata__.modified_class
    env = object.__new__(cls)
    env.cfg = {}
    env.rch = Helper()
    env.head_server_config = object()
    env._tokenizer = object()
    env._token_capture_enabled = mode != "legacy"
    env._context_compaction = mode == "cc"
    env._require_spinup = lambda: None
    # Token decoding and storage are irrelevant to verifier-field propagation.
    env._postprocess_nemo_gym_to_nemo_rl_result = lambda row, result, tokenizer, **kw: {
        "message_log": [],
        "full_result": result,
    }
    env._postprocess_captured_history = AsyncMock(
        side_effect=lambda row, result: {"message_log": [], "full_result": result}
    )
    env._drained_manifest = AsyncMock(side_effect=OSError("storage unavailable"))

    async def run() -> list:
        return [item async for item in env.run_rollouts([row], "test")]

    actual = asyncio.run(run())[0][2]["full_result"]
    assert actual["instance_config"]["mask_sample"] == (
        top or nested if mode == "cc" else nested
    )
    if mode != "cc":
        assert actual == raw
    assert raw["instance_config"]["mask_sample"] == nested
