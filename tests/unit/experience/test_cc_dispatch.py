# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CC dispatch over real Gym custody, receipt assembly and group publication.

Model generation and DataPlane storage use the existing boundary doubles. This
does not qualify a live Ray cluster, vLLM or TransferQueue deployment.
"""

import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest
import torch

import nemo_rl.environments.nemo_gym as gym_environment
from nemo_rl.environments.nemo_gym import NemoGym
from nemo_rl.experience.failures import GymTransportError
from nemo_rl.algorithms.single_controller_utils.config import RolloutRecoveryConfig
from nemo_rl.experience.interfaces import Completion, PromptGroupRecord
from nemo_rl.experience.rollout_manager import RolloutRetryPolicy
from nemo_rl.experience.rollout_recovery import RolloutRecoveryLedger
from nemo_rl.experience.rollout_reassembler_actor import (
    RolloutReassemblerActor,
    assert_metadata_only,
)
from nemo_rl.utils.timer import Timer
from tests.unit.experience.test_logical_owner_finalization import (
    capture_segment,
    finalize,
)
from tests.unit.experience.test_logical_owner_finalization import (
    owner_stack as _owner_stack,
)
from tests.unit.experience.test_rollout_generation_failures import _make_gym_impl
from tests.unit.experience.test_rollout_manager import (
    _FakeCaptureBuffer,
    _make_capture_manager,
    _nemo_gym_impl,
)

pytestmark = pytest.mark.nemo_gym
owner_stack = _owner_stack  # Reuse the real capture/publication fixture.


def test_nonunit_dataset_weight_is_rejected_before_cc_dispatch():
    manager = _make_capture_manager(_FakeCaptureBuffer(), context_compaction=True)
    manager._impl.run_rollout = AsyncMock()
    with pytest.raises(ValueError, match="unit dataset loss_multiplier"):
        asyncio.run(
            manager.generate_for_finalization({"idx": 99, "loss_multiplier": 0.25})
        )
    manager._impl.run_rollout.assert_not_awaited()
    assert not manager.recovery_ledger.groups()


def _result(owner, segments, *, outcome="completed"):
    return {
        "reward": 1.0,
        "response": {"id": segments[-1].selected_response_ids[-1]},
        "instance_config": {"mask_sample": False},
        "context_compaction_result": {
            "logical_rollout_id": owner,
            "segments": [
                {
                    "capture_rollout_id": s.capture_rollout_id,
                    "segment_index": i,
                    "selected_actions": [
                        {
                            "response_id": response_id,
                            "finish_reason": "stop",
                            "last_output_item": None,
                        }
                        for response_id in s.selected_response_ids
                    ],
                    "media_occurrence_refs": [],
                }
                for i, s in enumerate(segments)
            ],
            "media_assets": {},
            "outcome": outcome,
        },
    }


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


def test_existing_output_detector_masks_only_each_selected_generation(owner_stack):
    harness, plane, finalizer = owner_stack
    env = _env(harness)
    env.cfg = {
        "invalid_tool_call_patterns": ["[[BAD]]"],
        "thinking_tags": ["<badthink>"],
    }
    segments = [
        capture_segment(harness, "group_g0_s0", child=True),
        capture_segment(harness, "group_g0_s1"),
    ]
    result = _result("group_g0", segments)
    selected = result["context_compaction_result"]["segments"]
    selected[0]["selected_actions"][0]["last_output_item"] = {
        "type": "message",
        "id": "message-0",
        "role": "assistant",
        "status": "completed",
        "content": [
            {"type": "output_text", "text": "[[BAD]] <badthink>", "annotations": []}
        ],
    }
    selected[1]["selected_actions"][0]["last_output_item"] = {
        "type": "reasoning",
        "id": "reasoning-1",
        "summary": [{"type": "summary_text", "text": "<badthink><badthink>"}],
    }
    processed = asyncio.run(
        env._postprocess_cc_receipt_mode({"_ng_rollout_id": "group_g0"}, result)
    )
    assert_metadata_only(processed)
    receipts = processed["logical_segments"]
    assert receipts[0].action_flags[0].invalid_tool_call
    assert receipts[0].action_flags[0].malformed_thinking
    assert not receipts[0].action_flags[1].invalid_tool_call
    sibling = capture_segment(harness, "group_g1_s0")
    finished = finalize(finalizer, [list(receipts), [sibling]])
    rows = [plane.rows["canonical", key] for key in finished.meta.sample_ids]
    assert rows[0]["invalid_tool_call_mask"][0].tolist() == [
        False,
        False,
        True,
        False,
        False,
        False,
    ]
    assert rows[0]["malformed_thinking_mask"][0].tolist() == [
        False,
        False,
        True,
        False,
        False,
        False,
    ]
    assert rows[1]["invalid_tool_call_mask"][0].tolist() == [False, False, False]
    assert rows[1]["malformed_thinking_mask"][0].tolist() == [False, False, True]
    assert not rows[2]["invalid_tool_call_mask"].any()
    for row in rows:
        assert not torch.any(row["malformed_thinking_mask"] & ~row["token_mask"].bool())
    assert sum(tag["num_invalid_tool_calls"] for tag in finished.meta.tags) == 1
    assert sum(tag["num_malformed_thinking"] for tag in finished.meta.tags) == 2
    assert sum(tag["num_assistant_messages"] for tag in finished.meta.tags) == 4


@pytest.mark.parametrize(
    "outcome", ["completed", "max_steps", "max_output_tokens", "execution_failure"]
)
def test_agent_result_reaches_existing_finalizer_with_exact_selection(
    owner_stack, outcome
):
    harness, plane, finalizer = owner_stack
    env = _env(harness)
    buf = _FakeCaptureBuffer()
    manager = _make_capture_manager(
        buf, context_compaction=True, execution_row_multiple=4
    )
    converter = _nemo_gym_impl(True)
    converter._context_compaction = True
    selected = []

    async def run(
        _sample, *, rollout_ids, generation_indices, on_completion, recovery_granularity
    ):
        processed = []
        for index, owner in enumerate(rollout_ids):
            segments = [
                await asyncio.to_thread(
                    capture_segment, harness, f"{owner}_s0", child=True, retry=True
                )
            ]
            if index == 0:
                segments.append(
                    await asyncio.to_thread(capture_segment, harness, f"{owner}_s1")
                )
            selected.append(segments)
            result = _result(owner, segments, outcome=outcome)
            result["reward"] = float(index)
            result["responses_create_params"] = {"input": "must not cross RPC"}
            processed.append(
                await env._postprocess_cc_receipt_mode(
                    {"_ng_rollout_id": owner}, result
                )
            )
        completions, _ = converter._results_to_completions(processed)
        for index, completion in zip(generation_indices, completions, strict=True):
            await on_completion(index, completion)
        assert [c.truncated for c in completions] == [
            outcome == "max_output_tokens"
        ] * 2
        assert all("responses_create_params" not in c.env_extras for c in completions)
        return PromptGroupRecord(
            prompt_idx=99,
            prompt=[],
            extra_env_info={},
            metadata={},
            completions=completions,
            rollout_metrics={},
        )

    manager._impl.run_rollout = run
    request = asyncio.run(manager.generate_for_finalization({"idx": 99}))
    assert_metadata_only(request)
    assert request.rewards == (0.0, 1.0)
    assert all(
        scope.startswith(owner + "_a") and len(scope) == len(owner) + 34
        for scope, owner in zip(
            request.rollout_ids, request.canonical_sample_ids, strict=True
        )
    )
    assert request.execution_row_multiple == 4
    assert env._control.await_count == 3  # once per segment, not per action/prefix
    assert request.canonical_sample_ids == tuple(
        f"{request.group_id}_g{i}" for i in range(2)
    )
    assert request.cleanup_sample_ids == tuple(
        f"{request.group_id}_g{i}_s{j}"
        for i, owner in enumerate(selected)
        for j in range(len(owner))
    ) + tuple(f"{request.group_id}_pad{i}" for i in range(3))
    actor = object.__new__(RolloutReassemblerActor.__ray_metadata__.modified_class)
    actor._finalizer = finalizer
    if outcome == "execution_failure":
        with pytest.raises(ValueError, match="no verified input layout"):
            actor.finalize(request)
        assert not any(partition == "canonical" for partition, _ in plane.rows)
        assert any(partition == "staged" for partition, _ in plane.rows)
        return
    result = actor.finalize(request)
    assert result.meta.sample_ids == [
        f"{request.group_id}_g0_s0",
        f"{request.group_id}_g0_s1",
        f"{request.group_id}_g1_s0",
        f"{request.group_id}_pad0",
    ]
    assert result.total_row_count == result.valid_row_count == 3
    rows = [
        row for (partition, _), row in plane.rows.items() if partition == "canonical"
    ]
    assert len(rows) == 4
    assert all(
        bool(row["truncated"][0]) == (outcome == "max_output_tokens")
        for row in rows[:3]
    )
    assert not rows[3]["truncated"].item()
    assert sorted(row["token_mask"].sum().item() for row in rows) == [0, 1, 2, 2]
    assert (
        sum(event[0] == "put" and event[1] == "canonical" for event in plane.events)
        == 1
    )
    assert not any(partition == "staged" for partition, _ in plane.rows)


@pytest.mark.parametrize(
    "damage",
    [
        "missing",
        "foreign_owner",
        "gap",
        "terminal",
        "duplicate",
        "nan",
        "missing_reward",
    ],
)
def test_invalid_cc_result_fails_before_manifest_lookup(owner_stack, damage):
    harness, plane, _ = owner_stack
    segment = capture_segment(harness, "group_g0_s0", child=True)
    result = _result("group_g0", [segment])
    cc = result["context_compaction_result"]
    if damage == "missing":
        del result["context_compaction_result"]
    elif damage == "foreign_owner":
        cc["logical_rollout_id"] = "other_g0"
        cc["segments"][0]["capture_rollout_id"] = "other_g0_s0"
    elif damage == "gap":
        cc["segments"][0]["segment_index"] = 1
    elif damage == "terminal":
        result["response"]["id"] = "different-response"
    elif damage == "duplicate":
        cc["segments"][0]["selected_actions"] *= 2
    elif damage == "nan":
        result["reward"] = float("nan")
    else:
        del result["reward"]
    env = _env(harness)
    with pytest.raises(ValueError):
        asyncio.run(
            env._postprocess_cc_receipt_mode({"_ng_rollout_id": "group_g0"}, result)
        )
    env._control.assert_not_awaited()
    assert plane.delete_count == 0


def test_manifest_failure_is_fatal_and_preserves_staging(owner_stack):
    harness, plane, _ = owner_stack
    segment = capture_segment(harness, "group_g0_s0")
    env = _env(harness)
    env._control.side_effect = OSError("manifest unavailable")
    with pytest.raises(OSError, match="unavailable"):
        asyncio.run(
            env._postprocess_cc_receipt_mode(
                {"_ng_rollout_id": "group_g0"}, _result("group_g0", [segment])
            )
        )
    assert env._control.await_count == 1
    assert plane.delete_count == 0 and len(plane.rows) == 1


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
        buffer,
        context_compaction=True,
        retry_policy=policy,
        on_run=execute_then_lose_response,
    )
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


@pytest.mark.parametrize("granularity", ["sibling", "prompt_group"])
def test_cc_rejects_foreign_staging_before_custody_and_cleanup(
    owner_stack: tuple, granularity: str
) -> None:
    harness, _, _ = owner_stack
    buffer = _FakeCaptureBuffer()
    manager = _make_capture_manager(
        buffer,
        context_compaction=True,
        recovery_config=RolloutRecoveryConfig(default_granularity=granularity),
    )

    async def run(
        _sample: dict,
        *,
        rollout_ids: list[str],
        generation_indices: list[int],
        on_completion: Callable[[int, Completion], Awaitable[None]],
        **_: object,
    ) -> None:
        for index in generation_indices:
            owner = rollout_ids[index]
            segment = await asyncio.to_thread(capture_segment, harness, f"{owner}_s0")
            segment.receipt["manifest"][0]["staging_key"] = "other_owner_s0/call"
            await on_completion(
                index,
                Completion(
                    message_log=[],
                    env_extras={
                        "ng_receipt": None,
                        "ng_rollout_id": owner,
                        "ng_logical_segments": (segment,),
                    },
                    reward=0.5,
                    truncated=False,
                ),
            )

    async def exercise() -> None:
        manager._impl.run_rollout = run
        with pytest.raises(ValueError, match="staging ownership"):
            await manager.generate_for_finalization({"idx": 99})
        assert manager.recovery_ledger.expected_staging_keys() == set()
        async with manager._recovery_mutation() as cut:
            await manager.discard_recovery_group(
                cut, manager.recovery_ledger.groups()[0].group_id
            )
        assert buffer.cleared_staging_key_batches == [[]]

    asyncio.run(exercise())


@pytest.mark.parametrize("granularity", ["sibling", "prompt_group"])
@pytest.mark.parametrize(
    "interruption", ["runtime", "data_retry", "cold", "sealed_cold"]
)
def test_cc_retry_and_restart_reuse_durable_evidence(
    owner_stack: tuple, tmp_path: Path, granularity: str, interruption: str
) -> None:
    harness, plane, finalizer = owner_stack
    policy = RolloutRetryPolicy(
        max_infra_attempts=2,
        max_data_attempts=2,
        max_gym_row_attempts=1,
        backoff_base_s=0,
        max_backoff_s=0,
    )

    def make_manager():
        return _make_capture_manager(
            _FakeCaptureBuffer(),
            context_compaction=True,
            retry_policy=policy,
            recovery_config=RolloutRecoveryConfig(default_granularity=granularity),
        )

    manager = make_manager()
    calls, observed = [], {}

    async def run(
        _sample, *, rollout_ids, generation_indices, on_completion, recovery_granularity
    ):
        calls.append((list(rollout_ids), list(generation_indices)))
        for index in generation_indices:
            owner = rollout_ids[index]
            segments = tuple(
                [
                    await asyncio.to_thread(
                        capture_segment, harness, f"{owner}_s{s}", child=True
                    )
                    for s in range(2 if index == 0 else 1)
                ]
            )
            observed[owner] = segments
            await on_completion(
                index,
                Completion(
                    message_log=[],
                    env_extras={
                        "ng_receipt": None,
                        "ng_rollout_id": owner,
                        "ng_logical_segments": segments,
                    },
                    reward=float(index),
                    truncated=False,
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
            assert len(calls) == 1  # Cancellation must not consume retry budgets.
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
    assert request.logical_segments == tuple(
        observed[owner] for owner in request.rollout_ids
    )
    expected_keys = {
        entry["staging_key"]
        for segments in request.logical_segments
        for segment in segments
        for entry in segment.receipt["manifest"]
    }
    assert manager.recovery_ledger.expected_staging_keys() == expected_keys
    actor = object.__new__(RolloutReassemblerActor.__ray_metadata__.modified_class)
    actor._finalizer = finalizer
    result = actor.finalize(request)
    assert result.valid_row_count == result.total_row_count == 3
    assert len(set(result.meta.sample_ids)) == 3
    assert [tag["logical_slot"] for tag in result.meta.tags] == [0, 0, 1]
    assert not any(
        part == "staged" and key in expected_keys for part, key in plane.rows
    )


def test_cc_converter_rejects_ordinary_receipt_fallback():
    converter = _nemo_gym_impl(True)
    converter._context_compaction = True
    result = {
        "receipt": None,
        "rollout_id": "group_g0",
        "message_log": [],
        "full_result": {"reward": 1.0},
    }
    with pytest.raises(ValueError, match="logical segment"):
        converter._results_to_completions([result])


def test_cc_row_stream_disables_ray_replay_and_row_redispatch():
    class Remote:
        def __init__(self):
            self.calls = 0
            self.call_options = None

        def options(self, **kwargs):
            self.call_options = kwargs
            return self

        async def remote(self, *_args):
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


@pytest.mark.parametrize("cc", [False, True])
def test_actor_creation_replay_options_are_scoped(monkeypatch, cc):
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
    if cc:
        expected.update(max_restarts=0, max_task_retries=0)
    options.assert_called_once_with(**expected)
    actor._spinup.remote.assert_called_once()
    actor.set_tokenizer.remote.assert_called_once_with(None)


def test_actor_dispatch_binds_cc_before_run_and_rejects_missing_result():
    cls = NemoGym.__ray_metadata__.modified_class
    env = cls(
        {
            "model_name": "model",
            "base_urls": [],
            "initial_global_config_dict": {},
            "token_capture": {"enabled": True, "context_compaction": True},
        }
    )
    row = {"_ng_rollout_id": "group_g0", "_rowidx": 0, "agent_ref": {"name": "agent"}}
    env._require_spinup = lambda: None
    env._tokenizer = object()
    env._control = AsyncMock()
    env._postprocess_receipt_mode = AsyncMock()
    called = []

    class Helper:
        def run_examples(self, *, examples, head_server_config, retry_requests):
            called.append(retry_requests)

            async def result():
                return row, {"reward": 1.0, "response": {"id": "ordinary-response"}}

            return [result()]

    env.rch = Helper()

    async def collect():
        return [result async for result in env.run_rollouts([row], "test")]

    with pytest.raises(ValueError):
        asyncio.run(collect())
    assert called == [False]
    env._control.assert_not_awaited()
    env._postprocess_receipt_mode.assert_not_awaited()


def test_segment_manifest_reads_are_bounded_and_keep_selection_order(owner_stack):
    harness, _, _ = owner_stack
    segments = [capture_segment(harness, f"group_g0_s{i}") for i in range(10)]
    env = _env(harness)
    original = env._control.side_effect
    active = peak = 0

    async def delayed(method, path):
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0)
        result = await original(method, path)
        active -= 1
        return result

    env._control.side_effect = delayed
    processed = asyncio.run(
        env._postprocess_cc_receipt_mode(
            {"_ng_rollout_id": "group_g0"}, _result("group_g0", segments)
        )
    )
    assert 1 < peak <= 8
    assert [s.capture_rollout_id for s in processed["logical_segments"]] == [
        s.capture_rollout_id for s in segments
    ]
    assert env._control.await_count == 10
