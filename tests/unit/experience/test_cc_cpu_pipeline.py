# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Composed CPU CC flow; generation, storage and learner execution are doubles.

Uses real shared-client policy decisions, HTTP capture/ledger, receipt processing,
finalization, replay selection and controller advantages. No real tokenizer,
image processor, vLLM, TransferQueue service or optimizer executes here.
"""

import asyncio
import io
from collections.abc import Iterator
from dataclasses import replace
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
import torch
from tensordict import TensorDict

pytest.importorskip("nemo_gym", reason="requires the paired Gym checkout")

from nemo_gym.config_types import ModelServerRef
from nemo_gym.context_management import (
    ContextHistoryConfig,
    ContextManagedResponsesClient,
)
from nemo_gym.openai_utils import NeMoGymResponseCreateParamsNonStreaming
from nemo_gym.server_utils import ServerClient
from responses_api_agents.simple_agent_with_compaction.tests.test_client import (
    http_response,
)

from nemo_rl.algorithms.async_utils.replay_buffer import TQReplayBuffer
from nemo_rl.algorithms.async_utils.staleness_sampler import InOrderSampler
from nemo_rl.data_plane.adapters.noop import NoOpDataPlaneClient
from nemo_rl.data_plane.tq_token_sink import TQTokenSink, TQTokenSource
from nemo_rl.experience.rollout_reassembler import ActionOutputFlags, RolloutReassembler
from tests.unit.experience.test_cc_dispatch import _env
from tests.unit.experience.test_logical_owner_finalization import (
    capture_segment,
    finalize,
    gym_harness,
)
from tests.unit.single_controller.test_cc_optimizer_batch import _setup

pytestmark = pytest.mark.nemo_gym


@pytest.fixture(params=[False])
def pipeline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, request: pytest.FixtureRequest
) -> Iterator[tuple]:
    plane = NoOpDataPlaneClient()
    plane.register_partition("staged", [], 128, ["finalize"])
    plane.register_partition("train", [], 128, ["train"])
    source = TQTokenSource(plane, staging_partition="staged")
    harness = gym_harness.make_capture_harness(
        monkeypatch,
        tmp_path,
        sink=TQTokenSink(plane, staging_partition="staged"),
        fetch_prefix=source.fetch_prefix_token_ids,
        root_prompt=[10, 11],
        reasoning=request.param,
    )
    finalizer = RolloutReassembler(
        plane,
        partition_id="train",
        staging_partition="staged",
        pad_token_id=0,
        max_seq_len=1024,
    )
    yield harness, plane, finalizer
    harness.client.close()
    asyncio.run(harness.ledger.close())
    plane.close()


async def ready_controller(
    monkeypatch: pytest.MonkeyPatch, plane: NoOpDataPlaneClient, meta: Any
) -> Any:
    data = plane.get_samples(meta.sample_ids, "train", meta.fields)
    ctrl, _ = _setup(monkeypatch, batches=[(meta, data)], plane=plane)
    ctrl._algo_cfg.baseline_population = "all_owners"
    buffer = TQReplayBuffer(
        plane,
        "train",
        pad_value_dict={"token_ids": 0},
        include_message_violation_fields=False,
        staging_partition_id="staged",
    )
    buffer.set_data_plane_checkpoint_barrier(ctrl._data_plane_checkpoint_barrier)
    group_id = meta.tags[0]["dispatch_group_id"]
    buffer.reserve(weight_version=0, target_step=0, group_id=group_id)
    async with ctrl._data_plane_checkpoint_barrier.mutation() as cut:
        await buffer.commit_finalized(
            cut, group_id, meta, group_min_wv=0, group_max_wv=0, staging_keys=[]
        )
    ctrl._buffer = buffer
    ctrl._sampler = InOrderSampler(buffer, max_lookahead_versions=0)
    return ctrl


@pytest.mark.parametrize(
    "compact,pipeline,chunk_size",
    [(False, False, 2), (True, True, 1), (True, True, 2)],
    indirect=["pipeline"],
    ids=["identity", "reasoning_k1", "reasoning_k2"],
)
def test_twenty_turn_shared_client_to_replay_and_advantages(
    pipeline: tuple,
    monkeypatch: pytest.MonkeyPatch,
    compact: bool,
    chunk_size: int,
) -> None:
    harness, plane, finalizer = pipeline
    env = _env(harness)
    expected_tokens, receipts, rewards = [], [], []

    async def exercise() -> None:
        for owner in range(2):
            transport = MagicMock(spec=ServerClient)

            async def post(**kwargs: Any) -> Any:
                assert kwargs["_retry"] is False
                response = harness.client.post(
                    kwargs["url_path"],
                    json=kwargs["json"].model_dump(mode="json"),
                    headers=kwargs["headers"],
                )
                gym_harness.assert_clean(response)
                return http_response(response.json())

            transport.post = AsyncMock(side_effect=post)
            client = ContextManagedResponsesClient(
                server_client=transport,
                model_server=ModelServerRef(name="policy", type="responses_api_models"),
                logical_rollout_id=f"group_g{owner}",
                initial_request=NeMoGymResponseCreateParamsNonStreaming(input="task"),
                config=ContextHistoryConfig.model_validate(
                    {
                        "enabled": compact,
                        "max_response_retries": 1,
                        "policy": {
                            "type": "recency",
                            "config": {
                                "reasoning": {"enabled": True, "keep_last_blocks": 0}
                            },
                        },
                        "schedule": {
                            "type": "turn_chunked_recency",
                            "actions_per_chunk": chunk_size,
                        },
                    }
                ),
            )
            selected = []
            for turn in range(20):
                # One definite, discarded response must never reach training.
                response = await client.create(
                    select_response=lambda _: len(harness.worker_calls) != 1
                )
                selected.append(1000 + len(harness.worker_calls))
                if turn < 19:
                    client.append_observation(
                        [
                            {
                                "role": "user",
                                "type": "message",
                                "content": f"observation {turn}",
                            }
                        ]
                    )
            result = client.finish(response)
            assert [len(s.selected_actions) for s in result.segments] == (
                [chunk_size] * (20 // chunk_size) if compact else [20]
            )
            processed = await env._postprocess_cc_receipt_mode(
                {"_ng_rollout_id": f"group_g{owner}"},
                {
                    "reward": float(owner),
                    "response": {"id": response.id},
                    "context_compaction_result": result.model_dump(mode="json"),
                },
            )
            receipts.append(list(processed["logical_segments"]))
            rewards.append(processed["full_result"]["reward"])
            expected_tokens.append(selected)

        result = finalizer.finalize_group(
            "group",
            ["group_g0", "group_g1"],
            [None, None],
            rewards,
            mask_sample=[False, False],
            fallback_weight_version=7,
            prompt_idx=99,
            logical_segments=receipts,
        )
        assert result.meta.size == (40 // chunk_size if compact else 2)
        assert result.valid_row_count == result.meta.size
        assert len(harness.worker_calls) == 41
        assert (
            sum(call.mode == "text" for call, _ in harness.worker_calls)
            == result.meta.size + 1
        )
        assert plane.list_sample_ids("staged") == []  # Includes the discarded root.
        ctrl = await ready_controller(monkeypatch, plane, result.meta)
        checked = []

        def learner(meta: Any, *args: Any, **kwargs: Any) -> None:
            data = plane.get_samples(
                meta.sample_ids,
                "train",
                ["input_ids", "token_mask", "advantages", "generation_logprobs"],
            )
            for owner in range(2):
                actual = []
                for index, tag in enumerate(meta.tags):
                    if tag["logical_slot"] != owner:
                        continue
                    mask = data["token_mask"][index].bool()
                    actual.extend(data["input_ids"][index][mask].tolist())
                    assert torch.all(data["generation_logprobs"][index][mask] == -0.25)
                    assert torch.all(data["advantages"][index][mask] == owner - 0.5)
                assert actual == expected_tokens[owner]
                assert len(actual) == len(set(actual)) == 20
            checked.append(meta.size)

        ctrl._trainer.train_microbatches_from_meta.side_effect = learner
        await asyncio.wait_for(ctrl._train_pump(), timeout=5)
        assert checked == [result.meta.size]
        ctrl._trainer.finish_train_step.assert_called_once()
        assert ctrl._trainer_version == 1 and ctrl._consumed_samples == 1
        assert ctrl._buffer.meta_list == [] and plane.list_sample_ids("train") == []

    asyncio.run(exercise())


@pytest.mark.parametrize("mask_all", [False, True])
def test_mixed_cc_snapshot_restores_segmented_replay_and_owner_advantages(
    pipeline: tuple,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mask_all: bool,
) -> None:
    harness, plane, finalizer = pipeline
    healthy = [capture_segment(harness, f"group_g0_s{i}") for i in range(3)]
    failed = capture_segment(harness, "group_g1_s0", child=True)
    failed = replace(failed, action_flags=(ActionOutputFlags(False, False),))
    result = finalize(finalizer, [healthy, [failed]], execution_row_multiple=3)
    meta = result.meta
    assert meta.size == 6
    if mask_all:
        # A structurally sound group can be fully loss-filtered after capture.
        # This is distinct from an all-invalid finalization with no input layout.
        plane.put_samples(
            sample_ids=meta.sample_ids,
            partition_id="train",
            fields=TensorDict(
                {"mask_sample": torch.ones(meta.size, dtype=torch.bool)},
                batch_size=[meta.size],
            ),
        )

    async def exercise() -> None:
        ctrl = await ready_controller(monkeypatch, plane, meta)
        async with ctrl._data_plane_checkpoint_barrier.checkpoint():
            state = ctrl._buffer.metadata_state_dict(saved_capacity=1)
            serialized = io.BytesIO()
            torch.save(state, serialized)
            plane.save_checkpoint(
                tmp_path / "snapshot",
                metadata={"manifest_digest": state["manifest_digest"]},
            )
        serialized.seek(0)
        state = torch.load(
            serialized, weights_only=False
        )  # Trusted test-created bytes.
        restored_plane = NoOpDataPlaneClient()
        native_metadata = restored_plane.load_checkpoint(tmp_path / "snapshot")
        restored_meta = state["groups"][0]["meta"]
        assert restored_meta.tags == meta.tags
        assert restored_meta.sample_ids == meta.sample_ids
        before = plane.get_samples(meta.sample_ids, "train", meta.fields)
        after = restored_plane.get_samples(meta.sample_ids, "train", meta.fields)
        for field in meta.fields:
            for left, right in zip(
                before[field].unbind(), after[field].unbind(), strict=True
            ):
                torch.testing.assert_close(left, right, rtol=0, atol=0)
        restored_buffer = TQReplayBuffer(
            restored_plane,
            "train",
            pad_value_dict={"token_ids": 0},
            include_message_violation_fields=False,
        )
        # Two logical owners, six physical rows, and capacity for one group.
        count = await restored_buffer.load_state_dict(
            state,
            max_groups=1,
            expected_partition_id="train",
            expected_group_size=2,
            expected_manifest_digest=native_metadata["manifest_digest"],
        )
        assert count == restored_buffer.size() == 1
        assert restored_buffer.meta_list == [restored_meta]
        assert restored_buffer.target_step_list == ctrl._buffer.target_step_list
        assert restored_buffer.start_weight_list == ctrl._buffer.start_weight_list
        assert restored_buffer.end_weight_list == ctrl._buffer.end_weight_list
        assert sorted(restored_plane.list_sample_ids("train")) == sorted(
            meta.sample_ids
        )
        restored_ctrl = await ready_controller(
            monkeypatch, restored_plane, restored_meta
        )
        await ctrl._advantage_stage(meta)
        await restored_ctrl._advantage_stage(restored_meta)
        for field in ("advantages", "sample_mask"):
            original = plane.get_samples(meta.sample_ids, "train", [field])[field]
            restored = restored_plane.get_samples(meta.sample_ids, "train", [field])[
                field
            ]
            for left, right in zip(original.unbind(), restored.unbind(), strict=True):
                torch.testing.assert_close(left, right, rtol=0, atol=0)
                if mask_all:
                    assert right.count_nonzero() == 0
        restored_plane.close()

    asyncio.run(exercise())


@pytest.mark.parametrize("supported", [False, True])
@pytest.mark.parametrize("logprobs", [False, True])
def test_mixed_failed_owner_padding_replay_and_controller(
    pipeline: tuple,
    monkeypatch: pytest.MonkeyPatch,
    supported: bool,
    logprobs: bool,
) -> None:
    harness, plane, finalizer = pipeline
    healthy = [capture_segment(harness, f"group_g0_s{i}") for i in range(3)]
    failed = capture_segment(harness, "group_g1_s0", child=True)
    failed = replace(failed, action_flags=(ActionOutputFlags(False, False),))
    result = finalize(finalizer, [healthy, [failed]], execution_row_multiple=3)
    meta = result.meta
    assert meta.size == 6 and result.valid_row_count == 3
    assert [tag["uses_borrowed_input"] for tag in meta.tags] == [False] * 3 + [True] * 3
    assert [tag["is_execution_padding"] for tag in meta.tags] == [False] * 4 + [
        True
    ] * 2
    assert plane.list_sample_ids("staged") == []

    async def exercise() -> None:
        ctrl = await ready_controller(monkeypatch, plane, meta)
        ctrl._policy_logprobs_required = ctrl._reference_logprobs_required = logprobs
        original_ids = list(meta.sample_ids)
        check = ctrl._trainer.worker_group.run_all_workers_single_data
        if not supported:
            check.side_effect = ValueError("unsafe router")
            with pytest.raises(ValueError, match="unsafe router"):
                await asyncio.wait_for(ctrl._train_pump(), timeout=5)
            ctrl._trainer.get_logprobs_from_meta.assert_not_called()
            ctrl._trainer.get_reference_policy_logprobs_from_meta.assert_not_called()
            ctrl._trainer.train_microbatches_from_meta.assert_not_called()
            ctrl._trainer.finish_train_step.assert_not_called()
            assert sorted(plane.list_sample_ids("train")) == sorted(original_ids)
        else:

            def learner(batch: Any, *args: Any, **kwargs: Any) -> None:
                assert batch.sample_ids == original_ids and batch.tags == meta.tags
                data = plane.get_samples(
                    batch.sample_ids,
                    "train",
                    ["total_reward", "sample_mask", "token_mask", "advantages"],
                )
                assert data["total_reward"].tolist() == [1, 1, 1, 2, 0, 0]
                assert data["sample_mask"].tolist() == [1, 1, 1, 0, 0, 0]
                for index in range(6):
                    mask = data["token_mask"][index].bool()
                    if index < 3:
                        assert torch.all(data["advantages"][index][mask] == -0.5)
                    else:
                        assert not mask.any()

            ctrl._trainer.train_microbatches_from_meta.side_effect = learner
            await asyncio.wait_for(ctrl._train_pump(), timeout=5)
            ctrl._trainer.finish_train_step.assert_called_once()
            assert plane.list_sample_ids("train") == []
            assert ctrl._trainer_version == 1 and ctrl._consumed_samples == 1
        check.assert_called_once_with("validate_cc_execution_padding")

    asyncio.run(exercise())
