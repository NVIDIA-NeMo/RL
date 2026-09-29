"""Actual train-pump/GRPO composition; storage, learner and sampler are doubles."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest
import torch
from tensordict import TensorDict

from nemo_rl.algorithms.async_utils.replay_buffer import TQReplayBuffer
from nemo_rl.algorithms.async_utils.staleness_sampler import InOrderSampler
from nemo_rl.algorithms.single_controller_utils.config import TokenCaptureConfig
from nemo_rl.data_plane import KVBatchMeta
from nemo_rl.data_plane.adapters.noop import NoOpDataPlaneClient
from tests.unit.single_controller.test_logical_advantage import _batch, _controller
from tests.unit.single_controller.test_single_controller_actor import (
    _FullStepSampler,
    _NoOpTrainer,
    _SequenceSampler,
    _train_pump_controller,
)


def _setup(
    monkeypatch,
    *,
    failure=None,
    padding=True,
    batches: list[tuple[KVBatchMeta, TensorDict]] | None = None,
    plane: NoOpDataPlaneClient | None = None,
):
    if batches is None:
        batches = [
            _batch([("a", [(0.0, 2), (1.0, 1)])], padding=padding),
            _batch([("b", [(0.0, 3), (1.0, 2)])], padding=padding),
        ]
    num_prompts = len(batches)
    for meta, _ in batches:
        if meta.sequence_lengths is None:
            meta.sequence_lengths = [3] * meta.size
        for tag in meta.tags:
            tag["weight_version"] = 0
            tag.setdefault("uses_borrowed_input", tag["is_execution_padding"])
    if failure == "missing_version":
        del batches[0][0].tags[0]["weight_version"]
    elif failure == "boolean_version":
        batches[0][0].tags[0]["weight_version"] = False
    elif failure == "duplicate":
        batches[1] = batches[0]
    elif failure == "short":
        batches.pop()
    elif failure == "missing_segment":
        meta, data = batches[0]
        batches[0] = meta.subset([0, 2, 3]), data[[0, 2, 3]]
    metas = [meta for meta, _ in batches]
    sampler = (
        _FullStepSampler(metas[0])
        if failure == "group_tally"
        else _SequenceSampler(metas)
    )
    ctrl = _train_pump_controller(sampler=sampler)
    stage = _controller(*batches[0])
    ctrl._algo_cfg = stage._algo_cfg
    ctrl._algo_cfg.num_prompts_per_step = num_prompts
    ctrl._algo_cfg.max_num_steps = 1
    ctrl._master_config.grpo = ctrl._algo_cfg
    ctrl._master_config.loss_fn = stage._master_config.loss_fn
    ctrl._master_config.policy = {
        "train_global_batch_size": 5 if failure == "gbs" else 2 * num_prompts
    }
    ctrl._master_config.token_capture = TokenCaptureConfig(
        enabled=True, context_compaction=True
    )
    ctrl._async_cfg.rollout_failure.min_step_batch_fraction = 1
    ctrl._advantage_estimator = stage._advantage_estimator
    ctrl._step_log_dict = stage._step_log_dict
    if plane is None:
        plane = NoOpDataPlaneClient()
        plane.register_partition("train", [], 10, ["train"])
    for meta, data in batches:
        meta.partition_id = "train"
        data["prev_logprobs"] = torch.zeros_like(data["token_mask"])
        data["reference_policy_logprobs"] = data["prev_logprobs"].clone()
        data.setdefault("generation_logprobs", data["prev_logprobs"].clone())
        plane.put_samples(meta.sample_ids, "train", data, meta.tags)
    ctrl._dp_client = plane
    ctrl._trainer = MagicMock(spec=_NoOpTrainer)
    ctrl._trainer.worker_group = MagicMock()
    ctrl._trainer.get_logprobs_from_meta = MagicMock()
    ctrl._trainer.get_reference_policy_logprobs_from_meta = MagicMock()
    ctrl._trainer.finish_train_step.return_value = {}
    ctrl._sync_weights = AsyncMock(return_value=0)
    ctrl._logger = MagicMock()
    monkeypatch.setattr(
        "nemo_rl.algorithms.single_controller.ray.cluster_resources", lambda: {}
    )
    monkeypatch.setattr(
        "nemo_rl.algorithms.single_controller.ray.get", lambda refs: None
    )
    return ctrl, metas


@pytest.mark.parametrize("padding", [False, True])
@pytest.mark.parametrize("logprobs", [False, True])
@pytest.mark.parametrize("supported", [False, True])
def test_padding_guard_precedes_every_forward_and_only_checks_dummy_batches(
    monkeypatch, padding, logprobs, supported
):
    ctrl, _ = _setup(monkeypatch, padding=padding)
    ctrl._policy_logprobs_required = logprobs
    ctrl._reference_logprobs_required = logprobs
    group = ctrl._trainer.worker_group
    capability = group.run_all_workers_single_data.return_value
    checks = []

    def await_capability(refs):
        assert refs is capability
        checks.append(len(checks) + 1)
        if not supported:
            raise ValueError("unsafe router")

    monkeypatch.setattr(
        "nemo_rl.algorithms.single_controller.ray.get", await_capability
    )
    if padding and not supported:
        with pytest.raises(ValueError, match="unsafe router"):
            asyncio.run(asyncio.wait_for(ctrl._train_pump(), timeout=3))
        for method in (
            "prepare_for_lp_inference",
            "get_logprobs_from_meta",
            "get_reference_policy_logprobs_from_meta",
            "begin_train_step",
            "train_microbatches_from_meta",
            "finish_train_step",
        ):
            getattr(ctrl._trainer, method).assert_not_called()
        ctrl._sync_weights.assert_not_awaited()
        assert checks == [1]
    else:
        asyncio.run(asyncio.wait_for(ctrl._train_pump(), timeout=3))
        assert checks == ([1, 2] if padding else [])
        assert ctrl._trainer.train_microbatches_from_meta.call_count == 2
        assert ctrl._trainer.get_logprobs_from_meta.call_count == (2 if logprobs else 0)
        ctrl._trainer.finish_train_step.assert_called_once()
    if padding:
        group.run_all_workers_single_data.assert_called_with(
            "validate_cc_execution_padding"
        )
    else:
        group.run_all_workers_single_data.assert_not_called()


@pytest.mark.parametrize("flag", [None, 0, "false"])
@pytest.mark.parametrize("field", ["is_execution_padding", "uses_borrowed_input"])
def test_malformed_input_layout_flag_fails_before_forward(monkeypatch, flag, field):
    ctrl, metas = _setup(monkeypatch, padding=False)
    ctrl._policy_logprobs_required = ctrl._reference_logprobs_required = True
    if flag is None:
        del metas[0].tags[0][field]
    else:
        metas[0].tags[0][field] = flag
    with pytest.raises(ValueError, match="padding flags"):
        asyncio.run(asyncio.wait_for(ctrl._train_pump(), timeout=3))
    ctrl._trainer.prepare_for_lp_inference.assert_not_called()
    ctrl._trainer.get_logprobs_from_meta.assert_not_called()
    ctrl._trainer.get_reference_policy_logprobs_from_meta.assert_not_called()
    ctrl._trainer.train_microbatches_from_meta.assert_not_called()
    ctrl._trainer.finish_train_step.assert_not_called()


@pytest.mark.parametrize("async_versions", [False, True])
def test_complete_logical_batch_streams_all_segments_but_steps_once(
    monkeypatch, async_versions
):
    ctrl, metas = _setup(monkeypatch)
    if async_versions:
        ctrl._trainer_version = 3
        for meta in metas:
            for index, tag in enumerate(meta.tags):
                tag["weight_version"] = index % 3
    versions = [[tag["weight_version"] for tag in meta.tags] for meta in metas]

    def check_advantages(meta: KVBatchMeta, *, train_fields: tuple[str, ...]) -> dict:
        data = ctrl._dp_client.get_samples(
            sample_ids=meta.sample_ids,
            partition_id=meta.partition_id,
            select_fields=["advantages"],
        )
        for index, tag in enumerate(meta.tags):
            advantage = 0 if tag["is_execution_padding"] else tag["logical_slot"] - 0.5
            torch.testing.assert_close(
                data["advantages"][index],
                torch.full_like(data["advantages"][index], advantage),
            )
        return {}

    ctrl._trainer.train_microbatches_from_meta.side_effect = check_advantages

    async def exercise():
        if not async_versions:
            await asyncio.wait_for(ctrl._train_pump(), timeout=3)
            return
        buffer = TQReplayBuffer(
            ctrl._dp_client,
            "train",
            pad_value_dict={},
            include_message_violation_fields=False,
        )
        buffer.set_data_plane_checkpoint_barrier(ctrl._data_plane_checkpoint_barrier)
        for meta, row_versions in zip(metas, versions):
            group_id = meta.tags[0]["dispatch_group_id"]
            buffer.reserve(
                weight_version=min(row_versions),
                target_step=ctrl._trainer_version,
                group_id=group_id,
            )
            async with ctrl._data_plane_checkpoint_barrier.mutation() as cut:
                await buffer.commit_finalized(
                    cut, group_id, meta, min(row_versions), max(row_versions)
                )
        ctrl._buffer = buffer
        ctrl._sampler = InOrderSampler(buffer, max_lookahead_versions=2)
        await asyncio.wait_for(ctrl._train_pump(), timeout=3)
        assert buffer.meta_list == []

    asyncio.run(exercise())
    submitted = [
        call.args[0]
        for call in ctrl._trainer.train_microbatches_from_meta.call_args_list
    ]
    assert [meta.size for meta in submitted] == ([10] if async_versions else [4, 6])
    assert [key for meta in submitted for key in meta.sample_ids] == [
        key for meta in metas for key in meta.sample_ids
    ]
    ctrl._trainer.begin_train_step.assert_called_once_with(None)
    ctrl._trainer.finish_train_step.assert_called_once_with()
    assert ctrl._master_config.policy["train_global_batch_size"] == 4
    assert [tag["weight_version"] for meta in submitted for tag in meta.tags] == [
        value for row_versions in versions for value in row_versions
    ]
    assert ctrl._trainer_version == (4 if async_versions else 1)
    assert ctrl._train_steps == 1
    assert ctrl._consumed_samples == 2  # Existing counter is dataset prompts.
    ctrl._sync_weights.assert_awaited_once()
    assert not ctrl._dp_client.list_sample_ids("train")


@pytest.mark.parametrize(
    "failure,trained_chunks,reason",
    [
        ("missing_version", 0, "integer generation versions"),
        ("boolean_version", 0, "integer generation versions"),
        ("duplicate", 1, "new, complete logical groups"),
        ("short", 1, "before a complete training step"),
        ("group_tally", 0, "new, complete logical groups"),
        ("gbs", 2, "full logical sample count"),
        ("missing_segment", 0, "every segment exactly once"),
    ],
)
def test_invalid_logical_batch_never_completes_optimizer_step(
    monkeypatch, failure, trained_chunks, reason
):
    ctrl, _ = _setup(monkeypatch, failure=failure)
    initial_ids = set(ctrl._dp_client.list_sample_ids("train"))
    with pytest.raises((ValueError, RuntimeError), match=reason):
        asyncio.run(asyncio.wait_for(ctrl._train_pump(), timeout=3))
    assert ctrl._trainer.train_microbatches_from_meta.call_count == trained_chunks
    ctrl._trainer.finish_train_step.assert_not_called()
    ctrl._sync_weights.assert_not_awaited()
    assert ctrl._trainer_version == ctrl._train_steps == ctrl._consumed_samples == 0
    assert set(ctrl._dp_client.list_sample_ids("train")) == initial_ids
