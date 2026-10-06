"""CPU queue-recycle contracts, including Miles' prefetch/version boundary."""

import asyncio
import copy
import io
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from omegaconf import OmegaConf
from pydantic import TypeAdapter, ValidationError

import nemo_rl.algorithms.async_utils.replay_buffer as replay
import nemo_rl.algorithms.single_controller as controller
from nemo_rl.algorithms.async_utils.replay_buffer import (
    QUEUE_DEQUEUE_VERSION,
    QUEUE_TRAIN_VERSION,
    DataPlaneCheckpointBarrier,
    TQReplayBuffer,
)
from nemo_rl.algorithms.async_utils.staleness_sampler import (
    QueueRecycleSampler,
    QueueRecycleSamplerConfig,
    SamplerConfig,
    WindowedSampler,
    create_sampler,
    required_buffer_capacity_for_config,
)
from nemo_rl.algorithms.single_controller_utils.config import (
    MasterConfig,
    validate_single_controller_config,
)
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.experience.interfaces import PromptGroupRecord
from nemo_rl.experience.metric_utils import calculate_staleness_metrics
from nemo_rl.experience.rollout_manager import RolloutOutcome
from nemo_rl.experience.rollout_recovery import RolloutRecoveryLedger
from nemo_rl.utils.config import load_config, register_omegaconf_resolvers
from tests.unit.single_controller.test_single_controller_actor import (
    _train_pump_controller,
)


class MemoryDataPlane:
    """Mock only storage RPCs; exercise the real buffer and sampler."""

    def __init__(self):
        self.rows = set()
        self.fail_clear = False

    def put_samples(self, *, sample_ids, **kwargs):
        self.rows.update(sample_ids)

    def clear_samples(self, *, sample_ids, **kwargs):
        if self.fail_clear:
            raise OSError("injected cleanup failure")
        self.rows.difference_update(sample_ids)


@pytest.fixture(autouse=True)
def tensorize(monkeypatch):
    def convert(record, **kwargs):
        return BatchedDataDict(
            {
                "input_ids": torch.ones((2, 3), dtype=torch.long),
                "input_lengths": torch.tensor([3, 3]),
                "total_reward": torch.tensor([1.0, 0.0]),
            }
        )

    monkeypatch.setattr(replay, "record_to_train_batch", convert)


def buffer_and_sampler(capacity=8, bound=8):
    dp = MemoryDataPlane()
    buffer = TQReplayBuffer(
        dp, "rollouts", pad_value_dict={}, include_message_violation_fields=False
    )
    buffer.set_data_plane_checkpoint_barrier(DataPlaneCheckpointBarrier())
    buffer.set_trainer_version_provider(lambda: 17)
    buffer.enable_queue_recycle(capacity)
    return buffer, QueueRecycleSampler(buffer, max_staleness_versions=bound), dp


async def publish(buffer, name, version, idx=0, *, reserve=True):
    if reserve:
        buffer.reserve(group_id=name, weight_version=version)
    record = PromptGroupRecord(
        prompt_idx=idx,
        prompt=[],
        extra_env_info={"task_source": "math"},
        metadata={"task_name": "math"},
        completions=[],
        rollout_metrics={"reward": 0.5},
    )
    return await buffer.commit(name, record, version, 17)


async def select(sampler, version, count=1):
    return await asyncio.wait_for(
        sampler.select(
            current_train_weight=version,
            min_prompt_groups=count,
            max_prompt_groups=count,
        ),
        timeout=2,
    )


def test_config_and_factory_keep_windowed_as_parent():
    cfg = TypeAdapter(SamplerConfig).validate_python(
        {"name": "queue_recycle", "max_staleness_versions": 8}
    )
    buffer, _, _ = buffer_and_sampler()
    sampler = create_sampler(buffer, cfg, min_groups_for_streaming_train=192)
    assert isinstance(sampler, WindowedSampler)
    assert sampler.max_staleness_versions == 8
    assert not sampler.is_on_policy
    assert sampler.required_buffer_capacity(192) == 1
    assert (
        required_buffer_capacity_for_config(
            cfg, 192, min_groups_for_streaming_train=192
        )
        == 1
    )
    with pytest.raises(ValidationError):
        QueueRecycleSamplerConfig(max_staleness_versions=0)
    with pytest.raises(ValidationError):
        QueueRecycleSamplerConfig(sample_freshest_first=True)
    # Reject the previous local name rather than silently using the default.
    with pytest.raises(ValidationError):
        QueueRecycleSamplerConfig(max_weight_staleness=7)


@pytest.mark.parametrize("use_nemo_gym", [False, True])
def test_full_recipe_schema_supports_cycling_and_guards_partial_batches(
    use_nemo_gym: bool,
) -> None:
    register_omegaconf_resolvers()
    config = load_config(
        str(
            Path(__file__).resolve().parents[3]
            / "examples/configs/grpo_math_1B_megatron_single_controller.yaml"
        )
    )
    config.grpo.max_num_epochs = None
    config.async_rl.sampler = {"name": "queue_recycle", "max_staleness_versions": 8}
    config.async_rl.min_groups_for_streaming_train = config.grpo.num_prompts_per_step
    config.policy.train_global_batch_size = (
        config.grpo.num_prompts_per_step * config.grpo.num_generations_per_prompt
    )
    config.loss_fn.force_on_policy_ratio = True
    config.loss_fn.use_importance_sampling_correction = True
    master = MasterConfig(**OmegaConf.to_container(config, resolve=True))
    master.env["should_use_nemo_gym"] = use_nemo_gym
    master.policy["generation"]["vllm_cfg"]["expose_http_server"] = use_nemo_gym
    validate_single_controller_config(master)
    assert master.grpo.max_num_epochs is None
    assert master.loss_fn.force_on_policy_ratio
    assert master.loss_fn.use_importance_sampling_correction
    capture_config = master.model_copy(deep=True)
    capture_config.token_capture.enabled = True
    with pytest.raises(ValueError, match="GRPO without token_capture"):
        validate_single_controller_config(capture_config)
    master.async_rl.min_groups_for_streaming_train = 1
    with pytest.raises(ValueError, match="complete prompt batches"):
        validate_single_controller_config(master)


@pytest.mark.asyncio
async def test_completion_fifo_not_submission_or_freshness_order():
    buffer, sampler, _ = buffer_and_sampler()
    buffer.reserve(group_id="slow", weight_version=16)
    buffer.reserve(group_id="fast", weight_version=15)
    await publish(buffer, "fast", 15, reserve=False)
    await publish(buffer, "slow", 16, reserve=False)
    await publish(buffer, "fresh", 17)
    meta, count = await select(sampler, 17, count=2)
    assert count == 2
    assert meta.sample_ids == ["fast_g0", "fast_g1", "slow_g0", "slow_g1"]
    assert buffer.group_ids == ("fresh",)


@pytest.mark.asyncio
async def test_strict_dequeue_boundary_then_one_update_allows_training_lag_eight():
    buffer, sampler, dp = buffer_and_sampler()
    await publish(buffer, "gap8", 9, idx=9)
    await publish(buffer, "gap7", 10, idx=10)
    # Miles starts selecting batch k+1 while batch k trains at version 17.
    sampler.start_prefetch(dequeue_version=17, num_groups=1)
    await asyncio.wait_for(sampler.finish_prefetch(), timeout=2)
    assert buffer.peek_recycled_prompt().prompt_ref.sample_id == "9"
    assert not {"gap8_g0", "gap8_g1"} & dp.rows
    assert dp.rows == {"gap7_g0", "gap7_g1"}
    group = buffer.queue_training_groups(18)[0]
    assert group["meta"].extra_info[QUEUE_DEQUEUE_VERSION] == 17
    assert group["meta"].extra_info[QUEUE_TRAIN_VERSION] == 18
    # Publishing version 18 must NOT trigger a second gap<8 rejection.
    meta, count = await select(sampler, 18)
    assert count == 1
    metrics = calculate_staleness_metrics(
        [(sid, tag["weight_version"]) for sid, tag in zip(meta.sample_ids, meta.tags)],
        train_weight_version=18,
    )
    assert metrics["staleness/total/max"] == 8
    assert sampler.drain_metrics()["queue_recycle/recycled_groups"] == 1


@pytest.mark.asyncio
async def test_bootstrap_has_no_intervening_update_and_recycles_only_drained_prefix():
    buffer, sampler, _ = buffer_and_sampler()
    await publish(buffer, "too_old", 9, idx=1)
    await publish(buffer, "accepted", 10, idx=2)
    await publish(buffer, "later_old", 8, idx=3)
    assert await sampler.evict(current_train_weight=17) == 0
    meta, _ = await select(sampler, 17)
    assert meta.sample_ids[0] == "accepted_g0"
    assert buffer.group_ids == ("later_old",)
    assert buffer.peek_recycled_prompt().prompt_ref.sample_id == "1"
    assert not sampler.should_abort_inflight(
        start_weight_version=0, current_train_weight=100
    )


@pytest.mark.asyncio
async def test_capacity_counts_completed_queue_not_reserved_or_claimed_groups():
    buffer, sampler, dp = buffer_and_sampler(capacity=1)
    await publish(buffer, "first", 17)
    buffer.reserve(group_id="second", weight_version=17)
    blocked = asyncio.create_task(publish(buffer, "second", 17, reserve=False))
    await asyncio.sleep(0)
    assert not blocked.done()
    assert len(buffer) == 2  # One complete and one waiting for publication.
    meta, count = await select(sampler, 17, count=2)
    await blocked
    assert count == 2
    assert len(meta.sample_ids) == 4
    assert buffer.group_ids == ()
    assert len(buffer.training_owned_group_ids()) == 2
    # A selected batch no longer holds either of the completed-queue slots.
    await asyncio.wait_for(publish(buffer, "third", 17), timeout=2)
    assert len(dp.rows) == 6


@pytest.mark.asyncio
async def test_publication_cancellation_does_not_leak_capacity():
    buffer, _, _ = buffer_and_sampler(capacity=1)
    await publish(buffer, "first", 17)
    blocked = asyncio.create_task(publish(buffer, "cancelled", 17))
    await asyncio.sleep(0)
    blocked.cancel()
    with pytest.raises(asyncio.CancelledError):
        await blocked
    await buffer.remove_group("cancelled")
    still_blocked = asyncio.create_task(publish(buffer, "third", 17))
    await asyncio.sleep(0)
    assert not still_blocked.done()
    await buffer.recycle_queue_group("first")
    await asyncio.wait_for(still_blocked, timeout=2)


@pytest.mark.asyncio
async def test_cleanup_failure_keeps_original_ownership_without_retry_duplicate():
    buffer, _, dp = buffer_and_sampler()
    await publish(buffer, "old", 9)
    dp.fail_clear = True
    with pytest.raises(RuntimeError, match="canonical cleanup failed"):
        await buffer.recycle_queue_group("old")
    assert buffer.group_ids == ("old",)
    assert buffer.peek_recycled_prompt() is None
    assert dp.rows == {"old_g0", "old_g1"}


async def snapshot(buffer):
    async with buffer.data_plane_checkpoint_barrier.checkpoint():
        state = buffer.metadata_state_dict(
            saved_capacity=3, additional_groups=buffer.training_owned_replay_groups()
        )
        stream = io.BytesIO()
        torch.save(state, stream)
    stream.seek(0)
    return torch.load(stream, weights_only=False)


async def restore(state):
    buffer, sampler, dp = buffer_and_sampler(capacity=3)
    await buffer.load_state_dict(
        state,
        max_groups=7,
        expected_partition_id="rollouts",
        expected_group_size=2,
        expected_manifest_digest=state["manifest_digest"],
    )
    dp.rows = {sid for group in state["groups"] for sid in group["meta"].sample_ids}
    return buffer, sampler, dp


@pytest.mark.asyncio
async def test_checkpoint_preserves_retry_fifo_ready_fifo_and_prefetched_batch():
    buffer, sampler, _ = buffer_and_sampler(capacity=3)
    await publish(buffer, "old0", 8, idx=0)
    await publish(buffer, "old1", 9, idx=1)
    await publish(buffer, "next_batch", 10, idx=2)
    sampler.start_prefetch(dequeue_version=17, num_groups=1)
    await sampler.finish_prefetch()
    await publish(buffer, "ready", 17, idx=3)
    state = await snapshot(buffer)
    loaded, next_sampler, _ = await restore(state)
    meta, _ = await select(next_sampler, 18)
    assert meta.sample_ids == ["next_batch_g0", "next_batch_g1"]
    assert loaded.group_ids == ("ready",)
    async with loaded.data_plane_checkpoint_barrier.mutation() as cut:
        assert loaded.pop_recycled_prompt(cut).prompt_ref.sample_id == "0"
        assert loaded.pop_recycled_prompt(cut).prompt_ref.sample_id == "1"
    assert loaded.peek_recycled_prompt() is None


@pytest.mark.asyncio
async def test_checkpoint_cut_survives_prefetch_before_sidecar_serialization() -> None:
    buffer, sampler, _ = buffer_and_sampler(capacity=3)
    await publish(buffer, "current_batch", 17)
    await select(sampler, 17)
    await publish(buffer, "ready_at_snapshot", 17)

    async with buffer.data_plane_checkpoint_barrier.checkpoint():
        state = buffer.metadata_state_dict(
            saved_capacity=3, additional_groups=buffer.training_owned_replay_groups()
        )
        sampler.start_prefetch(dequeue_version=17, num_groups=1)
        # The checkpoint barrier holds the prefetch claim until the cut is taken.
        await asyncio.sleep(0)
        assert not buffer.queue_training_groups(18)

    await asyncio.wait_for(sampler.finish_prefetch(), timeout=2)
    assert buffer.group_ids == ()
    assert [g["group_id"] for g in buffer.queue_training_groups(18)] == [
        "ready_at_snapshot"
    ]

    # Production writes the replay sidecar after releasing the barrier. Its
    # contents must still describe the cut before the live prefetch advanced.
    stream = io.BytesIO()
    torch.save(state, stream)
    stream.seek(0)
    loaded, resumed, _ = await restore(torch.load(stream, weights_only=False))
    assert loaded.group_ids == ("ready_at_snapshot",)
    assert loaded.training_owned_group_ids() == {"current_batch"}
    assert not loaded.queue_training_groups(18)
    current, _ = await select(resumed, 17)
    assert current.sample_ids == ["current_batch_g0", "current_batch_g1"]
    resumed.start_prefetch(dequeue_version=17, num_groups=1)
    await asyncio.wait_for(resumed.finish_prefetch(), timeout=2)
    next_batch, _ = await select(resumed, 18)
    assert next_batch.sample_ids == ["ready_at_snapshot_g0", "ready_at_snapshot_g1"]


@pytest.mark.asyncio
async def test_checkpoint_during_partial_prefetch_continues_without_duplicate_training():
    buffer, sampler, _ = buffer_and_sampler(capacity=3)
    await publish(buffer, "part1", 10)
    sampler.start_prefetch(dequeue_version=17, num_groups=2)
    for _ in range(100):
        if buffer.queue_training_groups(18):
            break
        await asyncio.sleep(0.001)
    assert len(buffer.queue_training_groups(18)) == 1
    state = await snapshot(buffer)
    await sampler.cancel_prefetch()
    loaded, resumed, _ = await restore(state)
    resumed.start_prefetch(dequeue_version=17, num_groups=2)
    await publish(loaded, "part2", 17)
    await asyncio.wait_for(resumed.finish_prefetch(), timeout=2)
    meta, count = await select(resumed, 18, count=2)
    assert count == 2
    assert meta.sample_ids == ["part1_g0", "part1_g1", "part2_g0", "part2_g1"]


@pytest.mark.asyncio
async def test_invalid_checkpoint_order_is_rejected_before_loading_rows():
    buffer, _, _ = buffer_and_sampler(capacity=3)
    await publish(buffer, "first", 17)
    state = await snapshot(buffer)
    changed = copy.deepcopy(state)
    changed["queue_recycle"]["next_sequence"] = 0
    with pytest.raises(ValueError, match="queue order"):
        await restore(changed)


@pytest.mark.asyncio
@pytest.mark.parametrize("dump_enabled", [False, True])
async def test_controller_prefetches_before_refit_and_logs_training_gap_eight(
    monkeypatch: pytest.MonkeyPatch,
    dump_enabled: bool,
) -> None:
    buffer, sampler, dp = buffer_and_sampler()
    await publish(buffer, "bootstrap", 17)
    await publish(buffer, "next", 10)
    ctrl = _train_pump_controller(sampler=sampler)
    ctrl._buffer = buffer
    ctrl._dp_client = dp
    ctrl._partition_id = "rollouts"
    ctrl._data_plane_checkpoint_barrier = buffer.data_plane_checkpoint_barrier
    ctrl._algo_cfg.num_prompts_per_step = 1
    ctrl._algo_cfg.max_num_steps = 2
    ctrl._async_cfg.min_groups_for_streaming_train = 1
    ctrl._trainer_version = 17
    ctrl._rollout_exhausted.clear()
    ctrl._logger = MagicMock()
    versions = []
    dump_versions: list[tuple[int, int]] = []

    def finish_dump(step: int, rows: int) -> None:
        dump_versions.append((step, ctrl._trainer_version))
        assert rows == 0  # The no-op advantage stage writes no dump rows.
        if step == 0:
            assert [g["group_id"] for g in buffer.queue_training_groups(18)] == ["next"]

    if dump_enabled:
        ctrl._train_data_dump = SimpleNamespace(finish_step=finish_dump)
        ctrl._train_data_dump_rows = 0

    async def sync_weights(**kwargs):
        versions.append(ctrl._trainer_version)
        if dump_enabled:
            assert dump_versions[-1] == (
                ctrl._train_steps - 1,
                ctrl._trainer_version - 1,
            )
        if ctrl._trainer_version == 18:
            # The actual train pump must have selected the NEXT batch before
            # updating the engine, and released only the CURRENT batch.
            assert [g["group_id"] for g in buffer.queue_training_groups(18)] == ["next"]
            assert dp.rows == {"next_g0", "next_g1"}
        return 0

    ctrl._sync_weights = sync_weights
    monkeypatch.setattr(controller.ray, "cluster_resources", lambda: {})
    await asyncio.wait_for(ctrl._train_pump(), timeout=3)
    metrics = [
        call.args[0]
        for call in ctrl._logger.log_metrics.call_args_list
        if "staleness/total/max" in call.args[0]
    ]
    assert versions == [18, 19]
    assert dump_versions == ([(0, 17), (1, 18)] if dump_enabled else [])
    assert [m["staleness/total/max"] for m in metrics] == [0, 8]
    assert metrics[1]["staleness/category/math/total/max"] == 8
    assert not buffer.training_owned_group_ids()
    assert not dp.rows


@pytest.mark.asyncio
async def test_controller_regenerates_original_prompt_before_next_fresh_prompt():
    buffer, sampler, _ = buffer_and_sampler()
    await publish(buffer, "old", 9, idx=0)
    await buffer.recycle_queue_group("old")
    ctrl = _train_pump_controller(sampler=sampler)
    ctrl._buffer = buffer
    ctrl._data_plane_checkpoint_barrier = buffer.data_plane_checkpoint_barrier
    ctrl._async_cfg.max_inflight_prompts = 1
    ctrl._rollout_permitted = asyncio.Event()
    ctrl._rollout_permitted.set()
    ctrl._inflight_rollouts = 0
    ctrl._dispatched_rollouts = set()
    ctrl._inflight_by_group_id = {}
    ctrl._current_epoch = 0
    ctrl._rollout_completion_durations_s = deque()
    ctrl._rollout_queue_wait_durations_s = deque()
    original = {"idx": 0, "task_name": "math", "message_log": ["original"]}
    fresh = {"idx": 1, "task_name": "math", "message_log": ["fresh"]}

    class Loader:
        dataset = [original, fresh]

        def __iter__(self):
            yield BatchedDataDict({key: [value] for key, value in fresh.items()})

    ctrl._dataloader = Loader()
    ledger = RolloutRecoveryLedger()
    generated = []
    finished = asyncio.Event()

    def reserve(cut, prompt, *, target_step):
        return ledger.reserve_group(
            cut,
            prompt_id=str(prompt["idx"]),
            prompt_payload=prompt,
            target_step=target_step,
            expected_generations=2,
            start_weight_version=17,
        ).group_id

    async def generate(prompt, *, lineage_group_id, **kwargs):
        generated.append(prompt)
        if len(generated) == 2:
            ctrl._rollout_permitted.clear()
        await publish(buffer, lineage_group_id, 17, idx=prompt["idx"])
        async with ctrl._data_plane_checkpoint_barrier.mutation() as cut:
            ledger.discard_group(cut, lineage_group_id)
        if len(generated) == 2:
            finished.set()
        return RolloutOutcome.COMMITTED

    ctrl._rollout_manager = SimpleNamespace(
        recovery_ledger=ledger,
        reserve_prompt_group=reserve,
        generate_and_push=generate,
    )
    task = asyncio.create_task(ctrl._rollout_pump())
    try:
        await asyncio.wait_for(finished.wait(), timeout=3)
        assert generated == [original, fresh]
        assert buffer.peek_recycled_prompt() is None
        assert len(ledger) == 0
        assert len(buffer.ready_queue_indices()) == 2
    finally:
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
