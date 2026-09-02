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

"""Tests for SingleController initialization and pump lifecycle."""

import asyncio
import math
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
import torch
from tensordict import TensorDict

import nemo_rl.algorithms.single_controller as single_controller
from nemo_rl.algorithms.async_utils.replay_buffer import DataPlaneCheckpointBarrier
from nemo_rl.algorithms.async_utils.staleness_sampler import BaseSampler
from nemo_rl.algorithms.grpo import GRPOConfig, _initial_grpo_save_state
from nemo_rl.algorithms.loss import ClippedPGLossConfig
from nemo_rl.algorithms.metric_utils import SetupTimingMetrics
from nemo_rl.algorithms.ppo import PPOConfig
from nemo_rl.algorithms.single_controller import (
    SingleControllerActor,
    _pooled_opd_metrics,
)
from nemo_rl.algorithms.single_controller_utils.config import (
    AdvantageConfig,
    AsyncRLConfig,
    MasterConfig,
)
from nemo_rl.data_plane import KVBatchMeta
from nemo_rl.data_plane.schema import ROLLOUT_METRICS
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.utils.timer import TimeoutChecker, Timer


class FakeWeightSynchronizer:
    pass


class _InitBuffer:
    """Minimal non-optional TQ buffer contract for actor-init tests."""

    def __init__(self) -> None:
        self.checkpoint_barrier: DataPlaneCheckpointBarrier | None = None

    def set_data_plane_checkpoint_barrier(
        self, barrier: DataPlaneCheckpointBarrier
    ) -> None:
        self.checkpoint_barrier = barrier


def _checkpointing_config(tmp_path) -> dict:
    """Minimal checkpointing block for actors built through __init__."""
    return {
        "enabled": False,
        "checkpoint_dir": str(tmp_path / "checkpoints"),
        "metric_name": None,
        "higher_is_better": True,
        "keep_top_k": None,
        "save_period": 10,
        "save_optimizer": True,
        "checkpoint_must_save_by": None,
    }


def _grpo_master_config(tmp_path) -> MasterConfig:
    """A minimal GRPO MasterConfig the real __init__ accepts."""
    return MasterConfig.model_construct(
        policy={
            "train_global_batch_size": 8,
            "generation": {"colocated": {"enabled": False}},
        },
        grpo=GRPOConfig.model_construct(
            num_prompts_per_step=2,
            num_generations_per_prompt=4,
        ),
        loss_fn=ClippedPGLossConfig(force_on_policy_ratio=False),
        async_rl=AsyncRLConfig(
            min_groups_for_streaming_train=1,
            max_buffered_rollouts=4,
        ),
        logger={},
        env={},
        checkpointing=_checkpointing_config(tmp_path),
    )


def _actor_args_for_init(**overrides) -> SimpleNamespace:
    """Minimal actor args for a controller built through the real __init__."""
    tq_buffer = _InitBuffer()
    args = dict(
        partition_id="rollout_data",
        dp_client=None,
        gen_handle=None,
        trainer_handle=None,
        dataloader=None,
        weight_synchronizer=FakeWeightSynchronizer(),
        advantage_estimator=None,
        loss_fn=None,
        tq_buffer=tq_buffer,
        rollout_manager=SimpleNamespace(_tq_buffer=tq_buffer),
        env_handles={},
        fleet_monitor=None,
        generation_router=None,
        train_cluster=None,
        inference_cluster=None,
        save_state=_initial_grpo_save_state(),
        last_checkpoint_path=None,
        data_plane_checkpoint_metadata=None,
    )
    args.update(overrides)
    return SimpleNamespace(**args)


def _init_controller(master_config, actor_args):
    controller_cls = SingleControllerActor.__ray_metadata__.modified_class
    return controller_cls(
        master_config=master_config,
        actor_args=actor_args,
        setup_timing_metrics=SetupTimingMetrics(),
    )


def test_rejects_multiple_optimizer_steps_per_rl_step(monkeypatch) -> None:
    monkeypatch.setattr(single_controller, "Logger", lambda _: object())
    master_config = MasterConfig.model_construct(
        policy={
            "train_global_batch_size": 4,
            "generation": {"colocated": {"enabled": False}},
        },
        grpo=GRPOConfig.model_construct(
            num_prompts_per_step=2,
            num_generations_per_prompt=4,
        ),
        async_rl=AsyncRLConfig(min_groups_for_streaming_train=1),
        logger={},
        env={},
    )
    tq_buffer = _InitBuffer()
    actor_args = SimpleNamespace(
        partition_id="rollout_data",
        dp_client=None,
        gen_handle=None,
        trainer_handle=None,
        dataloader=None,
        weight_synchronizer=None,
        advantage_estimator=None,
        loss_fn=None,
        tq_buffer=tq_buffer,
        rollout_manager=SimpleNamespace(_tq_buffer=tq_buffer),
        env_handles={},
        fleet_monitor=None,
        generation_router=None,
        train_cluster=None,
        inference_cluster=None,
    )
    controller_cls = SingleControllerActor.__ray_metadata__.modified_class

    with pytest.raises(
        ValueError,
        match=(
            r"num_prompts_per_step \* num_generations_per_prompt \(8\) "
            r"must equal policy.train_global_batch_size \(4\)"
        ),
    ):
        controller_cls(
            master_config=master_config,
            actor_args=actor_args,
            setup_timing_metrics=SetupTimingMetrics(),
        )


def test_logs_hyperparameters_and_concrete_weight_synchronizer(
    monkeypatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path,
) -> None:
    logger = MagicMock()
    monkeypatch.setattr(single_controller, "Logger", lambda _: logger)
    master_config = MasterConfig.model_construct(
        policy={
            "train_global_batch_size": 8,
            "generation": {"colocated": {"enabled": False}},
        },
        grpo=GRPOConfig.model_construct(
            num_prompts_per_step=2,
            num_generations_per_prompt=4,
        ),
        loss_fn=ClippedPGLossConfig(force_on_policy_ratio=False),
        async_rl=AsyncRLConfig(
            min_groups_for_streaming_train=1,
            max_buffered_rollouts=4,
        ),
        logger={},
        env={},
        # __init__ builds a CheckpointManager + TimeoutChecker from this block.
        checkpointing=_checkpointing_config(tmp_path),
    )
    actor_args = _actor_args_for_init()
    controller_cls = SingleControllerActor.__ray_metadata__.modified_class

    controller_cls(
        master_config=master_config,
        actor_args=actor_args,
        setup_timing_metrics=SetupTimingMetrics(),
    )

    logger.log_hyperparams.assert_called_once_with(master_config.model_dump())
    output = capsys.readouterr().out
    assert "weight_sync=FakeWeightSynchronizer" in output
    assert "transport=stub" not in output


@pytest.mark.parametrize(
    (
        "reference_policy_kl_penalty",
        "skip_reference_logprobs",
        "force_on_policy_ratio",
        "expected_policy_required",
        "expected_reference_required",
    ),
    [
        (0.0, False, False, True, False),
        (0.0, True, False, True, False),
        (0.01, False, False, True, True),
        (0.01, False, True, False, True),
    ],
)
def test_reference_logprobs_required_only_when_kl_enabled(
    monkeypatch,
    tmp_path,
    reference_policy_kl_penalty: float,
    skip_reference_logprobs: bool,
    force_on_policy_ratio: bool,
    expected_policy_required: bool,
    expected_reference_required: bool,
) -> None:
    """KL-disabled SingleController runs do not request reference logprobs."""
    monkeypatch.setattr(single_controller, "Logger", lambda _: MagicMock())
    master_config = MasterConfig.model_construct(
        policy={
            "train_global_batch_size": 8,
            "generation": {"colocated": {"enabled": False}},
        },
        grpo=GRPOConfig.model_construct(
            num_prompts_per_step=2,
            num_generations_per_prompt=4,
            skip_reference_policy_logprobs_calculation=skip_reference_logprobs,
        ),
        loss_fn=ClippedPGLossConfig(
            force_on_policy_ratio=force_on_policy_ratio,
            reference_policy_kl_penalty=reference_policy_kl_penalty,
        ),
        async_rl=AsyncRLConfig(
            min_groups_for_streaming_train=1,
            max_buffered_rollouts=4,
        ),
        logger={},
        env={},
        checkpointing=_checkpointing_config(tmp_path),
    )

    controller = _init_controller(master_config, _actor_args_for_init())

    assert controller._policy_logprobs_required is expected_policy_required
    assert controller._reference_logprobs_required is expected_reference_required
    assert ("prev_logprobs" in controller._train_fields) is expected_policy_required
    assert (
        "reference_policy_logprobs" in controller._train_fields
    ) is expected_reference_required


@pytest.mark.parametrize("with_critic", [True, False], ids=["ppo", "grpo"])
def test_init_picks_up_the_critic_handles(monkeypatch, tmp_path, with_critic) -> None:
    """The PPO path hands the critic and its loss in through actor_args.

    A GRPO run leaves both unset -- actor_args defaults them to None, and older
    args may omit them entirely.
    """
    monkeypatch.setattr(single_controller, "Logger", lambda _: MagicMock())
    handles = (
        {
            "value_handle": MagicMock(name="value"),
            "value_loss_fn": MagicMock(name="value_loss_fn"),
        }
        if with_critic
        else {}
    )

    ctrl = _init_controller(
        _grpo_master_config(tmp_path), _actor_args_for_init(**handles)
    )

    assert ctrl._value is handles.get("value_handle")
    assert ctrl._value_loss_fn is handles.get("value_loss_fn")


def test_logs_setup_timing_metrics(monkeypatch, tmp_path) -> None:
    """setup_timing_metrics is forwarded to Logger.log_metrics under timing/setup."""
    logger = MagicMock()
    monkeypatch.setattr(single_controller, "Logger", lambda _: logger)
    master_config = MasterConfig.model_construct(
        policy={
            "train_global_batch_size": 8,
            "generation": {"colocated": {"enabled": False}},
        },
        grpo=GRPOConfig.model_construct(
            num_prompts_per_step=2,
            num_generations_per_prompt=4,
        ),
        loss_fn=ClippedPGLossConfig(force_on_policy_ratio=False),
        async_rl=AsyncRLConfig(
            min_groups_for_streaming_train=1,
            max_buffered_rollouts=4,
        ),
        logger={},
        env={},
        # __init__ builds a CheckpointManager + TimeoutChecker from this block.
        checkpointing=_checkpointing_config(tmp_path),
    )
    setup_metrics = SetupTimingMetrics(
        generation_init_time_s=1.5, policy_init_time_s=2.5
    )
    actor_args = _actor_args_for_init()
    controller_cls = SingleControllerActor.__ray_metadata__.modified_class

    controller_cls(
        master_config=master_config,
        actor_args=actor_args,
        setup_timing_metrics=setup_metrics,
    )

    logger.log_metrics.assert_called_once_with(
        setup_metrics.to_metrics_dict(), step=0, prefix="timing/setup"
    )


def _lookahead_controller(
    *,
    trainer_version: int,
    policy_training_start_step: int,
    max_lookahead_versions: int = 1,
    warmup_lookahead_versions: int | None = None,
    is_ppo: bool = True,
):
    """Bare actor carrying only what the lookahead schedule reads."""
    controller_cls = SingleControllerActor.__ray_metadata__.modified_class
    ctrl = object.__new__(controller_cls)
    ctrl._is_ppo = is_ppo
    ctrl._trainer_version = trainer_version
    ctrl._algo_cfg = SimpleNamespace(
        policy_training_start_step=policy_training_start_step
    )
    ctrl._async_cfg = SimpleNamespace(
        sampler=SimpleNamespace(
            max_lookahead_versions=max_lookahead_versions,
            warmup_lookahead_versions=warmup_lookahead_versions,
        )
    )
    ctrl._sampler = MagicMock()
    return ctrl


class TestLookaheadSchedule:
    """Port of ppo.py's _async_ppo_generation_lead_steps.

    Generation may run further ahead while the policy is frozen, then the window
    has to converge back before warmup-era rollouts stop being trainable.
    """

    @pytest.mark.parametrize(
        ("trainer_version", "expected"),
        [
            # start=4, steady=1, warmup=5 -> frontier is 4+1=5.
            (0, 5),  # min(5, 5-0) = 5
            (1, 4),  # min(5, 5-1) = 4, already shrinking
            (3, 2),
            (4, 1),  # warmup over: back to steady
            (9, 1),
        ],
    )
    def test_window_widens_then_converges(self, trainer_version, expected):
        ctrl = _lookahead_controller(
            trainer_version=trainer_version,
            policy_training_start_step=4,
            max_lookahead_versions=1,
            warmup_lookahead_versions=5,
        )

        ctrl._retune_lookahead_versions()

        ctrl._sampler.set_gate_window.assert_called_once_with(expected)

    def test_steady_value_is_a_defensive_floor(self):
        """Kept because ppo.py has it, but unreachable through a valid config.

        The floor only binds when warmup < steady, which
        InOrderSamplerConfig.validate_warmup_lookahead already rejects -- inside
        the warmup branch the frontier term is always greater than steady. This
        builds the config by hand to reach it at all.
        """
        ctrl = _lookahead_controller(
            trainer_version=2,
            policy_training_start_step=100,
            max_lookahead_versions=3,
            warmup_lookahead_versions=2,
        )

        ctrl._retune_lookahead_versions()

        ctrl._sampler.set_gate_window.assert_called_once_with(3)

    def test_unset_warmup_window_pins_the_steady_value(self):
        ctrl = _lookahead_controller(
            trainer_version=0,
            policy_training_start_step=4,
            max_lookahead_versions=2,
            warmup_lookahead_versions=None,
        )

        ctrl._retune_lookahead_versions()

        ctrl._sampler.set_gate_window.assert_called_once_with(2)

    def test_retune_is_a_noop_off_the_ppo_path(self):
        ctrl = _lookahead_controller(
            trainer_version=0,
            policy_training_start_step=0,
            is_ppo=False,
        )

        ctrl._retune_lookahead_versions()

        ctrl._sampler.set_gate_window.assert_not_called()


@pytest.mark.parametrize(
    ("recompute_kv_cache", "expected_invalidation_calls"),
    [(False, 0), (True, 1)],
)
def test_sync_weights_honors_recompute_kv_cache_config(
    recompute_kv_cache: bool,
    expected_invalidation_calls: int,
) -> None:
    controller_cls = SingleControllerActor.__ray_metadata__.modified_class
    ctrl = object.__new__(controller_cls)
    ctrl._async_cfg = AsyncRLConfig(
        recompute_kv_cache_after_weight_updates=recompute_kv_cache
    )
    ctrl._rollout_permitted = asyncio.Event()
    ctrl._rollout_permitted.set()
    # No fleet health: _sync_weights reconciles refit membership first, and with no
    # monitor there is nothing to reconcile.
    ctrl._gen_fleet = None
    ctrl._weight_synchronizer = SimpleNamespace(sync_weights=MagicMock())
    ctrl._gen = SimpleNamespace(
        invalidate_kv_cache=MagicMock(),
        requires_kv_scale_sync=False,
    )
    ctrl._inflight_by_group_id = {}
    ctrl._rollout_recovery_enabled = False
    # env={} -> should_use_nemo_gym is False, so _sync_weights takes the native
    # abort path (empty registry -> no-op) instead of the gym gate.
    ctrl._master_config = SimpleNamespace(env={})

    asyncio.run(ctrl._sync_weights())

    ctrl._weight_synchronizer.sync_weights.assert_called_once_with(kv_scales=None)
    assert ctrl._gen.invalidate_kv_cache.call_count == expected_invalidation_calls
    assert ctrl._rollout_permitted.is_set()


def test_sync_weights_calibrates_and_forwards_fp8_kv_scales() -> None:
    controller_cls = SingleControllerActor.__ray_metadata__.modified_class
    ctrl = object.__new__(controller_cls)
    ctrl._async_cfg = AsyncRLConfig()
    ctrl._rollout_permitted = asyncio.Event()
    ctrl._rollout_permitted.set()
    # No fleet health: _sync_weights reconciles refit membership first, and with no
    # monitor there is nothing to reconcile.
    ctrl._gen_fleet = None
    ctrl._weight_synchronizer = SimpleNamespace(sync_weights=MagicMock())
    ctrl._gen = SimpleNamespace(
        invalidate_kv_cache=MagicMock(),
        requires_kv_scale_sync=True,
    )
    ctrl._trainer = SimpleNamespace(
        calibrate_qkv_fp8_scales=MagicMock(return_value={"layers": {"layer.0": 0.5}})
    )
    ctrl._inflight_by_group_id = {}
    ctrl._rollout_recovery_enabled = False
    # env={} -> should_use_nemo_gym is False, so _sync_weights takes the native
    # abort path (empty registry -> no-op) instead of the gym gate.
    ctrl._master_config = SimpleNamespace(env={})
    calibration_data = BatchedDataDict(
        {
            "input_ids": torch.tensor([[1, 2]]),
            "input_lengths": torch.tensor([2]),
        }
    )

    asyncio.run(ctrl._sync_weights(calibration_data=calibration_data))

    ctrl._trainer.calibrate_qkv_fp8_scales.assert_called_once_with(
        calibration_data,
        include_q=True,
    )
    ctrl._weight_synchronizer.sync_weights.assert_called_once_with(
        kv_scales={"layer.0": 0.5}
    )


class _AdvantageDataPlane:
    def __init__(self, data: TensorDict) -> None:
        self._data = data
        self.selected_fields: list[str] | None = None
        self.written_fields: TensorDict | None = None

    def get_samples(self, *, select_fields, **kwargs):
        del kwargs
        self.selected_fields = list(select_fields)
        return self._data

    def put_samples(self, *, fields, **kwargs) -> None:
        del kwargs
        self.written_fields = fields


class _MaskRecordingAdvantageEstimator:
    def __init__(self) -> None:
        self.mask: torch.Tensor | None = None

    def compute_advantage(self, *, rewards, mask, **kwargs) -> torch.Tensor:
        del kwargs
        self.mask = mask.clone()
        return rewards.unsqueeze(-1).expand_as(mask).clone()


def test_advantage_stage_composes_all_filters_before_computing_advantages(
    capsys: pytest.CaptureFixture[str],
) -> None:
    batch_size, sequence_length = 4, 5
    generation_logprobs = torch.zeros(batch_size, sequence_length)
    # Rows 1, 2, and 3 are removed by the environment, sequence-error,
    # and overlong masks respectively. Row 0 remains trainable.
    generation_logprobs[2, 1:] = 1.0
    data = TensorDict(
        {
            "prompt_ids_for_adv": torch.zeros(
                batch_size, sequence_length, dtype=torch.long
            ),
            "total_reward": torch.tensor([0.0, 0.0, 1.0, 0.0]),
            "token_mask": torch.ones(batch_size, sequence_length),
            "sample_mask": torch.ones(batch_size),
            "mask_sample": torch.tensor([False, True, False, False]),
            "truncated": torch.tensor([False, False, False, True]),
            "prev_logprobs": torch.zeros(batch_size, sequence_length),
            "generation_logprobs": generation_logprobs,
            # The sequence-error- and overlong-filtered rows are also flagged.
            # Their penalties must not leak back into streaming training.
            "invalid_tool_call_mask": torch.tensor(
                [[False] * sequence_length] * 2 + [[True] * sequence_length] * 2
            ),
            "malformed_thinking_mask": torch.zeros(batch_size, sequence_length),
        },
        batch_size=[batch_size],
    )
    data_plane = _AdvantageDataPlane(data)
    estimator = _MaskRecordingAdvantageEstimator()

    controller_cls = SingleControllerActor.__ray_metadata__.modified_class
    ctrl = object.__new__(controller_cls)
    ctrl._dp_client = data_plane
    ctrl._advantage_cfg = AdvantageConfig()
    ctrl._advantage_estimator = estimator
    ctrl._policy_logprobs_required = True
    ctrl._reference_logprobs_required = False
    ctrl._teacher_logprobs_required = False
    ctrl._is_ppo = False
    ctrl._master_config = SimpleNamespace(
        grpo=SimpleNamespace(
            seq_logprob_error_threshold=2.0,
            overlong_filtering=True,
            invalid_tool_call_advantage=-5.0,
            malformed_thinking_advantage=None,
        )
    )
    ctrl._algo_cfg = ctrl._master_config.grpo
    ctrl._message_level_advantage_penalties_enabled = True
    ctrl._step_log_dict = {
        "rewards": [],
        "masked_advantages": [],
        "sequence_lengths": [],
        "num_mask_sample_filtered": [],
        "seq_logprob_error_metrics": [],
    }
    meta = KVBatchMeta(
        partition_id="rollout_data",
        task_name="train",
        sample_ids=[f"sample-{i}" for i in range(batch_size)],
        fields=list(data.keys()),
    )

    result_meta, has_valid_training_tokens = asyncio.run(ctrl._advantage_stage(meta))
    capsys.readouterr()

    assert has_valid_training_tokens
    assert data_plane.selected_fields is not None
    assert "prev_logprobs" in data_plane.selected_fields
    assert "invalid_tool_call_mask" in data_plane.selected_fields
    assert "generation_logprobs" in data_plane.selected_fields
    assert data_plane.written_fields is not None
    # The estimator's values remain, but the penalty did not overwrite them with
    # -5; sample_mask below is what excludes these rows from streaming training.
    torch.testing.assert_close(
        data_plane.written_fields["advantages"][2], torch.ones(5)
    )
    torch.testing.assert_close(
        data_plane.written_fields["advantages"][3], torch.zeros(5)
    )
    assert torch.equal(
        data_plane.written_fields["sample_mask"],
        torch.tensor([1.0, 0.0, 0.0, 0.0]),
    )
    assert estimator.mask is not None
    assert estimator.mask[0].all()
    assert estimator.mask[1:].count_nonzero() == 0
    assert ctrl._step_log_dict["num_mask_sample_filtered"] == [1]
    metrics = ctrl._step_log_dict["seq_logprob_error_metrics"]
    assert len(metrics) == 1
    assert metrics[0]["num_masked_seqs_by_logprob_error"] == 1
    assert metrics[0]["max_seq_mult_prob_error"] == pytest.approx(math.e)
    assert metrics[0]["max_seq_mult_prob_error_after_mask"] == pytest.approx(1.0)
    assert "advantages" in (result_meta.fields or [])


@pytest.mark.parametrize(
    "overlong_filtering, mask_sample, truncated, expected_sample_mask",
    [
        (False, [True, False], [True, True], [0.0, 1.0]),
        (True, [False, False], [False, True], [1.0, 0.0]),
    ],
    ids=["env_mask_only", "overlong_only"],
)
def test_advantage_stage_writes_each_sample_filter_without_seq_threshold(
    overlong_filtering: bool,
    mask_sample: list[bool],
    truncated: list[bool],
    expected_sample_mask: list[float],
) -> None:
    batch_size, sequence_length = 2, 5
    data = TensorDict(
        {
            "prompt_ids_for_adv": torch.zeros(
                batch_size, sequence_length, dtype=torch.long
            ),
            "total_reward": torch.tensor([1.0, 0.0]),
            "token_mask": torch.ones(batch_size, sequence_length),
            "sample_mask": torch.ones(batch_size),
            "mask_sample": torch.tensor(mask_sample),
            "truncated": torch.tensor(truncated),
        },
        batch_size=[batch_size],
    )
    data_plane = _AdvantageDataPlane(data)
    estimator = _MaskRecordingAdvantageEstimator()

    controller_cls = SingleControllerActor.__ray_metadata__.modified_class
    ctrl = object.__new__(controller_cls)
    ctrl._dp_client = data_plane
    ctrl._advantage_cfg = AdvantageConfig()
    ctrl._advantage_estimator = estimator
    ctrl._policy_logprobs_required = False
    ctrl._reference_logprobs_required = False
    ctrl._teacher_logprobs_required = False
    ctrl._is_ppo = False
    ctrl._message_level_advantage_penalties_enabled = False
    ctrl._algo_cfg = SimpleNamespace(
        seq_logprob_error_threshold=None,
        overlong_filtering=overlong_filtering,
    )
    ctrl._step_log_dict = {
        "rewards": [],
        "masked_advantages": [],
        "num_mask_sample_filtered": [],
        "sequence_lengths": [],
        "seq_logprob_error_metrics": [],
    }
    meta = KVBatchMeta(
        partition_id="rollout_data",
        task_name="train",
        sample_ids=[f"sample-{i}" for i in range(batch_size)],
        fields=list(data.keys()),
    )

    _, has_valid_training_tokens = asyncio.run(ctrl._advantage_stage(meta))

    expected = torch.tensor(expected_sample_mask)
    assert has_valid_training_tokens
    assert data_plane.written_fields is not None
    assert torch.equal(data_plane.written_fields["sample_mask"], expected)
    assert estimator.mask is not None
    assert torch.equal(
        estimator.mask,
        data["token_mask"] * expected.unsqueeze(-1),
    )


def test_advantage_stage_reports_seq_logprob_metrics_without_masking() -> None:
    batch_size, sequence_length = 2, 5
    generation_logprobs = torch.zeros(batch_size, sequence_length)
    generation_logprobs[1, 1:] = 1.0
    data = TensorDict(
        {
            "prompt_ids_for_adv": torch.zeros(
                batch_size, sequence_length, dtype=torch.long
            ),
            "total_reward": torch.tensor([0.0, 1.0]),
            "token_mask": torch.ones(batch_size, sequence_length),
            "sample_mask": torch.ones(batch_size),
            "prev_logprobs": torch.zeros(batch_size, sequence_length),
            "generation_logprobs": generation_logprobs,
            "mask_sample": torch.zeros(batch_size, dtype=torch.bool),
            "truncated": torch.tensor([False, True]),
        },
        batch_size=[batch_size],
    )
    data_plane = _AdvantageDataPlane(data)
    estimator = _MaskRecordingAdvantageEstimator()

    controller_cls = SingleControllerActor.__ray_metadata__.modified_class
    ctrl = object.__new__(controller_cls)
    ctrl._dp_client = data_plane
    ctrl._advantage_cfg = AdvantageConfig()
    ctrl._advantage_estimator = estimator
    ctrl._policy_logprobs_required = True
    ctrl._reference_logprobs_required = False
    ctrl._teacher_logprobs_required = False
    ctrl._is_ppo = False
    ctrl._master_config = SimpleNamespace(
        grpo=SimpleNamespace(seq_logprob_error_threshold=None, overlong_filtering=False)
    )
    ctrl._algo_cfg = ctrl._master_config.grpo
    ctrl._message_level_advantage_penalties_enabled = False
    ctrl._step_log_dict = {
        "rewards": [],
        "masked_advantages": [],
        "num_mask_sample_filtered": [],
        "sequence_lengths": [],
        "seq_logprob_error_metrics": [],
    }
    meta = KVBatchMeta(
        partition_id="rollout_data",
        task_name="train",
        sample_ids=[f"sample-{i}" for i in range(batch_size)],
        fields=list(data.keys()),
    )

    _, has_valid_training_tokens = asyncio.run(ctrl._advantage_stage(meta))

    assert has_valid_training_tokens
    assert data_plane.selected_fields is not None
    assert "prev_logprobs" in data_plane.selected_fields
    assert "generation_logprobs" in data_plane.selected_fields
    assert data_plane.written_fields is not None
    assert "sample_mask" not in data_plane.written_fields
    assert estimator.mask is not None
    assert estimator.mask.all()
    metrics = ctrl._step_log_dict["seq_logprob_error_metrics"]
    assert len(metrics) == 1
    assert metrics[0]["num_masked_seqs_by_logprob_error"] == 0
    assert ctrl._step_log_dict["num_mask_sample_filtered"] == [0]
    assert metrics[0]["max_seq_mult_prob_error"] == pytest.approx(math.e)
    assert metrics[0]["max_seq_mult_prob_error_after_mask"] == pytest.approx(math.e)


def test_advantage_stage_skips_estimator_when_seq_mask_removes_whole_chunk(
    capsys: pytest.CaptureFixture[str],
) -> None:
    batch_size, sequence_length = 2, 5
    data = TensorDict(
        {
            "prompt_ids_for_adv": torch.zeros(
                batch_size, sequence_length, dtype=torch.long
            ),
            "total_reward": torch.tensor([1.0, 0.0]),
            "token_mask": torch.ones(batch_size, sequence_length),
            "sample_mask": torch.ones(batch_size),
            "prev_logprobs": torch.zeros(batch_size, sequence_length),
            "generation_logprobs": torch.ones(batch_size, sequence_length),
            "mask_sample": torch.zeros(batch_size, dtype=torch.bool),
            "truncated": torch.zeros(batch_size, dtype=torch.bool),
        },
        batch_size=[batch_size],
    )
    data_plane = _AdvantageDataPlane(data)
    estimator = MagicMock()

    controller_cls = SingleControllerActor.__ray_metadata__.modified_class
    ctrl = object.__new__(controller_cls)
    ctrl._dp_client = data_plane
    ctrl._advantage_cfg = AdvantageConfig()
    ctrl._advantage_estimator = estimator
    ctrl._policy_logprobs_required = True
    ctrl._reference_logprobs_required = False
    ctrl._teacher_logprobs_required = False
    ctrl._is_ppo = False
    ctrl._master_config = SimpleNamespace(
        grpo=SimpleNamespace(seq_logprob_error_threshold=2.0, overlong_filtering=False)
    )
    ctrl._algo_cfg = ctrl._master_config.grpo
    ctrl._message_level_advantage_penalties_enabled = False
    ctrl._step_log_dict = {
        "rewards": [],
        "masked_advantages": [],
        "num_mask_sample_filtered": [],
        "sequence_lengths": [],
        "seq_logprob_error_metrics": [],
    }
    meta = KVBatchMeta(
        partition_id="rollout_data",
        task_name="train",
        sample_ids=[f"sample-{i}" for i in range(batch_size)],
        fields=list(data.keys()),
    )

    result_meta, has_valid_training_tokens = asyncio.run(ctrl._advantage_stage(meta))
    capsys.readouterr()

    assert not has_valid_training_tokens
    estimator.compute_advantage.assert_not_called()
    assert data_plane.written_fields is not None
    assert not data_plane.written_fields["sample_mask"].bool().any()
    assert torch.equal(
        data_plane.written_fields["advantages"],
        torch.zeros(batch_size, sequence_length),
    )
    assert "advantages" in (result_meta.fields or [])


def test_advantage_stage_skips_preexisting_empty_mask_without_seq_threshold() -> None:
    batch_size, sequence_length = 2, 5
    data = TensorDict(
        {
            "prompt_ids_for_adv": torch.zeros(
                batch_size, sequence_length, dtype=torch.long
            ),
            "total_reward": torch.tensor([1.0, 0.0]),
            "token_mask": torch.ones(batch_size, sequence_length),
            "sample_mask": torch.zeros(batch_size),
            "mask_sample": torch.zeros(batch_size, dtype=torch.bool),
            "truncated": torch.zeros(batch_size, dtype=torch.bool),
        },
        batch_size=[batch_size],
    )
    data_plane = _AdvantageDataPlane(data)
    estimator = MagicMock()

    controller_cls = SingleControllerActor.__ray_metadata__.modified_class
    ctrl = object.__new__(controller_cls)
    ctrl._dp_client = data_plane
    ctrl._advantage_cfg = AdvantageConfig()
    ctrl._advantage_estimator = estimator
    ctrl._policy_logprobs_required = False
    ctrl._reference_logprobs_required = False
    ctrl._teacher_logprobs_required = False
    ctrl._is_ppo = False
    ctrl._master_config = SimpleNamespace(
        grpo=SimpleNamespace(seq_logprob_error_threshold=None, overlong_filtering=False)
    )
    ctrl._algo_cfg = ctrl._master_config.grpo
    ctrl._message_level_advantage_penalties_enabled = False
    ctrl._step_log_dict = {
        "rewards": [],
        "masked_advantages": [],
        "num_mask_sample_filtered": [],
        "sequence_lengths": [],
        "seq_logprob_error_metrics": [],
    }
    meta = KVBatchMeta(
        partition_id="rollout_data",
        task_name="train",
        sample_ids=[f"sample-{i}" for i in range(batch_size)],
        fields=list(data.keys()),
    )

    result_meta, has_valid_training_tokens = asyncio.run(ctrl._advantage_stage(meta))

    assert not has_valid_training_tokens
    estimator.compute_advantage.assert_not_called()
    assert data_plane.selected_fields is not None
    assert "prev_logprobs" not in data_plane.selected_fields
    assert "generation_logprobs" not in data_plane.selected_fields
    assert data_plane.written_fields is not None
    assert "sample_mask" not in data_plane.written_fields
    assert torch.equal(
        data_plane.written_fields["advantages"],
        torch.zeros(batch_size, sequence_length),
    )
    assert "advantages" in (result_meta.fields or [])


def test_opd_advantage_stage_reads_teacher_and_student_logprobs() -> None:
    """SC passes the TQ teacher column under OPD's estimator contract."""
    controller_cls = SingleControllerActor.__ray_metadata__.modified_class
    ctrl = object.__new__(controller_cls)
    captured_kwargs = {}

    class FakeEstimator:
        def compute_advantage(self, **kwargs):
            captured_kwargs.update(kwargs)
            return kwargs["teacher_logprobs"] - kwargs["prev_logprobs"]

    class FakeDataPlane:
        def __init__(self):
            self.put_fields = None

        def get_samples(self, sample_ids, partition_id, select_fields):
            del sample_ids, partition_id
            assert "teacher_reference_logprobs" in select_fields
            assert "generation_logprobs" in select_fields
            return TensorDict(
                {
                    "prompt_ids_for_adv": torch.zeros(2, 3, dtype=torch.long),
                    "total_reward": torch.zeros(2),
                    "token_mask": torch.tensor([[1.0, 1.0, 1.0], [1.0, 0.0, 0.0]]),
                    "sample_mask": torch.ones(2),
                    "mask_sample": torch.zeros(2, dtype=torch.bool),
                    "truncated": torch.zeros(2, dtype=torch.bool),
                    "generation_logprobs": torch.full((2, 3), 0.5),
                    "prev_logprobs": torch.full((2, 3), 0.5),
                    "teacher_reference_logprobs": torch.full((2, 3), 0.75),
                },
                batch_size=(2,),
            )

        def put_samples(self, sample_ids, partition_id, fields):
            del sample_ids, partition_id
            self.put_fields = fields

    ctrl._advantage_cfg = AdvantageConfig()
    ctrl._advantage_estimator = FakeEstimator()
    ctrl._policy_logprobs_required = True
    ctrl._reference_logprobs_required = False
    ctrl._teacher_logprobs_required = True
    ctrl._is_ppo = False
    ctrl._dp_client = FakeDataPlane()
    ctrl._master_config = SimpleNamespace(
        grpo=SimpleNamespace(seq_logprob_error_threshold=None, overlong_filtering=False)
    )
    ctrl._algo_cfg = ctrl._master_config.grpo
    ctrl._message_level_advantage_penalties_enabled = False
    ctrl._step_log_dict = {
        "rewards": [],
        "masked_advantages": [],
        "sequence_lengths": [],
        "seq_logprob_error_metrics": [],
        "num_mask_sample_filtered": [],
    }
    ctrl._opd_stat_sum = 0.0
    ctrl._opd_stat_sumsq = 0.0
    ctrl._opd_stat_count = 0
    meta = KVBatchMeta(
        partition_id="rollout_data",
        task_name="train",
        sample_ids=["a", "b"],
        fields=[],
        sequence_lengths=[3, 3],
    )

    enriched, has_valid_training_tokens = asyncio.run(ctrl._advantage_stage(meta))

    assert has_valid_training_tokens
    assert set(captured_kwargs) >= {
        "teacher_logprobs",
        "prev_logprobs",
        "prompt_ids",
        "rewards",
        "mask",
        "repeated_batch",
    }
    assert "logprobs_policy" not in captured_kwargs
    assert torch.allclose(
        captured_kwargs["teacher_logprobs"] - captured_kwargs["prev_logprobs"],
        torch.full((2, 3), 0.25),
    )
    assert "advantages" in (enriched.fields or [])
    assert ctrl._opd_stat_sum == pytest.approx(1.0)
    assert ctrl._opd_stat_sumsq == pytest.approx(0.25)
    assert ctrl._opd_stat_count == 4


def test_pooled_opd_metrics_weight_unequal_chunks_by_valid_token_count() -> None:
    """A small streaming chunk cannot receive the same weight as a large one."""
    # Chunk 1 has values [0, 2]; chunk 2 has [4]. Averaging chunk means
    # would incorrectly produce 2.5. Exact pooling produces mean=2, std=2.
    metrics = _pooled_opd_metrics(
        stat_sum=6.0,
        stat_sumsq=20.0,
        count=3,
    )

    assert metrics == pytest.approx(
        {
            "on_policy_distillation/teacher_student_logprob_gap_mean": 2.0,
            "on_policy_distillation/adv_mean": 2.0,
            "on_policy_distillation/adv_std": 2.0,
        }
    )


class _EmptySampler:
    async def evict(self, *, current_train_weight: int) -> int:
        del current_train_weight
        return 0

    def set_gate_window(self, gate_window: int) -> None:
        self.gate_window = gate_window

    async def select(self, **kwargs):
        del kwargs
        return None, 0


class _OneThenEmptySampler(_EmptySampler):
    def __init__(self, meta: KVBatchMeta) -> None:
        self._meta: KVBatchMeta | None = meta

    async def select(self, **kwargs):
        del kwargs
        if self._meta is None:
            return None, 0
        meta = self._meta
        self._meta = None
        return meta, 1


class _EvictingSampler(_OneThenEmptySampler):
    async def evict(self, *, current_train_weight: int) -> int:
        del current_train_weight
        return 2

    async def select(self, **kwargs):
        meta, num_groups = await super().select(**kwargs)
        return meta, 2 if num_groups else 0


class _FullStepSampler(_OneThenEmptySampler):
    async def select(self, **kwargs):
        meta, num_groups = await super().select(**kwargs)
        return meta, 2 if num_groups else 0


class _ChunkedSampler(_EmptySampler):
    """Assembles one step out of several single-group chunks, then goes empty.

    This is the shape the streaming path actually produces and the reason
    ``keep_train_buffers`` exists: every chunk after the first runs against an
    already-open train step.
    """

    def __init__(self, meta: KVBatchMeta, chunks: int) -> None:
        self._meta = meta
        self._remaining = chunks

    async def select(self, **kwargs):
        del kwargs
        if self._remaining == 0:
            return None, 0
        self._remaining -= 1
        return self._meta, 1


class _SequenceSampler(_EmptySampler):
    def __init__(self, metas: list[KVBatchMeta]) -> None:
        self._metas = list(metas)

    async def select(self, **kwargs):
        del kwargs
        if not self._metas:
            return None, 0
        return self._metas.pop(0), 1


class _EmptyBuffer:
    def __len__(self) -> int:
        return 0


class _NoOpTrainer:
    def prepare_for_lp_inference(self, keep_train_buffers: bool = False) -> None:
        del keep_train_buffers

    def finish_inference(self) -> None:
        pass

    def prepare_for_training(self) -> None:
        pass

    def begin_train_step(self, loss_fn) -> None:
        del loss_fn

    def train_microbatches_from_meta(
        self, meta: KVBatchMeta, *, train_fields: tuple[str, ...]
    ) -> None:
        del meta, train_fields

    def finish_train_step(self) -> dict:
        return {}

    def offload_to_cpu(self) -> None:
        pass


class _LpRecordingTrainer(_NoOpTrainer):
    """Records the ``keep_train_buffers`` flag the pump passes on each chunk."""

    def __init__(self) -> None:
        self.keep_train_buffers_calls: list[bool] = []

    def prepare_for_lp_inference(self, keep_train_buffers: bool = False) -> None:
        self.keep_train_buffers_calls.append(keep_train_buffers)

    def get_logprobs_from_meta(self, meta: KVBatchMeta) -> None:
        del meta


class _LogprobRecordingTrainer(_NoOpTrainer):
    def __init__(self) -> None:
        self.policy_logprob_calls = 0
        self.reference_logprob_calls = 0
        self.train_fields_calls: list[tuple[str, ...]] = []

    def get_logprobs_from_meta(self, meta: KVBatchMeta) -> None:
        del meta
        self.policy_logprob_calls += 1

    def get_reference_policy_logprobs_from_meta(self, meta: KVBatchMeta) -> None:
        del meta
        self.reference_logprob_calls += 1

    def train_microbatches_from_meta(
        self, meta: KVBatchMeta, *, train_fields: tuple[str, ...]
    ) -> None:
        del meta
        self.train_fields_calls.append(train_fields)


class _OrderRecordingTrainer(_NoOpTrainer):
    """Records the policy lifecycle into a log shared with the critic double."""

    def __init__(self, calls: list[str]) -> None:
        self.calls = calls

    def prepare_for_lp_inference(self, keep_train_buffers: bool = False) -> None:
        del keep_train_buffers
        self.calls.append("policy.prepare_for_lp_inference")

    def get_logprobs_from_meta(self, meta: KVBatchMeta) -> None:
        del meta
        self.calls.append("policy.get_logprobs_from_meta")

    def finish_inference(self) -> None:
        self.calls.append("policy.finish_inference")

    def prepare_for_training(self) -> None:
        self.calls.append("policy.prepare_for_training")

    def offload_to_cpu(self) -> None:
        self.calls.append("policy.offload_to_cpu")


class _EpochRecordingTrainer(_OrderRecordingTrainer):
    """Also records the optimizer-step lifecycle, which the epoch loop repeats."""

    def begin_train_step(self, loss_fn) -> None:
        del loss_fn
        self.calls.append("policy.begin_train_step")

    def train_microbatches_from_meta(
        self, meta: KVBatchMeta, *, train_fields: tuple[str, ...]
    ) -> None:
        del meta, train_fields
        self.calls.append("policy.train_microbatches_from_meta")

    def finish_train_step(self) -> dict:
        self.calls.append("policy.finish_train_step")
        return {}


class _StepMetricRecordingTrainer(_NoOpTrainer):
    def __init__(self, events: list[str]) -> None:
        self._events = events

    def begin_train_step(self, loss_fn) -> None:
        del loss_fn
        self._events.append("begin_train_step")

    def train_microbatches_from_meta(
        self, meta: KVBatchMeta, *, train_fields: tuple[str, ...]
    ) -> None:
        del meta, train_fields
        self._events.append("train_microbatches")

    def finish_train_step(self) -> dict:
        self._events.append("finish_train_step")
        return {}


class _StepMetricRecordingGeneration:
    requires_kv_scale_sync = False

    def __init__(self, events: list[str]) -> None:
        self._events = events

    def snapshot_step_metrics(self) -> None:
        self._events.append("snapshot_step_metrics")

    def get_step_metrics(self) -> dict[str, float]:
        self._events.append("get_step_metrics")
        return {"vllm/spec_acceptance_rate": 0.8}


class _NoOpDataPlane:
    def clear_samples(self, **kwargs) -> None:
        del kwargs


def _train_pump_controller(*, sampler) -> object:
    controller_cls = SingleControllerActor.__ray_metadata__.modified_class
    ctrl = object.__new__(controller_cls)
    ctrl._master_config = SimpleNamespace(
        grpo=GRPOConfig.model_construct(
            num_prompts_per_step=2,
            max_num_steps=1,
        ),
        # The pump's step epilogue reads the save triggers even when saving
        # is disabled.
        checkpointing={"enabled": False, "save_period": 10},
    )
    ctrl._algo_cfg = ctrl._master_config.grpo
    ctrl._message_level_advantage_penalties_enabled = False
    ctrl._async_cfg = SimpleNamespace(
        min_groups_for_streaming_train=1,
        rollout_failure=SimpleNamespace(min_step_batch_fraction=0.9),
        sampler=SimpleNamespace(
            max_lookahead_versions=1,
            warmup_lookahead_versions=None,
        ),
    )
    ctrl._consumed_samples = 0
    ctrl._total_valid_tokens = 0
    ctrl._timeout = TimeoutChecker(timeout=None, fit_last_save_time=True)
    ctrl._timeout.start_iterations()
    ctrl._advantage_cfg = AdvantageConfig()
    ctrl._policy_logprobs_required = False
    ctrl._reference_logprobs_required = False
    ctrl._teacher_logprobs_required = False
    ctrl._train_fields = single_controller._train_fields_for_step(
        policy_logprobs_required=False,
        reference_logprobs_required=False,
    )
    ctrl._advantage_estimator = None
    ctrl._partition_id = "rollout_data"
    ctrl._sampler = sampler
    ctrl._buffer = _EmptyBuffer()
    ctrl._buffer_capacity = asyncio.Semaphore(2)
    ctrl._rollout_exhausted = asyncio.Event()
    ctrl._rollout_exhausted.set()
    ctrl._trainer = _NoOpTrainer()
    ctrl._is_ppo = False
    ctrl._ppo_epochs = 1
    ctrl._critic_ppo_epochs = 1
    ctrl._value = None
    ctrl._value_loss_fn = None
    ctrl._gen = SimpleNamespace(
        requires_kv_scale_sync=False,
        snapshot_step_metrics=lambda: None,
        get_step_metrics=lambda: {},
    )
    ctrl._rollout_manager = SimpleNamespace(set_weight_version=MagicMock())
    ctrl._loss_fn = None
    ctrl._dp_client = _NoOpDataPlane()
    ctrl._timer = Timer()
    ctrl._trainer_version = 0
    ctrl._train_steps = 0
    ctrl._data_plane_checkpoint_barrier = DataPlaneCheckpointBarrier()
    ctrl._batch_shortfall = {}
    ctrl._batch_replacements = {}
    ctrl._batch_promotions = {}
    ctrl._step_log_dict = {
        "rewards": [],
        "masked_advantages": [],
        "sequence_lengths": [],
        "num_mask_sample_filtered": [],
        "seq_logprob_error_metrics": [],
    }
    ctrl._opd_stat_sum = 0.0
    ctrl._opd_stat_sumsq = 0.0
    ctrl._opd_stat_count = 0
    ctrl._teacher_coordinator = None
    return ctrl


def test_train_pump_stops_after_rollout_exhaustion_and_buffer_drain() -> None:
    ctrl = _train_pump_controller(sampler=_EmptySampler())

    asyncio.run(asyncio.wait_for(ctrl._train_pump(), timeout=1.0))

    assert ctrl._train_steps == 0


def test_train_pump_fails_if_rollout_exhausts_during_partial_step() -> None:
    meta = KVBatchMeta(
        partition_id="rollout_data",
        task_name="train",
        sample_ids=["sample-0"],
        fields=[],
        sequence_lengths=[1],
        tags=[{"weight_version": 0}],
    )
    ctrl = _train_pump_controller(sampler=_OneThenEmptySampler(meta))

    with pytest.raises(
        RuntimeError,
        match=(
            r"rollout exhausted before a complete training step was assembled: "
            r"dispatched 1/2 prompt groups"
        ),
    ):
        asyncio.run(asyncio.wait_for(ctrl._train_pump(), timeout=1.0))


class _DroppingSampler(_OneThenEmptySampler):
    """Yields one group, then reports the second as never coming.

    Stands in for a prompt that was stamped for this step and then given up on: the
    credit lands while the pump is already waiting for the group, which is the only
    way it happens in a real run and the case that waits forever without the fix.

    ``select`` validates its bounds through the real sampler's own check, so a pump
    that asks for a non-positive batch fails here exactly as it would in production
    rather than being quietly tolerated by a permissive fake.
    """

    def __init__(self, meta: KVBatchMeta, *, credit_in_evict: bool) -> None:
        super().__init__(meta)
        self._credit_in_evict = credit_in_evict
        self.ctrl = None

    def _credit(self) -> None:
        self.ctrl._batch_shortfall[self.ctrl._trainer_version] = 1

    async def evict(self, *, current_train_weight: int) -> int:
        del current_train_weight
        # The window between the pump's two reads of the step target. A credit landing
        # here shrinks the target after the loop condition has already passed.
        if self._credit_in_evict and self._meta is None:
            self._credit()
        return 0

    async def select(self, **kwargs):
        BaseSampler._validate_group_bounds(
            kwargs["min_prompt_groups"], kwargs["max_prompt_groups"]
        )
        meta, num_groups = await super().select(**kwargs)
        if meta is None and not self._credit_in_evict:
            self._credit()
        return meta, num_groups


def _dropping_controller(*, credit_in_evict: bool):
    meta = KVBatchMeta(
        partition_id="rollout_data",
        task_name="train",
        sample_ids=["sample-0"],
        fields=[],
        sequence_lengths=[1],
        tags=[{"weight_version": 0}],
    )
    sampler = _DroppingSampler(meta, credit_in_evict=credit_in_evict)
    ctrl = _train_pump_controller(sampler=sampler)
    sampler.ctrl = ctrl
    # ceil(0.9 * 2) is 2, so the harness's default floor forbids any short step at
    # all; at 0.5 the floor is 1 and closing on the one group it got is legal.
    ctrl._async_cfg.rollout_failure.min_step_batch_fraction = 0.5
    # Rollouts are still running, so the "rollout exhausted" escape is unavailable:
    # a pump that does not act on the credit waits on a group nobody is generating.
    ctrl._rollout_exhausted.clear()
    ctrl._sync_weights = AsyncMock(return_value=0)
    ctrl._logger = MagicMock()
    return ctrl


@pytest.mark.parametrize("credit_in_evict", [False, True])
def test_train_pump_closes_a_step_short_when_a_stamped_prompt_is_dropped(
    monkeypatch, credit_in_evict
) -> None:
    """Both windows a shortfall can be credited in have to close the step.

    Credited before the loop condition is re-read, the target simply falls to what is
    already dispatched. Credited after it, between the two reads, the batch the pump
    would ask for is empty -- which the sampler rejects, so the pump has to notice
    instead of asking. Either way the step trains on what it got.
    """
    ctrl = _dropping_controller(credit_in_evict=credit_in_evict)
    ctrl._batch_replacements = {0: 1}
    ctrl._batch_promotions = {0: 2, 3: 1}
    monkeypatch.setattr(single_controller.ray, "cluster_resources", lambda: {})

    asyncio.run(asyncio.wait_for(ctrl._train_pump(), timeout=1.0))

    assert ctrl._train_steps == 1
    # One group, not the configured two: the step is what shrank, not the run.
    assert ctrl._consumed_samples == 1
    train_metrics = ctrl._logger.log_metrics.call_args_list[0].args[0]
    assert train_metrics["dropped_prompt_groups"] == 1
    assert train_metrics["replaced_prompt_groups"] == 1
    assert train_metrics["promoted_prompt_groups"] == 2
    # Read against version_during_step, not the already-incremented _trainer_version,
    # which would report 0 for every step forever.
    assert ctrl._trainer_version == 1
    # This step's counts are pruned; a later step's survive to be reported by it.
    assert ctrl._batch_shortfall == {}
    assert ctrl._batch_replacements == {}
    assert ctrl._batch_promotions == {3: 1}


def test_train_pump_prunes_stamps_older_than_the_step_that_just_closed(
    monkeypatch,
) -> None:
    """A straggler credited for a step that already closed must not outlive it.

    Popping only the current step's entry would leak those, and they are unreachable
    afterwards: the target is only ever read for the step being assembled.
    """
    meta = KVBatchMeta(
        partition_id="rollout_data",
        task_name="train",
        sample_ids=["sample-0"],
        fields=[],
        sequence_lengths=[1],
        tags=[{"weight_version": 3}],
    )
    ctrl = _train_pump_controller(sampler=_OneThenEmptySampler(meta))
    ctrl._async_cfg.rollout_failure.min_step_batch_fraction = 0.5
    ctrl._trainer_version = 3
    ctrl._sync_weights = AsyncMock(return_value=0)
    ctrl._logger = MagicMock()
    ctrl._batch_shortfall = {2: 1, 3: 1, 5: 1}
    monkeypatch.setattr(single_controller.ray, "cluster_resources", lambda: {})

    asyncio.run(asyncio.wait_for(ctrl._train_pump(), timeout=1.0))

    assert ctrl._batch_shortfall == {5: 1}


@pytest.mark.parametrize(
    (
        "policy_logprobs_required",
        "reference_logprobs_required",
        "expected_policy_calls",
        "expected_reference_calls",
    ),
    [
        (False, False, 0, 0),
        (True, False, 1, 0),
        (False, True, 0, 1),
        (True, True, 1, 1),
    ],
)
def test_train_pump_requests_and_fetches_only_required_logprobs(
    monkeypatch,
    policy_logprobs_required: bool,
    reference_logprobs_required: bool,
    expected_policy_calls: int,
    expected_reference_calls: int,
) -> None:
    meta = KVBatchMeta(
        partition_id="rollout_data",
        task_name="train",
        sample_ids=["sample-0", "sample-1"],
        fields=[],
        sequence_lengths=[1, 1],
        tags=[{"weight_version": 0}, {"weight_version": 0}],
    )
    ctrl = _train_pump_controller(sampler=_FullStepSampler(meta))
    ctrl._policy_logprobs_required = policy_logprobs_required
    ctrl._reference_logprobs_required = reference_logprobs_required
    ctrl._train_fields = single_controller._train_fields_for_step(
        policy_logprobs_required=policy_logprobs_required,
        reference_logprobs_required=reference_logprobs_required,
    )
    trainer = _LogprobRecordingTrainer()
    ctrl._trainer = trainer
    ctrl._sync_weights = AsyncMock(return_value=1)
    ctrl._logger = MagicMock()
    monkeypatch.setattr(single_controller.ray, "cluster_resources", lambda: {})

    asyncio.run(asyncio.wait_for(ctrl._train_pump(), timeout=1.0))

    assert trainer.policy_logprob_calls == expected_policy_calls
    assert trainer.reference_logprob_calls == expected_reference_calls
    assert trainer.train_fields_calls == [ctrl._train_fields]
    assert ("prev_logprobs" in ctrl._train_fields) is policy_logprobs_required
    assert (
        "reference_policy_logprobs" in ctrl._train_fields
    ) is reference_logprobs_required


def test_train_pump_rejects_step_with_no_valid_training_chunks() -> None:
    meta = KVBatchMeta(
        partition_id="rollout_data",
        task_name="train",
        sample_ids=["sample-0"],
        fields=[],
        sequence_lengths=[1],
        tags=[{"weight_version": 0}],
    )
    ctrl = _train_pump_controller(sampler=_OneThenEmptySampler(meta))
    ctrl._master_config.grpo.num_prompts_per_step = 1
    ctrl._advantage_stage = AsyncMock(return_value=(meta, False))
    trainer = MagicMock(spec=_NoOpTrainer)
    ctrl._trainer = trainer

    with pytest.raises(
        RuntimeError,
        match="no valid response tokens after filtering",
    ):
        asyncio.run(asyncio.wait_for(ctrl._train_pump(), timeout=1.0))

    trainer.prepare_for_training.assert_called_once_with()
    trainer.begin_train_step.assert_not_called()
    trainer.train_microbatches_from_meta.assert_not_called()
    trainer.finish_train_step.assert_not_called()


def test_train_pump_skips_empty_chunk_and_trains_later_valid_chunk(
    monkeypatch,
) -> None:
    empty_meta = KVBatchMeta(
        partition_id="rollout_data",
        task_name="train",
        sample_ids=["empty-sample"],
        fields=[],
        sequence_lengths=[1],
        tags=[{"weight_version": 0}],
    )
    valid_meta = KVBatchMeta(
        partition_id="rollout_data",
        task_name="train",
        sample_ids=["valid-sample"],
        fields=[],
        sequence_lengths=[1],
        tags=[{"weight_version": 0}],
    )
    ctrl = _train_pump_controller(sampler=_SequenceSampler([empty_meta, valid_meta]))
    ctrl._advantage_stage = AsyncMock(
        side_effect=[
            (empty_meta, False),
            (valid_meta, True),
        ]
    )
    trainer = MagicMock(spec=_NoOpTrainer)
    trainer.finish_train_step.return_value = {}
    ctrl._trainer = trainer
    ctrl._sync_weights = AsyncMock(return_value=0)
    ctrl._logger = MagicMock()
    monkeypatch.setattr(single_controller.ray, "cluster_resources", lambda: {})

    asyncio.run(asyncio.wait_for(ctrl._train_pump(), timeout=1.0))

    assert trainer.prepare_for_training.call_count == 2
    trainer.begin_train_step.assert_called_once_with(None)
    trainer.train_microbatches_from_meta.assert_called_once_with(
        valid_meta, train_fields=ctrl._train_fields
    )
    trainer.finish_train_step.assert_called_once_with()
    assert ctrl._train_steps == 1


def test_train_pump_logs_nonzero_stale_group_metrics(monkeypatch) -> None:
    meta = KVBatchMeta(
        partition_id="rollout_data",
        task_name="train",
        sample_ids=["sample-0", "sample-1"],
        fields=[],
        sequence_lengths=[1, 1],
        tags=[{"weight_version": 0}, {"weight_version": 0}],
    )
    ctrl = _train_pump_controller(sampler=_EvictingSampler(meta))
    ctrl._sync_weights = AsyncMock(return_value=1)
    ctrl._logger = MagicMock()
    monkeypatch.setattr(single_controller.ray, "cluster_resources", lambda: {})

    asyncio.run(asyncio.wait_for(ctrl._train_pump(), timeout=1.0))

    ctrl._sync_weights.assert_awaited_once_with(calibration_data=None)
    train_metrics = ctrl._logger.log_metrics.call_args_list[0].args[0]
    assert train_metrics["evicted_stale_prompt_groups"] == 2
    assert train_metrics["aborted_stale_inflight_groups"] == 1


def test_train_pump_aggregates_selected_rollout_metrics_across_chunks(
    monkeypatch,
    capsys,
) -> None:
    metas = [
        KVBatchMeta(
            partition_id="rollout_data",
            task_name="train",
            sample_ids=[f"sample-{index}"],
            fields=[],
            sequence_lengths=[1],
            extra_info={ROLLOUT_METRICS: [metrics]},
            tags=[{"weight_version": 0}],
        )
        for index, metrics in enumerate(
            [
                {
                    "gen_tokens/min": 7,
                    "gen_tokens/max": 10,
                    "total_turns": 2,
                    "accuracy": 0.25,
                    "trajectory_duration_s": 1.0,
                    "histogram/gen_tokens_length": [7, 10],
                },
                {
                    "gen_tokens/min": 3,
                    "gen_tokens/max": 20,
                    "total_turns": 5,
                    "accuracy": 0.75,
                    "trajectory_duration_s": 3.0,
                    "histogram/gen_tokens_length": [3, 20],
                },
            ]
        )
    ]
    ctrl = _train_pump_controller(sampler=_SequenceSampler(metas))
    ctrl._sync_weights = AsyncMock(return_value=0)
    ctrl._logger = MagicMock()
    monkeypatch.setattr(single_controller.ray, "cluster_resources", lambda: {})

    asyncio.run(asyncio.wait_for(ctrl._train_pump(), timeout=1.0))

    train_call = ctrl._logger.log_metrics.call_args_list[0]
    train_metrics = train_call.args[0]
    assert train_metrics["gen_tokens/min"] == 3
    assert train_metrics["gen_tokens/max"] == 20
    assert train_metrics["total_turns"] == 7
    assert train_metrics["accuracy"] == pytest.approx(0.5)
    assert train_metrics["trajectory_duration_s"] == pytest.approx(2.0)
    assert train_metrics["trajectory_duration_s/max"] == 3.0
    assert train_metrics["trajectory_duration_s/p95"] == 3.0
    assert train_metrics["histogram/gen_tokens_length"] == [7, 10, 3, 20]
    assert train_call.kwargs == {"step": 1, "prefix": "train"}
    assert all(ROLLOUT_METRICS not in meta.extra_info for meta in metas)
    assert "histogram/gen_tokens_length" not in capsys.readouterr().out


def test_train_pump_collects_generation_metrics_at_step_boundaries(
    monkeypatch,
) -> None:
    meta = KVBatchMeta(
        partition_id="rollout_data",
        task_name="train",
        sample_ids=["sample-0"],
        fields=[],
        sequence_lengths=[1],
        tags=[{"weight_version": 0}],
    )
    events: list[str] = []
    ctrl = _train_pump_controller(sampler=_ChunkedSampler(meta, chunks=2))
    ctrl._trainer = _StepMetricRecordingTrainer(events)
    ctrl._gen = _StepMetricRecordingGeneration(events)
    ctrl._sync_weights = AsyncMock(return_value=0)
    ctrl._logger = MagicMock()
    monkeypatch.setattr(single_controller.ray, "cluster_resources", lambda: {})

    asyncio.run(asyncio.wait_for(ctrl._train_pump(), timeout=1.0))

    assert events == [
        "snapshot_step_metrics",
        "begin_train_step",
        "train_microbatches",
        "train_microbatches",
        "finish_train_step",
        "get_step_metrics",
    ]
    train_metrics = ctrl._logger.log_metrics.call_args_list[0].args[0]
    assert train_metrics["vllm/spec_acceptance_rate"] == pytest.approx(0.8)


def test_train_pump_skips_generation_metrics_without_generation_handle(
    monkeypatch,
) -> None:
    meta = KVBatchMeta(
        partition_id="rollout_data",
        task_name="train",
        sample_ids=["sample-0"],
        fields=[],
        sequence_lengths=[1],
        tags=[{"weight_version": 0}],
    )
    ctrl = _train_pump_controller(sampler=_ChunkedSampler(meta, chunks=2))
    ctrl._gen = None
    ctrl._sync_weights = AsyncMock(return_value=0)
    ctrl._logger = MagicMock()
    monkeypatch.setattr(single_controller.ray, "cluster_resources", lambda: {})

    asyncio.run(asyncio.wait_for(ctrl._train_pump(), timeout=1.0))

    train_metrics = ctrl._logger.log_metrics.call_args_list[0].args[0]
    assert "vllm/spec_acceptance_rate" not in train_metrics


def test_train_pump_keeps_train_buffers_once_the_step_is_open(monkeypatch) -> None:
    """The logprob detour between chunks must not offload the trainer's grad
    buffers, because mcore's offload frees the gradients the earlier chunks of
    this step accumulated rather than copying them out.

    First chunk: no step open yet, nothing to preserve, so the offload is still
    worth taking. Every later chunk: step open, buffers must stay resident.
    """
    meta = KVBatchMeta(
        partition_id="rollout_data",
        task_name="train",
        sample_ids=["sample-0"],
        fields=[],
        sequence_lengths=[1],
        tags=[{"weight_version": 0}],
    )
    # num_prompts_per_step is 2 in the harness, so two single-group chunks close
    # the step.
    ctrl = _train_pump_controller(sampler=_ChunkedSampler(meta, chunks=2))
    ctrl._policy_logprobs_required = True
    trainer = _LpRecordingTrainer()
    ctrl._trainer = trainer
    ctrl._sync_weights = AsyncMock(return_value=1)
    ctrl._logger = MagicMock()
    monkeypatch.setattr(single_controller.ray, "cluster_resources", lambda: {})

    asyncio.run(asyncio.wait_for(ctrl._train_pump(), timeout=1.0))

    assert ctrl._train_steps == 1
    assert trainer.keep_train_buffers_calls == [False, True]


def test_train_pump_does_not_offload_the_policy_on_a_grpo_run(monkeypatch) -> None:
    """The pre-critic offload is PPO-only: GRPO has no critic to make room for."""
    meta = KVBatchMeta(
        partition_id="rollout_data",
        task_name="train",
        sample_ids=["sample-0"],
        fields=[],
        sequence_lengths=[1],
        tags=[{"weight_version": 0}],
    )
    calls: list[str] = []
    ctrl = _train_pump_controller(sampler=_ChunkedSampler(meta, chunks=2))
    ctrl._policy_logprobs_required = False
    ctrl._reference_logprobs_required = False
    ctrl._trainer = _OrderRecordingTrainer(calls)
    ctrl._sync_weights = AsyncMock(return_value=1)
    ctrl._logger = MagicMock()
    monkeypatch.setattr(single_controller.ray, "cluster_resources", lambda: {})

    asyncio.run(asyncio.wait_for(ctrl._train_pump(), timeout=1.0))

    assert ctrl._train_steps == 1
    assert "policy.offload_to_cpu" not in calls


# ── PPO ────────────────────────────────────────────────────────────────────


class _NoOpValue:
    """Records the critic lifecycle the pump drives around each stage.

    Pass a shared list and a prefix to interleave this log with the policy's.
    """

    def __init__(self, calls: list[str] | None = None, prefix: str = "") -> None:
        self.calls: list[str] = [] if calls is None else calls
        self._prefix = prefix

    def _record(self, name: str) -> None:
        self.calls.append(f"{self._prefix}{name}")

    def prepare_for_inference(self) -> None:
        self._record("prepare_for_inference")

    def get_values_from_meta(self, meta: KVBatchMeta) -> None:
        del meta
        self._record("get_values_from_meta")

    def finish_inference(self) -> None:
        self._record("finish_inference")

    def prepare_for_training(self) -> None:
        self._record("prepare_for_training")

    def train_from_meta(self, meta: KVBatchMeta, loss_fn) -> dict:
        del meta, loss_fn
        self._record("train_from_meta")
        return {
            "loss": torch.tensor([0.25]),
            "grad_norm": torch.tensor([1.5]),
            "all_mb_metrics": {"vf_clipfrac": [0.0], "values_min": [-1.0]},
        }

    def finish_training(self) -> None:
        self._record("finish_training")


def _ppo_train_pump_controller(
    *,
    sampler,
    policy_training_start_step: int = 0,
    value: _NoOpValue | None = None,
    ppo_epochs: int = 1,
    critic_ppo_epochs: int | None = None,
) -> tuple[object, _NoOpValue]:
    ctrl = _train_pump_controller(sampler=sampler)
    value = _NoOpValue() if value is None else value
    ctrl._is_ppo = True
    ctrl._ppo_epochs = ppo_epochs
    ctrl._critic_ppo_epochs = (
        ppo_epochs if critic_ppo_epochs is None else critic_ppo_epochs
    )
    ctrl._value = value
    ctrl._value_loss_fn = MagicMock(name="value_loss_fn")
    ctrl._master_config.grpo = None
    ctrl._master_config.ppo = PPOConfig.model_construct(
        num_prompts_per_step=1,
        max_num_steps=1,
        policy_training_start_step=policy_training_start_step,
        seq_logprob_error_threshold=None,
    )
    ctrl._algo_cfg = ctrl._master_config.ppo
    ctrl._message_level_advantage_penalties_enabled = False
    ctrl._sync_weights = AsyncMock(return_value=0)
    ctrl._logger = MagicMock()
    return ctrl, value


def _single_group_meta() -> KVBatchMeta:
    return KVBatchMeta(
        partition_id="rollout_data",
        task_name="train",
        sample_ids=["sample-0"],
        fields=[],
        sequence_lengths=[1],
        tags=[{"weight_version": 0}],
    )


def test_train_pump_parks_the_policy_on_cpu_across_the_critic_stages(
    monkeypatch,
) -> None:
    """The critic shares the training GPUs, so the two models never overlap.

    The critic forward runs after the log-prob pass rather than before it, as
    ppo.py does, so the policy reaches finish_inference with its grad buffers
    already freed.
    """
    meta = _single_group_meta()
    calls: list[str] = []
    ctrl, _ = _ppo_train_pump_controller(
        sampler=_OneThenEmptySampler(meta),
        value=_NoOpValue(calls=calls, prefix="critic."),
    )
    ctrl._policy_logprobs_required = True
    ctrl._trainer = _OrderRecordingTrainer(calls)
    ctrl._advantage_stage = AsyncMock(return_value=(meta, True))
    monkeypatch.setattr(single_controller.ray, "cluster_resources", lambda: {})

    asyncio.run(asyncio.wait_for(ctrl._train_pump(), timeout=1.0))

    assert calls == [
        "policy.prepare_for_lp_inference",
        "policy.get_logprobs_from_meta",
        "policy.finish_inference",
        "critic.prepare_for_inference",
        "critic.get_values_from_meta",
        "critic.finish_inference",
        "critic.prepare_for_training",
        "critic.train_from_meta",
        "critic.finish_training",
        "policy.prepare_for_training",
    ]


def test_train_pump_parks_the_policy_when_neither_logprob_is_needed(
    monkeypatch,
) -> None:
    """No logprob means no prepare_for_lp_inference, so nothing else would park
    the policy optimizer before the critic runs."""
    meta = _single_group_meta()
    calls: list[str] = []
    ctrl, _ = _ppo_train_pump_controller(
        sampler=_OneThenEmptySampler(meta),
        value=_NoOpValue(calls=calls, prefix="critic."),
    )
    ctrl._policy_logprobs_required = False
    ctrl._reference_logprobs_required = False
    ctrl._trainer = _OrderRecordingTrainer(calls)
    ctrl._advantage_stage = AsyncMock(return_value=(meta, True))
    monkeypatch.setattr(single_controller.ray, "cluster_resources", lambda: {})

    asyncio.run(asyncio.wait_for(ctrl._train_pump(), timeout=1.0))

    assert "policy.prepare_for_lp_inference" not in calls
    assert calls.index("policy.offload_to_cpu") < calls.index(
        "critic.prepare_for_inference"
    )


def test_train_pump_does_not_double_offload_when_logprobs_run(monkeypatch) -> None:
    """The logprob path already parks the optimizer, so the elif must not fire."""
    meta = _single_group_meta()
    calls: list[str] = []
    ctrl, _ = _ppo_train_pump_controller(
        sampler=_OneThenEmptySampler(meta),
        value=_NoOpValue(calls=calls, prefix="critic."),
    )
    ctrl._policy_logprobs_required = True
    ctrl._trainer = _OrderRecordingTrainer(calls)
    ctrl._advantage_stage = AsyncMock(return_value=(meta, True))
    monkeypatch.setattr(single_controller.ray, "cluster_resources", lambda: {})

    asyncio.run(asyncio.wait_for(ctrl._train_pump(), timeout=1.0))

    assert "policy.offload_to_cpu" not in calls


def test_train_pump_logs_critic_metrics(monkeypatch) -> None:
    meta = _single_group_meta()
    ctrl, _ = _ppo_train_pump_controller(sampler=_OneThenEmptySampler(meta))
    ctrl._advantage_stage = AsyncMock(return_value=(meta, True))
    monkeypatch.setattr(single_controller.ray, "cluster_resources", lambda: {})

    asyncio.run(asyncio.wait_for(ctrl._train_pump(), timeout=1.0))

    train_metrics = ctrl._logger.log_metrics.call_args_list[0].args[0]
    assert train_metrics["critic/loss"].item() == pytest.approx(0.25)
    assert train_metrics["critic/grad_norm"].item() == pytest.approx(1.5)
    assert "critic/explained_var" in train_metrics


def test_train_pump_skips_the_critic_on_an_empty_chunk(monkeypatch) -> None:
    """No training tokens means no GAE returns to regress against."""
    meta = _single_group_meta()
    ctrl, value = _ppo_train_pump_controller(sampler=_OneThenEmptySampler(meta))
    ctrl._advantage_stage = AsyncMock(return_value=(meta, False))
    monkeypatch.setattr(single_controller.ray, "cluster_resources", lambda: {})

    with pytest.raises(RuntimeError, match="no valid response tokens after filtering"):
        asyncio.run(asyncio.wait_for(ctrl._train_pump(), timeout=1.0))

    assert "train_from_meta" not in value.calls
    # The forward still ran -- it is what the advantage stage consumes.
    assert "get_values_from_meta" in value.calls


@pytest.mark.parametrize("ppo_epochs", [1, 2])
def test_train_pump_freezes_the_policy_during_critic_warmup(
    monkeypatch, capsys, ppo_epochs
) -> None:
    """Below policy_training_start_step the critic trains alone: no optimizer
    step, and no weight transfer to generation either. The frozen policy does
    not shorten the critic's own epoch loop."""
    critic_ppo_epochs = 3
    meta = _single_group_meta()
    ctrl, value = _ppo_train_pump_controller(
        sampler=_OneThenEmptySampler(meta),
        policy_training_start_step=1,
        ppo_epochs=ppo_epochs,
        critic_ppo_epochs=critic_ppo_epochs,
    )
    trainer = MagicMock(spec=_NoOpTrainer)
    ctrl._trainer = trainer
    ctrl._advantage_stage = AsyncMock(return_value=(meta, True))
    monkeypatch.setattr(single_controller.ray, "cluster_resources", lambda: {})

    asyncio.run(asyncio.wait_for(ctrl._train_pump(), timeout=1.0))

    assert value.calls.count("train_from_meta") == critic_ppo_epochs
    trainer.prepare_for_training.assert_not_called()
    trainer.begin_train_step.assert_not_called()
    trainer.finish_train_step.assert_not_called()
    ctrl._sync_weights.assert_not_awaited()
    # The step still closed and published the new version, so staleness
    # accounting keeps working through the warmup.
    assert ctrl._train_steps == 1
    assert ctrl._trainer_version == 1
    ctrl._rollout_manager.set_weight_version.assert_called_once_with(1)
    assert "Critic warmup complete" not in capsys.readouterr().out


def test_train_pump_trains_the_policy_once_warmup_is_over(monkeypatch, capsys) -> None:
    meta = _single_group_meta()
    ctrl, _ = _ppo_train_pump_controller(
        sampler=_OneThenEmptySampler(meta),
        policy_training_start_step=1,
    )
    ctrl._train_steps = 1
    ctrl._trainer_version = 1
    ctrl._algo_cfg.max_num_steps = 2
    trainer = MagicMock(spec=_NoOpTrainer)
    trainer.finish_train_step.return_value = {}
    ctrl._trainer = trainer
    ctrl._advantage_stage = AsyncMock(return_value=(meta, True))
    monkeypatch.setattr(single_controller.ray, "cluster_resources", lambda: {})

    asyncio.run(asyncio.wait_for(ctrl._train_pump(), timeout=1.0))

    trainer.begin_train_step.assert_called_once()
    trainer.finish_train_step.assert_called_once_with()
    ctrl._sync_weights.assert_awaited_once_with(calibration_data=None)
    # Announced exactly once, on the step that crosses the boundary.
    assert capsys.readouterr().out.count("Critic warmup complete") == 1


def test_train_pump_groups_ppo_epochs_by_model(monkeypatch) -> None:
    """Each model stays resident for all of its PPO epochs.

    The critic still finishes and leaves the shared training GPUs before the
    policy is loaded, but the models no longer move between epochs."""
    meta = _single_group_meta()
    calls: list[str] = []
    ctrl, _ = _ppo_train_pump_controller(
        sampler=_OneThenEmptySampler(meta),
        value=_NoOpValue(calls=calls, prefix="critic."),
        ppo_epochs=2,
    )
    ctrl._trainer = _EpochRecordingTrainer(calls)
    ctrl._advantage_stage = AsyncMock(return_value=(meta, True))
    monkeypatch.setattr(single_controller.ray, "cluster_resources", lambda: {})

    asyncio.run(asyncio.wait_for(ctrl._train_pump(), timeout=1.0))

    assert calls == [
        # Neither logprob is required here, so the policy is parked up front.
        "policy.offload_to_cpu",
        "policy.finish_inference",
        "critic.prepare_for_inference",
        "critic.get_values_from_meta",
        "critic.finish_inference",
        "critic.prepare_for_training",
        "critic.train_from_meta",
        "critic.train_from_meta",
        "critic.finish_training",
        "policy.prepare_for_training",
        "policy.begin_train_step",
        "policy.train_microbatches_from_meta",
        "policy.finish_train_step",
        "policy.begin_train_step",
        "policy.train_microbatches_from_meta",
        "policy.finish_train_step",
    ]
    # Still one RL step, so one refit and one version bump.
    ctrl._sync_weights.assert_awaited_once_with(calibration_data=None)
    assert ctrl._trainer_version == 1


def test_train_pump_runs_all_critic_epochs_before_actor_epochs(monkeypatch) -> None:
    """Independent critic epochs share one residency and do not update policy."""
    meta = _single_group_meta()
    calls: list[str] = []
    ctrl, _ = _ppo_train_pump_controller(
        sampler=_OneThenEmptySampler(meta),
        value=_NoOpValue(calls=calls, prefix="critic."),
        ppo_epochs=1,
        critic_ppo_epochs=3,
    )
    ctrl._trainer = _EpochRecordingTrainer(calls)
    ctrl._advantage_stage = AsyncMock(return_value=(meta, True))
    monkeypatch.setattr(single_controller.ray, "cluster_resources", lambda: {})

    asyncio.run(asyncio.wait_for(ctrl._train_pump(), timeout=1.0))

    assert calls == [
        # Neither logprob is required here, so the policy is parked up front.
        "policy.offload_to_cpu",
        "policy.finish_inference",
        "critic.prepare_for_inference",
        "critic.get_values_from_meta",
        "critic.finish_inference",
        "critic.prepare_for_training",
        "critic.train_from_meta",
        "critic.train_from_meta",
        "critic.train_from_meta",
        "critic.finish_training",
        "policy.prepare_for_training",
        "policy.begin_train_step",
        "policy.train_microbatches_from_meta",
        "policy.finish_train_step",
    ]
    ctrl._sync_weights.assert_awaited_once_with(calibration_data=None)


def test_advantage_stage_writes_gae_returns_alongside_advantages() -> None:
    """The critic's regression target has to reach TQ, or the value train step
    fetches a column nobody wrote."""
    batch_size, sequence_length = 2, 4
    data = TensorDict(
        {
            "prompt_ids_for_adv": torch.zeros(
                batch_size, sequence_length, dtype=torch.long
            ),
            "total_reward": torch.tensor([1.0, 0.0]),
            "token_mask": torch.ones(batch_size, sequence_length),
            "sample_mask": torch.ones(batch_size),
            "values": torch.zeros(batch_size, sequence_length),
            "mask_sample": torch.zeros(batch_size, dtype=torch.bool),
            "truncated": torch.zeros(batch_size, dtype=torch.bool),
        },
        batch_size=[batch_size],
    )
    data_plane = _AdvantageDataPlane(data)

    class _GaeLikeEstimator:
        def __init__(self) -> None:
            self.kwargs: dict | None = None

        def compute_advantage(self, *, rewards, mask, **kwargs):
            self.kwargs = kwargs
            adv = rewards.unsqueeze(-1).expand_as(mask).clone()
            return adv, adv + 1.0

    estimator = _GaeLikeEstimator()
    controller_cls = SingleControllerActor.__ray_metadata__.modified_class
    ctrl = object.__new__(controller_cls)
    ctrl._dp_client = data_plane
    ctrl._advantage_cfg = AdvantageConfig()
    ctrl._advantage_estimator = estimator
    ctrl._policy_logprobs_required = False
    ctrl._reference_logprobs_required = False
    ctrl._teacher_logprobs_required = False
    ctrl._is_ppo = True
    ctrl._master_config = SimpleNamespace(
        ppo=SimpleNamespace(seq_logprob_error_threshold=None, overlong_filtering=False)
    )
    ctrl._algo_cfg = ctrl._master_config.ppo
    ctrl._message_level_advantage_penalties_enabled = False
    ctrl._step_log_dict = {
        "rewards": [],
        "masked_advantages": [],
        "sequence_lengths": [],
        "num_mask_sample_filtered": [],
        "seq_logprob_error_metrics": [],
    }
    meta = KVBatchMeta(
        partition_id="rollout_data",
        task_name="train",
        sample_ids=[f"sample-{i}" for i in range(batch_size)],
        fields=list(data.keys()),
    )

    result_meta, has_valid_training_tokens = asyncio.run(ctrl._advantage_stage(meta))

    assert has_valid_training_tokens
    assert "values" in (data_plane.selected_fields or [])
    assert estimator.kwargs is not None
    assert torch.equal(estimator.kwargs["values"], torch.zeros(2, 4))
    assert data_plane.written_fields is not None
    assert torch.equal(
        data_plane.written_fields["returns"],
        torch.tensor([[2.0] * 4, [1.0] * 4]),
    )
    assert "returns" in (result_meta.fields or [])
    assert "advantages" in (result_meta.fields or [])
