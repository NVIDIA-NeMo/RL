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

"""Unit tests for Phase 1 SC checkpointing.

Covers (checkpointing-plan.md, Phase 1 "Testing"):
  - counter restore from save_state (train_steps / trainer_version /
    max_rollout_version invariant);
  - save trigger + write path through _train_pump with fakes (period
    boundary, last step, timeout, disabled);
  - metric_name behavior (val:* warn-and-save, train:* value recorded);
  - setup_single_controller resume-path wiring (get_resume_paths forwarded
    to the trainer factory, save_state loaded from training_info.json).
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any, Optional
from unittest.mock import MagicMock

import pytest

from nemo_rl.algorithms.grpo import _default_grpo_save_state
from nemo_rl.algorithms.loss import ClippedPGLossConfig
from nemo_rl.algorithms.single_controller import SingleControllerActor
from nemo_rl.algorithms.single_controller_utils import (
    AsyncRLConfig,
    MasterConfig,
    SingleControllerBundle,
    setup_single_controller,
)
from nemo_rl.data_plane import KVBatchMeta
from nemo_rl.utils.checkpoint import CheckpointManager

# Reuse the factory patches from the setup tests (same cross-module fixture
# import pattern as test_rollout_pump.py).
from tests.unit.single_controller.test_single_controller_setup import (
    patched_factories,  # noqa: F401
)

# Instantiate the underlying class in-process (same pattern as
# tests/unit/algorithms/test_async_utils.py for AsyncTrajectoryCollector).
_ACTOR_CLS = SingleControllerActor.__ray_metadata__.modified_class

_PARTITION_ID = "rollout_data"


# ── fakes ────────────────────────────────────────────────────────────────────


class _FakeTrainer:
    """TQPolicy stand-in: train methods are no-ops, save_checkpoint records calls."""

    def __init__(self, step_metrics: Optional[dict[str, Any]] = None) -> None:
        self._step_metrics = dict(step_metrics or {})
        self.save_calls: list[dict[str, Any]] = []

    def prepare_for_lp_inference(self) -> None:
        pass

    def get_logprobs_from_meta(self, meta: KVBatchMeta) -> None:
        pass

    def get_reference_policy_logprobs_from_meta(self, meta: KVBatchMeta) -> None:
        pass

    def prepare_for_training(self) -> None:
        pass

    def begin_train_step(self, step_id: str, loss_fn: Any = None) -> None:
        pass

    def train_microbatch_from_meta(self, step_id: str, meta: KVBatchMeta) -> None:
        pass

    def finish_train_step(self, step_id: str) -> dict[str, Any]:
        return dict(self._step_metrics)

    def save_checkpoint(
        self,
        *,
        weights_path: str,
        optimizer_path: Optional[str],
        tokenizer_path: str,
        checkpointing_cfg: dict[str, Any],
    ) -> None:
        self.save_calls.append(
            {
                "weights_path": weights_path,
                "optimizer_path": optimizer_path,
                "tokenizer_path": tokenizer_path,
                "checkpointing_cfg": checkpointing_cfg,
            }
        )
        # Mimic the real Policy: materialize the checkpoint subdirs.
        os.makedirs(weights_path, exist_ok=True)
        if optimizer_path is not None:
            os.makedirs(optimizer_path, exist_ok=True)
        os.makedirs(tokenizer_path, exist_ok=True)


class _FakeSampler:
    """StalenessSampler stand-in: always returns a full, fresh batch."""

    def __init__(self, partition_id: str = _PARTITION_ID) -> None:
        self._partition_id = partition_id
        self._step = 0

    async def evict(self, current_train_weight: int) -> int:
        return 0

    async def select(
        self,
        *,
        current_train_weight: int,
        min_prompt_groups: int,
        max_prompt_groups: int,
    ) -> tuple[KVBatchMeta, int]:
        n = max_prompt_groups
        sample_ids = [f"s{self._step}-{i}" for i in range(n)]
        self._step += 1
        meta = KVBatchMeta(
            partition_id=self._partition_id,
            task_name=None,
            sample_ids=sample_ids,
            sequence_lengths=[16] * n,
            tags=[{"weight_version": current_train_weight}] * n,
        )
        return meta, n


class _FakeDPClient:
    def __init__(self) -> None:
        self.clear_calls: list[tuple[list[str], str]] = []

    def clear_samples(self, sample_ids: list[str], partition_id: str) -> None:
        self.clear_calls.append((list(sample_ids), partition_id))


class _FakeWeightSynchronizer:
    def __init__(self) -> None:
        self.sync_count = 0

    def sync_weights(self) -> None:
        self.sync_count += 1


class _FakeRolloutManager:
    def __init__(self) -> None:
        self.weight_versions: list[int] = []

    def set_weight_version(self, version: int) -> None:
        self.weight_versions.append(version)


# ── builders ─────────────────────────────────────────────────────────────────


def _actor_master_config(
    tmp_path: Path,
    *,
    max_num_steps: int = 4,
    save_period: int = 2,
    enabled: bool = True,
    metric_name: Optional[str] = None,
    save_optimizer: bool = True,
    checkpoint_must_save_by: Optional[str] = None,
    num_prompts_per_step: int = 2,
) -> MasterConfig:
    """MasterConfig for in-process SingleControllerActor tests.

    All fields are populated (init_tmp_checkpoint dumps the whole config to
    config.yaml); values satisfy __init__'s quota/batch-size validation.
    """
    return MasterConfig.model_construct(
        policy={
            # One optimizer.step per RL step: prompts * generations == gbs.
            "train_global_batch_size": num_prompts_per_step * 2,
        },
        loss_fn=ClippedPGLossConfig(),
        env={},
        data={"shuffle": False, "num_workers": 0},
        grpo={
            "max_num_steps": max_num_steps,
            "max_num_epochs": 1,
            "num_prompts_per_step": num_prompts_per_step,
            "num_generations_per_prompt": 2,
            "seed": 42,
        },
        logger={
            "log_dir": str(tmp_path / "logs"),
            "wandb_enabled": False,
            "swanlab_enabled": False,
            "tensorboard_enabled": False,
            "mlflow_enabled": False,
            "monitor_gpus": False,
        },
        cluster={"num_nodes": 1, "gpus_per_node": 1},
        checkpointing={
            "enabled": enabled,
            "checkpoint_dir": str(tmp_path / "checkpoints"),
            "metric_name": metric_name,
            "higher_is_better": True,
            "keep_top_k": None,
            "save_period": save_period,
            "save_optimizer": save_optimizer,
            "checkpoint_must_save_by": checkpoint_must_save_by,
        },
        data_plane={"enabled": True, "impl": "transfer_queue"},
        async_rl=AsyncRLConfig(
            batch_selection_strategy="staleness_window",
            max_weight_staleness_versions=1,
            min_prompt_groups_per_batch=1,
            max_inflight_prompts=4,
            max_buffered_rollouts=4,
            over_sampling=True,
        ),
    )


def _make_bundle(
    *,
    trainer: Optional[_FakeTrainer] = None,
    save_state: Optional[dict[str, Any]] = None,
) -> SingleControllerBundle:
    return SingleControllerBundle(
        gen_handle=object(),
        trainer_handle=trainer if trainer is not None else _FakeTrainer(),
        env_handles={},
        train_cluster=None,
        inference_cluster=None,
        dp_client=_FakeDPClient(),
        dataloader=[],
        weight_synchronizer=_FakeWeightSynchronizer(),
        advantage_estimator=None,
        loss_fn=object(),
        rollout_manager=_FakeRolloutManager(),
        tq_buffer=object(),
        partition_id=_PARTITION_ID,
        save_state=(
            save_state if save_state is not None else _default_grpo_save_state()
        ),
    )


def _run_train_pump(mc: MasterConfig, bundle: SingleControllerBundle):
    """Construct the actor in-process and drive _train_pump to completion."""

    async def _main():
        actor = _ACTOR_CLS(mc, bundle)
        actor._sampler = _FakeSampler()
        await actor._train_pump()
        return actor

    return asyncio.run(_main())


def _step_dir_names(ckpt_dir: Path) -> set[str]:
    if not ckpt_dir.exists():
        return set()
    return {p.name for p in ckpt_dir.iterdir()}


def _training_info(ckpt_dir: Path, step: int) -> dict[str, Any]:
    with open(ckpt_dir / f"step_{step}" / "training_info.json") as f:
        return json.load(f)


# ── counter restore ──────────────────────────────────────────────────────────


class TestCounterRestore:
    def test_restore_from_step_n(self, tmp_path):
        save_state = _default_grpo_save_state()
        save_state["current_step"] = 7
        save_state["consumed_samples"] = 42
        save_state["total_valid_tokens"] = 1234

        actor = _ACTOR_CLS(_actor_master_config(tmp_path), _make_bundle(save_state=save_state))

        assert actor._train_steps == 7
        assert actor._trainer_version == 7
        # Fresh-start invariant: _max_rollout_version == _trainer_version - 1.
        assert actor._max_rollout_version == 6
        assert actor._consumed_samples == 42
        assert actor._total_valid_tokens == 1234

    def test_fresh_start_defaults(self, tmp_path):
        actor = _ACTOR_CLS(_actor_master_config(tmp_path), _make_bundle())

        assert actor._train_steps == 0
        assert actor._trainer_version == 0
        assert actor._max_rollout_version == -1
        assert actor._consumed_samples == 0
        assert actor._total_valid_tokens == 0

    def test_old_checkpoint_without_total_valid_tokens(self, tmp_path):
        # Older checkpoints may predate the total_valid_tokens key.
        save_state = {
            "consumed_samples": 10,
            "current_step": 5,
            "current_epoch": 0,
            "total_steps": 5,
        }

        actor = _ACTOR_CLS(_actor_master_config(tmp_path), _make_bundle(save_state=save_state))

        assert actor._train_steps == 5
        assert actor._max_rollout_version == 4
        assert actor._total_valid_tokens == 0


# ── save trigger + write path ────────────────────────────────────────────────


class TestSaveTrigger:
    def test_saves_on_period_boundary_and_last_step(self, tmp_path):
        mc = _actor_master_config(tmp_path, max_num_steps=4, save_period=2)
        trainer = _FakeTrainer()

        actor = _run_train_pump(mc, _make_bundle(trainer=trainer))

        assert actor._train_steps == 4
        ckpt_dir = tmp_path / "checkpoints"
        # Finalized exactly at steps 2 and 4; no tmp_step_* leftovers.
        assert _step_dir_names(ckpt_dir) == {"step_2", "step_4"}

        info_2 = _training_info(ckpt_dir, 2)
        assert info_2["current_step"] == 2
        assert info_2["total_steps"] == 2
        assert info_2["consumed_samples"] == 4  # 2 prompts/step * 2 steps
        # No validation ran, so the default val_reward is dropped.
        assert "val_reward" not in info_2

        info_4 = _training_info(ckpt_dir, 4)
        assert info_4["current_step"] == 4
        assert info_4["consumed_samples"] == 8

        # config.yaml is dumped next to training_info.json.
        assert (ckpt_dir / "step_2" / "config.yaml").exists()

        # save_checkpoint was called into the tmp dir with all three paths.
        assert len(trainer.save_calls) == 2
        first = trainer.save_calls[0]
        assert first["weights_path"] == str(
            ckpt_dir / "tmp_step_2" / "policy" / "weights"
        )
        assert first["optimizer_path"] == str(
            ckpt_dir / "tmp_step_2" / "policy" / "optimizer"
        )
        assert first["tokenizer_path"] == str(
            ckpt_dir / "tmp_step_2" / "policy" / "tokenizer"
        )
        assert first["checkpointing_cfg"] is mc.checkpointing
        assert trainer.save_calls[1]["weights_path"] == str(
            ckpt_dir / "tmp_step_4" / "policy" / "weights"
        )

        # The tmp dirs were finalized: policy/* survive under step_*.
        assert (ckpt_dir / "step_2" / "policy" / "weights").is_dir()
        assert (ckpt_dir / "step_2" / "policy" / "optimizer").is_dir()
        assert (ckpt_dir / "step_4" / "policy" / "tokenizer").is_dir()

    def test_last_step_saves_off_period_boundary(self, tmp_path):
        mc = _actor_master_config(tmp_path, max_num_steps=3, save_period=2)

        _run_train_pump(mc, _make_bundle())

        # step 2 (boundary) + step 3 (last step), no step_1.
        assert _step_dir_names(tmp_path / "checkpoints") == {"step_2", "step_3"}

    def test_save_optimizer_false_gates_optimizer_path(self, tmp_path):
        mc = _actor_master_config(
            tmp_path, max_num_steps=2, save_period=2, save_optimizer=False
        )
        trainer = _FakeTrainer()

        _run_train_pump(mc, _make_bundle(trainer=trainer))

        assert len(trainer.save_calls) == 1
        assert trainer.save_calls[0]["optimizer_path"] is None
        ckpt_dir = tmp_path / "checkpoints"
        assert (ckpt_dir / "step_2" / "policy" / "weights").is_dir()
        assert not (ckpt_dir / "step_2" / "policy" / "optimizer").exists()

    def test_no_save_when_disabled(self, tmp_path):
        mc = _actor_master_config(
            tmp_path, max_num_steps=2, save_period=1, enabled=False
        )
        trainer = _FakeTrainer()

        actor = _run_train_pump(mc, _make_bundle(trainer=trainer))

        assert actor._train_steps == 2
        assert trainer.save_calls == []
        assert _step_dir_names(tmp_path / "checkpoints") == set()

    def test_timeout_saves_and_stops_training_early(self, tmp_path):
        # 0-second budget: the first check_save() fires; the pump must save
        # at step 1 (off the period boundary) and break out of the loop.
        mc = _actor_master_config(
            tmp_path,
            max_num_steps=4,
            save_period=100,
            checkpoint_must_save_by="00:00:00:00",
        )
        trainer = _FakeTrainer()

        actor = _run_train_pump(mc, _make_bundle(trainer=trainer))

        assert actor._train_steps == 1
        assert len(trainer.save_calls) == 1
        assert _step_dir_names(tmp_path / "checkpoints") == {"step_1"}


# ── metric_name behavior ─────────────────────────────────────────────────────


class TestMetricName:
    def test_val_metric_without_validation_warns_and_still_saves(self, tmp_path):
        mc = _actor_master_config(
            tmp_path, max_num_steps=2, save_period=2, metric_name="val:accuracy"
        )

        with pytest.warns(UserWarning, match="no val metrics were collected"):
            _run_train_pump(mc, _make_bundle())

        ckpt_dir = tmp_path / "checkpoints"
        assert _step_dir_names(ckpt_dir) == {"step_2"}
        assert "val:accuracy" not in _training_info(ckpt_dir, 2)

    def test_train_metric_lands_in_training_info(self, tmp_path):
        mc = _actor_master_config(
            tmp_path, max_num_steps=2, save_period=2, metric_name="train:loss"
        )
        trainer = _FakeTrainer(step_metrics={"loss": 0.5})

        _run_train_pump(mc, _make_bundle(trainer=trainer))

        info = _training_info(tmp_path / "checkpoints", 2)
        assert info["train:loss"] == 0.5

    def test_train_metric_missing_key_raises(self, tmp_path):
        mc = _actor_master_config(
            tmp_path, max_num_steps=2, save_period=2, metric_name="train:not_a_metric"
        )

        with pytest.raises(ValueError, match="not found in train metrics"):
            _run_train_pump(mc, _make_bundle())

    def test_metric_name_requires_train_or_val_prefix(self, tmp_path):
        mc = _actor_master_config(
            tmp_path, max_num_steps=2, save_period=2, metric_name="reward"
        )

        with pytest.raises(AssertionError, match="must start with"):
            _run_train_pump(mc, _make_bundle())


# ── setup resume-path wiring ─────────────────────────────────────────────────


_STEP_3_SAVE_STATE = {
    "consumed_samples": 24,
    "current_step": 3,
    "current_epoch": 1,
    "total_steps": 3,
    "total_valid_tokens": 999,
}


def _write_checkpoint(
    ckpt_dir: Path,
    step: int,
    save_state: dict[str, Any],
    *,
    with_optimizer: bool = True,
) -> Path:
    step_dir = ckpt_dir / f"step_{step}"
    (step_dir / "policy" / "weights").mkdir(parents=True)
    if with_optimizer:
        (step_dir / "policy" / "optimizer").mkdir(parents=True)
    with open(step_dir / "training_info.json", "w") as f:
        json.dump(save_state, f)
    return step_dir


def _setup_master_config(checkpoint_dir: str) -> MasterConfig:
    """Partially-populated MasterConfig for setup_single_controller tests.

    Same shape as test_single_controller_setup._make_master_config, plus the
    checkpointing block setup now reads.
    """
    return MasterConfig.model_construct(
        data_plane={"enabled": True, "impl": "transfer_queue"},
        data={
            "use_multiple_dataloader": False,
            "shuffle": False,
            "num_workers": 0,
            "train": [{"env_name": "math"}],
        },
        grpo={
            "max_num_steps": 100,
            "max_num_epochs": 1,
            "num_prompts_per_step": 4,
            "num_generations_per_prompt": 2,
            "max_rollout_turns": 1,
            "seed": 42,
        },
        policy={
            "max_total_sequence_length": 32,
            "megatron_cfg": {"enabled": False},
            "generation": {
                "backend": "vllm",
                "colocated": {"enabled": True, "resources": {}},
            },
        },
        loss_fn=ClippedPGLossConfig(),
        env={},
        checkpointing={
            "enabled": True,
            "checkpoint_dir": checkpoint_dir,
            "metric_name": None,
            "higher_is_better": True,
            "keep_top_k": None,
            "save_period": 2,
            "save_optimizer": True,
            "checkpoint_must_save_by": None,
        },
    )


class TestGetResumePaths:
    def test_resume_paths_from_fixture_layout(self, tmp_path):
        step_dir = _write_checkpoint(tmp_path, 3, _STEP_3_SAVE_STATE)

        weights_path, optimizer_path = CheckpointManager.get_resume_paths(
            str(step_dir)
        )

        assert weights_path == step_dir / "policy" / "weights"
        assert optimizer_path == step_dir / "policy" / "optimizer"

    def test_resume_paths_without_optimizer_state(self, tmp_path):
        step_dir = _write_checkpoint(
            tmp_path, 3, _STEP_3_SAVE_STATE, with_optimizer=False
        )

        with pytest.warns(UserWarning, match="Optimizer state not found"):
            weights_path, optimizer_path = CheckpointManager.get_resume_paths(
                str(step_dir)
            )

        assert weights_path == step_dir / "policy" / "weights"
        assert optimizer_path is None

    def test_no_checkpoint_gives_none(self):
        assert CheckpointManager.get_resume_paths(None) == (None, None)


class TestSetupResumeWiring:
    def test_setup_forwards_latest_resume_paths(self, patched_factories, tmp_path):
        ckpt_dir = tmp_path / "ckpts"
        _write_checkpoint(ckpt_dir, 1, {**_STEP_3_SAVE_STATE, "current_step": 1})
        step_3 = _write_checkpoint(ckpt_dir, 3, _STEP_3_SAVE_STATE)
        mc = _setup_master_config(str(ckpt_dir))

        bundle = setup_single_controller(mc, MagicMock(pad_token_id=0))

        # Latest checkpoint (step_3) wins; its paths reach the trainer factory.
        trainer_kwargs = patched_factories["_build_trainer"].call_args.kwargs
        assert trainer_kwargs["weights_path"] == step_3 / "policy" / "weights"
        assert trainer_kwargs["optimizer_path"] == step_3 / "policy" / "optimizer"
        # training_info.json is loaded into the bundle for the actor.
        assert bundle.save_state == _STEP_3_SAVE_STATE

    def test_setup_fresh_start_passes_none_paths(self, patched_factories, tmp_path):
        ckpt_dir = tmp_path / "empty_ckpts"
        ckpt_dir.mkdir()
        mc = _setup_master_config(str(ckpt_dir))

        bundle = setup_single_controller(mc, MagicMock(pad_token_id=0))

        trainer_kwargs = patched_factories["_build_trainer"].call_args.kwargs
        assert trainer_kwargs["weights_path"] is None
        assert trainer_kwargs["optimizer_path"] is None
        assert bundle.save_state == _default_grpo_save_state()

    def test_setup_forwards_pretrained_checkpoint(self, patched_factories, tmp_path):
        mc = _setup_master_config(str(tmp_path / "ckpts"))
        pretrained = {"path": "/some/ckpt", "format": "megatron_bridge"}
        mc.checkpointing["pretrained_checkpoint"] = pretrained

        setup_single_controller(mc, MagicMock(pad_token_id=0))

        assert mc.policy["pretrained_checkpoint"] == pretrained
