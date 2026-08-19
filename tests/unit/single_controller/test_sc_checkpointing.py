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

"""Unit tests for SC checkpointing.

Covers:
  - counter restore from save_state (train_steps / trainer_version / sampler
    dispatch cursor, current_epoch);
  - save trigger + write path through _train_pump with fakes (period
    boundary, last step, timeout, disabled);
  - async-save finalization (rename deferred until finalize_async_save,
    background failure re-raised at the next save and, for the last save,
    at shutdown);
  - metric_name behavior (non-"train:" rejected at config validation,
    train:* value recorded);
  - dataloader state: train_dataloader.pt written at save, position
    round-trip through a real StatefulDataLoader, dataset-swap guard,
    setup restore wiring + missing-file corruption check;
  - native replay persistence requires both sampler support and TQ checkpointing;
  - setup_single_controller resume-path wiring (get_resume_paths forwarded
    to the trainer factory, save_state loaded from training_info.json).
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import threading
from pathlib import Path
from typing import Any, Optional, Union
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import torch
import yaml
from torchdata.stateful_dataloader import StatefulDataLoader

from nemo_rl.algorithms.async_utils.replay_buffer import (
    LEGACY_REPLAY_BUFFER_FILENAME,
    REPLAY_BUFFER_METADATA_FILENAME,
    REPLAY_BUFFER_METADATA_SCHEMA_VERSION,
    REPLAY_BUFFER_METADATA_STORAGE,
    DataPlaneCheckpointBarrier,
    replay_manifest_digest,
)
from nemo_rl.algorithms.async_utils.staleness_sampler import (
    WeightFifoSamplerConfig,
    WindowedSamplerConfig,
)
from nemo_rl.algorithms.grpo import (
    GRPOConfig,
    GRPOSaveState,
    _get_grpo_save_state,
    _initial_grpo_save_state,
)
from nemo_rl.algorithms.loss import ClippedPGLossConfig
from nemo_rl.algorithms.metric_utils import SetupTimingMetrics
from nemo_rl.algorithms.single_controller import SingleControllerActor
from nemo_rl.algorithms.single_controller_utils import (
    AsyncRLConfig,
    MasterConfig,
    RolloutCheckpointConfig,
    setup_single_controller,
)
from nemo_rl.algorithms.single_controller_utils.rollout_checkpoint import (
    BOOTSTRAP_DIRNAME,
    ROLLOUT_SNAPSHOT_COMMITTED_FILENAME,
    ROLLOUT_SNAPSHOT_MANIFEST_FILENAME,
)
from nemo_rl.algorithms.single_controller_utils.setup import SingleControllerActorArgs
from nemo_rl.data.utils import load_dataloader_state
from nemo_rl.data_plane import DATA_PLANE_CHECKPOINT_SCHEMA_VERSION, KVBatchMeta
from nemo_rl.experience.rollout_recovery import (
    ROLLOUT_RECOVERY_SCHEMA_VERSION,
    ROLLOUT_RECOVERY_STATE_FILENAME,
    PromptRef,
    RolloutRecoveryLedger,
)
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


class _SteppingClock:
    def __init__(self, *, start: float = 0.0, step: float = 1.0) -> None:
        self._next = start
        self._step = step

    def __call__(self) -> float:
        value = self._next
        self._next += self._step
        return value


_STAGING_PARTITION_ID = "rollout_staging"


# ── fakes ────────────────────────────────────────────────────────────────────


class _FakeTrainer:
    """TQPolicy stand-in: train methods are no-ops, save_checkpoint records calls."""

    def __init__(self, step_metrics: Optional[dict[str, Any]] = None) -> None:
        self._step_metrics = dict(step_metrics or {})
        self.save_calls: list[dict[str, Any]] = []
        self.finalize_calls: int = 0

    def prepare_for_lp_inference(self) -> None:
        pass

    def get_logprobs_from_meta(self, meta: KVBatchMeta) -> None:
        pass

    def get_reference_policy_logprobs_from_meta(self, meta: KVBatchMeta) -> None:
        pass

    def prepare_for_training(self) -> None:
        pass

    def begin_train_step(self, loss_fn: Any) -> None:
        pass

    def train_microbatches_from_meta(self, meta: KVBatchMeta) -> None:
        pass

    def finish_train_step(self) -> dict[str, Any]:
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

    def finalize_async_save(self) -> None:
        self.finalize_calls += 1


class _GatedFinalizeTrainer(_FakeTrainer):
    """Async-save stand-in: finalize_async_save blocks until released."""

    def __init__(self) -> None:
        super().__init__()
        self.release = threading.Event()

    def finalize_async_save(self) -> None:
        assert self.release.wait(timeout=30.0), "test never released the writer"
        super().finalize_async_save()


class _FailingFinalizeTrainer(_FakeTrainer):
    def finalize_async_save(self) -> None:
        raise RuntimeError("injected async-writer failure")


class _FakeSampler:
    """PromptGroupSampler stand-in: always returns a full, fresh batch."""

    def __init__(self, supports_buffer_checkpoint: bool = True) -> None:
        self._supports_buffer_checkpoint = supports_buffer_checkpoint
        self._step = 0

    async def admit(self, *, trainer_version_fn) -> Optional[int]:
        return None

    async def evict(self, *, current_train_weight: int) -> int:
        return 0

    async def select(
        self,
        *,
        current_train_weight: int,
        min_prompt_groups: int,
        max_prompt_groups: int,
    ) -> tuple[KVBatchMeta, int]:
        n = max_prompt_groups
        # Canonical rollout rows are named ``{group_id}_g{generation_index}``.
        # This fixture models one physical row per prompt group, so give each
        # synthetic group its generation-0 row rather than using the legacy
        # suffix-free IDs that the train pump now correctly rejects.
        sample_ids = [f"s{self._step}-{i}_g0" for i in range(n)]
        self._step += 1
        meta = KVBatchMeta(
            partition_id=_PARTITION_ID,
            task_name=None,
            sample_ids=sample_ids,
            sequence_lengths=[16] * n,
            tags=[{"weight_version": current_train_weight}] * n,
        )
        return meta, n

    @property
    def is_on_policy(self) -> bool:
        return False

    @property
    def supports_buffer_checkpoint(self) -> bool:
        return self._supports_buffer_checkpoint

    def required_buffer_capacity(self, groups_per_step: int) -> Optional[int]:
        return None

    def set_dispatch_index(self, resume_from_trainer_version: int) -> None:
        pass


class _ExhaustingSampler(_FakeSampler):
    """Serves exactly ``steps`` full batches, then reports no data forever."""

    def __init__(self, steps: int) -> None:
        super().__init__()
        self._remaining = steps

    async def select(self, **kwargs) -> tuple[Optional[KVBatchMeta], int]:
        if self._remaining == 0:
            return None, 0
        self._remaining -= 1
        return await super().select(**kwargs)


class _RestoredGroupsSampler(_FakeSampler):
    """Drain the exact groups represented by a restored metadata sidecar."""

    def __init__(self, groups: list[dict[str, Any]]) -> None:
        super().__init__()
        self._groups = list(groups)

    async def select(
        self,
        *,
        current_train_weight: int,
        min_prompt_groups: int,
        max_prompt_groups: int,
    ) -> tuple[Optional[KVBatchMeta], int]:
        del current_train_weight
        selected = self._groups[:max_prompt_groups]
        if len(selected) < min_prompt_groups:
            return None, 0
        del self._groups[: len(selected)]

        metas = [group["meta"] for group in selected]
        return (
            KVBatchMeta(
                partition_id=_PARTITION_ID,
                task_name=None,
                sample_ids=[sid for meta in metas for sid in meta.sample_ids],
                sequence_lengths=[
                    length for meta in metas for length in (meta.sequence_lengths or [])
                ],
                tags=[tag for meta in metas for tag in (meta.tags or [])],
            ),
            len(selected),
        )


class _FakeDPClient:
    def __init__(
        self,
        *,
        save_error: Optional[Exception] = None,
        sample_ids: Optional[list[str]] = None,
        staging_sample_ids: Optional[list[str]] = None,
    ) -> None:
        self.clear_calls: list[tuple[list[str], str]] = []
        self.clear_thread_ids: list[int] = []
        self.list_calls: list[str] = []
        self.save_calls: list[dict[str, Any]] = []
        self.save_error = save_error
        self.sample_ids_by_partition = {
            _PARTITION_ID: list(sample_ids or []),
            _STAGING_PARTITION_ID: list(staging_sample_ids or []),
        }
        # Preserve the existing canonical-row test surface while modelling the
        # staging partition independently.
        self.sample_ids = self.sample_ids_by_partition[_PARTITION_ID]

    def list_sample_ids(self, partition_id: str) -> list[str]:
        self.list_calls.append(partition_id)
        assert partition_id in self.sample_ids_by_partition
        return sorted(self.sample_ids_by_partition[partition_id])

    def clear_samples(self, sample_ids: list[str], partition_id: str) -> None:
        assert partition_id in self.sample_ids_by_partition
        self.clear_thread_ids.append(threading.get_ident())
        self.clear_calls.append((list(sample_ids), partition_id))
        cleared = set(sample_ids)
        partition_sample_ids = self.sample_ids_by_partition[partition_id]
        partition_sample_ids[:] = [
            sample_id for sample_id in partition_sample_ids if sample_id not in cleared
        ]

    def save_checkpoint(
        self,
        checkpoint_dir: str,
        *,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        self.save_calls.append(
            {
                "checkpoint_dir": checkpoint_dir,
                "metadata": dict(metadata or {}),
            }
        )
        if self.save_error is not None:
            raise self.save_error
        os.makedirs(checkpoint_dir, exist_ok=True)
        with open(os.path.join(checkpoint_dir, "metadata.json"), "w") as f:
            json.dump({"user_metadata": metadata or {}}, f)


class _BlockingDPClient(_FakeDPClient):
    def __init__(self) -> None:
        super().__init__()
        self.save_started = threading.Event()
        self.release_save = threading.Event()

    def save_checkpoint(
        self,
        checkpoint_dir: str,
        *,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        self.save_started.set()
        assert self.release_save.wait(timeout=30.0), "test never released TQ save"
        super().save_checkpoint(checkpoint_dir, metadata=metadata)


class _FakeWeightSynchronizer:
    def __init__(self) -> None:
        self.sync_count = 0
        self.shutdown_count = 0

    def sync_weights(self, *, kv_scales: Any = None) -> None:
        self.sync_count += 1

    def shutdown(self) -> None:
        self.shutdown_count += 1


class _FakeGeneration:
    """Generation stand-in for weight-version updates during actor startup."""

    requires_kv_scale_sync = False

    def __init__(self) -> None:
        self.rollout_weight_versions: list[int] = []

    def set_rollout_weight_version(self, version: int) -> None:
        self.rollout_weight_versions.append(version)

    def drain_latest_logger_metrics(self) -> dict[str, Any]:
        return {}


class _FakeRolloutManager:
    def __init__(self, recovery_ledger: Optional[RolloutRecoveryLedger] = None) -> None:
        self.weight_versions: list[int] = []
        self._tq_buffer = None
        self.recovery_ledger = recovery_ledger
        self.checkpoint_barrier: Optional[DataPlaneCheckpointBarrier] = None
        self.recovery_calls: list[str] = []
        self.discard_calls: list[str] = []
        self.telemetry = {
            "canonical_groups_finalized": 0,
            "canonical_output_tokens": 0,
            "recovery_siblings_reused": 0,
            "recovery_siblings_redispatched": 0,
        }

    def set_weight_version(self, version: int) -> None:
        self.weight_versions.append(version)

    def set_data_plane_checkpoint_barrier(
        self, barrier: DataPlaneCheckpointBarrier
    ) -> None:
        self.checkpoint_barrier = barrier

    async def recover_for_finalization(self, group_id: str, **_: Any) -> None:
        self.recovery_calls.append(group_id)
        assert self.recovery_ledger is not None
        self.recovery_ledger.discard_group(group_id)

    async def discard_recovery_group(self, group_id: str) -> None:
        self.discard_calls.append(group_id)
        assert self.recovery_ledger is not None
        self.recovery_ledger.discard_group(group_id)

    def telemetry_snapshot(self) -> dict[str, int]:
        return dict(self.telemetry)

    def record_canonical_publication(self, output_tokens: int) -> None:
        self.telemetry["canonical_groups_finalized"] += 1
        self.telemetry["canonical_output_tokens"] += output_tokens

    def record_recovery_siblings(self, *, reused: int, redispatched: int) -> None:
        self.telemetry["recovery_siblings_reused"] += reused
        self.telemetry["recovery_siblings_redispatched"] += redispatched


def _unfinished_recovery_ledger(group_count: int) -> RolloutRecoveryLedger:
    ledger = RolloutRecoveryLedger()
    for index in range(group_count):
        ledger.reserve_group(
            group_id=f"recovery-g{index}",
            prompt_id=str(index),
            prompt_ref=PromptRef(sample_id=str(index), task_name="nemo_gym"),
            prompt_payload={"idx": index, "task_name": "nemo_gym"},
            expected_generations=2,
            target_step=None,
            start_weight_version=0,
        )
    return ledger


class _FakeTQBuffer:
    """TQReplayBuffer stand-in for the SC save/restore integration tests."""

    def __init__(
        self,
        metadata_state: Optional[dict[str, Any]] = None,
        load_return: int = 0,
    ) -> None:
        # Empty like a drained buffer; the pump's exhaustion checks len() it.
        self._num_groups = 0
        self._metadata_state = metadata_state or {
            "schema_version": REPLAY_BUFFER_METADATA_SCHEMA_VERSION,
            "storage": REPLAY_BUFFER_METADATA_STORAGE,
            "partition_id": _PARTITION_ID,
            "saved_capacity": 4,
            "manifest_digest": "fake-manifest-digest",
            "groups": [],
        }
        self.load_return = load_return
        self._group_ids: tuple[str, ...] = ()
        self._target_step_list: list[Optional[int]] = []
        self.metadata_state_dict_calls: list[int] = []
        self.load_calls: list[dict[str, Any]] = []
        self.checkpoint_barrier: Optional[DataPlaneCheckpointBarrier] = None

    @property
    def group_ids(self) -> tuple[str, ...]:
        return self._group_ids

    @property
    def target_step_list(self) -> tuple[Optional[int], ...]:
        return tuple(self._target_step_list)

    def set_data_plane_checkpoint_barrier(
        self, barrier: DataPlaneCheckpointBarrier
    ) -> None:
        self.checkpoint_barrier = barrier

    def metadata_state_dict(
        self,
        *,
        saved_capacity: int,
        additional_groups: Optional[list[dict[str, Any]]] = None,
    ) -> dict[str, Any]:
        self.metadata_state_dict_calls.append(saved_capacity)
        state = dict(self._metadata_state)
        if additional_groups:
            state["groups"] = [*state["groups"], *additional_groups]
            state["manifest_digest"] = replay_manifest_digest(state["groups"])
        return state

    async def load_state_dict(
        self,
        state: dict[str, Any],
        *,
        max_groups: int,
        expected_partition_id: str,
        expected_group_size: int,
        expected_manifest_digest: str,
    ) -> int:
        self.load_calls.append(
            {
                "state": state,
                "max_groups": max_groups,
                "expected_partition_id": expected_partition_id,
                "expected_group_size": expected_group_size,
                "expected_manifest_digest": expected_manifest_digest,
            }
        )
        groups = state.get("groups", [])
        self._group_ids = tuple(group["group_id"] for group in groups)
        self._target_step_list = [group.get("target_step") for group in groups]
        self._num_groups = self.load_return
        return self.load_return

    def __len__(self) -> int:
        return self._num_groups


# Default position sentinel the fake dataloader reports via state_dict().
_SENTINEL_DL_STATE = {"fake_position": 42}


class _FakeDataloader(list):
    """List-backed dataloader with the StatefulDataLoader state_dict surface.

    The save block snapshots ``self._dataloader.state_dict()``; return a
    sentinel dict so tests can assert the exact object written to
    train_dataloader.pt.
    """

    def __init__(self, batches: Any = (), state: Optional[dict[str, Any]] = None):
        super().__init__(batches)
        self._state = dict(state) if state is not None else dict(_SENTINEL_DL_STATE)

    def state_dict(self) -> dict[str, Any]:
        return dict(self._state)


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
    ft_save_period: Optional[int] = None,
    num_prompts_per_step: int = 2,
    max_num_epochs: int = 1,
    buffer_checkpoint: bool = False,
    data_plane_checkpoint: bool = False,
    token_capture: bool = False,
    rollout_checkpoint_interval_s: Optional[float] = None,
    rollout_checkpoint_keep_latest_k: int = 2,
) -> MasterConfig:
    """MasterConfig for in-process SingleControllerActor tests.

    All fields are populated (init_tmp_checkpoint dumps the whole config to
    config.yaml); values satisfy validate_single_controller_config.
    """
    sampler_cfg = (
        WindowedSamplerConfig(max_staleness_versions=1)
        if buffer_checkpoint
        else WeightFifoSamplerConfig(max_staleness_versions=1)
    )
    config = MasterConfig.model_construct(
        policy={
            # One optimizer.step per RL step: prompts * generations == gbs.
            "train_global_batch_size": num_prompts_per_step * 2,
        },
        loss_fn=ClippedPGLossConfig(),
        env={},
        data={"shuffle": False, "num_workers": 0},
        grpo=GRPOConfig.model_construct(
            max_num_steps=max_num_steps,
            max_num_epochs=max_num_epochs,
            num_prompts_per_step=num_prompts_per_step,
            num_generations_per_prompt=2,
            seed=42,
        ),
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
            "ft_save_period": ft_save_period,
        },
        data_plane={
            "enabled": True,
            "impl": "transfer_queue",
            "backend": "simple",
            "checkpointing_enabled": data_plane_checkpoint,
        },
        async_rl=AsyncRLConfig(
            sampler=sampler_cfg,
            min_groups_for_streaming_train=1,
            max_inflight_prompts=4,
            max_buffered_rollouts=4,
        ),
        rollout_checkpointing=RolloutCheckpointConfig(
            interval_s=rollout_checkpoint_interval_s,
            keep_latest_k=rollout_checkpoint_keep_latest_k,
        ),
    )
    config.token_capture.enabled = token_capture
    return config


def _make_actor_args(
    *,
    trainer: Optional[_FakeTrainer] = None,
    save_state: Optional[GRPOSaveState] = None,
    dataloader: Optional[_FakeDataloader] = None,
    tq_buffer: Optional[_FakeTQBuffer] = None,
    dp_client: Optional[_FakeDPClient] = None,
    last_checkpoint_path: Optional[str] = None,
    data_plane_checkpoint_metadata: Optional[dict[str, Any]] = None,
    rollout_manager: Optional[_FakeRolloutManager] = None,
    bootstrap_fingerprint: Optional[str] = None,
    rollout_checkpoint_load_metrics: Optional[dict[str, float]] = None,
) -> SingleControllerActorArgs:
    return SingleControllerActorArgs(
        gen_handle=_FakeGeneration(),
        trainer_handle=trainer if trainer is not None else _FakeTrainer(),
        env_handles={},
        train_cluster=None,  # type: ignore[arg-type]
        inference_cluster=None,  # type: ignore[arg-type]
        dp_client=dp_client if dp_client is not None else _FakeDPClient(),
        dataloader=dataloader if dataloader is not None else _FakeDataloader(),
        weight_synchronizer=_FakeWeightSynchronizer(),  # type: ignore[arg-type]
        advantage_estimator=None,
        loss_fn=object(),  # type: ignore[arg-type]
        rollout_manager=(
            rollout_manager if rollout_manager is not None else _FakeRolloutManager()
        ),  # type: ignore[arg-type]
        tq_buffer=tq_buffer if tq_buffer is not None else _FakeTQBuffer(),  # type: ignore[arg-type]
        partition_id=_PARTITION_ID,
        finalizer_actors=[],
        save_state=(
            save_state if save_state is not None else _initial_grpo_save_state()
        ),
        last_checkpoint_path=last_checkpoint_path,
        data_plane_checkpoint_metadata=data_plane_checkpoint_metadata,
        bootstrap_fingerprint=bootstrap_fingerprint,
        rollout_checkpoint_load_metrics=rollout_checkpoint_load_metrics,
    )


def _run_train_pump(
    mc: MasterConfig,
    actor_args: SingleControllerActorArgs,
    *,
    flush: bool = True,
):
    """Construct the actor in-process and drive _train_pump to completion.

    flush=True joins the (possibly async) checkpoint finalization afterwards,
    like run()'s exit path does, so step_N dirs are visible to assertions.
    """

    async def _main():
        actor = _ACTOR_CLS(mc, actor_args, SetupTimingMetrics())
        actor._sampler = _FakeSampler(
            supports_buffer_checkpoint=(mc.async_rl.sampler.name == "windowed")
        )
        # In-process runs have no Ray runtime; the pump only reads the GPU
        # count for a throughput metric.
        with patch("ray.cluster_resources", return_value={"GPU": 0}):
            await actor._train_pump()
        if flush:
            actor._checkpointer.shutdown()
        return actor

    return asyncio.run(_main())


def _run_actor_run(mc: MasterConfig, actor_args: SingleControllerActorArgs):
    """Construct the actor in-process and drive run() to completion.

    max_num_steps=0 makes _train_pump exit immediately, so run() executes
    only the restore block + pump startup/teardown. The wait_for bounds the
    would-be-deadlock cases (an over-capacity permit acquisition would hang
    run() forever).
    """

    async def _main():
        actor = _ACTOR_CLS(mc, actor_args, SetupTimingMetrics())
        result = await asyncio.wait_for(actor.run(), timeout=60.0)
        return actor, result

    return asyncio.run(_main())


def _run_restore_then_train_pump(
    mc: MasterConfig,
    actor_args: SingleControllerActorArgs,
    *,
    restored_groups: list[dict[str, Any]],
):
    """Restore the replay buffer, then drive a live _train_pump.

    Composes what _run_actor_run (max_num_steps=0) and _run_train_pump (no
    restore) each cover in isolation: the permits taken by the restore must be
    released by a running pump. The wait_for bounds a stalled pump.
    """

    async def _main():
        actor = _ACTOR_CLS(mc, actor_args, SetupTimingMetrics())
        await actor._maybe_restore_replay_buffer()
        actor._sampler = _RestoredGroupsSampler(restored_groups)
        with patch("ray.cluster_resources", return_value={"GPU": 0}):
            await asyncio.wait_for(actor._train_pump(), timeout=60.0)
        actor._checkpointer.shutdown()
        return actor

    return asyncio.run(_main())


def _step_dir_names(ckpt_dir: Path) -> set[str]:
    if not ckpt_dir.exists():
        return set()
    return {
        p.name for p in ckpt_dir.iterdir() if p.name != "latest_checkpoint_status.json"
    }


def _training_info(ckpt_dir: Path, step: int) -> dict[str, Any]:
    with open(ckpt_dir / f"step_{step}" / "training_info.json") as f:
        return json.load(f)


# ── counter restore ──────────────────────────────────────────────────────────


class TestCounterRestore:
    def test_restore_from_step_n(self, tmp_path):
        save_state = _initial_grpo_save_state()
        save_state.current_step = 7
        save_state.current_epoch = 2
        save_state.consumed_samples = 42
        save_state.total_valid_tokens = 1234

        actor = _ACTOR_CLS(
            _actor_master_config(tmp_path),
            _make_actor_args(save_state=save_state),
            SetupTimingMetrics(),
        )

        assert actor._train_steps == 7
        assert actor._trainer_version == 7
        # The sampler dispatch cursor is seeded to preserve the fresh-start
        # invariant _dispatch_index == trainer_version - 1.
        assert actor._sampler._dispatch_index == 6
        assert actor._consumed_samples == 42
        assert actor._current_epoch == 2
        assert actor._total_valid_tokens == 1234

    def test_restores_trainer_version_independently_from_train_step(self, tmp_path):
        save_state = _initial_grpo_save_state()
        save_state.current_step = 7
        save_state.trainer_version = 11

        actor = _ACTOR_CLS(
            _actor_master_config(tmp_path),
            _make_actor_args(save_state=save_state),
            SetupTimingMetrics(),
        )

        assert actor._train_steps == 7
        assert actor._trainer_version == 11
        assert actor._sampler._dispatch_index == 10

    def test_fresh_start_defaults(self, tmp_path):
        actor = _ACTOR_CLS(
            _actor_master_config(tmp_path), _make_actor_args(), SetupTimingMetrics()
        )

        assert actor._train_steps == 0
        assert actor._trainer_version == 0
        assert actor._sampler._dispatch_index == -1
        assert actor._consumed_samples == 0
        assert actor._current_epoch == 0
        assert actor._total_valid_tokens == 0

    def test_old_checkpoint_without_total_valid_tokens(self, tmp_path):
        # Older checkpoints may predate the total_valid_tokens key;
        # _get_grpo_save_state backfills it with the default.
        save_state = _get_grpo_save_state(
            {
                "consumed_samples": 10,
                "current_step": 5,
                "current_epoch": 0,
                "total_steps": 5,
            }
        )

        actor = _ACTOR_CLS(
            _actor_master_config(tmp_path),
            _make_actor_args(save_state=save_state),
            SetupTimingMetrics(),
        )

        assert actor._train_steps == 5
        assert actor._sampler._dispatch_index == 4
        assert actor._total_valid_tokens == 0

    def test_resumed_pump_continues_to_max_steps(self, tmp_path):
        # Composes counter restore with a live pump: resuming at step 2 and
        # running to max_num_steps=4 must yield 4 total steps (not 2, not 6),
        # with only the post-resume boundary checkpointed.
        mc = _actor_master_config(tmp_path, max_num_steps=4, save_period=2)
        save_state = _initial_grpo_save_state()
        save_state.current_step = 2
        save_state.consumed_samples = 4

        actor = _run_train_pump(mc, _make_actor_args(save_state=save_state))

        assert actor._train_steps == 4
        assert actor._trainer_version == 4
        # Steps 3 and 4 ran: one save at the step-4 boundary, none re-written
        # for the pre-resume step 2.
        assert _step_dir_names(tmp_path / "checkpoints") == {"step_4"}
        info = _training_info(tmp_path / "checkpoints", 4)
        assert info["current_step"] == 4
        assert info["consumed_samples"] == 4 + 2 * 2


# ── save trigger + write path ────────────────────────────────────────────────


class TestSaveTrigger:
    def test_saves_on_period_boundary_and_last_step(self, tmp_path):
        mc = _actor_master_config(tmp_path, max_num_steps=4, save_period=2)
        trainer = _FakeTrainer()

        actor = _run_train_pump(mc, _make_actor_args(trainer=trainer))

        assert actor._train_steps == 4
        ckpt_dir = tmp_path / "checkpoints"
        # Finalized exactly at steps 2 and 4; no tmp_step_* leftovers.
        assert _step_dir_names(ckpt_dir) == {"step_2", "step_4"}

        info_2 = _training_info(ckpt_dir, 2)
        assert info_2["current_step"] == 2
        assert info_2["trainer_version"] == 2
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

        # The async writers were waited on before each rename.
        assert trainer.finalize_calls == 2

        # The tmp dirs were finalized: policy/* survive under step_*.
        assert (ckpt_dir / "step_2" / "policy" / "weights").is_dir()
        assert (ckpt_dir / "step_2" / "policy" / "optimizer").is_dir()
        assert (ckpt_dir / "step_4" / "policy" / "tokenizer").is_dir()

    def test_last_step_saves_off_period_boundary(self, tmp_path):
        mc = _actor_master_config(tmp_path, max_num_steps=3, save_period=2)

        _run_train_pump(mc, _make_actor_args())

        # step 2 (boundary) + step 3 (last step), no step_1.
        assert _step_dir_names(tmp_path / "checkpoints") == {"step_2", "step_3"}

    def test_save_optimizer_false_gates_optimizer_path(self, tmp_path):
        mc = _actor_master_config(
            tmp_path, max_num_steps=2, save_period=2, save_optimizer=False
        )
        trainer = _FakeTrainer()

        _run_train_pump(mc, _make_actor_args(trainer=trainer))

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

        actor = _run_train_pump(mc, _make_actor_args(trainer=trainer))

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

        actor = _run_train_pump(mc, _make_actor_args(trainer=trainer))

        assert actor._train_steps == 1
        assert len(trainer.save_calls) == 1
        assert _step_dir_names(tmp_path / "checkpoints") == {"step_1"}

    def test_rollout_exhaustion_saves_final_checkpoint(self, tmp_path):
        # A resumed run can exhaust its data before max_num_steps:
        # _clamp_max_num_steps budgets against the full per-epoch batch count,
        # but the restored loader holds only the remaining batches. Once
        # rollout is exhausted and the buffer is drained, every completed step
        # is potentially the last one, so it must save — previously the pump
        # exited right after it with nothing written and exit code 0.
        mc = _actor_master_config(tmp_path, max_num_steps=4, save_period=100)
        trainer = _FakeTrainer()

        async def _main():
            actor = _ACTOR_CLS(
                mc, _make_actor_args(trainer=trainer), SetupTimingMetrics()
            )
            actor._sampler = _ExhaustingSampler(steps=2)
            actor._rollout_exhausted.set()
            with patch("ray.cluster_resources", return_value={"GPU": 0}):
                await actor._train_pump()
            actor._checkpointer.shutdown()
            return actor

        actor = asyncio.run(_main())

        # Stopped short of max_num_steps=4, but the completed steps saved.
        assert actor._train_steps == 2
        assert _step_dir_names(tmp_path / "checkpoints") == {"step_1", "step_2"}

    def test_ft_save_period_triggers_saves(self, tmp_path):
        # checkpointing.ft_save_period ORs into the save trigger like on
        # every other algorithm; silently ignoring it would break crash
        # recovery for users who configured it.
        mc = _actor_master_config(
            tmp_path, max_num_steps=3, save_period=100, ft_save_period=2
        )

        _run_train_pump(mc, _make_actor_args())

        # step_2 from ft_save_period, step_3 from last-step.
        assert _step_dir_names(tmp_path / "checkpoints") == {"step_2", "step_3"}


class TestPeriodicRolloutCheckpoint:
    @staticmethod
    def _make_actor(
        tmp_path: Path,
        *,
        ledger: Optional[RolloutRecoveryLedger] = None,
        dp_client: Optional[_FakeDPClient] = None,
        tq_buffer: Optional[_FakeTQBuffer] = None,
    ) -> Any:
        mc = _actor_master_config(
            tmp_path,
            buffer_checkpoint=True,
            data_plane_checkpoint=True,
            token_capture=True,
            rollout_checkpoint_interval_s=120.0,
        )
        ledger = ledger or RolloutRecoveryLedger()
        return _ACTOR_CLS(
            mc,
            _make_actor_args(
                rollout_manager=_FakeRolloutManager(ledger),
                dp_client=dp_client,
                tq_buffer=tq_buffer,
                bootstrap_fingerprint="bootstrap-v1",
            ),
            SetupTimingMetrics(),
        )

    def test_pre_step_snapshot_omits_trainer_payload(self, tmp_path: Path) -> None:
        actor = self._make_actor(tmp_path)

        assert asyncio.run(actor._save_rollout_checkpoint(force=True))
        actor._checkpointer.shutdown()

        snapshot = (
            tmp_path
            / "checkpoints"
            / BOOTSTRAP_DIRNAME
            / "rollout_snapshots"
            / "snapshot_000001"
        )
        assert (snapshot / ROLLOUT_SNAPSHOT_COMMITTED_FILENAME).is_file()
        assert (snapshot / ROLLOUT_SNAPSHOT_MANIFEST_FILENAME).is_file()
        assert (snapshot / "data_plane" / "metadata.json").is_file()
        assert (snapshot / "train_dataloader.pt").is_file()
        assert (snapshot / REPLAY_BUFFER_METADATA_FILENAME).is_file()
        assert (snapshot / ROLLOUT_RECOVERY_STATE_FILENAME).is_file()
        assert (snapshot / "config.yaml").is_file()
        assert not (snapshot / "policy").exists()

    def test_logs_snapshot_phase_durations(self, tmp_path: Path) -> None:
        actor = self._make_actor(tmp_path)
        actor._logger = MagicMock()
        actor._telemetry_started_at = 0.0

        with patch(
            "nemo_rl.algorithms.single_controller.time.monotonic",
            new=_SteppingClock(),
        ):
            assert asyncio.run(actor._save_rollout_checkpoint(force=True))

        logged = actor._logger.log_metrics.call_args.args[0]
        assert logged["snapshot_sequence"] == 1.0
        assert logged["total_save_seconds"] > 0
        assert 0 <= logged["barrier_wait_seconds"] <= logged["total_save_seconds"]
        assert 0 <= logged["tq_save_seconds"] <= logged["exclusive_hold_seconds"]
        assert logged["rolled_back_train_groups"] == 0.0
        assert actor._logger.log_metrics.call_args.kwargs == {
            "step": 1,
            "prefix": "timing/rollout_checkpoint",
            "step_metric": "telemetry/wall_time_seconds",
        }

    def test_logs_raw_and_canonical_rollout_throughput(self, tmp_path: Path) -> None:
        actor = self._make_actor(tmp_path)
        actor._logger = MagicMock()
        actor._telemetry_started_at = 0.0
        snapshots = iter(
            [
                {
                    "canonical_groups_finalized": 0,
                    "canonical_output_tokens": 0,
                    "recovery_siblings_reused": 0,
                    "recovery_siblings_redispatched": 0,
                },
                {
                    "canonical_groups_finalized": 2,
                    "canonical_output_tokens": 40,
                    "recovery_siblings_reused": 1,
                    "recovery_siblings_redispatched": 3,
                },
            ]
        )
        actor._rollout_manager.telemetry_snapshot = lambda: next(snapshots)
        generation_snapshots = iter(
            [
                {"generation_tokens": {0: [60], 1: [40]}},
                {
                    "generation_tokens": {0: [180], 1: [120]},
                    "inflight_batch_sizes": {0: [2], 1: [1]},
                    "num_pending_samples": {0: [1], 1: [3]},
                    "kv_cache_usage_perc": {0: [0.5], 1: [0.7]},
                },
            ]
        )
        actor._gen.drain_latest_logger_metrics = lambda: next(generation_snapshots)

        async def _sample_twice() -> None:
            await actor._log_rollout_throughput_metrics(emit=False)
            actor._rollout_completion_durations_s.extend([2.0, 4.0])
            actor._rollout_queue_wait_durations_s.extend([1.0, 3.0])
            await actor._log_rollout_throughput_metrics()

        with patch(
            "nemo_rl.algorithms.single_controller.time.monotonic",
            new=_SteppingClock(start=10.0, step=10.0),
        ):
            asyncio.run(_sample_twice())

        logged = actor._logger.log_metrics.call_args.args[0]
        assert logged["vllm_output_tokens_per_second"] == pytest.approx(20.0)
        assert logged["canonical_output_tokens_per_second"] == pytest.approx(4.0)
        assert logged["canonical_groups_per_second"] == pytest.approx(0.2)
        assert logged["vllm_requests_running"] == 3
        assert logged["vllm_requests_waiting"] == 4
        assert logged["vllm_kv_cache_usage_mean"] == pytest.approx(0.6)
        assert logged["group_completion_seconds_p50"] == pytest.approx(2.0)
        assert logged["group_completion_seconds_p95"] == pytest.approx(4.0)
        assert logged["group_queue_wait_seconds_p95"] == pytest.approx(3.0)

    def test_unchanged_state_is_not_saved_twice(self, tmp_path: Path) -> None:
        actor = self._make_actor(tmp_path)

        async def _save_twice() -> tuple[bool, bool]:
            return (
                await actor._save_rollout_checkpoint(),
                await actor._save_rollout_checkpoint(),
            )

        assert asyncio.run(_save_twice()) == (True, False)
        actor._checkpointer.shutdown()
        snapshot_root = (
            tmp_path / "checkpoints" / BOOTSTRAP_DIRNAME / "rollout_snapshots"
        )
        assert sorted(path.name for path in snapshot_root.glob("snapshot_*")) == [
            "snapshot_000001"
        ]

    def test_failed_save_retains_previous_committed_snapshot(
        self, tmp_path: Path
    ) -> None:
        dp_client = _FakeDPClient()
        actor = self._make_actor(tmp_path, dp_client=dp_client)

        async def _save_then_fail() -> None:
            assert await actor._save_rollout_checkpoint(force=True)
            dp_client.save_error = RuntimeError("injected periodic save failure")
            await actor._save_rollout_checkpoint(force=True)

        with pytest.raises(RuntimeError, match="injected periodic save failure"):
            asyncio.run(_save_then_fail())
        actor._checkpointer.shutdown()

        snapshot_root = (
            tmp_path / "checkpoints" / BOOTSTRAP_DIRNAME / "rollout_snapshots"
        )
        assert (snapshot_root / "snapshot_000001" / "COMMITTED").is_file()
        assert (snapshot_root / "LATEST").read_text().strip() == "snapshot_000001"
        assert not (snapshot_root / "snapshot_000002").exists()
        assert not (snapshot_root / "tmp_snapshot_000002").exists()

    def test_retention_keeps_recent_fallbacks(self, tmp_path: Path) -> None:
        mc = _actor_master_config(
            tmp_path,
            buffer_checkpoint=True,
            data_plane_checkpoint=True,
            token_capture=True,
            rollout_checkpoint_interval_s=120.0,
            rollout_checkpoint_keep_latest_k=2,
        )
        ledger = RolloutRecoveryLedger()
        actor = _ACTOR_CLS(
            mc,
            _make_actor_args(
                rollout_manager=_FakeRolloutManager(ledger),
                bootstrap_fingerprint="bootstrap-v1",
            ),
            SetupTimingMetrics(),
        )

        async def _save_three() -> None:
            for _ in range(3):
                assert await actor._save_rollout_checkpoint(force=True)

        asyncio.run(_save_three())
        actor._checkpointer.shutdown()
        snapshot_root = (
            tmp_path / "checkpoints" / BOOTSTRAP_DIRNAME / "rollout_snapshots"
        )
        assert sorted(path.name for path in snapshot_root.glob("snapshot_*")) == [
            "snapshot_000002",
            "snapshot_000003",
        ]
        assert (snapshot_root / "LATEST").read_text().strip() == "snapshot_000003"

    def test_post_step_snapshot_requires_matching_trainer_anchor(
        self, tmp_path: Path, capsys: Any
    ) -> None:
        actor = self._make_actor(tmp_path)
        actor._train_steps = 1
        actor._trainer_version = 1
        step_dir = tmp_path / "checkpoints" / "step_1"

        async def _save_without_then_with_anchor() -> tuple[bool, bool, bool]:
            first = await actor._save_rollout_checkpoint(force=True)
            second = await actor._save_rollout_checkpoint(force=True)
            step_dir.mkdir(parents=True)
            third = await actor._save_rollout_checkpoint(force=True)
            return first, second, third

        assert asyncio.run(_save_without_then_with_anchor()) == (False, False, True)
        actor._checkpointer.shutdown()
        assert (
            capsys.readouterr().out.count(
                "rollout checkpoint skipped: matching trainer checkpoint"
            )
            == 1
        )
        manifest = json.loads(
            (
                step_dir
                / "rollout_snapshots"
                / "snapshot_000001"
                / ROLLOUT_SNAPSHOT_MANIFEST_FILENAME
            ).read_text()
        )
        assert manifest["base_train_step"] == 1
        assert manifest["trainer_version"] == 1
        assert manifest["bootstrap_fingerprint"] is None

    def test_pre_step_deadline_retries_guarded_cut_before_stopping(
        self, tmp_path: Path
    ) -> None:
        actor = self._make_actor(tmp_path)
        actor._master_config.rollout_checkpointing.interval_s = 0.001
        actor._timeout.last_save_time = 0
        save = AsyncMock(side_effect=[False, True])

        with patch.object(actor, "_save_rollout_checkpoint", new=save):
            asyncio.run(asyncio.wait_for(actor._rollout_checkpoint_pump(), timeout=5))
        actor._checkpointer.shutdown()

        assert save.await_count == 2
        assert all(call.kwargs == {"force": True} for call in save.await_args_list)
        assert actor._timeout.last_saved is True
        assert actor._rollout_checkpoint_stop_requested.is_set()

    def test_active_streamed_step_is_persisted_as_replayable(
        self, tmp_path: Path
    ) -> None:
        ledger = RolloutRecoveryLedger()
        group = ledger.reserve_group(
            group_id="claimed",
            prompt_id="17",
            prompt_ref=PromptRef(sample_id="17", task_name="nemo_gym"),
            prompt_payload={"idx": 17, "task_name": "nemo_gym"},
            expected_generations=2,
            target_step=0,
            start_weight_version=0,
        )
        ledger.mark_group_dispatched(group.group_id)
        for generation_index, gate_id in enumerate(group.gate_rollout_ids):
            ledger.mark_sibling_sealed(
                group.group_id,
                generation_index=generation_index,
                gate_rollout_id=gate_id,
                receipt={
                    "rollout_id": gate_id,
                    "manifest": [{"staging_key": f"{gate_id}/call"}],
                },
                reward=float(generation_index),
            )
        canonical_meta = KVBatchMeta(
            partition_id=_PARTITION_ID,
            task_name="train",
            sample_ids=group.logical_rollout_ids,
            fields=["input_ids"],
            sequence_lengths=[16, 16],
            tags=[{"weight_version": 0}, {"weight_version": 0}],
        )
        ledger.mark_finalization_started(group.group_id)
        ledger.mark_group_finalized(
            group.group_id,
            meta=canonical_meta,
            group_min_weight_version=0,
            group_max_weight_version=0,
        )
        ledger.claim_groups_for_training(
            [group.group_id],
            train_step=0,
            trainer_version=0,
            expected_group_count=2,
        )
        dp_client = _FakeDPClient(
            sample_ids=list(canonical_meta.sample_ids),
            staging_sample_ids=sorted(ledger.expected_staging_keys()),
        )
        actor = self._make_actor(
            tmp_path,
            ledger=ledger,
            dp_client=dp_client,
            tq_buffer=_FakeTQBuffer(),
        )

        assert asyncio.run(actor._save_rollout_checkpoint(force=True))
        actor._checkpointer.shutdown()

        snapshot = (
            tmp_path
            / "checkpoints"
            / BOOTSTRAP_DIRNAME
            / "rollout_snapshots"
            / "snapshot_000001"
        )
        replay_state = torch.load(
            snapshot / REPLAY_BUFFER_METADATA_FILENAME,
            weights_only=False,
        )
        lineage_state = torch.load(
            snapshot / ROLLOUT_RECOVERY_STATE_FILENAME,
            weights_only=False,
        )
        restored = RolloutRecoveryLedger.from_state_dict(lineage_state)
        assert [item["group_id"] for item in replay_state["groups"]] == ["claimed"]
        assert restored.open_train_step is None
        assert restored.get_group("claimed").status.value == "finalized"
        assert ledger.open_train_step is not None
        manifest = json.loads(
            (snapshot / ROLLOUT_SNAPSHOT_MANIFEST_FILENAME).read_text()
        )
        assert manifest["rolled_back_train_group_count"] == 1


class TestDataPlaneCheckpoint:
    def test_saves_digest_bound_rollout_lineage_without_prompt_payload(self, tmp_path):
        mc = _actor_master_config(
            tmp_path,
            max_num_steps=1,
            save_period=1,
            buffer_checkpoint=True,
            data_plane_checkpoint=True,
        )
        mc.token_capture.enabled = True
        ledger = RolloutRecoveryLedger()
        ledger.reserve_group(
            group_id="partial-group",
            prompt_id="17",
            prompt_ref=PromptRef(sample_id="17", task_name="nemo_gym"),
            prompt_payload={
                "idx": 17,
                "task_name": "nemo_gym",
                "message_log": [{"role": "user", "content": "solve"}],
            },
            expected_generations=2,
            target_step=None,
            start_weight_version=0,
        )
        manager = _FakeRolloutManager(ledger)
        dp_client = _FakeDPClient()

        async def _main() -> None:
            actor = _ACTOR_CLS(
                mc,
                _make_actor_args(
                    dp_client=dp_client,
                    rollout_manager=manager,
                ),
                SetupTimingMetrics(),
            )
            actor._train_steps = 1
            actor._trainer_version = 1
            await actor._save_checkpoint({})
            actor._checkpointer.shutdown()

        asyncio.run(_main())

        step_dir = tmp_path / "checkpoints" / "step_1"
        payload = (step_dir / ROLLOUT_RECOVERY_STATE_FILENAME).read_bytes()
        metadata = dp_client.save_calls[0]["metadata"]
        assert metadata["rollout_recovery_schema_version"] == (
            ROLLOUT_RECOVERY_SCHEMA_VERSION
        )
        assert metadata["rollout_recovery_group_count"] == 1
        assert (
            metadata["rollout_recovery_payload_sha256"]
            == hashlib.sha256(payload).hexdigest()
        )
        state = torch.load(
            step_dir / ROLLOUT_RECOVERY_STATE_FILENAME, weights_only=False
        )
        assert state["groups"][0]["group_id"] == "partial-group"
        assert "prompt_payload" not in state["groups"][0]
        assert manager.checkpoint_barrier is not None
        assert dp_client.list_calls == [
            _PARTITION_ID,
            mc.token_capture.staging_partition,
        ]

    def test_saves_authoritative_tq_state_and_metadata_only_replay_index(
        self, tmp_path
    ):
        mc = _actor_master_config(
            tmp_path,
            max_num_steps=1,
            save_period=1,
            buffer_checkpoint=True,
            data_plane_checkpoint=True,
        )
        sample_ids = ["g0_g0", "g0_g1"]
        dp_client = _FakeDPClient(sample_ids=sample_ids)
        replay_metadata = {
            "schema_version": REPLAY_BUFFER_METADATA_SCHEMA_VERSION,
            "storage": REPLAY_BUFFER_METADATA_STORAGE,
            "partition_id": _PARTITION_ID,
            "saved_capacity": 4,
            "manifest_digest": "digest-1",
            "groups": [
                {
                    "meta": KVBatchMeta(
                        partition_id=_PARTITION_ID,
                        task_name="train",
                        sample_ids=sample_ids,
                        fields=["input_ids"],
                        sequence_lengths=[16, 16],
                        tags=[
                            {"weight_version": 0},
                            {"weight_version": 0},
                        ],
                    ),
                    "start_weight": 0,
                    "end_weight": 0,
                    "target_step": None,
                    "group_id": "g0",
                }
            ],
        }
        buffer = _FakeTQBuffer(metadata_state=replay_metadata)

        _run_train_pump(
            mc,
            _make_actor_args(dp_client=dp_client, tq_buffer=buffer),
        )

        assert len(dp_client.save_calls) == 1
        save_call = dp_client.save_calls[0]
        assert save_call["checkpoint_dir"] == str(
            tmp_path / "checkpoints" / "tmp_step_1" / "data_plane"
        )
        assert save_call["metadata"] == {
            "data_plane_checkpoint_schema_version": (
                DATA_PLANE_CHECKPOINT_SCHEMA_VERSION
            ),
            "single_controller_train_steps": 1,
            "single_controller_trainer_version": 1,
            "single_controller_epoch": 0,
            "partition_id": _PARTITION_ID,
            "sampler_name": "windowed",
            "mode": "authoritative",
            "replay_metadata_schema_version": REPLAY_BUFFER_METADATA_SCHEMA_VERSION,
            "replay_manifest_digest": "digest-1",
            "replay_group_count": 1,
        }
        step_dir = tmp_path / "checkpoints" / "step_1"
        assert (step_dir / "data_plane" / "metadata.json").is_file()
        assert (
            torch.load(step_dir / REPLAY_BUFFER_METADATA_FILENAME, weights_only=False)
            == replay_metadata
        )
        assert not (step_dir / "replay_buffer.pt").exists()
        assert buffer.metadata_state_dict_calls == [4]

    @pytest.mark.parametrize(
        ("actual_sample_ids", "error_fragment"),
        [
            (["g0_g0"], r"missing=\['g0_g1'\]"),
            (
                ["g0_g0", "g0_g1", "orphan_g0"],
                r"unexpected=\['orphan_g0'\]",
            ),
        ],
    )
    def test_tq_save_rejects_inventory_mismatch(
        self, tmp_path, actual_sample_ids, error_fragment
    ):
        mc = _actor_master_config(
            tmp_path,
            max_num_steps=1,
            save_period=1,
            buffer_checkpoint=True,
            data_plane_checkpoint=True,
        )
        sample_ids = ["g0_g0", "g0_g1"]
        replay_metadata = {
            "schema_version": REPLAY_BUFFER_METADATA_SCHEMA_VERSION,
            "storage": REPLAY_BUFFER_METADATA_STORAGE,
            "partition_id": _PARTITION_ID,
            "saved_capacity": 4,
            "manifest_digest": "digest-1",
            "groups": [
                {
                    "meta": KVBatchMeta(
                        partition_id=_PARTITION_ID,
                        task_name="train",
                        sample_ids=sample_ids,
                        fields=["input_ids"],
                        sequence_lengths=[16, 16],
                        tags=[
                            {"weight_version": 0},
                            {"weight_version": 0},
                        ],
                    ),
                    "start_weight": 0,
                    "end_weight": 0,
                    "target_step": None,
                    "group_id": "g0",
                }
            ],
        }

        with pytest.raises(RuntimeError, match=error_fragment):
            _run_train_pump(
                mc,
                _make_actor_args(
                    dp_client=_FakeDPClient(sample_ids=actual_sample_ids),
                    tq_buffer=_FakeTQBuffer(metadata_state=replay_metadata),
                ),
            )

        assert not (tmp_path / "checkpoints" / "step_1").exists()

    def test_gated_sampler_keeps_tq_checkpoint_in_shadow_mode(self, tmp_path):
        mc = _actor_master_config(
            tmp_path,
            max_num_steps=1,
            save_period=1,
            buffer_checkpoint=False,
            data_plane_checkpoint=True,
        )
        dp_client = _FakeDPClient()
        buffer = _FakeTQBuffer()

        _run_train_pump(
            mc,
            _make_actor_args(dp_client=dp_client, tq_buffer=buffer),
        )

        assert dp_client.save_calls[0]["metadata"]["mode"] == "shadow"
        step_dir = tmp_path / "checkpoints" / "step_1"
        assert not (step_dir / REPLAY_BUFFER_METADATA_FILENAME).exists()
        assert not (step_dir / "replay_buffer.pt").exists()
        assert buffer.metadata_state_dict_calls == []

    def test_tq_save_failure_aborts_checkpoint(self, tmp_path):
        mc = _actor_master_config(
            tmp_path,
            max_num_steps=1,
            save_period=1,
            data_plane_checkpoint=True,
        )
        dp_client = _FakeDPClient(save_error=RuntimeError("injected TQ failure"))

        with pytest.raises(RuntimeError, match="injected TQ failure"):
            _run_train_pump(mc, _make_actor_args(dp_client=dp_client))

        assert not (tmp_path / "checkpoints" / "step_1").exists()

    def test_consumed_clear_waits_for_tq_save(self, tmp_path):
        mc = _actor_master_config(
            tmp_path,
            max_num_steps=1,
            save_period=1,
            data_plane_checkpoint=True,
        )
        dp_client = _BlockingDPClient()

        async def _main() -> None:
            actor = _ACTOR_CLS(
                mc, _make_actor_args(dp_client=dp_client), SetupTimingMetrics()
            )
            actor._train_steps = 1
            actor._trainer_version = 1
            save_task = asyncio.create_task(actor._save_checkpoint({"loss": 1.0}))
            started = await asyncio.to_thread(dp_client.save_started.wait, 30.0)
            assert started

            clear_task = asyncio.create_task(
                actor._clear_data_plane_samples(["sample-0"])
            )
            await asyncio.sleep(0)
            assert dp_client.clear_calls == []

            dp_client.release_save.set()
            await save_task
            await clear_task
            actor._checkpointer.shutdown()

        asyncio.run(_main())
        assert dp_client.clear_calls == [(["sample-0"], _PARTITION_ID)]

    def test_consumed_clear_does_not_block_actor_event_loop(self, tmp_path):
        mc = _actor_master_config(tmp_path, max_num_steps=1, save_period=1)
        dp_client = _FakeDPClient()

        async def _main() -> int:
            actor = _ACTOR_CLS(
                mc, _make_actor_args(dp_client=dp_client), SetupTimingMetrics()
            )
            event_loop_thread_id = threading.get_ident()
            await actor._clear_data_plane_samples(["sample-0"])
            actor._checkpointer.shutdown()
            return event_loop_thread_id

        event_loop_thread_id = asyncio.run(_main())
        assert dp_client.clear_thread_ids
        assert dp_client.clear_thread_ids[0] != event_loop_thread_id


# ── async-save finalization ──────────────────────────────────────────────────


class TestAsyncSaveFinalization:
    def test_rename_deferred_until_async_writes_finish(self, tmp_path):
        mc = _actor_master_config(tmp_path, max_num_steps=2, save_period=2)
        trainer = _GatedFinalizeTrainer()

        actor = _run_train_pump(mc, _make_actor_args(trainer=trainer), flush=False)

        # The async writer hasn't finished: the checkpoint must still be a
        # tmp dir, invisible to resume lookups.
        ckpt_dir = tmp_path / "checkpoints"
        assert (ckpt_dir / "tmp_step_2").is_dir()
        assert not (ckpt_dir / "step_2").exists()

        trainer.release.set()
        actor._checkpointer.shutdown()

        assert (ckpt_dir / "step_2").is_dir()
        assert not (ckpt_dir / "tmp_step_2").exists()
        assert trainer.finalize_calls == 1

    def test_failed_background_finalization_raises_at_next_save(self, tmp_path):
        # The step-2 checkpoint's background finalization fails; the failure
        # must surface at the step-4 save's finalize_pending, not vanish.
        mc = _actor_master_config(tmp_path, max_num_steps=4, save_period=2)
        trainer = _FailingFinalizeTrainer()

        with pytest.raises(RuntimeError, match="finalization failed"):
            _run_train_pump(mc, _make_actor_args(trainer=trainer), flush=False)

    def test_failed_background_finalization_raises_at_shutdown(self, tmp_path):
        # Companion to the test above for the *last* save: nothing after it
        # calls finalize_pending, so the only thing that can surface the
        # failure is run()'s exit path, which goes through shutdown().
        mc = _actor_master_config(tmp_path, max_num_steps=2, save_period=2)
        trainer = _FailingFinalizeTrainer()

        actor = _run_train_pump(mc, _make_actor_args(trainer=trainer), flush=False)

        # The rename never happened: the checkpoint is still a tmp dir.
        ckpt_dir = tmp_path / "checkpoints"
        assert (ckpt_dir / "tmp_step_2").is_dir()
        assert not (ckpt_dir / "step_2").exists()

        with pytest.raises(RuntimeError, match="finalization failed"):
            actor._checkpointer.shutdown()


# ── metric_name behavior ─────────────────────────────────────────────────────


class TestMetricName:
    def test_val_metric_rejected_at_validation(self, tmp_path):
        # SC has no validation loop, so a "val:" metric would never be
        # collected and top-k retention would silently no-op. The config
        # validation rejects it up front instead of warning at every save.
        mc = _actor_master_config(
            tmp_path, max_num_steps=2, save_period=2, metric_name="val:accuracy"
        )

        with pytest.raises(ValueError, match="no validation loop yet"):
            _run_train_pump(mc, _make_actor_args())

        assert _step_dir_names(tmp_path / "checkpoints") == set()

    def test_train_metric_lands_in_training_info(self, tmp_path):
        mc = _actor_master_config(
            tmp_path, max_num_steps=2, save_period=2, metric_name="train:loss"
        )
        trainer = _FakeTrainer(step_metrics={"loss": 0.5})

        _run_train_pump(mc, _make_actor_args(trainer=trainer))

        info = _training_info(tmp_path / "checkpoints", 2)
        assert info["train:loss"] == 0.5

    def test_train_metric_missing_key_raises(self, tmp_path):
        mc = _actor_master_config(
            tmp_path, max_num_steps=2, save_period=2, metric_name="train:not_a_metric"
        )

        with pytest.raises(ValueError, match="not found in train metrics"):
            _run_train_pump(mc, _make_actor_args())

    def test_metric_name_requires_train_prefix(self, tmp_path):
        mc = _actor_master_config(
            tmp_path, max_num_steps=2, save_period=2, metric_name="reward"
        )

        with pytest.raises(ValueError, match="is not usable on the SingleController"):
            _run_train_pump(mc, _make_actor_args())


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
    save_state: Union[dict[str, Any], GRPOSaveState],
    *,
    with_optimizer: bool = True,
    dataloader_state: Optional[dict[str, Any]] = None,
    config: Optional[dict[str, Any]] = None,
) -> Path:
    step_dir = ckpt_dir / f"step_{step}"
    (step_dir / "policy" / "weights").mkdir(parents=True)
    if with_optimizer:
        (step_dir / "policy" / "optimizer").mkdir(parents=True)
    with open(step_dir / "training_info.json", "w") as f:
        # Mirror production serialization: the actor writes vars(save_state)
        # of the GRPOSaveState dataclass; plain dicts model legacy files.
        json.dump(save_state if isinstance(save_state, dict) else vars(save_state), f)
    if dataloader_state is not None:
        torch.save(dataloader_state, step_dir / "train_dataloader.pt")
    if config is not None:
        with open(step_dir / "config.yaml", "w") as f:
            yaml.safe_dump(config, f)
    return step_dir


def _setup_master_config(checkpoint_dir: str) -> MasterConfig:
    """Partially-populated MasterConfig for setup_single_controller tests.

    Same shape as test_single_controller_setup._make_master_config, plus the
    checkpointing block setup now reads.
    """
    return MasterConfig.model_construct(
        data_plane={
            "enabled": True,
            "impl": "transfer_queue",
            "backend": "simple",
            "checkpointing_enabled": True,
        },
        data={
            "use_multiple_dataloader": False,
            "shuffle": False,
            "num_workers": 0,
            "train": [{"env_name": "math"}],
        },
        grpo=GRPOConfig.model_construct(
            max_num_steps=100,
            max_num_epochs=1,
            num_prompts_per_step=4,
            num_generations_per_prompt=2,
            max_rollout_turns=1,
            seed=42,
            val_period=0,
            val_at_start=False,
            val_at_end=False,
        ),
        policy={
            "train_global_batch_size": 8,
            "max_total_sequence_length": 32,
            "tokenizer": {"use_fastokens": False},
            "megatron_cfg": {"enabled": False},
            "generation": {
                "backend": "vllm",
                "colocated": {"enabled": True, "resources": {}},
            },
        },
        loss_fn=ClippedPGLossConfig(),
        env={},
        async_rl=AsyncRLConfig(
            min_groups_for_streaming_train=4,
            max_buffered_rollouts=8,
        ),
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

        weights_path, optimizer_path = CheckpointManager.get_resume_paths(str(step_dir))

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
    def test_setup_forwards_latest_resume_paths(
        self,
        patched_factories,  # noqa: F811
        tmp_path,
    ):
        ckpt_dir = tmp_path / "ckpts"
        _write_checkpoint(
            ckpt_dir,
            1,
            {**_STEP_3_SAVE_STATE, "current_step": 1},
            dataloader_state={"fake_position": 1},
        )
        step_3 = _write_checkpoint(
            ckpt_dir, 3, _STEP_3_SAVE_STATE, dataloader_state={"fake_position": 3}
        )
        mc = _setup_master_config(str(ckpt_dir))

        actor_args, _ = setup_single_controller(mc, MagicMock(pad_token_id=0))

        # Latest checkpoint (step_3) wins; its paths reach the trainer factory.
        trainer_kwargs = patched_factories["_build_trainer"].call_args.kwargs
        assert trainer_kwargs["weights_path"] == step_3 / "policy" / "weights"
        assert trainer_kwargs["optimizer_path"] == step_3 / "policy" / "optimizer"
        # training_info.json is loaded into the actor args for the actor
        # (missing fields backfilled with the GRPOSaveState defaults).
        assert actor_args.save_state == _get_grpo_save_state(dict(_STEP_3_SAVE_STATE))
        assert actor_args.last_checkpoint_path == str(step_3)

    def test_setup_fresh_start_passes_none_paths(
        self,
        patched_factories,  # noqa: F811
        tmp_path,
    ):
        ckpt_dir = tmp_path / "empty_ckpts"
        ckpt_dir.mkdir()
        mc = _setup_master_config(str(ckpt_dir))

        actor_args, _ = setup_single_controller(mc, MagicMock(pad_token_id=0))

        trainer_kwargs = patched_factories["_build_trainer"].call_args.kwargs
        assert trainer_kwargs["weights_path"] is None
        assert trainer_kwargs["optimizer_path"] is None
        assert actor_args.save_state == _initial_grpo_save_state()
        assert actor_args.last_checkpoint_path is None

    def test_latest_restore_mode_selects_newer_rollout_snapshot(
        self,
        patched_factories,  # noqa: F811
        tmp_path: Path,
    ) -> None:
        ckpt_dir = tmp_path / "ckpts"
        step_3 = _write_checkpoint(
            ckpt_dir,
            3,
            _STEP_3_SAVE_STATE,
            dataloader_state={"fake_position": 3},
        )
        snapshot = step_3 / "rollout_snapshots" / "snapshot_000001"
        snapshot.mkdir(parents=True)
        (snapshot / ROLLOUT_SNAPSHOT_COMMITTED_FILENAME).write_text("committed\n")
        (snapshot / ROLLOUT_SNAPSHOT_MANIFEST_FILENAME).write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "base_train_step": 3,
                    "trainer_version": 3,
                    "current_epoch": 7,
                    "mutation_version": 1,
                    "rolled_back_train_group_count": 0,
                    "bootstrap_fingerprint": None,
                }
            )
        )
        torch.save({"fake_position": 7}, snapshot / "train_dataloader.pt")
        mc = _setup_master_config(str(ckpt_dir))

        actor_args, _ = setup_single_controller(mc, MagicMock(pad_token_id=0))

        assert actor_args.last_checkpoint_path == str(snapshot)
        assert actor_args.save_state.current_step == 3
        assert actor_args.save_state.current_epoch == 7
        trainer_kwargs = patched_factories["_build_trainer"].call_args.kwargs
        assert trainer_kwargs["weights_path"] == step_3 / "policy" / "weights"

    def test_trainer_restore_mode_ignores_periodic_snapshot(
        self,
        patched_factories,  # noqa: F811
        tmp_path: Path,
    ) -> None:
        ckpt_dir = tmp_path / "ckpts"
        step_3 = _write_checkpoint(
            ckpt_dir,
            3,
            _STEP_3_SAVE_STATE,
            dataloader_state={"fake_position": 3},
        )
        snapshot = step_3 / "rollout_snapshots" / "snapshot_000001"
        snapshot.mkdir(parents=True)
        (snapshot / ROLLOUT_SNAPSHOT_COMMITTED_FILENAME).write_text("committed\n")
        mc = _setup_master_config(str(ckpt_dir))
        mc.rollout_checkpointing.restore_mode = "trainer_checkpoint"

        actor_args, _ = setup_single_controller(mc, MagicMock(pad_token_id=0))

        assert actor_args.last_checkpoint_path == str(step_3)
        assert actor_args.save_state.current_epoch == 1

    def test_setup_forwards_pretrained_checkpoint(
        self,
        patched_factories,  # noqa: F811
        tmp_path,
    ):
        mc = _setup_master_config(str(tmp_path / "ckpts"))
        pretrained = {"path": "/some/ckpt", "format": "megatron_bridge"}
        mc.checkpointing["pretrained_checkpoint"] = pretrained

        setup_single_controller(mc, MagicMock(pad_token_id=0))

        assert mc.policy["pretrained_checkpoint"] == pretrained


def _make_int_dataloader() -> StatefulDataLoader:
    """8 ints, batch_size=2 → batches [0,1], [2,3], [4,5], [6,7]."""
    return StatefulDataLoader(
        list(range(8)),
        batch_size=2,
        shuffle=False,
        drop_last=True,
        num_workers=0,
    )


class TestDataloaderState:
    def test_save_writes_dataloader_state(self, tmp_path):
        mc = _actor_master_config(tmp_path, max_num_steps=2, save_period=2)
        save_state = _initial_grpo_save_state()
        save_state.current_epoch = 3
        dataloader = _FakeDataloader(state={"fake_position": 7})

        _run_train_pump(
            mc, _make_actor_args(save_state=save_state, dataloader=dataloader)
        )

        ckpt_dir = tmp_path / "checkpoints"
        dl_state_path = ckpt_dir / "step_2" / "train_dataloader.pt"
        assert dl_state_path.exists()
        # The snapshot taken at save time round-trips through torch.save.
        assert torch.load(dl_state_path) == {"fake_position": 7}
        # current_epoch flows save_state → actor → training_info.json.
        assert _training_info(ckpt_dir, 2)["current_epoch"] == 3

    def test_stateful_dataloader_position_roundtrip(self, tmp_path):
        data_config = {"train": [{"dataset_name": "math_train"}]}
        dataloader = _make_int_dataloader()
        it = iter(dataloader)
        assert [next(it).tolist() for _ in range(2)] == [[0, 1], [2, 3]]

        step_dir = _write_checkpoint(
            tmp_path,
            5,
            _initial_grpo_save_state(),
            dataloader_state=dataloader.state_dict(),
            config={"data": {"train": [{"dataset_name": "math_train"}]}},
        )

        restored = _make_int_dataloader()
        load_dataloader_state(restored, str(step_dir), data_config)

        # Resumes at batch k+1, not from the top.
        assert next(iter(restored)).tolist() == [4, 5]

    def test_dataset_swap_skips_restore(self, tmp_path):
        dataloader = _make_int_dataloader()
        it = iter(dataloader)
        assert [next(it).tolist() for _ in range(2)] == [[0, 1], [2, 3]]

        step_dir = _write_checkpoint(
            tmp_path,
            5,
            _initial_grpo_save_state(),
            dataloader_state=dataloader.state_dict(),
            config={"data": {"train": [{"dataset_name": "old_dataset"}]}},
        )

        restored = _make_int_dataloader()
        load_dataloader_state(
            restored, str(step_dir), {"train": [{"dataset_name": "new_dataset"}]}
        )

        # Restore skipped on dataset swap: the new dataset starts from index 0.
        assert next(iter(restored)).tolist() == [0, 1]

    def test_setup_restores_dataloader_state(
        self,
        patched_factories,  # noqa: F811
        tmp_path,
    ):
        ckpt_dir = tmp_path / "ckpts"
        sentinel = {"fake_position": 123}
        _write_checkpoint(ckpt_dir, 3, _STEP_3_SAVE_STATE, dataloader_state=sentinel)
        mc = _setup_master_config(str(ckpt_dir))

        setup_single_controller(mc, MagicMock(pad_token_id=0))

        fake_dataloader = patched_factories["dataloader"]
        fake_dataloader.load_state_dict.assert_called_once()
        assert fake_dataloader.load_state_dict.call_args.args[0] == sentinel

    def test_setup_missing_dataloader_state_raises(
        self,
        patched_factories,  # noqa: F811
        tmp_path,
    ):
        # Checkpoint with training_info.json + policy/ but no
        # train_dataloader.pt. SC always writes it on save, so a missing file
        # means a corrupted checkpoint — setup must raise, not silently start
        # from a fresh dataloader position (matching GRPO's contract).
        ckpt_dir = tmp_path / "ckpts"
        _write_checkpoint(ckpt_dir, 3, _STEP_3_SAVE_STATE)
        mc = _setup_master_config(str(ckpt_dir))

        with pytest.raises(FileNotFoundError):
            setup_single_controller(mc, MagicMock(pad_token_id=0))

        patched_factories["dataloader"].load_state_dict.assert_not_called()


# ── replay buffer persistence ────────────────────────────────────────────────


class TestReplayBufferPersistence:
    def test_run_completes_unfinished_recovery_with_zero_train_steps(self, tmp_path):
        ckpt_dir = tmp_path / "resume_ckpt"
        ckpt_dir.mkdir()
        torch.save({"groups": []}, ckpt_dir / REPLAY_BUFFER_METADATA_FILENAME)
        mc = _actor_master_config(
            tmp_path,
            max_num_steps=0,
            buffer_checkpoint=True,
            data_plane_checkpoint=True,
        )
        mc.token_capture.enabled = True
        ledger = RolloutRecoveryLedger()
        ledger.reserve_group(
            group_id="partial-group",
            prompt_id="17",
            prompt_ref=PromptRef(sample_id="17", task_name="nemo_gym"),
            prompt_payload={"idx": 17, "task_name": "nemo_gym"},
            expected_generations=2,
            target_step=None,
            start_weight_version=0,
        )
        manager = _FakeRolloutManager(ledger)
        metadata = {
            "replay_manifest_digest": "digest-1",
            "replay_group_count": 0,
            "rollout_recovery_payload_sha256": "recovery-digest",
            "rollout_recovery_group_count": 1,
        }

        actor, result = _run_actor_run(
            mc,
            _make_actor_args(
                rollout_manager=manager,
                tq_buffer=_FakeTQBuffer(load_return=0),
                last_checkpoint_path=str(ckpt_dir),
                data_plane_checkpoint_metadata=metadata,
            ),
        )

        assert manager.recovery_calls == ["partial-group"]
        assert len(ledger) == 0
        assert actor._buffer_capacity._value == 4
        assert result["train_steps"] == 0

    def test_restore_recovers_groups_in_parallel_with_bounded_concurrency(
        self, tmp_path
    ):
        class _BlockingRecoveryManager(_FakeRolloutManager):
            def __init__(self, ledger: RolloutRecoveryLedger) -> None:
                super().__init__(ledger)
                self.active = 0
                self.peak_active = 0
                self.two_started = asyncio.Event()
                self.release = asyncio.Event()

            async def recover_for_finalization(
                self, group_id: str, **kwargs: Any
            ) -> None:
                del kwargs
                self.active += 1
                self.peak_active = max(self.peak_active, self.active)
                if self.active == 2:
                    self.two_started.set()
                try:
                    await self.release.wait()
                    await super().recover_for_finalization(group_id)
                finally:
                    self.active -= 1

        async def _main() -> None:
            mc = _actor_master_config(
                tmp_path,
                max_num_steps=0,
                buffer_checkpoint=True,
                data_plane_checkpoint=True,
            )
            mc.token_capture.enabled = True
            mc.async_rl.max_inflight_prompts = 2
            ledger = _unfinished_recovery_ledger(3)
            manager = _BlockingRecoveryManager(ledger)
            actor = _ACTOR_CLS(
                mc,
                _make_actor_args(
                    rollout_manager=manager,
                    data_plane_checkpoint_metadata={
                        "rollout_recovery_payload_sha256": "digest"
                    },
                ),
                SetupTimingMetrics(),
            )

            recovery_task = asyncio.create_task(
                actor._maybe_restore_rollout_recovery(restored_replay_groups=0)
            )
            await asyncio.wait_for(manager.two_started.wait(), timeout=1.0)

            assert manager.peak_active == 2
            assert actor._rollout_slots._value == 0
            # All restored groups own capacity, including the one queued behind
            # the two-worker recovery limit.
            assert actor._buffer_capacity._value == 1

            manager.release.set()
            await asyncio.wait_for(recovery_task, timeout=1.0)
            actor._checkpointer.shutdown()

            assert manager.peak_active == 2
            assert sorted(manager.recovery_calls) == [
                "recovery-g0",
                "recovery-g1",
                "recovery-g2",
            ]
            assert actor._rollout_slots._value == 2
            assert actor._buffer_capacity._value == 4
            assert actor._rollout_recovery_complete.is_set()

        asyncio.run(_main())

    def test_parallel_recovery_failure_releases_uncommitted_capacity(self, tmp_path):
        class _FailingRecoveryManager(_FakeRolloutManager):
            async def recover_for_finalization(
                self, group_id: str, **kwargs: Any
            ) -> None:
                del kwargs
                await asyncio.sleep(0)
                if group_id == "recovery-g0":
                    raise RuntimeError("injected recovery failure")
                await asyncio.Event().wait()

        async def _main() -> None:
            mc = _actor_master_config(
                tmp_path,
                max_num_steps=0,
                buffer_checkpoint=True,
                data_plane_checkpoint=True,
            )
            mc.token_capture.enabled = True
            mc.async_rl.max_inflight_prompts = 2
            ledger = _unfinished_recovery_ledger(3)
            actor = _ACTOR_CLS(
                mc,
                _make_actor_args(
                    rollout_manager=_FailingRecoveryManager(ledger),
                    data_plane_checkpoint_metadata={
                        "rollout_recovery_payload_sha256": "digest"
                    },
                ),
                SetupTimingMetrics(),
            )

            with pytest.raises(ExceptionGroup) as exc_info:
                await asyncio.wait_for(
                    actor._maybe_restore_rollout_recovery(restored_replay_groups=0),
                    timeout=1.0,
                )
            actor._checkpointer.shutdown()

            assert exc_info.value.subgroup(RuntimeError) is not None
            assert actor._rollout_slots._value == 2
            assert actor._buffer_capacity._value == 4
            assert actor._rollout_recovery_complete.is_set()

        asyncio.run(_main())

    def test_run_overlaps_recovery_with_fresh_rollout_and_train_pumps(self, tmp_path):
        class _BlockingRecoveryManager(_FakeRolloutManager):
            def __init__(self, ledger: RolloutRecoveryLedger) -> None:
                super().__init__(ledger)
                self.started = asyncio.Event()
                self.release = asyncio.Event()

            async def recover_for_finalization(
                self, group_id: str, **kwargs: Any
            ) -> None:
                del kwargs
                self.started.set()
                await self.release.wait()
                await super().recover_for_finalization(group_id)

        async def _main() -> None:
            mc = _actor_master_config(
                tmp_path,
                max_num_steps=1,
                buffer_checkpoint=True,
                data_plane_checkpoint=True,
            )
            mc.token_capture.enabled = True
            mc.async_rl.max_inflight_prompts = 2
            ledger = _unfinished_recovery_ledger(1)
            manager = _BlockingRecoveryManager(ledger)
            actor = _ACTOR_CLS(
                mc,
                _make_actor_args(
                    rollout_manager=manager,
                    data_plane_checkpoint_metadata={
                        "rollout_recovery_payload_sha256": "digest"
                    },
                ),
                SetupTimingMetrics(),
            )

            rollout_started = asyncio.Event()
            train_started = asyncio.Event()
            finish_pumps = asyncio.Event()

            async def _no_sync() -> None:
                return None

            async def _no_replay() -> int:
                return 0

            async def _rollout_pump() -> None:
                acquired = 0
                rollout_slot_owned = False
                try:
                    for _ in range(2):
                        await actor._buffer_capacity.acquire()
                        acquired += 1
                    await actor._rollout_slots.acquire()
                    rollout_slot_owned = True
                    rollout_started.set()
                    await finish_pumps.wait()
                finally:
                    if rollout_slot_owned:
                        actor._rollout_slots.release()
                    for _ in range(acquired):
                        actor._buffer_capacity.release()

            async def _train_pump() -> None:
                train_started.set()
                await finish_pumps.wait()

            async def _watchdog_pump() -> None:
                await finish_pumps.wait()

            actor._sync_weights = _no_sync
            actor._maybe_restore_replay_buffer = _no_replay
            actor._rollout_pump = _rollout_pump
            actor._train_pump = _train_pump
            actor._watchdog_pump = _watchdog_pump

            run_task = asyncio.create_task(actor.run())
            await asyncio.wait_for(
                asyncio.gather(
                    manager.started.wait(),
                    rollout_started.wait(),
                    train_started.wait(),
                ),
                timeout=1.0,
            )

            # One restored permit and two fresh permits coexist while recovery
            # is still blocked, and both paths share the two generation slots.
            assert actor._buffer_capacity._value == 1
            assert actor._rollout_slots._value == 0
            assert not actor._rollout_recovery_complete.is_set()

            manager.release.set()
            finish_pumps.set()
            result = await asyncio.wait_for(run_task, timeout=1.0)

            assert result["train_steps"] == 0
            assert manager.recovery_calls == ["recovery-g0"]
            assert actor._buffer_capacity._value == 4
            assert actor._rollout_recovery_complete.is_set()

        asyncio.run(_main())

    def test_checkpoint_capable_sampler_without_native_tq_is_rejected(self, tmp_path):
        mc = _actor_master_config(
            tmp_path,
            max_num_steps=2,
            save_period=2,
            buffer_checkpoint=True,
        )

        with pytest.raises(
            ValueError,
            match="replay-checkpoint-capable sampler requires",
        ):
            _ACTOR_CLS(mc, _make_actor_args(), SetupTimingMetrics())

    def test_no_replay_buffer_with_gated_sampler(self, tmp_path):
        mc = _actor_master_config(
            tmp_path, max_num_steps=2, save_period=2, buffer_checkpoint=False
        )
        buffer = _FakeTQBuffer()

        _run_train_pump(mc, _make_actor_args(tq_buffer=buffer))

        ckpt_dir = tmp_path / "checkpoints"
        assert (ckpt_dir / "step_2" / "training_info.json").exists()
        assert not (ckpt_dir / "step_2" / "replay_buffer.pt").exists()
        assert not (ckpt_dir / "step_2" / REPLAY_BUFFER_METADATA_FILENAME).exists()

    def test_run_rejects_legacy_replay_file(self, tmp_path):
        ckpt_dir = tmp_path / "resume_ckpt"
        ckpt_dir.mkdir()
        torch.save({"groups": ["legacy"]}, ckpt_dir / LEGACY_REPLAY_BUFFER_FILENAME)
        mc = _actor_master_config(
            tmp_path,
            max_num_steps=0,
            buffer_checkpoint=True,
            data_plane_checkpoint=True,
        )
        buffer = _FakeTQBuffer()

        with pytest.raises(RuntimeError, match="legacy replay_buffer.pt"):
            _run_actor_run(
                mc,
                _make_actor_args(tq_buffer=buffer, last_checkpoint_path=str(ckpt_dir)),
            )

        assert buffer.load_calls == []

    def test_run_restores_native_tq_replay_metadata_without_payload_reput(
        self, tmp_path
    ):
        ckpt_dir = tmp_path / "resume_ckpt"
        ckpt_dir.mkdir()
        sample_ids = ["g0_g0", "g0_g1", "g1_g0", "g1_g1"]
        groups = [
            {
                "meta": KVBatchMeta(
                    partition_id=_PARTITION_ID,
                    task_name=None,
                    sample_ids=[f"g{i}_g0", f"g{i}_g1"],
                    sequence_lengths=[16, 16],
                    tags=[{"weight_version": 0}, {"weight_version": 0}],
                ),
                "start_weight": 0,
                "end_weight": 0,
                "target_step": None,
                "group_id": f"g{i}",
            }
            for i in range(2)
        ]
        envelope = {"groups": groups}
        torch.save(envelope, ckpt_dir / REPLAY_BUFFER_METADATA_FILENAME)
        tq_metadata = {
            "replay_manifest_digest": "digest-1",
            "replay_group_count": 2,
        }
        mc = _actor_master_config(
            tmp_path,
            max_num_steps=0,
            buffer_checkpoint=True,
            data_plane_checkpoint=True,
        )
        buffer = _FakeTQBuffer(load_return=2)

        actor, result = _run_actor_run(
            mc,
            _make_actor_args(
                tq_buffer=buffer,
                dp_client=_FakeDPClient(sample_ids=sample_ids),
                last_checkpoint_path=str(ckpt_dir),
                data_plane_checkpoint_metadata=tq_metadata,
            ),
        )

        assert buffer.load_calls == [
            {
                "state": envelope,
                "max_groups": 4,
                "expected_partition_id": _PARTITION_ID,
                "expected_group_size": 2,
                "expected_manifest_digest": "digest-1",
            }
        ]
        assert actor._buffer_capacity._value == 2
        assert result["train_steps"] == 0
        # run()'s finally must tear the synchronizer down exactly once.
        assert actor._weight_synchronizer.shutdown_count == 1

    def test_restored_permits_are_released_by_a_live_pump(self, tmp_path):
        # The restore takes one capacity permit per group; a running pump must
        # give them all back. Every other restore test uses max_num_steps=0,
        # so the pump body never runs and a regression that leaks restored
        # permits (starving the rollout pump) would go unnoticed.
        # K == max_buffered_rollouts here, which also covers the full-capacity
        # acquisition shape.
        ckpt_dir = tmp_path / "resume_ckpt"
        ckpt_dir.mkdir()
        sample_ids = [f"g{i}_g{j}" for i in range(4) for j in range(2)]
        groups = [
            {
                "meta": KVBatchMeta(
                    partition_id=_PARTITION_ID,
                    task_name=None,
                    sample_ids=[f"g{i}_g0", f"g{i}_g1"],
                    sequence_lengths=[16, 16],
                    tags=[{"weight_version": 0}, {"weight_version": 0}],
                ),
                "start_weight": 0,
                "end_weight": 0,
                "target_step": None,
                "group_id": f"g{i}",
            }
            for i in range(4)
        ]
        torch.save({"groups": groups}, ckpt_dir / REPLAY_BUFFER_METADATA_FILENAME)
        tq_metadata = {
            "replay_manifest_digest": "digest-1",
            "replay_group_count": 4,
        }
        mc = _actor_master_config(
            tmp_path,
            max_num_steps=2,
            save_period=2,
            buffer_checkpoint=True,
            data_plane_checkpoint=True,
        )
        buffer = _FakeTQBuffer(load_return=4)

        actor = _run_restore_then_train_pump(
            mc,
            _make_actor_args(
                tq_buffer=buffer,
                dp_client=_FakeDPClient(sample_ids=sample_ids),
                last_checkpoint_path=str(ckpt_dir),
                data_plane_checkpoint_metadata=tq_metadata,
            ),
            restored_groups=groups,
        )

        assert len(buffer.load_calls) == 1
        assert actor._train_steps == 2
        # Restore drained the semaphore to 0; the pump released one permit per
        # selected group (2 steps x 2 prompt groups), so all 4 came back.
        assert actor._buffer_capacity._value == 4

    @pytest.mark.parametrize(
        ("actual_sample_ids", "error_fragment"),
        [
            (["g0_g0"], r"missing=\['g0_g1'\]"),
            (
                ["g0_g0", "g0_g1", "orphan_g0"],
                r"unexpected=\['orphan_g0'\]",
            ),
        ],
    )
    def test_native_restore_rejects_tq_inventory_mismatch(
        self, tmp_path, actual_sample_ids, error_fragment
    ):
        ckpt_dir = tmp_path / "resume_ckpt"
        ckpt_dir.mkdir()
        envelope = {
            "groups": [
                {
                    "meta": KVBatchMeta(
                        partition_id=_PARTITION_ID,
                        task_name=None,
                        sample_ids=["g0_g0", "g0_g1"],
                        sequence_lengths=[16, 16],
                        tags=[{"weight_version": 0}, {"weight_version": 0}],
                    ),
                    "start_weight": 0,
                    "end_weight": 0,
                    "target_step": None,
                    "group_id": "g0",
                }
            ]
        }
        torch.save(envelope, ckpt_dir / REPLAY_BUFFER_METADATA_FILENAME)
        tq_metadata = {
            "replay_manifest_digest": "digest-1",
            "replay_group_count": 1,
        }
        mc = _actor_master_config(
            tmp_path,
            max_num_steps=0,
            buffer_checkpoint=True,
            data_plane_checkpoint=True,
        )

        with pytest.raises(RuntimeError, match=error_fragment):
            _run_actor_run(
                mc,
                _make_actor_args(
                    tq_buffer=_FakeTQBuffer(load_return=1),
                    dp_client=_FakeDPClient(sample_ids=actual_sample_ids),
                    last_checkpoint_path=str(ckpt_dir),
                    data_plane_checkpoint_metadata=tq_metadata,
                ),
            )

    def test_native_replay_metadata_requires_setup_side_tq_restore(self, tmp_path):
        ckpt_dir = tmp_path / "resume_ckpt"
        ckpt_dir.mkdir()
        torch.save({"groups": []}, ckpt_dir / REPLAY_BUFFER_METADATA_FILENAME)
        mc = _actor_master_config(
            tmp_path,
            max_num_steps=0,
            buffer_checkpoint=True,
            data_plane_checkpoint=True,
        )

        with pytest.raises(RuntimeError, match="native TQ checkpoint was not restored"):
            _run_actor_run(
                mc,
                _make_actor_args(last_checkpoint_path=str(ckpt_dir)),
            )

    def test_run_missing_native_replay_metadata_starts_empty(self, tmp_path):
        ckpt_dir = tmp_path / "resume_ckpt"
        ckpt_dir.mkdir()
        mc = _actor_master_config(
            tmp_path,
            max_num_steps=0,
            buffer_checkpoint=True,
            data_plane_checkpoint=True,
        )
        buffer = _FakeTQBuffer()

        actor, _ = _run_actor_run(
            mc,
            _make_actor_args(tq_buffer=buffer, last_checkpoint_path=str(ckpt_dir)),
        )

        assert buffer.load_calls == []
        assert actor._buffer_capacity._value == 4  # zero permits consumed

    def test_run_rejects_native_replay_state_with_gated_sampler(self, tmp_path):
        ckpt_dir = tmp_path / "resume_ckpt"
        ckpt_dir.mkdir()
        torch.save({"groups": []}, ckpt_dir / REPLAY_BUFFER_METADATA_FILENAME)
        mc = _actor_master_config(tmp_path, max_num_steps=0, buffer_checkpoint=False)

        with pytest.raises(RuntimeError, match="does not support replay-buffer"):
            _run_actor_run(
                mc,
                _make_actor_args(last_checkpoint_path=str(ckpt_dir)),
            )
