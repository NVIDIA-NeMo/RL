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

"""Dry-run test: SC._train_pump drives the split-API cycle per mini-batch.

TODO: once the SingleController setup entrypoint lands (real TQPolicy,
WeightSynchronizer, and advantage_estimator wiring), replace the fakes
below with the real components and turn this into an integration test.
Today the trainer / weight_synchronizer / advantage_estimator are all
stubs — the test exercises the pump's control flow (claim → adv-stage →
split-API cycle → clear → sync) but not the collaborators' semantics.
"""

from __future__ import annotations

import pytest
import ray
import torch
from tensordict import TensorDict

from nemo_rl.algorithms.single_controller import (
    SingleControllerActor,
    SingleControllerConfig,
)
from nemo_rl.data_plane.adapters.noop import NoOpDataPlaneClient

_PARTITION_ID = "rollout_data"
_ROLLOUT_FIELDS = [
    "input_ids",
    "input_lengths",
    "token_mask",
    "sample_mask",
    "total_reward",
    "prompt_ids_for_adv",
]
_ADV_FIELD = "advantages"


@ray.remote(num_cpus=0)
class _DPActor(NoOpDataPlaneClient):
    """Ray-wrapped NoOpDataPlaneClient for cross-process DP inspection."""

    def claim_meta(self, *args, **kwargs):
        meta = super().claim_meta(*args, **kwargs)
        rec = self._partitions[meta.partition_id]
        meta.tags = [dict(rec.tags.get(sid, {})) for sid in meta.sample_ids]
        return meta

    def peek_count(self, partition_id: str) -> int:
        return len(self._partitions[partition_id].rows)


@ray.remote(num_cpus=0)
class _CallLog:
    """Ordered append-only log the fakes below write to."""

    def __init__(self) -> None:
        self._entries: list[tuple[str, dict]] = []

    def record(self, kind: str, payload: dict) -> int:
        self._entries.append((kind, dict(payload)))
        return len(self._entries)

    def get(self) -> list[tuple[str, dict]]:
        return list(self._entries)


class _FakeTrainer:
    """Sync driver-side TQPolicy stand-in."""

    def __init__(self, log_handle) -> None:
        self._log = log_handle

    def begin_train_step(self, loss_fn) -> None:
        ray.get(self._log.record.remote("begin_train_step", {}))

    def train_microbatches_from_meta(self, meta) -> None:
        ray.get(
            self._log.record.remote(
                "train_microbatches_from_meta", {"meta_size": int(meta.size)}
            )
        )

    def finish_train_step(self) -> dict:
        ray.get(self._log.record.remote("finish_train_step", {}))
        return {"loss": 0.0}


class _FakeWeightSync:
    """Async WeightSynchronizer stand-in."""

    def __init__(self, log_handle) -> None:
        self._log = log_handle

    async def sync_weights(self, version: int) -> None:
        ray.get(self._log.record.remote("sync_weights", {"version": int(version)}))


class _FakeAdvEstimator:
    """Records shapes SC hands the estimator; returns a per-sample scalar."""

    def __init__(self, log_handle) -> None:
        self._log = log_handle

    def compute_advantage(
        self,
        prompt_ids: torch.Tensor,
        rewards: torch.Tensor,
        mask: torch.Tensor,
        repeated_batch: dict,
        **kwargs,
    ) -> torch.Tensor:
        ray.get(
            self._log.record.remote(
                "compute_advantage",
                {
                    "rewards_shape": tuple(rewards.shape),
                    "mask_shape": tuple(mask.shape),
                    "prompt_ids_shape": tuple(prompt_ids.shape),
                    "rb_keys": sorted(repeated_batch.keys()),
                    "kwargs_keys": sorted(kwargs.keys()),
                },
            )
        )
        return rewards.detach().clone()


def _populate_group(
    dp,
    group_uuid: str,
    group_size: int,
    seq_len: int,
    weight_version: int,
) -> list[str]:
    """Write one complete prompt group to DP."""
    sample_ids = [f"{group_uuid}_g{i}" for i in range(group_size)]
    fields = TensorDict(
        {
            "input_ids": torch.arange(group_size * seq_len)
            .reshape(group_size, seq_len)
            .long(),
            "input_lengths": torch.tensor([seq_len] * group_size).long(),
            "token_mask": torch.ones(group_size, seq_len, dtype=torch.long),
            "sample_mask": torch.ones(group_size, 1, dtype=torch.long),
            "total_reward": (
                torch.arange(group_size, dtype=torch.float32) * 0.5 + 1.0
            ).unsqueeze(-1),
            "prompt_ids_for_adv": torch.zeros(group_size, seq_len, dtype=torch.long),
        },
        batch_size=(group_size,),
    )
    tags = [
        {
            "weight_version": int(weight_version),
            "expected_num_samples": int(group_size),
            "committed": True,
            "input_lengths": int(seq_len),
        }
        for _ in range(group_size)
    ]
    ray.get(
        dp.put_samples.remote(
            sample_ids=sample_ids,
            partition_id=_PARTITION_ID,
            fields=fields,
            tags=tags,
        )
    )
    return sample_ids


@pytest.fixture(scope="module")
def ray_cluster():
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, num_cpus=4, log_to_driver=False)
    yield
    ray.shutdown()


def test_train_pump_dry_run(ray_cluster, tmp_path):
    """SC._train_pump runs 2 mini-batches × 1 group each, then a single _sync_weights."""
    del ray_cluster
    group_size = 2
    seq_len = 4
    num_groups = 2  # target_groups_per_step
    # train_gbs == group_size → groups_per_minibatch=1, num_minibatches=2.
    train_gbs = group_size

    dp = _DPActor.remote()
    log = _CallLog.remote()
    ray.get(
        dp.register_partition.remote(
            partition_id=_PARTITION_ID,
            fields=[*_ROLLOUT_FIELDS, _ADV_FIELD],
            num_samples=group_size * num_groups * 4,
            consumer_tasks=["train"],
        )
    )
    for g in range(num_groups):
        _populate_group(
            dp,
            group_uuid=f"prompt{g}",
            group_size=group_size,
            seq_len=seq_len,
            weight_version=0,
        )

    trainer = _FakeTrainer(log)
    weight_sync = _FakeWeightSync(log)
    adv_est = _FakeAdvEstimator(log)

    cfg = SingleControllerConfig.model_construct(
        max_weight_staleness_versions=1,
        min_groups_per_batch=num_groups,
        target_groups_per_step=num_groups,
        group_size=group_size,
        batch_selection_strategy="staleness_window",
        max_inflight_prompts=8,
        max_buffered_rollouts=8,
        max_train_steps=1,
        max_rollout_prompts=num_groups,
        train_global_batch_size=train_gbs,
        partition_id=_PARTITION_ID,
        advantage_enabled=True,
        advantage_repeated_batch_fields=[],
        advantage_policy_logprobs_field=None,
        advantage_reference_logprobs_field=None,
        diagnostics=False,
        logger={
            "log_dir": str(tmp_path / "logs"),
            "wandb_enabled": False,
            "swanlab_enabled": False,
            "tensorboard_enabled": False,
            "mlflow_enabled": False,
            "monitor_gpus": False,
        },
    )

    ctrl = SingleControllerActor.remote(
        cfg=cfg,
        prompts=[],  # rollout pump is not started
        dp_client_handle=dp,
        gen_handle=None,
        trainer_handle=trainer,
        weight_synchronizer=weight_sync,
        loss_fn=object(),
        advantage_estimator=adv_est,
    )

    # _train_pump exits when max_train_steps is reached — no cancel needed.
    ray.get(ctrl._train_pump.remote())

    state = ray.get(ctrl.ping.remote())
    assert state["train_steps"] == 1
    assert state["trainer_version"] == 2  # one bump per mini-batch

    entries = ray.get(log.get.remote())
    kinds = [k for k, _ in entries]
    assert kinds == [
        "compute_advantage",
        "begin_train_step",
        "train_microbatches_from_meta",
        "finish_train_step",
        "compute_advantage",
        "begin_train_step",
        "train_microbatches_from_meta",
        "finish_train_step",
        "sync_weights",
    ]

    # sync_weights fires with the post-bump trainer_version.
    sync_payload = next(p for k, p in entries if k == "sync_weights")
    assert sync_payload["version"] == 2

    for k, p in entries:
        if k == "train_microbatches_from_meta":
            assert p["meta_size"] == group_size
        elif k == "compute_advantage":
            assert p["rewards_shape"] == (group_size,)
            assert p["mask_shape"] == (group_size, seq_len)
            assert p["prompt_ids_shape"] == (group_size, seq_len)
            assert p["rb_keys"] == ["total_reward"]
            assert p["kwargs_keys"] == []

    # clear_samples was called after each finish_train_step.
    assert ray.get(dp.peek_count.remote(_PARTITION_ID)) == 0
