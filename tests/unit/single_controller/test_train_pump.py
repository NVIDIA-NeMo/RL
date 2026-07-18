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

"""End-to-end tests for SC._train_pump."""

from __future__ import annotations

import gc
from types import SimpleNamespace

import pytest
import ray
import torch
from tensordict import TensorDict

from nemo_rl.algorithms.async_utils.replay_buffer import TQReplayBuffer
from nemo_rl.algorithms.single_controller import (
    SingleControllerActor,
    SingleControllerConfig,
)
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.data_plane import KVBatchMeta
from nemo_rl.distributed.virtual_cluster import RayVirtualCluster
from nemo_rl.models.generation import configure_generation_config
from nemo_rl.models.policy.tq_policy import TQPolicy
from tests.unit.models.policy.test_megatron_worker import create_megatron_test_config
from tests.unit.single_controller._dp_fakes import _PARTITION_ID
from tests.unit.test_utils import SimpleLossFn

# Union of DP_TRAIN_FIELDS (TQPolicy) + rollout extras (total_reward,
# prompt_ids_for_adv) — the partition schema must cover every field any
# producer/consumer touches.
_REGISTERED_FIELDS = [
    "input_ids",
    "input_lengths",
    "generation_logprobs",
    "prev_logprobs",
    "reference_policy_logprobs",
    "advantages",
    "token_mask",
    "sample_mask",
    "total_reward",
    "prompt_ids_for_adv",
]


def _simple_tq_cfg() -> dict:
    """Simple in-process TQ backend cfg (mirrors tests/unit/data_plane/conftest.py)."""
    return {
        "enabled": True,
        "impl": "transfer_queue",
        "backend": "simple",
        "storage_capacity": 1024,
        "num_storage_units": 1,
        "claim_meta_poll_interval_s": 0.5,
        "global_segment_size": 8589934592,  # 8 GiB
        "local_buffer_size": 1073741824,  # 1 GiB
    }


def _populate_group(
    dp_client,
    *,
    group_uuid: str,
    group_size: int,
    seq_len: int,
    weight_version: int,
) -> KVBatchMeta:
    """Write one complete prompt group to DP and return its meta."""
    sample_ids = [f"{group_uuid}_g{i}" for i in range(group_size)]
    fields = TensorDict(
        {
            "input_ids": torch.randint(0, 1000, (group_size, seq_len)).long(),
            "input_lengths": torch.tensor([seq_len] * group_size).long(),
            "token_mask": torch.ones(group_size, seq_len, dtype=torch.long),
            "sample_mask": torch.ones(group_size, dtype=torch.long),
            "generation_logprobs": torch.zeros(
                group_size, seq_len, dtype=torch.float32
            ),
            # prev_logprobs / reference_policy_logprobs are read by
            # TQPolicy workers as part of DP_TRAIN_FIELDS; pre-seed with
            # zeros since the loss (SimpleLossFn) ignores them.
            "prev_logprobs": torch.zeros(group_size, seq_len, dtype=torch.float32),
            "reference_policy_logprobs": torch.zeros(
                group_size, seq_len, dtype=torch.float32
            ),
            "total_reward": torch.arange(group_size, dtype=torch.float32) * 0.5 + 1.0,
            "prompt_ids_for_adv": torch.zeros(group_size, seq_len, dtype=torch.long),
        },
        batch_size=(group_size,),
    )
    tags = [{"weight_version": int(weight_version)} for _ in range(group_size)]
    dp_client.put_samples(
        sample_ids=sample_ids,
        partition_id=_PARTITION_ID,
        fields=fields,
        tags=tags,
    )
    return KVBatchMeta(
        partition_id=_PARTITION_ID,
        task_name="train",
        sample_ids=sample_ids,
        fields=list(fields.keys()),
        sequence_lengths=[seq_len] * group_size,
        tags=[dict(t) for t in tags],
    )


def _prepopulate_buffer(
    buffer: TQReplayBuffer, meta: KVBatchMeta, *, weight_version: int
) -> None:
    """Insert a ready slot into TQReplayBuffer, mirroring what commit() sets."""
    buffer.meta_list.append(meta)
    buffer.start_weight_list.append(int(weight_version))
    buffer.end_weight_list.append(int(weight_version))
    buffer.target_step_list.append(None)
    buffer.ready_list.append(True)
    # Group id follows pack_payload's "{group_uuid}_g{i}" convention.
    group_id = meta.sample_ids[0].rpartition("_g")[0]
    buffer._group_ids.append(group_id)


@pytest.fixture(scope="function")
def train_cluster():
    """Single-GPU virtual cluster for the trainer worker group."""
    cluster = RayVirtualCluster(
        name="test-sc-train-cluster",
        bundle_ct_per_node_list=[1],
        use_gpus=True,
        num_gpus_per_node=1,
        max_colocated_worker_groups=1,
    )
    try:
        yield cluster
    finally:
        cluster.shutdown()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class _FakeAdvEstimator:
    """Passthrough advantage: returns rewards unchanged."""

    def compute_advantage(
        self,
        prompt_ids: torch.Tensor,
        rewards: torch.Tensor,
        mask: torch.Tensor,
        repeated_batch: dict,
        **kwargs,
    ) -> torch.Tensor:
        return rewards.detach().clone()


@ray.remote(num_cpus=0)  # pragma: no cover
class _CallLog:
    """Ordered append-only log the fakes below write to."""

    def __init__(self) -> None:
        self._entries: list[tuple[str, dict]] = []

    def record(self, kind: str, payload: dict) -> int:
        self._entries.append((kind, dict(payload)))
        return len(self._entries)

    def get(self) -> list[tuple[str, dict]]:
        return list(self._entries)


class _FakeWeightSync:
    """Records the version SC syncs at end-of-step, via a Ray actor log."""

    def __init__(self, log_handle) -> None:
        self._log = log_handle

    async def sync_weights(self, version: int) -> None:
        ray.get(self._log.record.remote("sync_weights", {"version": int(version)}))


@pytest.mark.mcore
@pytest.mark.hf_gated
def test_train_pump_drives_mcore_training_step(
    train_cluster,
    tiny_llama_model_path,
    tmp_path,
):
    """SC._train_pump runs one outer step against a real Megatron TQPolicy."""
    num_generations = 2
    num_prompts = 2
    seq_len = 16
    train_gbs = num_prompts * num_generations

    policy_cfg = create_megatron_test_config(tiny_llama_model_path, tp=1, pp=1)
    policy_cfg["train_global_batch_size"] = train_gbs
    policy_cfg["train_micro_batch_size"] = num_generations
    tokenizer = get_tokenizer(policy_cfg["tokenizer"])
    policy_cfg["generation"] = configure_generation_config(
        policy_cfg["generation"], tokenizer
    )

    dp_cfg = _simple_tq_cfg()
    # TQPolicy's ctor bootstraps the TQ controller (bootstrap=True) and
    # fans out setup_data_plane to workers (bootstrap=False on workers).
    trainer = TQPolicy(
        cluster=train_cluster,
        config=policy_cfg,
        tokenizer=tokenizer,
        dp_cfg=dp_cfg,
        tq_partition_id=_PARTITION_ID,
    )

    try:
        # Reuse TQPolicy's driver-side dp_client for SC — same controller,
        # no double-bootstrap.
        dp_client = trainer.dp_client

        # Register the partition once with the full field union.
        dp_client.register_partition(
            partition_id=_PARTITION_ID,
            fields=_REGISTERED_FIELDS,
            num_samples=train_gbs * 4,
            consumer_tasks=["prev_lp", "ref_lp", "train"],
        )

        # Real TQReplayBuffer, pre-populated with 2 ready groups so
        # StalenessSampler.select fires immediately.
        tq_buffer = TQReplayBuffer(
            dp_client,
            partition_id=_PARTITION_ID,
            pad_value_dict={"input_ids": int(tokenizer.pad_token_id or 0)},
        )
        for g in range(num_prompts):
            meta = _populate_group(
                dp_client,
                group_uuid=f"prompt{g}",
                group_size=num_generations,
                seq_len=seq_len,
                weight_version=0,
            )
            _prepopulate_buffer(tq_buffer, meta, weight_version=0)

        log = _CallLog.remote()
        weight_sync = _FakeWeightSync(log)
        adv_est = _FakeAdvEstimator()
        # Rollout manager stub — SC.__init__ only touches ._tq_buffer.
        rollout_manager = SimpleNamespace(_tq_buffer=None)

        cfg = SingleControllerConfig.model_construct(
            max_weight_staleness_versions=1,
            min_groups_per_batch=num_prompts,
            target_groups_per_step=num_prompts,
            group_size=num_generations,
            batch_selection_strategy="staleness_window",
            max_inflight_prompts=num_prompts,
            max_buffered_rollouts=num_prompts,
            max_train_steps=1,
            max_num_epochs=None,
            over_sampling=True,
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
            dp_client_handle=dp_client,
            gen_handle=None,
            trainer_handle=trainer,
            weight_synchronizer=weight_sync,
            loss_fn=SimpleLossFn(),
            advantage_estimator=adv_est,
            rollout_manager=rollout_manager,
            dataloader=None,
            tq_buffer=tq_buffer,
        )

        # One outer step: sampler.select → advantage stage → begin/microbatches/finish → sync.
        ray.get(ctrl._train_pump.remote())

        state = ray.get(ctrl.ping.remote())
        assert state["train_steps"] == 1
        assert state["trainer_version"] == 1

        entries = ray.get(log.get.remote())
        sync_versions = [p["version"] for k, p in entries if k == "sync_weights"]
        assert sync_versions == [1]

    finally:
        trainer.shutdown()
