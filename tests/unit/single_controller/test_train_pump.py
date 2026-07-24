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
import math
from types import SimpleNamespace
from typing import Any

import pytest
import ray
import torch
from tensordict import TensorDict

from nemo_rl.algorithms.async_utils.replay_buffer import TQReplayBuffer
from nemo_rl.algorithms.async_utils.staleness_sampler import WindowedSamplerConfig
from nemo_rl.algorithms.single_controller import SingleControllerActor
from nemo_rl.algorithms.single_controller_utils.config import (
    AsyncRLConfig,
    MasterConfig,
)
from nemo_rl.algorithms.single_controller_utils.setup import SingleControllerActorArgs
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
    """Per-token advantage: broadcasts per-sample reward across the token dim."""

    def compute_advantage(
        self,
        prompt_ids: torch.Tensor,
        rewards: torch.Tensor,
        mask: torch.Tensor,
        repeated_batch: dict,
        **kwargs,
    ) -> torch.Tensor:
        return rewards.detach().unsqueeze(-1).expand_as(mask).clone()


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


class _RecordingLogger:
    """Forward metrics logged inside SC to a Ray actor visible to the test."""

    def __init__(self, log_handle: Any) -> None:
        self._log = log_handle

    def log_metrics(
        self,
        metrics: dict[str, Any],
        step: int,
        prefix: str | None = "",
    ) -> None:
        ray.get(
            self._log.record.remote(
                "metrics",
                {
                    "metrics": dict(metrics),
                    "step": int(step),
                    "prefix": prefix,
                },
            )
        )


@ray.remote(num_cpus=1, num_gpus=0)
class _RecordingSingleControllerActor(
    SingleControllerActor.__ray_metadata__.modified_class
):
    """SingleControllerActor variant with an observable logger."""

    def __init__(self, *, metric_log_handle: Any, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._logger = _RecordingLogger(metric_log_handle)


@pytest.mark.mcore
@pytest.mark.hf_gated
def test_train_pump_drives_mcore_training_step(
    train_cluster,
    tiny_llama_model_path,
    tmp_path,
):
    """SC._train_pump runs one outer step against a real Megatron TQPolicy."""
    train_steps = 2
    train_gbs = 4
    num_generations = 2
    num_prompts = 2

    policy_cfg = create_megatron_test_config(tiny_llama_model_path, tp=1, pp=1)
    policy_cfg["train_global_batch_size"] = train_gbs
    policy_cfg["train_micro_batch_size"] = train_gbs
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

        # Register the partition once with the full field union. All
        # pre-populated samples across every step live here at once.
        dp_client.register_partition(
            partition_id=_PARTITION_ID,
            fields=_REGISTERED_FIELDS,
            num_samples=train_steps * train_gbs,
            consumer_tasks=["prev_lp", "ref_lp", "train"],
        )

        # Real TQReplayBuffer, pre-populated with one ready batch (num_prompts
        # groups of num_generations samples) per step, stamped with the
        # step's weight_version so the sampler can advance across steps.
        tq_buffer = TQReplayBuffer(
            dp_client,
            partition_id=_PARTITION_ID,
            pad_value_dict={"input_ids": int(tokenizer.pad_token_id or 0)},
        )
        for step in range(train_steps):
            for g in range(num_prompts):
                meta = _populate_group(
                    dp_client,
                    group_uuid=f"step{step}_prompt{g}",
                    group_size=num_generations,
                    seq_len=16,
                    weight_version=step,
                )
                _prepopulate_buffer(tq_buffer, meta, weight_version=step)

        log = _CallLog.remote()
        weight_sync = SimpleNamespace(
            sync_weights=lambda *, kv_scales=None: None,
        )
        adv_est = _FakeAdvEstimator()
        # Rollout manager stub — SC.__init__ only touches ._tq_buffer.
        rollout_manager = SimpleNamespace(
            _tq_buffer=None,
            set_weight_version=lambda v: ray.get(
                log.record.remote("set_weight_version", {"version": int(v)})
            ),
        )

        master_config = MasterConfig.model_construct(
            policy={"train_global_batch_size": train_gbs},
            grpo={
                "num_prompts_per_step": num_prompts,
                "num_generations_per_prompt": num_generations,
                "max_num_steps": train_steps,
                "max_num_epochs": None,
            },
            loss_fn=SimpleNamespace(force_on_policy_ratio=False),
            async_rl=AsyncRLConfig(
                sampler=WindowedSamplerConfig(max_staleness_versions=1),
                min_groups_for_streaming_train=num_prompts,
                max_inflight_prompts=num_prompts,
                max_buffered_rollouts=num_prompts,
            ),
            logger={
                "log_dir": str(tmp_path / "logs"),
                "wandb_enabled": False,
                "swanlab_enabled": False,
                "tensorboard_enabled": False,
                "mlflow_enabled": False,
                "monitor_gpus": False,
            },
        )

        actor_args = SingleControllerActorArgs(
            gen_handle=None,
            trainer_handle=trainer,
            env_handles={},
            train_cluster=None,  # type: ignore[arg-type]
            inference_cluster=None,  # type: ignore[arg-type]
            dp_client=dp_client,
            dataloader=None,  # type: ignore[arg-type]  # _rollout_pump not started
            weight_synchronizer=weight_sync,  # type: ignore[arg-type]
            advantage_estimator=adv_est,
            loss_fn=SimpleLossFn(),
            rollout_manager=rollout_manager,
            tq_buffer=tq_buffer,
            partition_id=_PARTITION_ID,
        )
        ctrl = _RecordingSingleControllerActor.remote(
            metric_log_handle=log,
            master_config=master_config,
            actor_args=actor_args,
        )

        # train_steps outer steps, each: sampler.select → advantage stage → begin/microbatches/finish → sync.
        ray.get(ctrl._train_pump.remote())

        state = ray.get(ctrl.ping.remote())
        assert state["train_steps"] == train_steps
        assert state["trainer_version"] == train_steps

        entries = ray.get(log.get.remote())
        sync_versions = [p["version"] for k, p in entries if k == "set_weight_version"]
        assert sync_versions == list(range(1, train_steps + 1))

        train_metrics = [
            p["metrics"]
            for kind, p in entries
            if kind == "metrics" and p["prefix"] == "train"
        ]
        assert len(train_metrics) == train_steps
        for metrics in train_metrics:
            assert math.isfinite(metrics["reward"])
            assert math.isfinite(metrics["advantages/mean"])

    finally:
        trainer.shutdown()
