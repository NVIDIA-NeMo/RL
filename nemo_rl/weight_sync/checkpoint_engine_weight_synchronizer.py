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

from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, Optional

import ray

from nemo_rl.models.generation.interfaces import CheckpointEngineConfig
from nemo_rl.utils.timer import Timer
from nemo_rl.weight_sync.interfaces import WeightSynchronizer


def _flatten_metadata(results: list[Any]) -> list[Any]:
    return [
        item
        for result in results
        for item in (result if isinstance(result, list) else [result])
    ]


def _sort_ranked_metadata(metadata: list[Any]) -> list[Any]:
    if all(isinstance(item, dict) and "rank" in item for item in metadata):
        return sorted(metadata, key=lambda item: item["rank"])
    return metadata


def _ordered_generation_metadata(generation_results: list[Any]) -> list[Any]:
    """Order vLLM generation metadata by global rollout rank.

    Each result belongs to one vLLM data-parallel group. Engine-local ranks
    are unique only within a group, so sort each group before concatenating
    them in worker-group order.
    """
    metadata: list[Any] = []
    for group_result in generation_results:
        group_metadata = (
            group_result if isinstance(group_result, list) else [group_result]
        )
        metadata.extend(_sort_ranked_metadata(group_metadata))
    return metadata


@dataclass
class CheckpointEngineWeightSynchronizer(WeightSynchronizer):
    _policy: Any
    _generation: Any
    _checkpoint_engine_config: CheckpointEngineConfig
    _stale: bool = True
    _checkpoint_engine_ready: bool = False

    def init_communicator(self) -> None:
        self._generation.prepare_refit_info(self._policy.prepare_refit_info())
        self._ensure_checkpoint_engine_ready()

    @property
    def is_stale(self) -> bool:
        return self._stale

    def mark_stale(self) -> None:
        self._stale = True

    def _run_policy(
        self, checkpoint_method: str, **method_kwargs: Any
    ) -> list[ray.ObjectRef]:
        return self._policy.worker_group.run_all_workers_single_data(
            "checkpoint_engine_rpc",
            checkpoint_method=checkpoint_method,
            method_kwargs=method_kwargs,
        )

    def _generation_rpc(self) -> str:
        return (
            "checkpoint_engine_rpc_async"
            if self._generation.cfg["vllm_cfg"]["async_engine"]
            else "checkpoint_engine_rpc"
        )

    def _run_generation(
        self, checkpoint_method: str, method_args: tuple[Any, ...] = ()
    ) -> list[ray.ObjectRef]:
        return self._generation.worker_group.run_all_workers_single_data(
            self._generation_rpc(),
            checkpoint_method=checkpoint_method,
            method_args=method_args,
            run_rank_0_only_axes=["tensor_parallel", "pipeline_parallel"],
        )

    def _ensure_checkpoint_engine_ready(self) -> None:
        if self._checkpoint_engine_ready:
            return

        cfg = self._checkpoint_engine_config
        backend = cfg["backend"]
        bucket_size_bytes = cfg["update_weights_bucket_megabytes"] * 1024 * 1024
        engine_kwargs = cfg["engine_kwargs"][backend]

        ray.get(
            self._run_policy(
                "init_checkpoint_engine",
                backend=backend,
                bucket_size_bytes=bucket_size_bytes,
                engine_kwargs=engine_kwargs,
            )
            + self._run_generation(
                "init_checkpoint_engine",
                (backend, bucket_size_bytes, engine_kwargs),
            )
        )

        policy_prepare_refs = self._run_policy("prepare_checkpoint_engine")
        generation_prepare_refs = self._run_generation("prepare_checkpoint_engine")
        prepare_results = ray.get(policy_prepare_refs + generation_prepare_refs)
        policy_metadata = _sort_ranked_metadata(
            _flatten_metadata(prepare_results[: len(policy_prepare_refs)])
        )
        generation_metadata = _ordered_generation_metadata(
            prepare_results[len(policy_prepare_refs) :]
        )
        topology = {
            "metadata": policy_metadata + generation_metadata,
            "train_world_size": len(policy_metadata),
            "rollout_world_size": len(generation_metadata),
        }
        worker_count = len(self._generation.worker_group.workers)
        workers_per_group = worker_count // self._generation.dp_size
        ray.get(
            self._run_policy("init_checkpoint_engine_process_group", **topology)
            + self._generation.worker_group.run_all_workers_multiple_data(
                self._generation_rpc(),
                method_args=[
                    (
                        rank_prefix,
                        topology["train_world_size"],
                        topology["rollout_world_size"],
                        topology["metadata"],
                    )
                    for rank_prefix in range(0, worker_count, workers_per_group)
                ],
                run_rank_0_only_axes=["tensor_parallel", "pipeline_parallel"],
                common_kwargs={
                    "checkpoint_method": "init_checkpoint_engine_process_group"
                },
            )
        )
        self._checkpoint_engine_ready = True

    def sync_weights(
        self,
        *,
        timer: Optional[Timer] = None,
        kv_scales: Optional[dict[str, float]] = None,
    ) -> None:
        self._stale = True
        self._ensure_checkpoint_engine_ready()
        context = (
            timer.time("prepare_for_generation/transfer_and_update_weights")
            if timer is not None
            else nullcontext()
        )

        with context:
            policy_refs = self._run_policy(
                "send_weights_via_checkpoint_engine", kv_scales=kv_scales
            )
            results = ray.get(
                policy_refs
                + self._run_generation("update_weights_from_checkpoint_engine")
            )
            if not all(
                result for result in results[len(policy_refs) :] if result is not None
            ):
                raise RuntimeError(
                    "Weight transfer failed during "
                    f"{self._checkpoint_engine_config['backend']} "
                    "checkpoint-engine sync."
                )
            self._stale = False

    def shutdown(self) -> None:
        if not self._checkpoint_engine_ready:
            return
        ray.get(
            self._run_policy("finalize_checkpoint_engine")
            + self._run_generation("finalize_checkpoint_engine")
        )
        self._checkpoint_engine_ready = False


def sync_weights_with_checkpoint_engine(
    policy: Any,
    generation: Any,
    *,
    timer: Optional[Timer] = None,
    kv_scales: Optional[dict[str, float]] = None,
) -> None:
    """Run one checkpoint-engine refit and release its peer connections."""
    synchronizer = CheckpointEngineWeightSynchronizer(
        policy, generation, generation.cfg["checkpoint_engine"]
    )
    try:
        synchronizer.sync_weights(timer=timer, kv_scales=kv_scales)
    finally:
        synchronizer.shutdown()
