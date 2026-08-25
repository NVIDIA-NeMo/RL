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
"""Controller-side ``FlowGRPOPolicy`` facade.

Owns one :class:`nemo_rl.distributed.worker_groups.RayWorkerGroup` of
:class:`nemo_rl.models.diffusion.workers.flow_grpo_worker.FlowGRPOPolicyWorker`
actors. With N workers (``cluster.gpus_per_node = N``) the policy runs
data-parallel: rollout prompts are scattered across workers (each worker gets
a distinct seed), trajectories are gathered/concatenated on the controller,
and training data is re-scattered along the same split; workers all-reduce
their gradients so every rank applies the identical update.

``sample_trajectory`` falls back to worker 0 when the prompt count doesn't
split evenly (e.g. the K=1 validation path); training requires
``num_prompts_per_step % num_workers == 0``.

Method names mirror the API the trainer calls: ``sample_trajectory``,
``train``, ``save_checkpoint``, ``shutdown``.
"""

from __future__ import annotations

from typing import Any

import ray
import torch

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import RayVirtualCluster
from nemo_rl.distributed.worker_groups import RayWorkerBuilder, RayWorkerGroup
from nemo_rl.models.diffusion.interfaces import (
    DiffusionPolicyConfig,
    DiffusionTrainDataSpec,
    DiffusionTrajectorySpec,
)


def derive_rollout_seed(
    seed: int | None, worker_idx: int, *, single_seed: bool = False
) -> int | None:
    """Per-worker rollout seed for scattered sample_trajectory calls.

    Training uses a distinct seed per rank so initial latents decorrelate
    across DP workers while staying reproducible. `single_seed=True` (the
    single-seed validation mode) returns the caller's seed verbatim on every
    rank instead.
    """
    if seed is None:
        return None
    if single_seed:
        return seed
    return seed + worker_idx * 7919


def aggregate_worker_metrics(
    per_worker: list[dict[str, float]],
) -> dict[str, float]:
    """Reduce per-DP-worker train metrics into one dict.

    `ratio_min`/`ratio_max` take min/max across workers; every other key is
    averaged over the workers that reported it.
    """
    keys = set().union(*(d.keys() for d in per_worker))
    out: dict[str, float] = {}
    for k in keys:
        vals = [d[k] for d in per_worker if k in d]
        if k == "ratio_min":
            out[k] = min(vals)
        elif k == "ratio_max":
            out[k] = max(vals)
        else:
            out[k] = sum(vals) / len(vals)
    return out


class FlowGRPOPolicy:
    """Controller-side facade around a Ray pool of diffusion workers."""

    def __init__(
        self,
        cluster: RayVirtualCluster,
        config: DiffusionPolicyConfig,
        *,
        name_prefix: str = "flow_grpo_policy",
    ) -> None:
        self.cluster = cluster
        self.config = config
        builder = RayWorkerBuilder(
            "nemo_rl.models.diffusion.workers.flow_grpo_worker.FlowGRPOPolicyWorker",
            config,
        )
        self.worker_group = RayWorkerGroup(
            cluster=cluster,
            remote_worker_builder=builder,
            name_prefix=name_prefix,
        )

    # ------------------------------------------------------------------
    # Forward to worker(s) — single-worker path
    # ------------------------------------------------------------------
    def _call_all(self, method_name: str, **kwargs) -> list:
        """Invoke a method on every worker; return list of results (ray.get).

        Bypasses ``run_all_workers_single_data`` because that path requires
        ``sharding_annotations`` which we don't define for the diffusion
        single-worker v1.
        """
        futures = [
            getattr(w, method_name).remote(**kwargs) for w in self.worker_group.workers
        ]
        return ray.get(futures)

    @property
    def num_workers(self) -> int:
        return len(self.worker_group.workers)

    @staticmethod
    def _merge_trajectories(
        trajs: list[DiffusionTrajectorySpec],
    ) -> DiffusionTrajectorySpec:
        """Concatenate per-worker trajectories along the batch dim.

        Prompt-embedding tensors are padded on dim 1 to the max sequence
        length across workers (each worker pads only to its local batch max);
        the matching masks zero out the padding.
        """
        if len(trajs) == 1:
            return trajs[0]
        trajs_d: list[dict[str, Any]] = [dict(t) for t in trajs]
        merged: dict[str, Any] = {}
        for key, first in trajs_d[0].items():
            if not torch.is_tensor(first):
                merged[key] = [x for t in trajs_d for x in t[key]]
                continue
            parts = [t[key] for t in trajs_d]
            if first.ndim >= 2 and len({p.shape[1] for p in parts}) > 1:
                seq_max = max(p.shape[1] for p in parts)
                parts = [
                    torch.nn.functional.pad(
                        p, (0,) * (2 * (p.ndim - 2)) + (0, seq_max - p.shape[1])
                    )
                    for p in parts
                ]
            merged[key] = torch.cat(parts, dim=0)
        return merged  # type: ignore[return-value]

    def sample_trajectory(
        self,
        prompts: list[str],
        negative_prompts: list[str],
        metadata: list[dict[str, Any]],
        *,
        K: int,
        seed: int | None = None,
        generation_overrides: dict[str, Any] | None = None,
    ) -> DiffusionTrajectorySpec:
        n = self.num_workers
        if n == 1 or len(prompts) < n or len(prompts) % n != 0:
            # Uneven splits (e.g. the K=1 validation path with 1 prompt per
            # batch) run on worker 0 alone.
            if n > 1:
                print(
                    f"[FlowGRPOPolicy] rollout of {len(prompts)} prompts does "
                    f"not split across {n} workers; running on worker 0 only",
                    flush=True,
                )
            future = self.worker_group.workers[0].sample_trajectory.remote(
                prompts=prompts,
                negative_prompts=negative_prompts,
                metadata=metadata,
                K=K,
                seed=seed,
                generation_overrides=generation_overrides,
            )
            return ray.get(future)
        # generation_overrides is a full model_dump() of
        # FlowGRPOValGenerationCfg (or None on the training path).
        single_seed = generation_overrides is not None and bool(
            generation_overrides["single_seed"]
        )
        shard = len(prompts) // n
        futures = []
        for i, worker in enumerate(self.worker_group.workers):
            lo, hi = i * shard, (i + 1) * shard
            futures.append(
                worker.sample_trajectory.remote(
                    prompts=prompts[lo:hi],
                    negative_prompts=negative_prompts[lo:hi],
                    metadata=metadata[lo:hi],
                    K=K,
                    seed=derive_rollout_seed(seed, i, single_seed=single_seed),
                    generation_overrides=generation_overrides,
                )
            )
        return self._merge_trajectories(ray.get(futures))

    def train(
        self,
        data: BatchedDataDict[DiffusionTrainDataSpec],
        # model_dump() dict view of FlowGRPOLossConfig.
        loss_cfg: dict[str, Any],
    ) -> dict[str, float]:
        n = self.num_workers
        total = int(data["generation_logprobs"].shape[0])
        if n == 1:
            per_worker = self._call_all("train_step", data=data, loss_cfg=loss_cfg)
        else:
            if total % n != 0:
                raise ValueError(
                    f"train batch of {total} samples is not divisible by "
                    f"{n} DP workers; set grpo.num_prompts_per_step to a "
                    f"multiple of cluster.gpus_per_node"
                )
            # Same contiguous split as the rollout scatter, so each worker
            # trains on the samples it generated.
            shard = total // n
            futures = [
                w.train_step.remote(
                    data=data.slice(i * shard, (i + 1) * shard),
                    loss_cfg=loss_cfg,
                )
                for i, w in enumerate(self.worker_group.workers)
            ]
            per_worker = ray.get(futures)
        return aggregate_worker_metrics(per_worker)

    def trainable_checksums(self) -> list[float]:
        """Per-worker trainable-param checksums; DP ranks must agree."""
        return self._call_all("report_trainable_checksum")

    def prepare_for_generation(self) -> None:
        self._call_all("prepare_for_generation")

    def prepare_for_training(self) -> None:
        self._call_all("prepare_for_training")

    def save_checkpoint(self, path: str, *, save_optimizer: bool = True) -> None:
        self._call_all("save_checkpoint", path=path, save_optimizer=save_optimizer)

    def load_checkpoint(self, path: str) -> None:
        self._call_all("load_checkpoint", path=path)

    def shutdown(self) -> bool:
        try:
            self._call_all("shutdown")
        finally:
            self.worker_group.shutdown()
        return True
