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

"""NCCL-xfer (shard-to-shard) weight synchronizer for non-colocated deployments.

Handles disaggregated Megatron-train -> vLLM-gen weight refit via the
``xferdtensor`` reshard: bulk FFN/expert params are resharded shard-to-shard
between the train and gen parallelism layouts over a dedicated per-PP-stage NCCL
communicator, while the remaining "misc" params ride a packed broadcast over the
shared ``model_update_group``. Unlike the plain collective synchronizer (which
broadcasts every full tensor), this path redistributes each param directly
between layouts, avoiding a full gather + broadcast.

Lifecycle:
  init_communicator():
    1. policy/generation.init_collective()           -- model_update_group (misc)
    2. policy/generation.init_nccl_reshard_comm_group()  -- per-PP-stage bulk groups
    3. policy.prepare_nccl_reshard_refit_info()
       -> generation.prepare_nccl_reshard_refit_info()   -- backend-agnostic metadata
  sync_weights():
    policy.nccl_reshard_refit(kv_scales) + generation.nccl_reshard_refit(); verify.

Like the collective transport, this is a pure data mover: policy and generation
run on separate GPU clusters, so the phase transitions (offload / restore) are
owned by the orchestrator, not here.
"""

from contextlib import nullcontext
from typing import Any, Optional

import ray

from nemo_rl.utils.timer import Timer
from nemo_rl.weight_sync.interfaces import WeightSynchronizer


class NcclReshardWeightSynchronizer(WeightSynchronizer):
    """Weight synchronizer using the ``xferdtensor`` shard-to-shard reshard.

    For non-colocated Megatron-train -> vLLM-gen deployments where weights are
    redistributed directly between the two parallelism layouts (bulk path) plus
    a packed broadcast for the misc params. Mirrors
    :class:`CollectiveWeightSynchronizer` but additionally bootstraps the
    per-PP-stage bulk communicators and the nccl_reshard refit metadata.

    The train/gen parallelism and per-node GPU count are derived from the
    ``policy``/``generation`` configs and the clusters, so construction matches
    the collective synchronizer's signature.

    Args:
        policy: Policy object implementing ColocatablePolicyInterface (Megatron).
        generation: Generation object implementing GenerationInterface (vLLM).
        train_cluster: RayVirtualCluster for the training workers.  Only used by
            ``init_communicator()``; may be ``None`` for sync-only instances.
        inference_cluster: RayVirtualCluster for the inference workers.  Only
            used by ``init_communicator()``; may be ``None`` for sync-only
            instances.
    """

    def __init__(
        self,
        policy: Any,
        generation: Any,
        train_cluster: Any,
        inference_cluster: Any,
    ):
        self._policy = policy
        self._generation = generation
        self._train_cluster = train_cluster
        self._inference_cluster = inference_cluster
        self._stale = True

    def _train_parallelism(self) -> dict[str, int]:
        megatron_cfg = self._policy.cfg["megatron_cfg"]
        return {
            "tp_size": megatron_cfg.get("tensor_model_parallel_size", 1),
            "ep_size": megatron_cfg.get("expert_model_parallel_size", 1),
            "pp_size": megatron_cfg.get("pipeline_model_parallel_size", 1),
        }

    def _gen_parallelism(self) -> dict[str, int]:
        vllm_cfg = self._policy.cfg["generation"].get("vllm_cfg", {})
        return {
            "tp_size": vllm_cfg.get("tensor_parallel_size", 1),
            "ep_size": vllm_cfg.get("expert_parallel_size", 1),
            "pp_size": vllm_cfg.get("pipeline_parallel_size", 1),
        }

    def sync_weights(
        self,
        *,
        timer: Optional[Timer] = None,
        kv_scales: Optional[dict[str, float]] = None,
    ) -> None:
        timer_context = (
            timer.time("prepare_for_generation/transfer_and_update_weights")
            if timer is not None
            else nullcontext()
        )
        with timer_context:
            # Shard-to-shard reshard: train sends its TP/EP-local shards, gen
            # receives directly into its own (different) layout.  kv_scales ride
            # the misc packed-broadcast for FP8 KV cache.
            futures_train = self._policy.nccl_reshard_refit(kv_scales=kv_scales)
            futures_inference = self._generation.nccl_reshard_refit()

            ray.get(futures_train)
            results = ray.get(futures_inference)
            update_success = all(result for result in results if result is not None)

            if not update_success:
                raise RuntimeError(
                    "Weight transfer failed during nccl_reshard reshard sync. "
                    "This often indicates an issue with the NCCL process group "
                    "or the generation backend worker."
                )

        self._stale = False

    @property
    def is_stale(self) -> bool:
        return self._stale

    def mark_stale(self) -> None:
        self._stale = True

    def init_communicator(self) -> None:
        train_parallelism = self._train_parallelism()
        gen_parallelism = self._gen_parallelism()
        train_world_size = self._train_cluster.world_size()
        inference_world_size = self._inference_cluster.world_size()
        world_size = train_world_size + inference_world_size

        # 1. model_update_group: shared channel for the misc packed-broadcast
        #    (and the FP8 KV-cache scales).  Same setup as the collective path.
        ip, port = self._train_cluster.get_master_address_and_port()
        futures_train = self._policy.init_collective(
            ip, port, world_size, train_world_size=train_world_size
        )
        futures_inference = self._generation.init_collective(
            ip, port, world_size, train_world_size=train_world_size
        )
        ray.get(futures_train + futures_inference)

        # 2. Bulk-path comm group(s): one per PP stage, each spanning that
        #    stage's train ranks + all gen ranks (non-PP == a single stage over
        #    all train + gen ranks).  Separate NCCL communicator from
        #    model_update_group; the workers run the misc broadcast strictly
        #    after the bulk reshard (concurrent communicators can deadlock).
        pp_size = train_parallelism["pp_size"]
        train_gpus_per_node = self._train_cluster.num_gpus_per_node
        train_ranks_per_stage = train_world_size // pp_size
        sub_world_size = train_ranks_per_stage + inference_world_size
        pp_stages = [r // train_ranks_per_stage for r in range(train_world_size)]
        ranks_in_group = [r % train_ranks_per_stage for r in range(train_world_size)]
        # An IP and free port for each stage's group (one when non-PP).
        pp_ips: list[str] = []
        pp_ports: list[int] = []
        for stage in range(pp_size):
            node_idx = stage * train_ranks_per_stage // train_gpus_per_node
            stage_ip, stage_port = self._train_cluster.get_available_address_and_port(
                pg_idx=node_idx, bundle_idx=0
            )
            pp_ips.append(stage_ip)
            pp_ports.append(stage_port)
        print(
            f"nccl_reshard bulk comm group IPs/ports ({pp_size} stage(s)): "
            f"{list(zip(pp_ips, pp_ports))}",
            flush=True,
        )
        futures_train = self._policy.init_nccl_reshard_comm_group(
            pp_ips=pp_ips,
            pp_ports=pp_ports,
            pp_size=pp_size,
            pp_stages=pp_stages,
            sub_world_size=sub_world_size,
            ranks_in_group=ranks_in_group,
        )
        futures_inference = self._generation.init_nccl_reshard_comm_group(
            pp_ips=pp_ips,
            pp_ports=pp_ports,
            pp_size=pp_size,
            train_ranks_per_stage=train_ranks_per_stage,
            sub_world_size=sub_world_size,
        )
        ray.get(futures_train + futures_inference)

        # 3. Refit metadata.  Train builds backend-agnostic per-layer metadata
        #    (HF naming convention); gen maps it into its own fused layout
        #    (e.g. vLLM's w13/w2).
        nccl_reshard_refit_info = self._policy.prepare_nccl_reshard_refit_info(
            train_parallelism,
            gen_parallelism,
            train_world_size,
            inference_world_size,
        )
        self._generation.prepare_nccl_reshard_refit_info(nccl_reshard_refit_info)

    def shutdown(self) -> None:
        # The NCCL process groups' lifecycle is managed by Ray actor teardown;
        # the workers that own the groups are destroyed with the cluster.
        pass
