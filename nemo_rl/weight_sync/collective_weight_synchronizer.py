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

"""NCCL collective weight synchronizer for non-colocated deployments.

Handles weight transfer between policy and generation workers running on
separate GPU clusters using NCCL collective communication. The policy
broadcasts its weights, and generation workers receive them via the
established NCCL process group.

Lifecycle per sync:
  1. policy.broadcast_weights_for_collective()    -- send via NCCL
     generation.update_weights_from_collective()  -- receive via NCCL
  2. Verify transfer success

No offload/restore steps are needed since policy and generation run on
separate GPUs with dedicated memory.
"""

from contextlib import nullcontext
from typing import Any, Optional

import ray

from nemo_rl.utils.timer import Timer
from nemo_rl.weight_sync.interfaces import WeightSynchronizer
from nemo_rl.weight_sync.refit_transaction import (
    shutdown_refit_participants,
    wait_for_real_quant_refit,
)


class CollectiveWeightSynchronizer(WeightSynchronizer):
    """Weight synchronizer using NCCL collectives for non-colocated deployments.

    Policy and generation workers run on separate GPU clusters. Weights are
    synchronized via NCCL broadcast over a pre-established process group.

    Args:
        policy: Policy object implementing ColocatablePolicyInterface.
        generation: Generation object implementing GenerationInterface.
        train_cluster: RayVirtualCluster for the training workers, used to
            obtain the master address/port and world size for collective init.
        inference_cluster: RayVirtualCluster for the inference workers.
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
        self._fatal_refit_error: Exception | None = None

    def sync_weights(
        self,
        *,
        timer: Optional[Timer] = None,
        kv_scales: Optional[dict[str, float]] = None,
    ) -> None:
        if self._fatal_refit_error is not None:
            raise RuntimeError(
                "Real-quant refit is unusable after a prior failed transaction"
            ) from self._fatal_refit_error

        self._stale = True
        timer_context = (
            timer.time("prepare_for_generation/transfer_and_update_weights")
            if timer is not None
            else nullcontext()
        )
        with timer_context:
            try:
                futures_train = self._policy.broadcast_weights_for_collective(
                    kv_scales=kv_scales
                )
                futures_inference = self._generation.update_weights_from_collective()
                if self._generation.cfg.get("real_quant"):
                    results = wait_for_real_quant_refit(
                        futures_train, futures_inference
                    )
                else:
                    ray.get(futures_train)
                    results = ray.get(futures_inference)
                update_success = all(
                    result for result in results if result is not None
                )
                if not update_success:
                    raise RuntimeError(
                        "Weight transfer failed during NCCL collective sync. "
                        "This often indicates an issue with the NCCL process group "
                        "or the generation backend worker."
                    )
                if self._generation.cfg.get("real_quant"):
                    print(
                        "ModelOpt real-quant refit transaction completed", flush=True
                    )
            except Exception as error:
                if self._generation.cfg.get("real_quant"):
                    self._fatal_refit_error = error
                    shutdown_refit_participants(
                        self._policy, self._generation, error
                    )
                raise

        self._stale = False

    @property
    def is_stale(self) -> bool:
        return self._stale

    def mark_stale(self) -> None:
        self._stale = True

    def init_communicator(self) -> None:
        # prepare_refit_info is called before init_collective. This matches
        # distillation.py ordering. Neither call depends on the other today,
        # but we document this as the canonical ordering for future reference.
        state_dict_info = self._policy.prepare_refit_info()
        self._generation.prepare_refit_info(state_dict_info)

        ip, port = self._train_cluster.get_master_address_and_port()
        train_world_size = self._train_cluster.world_size()
        inference_world_size = self._inference_cluster.world_size()
        world_size = train_world_size + inference_world_size

        futures_train = self._policy.init_collective(
            ip, port, world_size, train_world_size=train_world_size
        )
        futures_inference = self._generation.init_collective(
            ip, port, world_size, train_world_size=train_world_size
        )
        ray.get(futures_train + futures_inference)

    def shutdown(self) -> None:
        # The NCCL process group lifecycle is managed by Ray actor teardown.
        # Explicit destroy_process_group() is not needed here because the
        # workers that own the group are destroyed when the cluster shuts down.
        pass
