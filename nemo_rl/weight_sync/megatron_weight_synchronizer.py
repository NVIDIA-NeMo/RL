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
from typing import Any, Optional

import ray

from nemo_rl.utils.timer import Timer
from nemo_rl.weight_sync.interfaces import WeightSynchronizer


class MegatronWeightSynchronizer(WeightSynchronizer):
    """Weight synchronization for the Megatron generation backend, both colocation modes.

    Colocated is the degenerate path: generation either aliases the training
    weights outright (dual-mode) or re-partitions them into the worker's
    dedicated inference model inside ``prepare_for_generation`` (when the
    configured inference layout/impl differs) — a genuine parallelism-changing
    transfer, but one the worker performs internally on wake. Sync therefore
    reduces to dropping training-only buffers and re-entering inference mode.

    Non-colocated adds the cross-group collective: the training and inference
    workers are disjoint actor groups that rendezvous in mcore's
    reshard-capable weight swap, with the engine suspended around the
    transfer. That wiring (a joint refit process group over the configured
    copy-service backend) is established once in ``init_communicator``.
    """

    def __init__(
        self,
        policy: Any,
        generation: Any,
        *,
        colocated: bool,
        train_cluster: Optional[Any] = None,
        inference_cluster: Optional[Any] = None,
    ):
        if not colocated and (train_cluster is None or inference_cluster is None):
            raise ValueError(
                "train_cluster and inference_cluster are required for "
                "non-colocated Megatron weight synchronization."
            )
        self._policy = policy
        self._generation = generation
        self._colocated = colocated
        self._train_cluster = train_cluster
        self._inference_cluster = inference_cluster
        self._refit_backend: Optional[str] = None
        self._stale = True

    def init_communicator(self) -> None:
        """Wire the cross-group refit collective (non-colocated only).

        Colocated generation shares the training worker group, so there is
        nothing to wire.
        """
        if self._colocated:
            return
        ip, port = self._train_cluster.get_master_address_and_port()
        print(f"Using ip: {ip}, port: {port} for collective communication", flush=True)
        train_world_size = self._train_cluster.world_size()
        world_size = train_world_size + self._inference_cluster.world_size()
        self._refit_backend = self._generation.cfg["mcore_generation_config"][
            "refit_backend"
        ]
        futures_train = self._policy.init_collective_mcore_generation(
            ip,
            port,
            world_size,
            rank_offset=0,
            refit_backend=self._refit_backend,
        )
        futures_inference = self._generation.init_collective(
            ip,
            port,
            world_size,
            train_world_size=train_world_size,
            refit_backend=self._refit_backend,
        )
        ray.get(futures_train + futures_inference)

    def sync_weights(
        self,
        *,
        timer: Optional[Timer] = None,
        kv_scales: Optional[dict[str, float]] = None,
    ) -> Optional[dict[str, float]]:
        if self._colocated:
            # The wake below carries any configured reshard; the loop already
            # slept the engine before training (or it has not started yet),
            # so no suspend is needed.
            self._policy.offload_before_refit()
            self._generation.prepare_for_generation()
            self._stale = False
            return {}

        # The engine serves continuously in non-colocated mode; pause it
        # exactly around the swap.
        self._generation.suspend_for_refit()
        self._policy.offload_before_refit()
        self._generation.prepare_for_generation(tags=["weights"])

        if self._refit_backend == "nvshmem":
            futures_train = self._policy.preinit_nvshmem()
            futures_inference = self._generation.preinit_nvshmem_collective()
            ray.get(futures_train + futures_inference)

        timer_context = (
            timer.time("prepare_for_generation/transfer_and_update_weights")
            if timer is not None
            else nullcontext()
        )
        with timer_context:
            futures_train = self._policy.swap_weights_via_reshard(is_source=True)
            futures_inference = self._generation.update_weights_from_collective()
            ray.get(futures_train)
            results = ray.get(futures_inference)
            if not all(result for result in results if result is not None):
                raise RuntimeError(
                    "❌ Error: Updating weights for the generation policy failed "
                    "during refit.\nThis often indicates an issue with the "
                    "refit copy service or a problem within the generation "
                    "backend.\n"
                )

        self._generation.prepare_for_generation(tags=["kv_cache"])
        self._generation.resume_after_refit()
        self._stale = False
        return {}

    @property
    def is_stale(self) -> bool:
        return self._stale

    def shutdown(self) -> None:
        """Nothing to tear down; the collective lives in the worker groups."""
