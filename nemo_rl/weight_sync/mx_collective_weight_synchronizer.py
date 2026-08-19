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

"""ModelExpress-brokered variant of the nccl_reshard weight synchronizer.

Same wire path as ``NcclReshardWeightSynchronizer``: bulk FFN parameters move
through ``nccl.m2n.reshard`` between the train and generation layouts, and the
remaining parameters ride a packed broadcast. The difference is entirely in how
the communicators come to exist.

The native path bootstraps through a ``StatelessProcessGroup``: a ``TCPStore``
whose only job is to move 128 bytes of ``ncclUniqueId`` from rank 0 to everyone
else, at an IP and port the Ray driver allocates and plumbs through both actor
sets, once per pipeline stage. That works, and it gives up three things a
coordination service can provide:

* **Admission.** Membership is whoever happens to connect. There is no expected
  set, so a missing worker is a hang rather than a bounded failure naming it.
* **Fencing.** A worker that dies and restarts rejoins silently, and the
  surviving ranks cannot tell the generation changed underneath them.
* **Observable readiness.** The trainer has no way to check that the generators
  it is about to push into have prepared their destinations; it can only enter
  the collective and block.

Routing the bootstrap through ModelExpress supplies all three, and removes the
per-stage port allocation from the driver. Everything below the rendezvous is
unchanged, which is deliberate: the two transports should move identical bytes
so they can be compared directly.
"""

from contextlib import nullcontext
from typing import Any, Optional

import ray

from nemo_rl.utils.timer import Timer
from nemo_rl.weight_sync.interfaces import WeightSynchronizer

MX_COLLECTIVE_TRANSPORT = "mx_nccl_reshard"


class MxCollectiveWeightSynchronizer(WeightSynchronizer):
    """Weight synchronizer whose NCCL rendezvous is brokered by ModelExpress.

    Args:
        policy: Policy object implementing ColocatablePolicyInterface (Megatron).
        generation: Generation object implementing GenerationInterface (vLLM).
        train_cluster: RayVirtualCluster for the training workers.
        inference_cluster: RayVirtualCluster for the inference workers.
        mx_server_url: Address of the ModelExpress server that brokers the
            group. Every worker on both sides must be given the same one, or
            they form two groups that never reach READY.
    """

    def __init__(
        self,
        policy: Any,
        generation: Any,
        train_cluster: Any,
        inference_cluster: Any,
        mx_server_url: Optional[str] = None,
    ):
        self._policy = policy
        self._generation = generation
        self._train_cluster = train_cluster
        self._inference_cluster = inference_cluster
        self._mx_server_url = mx_server_url
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

    def init_communicator(self) -> None:
        """Form the ModelExpress group and build every lane's communicator.

        Runs once. The expensive part is the per-lane ``Communicator.init``,
        which MX lets us keep across refits: the group is keyed by membership,
        and only a membership or plan change moves its epoch and invalidates
        the cached communicators.

        Note what is *absent* compared with the native path: no per-stage IP and
        port allocation, and no rank arithmetic in the driver. MX assigns
        ``rank_in_lane`` from the role, the ordinal within the role, and the
        pipeline stage, using the same convention the native path hardcodes --
        trainer ranks first, generators after -- so the mesh metadata carries
        over unchanged.
        """
        train_parallelism = self._train_parallelism()
        gen_parallelism = self._gen_parallelism()
        train_world_size = self._train_cluster.world_size()
        inference_world_size = self._inference_cluster.world_size()

        refit_info = self._policy.prepare_nccl_reshard_refit_info(
            train_parallelism,
            gen_parallelism,
            train_world_size,
            inference_world_size,
        )
        self._generation.prepare_nccl_reshard_refit_info(refit_info)

        futures_train = self._policy.init_mx_collective_group(
            mx_server_url=self._mx_server_url,
            train_world_size=train_world_size,
            gen_world_size=inference_world_size,
            source_partition_count=train_parallelism["pp_size"],
        )
        futures_inference = self._generation.init_mx_collective_group(
            mx_server_url=self._mx_server_url,
            train_world_size=train_world_size,
            gen_world_size=inference_world_size,
            source_partition_count=train_parallelism["pp_size"],
        )
        ray.get(futures_train + futures_inference)

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
            futures_train = self._policy.mx_collective_refit(kv_scales=kv_scales)
            futures_inference = self._generation.mx_collective_refit()

            ray.get(futures_train)
            results = ray.get(futures_inference)
            update_success = all(result for result in results if result is not None)

            if not update_success:
                raise RuntimeError(
                    "Weight transfer failed during the ModelExpress collective refit. "
                    "Check the ModelExpress server logs for the group's state: a group "
                    "that never reached READY names the participants it was waiting on."
                )

        self._stale = False

    @property
    def is_stale(self) -> bool:
        return self._stale

    def mark_stale(self) -> None:
        self._stale = True

    def shutdown(self) -> None:
        # Communicator teardown follows Ray actor teardown, as with the native
        # path. The MX group is reclaimed on its own once every participant's
        # registration lapses, so there is nothing to delete here.
        pass
