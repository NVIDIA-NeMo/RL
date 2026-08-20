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

Identical wire path to ``NcclReshardWeightSynchronizer``. The only difference
is where each lane's ``ncclUniqueId`` comes from: ModelExpress rather than a
``TCPStore`` the driver has to allocate a port for per pipeline stage.

That containment is deliberate. ``sync_weights`` below calls the *existing*
``nccl_reshard_refit`` on both sides, unchanged, so the two transports cannot
drift apart -- which is what makes a measured comparison between them mean
anything.
"""

from contextlib import nullcontext
from typing import Any, Optional

import ray

from nemo_rl.utils.timer import Timer
from nemo_rl.weight_sync.mx_collective_bootstrap import mx_lane_order
from nemo_rl.weight_sync.interfaces import WeightSynchronizer

MX_COLLECTIVE_TRANSPORT = "mx_nccl_reshard"


class MxCollectiveWeightSynchronizer(WeightSynchronizer):
    """nccl_reshard refit whose communicator bootstrap is brokered by ModelExpress."""

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
        if not mx_server_url:
            raise ValueError(
                "policy.generation.mx_server_url is required for "
                f"refit_transport={MX_COLLECTIVE_TRANSPORT!r}. Every worker on "
                "both sides must be given the same address; two different "
                "addresses form two groups, neither of which reaches READY."
            )
        self._mx_server_url = mx_server_url
        self._stale = True

    def _train_parallelism(self) -> dict:
        cfg = self._policy.cfg["megatron_cfg"]
        return {
            "tp_size": cfg.get("tensor_model_parallel_size", 1),
            "ep_size": cfg.get("expert_model_parallel_size", 1),
            "pp_size": cfg.get("pipeline_model_parallel_size", 1),
        }

    def _gen_parallelism(self) -> dict:
        cfg = self._policy.cfg["generation"].get("vllm_cfg", {})
        return {
            "tp_size": cfg.get("tensor_parallel_size", 1),
            "ep_size": cfg.get("expert_parallel_size", 1),
            "pp_size": cfg.get("pipeline_parallel_size", 1),
        }

    def init_communicator(self) -> None:
        train_parallelism = self._train_parallelism()
        gen_parallelism = self._gen_parallelism()
        train_world_size = self._train_cluster.world_size()
        gen_world_size = self._inference_cluster.world_size()
        pp_size = train_parallelism["pp_size"]
        model_name = self._policy.cfg.get("model_name", "unknown-model")

        refit_info = self._policy.prepare_nccl_reshard_refit_info(
            train_parallelism, gen_parallelism, train_world_size, gen_world_size
        )

        # Both sides must agree on the digest or the group never reaches READY,
        # so it is derived from the metadata the trainer already published
        # rather than computed independently on each side.
        try:
            from nemo_rl.weight_sync.mx_collective_plan import build_mx_plan
            from modelexpress_rl.collective import plan_digest

            digest = plan_digest(build_mx_plan(refit_info))
        except Exception as error:  # noqa: BLE001 - digest is an agreement token
            print(f"[mx] plan digest unavailable ({error!r}); using a constant", flush=True)
            digest = "nccl-reshard-plan"

        trainers = [f"train/{r}" for r in range(train_world_size)]
        generators = [f"gen/{r}" for r in range(gen_world_size)]
        ranks_per_stage = max(train_world_size // pp_size, 1)
        pp_stages = [r // ranks_per_stage for r in range(train_world_size)]

        print(
            f"[mx] forming group via {self._mx_server_url}: "
            f"{train_world_size} trainers + {gen_world_size} generators, "
            f"{pp_size} source partition(s), digest={digest[:12]}",
            flush=True,
        )

        common = dict(
            mx_server_url=self._mx_server_url,
            model_name=model_name,
            trainer_slots=trainers,
            generator_slots=generators,
            source_partition_count=pp_size,
            plan_digest=digest,
        )

        def both(phase):
            """Run one phase on every worker and wait for all of them.

            The wait is the point. Creating two different NCCL communicators
            concurrently across overlapping rank sets deadlocks, so no worker
            may start lane N+1 while another is still inside lane N. The
            TCPStore path separates its all-ranks group from its per-stage
            groups for exactly this reason.
            """
            futures_train = self._policy.init_mx_reshard_comm_group(
                pp_stages=pp_stages, phase=phase, **common
            )
            futures_gen = self._generation.init_mx_reshard_comm_group(
                phase=phase, **common
            )
            ray.get(futures_train + futures_gen)

        # Both sides block inside the rendezvous, so they have to be in flight
        # together; no communicator is created yet.
        both("rendezvous")

        # Broadcast lane first: every rank is in it, so the barrier after it is
        # a full-cluster sync point. Then each reshard lane, one at a time.
        for lane_id in mx_lane_order(pp_size):
            both(lane_id)
        both("finish")
        print("[mx] communicators ready", flush=True)

        self._generation.prepare_nccl_reshard_refit_info(refit_info)

    def sync_weights(
        self,
        *,
        timer: Optional[Timer] = None,
        kv_scales: Optional[dict] = None,
    ) -> None:
        ctx = (
            timer.time("prepare_for_generation/transfer_and_update_weights")
            if timer is not None
            else nullcontext()
        )
        with ctx:
            futures_train = self._policy.nccl_reshard_refit(kv_scales=kv_scales)
            futures_inference = self._generation.nccl_reshard_refit()
            ray.get(futures_train)
            results = ray.get(futures_inference)
            if not all(r for r in results if r is not None):
                raise RuntimeError(
                    "Weight transfer failed during the ModelExpress collective refit."
                )
        self._stale = False

    @property
    def is_stale(self) -> bool:
        return self._stale

    def mark_stale(self) -> None:
        self._stale = True

    def shutdown(self) -> None:
        pass
