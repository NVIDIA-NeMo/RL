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

Handles disaggregated Megatron-train -> vLLM/Megatron-gen weight refit via the
``xferdtensor`` reshard: bulk FFN/expert params are resharded shard-to-shard
between the train and gen parallelism layouts over a dedicated per-PP-stage NCCL
communicator, while the remaining "misc" params ride a packed broadcast over the
shared ``model_update_group``. Unlike the plain collective synchronizer (which
broadcasts every full tensor), this path redistributes each param directly
between layouts, avoiding a full gather + broadcast.

Lifecycle:
  init_communicator():
    1. policy/generation.init_collective()           -- model_update_group (misc)
    2. policy.init_nccl_reshard_comm_group() and
       generation.rebuild_nccl_reshard_comm_group()     -- per-PP-stage bulk groups
    3. policy.prepare_nccl_reshard_refit_info()
       -> generation.prepare_nccl_reshard_refit_info()   -- backend-agnostic metadata
  sync_weights():
    policy.offload_before_refit()                       -- optional trainer memory release
    policy.nccl_reshard_refit(kv_scales) + generation.nccl_reshard_refit(); verify.

Like the collective transport, this is normally a pure data mover: policy and
generation run on separate GPU clusters, so restore is owned by the orchestrator,
not here. Trainer offload is disabled by default, but large quantized exports can
opt in when their temporary tensors need more trainer GPU headroom -- the reshard
only moves params, so releasing grad buffers/optimizer state/caches first is safe.
"""

from collections.abc import Sequence
from contextlib import nullcontext
from typing import Any, Optional

import ray

from nemo_rl.models.generation.megatron.config import merged_inference_megatron_cfg
from nemo_rl.utils.timer import Timer
from nemo_rl.weight_sync.interfaces import WeightSynchronizer
from nemo_rl.weight_sync.membership import (
    RefitMembership,
    desired_membership,
    should_rebuild,
)
from nemo_rl.weight_sync.nccl_reshard_utils import (
    make_nccl_reshard_refit_info_wire_safe,
)


def _settle_before_propagating(futures, budget_s, what: str) -> None:
    """Let every rank finish unwinding before a refit failure reaches the caller.

    ``ray.get`` raises on the FIRST future that fails and leaves the rest running. That is
    fine when the caller is going to stop, and wrong when it is going to rebuild: a
    communicator rebuild is itself a collective, so dispatching ``init_collective`` while
    some ranks are still inside the old refit means they join late or not at all, and the
    rendezvous times out instead of coming up.

    Job 6512153 measured exactly that on the reshard kill variant. Rank 0 gave up on its
    own deadline, the controller went straight into the recovery, and the rebuild began --
    line 963 of the log -- two lines BEFORE rank 1's watchdog fired at all. The surviving
    generation worker then spent 300s twice failing to reach a store that never came up,
    and the run died at 690s having done everything else right.

    Bounded, and swallowing whatever the stragglers raise: they are unwinding from the same
    failure the caller is already holding, and replacing it with a straggler's version
    would lose the diagnosis. If the budget runs out, propagate anyway -- a caller stuck
    here would be a worse wedge than the one being recovered from.
    """
    if not futures:
        return
    try:
        ready, pending = ray.wait(
            list(futures), num_returns=len(futures), timeout=budget_s
        )
        if pending:
            print(
                f"  refit: {len(pending)} of {len(futures)} {what} rank(s) had not "
                f"unwound after {budget_s}s; rebuilding anyway",
                flush=True,
            )
    except Exception:  # noqa: BLE001 - the caller's failure is the one that matters
        pass


class NcclReshardWeightSynchronizer(WeightSynchronizer):
    """Weight synchronizer using the ``xferdtensor`` shard-to-shard reshard.

    For non-colocated Megatron-train -> vLLM/Megatron-gen deployments where weights are
    redistributed directly between the two parallelism layouts (bulk path) plus
    a packed broadcast for the misc params. Mirrors
    :class:`CollectiveWeightSynchronizer` but additionally bootstraps the
    per-PP-stage bulk communicators and the nccl_reshard refit metadata.

    The train/gen parallelism and per-node GPU count are derived from the
    ``policy``/``generation`` configs and the clusters, so construction matches
    the collective synchronizer's signature.

    Args:
        policy: Policy object implementing ColocatablePolicyInterface (Megatron).
        generation: Generation object implementing GenerationInterface.
        train_cluster: RayVirtualCluster for the training workers.  Only used by
            ``init_communicator()``; may be ``None`` for sync-only instances.
        inference_cluster: RayVirtualCluster for the inference workers.  Only
            used by ``init_communicator()``; may be ``None`` for sync-only
            instances.
        refit_timeout_s: Deadline for one refit collective. Each participating worker
            arms a watchdog and aborts its own communicator when it expires, which is
            what lets the controller rebuild over the survivors instead of blocking in
            NCCL forever. ``None`` disarms it entirely, so the hang protection is lost.
        sync_policy_params: Whether this synchronizer owns the pre-transfer policy
            parameter sync. A lifecycle wrapper may perform it earlier and disable it
            here to avoid a duplicate worker round trip.
        release_grads_before_refit: Whether to run the policy's existing refit
            offload lifecycle before the reshard transfer. Mirrors
            :class:`CollectiveWeightSynchronizer`; only touches gradient buffers,
            optimizer state, and cached tensors, never params, so it is safe for a
            transport that only moves params.
    """

    def __init__(
        self,
        policy: Any,
        generation: Any,
        train_cluster: Any,
        inference_cluster: Any,
        refit_timeout_s: Optional[float] = None,
        *,
        sync_policy_params: bool = True,
        release_grads_before_refit: bool = False,
    ) -> None:
        self._policy = policy
        self._generation = generation
        self._train_cluster = train_cluster
        self._inference_cluster = inference_cluster
        self._refit_timeout_s = refit_timeout_s
        self._sync_policy_params = sync_policy_params
        self._release_grads_before_refit = release_grads_before_refit
        self._stale = True
        # What the communicators were last built over. None until init_communicator.
        self._built_membership: Optional[RefitMembership] = None

    def _train_parallelism(self) -> dict[str, int]:
        megatron_cfg = self._policy.cfg["megatron_cfg"]
        tp_size = megatron_cfg.get("tensor_model_parallel_size", 1)
        etp_size = megatron_cfg.get("expert_tensor_parallel_size")
        return {
            "tp_size": tp_size,
            "ep_size": megatron_cfg.get("expert_model_parallel_size", 1),
            # MCore accepts None and resolves it to TP in ModelParallelConfig;
            # normalize at this boundary so the plan builder receives only ints.
            "etp_size": tp_size if etp_size is None else etp_size,
            "pp_size": megatron_cfg.get("pipeline_model_parallel_size", 1),
        }

    def _gen_parallelism(self) -> dict[str, int]:
        """The single generation layout (vLLM / Megatron generation)."""
        layouts = self._gen_layouts()
        if len(layouts) != 1:
            raise ValueError(
                "generation backend exposes several layouts; use _gen_layouts()"
            )
        return {
            key: layouts[0][key] for key in ("tp_size", "ep_size", "etp_size", "pp_size")
        }

    def _gen_layouts(self) -> list[dict[str, Any]]:
        """Destination layouts, each ``{tp_size, ep_size, etp_size, pp_size, world_size}``.

        vLLM and Megatron generation are one layout spanning every inference
        rank. TRT-LLM reports one per engine role: under prefill/decode
        disaggregation the context and generation engines have different
        TP/EP, and xferdtensor needs one destination mesh per transfer, so the
        bulk path runs once per layout (its own communicators and plan).
        """
        generation_cfg = self._policy.cfg["generation"]
        backend = generation_cfg["backend"]
        if backend == "vllm":
            vllm_cfg = generation_cfg.get("vllm_cfg", {})
            tp_size = vllm_cfg.get("tensor_parallel_size", 1)
            ep_size = vllm_cfg.get("expert_parallel_size", 1)
            return [
                {
                    "tp_size": tp_size,
                    "ep_size": ep_size,
                    "etp_size": tp_size if ep_size == 1 else 1,
                    "pp_size": vllm_cfg.get("pipeline_parallel_size", 1),
                    "world_size": None,
                }
            ]
        if backend == "megatron":
            # Resolve through the same merge the generation model is built from,
            # so the reshard mesh cannot drift from the layout MCore actually
            # instantiates (e.g. the inference_optimized ETP pin).
            megatron_cfg = merged_inference_megatron_cfg(self._policy.cfg)
            tp_size = megatron_cfg["tensor_model_parallel_size"]
            etp_size = megatron_cfg.get("expert_tensor_parallel_size")
            return [
                {
                    "tp_size": tp_size,
                    "ep_size": megatron_cfg["expert_model_parallel_size"],
                    # Match MCore's effective default: an omitted/None ETP uses TP.
                    "etp_size": tp_size if etp_size is None else etp_size,
                    "pp_size": megatron_cfg["pipeline_model_parallel_size"],
                    "world_size": None,
                }
            ]
        if backend == "trtllm":
            layouts = []
            for layout in self._generation.get_nccl_reshard_layouts():
                if layout["etp_size"] != 1:
                    raise ValueError(
                        f"TRT-LLM {layout['role']} engines split experts by TP "
                        f"(moe_tensor_parallel_size {layout['etp_size']}); the reshard "
                        "refit needs pure expert parallelism (moe_expert_parallel_size "
                        "== tensor_parallel_size)"
                    )
                layouts.append(
                    {
                        "role": layout["role"],
                        "tp_size": layout["tp_size"],
                        "ep_size": layout["ep_size"],
                        "etp_size": layout["etp_size"],
                        "pp_size": layout["pp_size"],
                        "world_size": layout["world_size"],
                    }
                )
            return layouts
        raise ValueError(
            "NCCL M-to-N refit only supports vLLM, Megatron or TRT-LLM generation, got "
            f"{backend!r}."
        )

    def sync_weights(
        self,
        *,
        timer: Optional[Timer] = None,
        kv_scales: Optional[dict[str, float]] = None,
    ) -> None:
        import time

        refit_t0 = time.perf_counter()
        if self._sync_policy_params:
            self._policy.sync_params_before_refit()
        sync_params_s = time.perf_counter() - refit_t0
        t_offload = time.perf_counter()
        if self._release_grads_before_refit:
            self._policy.offload_before_refit()
        offload_s = time.perf_counter() - t_offload

        timer_context = (
            timer.time("prepare_for_generation/transfer_and_update_weights")
            if timer is not None
            else nullcontext()
        )
        with timer_context:
            # Shard-to-shard reshard: train sends its TP/EP-local shards, gen
            # receives directly into its own (different) layout.  kv_scales ride
            # the misc packed-broadcast for FP8 KV cache.
            futures_train = self._policy.nccl_reshard_refit(
                kv_scales=kv_scales, refit_timeout_s=self._refit_timeout_s
            )
            futures_inference = self._generation.nccl_reshard_refit(
                refit_timeout_s=self._refit_timeout_s
            )

            try:
                ray.get(futures_train)
            except BaseException:
                # Every rank must be out of the old refit before the caller can rebuild
                # over the survivors; see _settle_before_propagating. BOTH sides: the
                # rebuild dispatches init_collective to the generation ranks too, and
                # ray.get(futures_train) raising leaves futures_inference running. Settling
                # only the train half is the same half-applied fix as the widened
                # signatures in design_vllm_fault_tolerance.md section 8.5.5.
                _settle_before_propagating(
                    futures_train, self._settle_budget_s(), "train"
                )
                _settle_before_propagating(
                    futures_inference, self._settle_budget_s(), "generation"
                )
                raise
            results = ray.get(futures_inference)
            print(
                f"[refit-timing] controller sync_params={sync_params_s:.2f}s "
                f"offload_before_refit={offload_s:.2f}s "
                f"transfer_and_update="
                f"{time.perf_counter() - refit_t0 - sync_params_s - offload_s:.2f}s "
                f"total={time.perf_counter() - refit_t0:.2f}s",
                flush=True,
            )
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

    def init_communicator(self) -> None:
        """Build both communicator families and the refit plan, over the whole fleet."""
        self._build(
            desired_membership(
                absent_shards=[],
                dp_size=self._generation.worker_group.dp_size,
                total_gen_workers=len(self._generation.worker_group.workers),
                train_world_size=self._train_cluster.world_size(),
            )
        )

    def _build(self, membership: RefitMembership) -> None:
        """Build everything this transport needs for the given fleet membership.

        Shared by the initial build and by a rebuild after a shard is lost, deliberately.
        All three pieces below are functions of the inference world size, so keeping two
        copies of this arithmetic is how the communicators and the refit plan would come
        to disagree -- and that disagreement is silent, because a mesh sized for the old
        fleet still runs, it just writes the wrong slices. Sharing the path also means
        every normal run exercises the rebuild code.
        """
        # First, so that everything below dispatches to the new membership. Step 3
        # distributes the regenerated plan through prepare_nccl_reshard_refit_info,
        # which consults it; setting it afterwards sends the new plan to the shard the
        # rebuild just excluded.
        self._generation.set_refit_membership(membership)

        train_parallelism = self._train_parallelism()
        train_world_size = membership.train_world_size
        world_size = membership.world_size
        inference_world_size = world_size - train_world_size

        # 1. model_update_group: shared channel for the misc packed-broadcast
        #    (and the FP8 KV-cache scales).  Same setup as the collective path.
        ip, port = self._train_cluster.get_master_address_and_port()
        # nccl_peer, for the same reason the collective synchronizer passes it: the
        # receiver's bootstrap is not negotiable, and omitting it rebuilds with the "nemo"
        # default. Mismatched warmups on one communicator hang rather than error.
        sender_spec = self._generation.get_collective_sender_spec()
        # The dispatch, so a missing [train] line below distinguishes "never asked" from
        # "asked and did not answer". The collective synchronizer prints its equivalent;
        # this path had none, which is why three rounds of diagnosis could not tell which
        # of the two it was.
        print(
            f"  refit: dispatching model_update_group rebuild addr={ip}:{port} "
            f"world_size={world_size} train_world_size={train_world_size} "
            f"peer={sender_spec.nccl_peer}",
            flush=True,
        )
        futures_train = self._policy.init_collective(
            ip,
            port,
            world_size,
            train_world_size=train_world_size,
            nccl_peer=sender_spec.nccl_peer,
        )
        futures_inference = self._generation.rebuild_collective(membership, ip, port)
        ray.get(futures_train + futures_inference)

        # 2. Bulk-path comm group(s): one per (generation layout, PP stage),
        #    each spanning that stage's train ranks + the layout's gen ranks
        #    (non-PP, one layout == a single group over all train + gen ranks).
        #    Separate NCCL communicators from model_update_group; the workers
        #    run the misc broadcast strictly after the bulk reshard (concurrent
        #    communicators can deadlock).
        pp_size = train_parallelism["pp_size"]
        train_gpus_per_node = self._train_cluster.num_gpus_per_node
        train_ranks_per_stage = train_world_size // pp_size
        pp_stages = [r // train_ranks_per_stage for r in range(train_world_size)]
        ranks_in_group = [r % train_ranks_per_stage for r in range(train_world_size)]
        layouts = self._gen_layouts()
        multi_layout = len(layouts) > 1 or "role" in layouts[0]
        layout_worlds = [
            inference_world_size if layout["world_size"] is None else layout["world_size"]
            for layout in layouts
        ]
        if sum(layout_worlds) != inference_world_size:
            raise RuntimeError(
                f"generation layouts cover {sum(layout_worlds)} ranks but the inference "
                f"world has {inference_world_size}"
            )
        for layout_index, (layout, layout_world) in enumerate(zip(layouts, layout_worlds)):
            sub_world_size = train_ranks_per_stage + layout_world
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
                f"nccl_reshard bulk comm group IPs/ports (layout {layout_index}"
                f"{' ' + layout['role'] if 'role' in layout else ''}, {pp_size} stage(s), "
                f"sub_world_size {sub_world_size}): {list(zip(pp_ips, pp_ports))}",
                flush=True,
            )
            # Single-layout backends keep the upstream call shapes (no
            # ``layout_index``); only the multi-layout TRT-LLM path adds it.
            train_kwargs = dict(
                pp_ips=pp_ips,
                pp_ports=pp_ports,
                pp_size=pp_size,
                pp_stages=pp_stages,
                sub_world_size=sub_world_size,
                ranks_in_group=ranks_in_group,
            )
            if multi_layout:
                train_kwargs["layout_index"] = layout_index
            futures_train = self._policy.init_nccl_reshard_comm_group(**train_kwargs)
            gen_kwargs = dict(
                pp_ips=pp_ips,
                pp_ports=pp_ports,
                pp_size=pp_size,
                train_ranks_per_stage=train_ranks_per_stage,
                sub_world_size=sub_world_size,
            )
            if multi_layout:
                gen_kwargs["layout_index"] = layout_index
            futures_inference = self._generation.rebuild_nccl_reshard_comm_group(
                membership, **gen_kwargs
            )
            ray.get(futures_train + futures_inference)
        # 3. Refit metadata.  Train builds backend-agnostic per-layer metadata
        #    (HF naming convention) per layout; gen maps it into its own fused
        #    layout (e.g. vLLM's w13/w2, TRT-LLM's EP slots).
        #
        #    Regenerated, not reused, on a rebuild. Each parameter's destination
        #    placements are derived from the layout's world size, so a plan built
        #    for the old fleet would have survivors writing the slices the dead
        #    shard used to own and leaving their own unwritten -- with no error,
        #    because a stale mesh is still a valid mesh.
        wire_refit_infos = []
        for layout_index, (layout, layout_world) in enumerate(zip(layouts, layout_worlds)):
            gen_parallelism = {
                key: layout[key] for key in ("tp_size", "ep_size", "etp_size", "pp_size")
            }
            refit_kwargs = dict(
                refit_payload_mode=self._generation.get_refit_payload_mode(),
            )
            if multi_layout:
                refit_kwargs["layout_index"] = layout_index
            nccl_reshard_refit_info = self._policy.prepare_nccl_reshard_refit_info(
                train_parallelism,
                gen_parallelism,
                train_world_size,
                layout_world,
                **refit_kwargs,
            )
            # nccl_reshard_refit_info holds MeshInfo rank tensors created under
            # Megatron, whose pickles resolve a Megatron-patched storage loader and
            # therefore need `import megatron` on unpickle. Convert them to plain
            # lists here; the receiving worker rebuilds them in
            # `restore_refit_info_placements()`.
            wire_refit_infos.append(
                make_nccl_reshard_refit_info_wire_safe(nccl_reshard_refit_info)
            )
        self._generation.prepare_nccl_reshard_refit_info(
            wire_refit_infos if multi_layout else wire_refit_infos[0]
        )
        self._built_membership = membership

    def _settle_budget_s(self) -> float:
        """How long to let stragglers unwind: their own deadline, plus a little.

        A rank that has not given up yet will do so when its watchdog fires, which is the
        same ``refit_timeout_s`` every rank was armed with. Without a configured deadline
        there is nothing bounding them, so fall back to a fixed wait rather than blocking
        the recovery indefinitely.
        """
        return (self._refit_timeout_s or 60.0) + 30.0

    def reconcile_communicator(
        self, absent_shards: Sequence[int], force: bool = False
    ) -> bool:
        """Rebuild both communicator families and regenerate the refit plan.

        This transport is harder to recover than the plain broadcast, and the difference
        is worth stating rather than discovering. Two families must be reconciled: the
        shared ``model_update_group``, and the per-PP-stage bulk groups whose
        ``sub_world_size`` is itself a function of the inference world size.

        More importantly the bulk path is a mesh-to-mesh redistribute, not a broadcast:
        ``prepare_nccl_reshard_refit_info`` derives each parameter's destination
        placements from ``gen_world_size``, so every gen rank receives its own slice
        rather than the same bytes. Dropping a rank therefore does not merely reduce the
        number of receivers -- it orphans the slices that rank owned, and the survivors
        would come back holding weights that were never written. Resizing the
        communicators without regenerating the plan would corrupt the refit silently,
        which is why the plan is regenerated rather than reused.
        """
        membership = desired_membership(
            absent_shards=absent_shards,
            dp_size=self._generation.worker_group.dp_size,
            total_gen_workers=len(self._generation.worker_group.workers),
            train_world_size=self._train_cluster.world_size(),
        )
        # An unrecorded membership means the full fleet -- see should_rebuild's docstring,
        # which owns the rest of this rule for both hardened transports.
        if self._built_membership is None:
            self._built_membership = desired_membership(
                absent_shards=[],
                dp_size=self._generation.worker_group.dp_size,
                total_gen_workers=len(self._generation.worker_group.workers),
                train_world_size=membership.train_world_size,
            )
        if not should_rebuild(
            desired=membership,
            built=self._built_membership,
            absent_shards=absent_shards,
            force=force,
        ):
            return False
        print(
            f"  refit: rebuilding nccl_reshard communicators over shards "
            f"{membership.surviving_shards}; gen world "
            f"{membership.world_size - membership.train_world_size}",
            flush=True,
        )
        self._build(membership)
        return True

    def shutdown(self) -> None:
        # The NCCL process groups' lifecycle is managed by Ray actor teardown;
        # the workers that own the groups are destroyed with the cluster.
        # Break the VllmGeneration <-> synchronizer reference cycle so the
        # generation wrapper is garbage-collectable after teardown. The
        # synchronizer is never used again after shutdown(), so losing the
        # handle is safe.
        self._generation = None
