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

"""``GenerationInterface`` implementation backed by TRT-LLM.

Non-colocated: separate train / inference GPU sets, NCCL broadcast for
weight sync. Colocated: shares GPUs with the policy and uses sleep/wakeup
to time-multiplex GPU memory between training and inference phases.
"""

import asyncio
import os
from collections import defaultdict
from typing import Any, AsyncGenerator, Optional, Union, cast

import numpy as np
import ray

from nemo_rl.distributed.batched_data_dict import BatchedDataDict, SlicedDataDict
from nemo_rl.distributed.named_sharding import NamedSharding
from nemo_rl.distributed.virtual_cluster import RayVirtualCluster
from nemo_rl.distributed.worker_groups import RayWorkerBuilder, RayWorkerGroup
from nemo_rl.models.generation.interfaces import (
    GenerationDatumSpec,
    GenerationInterface,
    GenerationOutputSpec,
)
from nemo_rl.models.generation.trtllm.config import TrtllmConfig


class TrtllmGeneration(GenerationInterface):
    """TRT-LLM generation backend (requires trtllm_cfg.async_engine=true)."""

    @staticmethod
    def init_cluster_placement_groups(
        cluster: RayVirtualCluster,
        config: TrtllmConfig,
    ) -> None:
        """Pre-initialize placement groups matching TRT-LLM's topology."""
        trtllm_cfg = config["trtllm_cfg"]
        disagg = trtllm_cfg.get("disaggregation") or {}
        engine_tp = trtllm_cfg["tensor_parallel_size"]
        if disagg.get("enabled"):
            engine_tp = max(
                int((disagg.get(f"{role}_trtllm_kwargs") or {}).get(
                    "tensor_parallel_size", engine_tp
                ))
                for role in ("ctx", "gen")
            )
        pp = trtllm_cfg.get("pipeline_parallel_size", 1)
        assert pp == 1, (
            "TRT-LLM backend does not support pipeline parallelism yet "
            f"(pipeline_parallel_size={pp}, must be 1)."
        )
        # GPUs held by the *widest single engine*. Deliberately not a replica's
        # width: this decides use_unified_pg, which exists so one tied worker
        # group can take bundles across nodes, and the tied group is an engine
        # (_get_tied_worker_bundle_indices slices per engine). A replica is not
        # a scheduling unit -- its engines are independent actor groups talking
        # over HTTP and the KV transceiver, so it may span nodes, and we only
        # soft-pin it for locality. Sizing this by the replica would push
        # node-local layouts (e.g. 4 x TP2 engines on 4-GPU nodes) into a
        # unified PG for nothing.
        widest_engine_gpus = engine_tp * pp
        colocated = bool(config.get("colocated", {}).get("enabled", False))

        # Colocated time-multiplexes the GPUs: engines sleep (dropping the whole
        # KV pool) while the policy trains. Disaggregation cannot survive that --
        # a replica's context and generation engines must be resident *at the
        # same time* for the transceiver to hand a prefilled cache over, and the
        # disagg servers hold HTTP connections to engines that would be asleep.
        # Reject the combination instead of hanging on the first request after a
        # sleep.
        assert not (disagg.get("enabled") and colocated), (
            "PD disaggregation requires non-colocated generation: colocated mode "
            "sleeps the engines between rollouts, which drops the KV cache the "
            "transceiver needs. Set colocated.enabled=false or "
            "trtllm_cfg.disaggregation.enabled=false."
        )

        needs_cross_node = widest_engine_gpus > cluster.num_gpus_per_node
        assert not (needs_cross_node and colocated), (
            "TRT-LLM cross-node tensor parallelism is only supported for "
            "non-colocated generation."
        )

        cluster._init_placement_groups(
            strategy=None if colocated else "PACK",
            use_unified_pg=needs_cross_node,
        )

    def __init__(
        self,
        cluster: RayVirtualCluster,
        config: TrtllmConfig,
        name_prefix: str = "trtllm_policy",
        workers_per_node: Optional[Union[int, list[int]]] = None,
    ):
        self.cfg = config
        self.tp_size = self.cfg["trtllm_cfg"]["tensor_parallel_size"]

        # Per-engine role and TP width, in engine order -- the single source of
        # truth for how the cluster is sliced. Without disaggregation every
        # engine is identical; under it, each replica contributes its context
        # engines followed by its generation engines.
        (
            self._engine_roles,
            self._engine_tps,
            self.num_replicas,
        ) = self._plan_engines(cluster.world_size())
        # DP keeps its general meaning -- the number of independent rollout
        # shards, i.e. replicas. It is deliberately *not* the engine count:
        # under disaggregation a replica is several engines, and overloading
        # "dp" to mean engines is what makes the two concepts collide.
        self.dp_size = self.num_replicas
        self.num_engines = len(self._engine_tps)

        # GPUs held by the widest single engine; see
        # init_cluster_placement_groups for why this is per engine, not per
        # replica.
        self.widest_engine_gpus = max(self._engine_tps)

        assert sum(self._engine_tps) == cluster.world_size(), (
            f"Engine layout needs {sum(self._engine_tps)} GPUs "
            f"({self._engine_tps}) but the cluster has {cluster.world_size()}."
        )

        # MoE: TRT-LLM partitions TP on MoE layers into moe_tp x moe_ep, so the
        # product must equal that engine's TP width. Validate here to fail fast
        # -- the LLM constructor would otherwise raise a less actionable error
        # deep inside the engine. Under disaggregation this is per role, since
        # both TP and the MoE split can differ between prefill and decode.
        if self._disagg_cfg.get("enabled"):
            engine_configs = [
                (f"{role}_trtllm_kwargs", self._role_kwargs(role))
                for role in ("ctx", "gen")
            ]
        else:
            engine_configs = [("trtllm_cfg", self.cfg["trtllm_cfg"])]

        for label, kwargs in engine_configs:
            m_tp = kwargs.get("moe_tensor_parallel_size")
            m_ep = kwargs.get("moe_expert_parallel_size")
            if m_tp is None and m_ep is None:
                continue
            product = (m_tp or 1) * (m_ep or 1)
            engine_tp = int(kwargs["tensor_parallel_size"])
            assert product == engine_tp, (
                f"{label}: moe_tensor_parallel_size ({m_tp}) * "
                f"moe_expert_parallel_size ({m_ep}) = {product} must equal "
                f"tensor_parallel_size ({engine_tp})."
            )

        missing_keys = [k for k in TrtllmConfig.__required_keys__ if k not in self.cfg]
        if "model_name" not in self.cfg:
            missing_keys.append("model_name")
        assert not missing_keys, f"TrtllmConfig missing keys: {missing_keys}"

        # [data_parallel, tensor_parallel] = [replicas, GPUs per replica]. A
        # replica is the unit data is actually parallelised over -- one rollout
        # shard, one URL -- and its GPU span tiles the cluster exactly, so the
        # grid is rectangular by construction with no special case for
        # asymmetric per-role TP. Without disaggregation a replica is a single
        # engine, so this is the usual arange(world).reshape(dp, tp).
        #
        # Under disaggregation the second axis spans a whole replica, not one
        # engine's TP group, so get_axis_size("tensor_parallel") is not any
        # engine's TP width -- read self._engine_tps for that. Nothing
        # dispatches on the axis: per-engine fan-out goes through
        # _run_on_engines, and the two grid-driven worker-group calls sit
        # behind _assert_direct_dispatch_allowed, which disaggregation rejects.
        world_size = sum(self._engine_tps)
        self.sharding_annotations = NamedSharding(
            layout=np.arange(world_size).reshape(self.num_replicas, -1),
            names=["data_parallel", "tensor_parallel"],
        )

        self.colocated_enabled = bool(self.cfg["colocated"]["enabled"])
        self.async_engine = bool(self.cfg["trtllm_cfg"].get("async_engine", False))
        # One disagg server actor and one URL per replica; stay empty unless
        # disaggregation is on. See _start_disagg_servers.
        self._disagg_actors: list[ray.actor.ActorHandle] = []
        self._disagg_server_urls: list[Optional[str]] = []
        # The synchronous TRT-LLM engine path is no longer supported: only the
        # async worker wires up colocated sleep/wakeup, IPC-ZMQ refit, and
        # per-sample streaming. Fail loudly at setup rather than silently
        # running a half-supported path.
        assert self.async_engine, (
            "TRT-LLM backend requires trtllm_cfg.async_engine=true; the "
            "synchronous engine path (async_engine=false) is no longer supported."
        )

        self.init_cluster_placement_groups(cluster, config)

        # Engines differ from one another only under disaggregation, so the
        # explicit bundle list (which fixes engine order) is only required
        # there; a uniform run keeps the original workers_per_node path.
        use_explicit_bundles = (
            self.widest_engine_gpus > 1 or self._disagg_cfg.get("enabled")
        )
        node_bundle_indices = (
            self._get_tied_worker_bundle_indices(cluster)
            if use_explicit_bundles
            else None
        )
        # Global rank of each engine's model owner, in engine order -- the list
        # every per-engine call is dispatched on (see _run_on_engines). The
        # worker group hands bundle_indices (the "you build the AsyncLLM"
        # signal) to each tied group's local rank 0 and assigns ranks in bundle
        # order, so the owners are the prefix sums of the tied group widths.
        # Derived from the very bundle split handed to the worker group so
        # engine fan-out cannot drift from it; the worker group's DP leaders are
        # a strict subset of these (one per replica) and stop coinciding with
        # them under disaggregation, which is why engine fan-out keeps its own
        # list.
        engine_widths = (
            [len(indices) for _, indices in node_bundle_indices]
            if node_bundle_indices is not None
            # No explicit bundles means every engine is one worker of its own.
            else [1] * self.num_engines
        )
        assert engine_widths == self._engine_tps, (
            f"Bundle split gives engine widths {engine_widths} but the engine "
            f"layout expects {self._engine_tps}; every per-engine kwarg would "
            "land on the wrong engine."
        )
        self._engine_owner_indices: list[int] = []
        offset = 0
        for width in engine_widths:
            self._engine_owner_indices.append(offset)
            offset += width

        # Owner of each replica's first engine: the DP leaders the worker group
        # would otherwise derive per engine, which is only right while a replica
        # is a single engine.
        self._replica_owner_indices = self._engine_owner_indices[
            :: self.num_engines // self.num_replicas
        ]

        worker_cls = "nemo_rl.models.generation.trtllm.trtllm_worker_async.TrtllmAsyncGenerationWorker"
        worker_builder = RayWorkerBuilder(
            worker_cls, self._config_with_engine_overrides(node_bundle_indices)
        )

        # NCCL_CUMEM_ENABLE=1 is needed for the non-colocated NCCL collective
        # broadcast; colocated shares the policy's NCCL group so don't touch it.
        env_vars: dict[str, str] = {}
        if not self.colocated_enabled:
            env_vars["NCCL_CUMEM_ENABLE"] = "1"

        if node_bundle_indices is not None:
            self.worker_group = RayWorkerGroup(
                cluster,
                worker_builder,
                name_prefix=name_prefix,
                bundle_indices_list=node_bundle_indices,
                sharding_annotations=self.sharding_annotations,
                env_vars=env_vars,
                dp_leader_worker_indices=self._replica_owner_indices,
            )
        else:
            self.worker_group = RayWorkerGroup(
                cluster,
                worker_builder,
                name_prefix=name_prefix,
                workers_per_node=workers_per_node,
                sharding_annotations=self.sharding_annotations,
                env_vars=env_vars,
            )

        # post-init on workers (starts HTTP server when expose_http_server=true,
        # finishes async engine setup for the async worker variant).
        post_init_method = "post_init_async"
        futures = self._run_on_engines(
            post_init_method,
        )
        ray.get(futures)

        # Round-robin DP shard used by generate_async for per-sample dispatch.
        self.current_generate_dp_shard_idx = 0

        self.dp_openai_server_base_urls = self._report_dp_openai_server_base_urls()

        self.device_uuids = self._report_device_id()

        # DP is replicas, not engines: only equal when a replica is one engine.
        assert self.dp_size == self.worker_group.dp_size, (
            f"Replica count {self.dp_size} does not match the worker group's DP "
            f"size {self.worker_group.dp_size}."
        )

    # ------------------------------------------------------------------ #
    #  Engine layout
    # ------------------------------------------------------------------ #

    def _role_kwargs(self, role: str) -> dict[str, Any]:
        """This role's engine overrides, merged over the base ``trtllm_cfg``.

        Any TRT-LLM kwarg may be overridden per role; TP and the MoE split are
        the ones that usually differ between prefill and decode.
        """
        overrides = self._disagg_cfg.get(f"{role}_trtllm_kwargs") or {}
        return {**self.cfg["trtllm_cfg"], **overrides}

    def _plan_engines(self, world_size: int) -> tuple[list[str], list[int], int]:
        """Per-engine ``(role, tp_width)``, in engine order.

        Without disaggregation every engine is a plain generation engine of
        ``tensor_parallel_size`` GPUs. Under disaggregation each replica
        contributes its context engines followed by its generation engines:

            [ctx_tp x M, gen_tp x K,   ctx_tp x M, gen_tp x K, ...]
             \\______ replica 0 _____/  \\____ replica 1 ...

        The replica *count* is derived, not configured: it follows from the
        cluster size, the same way the DP-shard count does without
        disaggregation. Same-role engines stay contiguous so each role's GPUs
        are adjacent.
        """
        disagg = self._disagg_cfg
        if not disagg.get("enabled"):
            assert world_size % self.tp_size == 0, (
                f"Cluster world_size ({world_size}) must be divisible by "
                f"TP size ({self.tp_size})."
            )
            n = world_size // self.tp_size
            # Without disaggregation a replica *is* an engine: one DP shard,
            # one URL, nothing below it to front. Reporting n rather than 1
            # keeps "replica" meaning the same thing on both paths.
            return ["generation"] * n, [self.tp_size] * n, n

        ctx_tp = int(self._role_kwargs("ctx")["tensor_parallel_size"])
        gen_tp = int(self._role_kwargs("gen")["tensor_parallel_size"])
        for role, value in (("ctx", ctx_tp), ("gen", gen_tp)):
            assert value >= 1, f"{role}_trtllm_kwargs.tensor_parallel_size must be >= 1"

        num_ctx = int(disagg["num_context_engines"])
        num_gen = int(disagg["num_generation_engines"])
        assert num_ctx >= 1 and num_gen >= 1, (
            f"a replica needs at least one engine of each role, got "
            f"num_context_engines={num_ctx}, num_generation_engines={num_gen}"
        )

        replica_width = num_ctx * ctx_tp + num_gen * gen_tp
        assert world_size % replica_width == 0, (
            f"replica width {replica_width} GPUs "
            f"({num_ctx} ctx x TP{ctx_tp} + {num_gen} gen x TP{gen_tp}) does not "
            f"divide the {world_size} inference GPUs."
        )
        num_replicas = world_size // replica_width

        roles = (["context"] * num_ctx + ["generation"] * num_gen) * num_replicas
        tps = ([ctx_tp] * num_ctx + [gen_tp] * num_gen) * num_replicas
        return roles, tps, num_replicas

    def _config_with_engine_overrides(
        self, node_bundle_indices: Optional[list[tuple[int, list[int]]]]
    ) -> TrtllmConfig:
        """Attach each engine's role and construction overrides to the worker config.

        Engine parallelism is fixed when ``AsyncLLM`` is built, and a worker
        cannot infer its own values from its TP width -- two roles may share a
        width and still want different expert layouts. ``RayWorkerBuilder`` hands
        every worker the same config, so the driver attaches a map keyed by
        :meth:`_engine_key` and each worker looks up its own entry.

        The context/generation *role* is per request in TRT-LLM, so an engine
        does not strictly need to know it; it is recorded anyway because the
        transceiver config and the role's kwargs are chosen from it.
        """
        if node_bundle_indices is None or not self._disagg_cfg.get("enabled"):
            return self.cfg

        overrides: dict[str, dict[str, Any]] = {}
        role_counts: dict[str, int] = {}
        for (pg_idx, bundles), role in zip(
            node_bundle_indices, self._engine_roles, strict=True
        ):
            prefix = "ctx" if role == "context" else "gen"
            # Ordinal within the role, so a layout with several engines of one
            # role (CTX_ENGINES=2, or more than one replica) can still tell them
            # apart. Only consumers that need a stable per-engine name use it --
            # the nsys report filename, so far.
            ordinal = role_counts.get(role, 0)
            role_counts[role] = ordinal + 1
            # Every engine gets an entry: the worker treats a missing key as a
            # driver/worker mismatch, so absence must never read as "no
            # overrides for this engine".
            overrides[self._engine_key(pg_idx, bundles)] = {
                "_disagg_role": role,
                "_disagg_role_ordinal": ordinal,
                **(self._disagg_cfg.get(f"{prefix}_trtllm_kwargs") or {}),
            }

        cfg = dict(self.cfg)
        cfg["trtllm_cfg"] = {**self.cfg["trtllm_cfg"], "_engine_overrides": overrides}
        return cast(TrtllmConfig, cfg)

    @staticmethod
    def _engine_key(pg_idx: int, local_bundle_indices: list[int]) -> str:
        """Stable id for the engine occupying these bundles.

        Both sides of the worker boundary compute this from the same
        ``(pg_idx, local_bundle_indices)`` tuple, so the worker can look up its
        own per-engine overrides without the driver needing per-worker init
        kwargs. Keying on the tuple rather than deriving a global ordinal keeps
        it correct for both the unified-PG and per-node-PG layouts, whose local
        bundle indices mean different things.
        """
        return f"{pg_idx}:" + ",".join(str(i) for i in local_bundle_indices)

    # ------------------------------------------------------------------ #
    #  Placement helpers (simplified from VllmGeneration)
    # ------------------------------------------------------------------ #

    def _get_tied_worker_bundle_indices(
        self,
        cluster: RayVirtualCluster,
    ) -> list[tuple[int, list[int]]]:
        """Calculate bundle indices for tensor-parallel worker groups.

        Handles both unified placement groups (cross-node model parallelism) and
        per-node placement groups (node-local model parallelism). For unified
        PGs, bundles are reordered by physical node before slicing so each TP
        group stays as node-local as possible.

        Bundles are consumed engine by engine following ``self._engine_tps``, so
        engines of differing width (PD disaggregation with asymmetric TP) each
        get exactly their own number of bundles.
        """
        placement_groups = cluster.get_placement_groups()
        if not placement_groups:
            raise ValueError("No placement groups available in the cluster")

        engine_tps = self._engine_tps

        if len(placement_groups) == 1:
            # Single unified PG: TP > GPUs/node, so model parallelism may span
            # nodes. Reorder bundles by physical node so consecutive indices in
            # `flat` belong to the same node — keeps TP siblings co-located
            # when TP <= GPUs/node and only crosses node boundaries when forced.
            unified_pg = placement_groups[0]
            try:
                pg_table = ray.util.placement_group_table(unified_pg)
                bundle_to_node = pg_table["bundles_to_node_id"]
            except Exception as e:
                raise RuntimeError(
                    "Failed to retrieve bundle/node mapping from placement group"
                ) from e

            node_bundles: dict[str, list[int]] = defaultdict(list)
            for bundle_idx, node_id in bundle_to_node.items():
                node_bundles[node_id].append(bundle_idx)
            for bundles in node_bundles.values():
                bundles.sort()

            if not node_bundles:
                raise ValueError("Placement group contains no bundles")

            counts = [len(b) for b in node_bundles.values()]
            assert len(set(counts)) == 1, "All nodes must have identical bundle counts"

            # RayVirtualCluster records the physical-node bundle order when it
            # builds a unified PG. Preserve it so TP replicas occupy contiguous
            # nodes in the topology-aware order selected by the cluster.
            flat = list(cluster._sorted_bundle_indices or [])
            if not flat:
                for nid in sorted(node_bundles):
                    flat.extend(node_bundles[nid])

            if len(flat) < sum(engine_tps):
                raise ValueError(
                    f"Engine layout needs {sum(engine_tps)} bundles but the "
                    f"unified placement group has {len(flat)}."
                )

            tied_groups: list[tuple[int, list[int]]] = []
            cursor = 0
            for tp in engine_tps:
                # The first value is a placement-group index.
                # A unified cluster has exactly one PG (index 0).
                tied_groups.append((0, flat[cursor : cursor + tp]))
                cursor += tp
        else:
            tied_groups = []
            engine_idx = 0
            # Consume placement groups in topology order, not creation order. Ray
            # picks the physical node behind each per-node PG, so consecutive
            # pg_idx values can land in different NVLink domains -- and engines
            # are laid out replica by replica, which would put a replica's
            # context and generation engines on opposite sides of the fabric and
            # push its KV handoff onto InfiniBand. Only the *order* changes;
            # every PG is still used exactly once.
            for pg_idx in cluster.get_topology_sorted_pg_indices():
                pg = placement_groups[pg_idx]
                if pg.bundle_count == 0:
                    continue
                cursor = 0
                while engine_idx < len(engine_tps):
                    tp = engine_tps[engine_idx]
                    if cursor + tp > pg.bundle_count:
                        break
                    tied_groups.append((pg_idx, list(range(cursor, cursor + tp))))
                    cursor += tp
                    engine_idx += 1
                if cursor != pg.bundle_count:
                    # An engine may not straddle a placement group (== node):
                    # TRT-LLM does not support cross-node TP, and a partially
                    # filled node would leave GPUs idle while the engine count
                    # silently drops.
                    raise ValueError(
                        f"Engine widths {engine_tps} do not tile placement group "
                        f"{pg_idx} ({pg.bundle_count} bundles): {cursor} bundles "
                        f"used. Choose per-role TP sizes whose group total "
                        f"divides the GPUs per node."
                    )

        if not tied_groups:
            raise ValueError(
                "Unable to allocate any worker groups with the available resources."
            )
        return tied_groups

    # ------------------------------------------------------------------ #
    #  PD disaggregation
    # ------------------------------------------------------------------ #

    @property
    def _disagg_cfg(self) -> dict[str, Any]:
        return self.cfg["trtllm_cfg"].get("disaggregation") or {}

    def _assert_direct_dispatch_allowed(self) -> None:
        """Reject the token-in-token-out path while PD is enabled.

        ``generate`` / ``generate_async`` round-robin straight to DP leaders,
        bypassing the group entry points. That would still produce correct
        tokens -- each engine can prefill and decode on its own -- but it would
        run *without* disaggregation while the user believes they are
        exercising it. Fail loudly instead.
        """
        if self._disagg_cfg.get("enabled"):
            raise RuntimeError(
                "PD disaggregation is only wired for the HTTP/NeMo-Gym rollout "
                "path; TrtllmGeneration.generate()/generate_async() dispatch "
                "directly to engines and would silently bypass it. Use the "
                "NeMo-Gym entrypoint, or set "
                "trtllm_cfg.disaggregation.enabled=false."
            )

    def _start_disagg_servers(self) -> list[Optional[str]]:
        """Start one disagg server per replica and return their URLs.

        Every engine already runs its own HTTP server; an address is the only
        thing a disagg server is given about one. This slices the engine list
        into replicas, hands each disagg server its two address pools, and
        collects the single URL it exposes. Those URLs are what NeMo-Gym sees.

        Starting a server is not idempotent, so a second call returns the URLs
        of the servers already running instead of doubling them.
        """
        if self._disagg_server_urls:
            return self._disagg_server_urls

        disagg = self._disagg_cfg
        num_ctx = int(disagg["num_context_engines"])
        num_gen = int(disagg["num_generation_engines"])
        per_replica = num_ctx + num_gen

        assert self.cfg["trtllm_cfg"].get("expose_http_server"), (
            "PD disaggregation requires trtllm_cfg.expose_http_server=true: an "
            "address is the only way the disagg server can reach an engine."
        )

        addrs = self._report_engine_addrs()
        missing = [i for i, a in enumerate(addrs) if not a]
        assert not missing, f"engines {missing} reported no HTTP address"

        from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

        from nemo_rl.models.generation.trtllm.trtllm_disagg_server import (
            DisaggServerActor,
        )

        self._disagg_actors = []
        futures = []
        for replica_idx in range(self.num_replicas):
            base = replica_idx * per_replica
            # Its own CPU-only actor rather than an engine worker's process:
            # this is the request hot path for the whole replica, and sharing a
            # process with an engine would couple the replica's routing latency
            # to that one engine's load. Soft-pinned to the node holding its
            # first context engine so routing hops stay local when they can.
            actor = DisaggServerActor.options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node_id=addrs[base]["node_id"], soft=True
                ),
                name=f"trtllm_disagg_server_{replica_idx}",
                # The server imports tensorrt_llm (OpenAIDisaggServer,
                # disagg_utils), which only exists in the engine workers' venv.
                # Without this the actor starts in the driver's environment and
                # dies with ModuleNotFoundError: No module named 'tensorrt_llm'.
                # The venv already exists on every node -- the worker group
                # created it before these actors are spawned.
                runtime_env={"py_executable": self.worker_group.py_executable},
            ).remote(replica_idx)
            self._disagg_actors.append(actor)

            futures.append(
                actor.start.remote(
                    ctx_addrs=[
                        (a["host"], a["port"]) for a in addrs[base : base + num_ctx]
                    ],
                    gen_addrs=[
                        (a["host"], a["port"])
                        for a in addrs[base + num_ctx : base + per_replica]
                    ],
                    ctx_router=disagg["ctx_router"],
                    gen_router=disagg["gen_router"],
                )
            )

        self._disagg_server_urls = ray.get(futures)
        print(
            f"  ✓ PD disaggregation: {self.num_replicas} replica(s) x "
            f"({num_ctx} context + {num_gen} generation) engines; "
            f"disagg servers: {self._disagg_server_urls}",
            flush=True,
        )
        return self._disagg_server_urls

    def _run_on_engines(
        self,
        method_name: str,
        per_engine: Optional[dict[str, list[Any]]] = None,
        **common: Any,
    ) -> list[ray.ObjectRef]:
        """Fan out to each engine's model owner, one call per engine.

        Dispatches by :attr:`_engine_owner_indices`, computed once from the
        bundle split. Neither the sharding grid's ``run_rank_0_only_axes`` gate
        nor the worker group's DP-leader list is used: the grid cannot model
        engines of differing width, and the DP leaders are one worker per
        *replica*, which under disaggregation skips every engine but the
        replica's first. This keeps one source of truth for who owns an engine.

        Args:
            per_engine: kwargs whose value is a per-engine list, in engine order.
            **common: kwargs passed unchanged to every engine.
        """
        futures = []
        for engine_idx, worker_idx in enumerate(self._engine_owner_indices):
            kwargs = dict(common)
            for key, values in (per_engine or {}).items():
                kwargs[key] = values[engine_idx]
            futures.append(
                self.worker_group.run_single_worker_single_data(
                    method_name=method_name, worker_idx=worker_idx, **kwargs
                )
            )
        return futures

    def _report_engine_addrs(self) -> list[Optional[dict[str, Any]]]:
        """Collect ``{host, port, node_id}`` from every engine's HTTP server.

        ``None`` for an engine whose server never started. The node id is what
        lets the replica's disagg server be pinned beside its engines.
        """
        futures = self._run_on_engines(
            "report_http_addr",
        )
        return ray.get(futures)

    def _report_dp_openai_server_base_urls(self) -> list[Optional[str]]:
        """One OpenAI-compatible base URL per DP shard, in shard order.

        A shard's entry point is not always an engine: under disaggregation the
        caller must talk to the replica's disagg server, which routes prefill and
        decode itself. Branching here rather than at the call site keeps the
        contract "one URL per DP shard" true on both paths -- an engine-per-URL
        list would have ``num_engines`` entries, which is no longer ``dp_size``.
        """
        if self._disagg_cfg.get("enabled"):
            return self._start_disagg_servers()
        if not self.cfg["trtllm_cfg"].get("expose_http_server"):
            return [cast(Optional[str], None)] * self.dp_size
        futures = self._run_on_engines(
            "report_dp_openai_server_base_url",
        )
        return ray.get(futures)

    def _report_device_id(self) -> list[list[str]]:
        futures = self._run_on_engines(
            "report_device_id_async",
        )
        return ray.get(futures)

    # ------------------------------------------------------------------ #
    #  GenerationInterface
    # ------------------------------------------------------------------ #

    def init_collective(
        self,
        ip: str,
        port: int,
        world_size: int,
        *,
        train_world_size: int,
    ) -> list[ray.ObjectRef]:
        if not self.worker_group or not self.worker_group.workers:
            raise RuntimeError("Worker group not initialised")

        # The caller sizes the group from the cluster config while the ranks
        # below come from the engine layout. If the two disagree the group never
        # fills up and StatelessProcessGroup's TCPStore blocks forever, so check
        # it here rather than debugging a hang at startup.
        inference_world_size = sum(self._engine_tps)
        assert train_world_size + inference_world_size == world_size, (
            f"Collective world_size {world_size} != train {train_world_size} + "
            f"inference {inference_world_size} (engine widths {self._engine_tps})."
        )

        # rank_prefix is each engine's first rank inside the inference half of
        # the group; the engine's own TP ranks follow contiguously. Not a fixed
        # stride -- with asymmetric TP the engines are not equally wide, and a
        # wrong offset would map generation ranks onto the wrong slice of the
        # training-side broadcast, silently loading the wrong weights.
        return self._run_on_engines(
            "init_collective_async",
            per_engine={"rank_prefix": self._engine_owner_indices},
            ip=ip,
            port=port,
            world_size=world_size,
            train_world_size=train_world_size,
        )

    def generate(
        self,
        data: BatchedDataDict[GenerationDatumSpec],
        greedy: bool = False,
    ) -> BatchedDataDict[GenerationOutputSpec]:
        self._assert_direct_dispatch_allowed()
        assert isinstance(data, BatchedDataDict)
        assert "input_ids" in data and "input_lengths" in data

        dp_size = self.dp_size
        sharded_data = cast(
            list[SlicedDataDict],
            data.shard_by_batch_size(
                dp_size,
                allow_uneven_shards=True,
            ),
        )
        future_bundle = self.worker_group.run_all_workers_sharded_data(
            "generate_async",
            data=sharded_data,
            in_sharded_axes=["data_parallel"],
            replicate_on_axes=None,
            output_is_replicated=None,
            common_kwargs={"greedy": greedy},
        )
        results = self.worker_group.get_all_worker_results(future_bundle)

        combined: BatchedDataDict[GenerationOutputSpec] = BatchedDataDict.from_batches(
            results,
            pad_value_dict={"output_ids": self.cfg["_pad_token_id"]},
        )

        required = [
            "output_ids",
            "generation_lengths",
            "unpadded_sequence_lengths",
            "logprobs",
        ]
        missing = [k for k in required if k not in combined]
        if missing:
            raise ValueError(f"Missing generation output keys: {missing}")
        return combined

    async def generate_async(
        self,
        data: BatchedDataDict[GenerationDatumSpec],
        greedy: bool = False,
    ) -> AsyncGenerator[tuple[int, BatchedDataDict[GenerationOutputSpec]], None]:
        """Yield a single-sample generation result.

        Called by run_async_multi_turn_rollout, which dispatches one sample at
        a time per coroutine. The async worker's max_concurrency lets multiple
        in-flight Ray calls share the same AsyncLLM, which batches them
        internally via asyncio.gather.
        """
        self._assert_direct_dispatch_allowed()
        if "input_ids" not in data or "input_lengths" not in data:
            raise AssertionError(
                "input_ids and input_lengths are required in data for generate_async"
            )
        if len(data["input_ids"]) == 0:
            return
        assert data.size == 1, (
            f"generate_async expects single-sample data, got batch_size={data.size}."
        )

        leader_worker_idx = self._engine_owner_indices[
            self.current_generate_dp_shard_idx
        ]
        worker_result_ref = self.worker_group.run_single_worker_single_data(
            method_name="generate_async",
            worker_idx=leader_worker_idx,
            data=data,
            greedy=greedy,
        )
        self.current_generate_dp_shard_idx = (
            self.current_generate_dp_shard_idx + 1
        ) % self.num_engines

        timeout_seconds = float(
            os.environ.get("NRL_TRTLLM_ASYNC_TIMEOUT_SECONDS", "900")
        )
        try:
            result = await asyncio.wait_for(worker_result_ref, timeout=timeout_seconds)
        except asyncio.TimeoutError:
            raise RuntimeError(
                f"TRT-LLM async generation timed out after {timeout_seconds}s. "
                f"Tune with NRL_TRTLLM_ASYNC_TIMEOUT_SECONDS."
            )

        result["gen_leader_worker_idx"] = [int(leader_worker_idx)]
        # Worker.generate_async returns a single-sample BatchedDataDict; idx in
        # the input batch is always 0 (caller already split per-sample).
        yield (0, result)

    def prepare_for_generation(self, *args: Any, **kwargs: Any) -> bool:
        """Wake inference workers up. No-op for non-colocated."""
        if not self.colocated_enabled:
            return True
        try:
            futures = self._run_on_engines(
                "wake_up_async",
                **kwargs,
            )
            results = ray.get(futures)
            return all(r for r in results if r is not None)
        except Exception as e:
            print(f"Error in prepare_for_generation: {e}")
            return False

    def finish_generation(self, *args: Any, **kwargs: Any) -> bool:
        """Sleep workers (colocated) or reset prefix cache (non-colocated)."""
        try:
            if self.colocated_enabled:
                method_name = "sleep_async"
            else:
                method_name = "reset_prefix_cache_async"
            futures = self._run_on_engines(
                method_name,
            )
            results = ray.get(futures)
            return all(r for r in results if r is not None)
        except Exception as e:
            print(f"Error in finish_generation: {e}")
            return False

    def prepare_refit_info(self, state_dict_info: dict[str, Any]) -> None:
        futures = self._run_on_engines(
            "prepare_refit_info_async",
            state_dict_info=state_dict_info,
        )
        ray.get(futures)

    def start_gpu_profiling(self) -> None:
        """Grpo profiling protocol: start nsys capture on the GPU workers."""
        if not self.worker_group or not self.worker_group.workers:
            return
        futures = self._run_on_engines(
            "start_gpu_profiling_async" if self.async_engine else "start_gpu_profiling",
        )
        ray.get(futures)

    def stop_gpu_profiling(self) -> None:
        """Grpo profiling protocol: stop nsys capture on the GPU workers."""
        if not self.worker_group or not self.worker_group.workers:
            return
        futures = self._run_on_engines(
            "stop_gpu_profiling_async" if self.async_engine else "stop_gpu_profiling",
        )
        ray.get(futures)

    def update_weights_from_collective(self) -> list[ray.ObjectRef]:
        if not self.worker_group or not self.worker_group.workers:
            raise RuntimeError("Worker group not initialised")
        trtllm_cfg = self.cfg["trtllm_cfg"]
        in_flight = bool(trtllm_cfg.get("in_flight_weight_updates"))
        recompute_kv = bool(trtllm_cfg.get("recompute_kv_cache_after_weight_updates"))
        return self._run_on_engines(
            "update_weights_from_collective_async",
            drain=not in_flight,
            recompute_kv=recompute_kv,
        )

    def update_weights_via_ipc_zmq(self) -> list[ray.ObjectRef]:
        """Receive weights via CUDA-IPC + ZMQ (colocated mode)."""
        if not self.worker_group or not self.worker_group.workers:
            raise RuntimeError("Worker group not initialised")
        return self._run_on_engines(
            "update_weights_via_ipc_zmq_async",
        )

    def invalidate_kv_cache(self) -> bool:
        """No-op for TRT-LLM: KV-cache invalidation happens inside the refit path.

        For async RL correctness, KV/prefix-cache invalidation must happen in
        the same engine step boundary as the weight update — otherwise
        in-flight requests forward several decode steps with new weights ×
        old KV, opening a race window.

        TRT-LLM avoids this by performing the invalidation inside the refit
        function itself, under the same ``control_action`` context:
          * ``NcclExtension.update_weights_from_collective`` (NCCL path)
          * ``NcclExtension.update_weights_via_ipc_zmq`` (IPC-ZMQ path)
        """
        return True

    def clear_logger_metrics(self) -> None:
        """No-op: TRT-LLM rollout telemetry is not yet wired up.

        vLLM overrides this (``vllm_generation.py``) to reset AsyncLLM
        iteration stats (rollout throughput, KV-cache utilization, in-flight
        batch counters). TRT-LLM has no equivalent plumbing yet, so this stays
        a no-op rather than inheriting silently — making the gap explicit.

        TODO: fetch AsyncLLM iteration stats from the TRT-LLM workers and
        clear them here, mirroring ``clear_vllm_logger_metrics``.
        """
        return None

    def get_logger_metrics(self) -> dict[str, Any]:
        """Return empty metrics: TRT-LLM rollout telemetry is not yet wired up.

        See :meth:`clear_logger_metrics`. Returning ``{}`` explicitly documents
        that rollout observability metrics (present for vLLM) are absent for
        TRT-LLM runs, rather than letting them silently disappear via the
        inherited no-op default.

        TODO: surface AsyncLLM iteration stats here, mirroring
        ``get_vllm_logger_metrics``.
        """
        return {}

    def shutdown(self) -> bool:
        try:
            # Stop routing before the engines go away, so in-flight requests are
            # not handed to an endpoint that is already tearing down.
            for actor in getattr(self, "_disagg_actors", []):
                try:
                    ray.get(actor.shutdown.remote(), timeout=30)
                except Exception as e:
                    print(f"Error stopping disagg server: {e}")
                ray.kill(actor)
            self._disagg_actors = []
            self._disagg_server_urls = []

            return self.worker_group.shutdown(cleanup_method="shutdown")
        except Exception as e:
            print(f"Error during TRT-LLM shutdown: {e}")
            return False

    def __del__(self) -> None:
        self.shutdown()
