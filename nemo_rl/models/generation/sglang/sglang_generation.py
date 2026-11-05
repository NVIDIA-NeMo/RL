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

import asyncio
import logging
import os
from collections.abc import Mapping
from typing import Any, AsyncGenerator, Optional

import ray
import torch
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import (
    DEFAULT_GENERATION_PORT_RANGE_HIGH,
    DEFAULT_GENERATION_PORT_RANGE_LOW,
    RayVirtualCluster,
    get_reordered_bundle,
)
from nemo_rl.distributed.worker_group_utils import get_nsight_config_if_pattern_matches
from nemo_rl.models.generation.interfaces import (
    GenerationDatumSpec,
    GenerationInterface,
    GenerationOutputSpec,
    reject_unenforceable_refit_deadline,
    verify_right_padding,
)
from nemo_rl.models.generation.sglang.config import (
    SGLangConfig,
    SGLangDisaggConfig,
    SGLangFaultToleranceConfig,
    resolve_sglang_disagg_config,
    resolve_sglang_disagg_layout,
    resolve_sglang_role_overrides,
    sglang_disagg_server_kwargs,
    sglang_engine_width,
    sglang_role_engine_config,
    validate_sglang_disaggregation,
)
from nemo_rl.models.generation.sglang.fault_tolerance import RolloutHealthMonitor
from nemo_rl.models.generation.sglang.sglang_router import _start_router
from nemo_rl.models.generation.sglang.sglang_worker import SGLangGenerationWorker
from nemo_rl.models.generation.sglang.utils.async_utils import AsyncLoopThread
from nemo_rl.models.generation.sglang.utils.disagg_dispatch import ReplicaDispatcher
from nemo_rl.models.generation.sglang.utils.http_utils import HttpClient
from nemo_rl.models.generation.sglang.utils.ip_port_utils import (
    _allocate_disagg_engine_addr_and_ports,
    _allocate_rollout_engine_addr_and_ports_normal,
    _format_v6_uri,
)
from nemo_rl.models.generation.sglang.utils.layout import (
    SGLangEngineSpec,
    plan_sglang_engines,
)
from nemo_rl.models.generation.sglang.utils.ray_utils import (
    NOSET_VISIBLE_DEVICES_ENV_VARS_LIST,
    Lock,
)
from nemo_rl.utils.nsys import wrap_with_nvtx_name
from nemo_rl.utils.venvs import make_actor_runtime_env
from nemo_rl.weight_sync.interfaces import WeightSynchronizer

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

# Present when the nvidia-peermem kernel module is loaded, which Mooncake's
# default GPU-memory registration (ibv_reg_mr) relies on.
NVIDIA_PEERMEM_MODULE = "/sys/module/nvidia_peermem"


def disagg_transport_env(
    pd_cfg: SGLangDisaggConfig,
    environ: Mapping[str, str],
    *,
    peermem_loaded: bool | None = None,
) -> dict[str, str]:
    """Engine environment the configured KV transport needs on this host.

    A value already present in ``environ`` is kept, so an operator can override
    either default.

    * ``nixl``: ``NCCL_NET_PLUGIN=none``. HPC-X's NCCL plugin loads a UCX
      without CUDA support into TP>1 engines, which breaks NIXL's VRAM
      registration; the same default the vLLM NIXL path uses.
    * ``mooncake``: ``WITH_NVIDIA_PEERMEM=0`` when nvidia-peermem is not
      loaded. Mooncake then registers GPU memory for RDMA through DMA-BUF
      instead of ``ibv_reg_mr``, which without the module fails with
      "Failed to register memory ...: Bad address" on every engine (GB200
      containers, for one) and turns every KV transfer into an error.

    Args:
        pd_cfg: The resolved ``sglang_cfg.disagg`` block.
        environ: The process environment to take overrides from.
        peermem_loaded: Whether nvidia-peermem is loaded; probed from
            :data:`NVIDIA_PEERMEM_MODULE` when None.
    """
    env: dict[str, str] = {}
    if pd_cfg.transfer_backend == "nixl":
        env["NCCL_NET_PLUGIN"] = environ.get("NCCL_NET_PLUGIN", "none")
    elif pd_cfg.transfer_backend == "mooncake":
        if peermem_loaded is None:
            peermem_loaded = os.path.exists(NVIDIA_PEERMEM_MODULE)
        if not peermem_loaded:
            env["WITH_NVIDIA_PEERMEM"] = environ.get("WITH_NVIDIA_PEERMEM", "0")
    return env


class SGLangGeneration(GenerationInterface):
    """The class to run rollout and convert rollout data to training data.

    This class owns the full rollout server topology: the placement group,
    the router subprocess, and every ``SGLangGenerationWorker`` Ray actor.
    The former ``ServerGroup`` dataclass has been folded in so there is a
    single source of truth for engine state.

    Prefill/decode disaggregation (``generation.disaggregation``) lays the
    engines out as replicas of M prefill plus K decode engines
    (:mod:`utils.layout`), starts the sglang router in PD mode, one per replica
    or one for the fleet (``sglang_cfg.disagg.router_scope``), and spreads the
    rollout's requests over the routers (:mod:`utils.disagg_dispatch`). The
    router is the disagg server: it pairs each request with one prefill and
    one decode engine and relays the decode output, so the rollout keeps
    posting token ids to ``/generate`` exactly as without disaggregation.
    """

    def __init__(
        self,
        cluster: RayVirtualCluster,
        sglang_cfg: SGLangConfig,
    ):
        # __del__ also runs when validation or resource allocation fails.
        # Establish inert cleanup state before either can raise.
        self._health_monitor: RolloutHealthMonitor | None = None
        # Set by ``grpo.setup``; ``refit_policy_generation`` dispatches on it.
        self.weight_synchronizer: WeightSynchronizer | None = None
        self._async_loop: AsyncLoopThread | None = None
        self._http_client: HttpClient | None = None
        self.all_engines: list = []
        self._router_actor: ray.actor.ActorHandle | None = None
        self._router_actors: list[ray.actor.ActorHandle] = []
        self._router_endpoints: list[tuple[str, int]] = []
        self._engine_specs: list[SGLangEngineSpec] = []
        self._bootstrap_ports: dict[int, int] = {}
        self._dispatcher: ReplicaDispatcher | None = None
        self.rollout_engine_lock: ray.actor.ActorHandle | None = None

        fault_tolerance_config = SGLangFaultToleranceConfig.model_validate(
            sglang_cfg["sglang_cfg"]["sglang_fault_tolerance_config"]
        )
        # Prefill/decode disaggregation: refuse what cannot run before any
        # allocation (the NeMo-Gym gate is grpo.setup's), then resolve the
        # per-role engine configs once; everything below reads the resolved
        # config. Both are no-ops without disaggregation.
        validate_sglang_disaggregation(sglang_cfg, use_nemo_gym=False)
        self._disagg_cfg = resolve_sglang_disagg_layout(sglang_cfg)
        self._pd_cfg = resolve_sglang_disagg_config(sglang_cfg)
        sglang_cfg = resolve_sglang_role_overrides(sglang_cfg)
        self.cluster = cluster
        self.sglang_cfg = sglang_cfg
        # GenerationInterface consumers (create_weight_synchronizer, the refit
        # transports) read ``cfg``; keep the sglang-specific name as the alias.
        self.cfg = sglang_cfg
        self._async_loop = AsyncLoopThread()

        pgs = cluster._init_placement_groups(
            strategy="PACK",
            use_unified_pg=True,
        )
        self.pg = pgs[0]
        self.pg_reordered_bundle_indices, self.pg_reordered_gpu_ids, _ = (
            get_reordered_bundle(self.pg)
        )
        self._http_client = HttpClient(sglang_cfg)

        # --- Engine topology (formerly ``ServerGroup``) ------------------
        sglang_server_cfg = sglang_cfg["sglang_cfg"]["sglang_server_config"]
        gpus_per_engine = sglang_server_cfg["num_gpus_per_engine"]
        num_gpus_per_node = cluster.num_gpus_per_node
        if self._disagg_cfg.enabled:
            prefill_width = sglang_engine_width(
                sglang_role_engine_config(sglang_cfg, "prefill")
            )
            decode_width = sglang_engine_width(
                sglang_role_engine_config(sglang_cfg, "decode")
            )
        else:
            prefill_width = decode_width = gpus_per_engine
        # One spec per logical engine: role, width, GPUs, actor slots, replica
        # and router group. ``all_engines`` holds one actor per slot (one per
        # node of a multi-node engine), in spec order.
        self._engine_specs, self.num_replicas, num_routers = plan_sglang_engines(
            num_gpus=sglang_server_cfg["num_gpus"],
            gpus_per_node=num_gpus_per_node,
            layout=self._disagg_cfg,
            aggregated_width=gpus_per_engine,
            prefill_width=prefill_width,
            decode_width=decode_width,
            router_scope=self._pd_cfg.router_scope,
        )
        num_slots = sum(spec.num_slots for spec in self._engine_specs)
        if self._disagg_cfg.enabled:
            logger.info(
                "SGLang prefill/decode layout: %d replica(s) of %d prefill x %d GPUs "
                "+ %d decode x %d GPUs behind %d router(s)",
                self.num_replicas,
                self._disagg_cfg.num_prefill_engines,
                prefill_width,
                self._disagg_cfg.num_decode_engines,
                decode_width,
                num_routers,
            )

        self.num_gpus_per_engine: int = gpus_per_engine
        self.num_gpus_per_node: int = num_gpus_per_node
        self.all_engines = [None] * num_slots
        # Keep endpoints outside actors so a dead or wedged actor can still be
        # removed from the router. Entries follow all_engines, including peers.
        self._engine_urls: list[str | None] = [None] * num_slots
        # It will be useful for future features which involve pd disaggregation, mixture sglang config setup
        self.rank_offset: int = 0
        self.gpu_offset: int = 0
        self.needs_offload: bool = sglang_server_cfg["needs_offload"]
        self.model_path: str | None = sglang_cfg["sglang_cfg"]["model_path"]

        # --- Weight-refit / fault-tolerance state ------------------------
        # Number of engines created by the most recent ``_start_engines``
        # call that the refit dispatch has not connected yet.
        self.num_new_engines: int = 0
        self.pause_generation_mode: str = sglang_server_cfg["pause_generation_mode"]

        # --- Router bootstrap --------------------------------------------
        # Resolved router endpoints are held only on the instance; we don't
        # mutate the caller's config dict. Workers receive theirs as explicit
        # ``router_ip`` / ``router_port`` kwargs in ``init.remote(...)``.
        # Without disaggregation there is one router; with it, one per router
        # group (replica or fleet), started in PD mode with empty pools that
        # the engines join as they come up. Routers start one after another:
        # each picks a free port on its node, and a router that is already
        # listening cannot collide with the next one's choice.
        router_cfg = sglang_cfg["sglang_cfg"].get("sglang_router_config") or {}
        pd_args = None
        if self._disagg_cfg.enabled:
            pd_args = {
                "prefill_policy": self._pd_cfg.prefill_policy,
                "decode_policy": self._pd_cfg.decode_policy,
            }
        for _ in range(num_routers):
            router_ip, router_port, router_actor = _start_router(
                router_cfg, pd_args=pd_args
            )
            self._router_endpoints.append((router_ip, router_port))
            # Only set when ``_start_router`` actually spawned the router (i.e.
            # sglang_router_ip was not already configured). Kept so ``shutdown``
            # can terminate it cleanly.
            if router_actor is not None:
                self._router_actors.append(router_actor)
        self.router_ip: str
        self.router_port: int
        self.router_ip, self.router_port = self._router_endpoints[0]
        self._router_actor = self._router_actors[0] if self._router_actors else None
        self._dispatcher = ReplicaDispatcher(num_routers)

        # --- Start engines -----------------------------------------------
        init_handles, _ = self._start_engines({})
        if init_handles:
            ray.get(init_handles)

        # Serializes weight-update broadcasts. Acquired per bucket by trainer
        # rank 0 in policy/utils.py; nothing on the generation side takes it.
        self.rollout_engine_lock = Lock.options(num_cpus=0, num_gpus=0).remote()

        if fault_tolerance_config.use_fault_tolerance:
            monitor = RolloutHealthMonitor(self, fault_tolerance_config)
            monitor.start()
            self._health_monitor = monitor

    # ------------------------------------------------------------------
    # Engine topology properties (formerly ``ServerGroup``)
    # ------------------------------------------------------------------
    @property
    def nodes_per_engine(self) -> int:
        """Actors per engine when every engine has the shared width."""
        return max(1, self.num_gpus_per_engine // self.num_gpus_per_node)

    @property
    def engine_specs(self) -> list[SGLangEngineSpec]:
        """One spec per logical engine, in ``all_engines`` order.

        An instance whose engines were assigned directly (tests, partial
        construction) has no planned specs; it is read as the uniform layout
        the properties below always assumed: ``nodes_per_engine`` actors per
        engine, every engine ``num_gpus_per_engine`` wide, one router.
        """
        specs = getattr(self, "_engine_specs", None)
        if specs:
            return specs
        all_engines = getattr(self, "all_engines", None) or []
        width = getattr(self, "num_gpus_per_engine", 1)
        nodes_per_engine = max(1, width // getattr(self, "num_gpus_per_node", width))
        return [
            SGLangEngineSpec(
                index=i,
                role="aggregated",
                width=width,
                gpu_offset=i * width,
                first_slot=i * nodes_per_engine,
                num_slots=nodes_per_engine,
                replica=i,
                router=0,
            )
            for i in range(len(all_engines) // nodes_per_engine)
        ]

    @property
    def engines(self) -> list:
        """Node-0 engines only (one entry per logical engine)."""
        return [self.all_engines[spec.first_slot] for spec in self.engine_specs]

    @property
    def rollout_engines(self) -> list:
        """Alias for ``engines`` — node-0 engines across all servers / models."""
        return self.engines

    @property
    def engine_gpu_counts(self) -> list[int]:
        """Per-engine GPU count, parallel to ``engines``."""
        return [spec.width for spec in self.engine_specs]

    @property
    def engine_gpu_offsets(self) -> list[int]:
        return [self.gpu_offset + spec.gpu_offset for spec in self.engine_specs]

    def engine_slot_range(self, engine_idx: int) -> range:
        """Indices in ``all_engines`` of the actors driving engine ``engine_idx``."""
        return self.engine_specs[engine_idx].slots

    def engine_index_of_slot(self, slot: int) -> int:
        """The logical engine whose actor sits at ``all_engines[slot]``."""
        for spec in self.engine_specs:
            if slot in spec.slots:
                return spec.index
        raise IndexError(f"all_engines has no slot {slot}")

    def engine_url(self, engine_idx: int) -> str | None:
        """Cached HTTP endpoint of engine ``engine_idx`` (its node-0 server)."""
        return self._engine_urls[self.engine_specs[engine_idx].first_slot]

    def router_endpoint_for_engine(self, engine_idx: int) -> tuple[str, int]:
        """``(ip, port)`` of the router that fronts engine ``engine_idx``."""
        endpoints = getattr(self, "_router_endpoints", None)
        if not endpoints:
            return self.router_ip, self.router_port
        return endpoints[self.engine_specs[engine_idx].router]

    def _engine_sglang_cfg(self, spec: SGLangEngineSpec) -> SGLangConfig:
        """The config an engine of ``spec`` is built from: its role's resolved block."""
        if spec.role not in ("prefill", "decode"):
            return self.sglang_cfg
        return {
            **self.sglang_cfg,
            "sglang_cfg": sglang_role_engine_config(self.sglang_cfg, spec.role),
        }

    def _engine_groups(self, order: tuple[str, ...]) -> list[list]:
        """Live node-0 engines grouped by role, groups in ``order``.

        Roles outside ``order`` (the aggregated engines) form one leading
        group, so a non-disaggregated fleet is addressed all at once.
        """
        specs = self.engine_specs
        engines = self.engines
        groups = [
            [
                engine
                for spec, engine in zip(specs, engines)
                if spec.role not in order and engine is not None
            ]
        ]
        for role in order:
            groups.append(
                [
                    engine
                    for spec, engine in zip(specs, engines)
                    if spec.role == role and engine is not None
                ]
            )
        return [group for group in groups if group]

    def _start_engines(self, port_cursors: dict | None = None) -> tuple[list, dict]:
        """Create Ray actors, allocate ports, and fire ``engine.init()`` without waiting.

        Walks the engine specs: every slot (one actor per node of an engine)
        whose entry in ``all_engines`` is empty gets a new actor on its GPUs.
        Returns ``(init_handles, port_cursors)`` where *init_handles* is a list
        of Ray ObjectRefs and *port_cursors* maps a node (its index, or its
        host under disaggregation) to its next free port.
        """
        if port_cursors is None:
            port_cursors = {}
        pg = self.pg
        reordered_bundle_indices = self.pg_reordered_bundle_indices
        reordered_gpu_ids = self.pg_reordered_gpu_ids
        disaggregated = self._disagg_cfg.enabled

        # Resolve the SGLang venv ONCE (mirrors the once-per-node RayWorkerBuilder path).
        sglang_runtime_env_base = make_actor_runtime_env(
            "nemo_rl.models.generation.sglang.sglang_worker.SGLangGenerationWorker"
        )
        sglang_runtime_env_base.update(
            get_nsight_config_if_pattern_matches("sglang_generation_worker")
        )

        local_all_engines = []
        local_specs: dict[int, SGLangEngineSpec] = {}
        for spec in self.engine_specs:
            for node_rank, slot in enumerate(spec.slots):
                if self.all_engines[slot] is not None:
                    continue

                global_rank = self.rank_offset + slot
                num_gpus = min(0.2, 1 / self.cluster.max_colocated_worker_groups)
                num_cpus = num_gpus

                gpu_index = self.gpu_offset + spec.slot_gpu_index(node_rank)
                base_gpu_id = int(reordered_gpu_ids[gpu_index])

                scheduling_strategy = PlacementGroupSchedulingStrategy(
                    placement_group=pg,
                    placement_group_capture_child_tasks=True,
                    placement_group_bundle_index=reordered_bundle_indices[gpu_index],
                )

                env_vars = {
                    name: "1" for name in NOSET_VISIBLE_DEVICES_ENV_VARS_LIST
                } | {
                    key: os.environ.get(key, default_val)
                    for key, default_val in {
                        "SGLANG_JIT_DEEPGEMM_PRECOMPILE": "false",
                        "SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK": "true",
                        "SGLANG_DISABLE_TP_MEMORY_INBALANCE_CHECK": "true",
                        "SGLANG_MEMORY_SAVER_CUDA_GRAPH": "true",
                        "SGLANG_BATCH_INVARIANT_OPS_ENABLE_MM_FALLBACK_VARIANT": "true",
                        "SGLANG_ENABLE_HEALTH_ENDPOINT_GENERATION": "false",
                        "SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_IDLE": "false",
                    }.items()
                }

                # Trainer and engine must agree on the NCCL transport; sglang's
                # scheduler subprocess defaults to NCCL_CUMEM_ENABLE=0.
                env_vars["NCCL_CUMEM_ENABLE"] = "0"

                if disaggregated:
                    # What the KV transport needs from the engine environment on
                    # this host (NIXL: no HPC-X NCCL plugin; Mooncake: DMA-BUF
                    # registration without nvidia-peermem); the environment wins.
                    env_vars.update(disagg_transport_env(self._pd_cfg, os.environ))

                # Explicitly pass CUDA_VISIBLE_DEVICES through to the engine actor so
                # all engines see the same global value (Ray would otherwise remap it
                # because we set the NOSET_* flags above).
                global_cvd = os.environ.get("CUDA_VISIBLE_DEVICES", None)
                if global_cvd:
                    env_vars["CUDA_VISIBLE_DEVICES"] = global_cvd

                # Resolve the SGLang venv on every node and pin VIRTUAL_ENV /
                # UV_PROJECT_ENVIRONMENT so the actor (and its spawned children)
                # actually run inside the sglang venv instead of inheriting the
                # raylet's base venv (e.g. /opt/nemo_rl_venv inside the container).
                sglang_runtime_env = {
                    **sglang_runtime_env_base,
                    "env_vars": {**sglang_runtime_env_base["env_vars"], **env_vars},
                }

                actor_options = {
                    "num_cpus": num_cpus,
                    "num_gpus": num_gpus,
                    "scheduling_strategy": scheduling_strategy,
                    "runtime_env": sglang_runtime_env,
                }
                # A disaggregated engine is built from its role's resolved block.
                init_args = (self.num_gpus_per_node, self._engine_sglang_cfg(spec))
                init_kwargs = {
                    "rank": global_rank,
                    "base_gpu_id": base_gpu_id,
                    "num_gpus_per_engine": spec.width,
                    "node_rank": node_rank,
                    "disagg_role": spec.role if disaggregated else None,
                }

                # Create worker actor directly — sglang_worker.py uses lazy imports
                # so it's importable in SYSTEM env; the actor runs in sglang env.
                engine = SGLangGenerationWorker.options(**actor_options).remote(
                    *init_args, **init_kwargs
                )

                local_all_engines.append((global_rank, engine))
                local_specs[global_rank] = spec
                self.all_engines[slot] = engine

        self.num_new_engines = len(local_all_engines)

        if self.num_new_engines == 0:
            return [], port_cursors

        # SGLang engine server/NCCL/dist_init ports come from the reserved
        # generation band (3000-4999 by default), below the ephemeral floor;
        # see the port layout in virtual_cluster.py. Under disaggregation a
        # prefill engine's bootstrap port comes from the same band.
        gen_port_low = self.sglang_cfg.get("port_range_low")
        if gen_port_low is None:
            gen_port_low = DEFAULT_GENERATION_PORT_RANGE_LOW
        gen_port_high = self.sglang_cfg.get("port_range_high")
        if gen_port_high is None:
            gen_port_high = DEFAULT_GENERATION_PORT_RANGE_HIGH
        if disaggregated:
            addr_and_ports, bootstrap_ports, port_cursors = (
                _allocate_disagg_engine_addr_and_ports(
                    specs=self.engine_specs,
                    local_all_engines=local_all_engines,
                    rank_offset=self.rank_offset,
                    sglang_dp_size=self.sglang_cfg["sglang_cfg"]["dp_size"],
                    port_range_low=gen_port_low,
                    port_range_high=gen_port_high,
                    host_port_cursor=port_cursors,
                )
            )
            self._bootstrap_ports.update(bootstrap_ports)
        else:
            addr_and_ports, port_cursors = (
                _allocate_rollout_engine_addr_and_ports_normal(
                    gpus_per_node=self.num_gpus_per_node,
                    sglang_cfg=self.sglang_cfg,
                    local_all_engines=local_all_engines,
                    rank_offset=self.rank_offset,
                    port_range_low=gen_port_low,
                    port_range_high=gen_port_high,
                    node_port_cursor=port_cursors,
                )
            )

        for rank, _engine in local_all_engines:
            address = addr_and_ports[rank]
            self._engine_urls[rank - self.rank_offset] = (
                f"http://{_format_v6_uri(address['host'])}:{address['port']}"
            )

        init_handles = []
        for rank, engine in local_all_engines:
            spec = local_specs[rank]
            router_ip, router_port = self._router_endpoints[spec.router]
            init_kwargs = dict(
                addr_and_ports[rank], router_ip=router_ip, router_port=router_port
            )
            if disaggregated:
                init_kwargs["disagg_server_kwargs"] = sglang_disagg_server_kwargs(
                    spec.role, self._pd_cfg, self._bootstrap_ports.get(spec.index)
                )
            init_handles.append(engine.init.remote(**init_kwargs))
        return init_handles, port_cursors

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def check_weights(self, action: str):
        """All node-0 engines across all servers / models."""
        return ray.get(
            [
                engine.check_weights.remote(action=action)
                for engine in self.engines
                if engine is not None
            ]
        )

    def _recover(self) -> None:
        """Recover dead engines, overlapping init."""
        if self._health_monitor is not None:
            # The serving probe is paused during training and refit. This
            # actor/process probe also works with weights and KV cache offloaded.
            self._health_monitor.check_liveness()
        dead_indices = [
            i for i, engine in enumerate(self.all_engines) if engine is None
        ]
        if not dead_indices:
            # ``_start_engines`` rewrites ``num_new_engines`` unconditionally, so
            # calling it with nothing to restart would clear a count the refit
            # dispatch has not consumed yet. That count is what gates ``_connect``
            # in the weight synchronizer, and ``_connect`` is the only place the
            # transport is ever built -- clearing it on the first refit leaves
            # every rank silently no-oping the weight send for the rest of the run.
            return

        if self._health_monitor is not None:
            self._health_monitor.record_restart_attempts(dead_indices)

        port_cursors: dict[int, int] = {}
        handles, _ = self._start_engines(port_cursors)
        if handles:
            ray.get(handles)

        assert self.num_new_engines == len(dead_indices), (
            "num_new_engines does not match dead_indices length"
        )

        # Replacement engines are freshly booted and still loading weights, so
        # give them the configured grace period before the monitor probes them.
        if self._health_monitor is not None:
            self._health_monitor.arm_first_wait()

        if self.needs_offload and dead_indices:
            new_engines = [self.all_engines[i] for i in dead_indices]
            ray.get(
                [
                    engine.release_memory_occupation.remote(tags=["weights"])
                    for engine in new_engines
                ]
            )
            ray.get(
                [
                    engine.release_memory_occupation.remote(tags=["kv_cache"])
                    for engine in new_engines
                ]
            )
            # The synchronizer onloads weights after recovery for all engines.
            # Resuming here too would onload a replacement twice.

    def get_updatable_engines(self) -> tuple[list, int, list[int], list[int]]:
        """Return engines eligible for weight updates."""
        return (
            self.engines,
            self.num_new_engines,
            self.engine_gpu_counts,
            self.engine_gpu_offsets,
        )

    def recover_updatable_engines(self) -> None:
        """Restart any dead rollout engines and update ``num_new_engines``."""
        # Resumed by prepare_for_generation; probing earlier races the weight stream.
        self._health_monitor.pause()

        self._recover()

    def clear_updatable_num_new_engines(self):
        # Called by the refit dispatch once it has connected the new engines, so
        # the next refit does not treat them as new again.
        self.num_new_engines = 0

    def pause_generation(self, mode: str) -> None:
        """Pause generation on every node-0 engine.

        Under disaggregation the prefill engines pause first, so no new KV
        handshake starts while the decode engines drain, then the decode
        engines. Aggregated engines pause all at once.

        Args:
            mode: Pause mode forwarded to every engine.
        """
        for group in self._engine_groups(("prefill", "decode")):
            ray.get([e.pause_generation.remote(mode=mode) for e in group])

    def continue_generation(self) -> None:
        """Resume generation on every node-0 engine, decode engines first."""
        for group in self._engine_groups(("decode", "prefill")):
            ray.get([e.continue_generation.remote() for e in group])

    def pause_generation_for_refit(
        self, *, clear_cache: bool, timeout_s: Optional[float] = None
    ) -> bool:
        """Acknowledge the pause owned by the SGLang weight synchronizer.

        Both SGLang transports pause and flush KV inside ``sync_weights`` before
        transferring weights. This hook issues no RPC; returning True prevents
        callers from flushing again after the synchronizer resumes generation.
        The synchronizer always clears KV, including when ``clear_cache=False``.
        ``timeout_s`` does not add a deadline to that existing lifecycle.
        """
        del clear_cache, timeout_s
        return True

    def resume_generation_after_refit(
        self, *, timeout_s: Optional[float] = None
    ) -> bool:
        """Acknowledge that the SGLang synchronizer already resumed generation.

        This hook issues no RPC and adds no deadline; ``sync_weights`` owns the
        pause, cache flush, weight transfer, and resume as one lifecycle.
        """
        del timeout_s
        return True

    def begin_weight_update(self) -> None:
        """Open a weight-update session on every node-0 engine."""
        engines = [e for e in self.engines if e is not None]
        if not engines:
            return
        ray.get([e.begin_weight_update.remote() for e in engines])

    def end_weight_update(self) -> None:
        """Close a weight-update session and rebuild quantized kernel layouts."""
        engines = [e for e in self.engines if e is not None]
        if not engines:
            return
        ray.get([e.end_weight_update.remote() for e in engines])

    def shutdown(self) -> bool:
        ok = True
        if self._health_monitor:
            self._health_monitor.stop()

        if self.weight_synchronizer is not None:
            # ``shutdown`` is reachable twice (explicit call + ``__del__``);
            # drop the handle so teardown stays one-shot.
            self.weight_synchronizer.shutdown()
            self.weight_synchronizer = None

        rollout_engine_lock = getattr(self, "rollout_engine_lock", None)
        if rollout_engine_lock is not None:
            try:
                ray.kill(rollout_engine_lock)
            except Exception as e:
                logger.warning(f"Rollout-engine lock terminate failed: {e}")
                ok = False
            del self.rollout_engine_lock

        engines = [e for e in self.all_engines if e is not None]
        if engines:
            try:
                ray.get([e.shutdown.remote() for e in engines])
            except Exception as e:
                logger.warning(f"Engine shutdown failed: {e}")
                ok = False
            self.all_engines = [None] * len(self.all_engines)

        router_actors = list(getattr(self, "_router_actors", None) or [])
        legacy_router_actor = getattr(self, "_router_actor", None)
        if legacy_router_actor is not None and legacy_router_actor not in router_actors:
            router_actors.append(legacy_router_actor)
        for router_actor in router_actors:
            try:
                ray.get(router_actor.stop.remote())
                ray.kill(router_actor)
            except Exception as e:
                logger.warning(f"Router terminate failed: {e}")
                ok = False
        self._router_actors = []
        self._router_actor = None

        if self._http_client is not None:
            try:
                if self._async_loop is not None:
                    self._async_loop.run(self._http_client.aclose())
                else:
                    self._http_client.shutdown()
            except Exception as e:
                logger.warning(f"HTTP client shutdown failed: {e}")
                ok = False
            self._http_client = None

        if self._async_loop is not None:
            try:
                self._async_loop.close()
            except Exception as e:
                logger.warning(f"AsyncLoopThread close failed: {e}")
                ok = False
            self._async_loop = None

        return ok

    def __del__(self) -> None:
        self.shutdown()

    def _merge_stop_strings(self, batch_stop_strings) -> list[list[str]]:
        """Merge stop strings from config and batch.

        Args:
            batch_stop_strings: List of stop strings from batch (one per sample)

        Returns:
            List of merged stop strings (one per sample)
        """
        stop_set: set[str] = set()

        # Add stop strings from config
        if self.sglang_cfg.get("stop_strings"):
            stop_set.update(self.sglang_cfg["stop_strings"])

        # Merge stop strings from batch
        merged_stop_strings = []
        for sample_ss in batch_stop_strings:
            sample_stop_set = stop_set.copy()
            if sample_ss:
                if isinstance(sample_ss, str):
                    sample_stop_set.add(sample_ss)
                elif isinstance(sample_ss, list):
                    sample_stop_set.update(sample_ss)

            merged_stop_strings.append(list(sample_stop_set))

        return merged_stop_strings

    def _build_sampling_params(
        self,
        *,
        greedy: bool,
        max_new_tokens: int,
        stop_strings: list[str] | None = None,
    ) -> dict[str, Any]:
        """Build sampling parameters dictionary for SGLang API.

        Args:
            greedy: Whether to use greedy decoding (temperature=0.0)
            max_new_tokens: Max new tokens for this sample (already clamped by caller
                against ``context_length - input_length - 1``).
            stop_strings: Merged stop strings for this sample.

        Returns:
            Dictionary of sampling parameters compatible with SGLang API.
        """
        temperature = 0.0 if greedy else self.sglang_cfg["temperature"]
        top_k_cfg = self.sglang_cfg.get("top_k")
        top_k_val = 1 if greedy else (top_k_cfg if top_k_cfg is not None else -1)

        # Build sampling params dict first, then patch in optional fields so we
        # never reference ``sampling_params`` before it's bound.
        sampling_params: dict[str, Any] = {
            "temperature": temperature,
            "top_p": self.sglang_cfg.get("top_p", 1.0),
            "max_new_tokens": max_new_tokens,
            "no_stop_trim": True,
            "spaces_between_special_tokens": False,
        }

        if top_k_val != -1:
            sampling_params["top_k"] = top_k_val

        stop_token_ids = self.sglang_cfg.get("stop_token_ids")
        if stop_token_ids is not None:
            sampling_params["stop_token_ids"] = stop_token_ids

        if stop_strings is not None and len(stop_strings) > 0:
            sampling_params["stop"] = stop_strings

        return sampling_params

    @wrap_with_nvtx_name("sglang_genertion/generate")
    def generate(
        self, data: BatchedDataDict[GenerationDatumSpec], greedy: bool = False
    ) -> BatchedDataDict[GenerationOutputSpec]:
        """Generate a batch of data using Sglang generation.

        Args:
            data: BatchedDataDict containing input_ids and input_lengths tensors
            greedy: Whether to use greedy decoding instead of sampling

        Returns:
            BatchedDataDict conforming to GenerationOutputSpec:
                - output_ids: input + generated token IDs with proper padding
                - logprobs: Log probabilities for tokens
                - generation_lengths: Lengths of each response
                - unpadded_sequence_lengths: Lengths of each input + generated sequence
        """
        # Handle empty input case
        if len(data["input_ids"]) == 0:
            # Return empty BatchedDataDict with all required fields
            return BatchedDataDict[GenerationOutputSpec](
                {
                    "output_ids": torch.zeros((0, 0), dtype=torch.long),
                    "logprobs": torch.zeros((0, 0), dtype=torch.float),
                    "generation_lengths": torch.zeros(0, dtype=torch.long),
                    "unpadded_sequence_lengths": torch.zeros(0, dtype=torch.long),
                    "truncated": torch.zeros(0, dtype=torch.bool),
                }
            )

        input_ids = data["input_ids"]
        input_lengths = data["input_lengths"]
        batch_stop_strings: list[list[str]] = data.get("stop_strings", [])
        stop_strings = self._merge_stop_strings(batch_stop_strings)

        batch_size = len(input_lengths)
        padded_input_length = input_ids.size(1)
        context_length = self.sglang_cfg["sglang_cfg"]["context_length"]

        # verify inputs have correct padding
        verify_right_padding(data, pad_value=self.sglang_cfg["_pad_token_id"])

        # Build per-sample requests (each sample gets its own sampling params because
        # max_new_tokens is adjusted against the per-sample input length).
        sample_requests: list[tuple[int, dict[str, Any], list[int]]] = []
        skip_results: set[int] = set()
        skip_max_length = 0
        for i in range(batch_size):
            input_length = input_lengths[i].item()
            valid_input_ids = input_ids[i, :input_length].tolist()

            if context_length is not None:
                max_new_tokens = min(
                    self.sglang_cfg["max_new_tokens"],
                    context_length - input_length - 1,
                )
            else:
                max_new_tokens = self.sglang_cfg["max_new_tokens"]
            max_new_tokens = max(0, max_new_tokens)

            if max_new_tokens == 0:
                skip_results.add(i)
                skip_max_length = max(skip_max_length, input_length)
                continue

            sample_sampling_params = self._build_sampling_params(
                greedy=greedy,
                max_new_tokens=max_new_tokens,
                stop_strings=stop_strings[i] if i < len(stop_strings) else None,
            )
            sample_requests.append((i, sample_sampling_params, valid_input_ids))

        # Dispatch concurrently to the SGLang router with bounded concurrency.
        # Max concurrency = per-engine concurrency * number of engines.
        sglang_server_cfg = self.sglang_cfg["sglang_cfg"]["sglang_server_config"]
        max_concurrency = (
            sglang_server_cfg["sglang_server_concurrency"]
            * sglang_server_cfg["num_gpus"]
            // sglang_server_cfg["num_gpus_per_engine"]
        )

        semaphore = asyncio.Semaphore(max_concurrency)

        async def _bounded_generate_one_sample(
            idx: int, sp: dict[str, Any], ids: list[int]
        ):
            async with semaphore:
                return await self.generate_one_sample(sp, ids, idx)

        async def _dispatch_all() -> dict[int, tuple[list[int], list[float], bool]]:
            gathered = await asyncio.gather(
                *(
                    _bounded_generate_one_sample(idx, sp, ids)
                    for idx, sp, ids in sample_requests
                )
            )
            # generate_one_sample returns (index, tokens, logprobs, truncated).
            # Re-key by the original sample index so downstream code can look up
            # results directly without sorting.
            return {
                returned_idx: (new_tokens, new_logprobs, is_truncated)
                for returned_idx, new_tokens, new_logprobs, is_truncated in gathered
            }

        router_results: dict[int, tuple[list[int], list[float], bool]] = (
            self._async_loop.run(_dispatch_all()) if sample_requests else {}
        )

        # Process the outputs - preserve the original input padding structure.
        pad_token_id = self.sglang_cfg["_pad_token_id"]
        output_ids_list: list[torch.Tensor] = []
        logprobs_list: list[torch.Tensor] = []
        generation_lengths_list: list[int] = []
        unpadded_sequence_lengths_list: list[int] = []
        truncated_list: list[bool] = []

        # First pass: compute total_length as the max over all samples of
        # (input_length + generation_length). Skipped samples contribute only
        # their input_length (already tracked in ``skip_max_length``).
        max_length = skip_max_length
        for returned_idx, (returned_tokens, _, _) in router_results.items():
            sample_input_length = input_lengths[returned_idx].item()
            max_length = max(max_length, sample_input_length + len(returned_tokens))
        total_length = max(max_length, padded_input_length)

        # Second pass: materialize the output tensors, using a single set of
        # local variable names (``generation_length`` / ``unpadded_length`` are
        # always Python ints; tensor promotion happens only at the final stack).
        for i in range(batch_size):
            input_length = input_lengths[i].item()
            full_output = torch.full(
                (total_length,), pad_token_id, dtype=input_ids.dtype
            )
            full_logprobs = torch.zeros(total_length, dtype=torch.float32)
            full_output[:input_length] = input_ids[i][:input_length]

            if i in skip_results:
                generation_length = 0
                is_truncated = False
            else:
                new_tokens, new_logprobs, is_truncated = router_results[i]
                generation_length = len(new_tokens)
                if new_tokens:
                    full_output[input_length : input_length + generation_length] = (
                        torch.tensor(new_tokens, dtype=input_ids.dtype)
                    )
                if new_logprobs:
                    full_logprobs[input_length : input_length + len(new_logprobs)] = (
                        torch.tensor(new_logprobs, dtype=torch.float32)
                    )

            unpadded_length = input_length + generation_length
            output_ids_list.append(full_output)
            logprobs_list.append(full_logprobs)
            generation_lengths_list.append(generation_length)
            unpadded_sequence_lengths_list.append(unpadded_length)
            truncated_list.append(bool(is_truncated))

        return_data = BatchedDataDict[GenerationOutputSpec](
            {
                "output_ids": torch.stack(output_ids_list),
                "logprobs": torch.stack(logprobs_list),
                "generation_lengths": torch.tensor(
                    generation_lengths_list, dtype=torch.long
                ),
                "unpadded_sequence_lengths": torch.tensor(
                    unpadded_sequence_lengths_list, dtype=torch.long
                ),
                "truncated": torch.tensor(truncated_list, dtype=torch.bool),
            }
        )

        return return_data

    async def generate_async(
        self,
        data: BatchedDataDict[GenerationDatumSpec],
        greedy: bool = False,
    ) -> AsyncGenerator[tuple[int, BatchedDataDict[GenerationOutputSpec]], None]:
        """Generate a single sample using SGLang, yielding the result when ready.

        Args:
            data: BatchedDataDict with input_ids and input_lengths (batch_size must be 1)
            greedy: Whether to use greedy decoding instead of sampling

        Yields:
            Tuple of (original_index, BatchedDataDict conforming to GenerationOutputSpec)
        """
        # Handle empty input case
        if len(data["input_ids"]) == 0:
            return

        verify_right_padding(data, pad_value=self.sglang_cfg["_pad_token_id"])

        input_ids_batch = data["input_ids"]
        input_lengths_batch = data["input_lengths"]
        batch_size = input_ids_batch.shape[0]

        # Restrict to single-sample batches, matching the vLLM async contract.
        assert batch_size == 1, (
            f"generate_async is restricted to handle only single samples, "
            f"but received batch_size={batch_size}. Please handle batching outside this method."
        )

        sample_idx = 0
        input_length = input_lengths_batch[sample_idx].item()
        original_input_ids_single_row = input_ids_batch[sample_idx]
        device = original_input_ids_single_row.device
        dtype = original_input_ids_single_row.dtype
        pad_token_id = self.sglang_cfg["_pad_token_id"]

        # Clamp max_new_tokens against the per-sample remaining context window,
        # mirroring the logic in ``generate``.
        context_length = self.sglang_cfg["sglang_cfg"].get("context_length")
        if context_length is not None:
            max_new_tokens = min(
                self.sglang_cfg["max_new_tokens"],
                context_length - input_length - 1,
            )
        else:
            max_new_tokens = self.sglang_cfg["max_new_tokens"]
        max_new_tokens = max(0, max_new_tokens)

        # Short-circuit when there is no room left in the context window. Yield
        # a pure-input row (generation_length=0, truncated=False) without
        # touching the SGLang router.
        if max_new_tokens == 0:
            output_ids_single_item_batched = original_input_ids_single_row[
                :input_length
            ].unsqueeze(0)
            logprobs_single_item = torch.zeros(
                (1, input_length), dtype=torch.float32, device=device
            )
            empty_result = BatchedDataDict[GenerationOutputSpec](
                {
                    "output_ids": output_ids_single_item_batched,
                    "logprobs": logprobs_single_item,
                    "generation_lengths": torch.tensor(
                        [0], dtype=torch.long, device=device
                    ),
                    "unpadded_sequence_lengths": torch.tensor(
                        [input_length], dtype=torch.long, device=device
                    ),
                    "truncated": torch.tensor([False], dtype=torch.bool, device=device),
                }
            )
            yield (sample_idx, empty_result)
            return

        # Merge stop strings for this single sample.
        batch_stop_strings: list[list[str]] = data.get("stop_strings", [])
        stop_strings = self._merge_stop_strings(batch_stop_strings)
        per_sample_stop_strings = (
            stop_strings[sample_idx] if sample_idx < len(stop_strings) else None
        )

        sampling_params = self._build_sampling_params(
            greedy=greedy,
            max_new_tokens=max_new_tokens,
            stop_strings=per_sample_stop_strings,
        )

        valid_input_ids = original_input_ids_single_row[:input_length].tolist()

        # batch_size == 1, so no task fan-out / as_completed is needed. Just
        # await the single coroutine directly.
        _, new_tokens, new_logprobs, is_truncated = await self.generate_one_sample(
            sampling_params,
            valid_input_ids,
            sample_idx,
        )

        # Build the single-sample output tensor: [input | generated].
        generation_length = len(new_tokens)
        unpadded_length = input_length + generation_length

        output_ids_single_item = torch.full(
            (unpadded_length,), pad_token_id, dtype=dtype, device=device
        )
        output_ids_single_item[:input_length] = original_input_ids_single_row[
            :input_length
        ]
        # Logprobs: zeros for input tokens, raw floats at generated positions.
        logprobs_single_item = torch.zeros(
            (1, unpadded_length), dtype=torch.float32, device=device
        )

        if new_tokens:
            output_ids_single_item[input_length:unpadded_length] = torch.tensor(
                new_tokens, dtype=dtype, device=device
            )
        if new_logprobs:
            logprobs_single_item[0, input_length : input_length + len(new_logprobs)] = (
                torch.tensor(new_logprobs, dtype=torch.float32, device=device)
            )

        result_batch = BatchedDataDict[GenerationOutputSpec](
            {
                "output_ids": output_ids_single_item.unsqueeze(0),
                "logprobs": logprobs_single_item,
                "generation_lengths": torch.tensor(
                    [generation_length], dtype=torch.long, device=device
                ),
                "unpadded_sequence_lengths": torch.tensor(
                    [unpadded_length], dtype=torch.long, device=device
                ),
                "truncated": torch.tensor(
                    [bool(is_truncated)], dtype=torch.bool, device=device
                ),
            }
        )

        yield (sample_idx, result_batch)

    # ---------------------------------------------------------------------------
    # Compatible with parent class or old interfaces
    # ---------------------------------------------------------------------------
    def init_collective(
        self, ip: str, port: int, world_size: int, *, train_world_size: int
    ) -> list[ray.ObjectRef]:
        return []

    def prepare_refit_info(
        self, state_dict_info: Optional[dict[str, Any]]
    ) -> Optional[list[str]]:
        return None

    def update_weights_via_ipc_zmq(self) -> list[ray.ObjectRef]:
        return []

    def update_weights_from_collective(
        self, refit_timeout_s: Optional[float] = None
    ) -> list[ray.ObjectRef]:
        reject_unenforceable_refit_deadline("SGLang", refit_timeout_s)
        return []

    def prepare_for_generation(self, *args: Any, **kwargs: Any) -> bool:
        """Wake workers up for colocated inference."""
        tags = kwargs.get("tags", None)
        if self.needs_offload:
            engines = [e for e in self.engines if e is not None]
            if engines:
                ray.get([e.resume_memory_occupation.remote(tags=tags) for e in engines])
        # The weights-only stage is mid-refit; only kv_cache can serve a probe.
        if self._health_monitor and (tags is None or "kv_cache" in tags):
            self._health_monitor.resume()

    def finish_generation(self, *args: Any, **kwargs: Any) -> bool:
        """Sleep workers and reset prefix cache."""
        if self._health_monitor:
            self._health_monitor.pause()
        dispatcher = getattr(self, "_dispatcher", None)
        if dispatcher is not None:
            # Nothing is in flight between steps; forget where this step's
            # prompts went so the next step spreads its own afresh.
            dispatcher.reset()
        if not self.needs_offload:
            return
        tags = kwargs.get("tags", None)
        engines = [e for e in self.engines if e is not None]
        if not engines:
            return
        ray.get([e.release_memory_occupation.remote(tags=tags) for e in engines])

    def invalidate_kv_cache(self) -> bool:
        """Invalidate KV cache before weight updates (Megatron-style).

        Flushes the cache on every node-0 engine so stale KV entries are
        discarded before new weights land. Returns ``True`` iff every engine
        reports success.
        """
        engines = [e for e in self.engines if e is not None]
        if not engines:
            return True
        try:
            ray.get([e.invalidate_kv_cache.remote() for e in engines])
        except Exception as e:
            logger.error(f"[sglang refit] Error flushing SGLang caches: {e}")
            return False

        logger.info("[sglang refit] All SGLang server caches flushed successfully")
        return True

    # ---------------------------------------------------------------------------
    # Generate one sample helper
    # ---------------------------------------------------------------------------
    async def generate_one_sample(
        self,
        sampling_params,
        input_ids,
        index: int,
    ):
        """Generate using traditional SGLang router with token-based workflow.

        With several routers (one per replica under disaggregation) the
        dispatcher picks one: the samples of a prompt share a router, and a
        new prompt goes to the router with the fewest requests in flight.
        """
        dispatcher = getattr(self, "_dispatcher", None)
        endpoints = getattr(self, "_router_endpoints", None)
        if dispatcher is not None and endpoints:
            target = dispatcher.acquire(ReplicaDispatcher.prompt_key(input_ids))
            router_ip, router_port = endpoints[target]
        else:
            target = None
            router_ip, router_port = self.router_ip, self.router_port
        url = f"http://{router_ip}:{router_port}/generate"

        # Prepare payload for sglang server
        payload = {
            "sampling_params": sampling_params,
            "return_logprob": True,
            "input_ids": input_ids,
        }

        try:
            output = await self._http_client.post(url, payload)
        finally:
            if target is not None:
                dispatcher.release(target)

        if "output_token_logprobs" in output["meta_info"]:
            response_tokens = [
                item[1] for item in output["meta_info"]["output_token_logprobs"]
            ]
            response_log_probs = [
                item[0] for item in output["meta_info"]["output_token_logprobs"]
            ]
        else:
            response_tokens, response_log_probs = [], []

        # SGLang reports the termination reason under meta_info.finish_reason.type;
        # "length" means the decoder hit max_new_tokens before EOS.
        finish_reason = output["meta_info"].get("finish_reason") or {}
        response_truncated = finish_reason.get("type") == "length"

        return index, response_tokens, response_log_probs, response_truncated
