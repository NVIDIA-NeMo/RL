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
import threading
import uuid
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
    verify_right_padding,
)
from nemo_rl.models.generation.sglang.config import (
    SGLangConfig,
    normalize_sglang_config,
)
from nemo_rl.models.generation.sglang.fault_tolerance import RolloutHealthMonitor
from nemo_rl.models.generation.sglang.sglang_router import _start_router
from nemo_rl.models.generation.sglang.sglang_worker import SGLangGenerationWorker
from nemo_rl.models.generation.sglang.utils.async_utils import AsyncLoopThread
from nemo_rl.models.generation.sglang.utils.http_utils import HttpClient
from nemo_rl.models.generation.sglang.utils.ip_port_utils import (
    _allocate_rollout_engine_addr_and_ports_normal,
)
from nemo_rl.models.generation.sglang.utils.refit_deadline import (
    SGLangRefitDeadline,
    SGLangRefitTimeoutError,
    cancel_ray_refs,
)
from nemo_rl.models.generation.sglang.utils.startup_deadline import (
    SGLangStartupDeadline,
)
from nemo_rl.models.generation.sglang.utils.ray_utils import (
    NOSET_VISIBLE_DEVICES_ENV_VARS_LIST,
    Lock,
)
from nemo_rl.utils.nsys import wrap_with_nvtx_name
from nemo_rl.utils.venvs import make_actor_runtime_env

logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

_BEST_EFFORT_REFIT_CLEANUP_TIMEOUT_S = 5.0
_OFFLOAD_MONITOR_SUSPENSION = "offload"


class SGLangGeneration(GenerationInterface):
    """The class to run rollout and convert rollout data to training data.

    This class owns the full rollout server topology: the placement group,
    the router subprocess, and every ``SGLangGenerationWorker`` Ray actor.
    The former ``ServerGroup`` dataclass has been folded in so there is a
    single source of truth for engine state.

    TODO: one sglang router(router ip, router port) --> different server group(eg: PD, different tp size, ...; each server group multiple engines/servsers with same settings)
    router + [[p, ..., p] [d, ..., d]] or router + [[tp = 2, ... tp = 2], ..., [tp = 8, ..., tp = 8]]
    """

    def __init__(
        self,
        cluster: RayVirtualCluster,
        sglang_cfg: SGLangConfig,
    ):
        normalize_sglang_config(sglang_cfg)
        self.cluster = cluster
        self.sglang_cfg = sglang_cfg
        self._owns_runtime = True
        self._async_loop: AsyncLoopThread | None = AsyncLoopThread()
        self._http_client: HttpClient | None = None

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
        num_gpu_per_engine_local = min(gpus_per_engine, num_gpus_per_node)
        num_engines = sglang_server_cfg["num_gpus"] // num_gpu_per_engine_local

        self.num_gpus_per_engine: int = gpus_per_engine
        self.num_gpus_per_node: int = num_gpus_per_node
        self.all_engines: list = [None] * num_engines
        # It will be useful for future features which involve pd disaggregation, mixture sglang config setup
        self.rank_offset: int = 0
        self.gpu_offset: int = 0
        self.needs_offload: bool = sglang_server_cfg["needs_offload"]
        self.model_path: str | None = sglang_cfg["sglang_cfg"]["model_path"]

        # --- Weight-refit / fault-tolerance state ------------------------
        # Number of engines created by the most recent ``_start_engines``
        # call that the refit dispatch has not connected yet.
        self.num_new_engines: int = 0
        # A failed child-process/router cleanup is not recoverable in place:
        # launching another actor on the same GPU/ports could overlap an
        # unreaped engine. This latch is intentionally persistent.
        self._engine_cleanup_error: str | None = None
        self.pause_generation_mode: str = sglang_server_cfg["pause_generation_mode"]
        self._health_monitor: RolloutHealthMonitor | None = None
        self._health_monitor_suspensions: dict[str, str] = {}
        self._health_monitor_state_lock = threading.Lock()
        if self.needs_offload:
            self._health_monitor_suspensions[_OFFLOAD_MONITOR_SUSPENSION] = (
                _OFFLOAD_MONITOR_SUSPENSION
            )

        # --- Router bootstrap --------------------------------------------
        # Resolved router endpoint is held only on the instance; we don't
        # mutate the caller's config dict. Workers receive these as explicit
        # ``router_ip`` / ``router_port`` kwargs in ``init.remote(...)``.
        self._router_actor: ray.actor.ActorHandle | None = None
        init_handles = []
        startup_deadline = SGLangStartupDeadline(self.engine_startup_timeout_s)
        try:
            router_ip, router_port, router_actor = _start_router(
                sglang_cfg["sglang_cfg"].get("sglang_router_config") or {},
                deadline=startup_deadline,
            )
            self.router_ip = router_ip
            self.router_port = router_port
            # Only set when ``_start_router`` actually spawned the router (i.e.
            # sglang_router_ip was not already configured). Kept so ``shutdown``
            # can terminate it cleanly.
            self._router_actor = router_actor

            # --- Start engines -------------------------------------------
            init_handles, _ = self._start_engines(
                {},
                deadline=startup_deadline,
            )
            self._wait_for_engine_startup(init_handles, deadline=startup_deadline)
        except BaseException as exc:
            self._cleanup_failed_startup(init_handles, exc)
            raise

        # Serializes weight refits against engine recovery across processes.
        self.rollout_engine_lock = Lock.options(num_cpus=1, num_gpus=0).remote()

        if sglang_cfg["sglang_cfg"]["use_fault_tolerance"]:
            monitor = RolloutHealthMonitor(self, sglang_cfg)
            monitor.start()
            self._health_monitor = monitor
            if not self._health_monitor_suspensions:
                monitor.resume()

    # ------------------------------------------------------------------
    # Engine topology properties (formerly ``ServerGroup``)
    # ------------------------------------------------------------------
    @property
    def nodes_per_engine(self) -> int:
        return max(1, self.num_gpus_per_engine // self.num_gpus_per_node)

    @property
    def engines(self) -> list:
        """Node-0 engines only (one entry per logical engine)."""
        return self.all_engines[:: self.nodes_per_engine]

    @property
    def rollout_engines(self) -> list:
        """Alias for ``engines`` — node-0 engines across all servers / models."""
        return self.engines

    @property
    def engine_gpu_counts(self) -> list[int]:
        """Per-engine GPU count, parallel to ``engines``."""
        return [self.num_gpus_per_engine for _ in self.engines]

    @property
    def cfg(self):
        """Full generation config dict (parity with ``VllmGeneration.cfg``).

        Shared rollout code (``nemo_rl.experience.rollouts``) inspects
        ``policy_generation.cfg`` to find the backend-specific sub-config; return
        the same full ``SGLangConfig`` the engines were built from so the
        ``"sglang_cfg"`` branch resolves.
        """
        return self.sglang_cfg

    # --- pickling support -------------------------------------------------
    # Async GRPO ships this handle into the trajectory-collector Ray actor.
    # ``_http_client`` (httpx) and ``_async_loop`` (a live thread + asyncio
    # loop) hold contextvars/threads and cannot pickle; ``_health_monitor``
    # owns engine-liveness threads that belong to the driver only. Drop all
    # three and mark the deserialized copy as a non-owner before rebuilding its
    # local client/loop. A collector copy may use actor handles but must never
    # terminate the driver-owned engines, router, or health monitor.
    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_owns_runtime"] = False
        state["_http_client"] = None
        state["_async_loop"] = None
        state["_health_monitor"] = None
        state["_health_monitor_state_lock"] = None
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self._owns_runtime = False
        if self.__dict__.get("_async_loop") is None:
            self._async_loop = AsyncLoopThread()
        if self.__dict__.get("_http_client") is None:
            self._http_client = HttpClient(self.sglang_cfg)
        self._health_monitor_state_lock = threading.Lock()

    @property
    def engine_gpu_offsets(self) -> list[int]:
        return [
            self.gpu_offset + j * self.num_gpus_per_engine
            for j in range(len(self.engines))
        ]

    @property
    def engine_startup_timeout_s(self) -> float:
        """Configured end-to-end timeout for initial SGLang engine startup."""
        timeout_s = float(self.sglang_cfg["sglang_cfg"]["engine_startup_timeout_s"])
        if timeout_s <= 0:
            raise ValueError(
                f"sglang_cfg.engine_startup_timeout_s must be positive, got {timeout_s}"
            )
        return timeout_s

    @property
    def refit_timeout_s(self) -> float:
        """Configured end-to-end timeout for disaggregated SGLang refits."""
        timeout_s = float(self.sglang_cfg["sglang_cfg"]["refit_timeout_s"])
        if timeout_s <= 0:
            raise ValueError(
                f"sglang_cfg.refit_timeout_s must be positive, got {timeout_s}"
            )
        return timeout_s

    def _wait_for_engine_startup(
        self,
        init_handles,
        *,
        deadline: SGLangStartupDeadline,
    ) -> None:
        """Wait for every engine using the constructor's shared deadline."""
        if not init_handles:
            return
        deadline.ray_get(
            init_handles,
            stage="waiting for engine initialization",
            cancel_on_error=True,
        )

    def _cleanup_failed_startup(
        self, init_handles, startup_error: BaseException
    ) -> None:
        """Best-effort cleanup that preserves the original startup exception."""
        cleanup_ok = True
        try:
            cancel_ray_refs(init_handles)
        except BaseException:
            logger.exception("Cancelling SGLang startup refs raised during cleanup")
            cleanup_ok = False
        try:
            cleanup_ok = self.shutdown() and cleanup_ok
        except BaseException:
            logger.exception("SGLang startup cleanup raised after bootstrap failed")
            cleanup_ok = False
        if not cleanup_ok:
            detail = (
                "SGLang startup cleanup could not be confirmed; terminate the "
                "enclosing Ray runtime before retrying"
            )
            logger.error(detail)
            startup_error.add_note(detail)

    @staticmethod
    def _wait_for_refs(
        refs,
        *,
        deadline: SGLangStartupDeadline | SGLangRefitDeadline | None,
        stage: str,
        cancel_on_error: bool = True,
    ):
        if deadline is None:
            return ray.get(refs)
        return deadline.ray_get(
            refs,
            stage=stage,
            cancel_on_error=cancel_on_error,
        )

    def get_rollout_engine_urls(self) -> list[str]:
        """Resolve node-0 engine HTTP base URLs once on the driver."""
        return ray.get([e.get_base_url.remote() for e in self.rollout_engines])

    def _start_engines(
        self,
        port_cursors: dict[int, int] | None = None,
        *,
        deadline: SGLangStartupDeadline | SGLangRefitDeadline,
    ) -> tuple[list, dict[int, int]]:
        """Create Ray actors, allocate ports, and fire ``engine.init()`` without waiting.

        Returns ``(init_handles, port_cursors)`` where *init_handles* is a list
        of Ray ObjectRefs and *port_cursors* maps node index -> next free port.
        """
        if port_cursors is None:
            port_cursors = {}
        num_gpu_per_engine = min(self.num_gpus_per_engine, self.num_gpus_per_node)
        pg = self.pg
        reordered_bundle_indices = self.pg_reordered_bundle_indices
        reordered_gpu_ids = self.pg_reordered_gpu_ids

        # Resolve the SGLang venv ONCE (mirrors the once-per-node RayWorkerBuilder path).
        sglang_runtime_env_base = make_actor_runtime_env(
            "nemo_rl.models.generation.sglang.sglang_worker.SGLangGenerationWorker"
        )
        sglang_runtime_env_base.update(
            get_nsight_config_if_pattern_matches("sglang_generation_worker")
        )

        local_all_engines = []
        for i in range(len(self.all_engines)):
            if self.all_engines[i] is not None:
                continue

            global_rank = self.rank_offset + i
            num_gpus = min(0.2, 1 / self.cluster.max_colocated_worker_groups)
            num_cpus = num_gpus

            gpu_index = self.gpu_offset + i * num_gpu_per_engine
            base_gpu_id = int(reordered_gpu_ids[gpu_index])

            scheduling_strategy = PlacementGroupSchedulingStrategy(
                placement_group=pg,
                placement_group_capture_child_tasks=True,
                placement_group_bundle_index=reordered_bundle_indices[gpu_index],
            )

            env_vars = {name: "1" for name in NOSET_VISIBLE_DEVICES_ENV_VARS_LIST} | {
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

            # Explicitly pass CUDA_VISIBLE_DEVICES through to the engine actor so
            # all engines see the same global value (Ray would otherwise remap it
            # because we set the NOSET_* flags above).
            # Trainer and engine must agree on the NCCL transport; sglang's
            # scheduler subprocess defaults to NCCL_CUMEM_ENABLE=0.
            env_vars["NCCL_CUMEM_ENABLE"] = "0"

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
            init_args = (self.num_gpus_per_node, self.sglang_cfg)
            init_kwargs = {
                "rank": global_rank,
                "base_gpu_id": base_gpu_id,
                "num_gpus_per_engine": self.num_gpus_per_engine,
            }

            # Create worker actor directly — sglang_worker.py uses lazy imports
            # so it's importable in SYSTEM env; the actor runs in sglang env.
            engine = SGLangGenerationWorker.options(**actor_options).remote(
                *init_args, **init_kwargs
            )

            local_all_engines.append((global_rank, engine))
            self.all_engines[i] = engine

        self.num_new_engines = len(local_all_engines)

        if self.num_new_engines == 0:
            return [], port_cursors

        # SGLang engine server/NCCL/dist_init ports come from the reserved
        # generation band (3000-4999 by default), below the ephemeral floor;
        # see the port layout in virtual_cluster.py.
        gen_port_low = self.sglang_cfg.get("port_range_low")
        if gen_port_low is None:
            gen_port_low = DEFAULT_GENERATION_PORT_RANGE_LOW
        gen_port_high = self.sglang_cfg.get("port_range_high")
        if gen_port_high is None:
            gen_port_high = DEFAULT_GENERATION_PORT_RANGE_HIGH
        addr_and_ports, port_cursors = _allocate_rollout_engine_addr_and_ports_normal(
            gpus_per_node=self.num_gpus_per_node,
            sglang_cfg=self.sglang_cfg,
            local_all_engines=local_all_engines,
            rank_offset=self.rank_offset,
            port_range_low=gen_port_low,
            port_range_high=gen_port_high,
            node_port_cursor=port_cursors,
            deadline=deadline,
        )

        init_handles = [
            engine.init.remote(
                **(addr_and_ports[rank]),
                router_ip=self.router_ip,
                router_port=self.router_port,
                startup_timeout_s=deadline.remaining(
                    f"dispatching initialization for engine rank {rank}"
                ),
            )
            for rank, engine in local_all_engines
        ]
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

    def _logical_engine_indices(self, indices: list[int]) -> list[int]:
        """Expand physical ranks to every rank of each affected logical engine."""
        expanded: set[int] = set()
        for index in indices:
            logical_engine = index // self.nodes_per_engine
            first_rank = logical_engine * self.nodes_per_engine
            expanded.update(
                range(
                    first_rank,
                    min(first_rank + self.nodes_per_engine, len(self.all_engines)),
                )
            )
        return sorted(expanded)

    def _latch_engine_cleanup_failure(self, reason: str) -> None:
        """Permanently disable in-place recovery after unconfirmed cleanup."""
        if getattr(self, "_engine_cleanup_error", None) is None:
            self._engine_cleanup_error = reason
        logger.error(
            "SGLang engine cleanup was not confirmed; automatic recovery is "
            f"disabled: {reason}"
        )

    def _quarantine_engine_indices(
        self,
        indices: list[int],
        *,
        timeout_s: float = _BEST_EFFORT_REFIT_CLEANUP_TIMEOUT_S,
    ) -> bool:
        """Detach, bounded-shutdown, and kill the selected physical engine actors."""
        selected = sorted(set(indices))
        actors = [
            (index, self.all_engines[index])
            for index in selected
            if 0 <= index < len(self.all_engines)
            and self.all_engines[index] is not None
        ]

        # Fail closed immediately: no caller may reuse an actor once quarantine
        # starts, even if graceful cleanup later times out.
        for index, _ in actors:
            self.all_engines[index] = None
        self.num_new_engines = 0

        if not actors:
            return getattr(self, "_engine_cleanup_error", None) is None

        timeout_s = max(float(timeout_s), 0.001)
        shutdown_refs = []
        ok = True
        try:
            shutdown_refs = [
                actor.shutdown.remote(timeout_s=timeout_s) for _, actor in actors
            ]
            results = ray.get(shutdown_refs, timeout=timeout_s + 1)
            if any(result is False for result in results):
                ok = False
        except Exception as exc:
            logger.warning(f"Bounded SGLang engine shutdown failed: {exc}")
            ok = False
        finally:
            for index, actor in actors:
                try:
                    ray.kill(actor, no_restart=True)
                except Exception as exc:
                    logger.warning(
                        f"Failed to kill quarantined SGLang actor at index {index}: "
                        f"{exc}"
                    )
                    ok = False
        if not ok:
            self._latch_engine_cleanup_failure(
                "bounded shutdown or Ray actor termination failed during quarantine"
            )
        return ok

    def quarantine_all_engines(
        self,
        *,
        timeout_s: float = _BEST_EFFORT_REFIT_CLEANUP_TIMEOUT_S,
    ) -> bool:
        """Fail closed by discarding every physical rank in the refit group.

        The caller must retain its refit health-monitor lease. A quarantined
        engine may contain only a prefix of the new weight stream and must
        never resume generation.
        """
        return self._quarantine_engine_indices(
            list(range(len(self.all_engines))),
            timeout_s=timeout_s,
        )

    def _recover(self, *, deadline: SGLangRefitDeadline | None = None) -> None:
        """Recover complete logical engines and roll back every failed attempt."""
        cleanup_error = getattr(self, "_engine_cleanup_error", None)
        if cleanup_error is not None:
            raise RuntimeError(
                "SGLang engine cleanup was not confirmed after an earlier "
                "failure; automatic recovery is disabled to avoid overlapping "
                f"engine processes: {cleanup_error}"
            )

        missing_indices = [
            i for i, engine in enumerate(self.all_engines) if engine is None
        ]
        if not missing_indices:
            self.num_new_engines = 0
            return

        restart_indices = self._logical_engine_indices(missing_indices)
        existing_peer_indices = [
            index for index in restart_indices if self.all_engines[index] is not None
        ]
        if existing_peer_indices and not self._quarantine_engine_indices(
            restart_indices
        ):
            raise RuntimeError(
                "Could not safely stop every rank of a partially failed SGLang "
                "logical engine; automatic restart is disabled."
            )

        port_cursors: dict[int, int] = {}
        engine_start_deadline = deadline or SGLangStartupDeadline(
            self.engine_startup_timeout_s
        )
        try:
            handles, _ = self._start_engines(
                port_cursors,
                deadline=engine_start_deadline,
            )
            if handles:
                self._wait_for_refs(
                    handles,
                    deadline=engine_start_deadline,
                    stage="restarting failed rollout engines",
                )

            assert self.num_new_engines == len(restart_indices), (
                "num_new_engines does not match the complete logical-engine restart set"
            )

            if self.needs_offload:
                new_engines = [self.all_engines[i] for i in restart_indices]
                assert all(engine is not None for engine in new_engines)
                timeout_s = (
                    deadline.remaining("offloading recovered engine weights")
                    if deadline is not None
                    else None
                )
                self._wait_for_refs(
                    [
                        engine.release_memory_occupation.remote(
                            tags=["weights"], timeout_s=timeout_s
                        )
                        for engine in new_engines
                    ],
                    deadline=deadline,
                    stage="offloading recovered engine weights",
                )
                timeout_s = (
                    deadline.remaining("offloading recovered engine KV caches")
                    if deadline is not None
                    else None
                )
                self._wait_for_refs(
                    [
                        engine.release_memory_occupation.remote(
                            tags=["kv_cache"], timeout_s=timeout_s
                        )
                        for engine in new_engines
                    ],
                    deadline=deadline,
                    stage="offloading recovered engine KV caches",
                )
                timeout_s = (
                    deadline.remaining("restoring recovered engine weights")
                    if deadline is not None
                    else None
                )
                self._wait_for_refs(
                    [
                        engine.resume_memory_occupation.remote(
                            tags=["weights"], timeout_s=timeout_s
                        )
                        for engine in new_engines
                    ],
                    deadline=deadline,
                    stage="restoring recovered engine weights",
                )
        except BaseException as exc:
            if not self._quarantine_engine_indices(restart_indices):
                exc.add_note(
                    "Rollback could not confirm clean shutdown of every newly "
                    "staged SGLang actor. Their slots remain quarantined."
                )
            raise

    def get_updatable_engines_and_lock(self):
        """Return engines eligible for weight updates."""
        return (
            self.engines,
            self.rollout_engine_lock,
            self.num_new_engines,
            self.engine_gpu_counts,
            self.engine_gpu_offsets,
        )

    def recover_updatable_engines(self, *, deadline: SGLangRefitDeadline | None = None):
        """Restart any dead rollout engines and update ``num_new_engines``."""
        self._recover(deadline=deadline)

        return (
            self.engines,
            self.rollout_engine_lock,
            self.num_new_engines,
            self.engine_gpu_counts,
            self.engine_gpu_offsets,
        )

    def clear_updatable_num_new_engines(self):
        # When fault tolerance is not enabled, num_new_engines must be cleared
        # manually after the refit dispatch connects the new engines.
        self.num_new_engines = 0

    def pause_generation(
        self,
        mode: Optional[str] = None,
        *,
        deadline: SGLangRefitDeadline | None = None,
    ) -> None:
        """Pause generation on every node-0 engine.

        Args:
            mode: Pause mode override. When ``None`` (default), the mode
                configured in ``sglang_server_config.pause_generation_mode``
                is used. Callers (e.g. the SGLang refit dispatch helpers)
                pass an explicit mode when they also need to gate follow-up
                steps such as ``invalidate_kv_cache`` on the same value.
        """
        engines = [e for e in self.engines if e is not None]
        if not engines:
            return
        if mode is None:
            mode = self.pause_generation_mode
        timeout_s = (
            deadline.remaining("dispatching generation pause")
            if deadline is not None
            else None
        )
        refs = [
            e.pause_generation.remote(mode=mode, timeout_s=timeout_s) for e in engines
        ]
        self._wait_for_refs(
            refs,
            deadline=deadline,
            stage="waiting for generation pause",
        )

    def continue_generation(
        self,
        *,
        deadline: SGLangRefitDeadline | None = None,
        best_effort: bool = False,
    ) -> None:
        """Resume generation on every node-0 engine."""
        engines = [e for e in self.engines if e is not None]
        if not engines:
            return
        if deadline is None:
            timeout_s = None
            wait_deadline = None
        elif best_effort:
            cleanup_timeout_s = max(
                deadline.remaining_or_zero(), _BEST_EFFORT_REFIT_CLEANUP_TIMEOUT_S
            )
            wait_deadline = SGLangRefitDeadline(cleanup_timeout_s)
            timeout_s = wait_deadline.remaining("dispatching generation resume")
        else:
            timeout_s = deadline.remaining("dispatching generation resume")
            wait_deadline = deadline
        refs = [e.continue_generation.remote(timeout_s=timeout_s) for e in engines]
        self._wait_for_refs(
            refs,
            deadline=wait_deadline,
            stage="waiting for generation resume",
            cancel_on_error=True,
        )

    def post_process_weights(
        self,
        *,
        restore_weights_before_load: bool = False,
        post_process_quantization: bool = True,
        deadline: SGLangRefitDeadline | None = None,
    ) -> None:
        """Run SGLang's ``/post_process_weights`` RPC on every node-0 engine.

        Called by the refit dispatch helpers after a colocate IPC or
        distributed broadcast refit so SGLang finalizes its weight tables
        (e.g. materializes quantized scales, swaps in the fresh buffer).
        """
        engines = [e for e in self.engines if e is not None]
        if not engines:
            return
        timeout_s = (
            deadline.remaining("dispatching weight post-processing")
            if deadline is not None
            else None
        )
        refs = [
            e.post_process_weights.remote(
                restore_weights_before_load=restore_weights_before_load,
                post_process_quantization=post_process_quantization,
                timeout_s=timeout_s,
            )
            for e in engines
        ]
        self._wait_for_refs(
            refs,
            deadline=deadline,
            stage="waiting for weight post-processing",
        )

    def health_monitoring_suspend_for_refit(
        self, *, deadline: SGLangRefitDeadline | None = None
    ) -> str | None:
        """Quiesce health checks and return a unique refit suspension lease.

        The lease remains registered if quiescence times out. This fail-closed
        latch prevents a later memory-onload call from re-enabling checks while
        a health action or failed refit may still own engine state.
        """
        if not self._health_monitor:
            return None

        lease = f"refit:{uuid.uuid4().hex}"
        with self._health_monitor_state_lock:
            self._health_monitor_suspensions[lease] = "refit"
            timeout_s = (
                deadline.remaining("quiescing rollout health monitoring")
                if deadline is not None
                else self._health_monitor.default_quiesce_timeout_s
            )
            self._health_monitor.pause(timeout_s=timeout_s)
        return lease

    def health_monitoring_release_refit(self, lease: str | None) -> None:
        """Release one successfully completed refit's monitor suspension."""
        if not self._health_monitor or lease is None:
            return
        with self._health_monitor_state_lock:
            reason = self._health_monitor_suspensions.get(lease)
            if reason != "refit":
                raise RuntimeError(f"Unknown SGLang refit monitor lease: {lease}")
            del self._health_monitor_suspensions[lease]
            if not self._health_monitor_suspensions:
                self._health_monitor.resume()

    def _health_monitoring_suspend_for_offload(self) -> None:
        """Quiesce checks before releasing engine memory."""
        if not self._health_monitor:
            return
        with self._health_monitor_state_lock:
            self._health_monitor_suspensions[_OFFLOAD_MONITOR_SUSPENSION] = (
                _OFFLOAD_MONITOR_SUSPENSION
            )
            self._health_monitor.pause(
                timeout_s=self._health_monitor.default_quiesce_timeout_s
            )

    def _health_monitoring_release_offload(self) -> None:
        """Release only the memory-offload suspension after a safe onload."""
        if not self._health_monitor:
            return
        with self._health_monitor_state_lock:
            self._health_monitor_suspensions.pop(_OFFLOAD_MONITOR_SUSPENSION, None)
            if not self._health_monitor_suspensions:
                self._health_monitor.resume()

    def shutdown(self) -> bool:
        ok = True
        if getattr(self, "_owns_runtime", True):
            health_monitor = getattr(self, "_health_monitor", None)
            if health_monitor:
                health_monitor.stop()

            all_engines = getattr(self, "all_engines", [])
            if not self._quarantine_engine_indices(list(range(len(all_engines)))):
                ok = False

            router_actor = getattr(self, "_router_actor", None)
            if router_actor is not None:
                try:
                    ray.get(
                        router_actor.stop.remote(),
                        timeout=_BEST_EFFORT_REFIT_CLEANUP_TIMEOUT_S,
                    )
                except Exception as e:
                    logger.warning(f"Router terminate failed: {e}")
                    ok = False
                finally:
                    try:
                        ray.kill(router_actor, no_restart=True)
                    except Exception as e:
                        logger.warning(f"Router kill failed: {e}")
                        ok = False
                self._router_actor = None

        http_client = getattr(self, "_http_client", None)
        async_loop = getattr(self, "_async_loop", None)
        if http_client is not None:
            try:
                if async_loop is not None:
                    async_loop.run(http_client.aclose())
                else:
                    http_client.shutdown()
            except Exception as e:
                logger.warning(f"HTTP client shutdown failed: {e}")
                ok = False
            self._http_client = None

        if async_loop is not None:
            try:
                async_loop.close()
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
        top_k_cfg = self.sglang_cfg["top_k"]
        top_k_val = 1 if greedy else (top_k_cfg if top_k_cfg is not None else -1)

        # Build sampling params dict first, then patch in optional fields so we
        # never reference ``sampling_params`` before it's bound.
        sampling_params: dict[str, Any] = {
            "temperature": temperature,
            "top_p": self.sglang_cfg["top_p"],
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
        context_length = self.sglang_cfg["sglang_cfg"]["context_length"]
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

    def prepare_refit_info(self, state_dict_info: dict[str, Any]) -> None:
        pass

    def update_weights_via_ipc_zmq(self) -> list[ray.ObjectRef]:
        return []

    def update_weights_from_collective(self) -> list[ray.ObjectRef]:
        return []

    def prepare_for_generation(self, *args: Any, **kwargs: Any) -> bool:
        """Wake workers up for colocated inference."""
        if not self.needs_offload:
            return
        tags = kwargs.get("tags", None)
        engines = [e for e in self.engines if e is not None]
        if not engines:
            return
        ray.get([e.resume_memory_occupation.remote(tags=tags) for e in engines])
        # ``tags=["weights"]`` is only the first half of colocated onload.
        # Health requests require the KV cache to be active as well.
        if tags is None or "kv_cache" in tags:
            self._health_monitoring_release_offload()

    def finish_generation(self, *args: Any, **kwargs: Any) -> bool:
        """Sleep workers and reset prefix cache."""
        if not self.needs_offload:
            return
        tags = kwargs.get("tags", None)
        engines = [e for e in self.engines if e is not None]
        if not engines:
            return
        # Quiesce before releasing any resource a health request may touch.
        # A failed release intentionally leaves monitoring paused.
        self._health_monitoring_suspend_for_offload()
        ray.get([e.release_memory_occupation.remote(tags=tags) for e in engines])

    def invalidate_kv_cache(
        self, *, deadline: SGLangRefitDeadline | None = None
    ) -> bool:
        """Invalidate KV cache before weight updates (Megatron-style).

        Flushes the cache on every node-0 engine so stale KV entries are
        discarded before new weights land. Returns ``True`` iff every engine
        reports success.
        """
        engines = [e for e in self.engines if e is not None]
        if not engines:
            return True
        timeout_s = (
            deadline.remaining("dispatching SGLang KV-cache invalidation")
            if deadline is not None
            else None
        )
        refs = [e.invalidate_kv_cache.remote(timeout_s=timeout_s) for e in engines]
        try:
            self._wait_for_refs(
                refs,
                deadline=deadline,
                stage="waiting for SGLang KV-cache invalidation",
            )
        except SGLangRefitTimeoutError:
            raise
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
        """Generate using traditional SGLang router with token-based workflow."""
        url = f"http://{self.router_ip}:{self.router_port}/generate"

        # Prepare payload for sglang server
        payload = {
            "sampling_params": sampling_params,
            "return_logprob": True,
            "input_ids": input_ids,
        }

        output = await self._http_client.post(url, payload)

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
