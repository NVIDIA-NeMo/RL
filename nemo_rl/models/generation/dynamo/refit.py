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

"""NCCL refit for a fixed managed Dynamo worker fleet.

Every managed engine loads NeMo RL's vLLM worker extension
(``nemo_rl.models.generation.vllm.dynamo_worker_extension``). The driver calls its
methods through Dynamo's per-worker ``/engine/*`` RL routes, which forward an
``engine_rpc`` name to vLLM ``collective_rpc``; the weights themselves move over
NCCL, exactly as for the vLLM backend.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import ray

from nemo_rl.distributed.refit_watchdog import REFIT_ABORTED_TOKEN, RefitAborted
from nemo_rl.models.generation.dynamo.http_client import (
    format_dynamo_error,
    http_post_json,
)
from nemo_rl.utils.staged_payload import encode_staged_payload
from nemo_rl.weight_sync.membership import RefitMembership, desired_membership

# Dynamo's per-worker admin routes. Setup calls go to the init route, whose server
# side kills the worker if the call outlives DYN_RL_INIT_WEIGHTS_TIMEOUT_S; refits go
# to the update route, which has no server deadline.
_INIT_ROUTE = "init_weights_update_group"
_UPDATE_ROUTE = "update_weights_from_distributed"
# Margin past the server's init deadline, so the client never gives up first and
# leaves a worker that is about to exit looking merely slow.
_INIT_ROUTE_CLIENT_MARGIN_S = 30.0
# Margin past a refit's own deadline for the HTTP call that carries it.
_REFIT_CLIENT_MARGIN_S = 60.0


@dataclass(frozen=True)
class DynamoWorkerEndpoint:
    """Serializable identity and admin endpoint for one Dynamo vLLM engine."""

    instance_id: str
    system_url: str

    @classmethod
    def from_metadata(cls, metadata: dict[str, Any]) -> "DynamoWorkerEndpoint":
        return cls(
            instance_id=str(metadata["instance_id"]),
            system_url=str(metadata["system_url"]),
        )


def _post_route(
    system_url: str, route: str, payload: dict[str, Any], timeout_s: float
) -> None:
    """POST one admin route and raise unless Dynamo reports success.

    A refit the NeMo RL extension aborted reaches Dynamo as a plain exception and
    comes back only as its message, so it is re-raised as RefitAborted here.
    """
    response = http_post_json(f"{system_url}/engine/{route}", payload, timeout_s)
    if response.get("status") == "ok":
        return
    message = (
        f"Dynamo worker {system_url} route {route} "
        f"(engine_rpc={payload.get('engine_rpc')}) failed: "
        f"{format_dynamo_error(response)}"
    )
    if REFIT_ABORTED_TOKEN in message:
        raise RefitAborted(message)
    raise RuntimeError(message)


# No retries: re-POSTing a collective call after a worker died can never complete,
# and would hold the engine's admin lock behind a call nobody else will join.
@ray.remote(num_cpus=0, max_retries=0)
def _post_worker_route(  # pragma: no cover
    *,
    system_url: str,
    route: str,
    payload: dict[str, Any],
    timeout_s: float,
) -> bool:
    _post_route(system_url, route, payload, timeout_s)
    return True


@ray.remote(num_cpus=0, max_retries=0)
def _stage_and_post_worker_route(  # pragma: no cover
    *,
    system_url: str,
    key: str,
    chunks: Sequence[str],
    payload: dict[str, Any],
    timeout_s: float,
) -> bool:
    """Send a staged payload chunk by chunk, then the call that consumes it."""
    for index, data in enumerate(chunks):
        _post_route(
            system_url,
            _INIT_ROUTE,
            {
                "engine_rpc": "nrl_stage_chunk",
                "key": key,
                "index": index,
                "total": len(chunks),
                "data": data,
            },
            timeout_s,
        )
    _post_route(system_url, _INIT_ROUTE, payload, timeout_s)
    return True


class DynamoRefitChannel:
    """Refit calls to every engine, shared by the driver and pickled client copies.

    Refit metadata is prepared once and kept inside the engines, so a client copy
    that SingleController ships to its actor needs only the endpoints.
    """

    def __init__(
        self,
        workers: Sequence[dict[str, Any] | DynamoWorkerEndpoint],
        *,
        engine_world_size: int,
        control_timeout_s: float,
        validate_workers: Callable[[list[dict[str, Any]]], list[dict[str, Any]]]
        | None = None,
    ) -> None:
        self._worker_metadata = [
            dict(worker) for worker in workers if isinstance(worker, dict)
        ]
        self._workers = tuple(
            DynamoWorkerEndpoint.from_metadata(worker)
            if isinstance(worker, dict)
            else worker
            for worker in workers
        )
        if not self._workers:
            raise ValueError("Dynamo refit requires at least one worker endpoint")
        self._engine_world_size = engine_world_size
        self._control_timeout_s = control_timeout_s
        self._validate_workers = validate_workers

    def client_copy(self) -> "DynamoRefitChannel":
        """Return a serializable endpoint-only channel for rollout actors."""
        return DynamoRefitChannel(
            self._workers,
            engine_world_size=self._engine_world_size,
            control_timeout_s=self._control_timeout_s,
        )

    def _validated_workers(self) -> tuple[DynamoWorkerEndpoint, ...]:
        if self._validate_workers is None:
            return self._workers
        current = self._validate_workers(self._worker_metadata)
        endpoints = tuple(DynamoWorkerEndpoint.from_metadata(item) for item in current)
        if endpoints != self._workers:
            raise RuntimeError(
                "Managed Dynamo worker membership changed after collective setup"
            )
        return endpoints

    @property
    def inference_world_size(self) -> int:
        return len(self._workers) * self._engine_world_size

    @property
    def num_engines(self) -> int:
        return len(self._workers)

    @property
    def _init_timeout_s(self) -> float:
        return self._control_timeout_s + _INIT_ROUTE_CLIENT_MARGIN_S

    def check_full_fleet(self, membership: RefitMembership) -> None:
        """Reject any membership but every engine, in order: the fleet is fixed."""
        full_fleet = {
            index: index * self._engine_world_size for index in range(self.num_engines)
        }
        if (
            membership.shard_prefixes != full_fleet
            or membership.workers_per_shard != self._engine_world_size
        ):
            raise NotImplementedError(
                "Managed Dynamo has a fixed worker fleet; refit membership must be "
                f"every engine {full_fleet} with {self._engine_world_size} ranks "
                f"each, got {membership.shard_prefixes} with "
                f"{membership.workers_per_shard}."
            )

    def _post_to_each_engine(
        self,
        membership: RefitMembership,
        engine_rpc: str,
        kwargs: dict[str, Any],
    ) -> list[ray.ObjectRef]:
        """Call a worker-extension method on every engine through the init route.

        Each engine gets its rank prefix; the extension adds its local rank to it.
        """
        self.check_full_fleet(membership)
        workers = self._validated_workers()
        return [
            _post_worker_route.remote(
                system_url=worker.system_url,
                route=_INIT_ROUTE,
                payload={
                    "engine_rpc": engine_rpc,
                    "rank_prefix": membership.shard_prefixes[index],
                    **kwargs,
                },
                timeout_s=self._init_timeout_s,
            )
            for index, worker in enumerate(workers)
        ]

    def call_on_every_engine(
        self, engine_rpc: str, kwargs: dict[str, Any]
    ) -> list[ray.ObjectRef]:
        """Call a worker-extension method on every engine through the init route."""
        return [
            _post_worker_route.remote(
                system_url=worker.system_url,
                route=_INIT_ROUTE,
                payload={"engine_rpc": engine_rpc, **kwargs},
                timeout_s=self._init_timeout_s,
            )
            for worker in self._validated_workers()
        ]

    def full_fleet_membership(self, train_world_size: int) -> RefitMembership:
        """Every engine, each a DP shard of TP x PP ranks after the trainers."""
        return desired_membership(
            absent_shards=[],
            dp_size=self.num_engines,
            total_gen_workers=self.inference_world_size,
            train_world_size=train_world_size,
        )

    def init_collective(
        self,
        ip: str,
        port: int,
        world_size: int,
        *,
        train_world_size: int,
    ) -> list[ray.ObjectRef]:
        """Join every engine to the NeMo RL ``model_update_group``."""
        expected_world_size = train_world_size + self.inference_world_size
        if world_size != expected_world_size:
            raise ValueError(
                f"NCCL world_size={world_size} does not match expected "
                f"{expected_world_size}"
            )
        return self.rebuild_collective(
            self.full_fleet_membership(train_world_size), ip, port
        )

    def rebuild_collective(
        self, membership: RefitMembership, ip: str, port: int
    ) -> list[ray.ObjectRef]:
        """Join every engine to the NeMo RL ``model_update_group``."""
        return self._post_to_each_engine(
            membership,
            "init_collective",
            {
                "ip": ip,
                "port": port,
                "world_size": membership.world_size,
                "train_world_size": membership.train_world_size,
            },
        )

    def rebuild_nccl_reshard_comm_group(
        self,
        membership: RefitMembership,
        *,
        pp_ips: list[str],
        pp_ports: list[int],
        pp_size: int,
        train_ranks_per_stage: int,
        sub_world_size: int,
    ) -> list[ray.ObjectRef]:
        """Join every engine to the per-training-PP-stage bulk reshard groups."""
        return self._post_to_each_engine(
            membership,
            "init_nccl_reshard_comm_group",
            {
                "pp_ips": list(pp_ips),
                "pp_ports": list(pp_ports),
                "pp_size": pp_size,
                "train_ranks_per_stage": train_ranks_per_stage,
                "sub_world_size": sub_world_size,
            },
        )

    def _stage_on_every_engine(self, payload: Any, consumer: str) -> None:
        """Stage ``payload`` on every engine and run the extension's consumer."""
        staged = encode_staged_payload(payload)
        ray.get(
            [
                _stage_and_post_worker_route.remote(
                    system_url=worker.system_url,
                    key=staged.key,
                    chunks=staged.chunks,
                    payload={
                        "engine_rpc": consumer,
                        "key": staged.key,
                        "sha256": staged.sha256,
                    },
                    timeout_s=self._init_timeout_s,
                )
                for worker in self._validated_workers()
            ]
        )

    def prepare_refit_info(self, state_dict_info: dict[str, Any] | None) -> None:
        """Stage the collective refit's ordered ``{name: (shape, dtype)}`` on every engine."""
        if state_dict_info is None:
            raise ValueError("state_dict_info must not be None for Dynamo refit")
        entries = [
            [name, list(shape), str(dtype).removeprefix("torch.")]
            for name, (shape, dtype) in state_dict_info.items()
        ]
        self._stage_on_every_engine(entries, "nrl_prepare_refit_info_from_stage")

    def prepare_nccl_reshard_refit_info(self, wire_refit_info: dict[str, Any]) -> None:
        """Stage the wire-safe reshard plan on every engine and build its map."""
        self._stage_on_every_engine(
            wire_refit_info, "nrl_prepare_nccl_reshard_refit_info_from_stage"
        )

    def _refit_every_engine(
        self,
        engine_rpc: str,
        refit_timeout_s: float | None,
        kwargs: dict[str, Any],
    ) -> list[ray.ObjectRef]:
        """Run one refit receive on every engine through the update route.

        The route has no server deadline; the extension's watchdog enforces
        ``refit_timeout_s``, and the HTTP call is given longer than that.
        """
        timeout_s = self._control_timeout_s
        if refit_timeout_s is not None:
            timeout_s = max(timeout_s, refit_timeout_s + _REFIT_CLIENT_MARGIN_S)
        return [
            _post_worker_route.remote(
                system_url=worker.system_url,
                route=_UPDATE_ROUTE,
                payload={
                    "engine_rpc": engine_rpc,
                    "allow_unpaused": True,
                    "reset_prefix_cache": False,
                    "refit_timeout_s": refit_timeout_s,
                    **kwargs,
                },
                timeout_s=timeout_s,
            )
            for worker in self._validated_workers()
        ]

    def update_weights_from_collective(
        self, refit_timeout_s: float | None, *, refit_with_reload_api: bool
    ) -> list[ray.ObjectRef]:
        """Receive one packed-broadcast refit on every engine."""
        return self._refit_every_engine(
            "nrl_update_weights_from_collective",
            refit_timeout_s,
            {"refit_with_reload_api": refit_with_reload_api},
        )

    def nccl_reshard_refit(self, refit_timeout_s: float | None) -> list[ray.ObjectRef]:
        """Receive one nccl_reshard refit on every engine."""
        return self._refit_every_engine("nrl_nccl_reshard_refit", refit_timeout_s, {})

    def _pause_workers(
        self, *, mode: str, clear_cache: bool, timeout_s: float
    ) -> tuple[list[DynamoWorkerEndpoint], list[str]]:
        """Pause every worker in vLLM's ``mode``; new requests wait for resume."""
        pause_futures = [
            _post_worker_route.remote(
                system_url=worker.system_url,
                route="pause_generation",
                payload={"mode": mode, "clear_cache": clear_cache},
                timeout_s=timeout_s,
            )
            for worker in self._workers
        ]
        paused_workers: list[DynamoWorkerEndpoint] = []
        pause_errors: list[str] = []
        for worker, future in zip(self._workers, pause_futures, strict=True):
            try:
                ray.get(future)
            except Exception as error:
                pause_errors.append(f"{worker.system_url}: {error}")
            else:
                paused_workers.append(worker)
        return paused_workers, pause_errors

    def _resume_workers(
        self, workers: Sequence[DynamoWorkerEndpoint], *, timeout_s: float
    ) -> list[str]:
        resume_futures = [
            (
                worker,
                _post_worker_route.remote(
                    system_url=worker.system_url,
                    route="resume_generation",
                    payload={},
                    timeout_s=timeout_s,
                ),
            )
            for worker in workers
        ]
        resume_errors: list[str] = []
        for worker, future in resume_futures:
            try:
                ray.get(future)
            except Exception as error:
                resume_errors.append(f"{worker.system_url}: {error}")
        return resume_errors

    def pause(self, *, clear_cache: bool, timeout_s: float | None) -> None:
        """Freeze every worker's requests in place ahead of a weight update.

        Frozen requests continue on the new weights after :meth:`resume`; with
        ``clear_cache`` their KV is dropped now, so they recompute it. Workers
        that confirmed the pause are resumed if any peer fails.
        """
        timeout_s = self._control_timeout_s if timeout_s is None else timeout_s
        paused_workers, pause_errors = self._pause_workers(
            mode="keep", clear_cache=clear_cache, timeout_s=timeout_s
        )
        if pause_errors:
            resume_errors = self._resume_workers(paused_workers, timeout_s=timeout_s)
            details = "pause/clear failed for " + "; ".join(pause_errors)
            if resume_errors:
                details += "; resume failed for " + "; ".join(resume_errors)
            raise RuntimeError(f"Dynamo pause before weight update failed: {details}")

    def resume(self, *, timeout_s: float | None) -> None:
        """Resume every worker after a weight update."""
        timeout_s = self._control_timeout_s if timeout_s is None else timeout_s
        resume_errors = self._resume_workers(self._workers, timeout_s=timeout_s)
        if resume_errors:
            raise RuntimeError(
                "Dynamo resume after weight update failed for "
                + "; ".join(resume_errors)
            )

    def flush_cache(self) -> bool:
        """Drain, clear, and resume every worker using immutable endpoints.

        Workers that confirmed the pause are resumed even if a peer fails.
        """
        paused_workers, pause_errors = self._pause_workers(
            mode="wait", clear_cache=True, timeout_s=self._control_timeout_s
        )
        resume_errors = self._resume_workers(
            paused_workers, timeout_s=self._control_timeout_s
        )

        if pause_errors or resume_errors:
            details = []
            if pause_errors:
                details.append("pause/clear failed for " + "; ".join(pause_errors))
            if resume_errors:
                details.append("resume failed for " + "; ".join(resume_errors))
            raise RuntimeError(
                "Dynamo KV cache invalidation failed: " + "; ".join(details)
            )
        return True
