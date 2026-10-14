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

"""NCCL refit protocol for a fixed managed Dynamo worker fleet."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import ray

from nemo_rl.distributed.refit_watchdog import REFIT_ABORTED_TOKEN, RefitAborted
from nemo_rl.models.generation.dynamo.config import (
    VLLM_PACKED_BUFFER_SIZE_BYTES,
    VLLM_PACKED_NUM_BUFFERS,
)
from nemo_rl.models.generation.dynamo.http_client import (
    format_dynamo_error,
    http_post_json,
)
from nemo_rl.models.generation.interfaces import CollectiveSenderSpec
from nemo_rl.utils.staged_payload import encode_staged_payload
from nemo_rl.weight_sync.membership import RefitMembership

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


@ray.remote(num_cpus=0)
def _update_worker_weights(  # pragma: no cover
    *,
    system_url: str,
    update_info: dict[str, Any],
    timeout_s: float,
) -> bool:
    common = {"allow_unpaused": True, "reset_prefix_cache": False}
    steps: tuple[tuple[str, dict[str, Any]], ...] = (
        ("start_weight_update", {}),
        ("update_weights", {"update_info": update_info}),
        ("finish_weight_update", {}),
    )
    for engine_rpc, kwargs in steps:
        response = http_post_json(
            f"{system_url}/engine/update_weights_from_distributed",
            {"engine_rpc": engine_rpc, **common, **kwargs},
            timeout_s,
        )
        if response.get("status") != "ok":
            raise RuntimeError(
                f"Dynamo worker {system_url} RPC {engine_rpc} failed: "
                f"{format_dynamo_error(response)}"
            )
    return True


class DynamoRefitChannel:
    """Closed refit protocol shared by driver and serialized rollout copies."""

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
        self._update_info: dict[str, Any] | None = None

    def client_copy(self) -> "DynamoRefitChannel":
        """Return a serializable copy without the driver-owned worker validator.

        The prepared update metadata travels with the copy: SingleController
        prepares refit on the driver and then pickles the synchronizer (and
        this generation client) into its actor, which issues every update.
        """
        copy = DynamoRefitChannel(
            self._workers,
            engine_world_size=self._engine_world_size,
            control_timeout_s=self._control_timeout_s,
        )
        copy._update_info = self._update_info
        return copy

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

    def prepare_nccl_reshard_refit_info(self, wire_refit_info: dict[str, Any]) -> None:
        """Stage the wire-safe reshard plan on every engine and build its map."""
        staged = encode_staged_payload(wire_refit_info)
        ray.get(
            [
                _stage_and_post_worker_route.remote(
                    system_url=worker.system_url,
                    key=staged.key,
                    chunks=staged.chunks,
                    payload={
                        "engine_rpc": "nrl_prepare_nccl_reshard_refit_info_from_stage",
                        "key": staged.key,
                        "sha256": staged.sha256,
                    },
                    timeout_s=self._init_timeout_s,
                )
                for worker in self._validated_workers()
            ]
        )

    def nccl_reshard_refit(self, refit_timeout_s: float | None) -> list[ray.ObjectRef]:
        """Receive one nccl_reshard refit on every engine."""
        timeout_s = self._control_timeout_s
        if refit_timeout_s is not None:
            timeout_s = max(timeout_s, refit_timeout_s + _REFIT_CLIENT_MARGIN_S)
        return [
            _post_worker_route.remote(
                system_url=worker.system_url,
                route=_UPDATE_ROUTE,
                payload={
                    "engine_rpc": "nrl_nccl_reshard_refit",
                    "allow_unpaused": True,
                    "reset_prefix_cache": False,
                    "refit_timeout_s": refit_timeout_s,
                },
                timeout_s=timeout_s,
            )
            for worker in self._validated_workers()
        ]

    @property
    def sender_spec(self) -> CollectiveSenderSpec:
        """Return the peer and packing geometry declared to workers at init."""
        return CollectiveSenderSpec(
            nccl_peer="vllm",
            buffer_size_bytes=VLLM_PACKED_BUFFER_SIZE_BYTES,
            num_buffers=VLLM_PACKED_NUM_BUFFERS,
        )

    def prepare(self, state_dict_info: dict[str, Any] | None) -> None:
        if state_dict_info is None:
            raise ValueError("state_dict_info must not be None for Dynamo refit")
        names: list[str] = []
        dtype_names: list[str] = []
        shapes: list[list[int]] = []
        for name, (shape, dtype) in state_dict_info.items():
            names.append(name)
            dtype_names.append(str(dtype).removeprefix("torch."))
            shapes.append(list(shape))
        self._update_info = {
            "names": names,
            "dtype_names": dtype_names,
            "shapes": shapes,
        }

    def init_collective(
        self,
        ip: str,
        port: int,
        world_size: int,
        *,
        train_world_size: int,
    ) -> list[ray.ObjectRef]:
        expected_world_size = train_world_size + self.inference_world_size
        if world_size != expected_world_size:
            raise ValueError(
                f"NCCL world_size={world_size} does not match expected "
                f"{expected_world_size}"
            )
        workers = self._validated_workers()
        return [
            _post_worker_route.remote(
                system_url=worker.system_url,
                route="init_weights_update_group",
                payload={
                    "engine_rpc": "init_weight_transfer_engine",
                    "init_info": {
                        "master_address": ip,
                        "master_port": port,
                        "rank_offset": train_world_size
                        + worker_index * self._engine_world_size,
                        "world_size": world_size,
                        # vLLM reads packing once at init and defaults it off;
                        # omitting it desynchronizes the packed broadcast.
                        "packed": True,
                        "packed_buffer_size_bytes": VLLM_PACKED_BUFFER_SIZE_BYTES,
                        "packed_num_buffers": VLLM_PACKED_NUM_BUFFERS,
                    },
                },
                timeout_s=self._control_timeout_s,
            )
            for worker_index, worker in enumerate(workers)
        ]

    def update_weights(self) -> list[ray.ObjectRef]:
        if self._update_info is None:
            raise RuntimeError(
                "prepare_refit_info() must be called before Dynamo weight updates"
            )
        return [
            _update_worker_weights.remote(
                system_url=worker.system_url,
                update_info=self._update_info,
                timeout_s=self._control_timeout_s,
            )
            for worker in self._validated_workers()
        ]

    def _pause_workers(
        self, *, clear_cache: bool
    ) -> tuple[list[DynamoWorkerEndpoint], list[str]]:
        """Drain in-flight requests, optionally clear caches; new requests wait."""
        pause_futures = [
            _post_worker_route.remote(
                system_url=worker.system_url,
                route="pause_generation",
                payload={"mode": "wait", "clear_cache": clear_cache},
                timeout_s=self._control_timeout_s,
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

    def _resume_workers(self, workers: Sequence[DynamoWorkerEndpoint]) -> list[str]:
        resume_futures = [
            (
                worker,
                _post_worker_route.remote(
                    system_url=worker.system_url,
                    route="resume_generation",
                    payload={},
                    timeout_s=self._control_timeout_s,
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

    def pause(self, *, clear_cache: bool) -> None:
        """Drain and pause every worker ahead of a weight update.

        Workers that confirmed the pause are resumed if any peer fails.
        """
        paused_workers, pause_errors = self._pause_workers(clear_cache=clear_cache)
        if pause_errors:
            resume_errors = self._resume_workers(paused_workers)
            details = "pause/clear failed for " + "; ".join(pause_errors)
            if resume_errors:
                details += "; resume failed for " + "; ".join(resume_errors)
            raise RuntimeError(f"Dynamo pause before weight update failed: {details}")

    def resume(self) -> None:
        """Resume every worker after a weight update."""
        resume_errors = self._resume_workers(self._workers)
        if resume_errors:
            raise RuntimeError(
                "Dynamo resume after weight update failed for "
                + "; ".join(resume_errors)
            )

    def flush_cache(self) -> bool:
        """Drain, clear, and resume every worker using immutable endpoints."""
        paused_workers, pause_errors = self._pause_workers(clear_cache=True)
        resume_errors = self._resume_workers(paused_workers)

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
