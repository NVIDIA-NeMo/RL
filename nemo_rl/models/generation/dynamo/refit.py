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

from nemo_rl.models.generation.dynamo.config import (
    VLLM_PACKED_BUFFER_SIZE_BYTES,
    VLLM_PACKED_NUM_BUFFERS,
)
from nemo_rl.models.generation.dynamo.http_client import (
    format_dynamo_error,
    http_post_json,
)
from nemo_rl.models.generation.interfaces import CollectiveSenderSpec


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


@ray.remote(num_cpus=0)
def _post_worker_route(  # pragma: no cover
    *,
    system_url: str,
    route: str,
    payload: dict[str, Any],
    timeout_s: float,
) -> bool:
    response = http_post_json(
        f"{system_url}/engine/{route}",
        payload,
        timeout_s,
    )
    if response.get("status") != "ok":
        raise RuntimeError(
            f"Dynamo worker {system_url} route {route} failed: "
            f"{format_dynamo_error(response)}"
        )
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
        ("start_weight_update", {"is_checkpoint_format": True}),
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
    def sender_spec(self) -> CollectiveSenderSpec:
        """Return vLLM's non-negotiated peer and packing contract."""
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
            "packed": True,
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

    def flush_cache(self) -> bool:
        """Drain, clear, and resume every worker using immutable endpoints."""
        pause_futures = [
            _post_worker_route.remote(
                system_url=worker.system_url,
                route="pause_generation",
                payload={"mode": "wait", "clear_cache": True},
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
            for worker in paused_workers
        ]
        resume_errors: list[str] = []
        for worker, future in resume_futures:
            try:
                ray.get(future)
            except Exception as error:
                resume_errors.append(f"{worker.system_url}: {error}")

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
