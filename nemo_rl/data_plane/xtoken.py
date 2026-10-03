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
"""Bounded, synchronous single-sample xToken teacher payloads."""

from __future__ import annotations

import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterator, Literal

import torch
from pydantic import BaseModel, PositiveFloat, PositiveInt
from tensordict import TensorDict

from nemo_rl.data_plane.factory import build_data_plane_client
from nemo_rl.data_plane.interfaces import DataPlaneClient, DataPlaneConfig

if TYPE_CHECKING:
    from nemo_rl.distributed.batched_data_dict import BatchedDataDict
    from nemo_rl.models.policy import PolicyConfig
    from nemo_rl.models.policy.lm_policy import Policy

XTOKEN_LOGITS_FIELD = "xtoken_dense_logits"


class XTokenTransportConfig(BaseModel, extra="allow"):
    """Transport selection and bounds; defaults preserve the IPC path."""

    backend: Literal["ipc", "tq"] = "ipc"
    max_payload_bytes: PositiveInt = 64 * 1024 * 1024
    timeout_s: PositiveFloat = 120.0


@dataclass(frozen=True)
class XTokenTQReference:
    """One FP32 ``[1, teacher_sequence, teacher_vocab]`` object in TQ."""

    partition_id: str
    sample_id: str
    shape: tuple[int, int, int]
    producer_node_id: str
    put_seconds: float

    @property
    def nbytes(self) -> int:
        """Logical application payload bytes, excluding transport overhead."""
        return self.shape[0] * self.shape[1] * self.shape[2] * 4


@dataclass
class XTokenTQReceiveResult:
    """Student-owned IPC descriptors and receipt; contains no tensor payload."""

    handles: list[dict[str, Any]]
    consumer_node_id: str
    nbytes: int
    get_seconds: float
    buffer_bytes: int


def validate_tq_support(
    *,
    data_plane: DataPlaneConfig | None,
    policies: list[PolicyConfig],
    num_nodes: int,
    gpus_per_node: int,
    batch_size: int,
) -> None:
    """Reject unsupported layouts before allocating models or Ray bundles."""
    if data_plane is None or not data_plane["enabled"]:
        raise ValueError("xToken TQ requires data_plane.enabled=true")
    if data_plane["impl"] != "transfer_queue" or data_plane["backend"] != "simple":
        raise ValueError("xToken TQ supports only the TransferQueue simple backend")
    if len(policies) != 2 or num_nodes != 2 or gpus_per_node != 1 or batch_size != 1:
        raise ValueError(
            "xToken TQ requires one teacher, two nodes, one GPU per node and batch=1"
        )
    for policy in policies:
        dtensor = policy["dtensor_cfg"]
        if dtensor["enabled"] is not True:
            raise ValueError("xToken TQ requires DTensor v2")
        if (
            dtensor.get("_v2") is not True
            or dtensor["tensor_parallel_size"] != 1
            or dtensor["context_parallel_size"] != 1
            or policy["train_global_batch_size"] != 1
            or policy["train_micro_batch_size"] != 1
            or policy["dynamic_batching"]["enabled"]
            or policy["sequence_packing"]["enabled"]
        ):
            raise ValueError(
                "xToken TQ requires DTensor v2, TP=CP=DP=GBS=MBS=1, no packing/dynamic batching"
            )


def check_payload_size(*, seq_len: int, vocab_size: int, max_bytes: int) -> None:
    """Check the padded FP32 size before teacher inference or allocation."""
    if seq_len <= 0 or vocab_size <= 0:
        raise ValueError("xToken TQ sequence and vocabulary sizes must be positive")
    nbytes = seq_len * vocab_size * 4
    if nbytes > max_bytes:
        raise ValueError(
            f"xToken TQ payload {nbytes} bytes exceeds max_payload_bytes={max_bytes}; "
            "reduce the sequence length (chunked transport is not supported)"
        )


def publish_logits(
    client: DataPlaneClient,
    logits: torch.Tensor,
    *,
    partition_id: str,
    sample_id: str,
    producer_node_id: str,
    max_payload_bytes: int,
) -> XTokenTQReference:
    """Write one teacher row directly from its worker; return only metadata."""
    if (
        logits.layout != torch.strided
        or logits.ndim != 3
        or logits.shape[0] != 1
        or logits.dtype != torch.float32
    ):
        raise ValueError("xToken TQ requires one dense FP32 [1, T, V] payload")
    check_payload_size(
        seq_len=logits.shape[1],
        vocab_size=logits.shape[2],
        max_bytes=max_payload_bytes,
    )
    started = time.perf_counter()
    client.put_samples(
        sample_ids=[sample_id],
        partition_id=partition_id,
        fields=TensorDict(
            {XTOKEN_LOGITS_FIELD: logits.detach().cpu().contiguous()}, batch_size=[1]
        ),
    )
    return XTokenTQReference(
        partition_id=partition_id,
        sample_id=sample_id,
        shape=(1, logits.shape[1], logits.shape[2]),
        producer_node_id=producer_node_id,
        put_seconds=time.perf_counter() - started,
    )


def fetch_logits(
    client: DataPlaneClient,
    reference: XTokenTQReference,
    *,
    consumer_node_id: str,
    max_payload_bytes: int,
) -> torch.Tensor:
    """Read the exact teacher row, rejecting local or malformed transfers."""
    if reference.producer_node_id == consumer_node_id:
        raise ValueError("xToken TQ teacher and student must be on different nodes")
    if reference.shape[0] != 1:
        raise ValueError("xToken TQ references must describe exactly one sample")
    check_payload_size(
        seq_len=reference.shape[1],
        vocab_size=reference.shape[2],
        max_bytes=max_payload_bytes,
    )
    fields = client.get_samples(
        sample_ids=[reference.sample_id],
        partition_id=reference.partition_id,
        select_fields=[XTOKEN_LOGITS_FIELD],
    )
    logits = fields[XTOKEN_LOGITS_FIELD]
    if (
        not isinstance(logits, torch.Tensor)
        or tuple(logits.shape) != reference.shape
        or logits.dtype != torch.float32
        or logits.layout != torch.strided
    ):
        raise ValueError("xToken TQ payload shape/dtype does not match its reference")
    return logits


def select_tq_nodes(
    nodes: list[dict[str, Any]],
) -> tuple[dict[str, float], dict[str, float]]:
    """Select two distinct live GPU nodes using their advertised node resources."""
    candidates = sorted(
        (n for n in nodes if n["Alive"] and n["Resources"].get("GPU", 0) >= 1),
        key=lambda n: n["NodeID"],
    )
    if len(candidates) < 2:
        raise ValueError(
            "xToken TQ requires two live Ray nodes with at least one GPU each"
        )
    constraints = []
    for node in candidates[:2]:
        resource = f"node:{node['NodeManagerAddress']}"
        if resource not in node["Resources"]:
            raise ValueError(f"Ray node {node['NodeID']} does not advertise {resource}")
        constraints.append({resource: 0.001})
    return constraints[0], constraints[1]


class XTokenTQTransport:
    """Driver-owned lifetime for synchronous TQ payloads and their workers."""

    def __init__(
        self,
        *,
        config: XTokenTransportConfig,
        data_plane: DataPlaneConfig,
        teacher: Policy,
        student: Policy,
    ) -> None:
        self.config = config
        self.data_plane = data_plane
        self.teacher = teacher
        self.student = student
        self.partition_id = f"xtoken-{uuid.uuid4().hex}"
        self.client: DataPlaneClient | None = None
        self._steps: list[str] = []
        self.metrics: dict[str, float] = {}
        self._failed = False
        self._stopped = False

    def __enter__(self) -> XTokenTQTransport:
        # Ray is optional for the pure payload helpers and CPU unit tests.
        import ray

        try:
            self.client = build_data_plane_client(self.data_plane, bootstrap=True)
            self.client.register_partition(
                self.partition_id, [XTOKEN_LOGITS_FIELD], 1, []
            )
            for policy in (self.teacher, self.student):
                ray.get(
                    policy.worker_group.run_all_workers_single_data(
                        "setup_data_plane", cfg=self.data_plane
                    ),
                    timeout=self.config.timeout_s,
                )
        except BaseException:
            try:
                self._stop_workers()
            finally:
                if self.client is not None:
                    self.client.close()
            raise
        return self

    def __exit__(self, *exc: object) -> None:
        try:
            if exc[0] is not None:
                self._stop_workers()
            if not self._failed:
                self.student.release_ipc_buffer()
        finally:
            if self.client is not None:
                self.client.close()

    def _stop_workers(self) -> None:
        # Actor termination prevents timed-out calls from reusing the buffers.
        import ray

        if self._stopped:
            return
        self._failed = True
        workers = self.teacher.worker_group.workers + self.student.worker_group.workers
        for worker in workers:
            ray.kill(worker, no_restart=True)
        # These queued calls must fail with ActorDiedError before any row is
        # cleared. A stop timeout aborts cleanup rather than racing a live PUT.
        for worker in workers:
            try:
                ray.get(
                    worker.get_free_memory_bytes.remote(), timeout=self.config.timeout_s
                )
            except ray.exceptions.ActorDiedError:
                pass
        self._stopped = True

    @contextmanager
    def step(self) -> Iterator[None]:
        """Retain a unique payload until training/evaluation has completed."""
        if self.client is None:
            raise RuntimeError("xToken TQ transport has not been initialized")
        sample_id = f"{self.partition_id}/{uuid.uuid4().hex}"
        self._steps.append(sample_id)
        try:
            yield
        except BaseException as step_error:
            try:
                self._stop_workers()
                self.client.clear_samples([sample_id], self.partition_id)
            except Exception as cleanup_error:
                raise BaseExceptionGroup(
                    "xToken TQ step and cleanup failed", [step_error, cleanup_error]
                ) from None
            raise
        else:
            self.client.clear_samples([sample_id], self.partition_id)
            remaining = self.client.list_sample_ids(self.partition_id)
            if sample_id in remaining:
                raise RuntimeError("xToken TQ cleanup left the step payload in storage")
            print(
                f"XTOKEN_TQ_CLEARED sample_id={sample_id} remaining_rows={len(remaining)}",
                flush=True,
            )
        finally:
            self._steps.pop()

    def transfer(self, data: BatchedDataDict[Any]) -> list[dict[str, Any]]:
        """Publish remotely, then expose student-local descriptors to the loss."""
        if not self._steps:
            raise RuntimeError("xToken TQ transfer requires an active step scope")
        reference = self.teacher.get_full_logits_tq(
            data,
            partition_id=self.partition_id,
            sample_id=self._steps[-1],
            max_payload_bytes=self.config.max_payload_bytes,
            timeout_s=self.config.timeout_s,
        )
        if (
            reference.partition_id != self.partition_id
            or reference.sample_id != self._steps[-1]
        ):
            raise ValueError("xToken TQ producer returned a stale or foreign key")
        receipt = self.student.materialize_full_logits_tq(
            reference,
            max_payload_bytes=self.config.max_payload_bytes,
            timeout_s=self.config.timeout_s,
        )
        if receipt.consumer_node_id == reference.producer_node_id:
            raise ValueError("xToken TQ did not cross a node boundary")
        if receipt.nbytes != reference.nbytes:
            raise ValueError("xToken TQ receipt byte count differs from the producer")
        self.metrics = {
            "put_payload_bytes": float(reference.nbytes),
            "get_payload_bytes": float(receipt.nbytes),
            "put_seconds": reference.put_seconds,
            "get_seconds": receipt.get_seconds,
            "student_buffer_bytes": float(receipt.buffer_bytes),
        }
        print(
            f"XTOKEN_TQ_TRANSFER producer={reference.producer_node_id} "
            f"consumer={receipt.consumer_node_id} shape={reference.shape} dtype=float32 "
            f"put_payload_bytes={reference.nbytes} get_payload_bytes={receipt.nbytes} "
            f"put_seconds={reference.put_seconds:.6f} get_seconds={receipt.get_seconds:.6f} "
            f"student_buffer_bytes={receipt.buffer_bytes}",
            flush=True,
        )
        return receipt.handles
