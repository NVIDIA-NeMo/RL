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

"""Actor-local coordination for NeMo Gym partial-rollout checkpoints."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import re
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import aiohttp

CHECKPOINT_CONTROL_TOKEN_ENV = "NEMO_GYM_CHECKPOINT_CONTROL_TOKEN"
GYM_CHECKPOINT_MANIFEST = "nemo_gym_checkpoint_manifest.json"
GYM_CHECKPOINT_SCHEMA_VERSION = 1
CHECKPOINT_OPERATION_TIMEOUT_S = 120.0
MAX_CHECKPOINT_RESPONSE_BYTES = 1024 * 1024
MAX_AGENT_COMPLETION_RECEIPTS = 4096
_CHECKPOINT_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_ROLLOUT_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_RESULT_DIGEST_PATTERN = re.compile(r"^[a-f0-9]{64}$")
_COMPONENT_KEYS = (
    "responses_api_models",
    "responses_api_agents",
    "resources_servers",
)


class CheckpointParticipantKind(str, Enum):
    """Checkpoint role of one Gym server instance."""

    POLICY_MODEL = "policy_model"
    AGENT = "agent"
    RESOURCES = "resources"


class ActorCheckpointPhase(str, Enum):
    """Actor-local checkpoint transaction phase."""

    IDLE = "idle"
    PREPARING = "preparing"
    PREPARED = "prepared"
    COMMITTING = "committing"
    COMMITTED_PAUSED = "committed_paused"
    RESTORING = "restoring"
    RESTORED_PAUSED = "restored_paused"
    RESUMING = "resuming"
    ABORTING = "aborting"


class LiveExecutionState(str, Enum):
    """Publication state of a rollout invocation owned by this actor."""

    RUNNING = "running"
    TERMINAL = "terminal"


@dataclass(frozen=True, order=True)
class ExecutionIdentity:
    """Stable identity supplied by the rollout controller."""

    rollout_id: str
    attempt_index: int

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> ExecutionIdentity:
        """Validate the identity already assigned to a Gym row."""
        rollout_id = row.get("_ng_rollout_id")
        attempt_index = row.get("_ng_attempt_index")
        if (
            not isinstance(rollout_id, str)
            or _ROLLOUT_ID_PATTERN.fullmatch(rollout_id) is None
        ):
            raise ValueError(
                "every NeMo Gym rollout row must carry a non-empty stable "
                "_ng_rollout_id containing only letters, digits, dots, dashes, or underscores"
            )
        if (
            not isinstance(attempt_index, int)
            or isinstance(attempt_index, bool)
            or attempt_index < 0
        ):
            raise ValueError(
                "every NeMo Gym rollout row must carry a nonnegative integer _ng_attempt_index"
            )
        return cls(rollout_id=rollout_id, attempt_index=attempt_index)


@dataclass(frozen=True)
class AgentCompletionReceipt:
    """Exact server-issued proof for one completed agent result."""

    rollout_id: str
    attempt_index: int
    execution_generation: int
    result_id: str
    result_digest: str
    _result_id_key: str = field(repr=False, compare=False)

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> AgentCompletionReceipt:
        """Validate a completion receipt without deriving any of its fields."""
        result_id_keys = [key for key in ("result_id", "result_identity") if key in raw]
        if len(result_id_keys) != 1:
            raise ValueError(
                "Gym agent completion receipt must contain exactly one result ID field"
            )
        result_id_key = result_id_keys[0]
        expected_keys = {
            "rollout_id",
            "attempt_index",
            "execution_generation",
            result_id_key,
            "result_digest",
        }
        if set(raw) != expected_keys:
            raise ValueError(
                "Gym agent completion receipt contains missing or unexpected fields"
            )
        identity = ExecutionIdentity.from_row(
            {
                "_ng_rollout_id": raw.get("rollout_id"),
                "_ng_attempt_index": raw.get("attempt_index"),
            }
        )
        execution_generation = raw.get("execution_generation")
        if (
            not isinstance(execution_generation, int)
            or isinstance(execution_generation, bool)
            or execution_generation < 1
        ):
            raise ValueError(
                "Gym agent completion receipt execution_generation must be a positive integer"
            )
        result_id = raw.get(result_id_key)
        if (
            not isinstance(result_id, str)
            or not result_id
            or len(result_id.encode("utf-8")) > 512
        ):
            raise ValueError(
                "Gym agent completion receipt result ID must contain 1 to 512 UTF-8 bytes"
            )
        result_digest = raw.get("result_digest")
        if (
            not isinstance(result_digest, str)
            or _RESULT_DIGEST_PATTERN.fullmatch(result_digest) is None
        ):
            raise ValueError(
                "Gym agent completion receipt result_digest must be a lowercase SHA-256 digest"
            )
        return cls(
            rollout_id=identity.rollout_id,
            attempt_index=identity.attempt_index,
            execution_generation=execution_generation,
            result_id=result_id,
            result_digest=result_digest,
            _result_id_key=result_id_key,
        )

    @property
    def identity(self) -> ExecutionIdentity:
        """Return the logical rollout attempt named by this receipt."""
        return ExecutionIdentity(self.rollout_id, self.attempt_index)

    def to_request(self) -> dict[str, Any]:
        """Return the exact wire schema issued by the Gym server."""
        return {
            "rollout_id": self.rollout_id,
            "attempt_index": self.attempt_index,
            "execution_generation": self.execution_generation,
            self._result_id_key: self.result_id,
            "result_digest": self.result_digest,
        }


@dataclass
class LiveExecution:
    """One actor-local rollout invocation."""

    identity: ExecutionIdentity
    state: LiveExecutionState = LiveExecutionState.RUNNING

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe execution record."""
        return {
            "rollout_id": self.identity.rollout_id,
            "attempt_index": self.identity.attempt_index,
            "state": self.state.value,
        }


class LiveExecutionRegistry:
    """Track live actor membership and fence new dispatch during checkpoints."""

    def __init__(self) -> None:
        self._live: dict[ExecutionIdentity, LiveExecution] = {}
        self._frozen_checkpoint_id: str | None = None
        self._frozen_membership: tuple[LiveExecution, ...] = ()

    def register(self, row: Mapping[str, Any]) -> ExecutionIdentity:
        """Register one pre-assigned rollout attempt without changing its identity."""
        if self._frozen_checkpoint_id is not None:
            raise RuntimeError(
                f"NeMo Gym dispatch is frozen for checkpoint {self._frozen_checkpoint_id!r}"
            )
        identity = ExecutionIdentity.from_row(row)
        if identity in self._live:
            raise ValueError(
                f"rollout {identity.rollout_id!r} attempt {identity.attempt_index} is already live"
            )
        self._live[identity] = LiveExecution(identity)
        return identity

    def mark_terminal(self, identity: ExecutionIdentity) -> None:
        """Mark a result terminal until the consumer acknowledges it."""
        execution = self._require_live(identity)
        execution.state = LiveExecutionState.TERMINAL

    def release(self, identity: ExecutionIdentity) -> None:
        """Release one result after it crosses the actor boundary."""
        self._live.pop(identity, None)

    def freeze(self, checkpoint_id: str) -> tuple[LiveExecution, ...]:
        """Close dispatch and snapshot actor-local membership."""
        _validate_checkpoint_id(checkpoint_id)
        if self._frozen_checkpoint_id is not None:
            if self._frozen_checkpoint_id != checkpoint_id:
                raise RuntimeError(
                    f"checkpoint {self._frozen_checkpoint_id!r} already owns the dispatch fence"
                )
            return self._frozen_membership
        self._frozen_checkpoint_id = checkpoint_id
        self._frozen_membership = tuple(
            LiveExecution(execution.identity, execution.state)
            for execution in sorted(self._live.values(), key=lambda item: item.identity)
        )
        return self._frozen_membership

    def restore(
        self,
        checkpoint_id: str,
        membership: Sequence[Mapping[str, Any]],
    ) -> None:
        """Restore a dispatch fence and its source-cut membership."""
        _validate_checkpoint_id(checkpoint_id)
        if self._live:
            raise RuntimeError(
                "cannot restore NeMo Gym dispatch while rollout calls are live"
            )
        if self._frozen_checkpoint_id not in (None, checkpoint_id):
            raise RuntimeError(
                f"checkpoint {self._frozen_checkpoint_id!r} already owns the dispatch fence"
            )
        restored: list[LiveExecution] = []
        for raw in membership:
            identity = ExecutionIdentity.from_row(
                {
                    "_ng_rollout_id": raw.get("rollout_id"),
                    "_ng_attempt_index": raw.get("attempt_index"),
                }
            )
            raw_state = raw.get("state")
            if raw_state == LiveExecutionState.RUNNING.value:
                state = LiveExecutionState.RUNNING
            elif raw_state == LiveExecutionState.TERMINAL.value:
                state = LiveExecutionState.TERMINAL
            else:
                raise ValueError(
                    f"invalid live execution state for {identity}: {raw_state!r}"
                )
            restored.append(LiveExecution(identity, state))
        self._frozen_checkpoint_id = checkpoint_id
        self._frozen_membership = tuple(
            sorted(restored, key=lambda item: item.identity)
        )

    def unfreeze(self, checkpoint_id: str) -> None:
        """Reopen dispatch for the active transaction."""
        if self._frozen_checkpoint_id != checkpoint_id:
            raise RuntimeError(
                f"checkpoint {checkpoint_id!r} does not own the dispatch fence "
                f"(owner={self._frozen_checkpoint_id!r})"
            )
        self._frozen_checkpoint_id = None
        self._frozen_membership = ()

    def status(self) -> dict[str, Any]:
        """Return bounded actor-local checkpoint diagnostics."""
        running = sum(
            execution.state == LiveExecutionState.RUNNING
            for execution in self._live.values()
        )
        terminal = len(self._live) - running
        return {
            "frozen_checkpoint_id": self._frozen_checkpoint_id,
            "live": len(self._live),
            "running": running,
            "terminal_unreleased": terminal,
            "frozen_membership": len(self._frozen_membership),
        }

    def frozen_membership(self) -> list[dict[str, Any]]:
        """Return the local source cut for the aggregate manifest."""
        return [execution.to_dict() for execution in self._frozen_membership]

    def _require_live(self, identity: ExecutionIdentity) -> LiveExecution:
        execution = self._live.get(identity)
        if execution is None:
            raise KeyError(
                f"rollout {identity.rollout_id!r} attempt {identity.attempt_index} is not live"
            )
        return execution


@dataclass(frozen=True)
class CheckpointCapabilities:
    """Validated capability declaration from one Gym server."""

    component: str
    name: str
    schema_version: int
    admission_states: tuple[str, ...]
    checkpoint_mode: str
    concurrency_contract: str
    multi_process_mode: str
    num_workers: int
    instance_role: str | None

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> CheckpointCapabilities:
        """Parse the fields required by actor-side orchestration."""
        multi_process = raw.get("multi_process")
        if not isinstance(multi_process, Mapping):
            raise ValueError("Gym capabilities are missing multi_process")
        admission_states = raw.get("admission_states")
        if not isinstance(admission_states, list) or not all(
            isinstance(state, str) for state in admission_states
        ):
            raise ValueError(
                "Gym capabilities admission_states must be a list of strings"
            )
        component = _required_capability_string(raw, "component")
        name = _required_capability_string(raw, "name")
        checkpoint_mode = _required_capability_string(raw, "checkpoint_mode")
        concurrency_contract = _required_capability_string(raw, "concurrency_contract")
        multi_process_mode = _required_capability_string(multi_process, "mode")
        schema_version = raw.get("schema_version")
        num_workers = multi_process.get("num_workers")
        if not isinstance(schema_version, int) or isinstance(schema_version, bool):
            raise ValueError("Gym capabilities schema_version must be an integer")
        if (
            not isinstance(num_workers, int)
            or isinstance(num_workers, bool)
            or num_workers < 1
        ):
            raise ValueError("Gym capabilities num_workers must be a positive integer")
        instance_role = raw.get("instance_role")
        if instance_role is not None and not isinstance(instance_role, str):
            raise ValueError("Gym capabilities instance_role must be a string or null")
        return cls(
            component=component,
            name=name,
            schema_version=schema_version,
            admission_states=tuple(admission_states),
            checkpoint_mode=checkpoint_mode,
            concurrency_contract=concurrency_contract,
            multi_process_mode=multi_process_mode,
            num_workers=num_workers,
            instance_role=instance_role,
        )


@dataclass(frozen=True)
class CheckpointParticipant:
    """One checkpoint-capable Gym server and its direct control endpoint."""

    participant_id: str
    name: str
    base_url: str
    kind: CheckpointParticipantKind
    capabilities: CheckpointCapabilities

    def to_dict(self) -> dict[str, Any]:
        """Return stable participant metadata for the aggregate manifest."""
        return {
            "participant_id": self.participant_id,
            "name": self.name,
            "base_url": self.base_url,
            "kind": self.kind.value,
            "capabilities": asdict(self.capabilities),
        }


class CheckpointParticipantError(RuntimeError):
    """A picklable participant failure retaining Gym's structured error fields."""

    def __init__(
        self,
        participant: str,
        endpoint: str,
        detail: str,
        *,
        status: int | None = None,
        gym_error_code: str | None = None,
    ) -> None:
        self.participant = participant
        self.endpoint = endpoint
        self.status = status
        self.gym_error_code = gym_error_code
        self.detail = detail
        fields = [f"participant={participant}", f"endpoint={endpoint}"]
        if status is not None:
            fields.append(f"status={status}")
        if gym_error_code is not None:
            fields.append(f"gym_error_code={gym_error_code}")
        super().__init__(f"Gym checkpoint call failed ({', '.join(fields)}): {detail}")

    def __reduce__(self) -> tuple[Any, tuple[Any, ...]]:
        return (
            _restore_participant_error,
            (
                self.participant,
                self.endpoint,
                self.detail,
                self.status,
                self.gym_error_code,
            ),
        )


def _restore_participant_error(
    participant: str,
    endpoint: str,
    detail: str,
    status: int | None,
    gym_error_code: str | None,
) -> CheckpointParticipantError:
    return CheckpointParticipantError(
        participant,
        endpoint,
        detail,
        status=status,
        gym_error_code=gym_error_code,
    )


class ActorCheckpointError(RuntimeError):
    """Actor transaction failure with phase and completed participants."""

    def __init__(
        self,
        checkpoint_id: str,
        phase: ActorCheckpointPhase,
        completed_participants: Sequence[str],
        cause: BaseException,
    ) -> None:
        self.checkpoint_id = checkpoint_id
        self.phase = phase
        self.completed_participants = tuple(completed_participants)
        self.cause_detail = str(cause)
        self.participant: str | None
        self.endpoint: str | None
        self.status: int | None
        self.gym_error_code: str | None
        if isinstance(cause, CheckpointParticipantError):
            self.participant = cause.participant
            self.endpoint = cause.endpoint
            self.status = cause.status
            self.gym_error_code = cause.gym_error_code
        else:
            self.participant = None
            self.endpoint = None
            self.status = None
            self.gym_error_code = None
        super().__init__(
            f"NeMo Gym checkpoint {checkpoint_id!r} failed in phase {phase.value!r} "
            f"after {list(completed_participants)!r}: {cause}"
        )

    def __reduce__(self) -> tuple[Any, tuple[Any, ...]]:
        return (
            _restore_actor_error,
            (
                self.checkpoint_id,
                self.phase,
                self.completed_participants,
                self.cause_detail,
                self.participant,
                self.endpoint,
                self.status,
                self.gym_error_code,
            ),
        )


def _restore_actor_error(
    checkpoint_id: str,
    phase: ActorCheckpointPhase,
    completed_participants: Sequence[str],
    cause_detail: str,
    participant: str | None,
    endpoint: str | None,
    status: int | None,
    gym_error_code: str | None,
) -> ActorCheckpointError:
    cause: BaseException
    if participant is not None and endpoint is not None:
        cause = CheckpointParticipantError(
            participant,
            endpoint,
            cause_detail,
            status=status,
            gym_error_code=gym_error_code,
        )
    else:
        cause = RuntimeError(cause_detail)
    return ActorCheckpointError(
        checkpoint_id,
        phase,
        completed_participants,
        cause,
    )


@dataclass(frozen=True)
class ActorCheckpointResult:
    """Opaque result returned through the Ray actor boundary."""

    checkpoint_id: str
    phase: ActorCheckpointPhase
    participant_results: Mapping[str, Mapping[str, Any]]
    manifest_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe and Ray-friendly result."""
        return {
            "checkpoint_id": self.checkpoint_id,
            "phase": self.phase.value,
            "participant_results": {
                participant: dict(result)
                for participant, result in self.participant_results.items()
            },
            "manifest_path": self.manifest_path,
        }


@dataclass
class ActorCheckpointTransaction:
    """Mutable actor-local state for one fenced checkpoint."""

    checkpoint_id: str
    phase: ActorCheckpointPhase
    deadline_ts: float
    participant_results: dict[str, dict[str, Any]] = field(default_factory=dict)
    prepared_participants: set[str] = field(default_factory=set)
    operation_results: dict[str, dict[str, Any]] = field(default_factory=dict)
    manifest_path: str | None = None


class CheckpointTransport:
    """Deadline-bounded aiohttp transport for Gym checkpoint control calls."""

    def __init__(self, bearer_token: str) -> None:
        if not bearer_token:
            raise ValueError("Gym checkpoint bearer token must not be empty")
        self._headers = {"Authorization": f"Bearer {bearer_token}"}

    async def request(
        self,
        participant: CheckpointParticipant,
        method: str,
        endpoint: str,
        *,
        deadline_ts: float,
        json_body: Mapping[str, Any] | None = None,
        params: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Issue one direct control request inside the shared absolute deadline."""
        remaining = _remaining(deadline_ts)
        url = f"{participant.base_url.rstrip('/')}{endpoint}"
        timeout = aiohttp.ClientTimeout(total=remaining)
        try:
            async with aiohttp.ClientSession(
                timeout=timeout, headers=self._headers
            ) as session:
                async with session.request(
                    method,
                    url,
                    json=dict(json_body) if json_body is not None else None,
                    params=dict(params) if params is not None else None,
                ) as response:
                    body = await _read_bounded_response(
                        participant,
                        endpoint,
                        response,
                    )
                    parsed = _decode_response(body)
                    if response.status < 200 or response.status >= 300:
                        error_code, detail = _gym_error(parsed, body)
                        raise CheckpointParticipantError(
                            participant.participant_id,
                            endpoint,
                            detail,
                            status=response.status,
                            gym_error_code=error_code,
                        )
        except CheckpointParticipantError:
            raise
        except (asyncio.TimeoutError, TimeoutError) as error:
            raise CheckpointParticipantError(
                participant.participant_id,
                endpoint,
                "absolute checkpoint deadline expired",
                gym_error_code="deadline_exceeded",
            ) from error
        except aiohttp.ClientError as error:
            raise CheckpointParticipantError(
                participant.participant_id,
                endpoint,
                str(error),
                gym_error_code="transport_error",
            ) from error
        if not isinstance(parsed, dict):
            raise CheckpointParticipantError(
                participant.participant_id,
                endpoint,
                "Gym checkpoint response must be a JSON object",
                status=response.status,
                gym_error_code="invalid_response",
            )
        return parsed


class NemoGymCheckpointCoordinator:
    """Coordinate one NemoGym actor's checkpoint transaction."""

    def __init__(self, bearer_token: str) -> None:
        self.registry = LiveExecutionRegistry()
        self._transport = CheckpointTransport(bearer_token)
        self._participants: list[CheckpointParticipant] = []
        self._transaction: ActorCheckpointTransaction | None = None
        self._retired_checkpoint_id: str | None = None
        self._retired_results: dict[str, dict[str, Any]] = {}
        self._lock = asyncio.Lock()

    async def discover(
        self,
        global_config: Mapping[str, Any],
        *,
        deadline_ts: float,
    ) -> list[CheckpointParticipant]:
        """Discover and validate the servers started by RunHelper."""
        self._participants = await discover_checkpoint_participants(
            global_config,
            self._transport,
            deadline_ts=deadline_ts,
        )
        return list(self._participants)

    async def prepare(self, checkpoint_id: str, deadline_ts: float) -> dict[str, Any]:
        """Freeze dispatch and prepare agents, policy admission, then resources."""
        _validate_checkpoint_id(checkpoint_id)
        _remaining(deadline_ts)
        async with self._lock:
            if self._retired_checkpoint_id == checkpoint_id:
                raise RuntimeError(f"checkpoint {checkpoint_id!r} is stale")
            transaction = self._transaction
            if transaction is not None:
                if transaction.checkpoint_id != checkpoint_id:
                    raise RuntimeError(
                        f"checkpoint {transaction.checkpoint_id!r} is already active "
                        f"in phase {transaction.phase.value!r}"
                    )
                recorded = transaction.operation_results.get("prepare")
                if recorded is not None:
                    return recorded
                if transaction.phase != ActorCheckpointPhase.PREPARING:
                    raise RuntimeError(
                        f"prepare is not valid in actor phase {transaction.phase.value!r}"
                    )
            else:
                transaction = ActorCheckpointTransaction(
                    checkpoint_id=checkpoint_id,
                    phase=ActorCheckpointPhase.PREPARING,
                    deadline_ts=deadline_ts,
                )
                self._transaction = transaction
                self.registry.freeze(checkpoint_id)

            try:
                if self.registry.status()["terminal_unreleased"]:
                    raise RuntimeError(
                        "cannot prepare a NeMo Gym checkpoint while streamed agent "
                        "results remain unacknowledged"
                    )
                agents = self._of_kind(CheckpointParticipantKind.AGENT)
                policy_models = self._of_kind(CheckpointParticipantKind.POLICY_MODEL)
                resources = self._of_kind(CheckpointParticipantKind.RESOURCES)
                await self._run_participants(
                    transaction,
                    "prepare",
                    agents,
                    deadline_ts=deadline_ts,
                )
                await self._run_participants(
                    transaction,
                    "prepare",
                    policy_models,
                    deadline_ts=deadline_ts,
                )
                for participant in policy_models:
                    await self._wait_for_policy_pause(
                        transaction,
                        participant,
                        deadline_ts=deadline_ts,
                    )
                await self._run_participants(
                    transaction,
                    "prepare",
                    resources,
                    deadline_ts=deadline_ts,
                )
            except (CheckpointParticipantError, RuntimeError, ValueError) as error:
                raise self._actor_error(transaction, error) from None

            transaction.phase = ActorCheckpointPhase.PREPARED
            result = self._result(transaction)
            transaction.operation_results["prepare"] = result
            return result

    async def commit(
        self, checkpoint_id: str, checkpoint_dir: str | Path
    ) -> dict[str, Any]:
        """Commit all participants and publish the aggregate manifest last."""
        async with self._lock:
            transaction = self._require_transaction(
                checkpoint_id,
                {
                    ActorCheckpointPhase.PREPARED,
                    ActorCheckpointPhase.COMMITTING,
                    ActorCheckpointPhase.COMMITTED_PAUSED,
                },
            )
            recorded = transaction.operation_results.get("commit")
            if recorded is not None:
                return recorded
            transaction.phase = ActorCheckpointPhase.COMMITTING
            deadline_ts = time.time() + CHECKPOINT_OPERATION_TIMEOUT_S
            transaction.deadline_ts = deadline_ts
            try:
                await self._run_participants(
                    transaction,
                    "commit",
                    self._participants,
                    deadline_ts=deadline_ts,
                    checkpoint_dir=checkpoint_dir,
                )
                manifest_path = await asyncio.to_thread(
                    write_checkpoint_manifest,
                    Path(checkpoint_dir),
                    checkpoint_id=checkpoint_id,
                    participants=self._participants,
                    participant_results=transaction.participant_results,
                    live_membership=self.registry.frozen_membership(),
                )
            except (
                CheckpointParticipantError,
                OSError,
                RuntimeError,
                ValueError,
            ) as error:
                raise self._actor_error(transaction, error) from None
            transaction.manifest_path = str(manifest_path)
            transaction.phase = ActorCheckpointPhase.COMMITTED_PAUSED
            result = self._result(transaction)
            transaction.operation_results["commit"] = result
            return result

    async def restore(
        self, checkpoint_id: str, checkpoint_dir: str | Path
    ) -> dict[str, Any]:
        """Verify the aggregate manifest and restore every participant paused."""
        _validate_checkpoint_id(checkpoint_id)
        async with self._lock:
            if (
                self._transaction is None
                and self._retired_checkpoint_id == checkpoint_id
            ):
                raise RuntimeError(f"checkpoint {checkpoint_id!r} is stale")
            if self._transaction is not None:
                transaction = self._require_transaction(
                    checkpoint_id,
                    {
                        ActorCheckpointPhase.RESTORING,
                        ActorCheckpointPhase.RESTORED_PAUSED,
                    },
                )
                recorded = transaction.operation_results.get("restore")
                if recorded is not None:
                    return recorded
            else:
                deadline_ts = time.time() + CHECKPOINT_OPERATION_TIMEOUT_S
                transaction = ActorCheckpointTransaction(
                    checkpoint_id=checkpoint_id,
                    phase=ActorCheckpointPhase.RESTORING,
                    deadline_ts=deadline_ts,
                )
                self._transaction = transaction
            transaction.phase = ActorCheckpointPhase.RESTORING
            deadline_ts = time.time() + CHECKPOINT_OPERATION_TIMEOUT_S
            transaction.deadline_ts = deadline_ts
            try:
                manifest = await asyncio.to_thread(
                    load_checkpoint_manifest,
                    Path(checkpoint_dir),
                )
                self._validate_manifest_participants(manifest)
                self.registry.restore(
                    checkpoint_id, manifest.get("live_membership", [])
                )
                await self._run_participants(
                    transaction,
                    "restore",
                    self._participants,
                    deadline_ts=deadline_ts,
                    checkpoint_dir=checkpoint_dir,
                )
            except (
                CheckpointParticipantError,
                OSError,
                RuntimeError,
                ValueError,
            ) as error:
                raise self._actor_error(transaction, error) from None
            transaction.manifest_path = str(
                Path(checkpoint_dir) / GYM_CHECKPOINT_MANIFEST
            )
            transaction.phase = ActorCheckpointPhase.RESTORED_PAUSED
            result = self._result(transaction)
            transaction.operation_results["restore"] = result
            return result

    async def resume(self, checkpoint_id: str) -> dict[str, Any]:
        """Resume model/resources, then agents, then actor dispatch."""
        async with self._lock:
            retired = self._retired_result(checkpoint_id, "resume")
            if retired is not None:
                return retired
            transaction = self._require_transaction(
                checkpoint_id,
                {
                    ActorCheckpointPhase.COMMITTED_PAUSED,
                    ActorCheckpointPhase.RESTORED_PAUSED,
                    ActorCheckpointPhase.RESUMING,
                },
            )
            transaction.phase = ActorCheckpointPhase.RESUMING
            deadline_ts = time.time() + CHECKPOINT_OPERATION_TIMEOUT_S
            transaction.deadline_ts = deadline_ts
            dependencies, agents = participant_groups(self._participants)
            try:
                await self._run_participants(
                    transaction,
                    "resume",
                    dependencies,
                    deadline_ts=deadline_ts,
                )
                await self._run_participants(
                    transaction,
                    "resume",
                    agents,
                    deadline_ts=deadline_ts,
                )
                self.registry.unfreeze(checkpoint_id)
            except (CheckpointParticipantError, RuntimeError, ValueError) as error:
                raise self._actor_error(transaction, error) from None
            result = ActorCheckpointResult(
                checkpoint_id=checkpoint_id,
                phase=ActorCheckpointPhase.IDLE,
                participant_results=transaction.participant_results,
                manifest_path=transaction.manifest_path,
            ).to_dict()
            self._retire_transaction(checkpoint_id, "resume", result)
            self._transaction = None
            return result

    async def abort(self, checkpoint_id: str) -> dict[str, Any]:
        """Resume participants reached by a partial transaction and reopen dispatch."""
        async with self._lock:
            retired = self._retired_result(checkpoint_id, "abort")
            if retired is not None:
                return retired
            transaction = self._require_transaction(
                checkpoint_id,
                {
                    ActorCheckpointPhase.PREPARING,
                    ActorCheckpointPhase.PREPARED,
                    ActorCheckpointPhase.COMMITTING,
                    ActorCheckpointPhase.COMMITTED_PAUSED,
                    ActorCheckpointPhase.RESTORING,
                    ActorCheckpointPhase.RESTORED_PAUSED,
                    ActorCheckpointPhase.RESUMING,
                    ActorCheckpointPhase.ABORTING,
                },
            )
            transaction.phase = ActorCheckpointPhase.ABORTING
            deadline_ts = time.time() + CHECKPOINT_OPERATION_TIMEOUT_S
            transaction.deadline_ts = deadline_ts
            prepared = [
                participant
                for participant in self._participants
                if participant.participant_id in transaction.prepared_participants
            ]
            dependencies, agents = participant_groups(prepared)
            try:
                await self._run_participants(
                    transaction,
                    "abort",
                    dependencies,
                    deadline_ts=deadline_ts,
                    route_operation="resume",
                )
                await self._run_participants(
                    transaction,
                    "abort",
                    agents,
                    deadline_ts=deadline_ts,
                    route_operation="resume",
                )
                self.registry.unfreeze(checkpoint_id)
            except (CheckpointParticipantError, RuntimeError, ValueError) as error:
                raise self._actor_error(transaction, error) from None
            result = ActorCheckpointResult(
                checkpoint_id=checkpoint_id,
                phase=ActorCheckpointPhase.IDLE,
                participant_results=transaction.participant_results,
                manifest_path=transaction.manifest_path,
            ).to_dict()
            self._retire_transaction(checkpoint_id, "abort", result)
            self._transaction = None
            return result

    async def release(
        self,
        identity: ExecutionIdentity,
        *,
        agent_name: str,
    ) -> None:
        """Acknowledge an exact delivered-result receipt, then release membership."""
        async with self._lock:
            matching_agents = [
                participant
                for participant in self._of_kind(CheckpointParticipantKind.AGENT)
                if agent_name in {participant.participant_id, participant.name}
            ]
            if len(matching_agents) > 1:
                raise RuntimeError(
                    f"multiple Gym checkpoint agents match resolved agent {agent_name!r}"
                )
            if matching_agents:
                deadline_ts = time.time() + CHECKPOINT_OPERATION_TIMEOUT_S
                checkpoint_id = (
                    self._transaction.checkpoint_id
                    if self._transaction is not None
                    else "actor-delivery"
                )
                participant = matching_agents[0]
                status = await self._transport.request(
                    participant,
                    "GET",
                    "/ng-control/v1/agent-checkpoint/status",
                    deadline_ts=deadline_ts,
                    params={"checkpoint_id": checkpoint_id},
                )
                receipt = _completion_receipt_for(
                    participant,
                    status,
                    identity,
                )
                acknowledgment = await self._transport.request(
                    participant,
                    "POST",
                    "/ng-control/v1/agent-checkpoint/acknowledge",
                    deadline_ts=deadline_ts,
                    json_body=receipt.to_request(),
                )
                acknowledged = acknowledgment.get("acknowledged")
                idempotent = acknowledgment.get("idempotent")
                if (
                    not isinstance(acknowledged, bool)
                    or not isinstance(idempotent, bool)
                    or not (acknowledged or idempotent)
                ):
                    raise CheckpointParticipantError(
                        participant.participant_id,
                        "/ng-control/v1/agent-checkpoint/acknowledge",
                        f"invalid Gym completion acknowledgment: {acknowledgment!r}",
                        gym_error_code="invalid_response",
                    )
            self.registry.release(identity)

    def status(self, checkpoint_id: str) -> dict[str, Any]:
        """Return actor-local transaction and bounded registry status."""
        transaction = self._transaction
        if transaction is None:
            return {
                "checkpoint_id": checkpoint_id,
                "phase": ActorCheckpointPhase.IDLE.value,
                "participants": [
                    participant.to_dict() for participant in self._participants
                ],
                "registry": self.registry.status(),
                "participant_results": {},
                "manifest_path": None,
            }
        if transaction.checkpoint_id != checkpoint_id:
            raise RuntimeError(
                f"checkpoint {transaction.checkpoint_id!r} is active, not {checkpoint_id!r}"
            )
        return {
            "checkpoint_id": checkpoint_id,
            "phase": transaction.phase.value,
            "participants": [
                participant.to_dict() for participant in self._participants
            ],
            "registry": self.registry.status(),
            "participant_results": dict(transaction.participant_results),
            "manifest_path": transaction.manifest_path,
        }

    async def _run_participants(
        self,
        transaction: ActorCheckpointTransaction,
        operation: str,
        participants: Sequence[CheckpointParticipant],
        *,
        deadline_ts: float,
        checkpoint_dir: str | Path | None = None,
        route_operation: str | None = None,
    ) -> None:
        route_operation = route_operation or operation
        for participant in participants:
            result_key = f"{operation}:{participant.participant_id}"
            if result_key in transaction.participant_results:
                continue
            if operation in {"prepare", "restore"}:
                # Once a request is sent, a lost response cannot prove whether
                # the server paused. Abort must conservatively try to resume it.
                transaction.prepared_participants.add(participant.participant_id)
            method, endpoint = participant_endpoint(participant, route_operation)
            body: dict[str, Any] = {
                "checkpoint_id": transaction.checkpoint_id,
                "deadline_ts": deadline_ts,
            }
            if checkpoint_dir is not None:
                body["checkpoint_dir"] = str(checkpoint_dir)
            try:
                result = await self._transport.request(
                    participant,
                    method,
                    endpoint,
                    deadline_ts=deadline_ts,
                    json_body=body,
                )
            except CheckpointParticipantError as error:
                if operation != "abort" or error.gym_error_code != "invalid_phase":
                    raise
                result = {
                    "state": "already_accepting",
                    "gym_error_code": error.gym_error_code,
                }
            transaction.participant_results[result_key] = result

    async def _wait_for_policy_pause(
        self,
        transaction: ActorCheckpointTransaction,
        participant: CheckpointParticipant,
        *,
        deadline_ts: float,
    ) -> None:
        result_key = f"prepare-status:{participant.participant_id}"
        if result_key in transaction.participant_results:
            return
        method, endpoint = participant_endpoint(participant, "status")
        remaining = _remaining(deadline_ts)
        result = await self._transport.request(
            participant,
            method,
            endpoint,
            deadline_ts=deadline_ts,
            params={
                "checkpoint_id": transaction.checkpoint_id,
                "deadline_ts": deadline_ts,
                "wait_state": "paused",
                "timeout_s": remaining,
            },
        )
        if result.get("state") != "paused":
            raise CheckpointParticipantError(
                participant.participant_id,
                endpoint,
                f"policy model did not drain before the deadline: {result!r}",
                gym_error_code="prepare_incomplete",
            )
        transaction.participant_results[result_key] = result
        transaction.prepared_participants.add(participant.participant_id)

    def _of_kind(self, kind: CheckpointParticipantKind) -> list[CheckpointParticipant]:
        return [
            participant
            for participant in self._participants
            if participant.kind == kind
        ]

    def _require_transaction(
        self,
        checkpoint_id: str,
        allowed_phases: set[ActorCheckpointPhase],
    ) -> ActorCheckpointTransaction:
        _validate_checkpoint_id(checkpoint_id)
        transaction = self._transaction
        if transaction is None:
            raise RuntimeError("no NeMo Gym checkpoint transaction is active")
        if transaction.checkpoint_id != checkpoint_id:
            raise RuntimeError(
                f"checkpoint {transaction.checkpoint_id!r} is already active "
                f"in phase {transaction.phase.value!r}"
            )
        if transaction.phase not in allowed_phases:
            raise RuntimeError(
                f"checkpoint operation is not valid in actor phase {transaction.phase.value!r}"
            )
        return transaction

    def _validate_manifest_participants(self, manifest: Mapping[str, Any]) -> None:
        raw_participants = manifest.get("participants")
        if not isinstance(raw_participants, list):
            raise ValueError("NeMo Gym checkpoint manifest participants must be a list")
        source: set[tuple[str, str, str]] = set()
        for entry in raw_participants:
            if not isinstance(entry, Mapping):
                raise ValueError("NeMo Gym checkpoint participant must be an object")
            participant_id = entry.get("participant_id")
            name = entry.get("name")
            kind = entry.get("kind")
            if not all(
                isinstance(value, str) for value in (participant_id, name, kind)
            ):
                raise ValueError(
                    "NeMo Gym checkpoint participant identity fields must be strings"
                )
            assert isinstance(participant_id, str)
            assert isinstance(name, str)
            assert isinstance(kind, str)
            source.add((participant_id, name, kind))
        current = {
            (participant.participant_id, participant.name, participant.kind.value)
            for participant in self._participants
        }
        if source != current:
            raise ValueError(
                f"NeMo Gym checkpoint participants do not match the running Gym fleet: "
                f"checkpoint={sorted(source)!r}, current={sorted(current)!r}"
            )

    def _retired_result(
        self,
        checkpoint_id: str,
        operation: str,
    ) -> dict[str, Any] | None:
        if self._retired_checkpoint_id != checkpoint_id:
            return None
        return self._retired_results.get(operation)

    def _retire_transaction(
        self,
        checkpoint_id: str,
        operation: str,
        result: dict[str, Any],
    ) -> None:
        self._retired_checkpoint_id = checkpoint_id
        self._retired_results = {operation: result}

    @staticmethod
    def _actor_error(
        transaction: ActorCheckpointTransaction,
        cause: BaseException,
    ) -> ActorCheckpointError:
        return ActorCheckpointError(
            transaction.checkpoint_id,
            transaction.phase,
            sorted(transaction.prepared_participants),
            cause,
        )

    @staticmethod
    def _result(transaction: ActorCheckpointTransaction) -> dict[str, Any]:
        return ActorCheckpointResult(
            checkpoint_id=transaction.checkpoint_id,
            phase=transaction.phase,
            participant_results=transaction.participant_results,
            manifest_path=transaction.manifest_path,
        ).to_dict()


async def discover_checkpoint_participants(
    global_config: Mapping[str, Any],
    transport: CheckpointTransport,
    *,
    deadline_ts: float,
) -> list[CheckpointParticipant]:
    """Discover checkpoint participants from RunHelper's resolved global config."""
    candidates = _server_candidates(global_config)
    declarations = await asyncio.gather(
        *[
            transport.request(
                candidate,
                "GET",
                "/ng-control/v1/capabilities",
                deadline_ts=deadline_ts,
            )
            for candidate in candidates
        ]
    )
    participants: list[CheckpointParticipant] = []
    for candidate, raw in zip(candidates, declarations):
        capabilities = CheckpointCapabilities.from_mapping(raw)
        if (
            capabilities.component != candidate.capabilities.component
            or capabilities.name != candidate.name
        ):
            raise ValueError(
                f"Gym capabilities identity mismatch for {candidate.participant_id!r}: "
                f"configured={candidate.capabilities.component}/{candidate.name}, "
                f"served={capabilities.component}/{capabilities.name}"
            )
        if capabilities.multi_process_mode == "unmanaged":
            raise ValueError(
                f"Gym server {candidate.participant_id!r} uses unmanaged multi-process mode "
                f"with {capabilities.num_workers} workers; checkpointing requires a single worker "
                "or a service-level coordinator"
            )
        kind = _participant_kind(capabilities)
        if kind is None:
            continue
        if capabilities.schema_version != GYM_CHECKPOINT_SCHEMA_VERSION:
            raise ValueError(
                f"Gym server {candidate.participant_id!r} declares unsupported checkpoint "
                f"schema version {capabilities.schema_version}"
            )
        if capabilities.checkpoint_mode != "export_restore":
            continue
        if kind == CheckpointParticipantKind.POLICY_MODEL and not {
            "accepting",
            "draining",
            "paused",
        }.issubset(capabilities.admission_states):
            raise ValueError(
                f"policy model {candidate.participant_id!r} cannot enter all required admission states"
            )
        participants.append(
            CheckpointParticipant(
                participant_id=candidate.participant_id,
                name=candidate.name,
                base_url=candidate.base_url,
                kind=kind,
                capabilities=capabilities,
            )
        )
    if not any(
        participant.kind == CheckpointParticipantKind.POLICY_MODEL
        for participant in participants
    ):
        raise ValueError(
            "NeMo Gym checkpointing requires one checkpoint-capable policy model"
        )
    return sorted(participants, key=lambda participant: participant.participant_id)


def write_checkpoint_manifest(
    checkpoint_dir: Path,
    *,
    checkpoint_id: str,
    participants: Sequence[CheckpointParticipant],
    participant_results: Mapping[str, Mapping[str, Any]],
    live_membership: Sequence[Mapping[str, Any]],
) -> Path:
    """Atomically publish the aggregate manifest after participant commits."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = checkpoint_dir / GYM_CHECKPOINT_MANIFEST
    files = _checkpoint_file_digests(checkpoint_dir)
    integrity = _integrity_digest(files)
    manifest = {
        "schema_version": GYM_CHECKPOINT_SCHEMA_VERSION,
        "checkpoint_id": checkpoint_id,
        "participants": [participant.to_dict() for participant in participants],
        "participant_results": {
            participant: dict(result)
            for participant, result in participant_results.items()
        },
        "live_membership": [dict(record) for record in live_membership],
        "files": files,
        "integrity_sha256": integrity,
    }
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n")
    _fsync_file(temporary)
    os.replace(temporary, manifest_path)
    _fsync_directory(checkpoint_dir)
    return manifest_path


def load_checkpoint_manifest(checkpoint_dir: Path) -> dict[str, Any]:
    """Load and verify a previously published aggregate Gym manifest."""
    checkpoint_dir = Path(checkpoint_dir)
    manifest_path = checkpoint_dir / GYM_CHECKPOINT_MANIFEST
    try:
        raw = json.loads(manifest_path.read_text())
    except FileNotFoundError:
        raise FileNotFoundError(
            f"NeMo Gym checkpoint manifest is missing at {manifest_path}"
        ) from None
    except json.JSONDecodeError as error:
        raise ValueError(
            f"NeMo Gym checkpoint manifest is invalid JSON: {error}"
        ) from error
    if not isinstance(raw, dict):
        raise ValueError("NeMo Gym checkpoint manifest must be a JSON object")
    if raw.get("schema_version") != GYM_CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported NeMo Gym checkpoint schema version: {raw.get('schema_version')!r}"
        )
    source_checkpoint_id = raw.get("checkpoint_id")
    if not isinstance(source_checkpoint_id, str):
        raise ValueError("NeMo Gym checkpoint manifest checkpoint_id must be a string")
    _validate_checkpoint_id(source_checkpoint_id)
    participants = raw.get("participants")
    if not isinstance(participants, list) or not all(
        isinstance(participant, Mapping) for participant in participants
    ):
        raise ValueError(
            "NeMo Gym checkpoint manifest participants must be a list of objects"
        )
    live_membership = raw.get("live_membership")
    if not isinstance(live_membership, list) or not all(
        isinstance(execution, Mapping) for execution in live_membership
    ):
        raise ValueError(
            "NeMo Gym checkpoint manifest live_membership must be a list of objects"
        )
    files = raw.get("files")
    if not isinstance(files, dict) or not all(
        isinstance(path, str) and isinstance(digest, str)
        for path, digest in files.items()
    ):
        raise ValueError("NeMo Gym checkpoint manifest files must map paths to digests")
    if raw.get("integrity_sha256") != _integrity_digest(files):
        raise ValueError(
            "NeMo Gym checkpoint manifest integrity digest does not match its file table"
        )
    for relative_path, expected_digest in files.items():
        relative = Path(relative_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(
                f"NeMo Gym checkpoint file path {relative_path!r} escapes the checkpoint directory"
            )
        path = checkpoint_dir / relative_path
        if path.is_symlink() or not path.is_file() or _digest(path) != expected_digest:
            raise ValueError(
                f"NeMo Gym checkpoint file {relative_path!r} is missing or corrupted"
            )
    return raw


def participant_endpoint(
    participant: CheckpointParticipant, operation: str
) -> tuple[str, str]:
    """Map an actor operation to the current Gym route contract."""
    if participant.kind == CheckpointParticipantKind.POLICY_MODEL:
        endpoints = {
            "prepare": ("POST", "/ng-control/v1/model-admission/pause"),
            "status": ("GET", "/ng-control/v1/model-admission/status"),
            "commit": ("POST", "/ng-control/v1/model-checkpoint/commit"),
            "restore": ("POST", "/ng-control/v1/model-checkpoint/restore"),
            "resume": ("POST", "/ng-control/v1/model-admission/resume"),
        }
    elif participant.kind == CheckpointParticipantKind.AGENT:
        endpoints = {
            operation_name: (
                "GET" if operation_name == "status" else "POST",
                f"/ng-control/v1/agent-checkpoint/{operation_name}",
            )
            for operation_name in ("prepare", "status", "commit", "restore", "resume")
        }
    else:
        endpoints = {
            operation_name: (
                "GET" if operation_name == "status" else "POST",
                f"/ng-control/v1/resources-checkpoint/{operation_name}",
            )
            for operation_name in ("prepare", "status", "commit", "restore", "resume")
        }
    try:
        return endpoints[operation]
    except KeyError as error:
        raise ValueError(
            f"unsupported Gym checkpoint operation {operation!r}"
        ) from error


def participant_groups(
    participants: Sequence[CheckpointParticipant],
) -> tuple[list[CheckpointParticipant], list[CheckpointParticipant]]:
    """Split dependencies from agents for safe resume ordering."""
    dependencies = [
        participant
        for participant in participants
        if participant.kind
        in (CheckpointParticipantKind.POLICY_MODEL, CheckpointParticipantKind.RESOURCES)
    ]
    agents = [
        participant
        for participant in participants
        if participant.kind == CheckpointParticipantKind.AGENT
    ]
    return dependencies, agents


def _completion_receipt_for(
    participant: CheckpointParticipant,
    status: Mapping[str, Any],
    identity: ExecutionIdentity,
) -> AgentCompletionReceipt:
    raw_attempts = status.get("completed_unacknowledged_attempts")
    endpoint = "/ng-control/v1/agent-checkpoint/status"
    if not isinstance(raw_attempts, list):
        raise CheckpointParticipantError(
            participant.participant_id,
            endpoint,
            "Gym agent status is missing completed_unacknowledged_attempts",
            gym_error_code="invalid_response",
        )
    if len(raw_attempts) > MAX_AGENT_COMPLETION_RECEIPTS:
        raise CheckpointParticipantError(
            participant.participant_id,
            endpoint,
            "Gym agent completion receipt inventory exceeds the actor safety bound",
            gym_error_code="response_too_large",
        )
    matches: list[AgentCompletionReceipt] = []
    try:
        for attempt in raw_attempts:
            if not isinstance(attempt, Mapping):
                raise ValueError(
                    "Gym completed-unacknowledged attempt must be an object"
                )
            raw_receipt = attempt.get("completion_receipt")
            if not isinstance(raw_receipt, Mapping):
                raise ValueError(
                    "Gym completed-unacknowledged attempt is missing its completion receipt"
                )
            receipt = AgentCompletionReceipt.from_mapping(raw_receipt)
            if receipt.identity == identity:
                matches.append(receipt)
    except ValueError as error:
        raise CheckpointParticipantError(
            participant.participant_id,
            endpoint,
            str(error),
            gym_error_code="invalid_response",
        ) from error
    if len(matches) != 1:
        raise CheckpointParticipantError(
            participant.participant_id,
            endpoint,
            f"expected exactly one completion receipt for rollout "
            f"{identity.rollout_id!r} attempt {identity.attempt_index}, "
            f"found {len(matches)}",
            gym_error_code="completion_receipt_mismatch",
        )
    return matches[0]


def _server_candidates(global_config: Mapping[str, Any]) -> list[CheckpointParticipant]:
    candidates: list[CheckpointParticipant] = []
    for top_level_name, raw_entry in global_config.items():
        if not isinstance(top_level_name, str) or not isinstance(raw_entry, Mapping):
            continue
        component_keys = [key for key in _COMPONENT_KEYS if key in raw_entry]
        if len(component_keys) != 1:
            continue
        component = component_keys[0]
        typed_entry = raw_entry[component]
        if not isinstance(typed_entry, Mapping) or len(typed_entry) != 1:
            continue
        _, config = next(iter(typed_entry.items()))
        if not isinstance(config, Mapping):
            continue
        host = config.get("host")
        port = config.get("port")
        if (
            not isinstance(host, str)
            or not isinstance(port, int)
            or isinstance(port, bool)
        ):
            raise ValueError(
                f"Gym server {top_level_name!r} is missing a concrete host and port"
            )
        placeholder = CheckpointCapabilities(
            component=component,
            name=top_level_name,
            schema_version=0,
            admission_states=(),
            checkpoint_mode="unknown",
            concurrency_contract="unknown",
            multi_process_mode="unknown",
            num_workers=1,
            instance_role=None,
        )
        candidates.append(
            CheckpointParticipant(
                participant_id=top_level_name,
                name=top_level_name,
                base_url=f"http://{host}:{port}",
                kind=CheckpointParticipantKind.RESOURCES,
                capabilities=placeholder,
            )
        )
    return sorted(candidates, key=lambda participant: participant.participant_id)


def _participant_kind(
    capabilities: CheckpointCapabilities,
) -> CheckpointParticipantKind | None:
    if capabilities.component == "responses_api_models":
        if capabilities.instance_role != "policy":
            return None
        return CheckpointParticipantKind.POLICY_MODEL
    if capabilities.component == "responses_api_agents":
        return CheckpointParticipantKind.AGENT
    if capabilities.component == "resources_servers":
        return CheckpointParticipantKind.RESOURCES
    raise ValueError(f"unknown Gym checkpoint component {capabilities.component!r}")


def _required_capability_string(raw: Mapping[str, Any], key: str) -> str:
    value = raw.get(key)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Gym capabilities {key} must be a non-empty string")
    return value


def _validate_checkpoint_id(checkpoint_id: str) -> None:
    if (
        not isinstance(checkpoint_id, str)
        or _CHECKPOINT_ID_PATTERN.fullmatch(checkpoint_id) is None
    ):
        raise ValueError(
            "checkpoint_id must start with a letter or digit, contain only letters, "
            "digits, dots, dashes, or underscores, and be at most 128 characters"
        )


def _remaining(deadline_ts: float) -> float:
    if (
        not isinstance(deadline_ts, (int, float))
        or isinstance(deadline_ts, bool)
        or not math.isfinite(deadline_ts)
    ):
        raise ValueError("checkpoint deadline_ts must be a finite Unix timestamp")
    remaining = float(deadline_ts) - time.time()
    if remaining <= 0:
        raise CheckpointParticipantError(
            "actor",
            "deadline",
            "absolute checkpoint deadline expired",
            gym_error_code="deadline_exceeded",
        )
    return remaining


async def _read_bounded_response(
    participant: CheckpointParticipant,
    endpoint: str,
    response: aiohttp.ClientResponse,
) -> bytes:
    if (
        response.content_length is not None
        and response.content_length > MAX_CHECKPOINT_RESPONSE_BYTES
    ):
        raise CheckpointParticipantError(
            participant.participant_id,
            endpoint,
            "Gym checkpoint response exceeds the actor safety bound",
            status=response.status,
            gym_error_code="response_too_large",
        )
    body = bytearray()
    async for chunk in response.content.iter_chunked(64 * 1024):
        body.extend(chunk)
        if len(body) > MAX_CHECKPOINT_RESPONSE_BYTES:
            raise CheckpointParticipantError(
                participant.participant_id,
                endpoint,
                "Gym checkpoint response exceeds the actor safety bound",
                status=response.status,
                gym_error_code="response_too_large",
            )
    return bytes(body)


def _decode_response(body: bytes) -> Any:
    if not body:
        return {}
    try:
        return json.loads(body)
    except json.JSONDecodeError:
        return None


def _gym_error(parsed: Any, body: bytes) -> tuple[str | None, str]:
    if isinstance(parsed, Mapping):
        error = parsed.get("error")
        if isinstance(error, Mapping):
            code = error.get("code")
            detail = error.get("detail")
            return (
                code if isinstance(code, str) else None,
                detail
                if isinstance(detail, str)
                else json.dumps(parsed, sort_keys=True),
            )
        detail = parsed.get("detail")
        if isinstance(detail, str):
            return None, detail
    return None, body.decode(errors="replace")


def _checkpoint_file_digests(checkpoint_dir: Path) -> dict[str, str]:
    files: dict[str, str] = {}
    for subdirectory in ("model-ledger", "agent", "resources"):
        root = checkpoint_dir / subdirectory
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            if path.is_file() and not path.is_symlink():
                files[path.relative_to(checkpoint_dir).as_posix()] = _digest(path)
    return files


def _integrity_digest(files: Mapping[str, str]) -> str:
    payload = json.dumps(dict(files), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_obj:
        for block in iter(lambda: file_obj.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _fsync_file(path: Path) -> None:
    with path.open("rb") as file_obj:
        os.fsync(file_obj.fileno())


def _fsync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    directory_fd = os.open(path, flags)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)
