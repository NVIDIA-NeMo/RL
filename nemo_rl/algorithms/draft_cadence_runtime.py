# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
"""Durable cadence checkpoint receipts and resume ledger handling.

The training checkpoint remains the authority.  This module only writes a
receipt after all three training components and the immutable decision prefix
are present, and re-hashes them before accepting a resume.
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from dataclasses import asdict, dataclass
from math import isfinite
from pathlib import Path
from typing import Any, Mapping, Self, cast

from pydantic import BaseModel, model_validator

from nemo_rl.algorithms.draft_update_schedule import (
    AppliedDraftSnapshot,
    DecisionLedgerReceipt,
    DraftDecisionLedger,
    DraftStepTransactionStore,
    DraftUpdateDecision,
    DraftUpdateScheduler,
    validate_decision_ledger_receipt,
)
from nemo_rl.models.policy.draft_config import DraftUpdateScheduleConfig


def write_json_exclusive_atomic(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, separators=(",", ":"), allow_nan=False
        ).encode()
    ).hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _sha256_path(path: Path) -> str:
    if path.is_file():
        return hashlib.sha256(path.read_bytes()).hexdigest()
    if path.is_dir():
        digest = hashlib.sha256()
        for member in sorted(item for item in path.rglob("*") if item.is_file()):
            digest.update(str(member.relative_to(path)).encode())
            digest.update(b"\0")
            digest.update(hashlib.sha256(member.read_bytes()).digest())
        return digest.hexdigest()
    raise ValueError(f"checkpoint artifact is absent: {path}")


def sha256_tree(root: Path, *, exclude: set[str] = set()) -> str:
    digest = hashlib.sha256()
    for member in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = str(member.relative_to(root))
        if relative in exclude:
            continue
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(hashlib.sha256(member.read_bytes()).digest())
    return digest.hexdigest()


class CadenceRuntimeConfig(BaseModel, extra="forbid"):
    enabled: bool = False
    result_dir: str | None = None
    required_checkpoint_steps: tuple[int, ...] = ()

    @model_validator(mode="after")
    def validate_paths(self) -> Self:
        if self.enabled and not self.result_dir:
            raise ValueError("cadence runtime result_dir is required")
        if any(
            type(step) is not int or step <= 0
            for step in self.required_checkpoint_steps
        ):
            raise ValueError("required checkpoint steps must be positive integers")
        return self


def disabled_draft_schedule_payload() -> dict[str, object]:
    return {
        "mode": "disabled",
        "state": {
            "decisions": 0,
            "next_decision_id": 1,
            "attempted_updates": 0,
            "successful_updates": 0,
            "failed_updates": 0,
            "skipped_updates": 0,
            "attempted_refits": 0,
            "successful_refits": 0,
            "failed_refits": 0,
            "skipped_refits": 0,
            "forced_updates": 0,
            "forced_refits": 0,
            "decision_history": [],
        },
        "events": [],
        "not_applicable_metrics": [
            "draft_loss",
            "draft_grad_norm",
            "applied_draft_version",
        ],
    }


def scheduler_decision_high_water(schedule: Mapping[str, object]) -> int:
    state = schedule.get("state")
    if not isinstance(state, Mapping):
        raise ValueError("checkpoint schedule state is absent")
    next_id = state.get("next_decision_id")
    if type(next_id) is not int or next_id < 1:
        raise ValueError("checkpoint schedule cursor is invalid")
    high_water = next_id - 1
    if "decisions" in state and state.get("decisions") != high_water:
        raise ValueError("legacy decisions field disagrees with schedule cursor")
    return high_water


def _checkpoint_member(root: Path, relative: object) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError("checkpoint member path must be a nonempty string")
    member = (root / relative).resolve()
    if root not in member.parents:
        raise ValueError("checkpoint member escapes checkpoint root")
    return member


def _read_ledger(path: Path) -> list[dict[str, object]]:
    raw = path.read_bytes()
    try:
        return [json.loads(line) for line in raw.splitlines()]
    except json.JSONDecodeError as error:
        raise ValueError("checkpoint decision ledger is invalid JSONL") from error


def seal_checkpoint_ledger(
    decision_ledger: DraftDecisionLedger, destination: Path, *, allow_empty: bool
) -> dict[str, object]:
    if decision_ledger.next_decision_id == 1:
        if decision_ledger.sealed_prefixes or not allow_empty:
            raise RuntimeError(
                "empty checkpoint ledger is valid only for disabled draft"
            )
        segments: tuple[DecisionLedgerReceipt, ...] = ()
    else:
        segments = (*decision_ledger.sealed_prefixes, decision_ledger.seal_prefix())
    raw = b""
    expected = 1
    for segment in segments:
        validate_decision_ledger_receipt(segment)
        if segment.first_decision_id != expected:
            raise ValueError("checkpoint ledger segments are not contiguous")
        raw += Path(segment.path).read_bytes()
        expected = segment.last_decision_id + 1
    rows = [json.loads(line) for line in raw.splitlines()]
    if [row.get("decision_id") for row in rows] != list(range(1, len(rows) + 1)):
        raise ValueError("checkpoint ledger prefix is not exactly 1..N")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    _fsync_directory(destination.parent)
    return {
        "relative_path": destination.name,
        "size_bytes": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "first_decision_id": 1 if rows else None,
        "last_decision_id": len(rows),
        "entry_count": len(rows),
    }


@dataclass(slots=True)
class CadenceTerminalEvidence:
    update_receipts_by_decision: dict[int, Mapping[str, object]]
    observations_by_refit_step: dict[int, Mapping[str, object]]

    def state_dict(self) -> dict[str, object]:
        return {
            "update_receipts_by_decision": {
                str(key): dict(value)
                for key, value in self.update_receipts_by_decision.items()
            },
            "observations_by_refit_step": {
                str(key): dict(value)
                for key, value in self.observations_by_refit_step.items()
            },
        }

    @classmethod
    def from_state(cls, state: Mapping[str, object]) -> CadenceTerminalEvidence:
        updates = state.get("update_receipts_by_decision")
        observations = state.get("observations_by_refit_step")
        if not isinstance(updates, Mapping) or not isinstance(observations, Mapping):
            raise ValueError("invalid checkpointed cadence terminal evidence")
        try:
            parsed_updates = {
                int(key): dict(value)
                for key, value in updates.items()
                if isinstance(value, Mapping)
            }
            parsed_observations = {
                int(key): dict(value)
                for key, value in observations.items()
                if isinstance(value, Mapping)
            }
        except (TypeError, ValueError) as error:
            raise ValueError(
                "invalid checkpointed cadence terminal evidence"
            ) from error
        if len(parsed_updates) != len(updates) or len(parsed_observations) != len(
            observations
        ):
            raise ValueError("invalid checkpointed cadence evidence entry")
        return cls(parsed_updates, parsed_observations)


def record_terminal_post_refit_observation(
    evidence: CadenceTerminalEvidence,
    *,
    decision: DraftUpdateDecision,
    last_applied_refit_step: int | None,
    acceptance_rate: float | None,
) -> CadenceTerminalEvidence:
    if (
        last_applied_refit_step is None
        or last_applied_refit_step != decision.global_step - 1
    ):
        return evidence
    if (
        type(acceptance_rate) not in (int, float)
        or not isfinite(float(acceptance_rate))
        or not 0.0 <= float(acceptance_rate) <= 1.0
        or decision.applied_draft_version <= 0
    ):
        raise ValueError("immediate post-refit science observation is invalid")
    observation: dict[str, object] = {
        "refit_step": last_applied_refit_step,
        "observation_step": decision.global_step,
        "applied_draft_version": decision.applied_draft_version,
        "acceptance_rate": float(acceptance_rate),
    }
    previous = evidence.observations_by_refit_step.setdefault(
        last_applied_refit_step, observation
    )
    if previous != observation:
        raise ValueError("conflicting post-refit science observation")
    return evidence


def load_checkpoint_bundle(checkpoint_path: Path) -> Mapping[str, object]:
    root = checkpoint_path.resolve()
    receipt_path = root / "cadence-checkpoint-receipt.json"
    try:
        receipt = json.loads(receipt_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("cadence checkpoint receipt is unreadable") from error
    required = {
        "schema_version",
        "successful",
        "checkpoint_id",
        "checkpoint_path",
        "completed_policy_steps",
        "current_step",
        "checkpoint_tree_sha256",
        "components",
        "scheduler_state_sha256",
        "draft_update_schedule",
        "applied_draft_snapshot",
        "decision_ledger",
        "decision_ledger_prefixes",
        "ledger_high_water",
        "resumed_from",
        "cadence_terminal_evidence",
    }
    if (
        not isinstance(receipt, Mapping)
        or set(receipt) != required
        or receipt.get("schema_version") != 1
        or receipt.get("successful") is not True
        or receipt.get("checkpoint_id") != root.name
        or receipt.get("checkpoint_path") != str(root)
        or type(receipt.get("current_step")) is not int
        or receipt["current_step"] <= 0
        or receipt.get("completed_policy_steps") != receipt["current_step"]
        or receipt.get("checkpoint_id") != f"step_{receipt['current_step']}"
    ):
        raise ValueError("invalid cadence checkpoint identity")
    components = receipt.get("components")
    if not isinstance(components, Mapping) or set(components) != {
        "model",
        "optimizer",
        "dataloader_rng",
    }:
        raise ValueError("cadence checkpoint component schema mismatch")
    for name, binding in components.items():
        if not isinstance(binding, Mapping) or set(binding) != {
            "relative_path",
            "sha256",
        }:
            raise ValueError(f"invalid {name} checkpoint binding")
        if binding.get("sha256") != _sha256_path(
            _checkpoint_member(root, binding.get("relative_path"))
        ):
            raise ValueError(f"{name} checkpoint digest mismatch")
    ledger = receipt.get("decision_ledger")
    ledger_keys = {
        "relative_path",
        "size_bytes",
        "sha256",
        "first_decision_id",
        "last_decision_id",
        "entry_count",
    }
    if not isinstance(ledger, Mapping) or set(ledger) != ledger_keys:
        raise ValueError("missing checkpoint decision-ledger binding")
    ledger_path = _checkpoint_member(root, ledger.get("relative_path"))
    raw_ledger = ledger_path.read_bytes()
    rows = _read_ledger(ledger_path)
    decision_ids = [row.get("decision_id") for row in rows]
    if (
        ledger.get("size_bytes") != len(raw_ledger)
        or ledger.get("sha256") != hashlib.sha256(raw_ledger).hexdigest()
        or ledger.get("entry_count") != len(rows)
        or decision_ids != list(range(1, len(rows) + 1))
        or ledger.get("first_decision_id") != (1 if rows else None)
        or ledger.get("last_decision_id") != len(rows)
        or receipt.get("decision_ledger_prefixes") != [ledger]
    ):
        raise ValueError("checkpoint decision-ledger receipt mismatch")
    schedule = receipt.get("draft_update_schedule")
    if (
        not isinstance(schedule, Mapping)
        or receipt.get("scheduler_state_sha256") != canonical_sha256(schedule)
        or scheduler_decision_high_water(schedule) != len(rows)
        or receipt.get("ledger_high_water") != len(rows)
    ):
        raise ValueError("checkpoint scheduler/ledger high-water mismatch")
    disabled = schedule.get("mode") == "disabled"
    if disabled:
        if (
            schedule != disabled_draft_schedule_payload()
            or rows
            or ledger.get("size_bytes") != 0
            or ledger.get("sha256") != hashlib.sha256(b"").hexdigest()
        ):
            raise ValueError("disabled draft checkpoint ledger must be exactly empty")
    elif not rows:
        raise ValueError("enabled draft checkpoint ledger cannot be empty")
    if not disabled:
        raw_snapshot = receipt.get("applied_draft_snapshot")
        if not isinstance(raw_snapshot, Mapping) or set(raw_snapshot) != {
            "version",
            "path",
            "size_bytes",
            "sha256",
        }:
            raise ValueError("enabled draft checkpoint requires an applied snapshot")
        try:
            snapshot = AppliedDraftSnapshot(**dict(raw_snapshot))
            raw_snapshot_bytes = Path(snapshot.path).read_bytes()
        except (OSError, TypeError) as error:
            raise ValueError("applied draft snapshot is unreadable") from error
        applied_version = schedule["state"].get("applied_draft_version")
        if (
            type(snapshot.version) is not int
            or snapshot.version != applied_version
            or type(snapshot.size_bytes) is not int
            or snapshot.size_bytes != len(raw_snapshot_bytes)
            or snapshot.sha256 != hashlib.sha256(raw_snapshot_bytes).hexdigest()
        ):
            raise ValueError("applied draft snapshot version or digest mismatch")
    if receipt.get("checkpoint_tree_sha256") != sha256_tree(
        root, exclude={"cadence-checkpoint-receipt.json"}
    ):
        raise ValueError("checkpoint tree digest mismatch")
    evidence = receipt.get("cadence_terminal_evidence")
    if not isinstance(evidence, Mapping):
        raise ValueError("checkpoint cadence terminal evidence is absent")
    CadenceTerminalEvidence.from_state(evidence)
    return receipt


@dataclass(frozen=True, slots=True)
class ResumeLedgerOpenResult:
    ledger: DraftDecisionLedger
    quarantine_receipt_path: Path


def _move_ledger_to_quarantine(source: Path, destination: Path) -> None:
    os.replace(source, destination)


def reconcile_ledger_quarantine(
    recovery_dir: Path, result_root: Path
) -> Mapping[str, object]:
    intent_path = recovery_dir / "ledger-quarantine-intent.json"
    intent = json.loads(intent_path.read_text())
    root = result_root.resolve()
    if (
        intent.get("schema_version") != 1
        or intent.get("state") != "intent"
        or not isinstance(intent.get("checkpoint_id"), str)
        or not isinstance(intent.get("recovery_id"), str)
        or not isinstance(intent.get("artifacts"), list)
        or not isinstance(intent.get("new_suffix_path"), str)
    ):
        raise ValueError("invalid ledger quarantine intent")
    for artifact in intent["artifacts"]:
        if not isinstance(artifact, Mapping):
            raise ValueError("invalid ledger quarantine artifact")
        source = Path(str(artifact.get("original_path"))).resolve()
        destination = Path(str(artifact.get("quarantine_path"))).resolve()
        if source.parent != root or destination.parent != recovery_dir.resolve():
            raise ValueError("ledger quarantine path escapes transaction roots")
        source_exists, destination_exists = source.is_file(), destination.is_file()
        if source_exists == destination_exists:
            raise RuntimeError(
                "ledger quarantine has ambiguous source/destination state"
            )
        if source_exists:
            raw = source.read_bytes()
            if (
                artifact.get("size_bytes") != len(raw)
                or artifact.get("sha256") != hashlib.sha256(raw).hexdigest()
            ):
                raise ValueError("ledger quarantine source digest mismatch")
            _move_ledger_to_quarantine(source, destination)
        raw = destination.read_bytes()
        if (
            artifact.get("size_bytes") != len(raw)
            or artifact.get("sha256") != hashlib.sha256(raw).hexdigest()
        ):
            raise ValueError("ledger quarantine destination digest mismatch")
    for directory in (root, recovery_dir.parent, recovery_dir):
        _fsync_directory(directory)
    receipt_path = recovery_dir / "ledger-quarantine-receipt.json"
    receipt = {
        **intent,
        "state": "resolved",
        "receipt_path": str(receipt_path.resolve()),
    }
    write_json_exclusive_atomic(receipt_path, receipt)
    return receipt


def open_resume_decision_ledger(
    checkpoint_path: Path, result_root: Path
) -> ResumeLedgerOpenResult:
    bundle = load_checkpoint_bundle(checkpoint_path)
    root = result_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if checkpoint_path.resolve().name != bundle["checkpoint_id"]:
        raise ValueError("resume checkpoint identity does not match cadence receipt")
    recovery_parent = root / "recovery"
    recovery_parent.mkdir(parents=True, exist_ok=True)
    incomplete = [
        path
        for path in recovery_parent.glob("resume-*")
        if (path / "ledger-quarantine-intent.json").is_file()
        and not (path / "ledger-quarantine-receipt.json").exists()
    ]
    if len(incomplete) > 1:
        raise RuntimeError("multiple incomplete ledger quarantine transactions")
    if incomplete:
        receipt = reconcile_ledger_quarantine(incomplete[0], root)
    else:
        recovery_id = str(uuid.uuid4())
        recovery_dir = (
            recovery_parent / f"resume-{bundle['checkpoint_id']}-{recovery_id}"
        )
        recovery_dir.mkdir(exist_ok=False)
        candidates = sorted(
            {
                *root.glob("draft-decision-ledger-after-step_*.jsonl"),
                *root.glob("draft-decision-ledger-resume-step_*.jsonl"),
            }
        )
        artifacts: list[dict[str, object]] = []
        for index, source in enumerate(candidates):
            raw = source.read_bytes()
            destination = recovery_dir / f"{index:04d}-{source.name}"
            artifacts.append(
                {
                    "original_path": str(source.resolve()),
                    "quarantine_path": str(destination.resolve()),
                    "size_bytes": len(raw),
                    "sha256": hashlib.sha256(raw).hexdigest(),
                }
            )
        suffix = (
            root
            / f"draft-decision-ledger-resume-{bundle['checkpoint_id']}-{recovery_id}.jsonl"
        )
        write_json_exclusive_atomic(
            recovery_dir / "ledger-quarantine-intent.json",
            {
                "schema_version": 1,
                "state": "intent",
                "checkpoint_id": bundle["checkpoint_id"],
                "recovery_id": recovery_id,
                "artifacts": artifacts,
                "new_suffix_path": str(suffix),
            },
        )
        receipt = reconcile_ledger_quarantine(recovery_dir, root)
    if receipt.get("checkpoint_id") != bundle["checkpoint_id"]:
        raise ValueError("ledger quarantine checkpoint mismatch")
    binding = bundle["decision_ledger"]
    assert isinstance(binding, Mapping)
    high_water = int(bundle["ledger_high_water"])
    prefixes: tuple[DecisionLedgerReceipt, ...] = ()
    if high_water:
        prefix = DecisionLedgerReceipt(
            path=str(checkpoint_path.resolve() / str(binding["relative_path"])),
            size_bytes=int(binding["size_bytes"]),
            sha256=str(binding["sha256"]),
            first_decision_id=int(binding["first_decision_id"]),
            last_decision_id=int(binding["last_decision_id"]),
            entry_count=int(binding["entry_count"]),
        )
        validate_decision_ledger_receipt(prefix)
        prefixes = (prefix,)
    suffix = Path(str(receipt["new_suffix_path"])).resolve()
    if suffix.parent != root or suffix.exists():
        raise FileExistsError("resume ledger suffix identity collision")
    return ResumeLedgerOpenResult(
        DraftDecisionLedger(suffix, sealed_prefixes=prefixes),
        Path(str(receipt["receipt_path"])),
    )


def recover_draft_step_transactions(
    *,
    config: DraftUpdateScheduleConfig | None,
    checkpoint_path: Path,
    transaction_store: DraftStepTransactionStore,
    decision_ledger: DraftDecisionLedger,
    save_state: Any,
) -> DraftUpdateScheduler | None:
    """Recover immutable transaction evidence without advancing past a full checkpoint."""
    checkpoint = load_checkpoint_bundle(checkpoint_path)
    checkpoint_id = str(checkpoint["checkpoint_id"])
    high_water = int(checkpoint["ledger_high_water"])
    schedule = checkpoint["draft_update_schedule"]
    assert isinstance(schedule, Mapping)
    disabled = schedule.get("mode") == "disabled"
    resolutions = transaction_store.resolutions_since(checkpoint_id)
    pending = transaction_store.pending_intents()
    if disabled and (config is not None or high_water != 0 or resolutions or pending):
        raise ValueError("disabled draft resume must have no scheduler transactions")
    if not disabled and config is None:
        raise ValueError("enabled draft resume requires schedule config")
    checkpoint_ledger = checkpoint["decision_ledger"]
    assert isinstance(checkpoint_ledger, Mapping)
    ledger_rows = _read_ledger(
        checkpoint_path.resolve() / str(checkpoint_ledger["relative_path"])
    )
    known = {item.transaction.transaction_id: item for item in resolutions}
    if len(known) != len(resolutions):
        raise ValueError("duplicate draft-step transaction resolution")
    for intent in pending:
        resolution = known.get(intent.transaction_id)
        if resolution is None:
            resolution = transaction_store.resolve_intent_for_recovery(
                intent,
                apply_receipt=transaction_store.lookup_durable_apply_receipt(intent),
            )
            known[intent.transaction_id] = resolution
        if resolution.transaction != intent or resolution.decision != intent.decision:
            raise ValueError("draft-step resolution differs from durable intent")
    for resolution in known.values():
        if resolution.decision.decision_id <= high_water:
            transaction_store.validate_checkpoint_contains(checkpoint_id, resolution)
            row = ledger_rows[resolution.decision.decision_id - 1]
            if (
                row.get("decision_id") != resolution.decision.decision_id
                or row.get("global_step") != resolution.decision.global_step
                or row.get("outcome") != dict(resolution.outcome)
            ):
                raise ValueError(
                    "checkpoint ledger differs from transaction resolution"
                )
            if resolution.applied_snapshot is not None:
                checkpoint_snapshot = checkpoint["applied_draft_snapshot"]
                if checkpoint_snapshot != asdict(resolution.applied_snapshot):
                    raise ValueError("checkpoint snapshot differs from transaction")
    transaction_store.discard_after_checkpoint(
        checkpoint_id=checkpoint_id, ledger_high_water=high_water
    )
    save_state.draft_update_schedule = schedule
    save_state.applied_draft_snapshot = checkpoint["applied_draft_snapshot"]
    save_state.draft_terminal_evidence = dict(
        cast(Mapping[str, object], checkpoint["cadence_terminal_evidence"])
    )
    save_state.draft_decision_ledger_prefixes = (
        []
        if disabled
        else list(
            cast(list[Mapping[str, object]], checkpoint["decision_ledger_prefixes"])
        )
    )
    if disabled:
        return None
    from nemo_rl.algorithms.grpo import restore_draft_update_scheduler

    assert config is not None
    return restore_draft_update_scheduler(
        config,
        schedule,
        origin_step=int(checkpoint["completed_policy_steps"]),
        resuming_from_checkpoint=True,
    )


class CadenceRuntimeWriter:
    def __init__(self, config: CadenceRuntimeConfig) -> None:
        if not config.enabled or config.result_dir is None:
            raise ValueError("cadence runtime writer requires enabled config")
        self.root = Path(config.result_dir).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.required_steps = frozenset(config.required_checkpoint_steps)
        self.receipt_session_id = str(uuid.uuid4())
        self.update_receipt_root = self.root / "update-receipts"
        self.update_receipt_root.mkdir(exist_ok=True)
        _fsync_directory(self.root)

    def successful_update_closed(
        self,
        *,
        decision: DraftUpdateDecision,
        worker_receipt: Mapping[str, object],
        evidence: CadenceTerminalEvidence,
        save_state: Any,
    ) -> CadenceTerminalEvidence:
        if not decision.update_requested:
            raise ValueError("cannot receipt a skipped draft update")
        required = {
            "successful": True,
            "decision_id": decision.decision_id,
            "global_step": decision.global_step,
        }
        if any(worker_receipt.get(key) != value for key, value in required.items()):
            raise ValueError("worker update receipt disagrees with decision")
        for key in ("draft_model_sha256", "draft_optimizer_sha256"):
            value = worker_receipt.get(key)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or set(value) - set("0123456789abcdef")
            ):
                raise ValueError(f"worker update receipt lacks {key}")
        if decision.decision_id in evidence.update_receipts_by_decision:
            raise RuntimeError("duplicate successful-update evidence")
        existing = getattr(save_state, "draft_terminal_evidence", None)
        if existing not in (None, evidence.state_dict()):
            raise RuntimeError("checkpointed terminal evidence diverged before update")
        path = (
            self.update_receipt_root
            / f"{self.receipt_session_id}-decision_{decision.decision_id}.json"
        )
        write_json_exclusive_atomic(
            path,
            {
                "schema_version": 1,
                **required,
                "draft_model_sha256": worker_receipt["draft_model_sha256"],
                "draft_optimizer_sha256": worker_receipt["draft_optimizer_sha256"],
            },
        )
        raw = path.read_bytes()
        evidence.update_receipts_by_decision[decision.decision_id] = {
            "successful": True,
            "decision_id": decision.decision_id,
            "global_step": decision.global_step,
            "path": str(path.resolve()),
            "size_bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
        }
        save_state.draft_terminal_evidence = evidence.state_dict()
        return evidence

    def checkpoint_closed(
        self,
        *,
        current_step: int,
        checkpoint_path: Path,
        save_state: Any,
        component_paths: Mapping[str, Path],
        decision_ledger: DraftDecisionLedger,
        terminal_evidence: CadenceTerminalEvidence,
        resumed_from: str | None = None,
    ) -> DraftDecisionLedger:
        expected = (self.root / "checkpoints" / f"step_{current_step}").resolve()
        if checkpoint_path.resolve() != expected or not checkpoint_path.is_dir():
            raise ValueError("checkpoint path is outside cadence result identity")
        if set(component_paths) != {"model", "optimizer", "dataloader_rng"}:
            raise RuntimeError("cadence cannot close a partial training checkpoint")
        components: dict[str, dict[str, str]] = {}
        for name, path in component_paths.items():
            try:
                relative = path.resolve().relative_to(expected)
            except ValueError as error:
                raise ValueError("checkpoint component escapes checkpoint") from error
            components[name] = {
                "relative_path": str(relative),
                "sha256": _sha256_path(path.resolve()),
            }
        schedule = save_state.draft_update_schedule
        if not isinstance(schedule, Mapping) or not isinstance(
            schedule.get("state"), Mapping
        ):
            raise ValueError("checkpoint requires scheduler state")
        disabled = schedule.get("mode") == "disabled"
        if disabled and schedule != disabled_draft_schedule_payload():
            raise ValueError("disabled draft schedule payload is not neutral")
        ledger = seal_checkpoint_ledger(
            decision_ledger,
            expected / "draft-decision-ledger.jsonl",
            allow_empty=disabled,
        )
        high_water = int(ledger["last_decision_id"])
        if scheduler_decision_high_water(schedule) != high_water:
            raise ValueError("terminal scheduler decisions differ from ledger")
        if not disabled and high_water == 0:
            raise ValueError("enabled draft checkpoint cannot have an empty ledger")
        payload: dict[str, object] = {
            "schema_version": 1,
            "successful": True,
            "checkpoint_id": f"step_{current_step}",
            "checkpoint_path": str(expected),
            "completed_policy_steps": current_step,
            "current_step": current_step,
            "checkpoint_tree_sha256": sha256_tree(
                expected, exclude={"cadence-checkpoint-receipt.json"}
            ),
            "components": components,
            "scheduler_state_sha256": canonical_sha256(schedule),
            "draft_update_schedule": schedule,
            "applied_draft_snapshot": save_state.applied_draft_snapshot,
            "cadence_terminal_evidence": terminal_evidence.state_dict(),
            "decision_ledger": ledger,
            "decision_ledger_prefixes": [ledger],
            "ledger_high_water": high_water,
            "resumed_from": resumed_from,
        }
        save_state.draft_terminal_evidence = terminal_evidence.state_dict()
        write_json_exclusive_atomic(
            expected / "cadence-checkpoint-receipt.json", payload
        )
        if current_step in self.required_steps:
            write_json_exclusive_atomic(
                self.root / f"checkpoint-runtime-step_{current_step}.json", payload
            )
        if disabled:
            save_state.draft_decision_ledger_prefixes = []
            return DraftDecisionLedger(
                self.root / f"draft-decision-ledger-after-step_{current_step}.jsonl"
            )
        prefix = DecisionLedgerReceipt(
            path=str(expected / str(ledger["relative_path"])),
            size_bytes=int(ledger["size_bytes"]),
            sha256=str(ledger["sha256"]),
            first_decision_id=int(ledger["first_decision_id"]),
            last_decision_id=int(ledger["last_decision_id"]),
            entry_count=int(ledger["entry_count"]),
        )
        save_state.draft_decision_ledger_prefixes = [asdict(prefix)]
        return DraftDecisionLedger(
            self.root / f"draft-decision-ledger-after-step_{current_step}.jsonl",
            sealed_prefixes=(prefix,),
        )

    def terminal_closed(
        self,
        *,
        current_step: int,
        final_checkpoint_path: Path,
        terminal_evidence: CadenceTerminalEvidence,
    ) -> None:
        missing = [
            step
            for step in sorted(self.required_steps)
            if not (self.root / f"checkpoint-runtime-step_{step}.json").is_file()
        ]
        if missing:
            raise RuntimeError(f"missing required cadence checkpoints: {missing}")
        checkpoint = load_checkpoint_bundle(final_checkpoint_path)
        if checkpoint.get("current_step") != current_step:
            raise ValueError("final checkpoint step disagrees with terminal step")
        write_json_exclusive_atomic(self.root / "checkpoint-runtime.json", checkpoint)
        write_json_exclusive_atomic(
            self.root / "schedule-runtime.json",
            build_terminal_schedule_payload(checkpoint, terminal_evidence),
        )


def build_terminal_schedule_payload(
    checkpoint: Mapping[str, object], evidence: CadenceTerminalEvidence
) -> dict[str, object]:
    if checkpoint.get("cadence_terminal_evidence") != evidence.state_dict():
        raise ValueError("terminal evidence differs from final checkpoint")
    schedule = checkpoint.get("draft_update_schedule")
    if not isinstance(schedule, Mapping):
        raise ValueError("terminal checkpoint lacks schedule state")
    current_step = int(checkpoint["current_step"])
    zero_fields = {
        key: 0
        for key in (
            "attempted_updates",
            "successful_updates",
            "failed_updates",
            "skipped_updates",
            "attempted_refits",
            "successful_refits",
            "failed_refits",
            "skipped_refits",
            "forced_updates",
            "forced_refits",
        )
    }
    if schedule.get("mode") == "disabled":
        if evidence.update_receipts_by_decision or evidence.observations_by_refit_step:
            raise ValueError("disabled draft cannot have terminal events")
        return {
            "mode": "disabled",
            "current_step": current_step,
            **zero_fields,
            "policy_refit_count": current_step,
            "decision_ids": [],
            "global_steps": [],
            "updated_steps": [],
            "refit_steps": [],
            "forced_update_steps": [],
            "forced_refit_steps": [],
            "update_receipts": [],
            "post_event_observations": [],
            "pending_post_event_steps": [],
            "refit_versions": [],
            "decision_reasons": [],
            "decision_ledger_segments": [],
            "not_applicable_metrics": [
                "applied_draft_version",
                "draft_grad_norm",
                "draft_loss",
            ],
        }
    ledger = checkpoint.get("decision_ledger")
    config = schedule.get("config")
    if not isinstance(ledger, Mapping) or not isinstance(config, Mapping):
        raise ValueError("terminal checkpoint lacks schedule config or ledger")
    path = Path(str(checkpoint["checkpoint_path"])) / str(ledger["relative_path"])
    rows = _read_ledger(path)
    if (
        len(rows) != scheduler_decision_high_water(schedule)
        or not rows
        or [row.get("decision_id") for row in rows] != list(range(1, len(rows) + 1))
        or rows[-1].get("global_step") != current_step
    ):
        raise ValueError("terminal ledger does not cover the scheduler cursor")
    update_rows = [row for row in rows if row.get("update_requested")]
    refit_rows = [row for row in rows if row.get("draft_refit_requested")]
    if set(evidence.update_receipts_by_decision) != {
        int(row["decision_id"]) for row in update_rows
    }:
        raise ValueError("terminal update-receipt cardinality mismatch")
    update_receipts: list[dict[str, object]] = []
    for row in update_rows:
        decision_id, global_step = int(row["decision_id"]), int(row["global_step"])
        binding = evidence.update_receipts_by_decision[decision_id]
        path_for_receipt = Path(str(binding.get("path"))).resolve()
        raw_receipt = path_for_receipt.read_bytes()
        receipt_payload = json.loads(raw_receipt)
        if (
            binding.get("successful") is not True
            or binding.get("decision_id") != decision_id
            or binding.get("global_step") != global_step
            or binding.get("size_bytes") != len(raw_receipt)
            or binding.get("sha256") != hashlib.sha256(raw_receipt).hexdigest()
            or not isinstance(receipt_payload, Mapping)
            or receipt_payload.get("schema_version") != 1
            or receipt_payload.get("successful") is not True
            or receipt_payload.get("decision_id") != decision_id
            or receipt_payload.get("global_step") != global_step
        ):
            raise ValueError("terminal update receipt is not digest bound")
        for key in ("draft_model_sha256", "draft_optimizer_sha256"):
            digest = receipt_payload.get(key)
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or set(digest) - set("0123456789abcdef")
            ):
                raise ValueError("terminal update receipt is not digest bound")
        update_receipts.append(dict(binding))
    if any(
        cast(Mapping[str, object], row["outcome"]).get("update_attempted")
        and not cast(Mapping[str, object], row["outcome"]).get("update_successful")
        or cast(Mapping[str, object], row["outcome"]).get("draft_refit_attempted")
        and not cast(Mapping[str, object], row["outcome"]).get("draft_refit_successful")
        for row in rows
    ):
        raise ValueError("successful terminal payload cannot contain failed work")
    observable_steps = [
        int(row["global_step"])
        for row in refit_rows
        if int(row["global_step"]) < current_step
    ]
    if set(evidence.observations_by_refit_step) != set(observable_steps):
        raise ValueError("terminal post-refit observation cardinality mismatch")
    resumed_from = checkpoint.get("resumed_from")
    resume_after = (
        None
        if resumed_from is None
        else int(Path(str(resumed_from)).name.removeprefix("step_"))
    )
    version_by_step = {
        int(row["global_step"]): int(row["decision_id"]) for row in refit_rows
    }
    observations: list[dict[str, object]] = []
    for row in refit_rows:
        step = int(row["global_step"])
        if step >= current_step:
            continue
        observation = evidence.observations_by_refit_step.get(step)
        if observation is None:
            raise ValueError("terminal post-refit observation cardinality mismatch")
        acceptance = observation.get("acceptance_rate")
        if (
            observation.get("refit_step") != step
            or observation.get("observation_step") != step + 1
            or observation.get("applied_draft_version") != version_by_step[step]
            or type(acceptance) not in (int, float)
            or not isfinite(float(acceptance))
            or not 0.0 <= float(acceptance) <= 1.0
        ):
            raise ValueError("terminal post-refit observation mismatch")
        if resume_after is None or step >= resume_after:
            observations.append(dict(observation))
    outcomes = [row["outcome"] for row in rows]

    def count(key: str) -> int:
        return sum(
            bool(cast(Mapping[str, object], outcome).get(key)) for outcome in outcomes
        )

    return {
        "mode": config["mode"],
        "current_step": current_step,
        "attempted_updates": count("update_attempted"),
        "successful_updates": count("update_successful"),
        "failed_updates": count("update_attempted") - count("update_successful"),
        "skipped_updates": count("update_skipped"),
        "attempted_refits": count("draft_refit_attempted"),
        "successful_refits": count("draft_refit_successful"),
        "failed_refits": count("draft_refit_attempted")
        - count("draft_refit_successful"),
        "skipped_refits": count("draft_refit_skipped"),
        "forced_updates": count("forced_update"),
        "forced_refits": count("forced_refit"),
        "policy_refit_count": current_step,
        "decision_ids": [int(row["decision_id"]) for row in rows],
        "global_steps": [int(row["global_step"]) for row in rows],
        "updated_steps": [int(row["global_step"]) for row in update_rows],
        "refit_steps": [int(row["global_step"]) for row in refit_rows],
        "forced_update_steps": [
            int(row["global_step"])
            for row in rows
            if cast(Mapping[str, object], row["outcome"]).get("forced_update")
        ],
        "forced_refit_steps": [
            int(row["global_step"])
            for row in rows
            if cast(Mapping[str, object], row["outcome"]).get("forced_refit")
        ],
        "update_receipts": update_receipts,
        "post_event_observations": observations,
        "pending_post_event_steps": [
            int(row["global_step"])
            for row in refit_rows
            if int(row["global_step"]) == current_step
        ],
        "refit_versions": [
            {
                "refit_step": int(row["global_step"]),
                "applied_draft_version": int(row["decision_id"]),
            }
            for row in refit_rows
        ],
        "decision_reasons": [str(row["reason"]) for row in rows],
        "decision_ledger_segments": [{"path": str(path.resolve()), **dict(ledger)}],
    }
