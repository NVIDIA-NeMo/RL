from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

from research.qwen3_8b_draft_cadence_200step.matrix import (
    ADAPTIVE_SCHEDULE,
    Arm,
    CHECKPOINT_STEPS,
)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"required receipt is absent: {path}")
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"receipt must be a JSON object: {path}")
    return payload


def _read_ledger(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise ValueError(f"decision ledger is absent: {path}")
    rows = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"ledger row {line_number} must be an object")
        rows.append(payload)
    return rows


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and not (set(value) - set("0123456789abcdef"))
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    for member in sorted(path for path in root.rglob("*") if path.is_file()):
        relative = str(member.relative_to(root))
        if relative == "cadence-checkpoint-receipt.json":
            continue
        digest.update(relative.encode())
        digest.update(b"\0")
        digest.update(bytes.fromhex(_sha256_file(member)))
    return digest.hexdigest()


def _validate_checkpoint_receipt(
    checkpoint: Path,
    receipt: dict[str, Any],
    *,
    expected_step: int,
    expected_high_water: int,
    verify_tree: bool,
) -> None:
    step_values = [
        receipt[key]
        for key in ("checkpoint_step", "current_step", "completed_policy_steps")
        if key in receipt
    ]
    high_water_values = [
        receipt[key]
        for key in ("last_decision_id", "ledger_high_water")
        if key in receipt
    ]
    if (
        receipt.get("successful") is not True
        or not step_values
        or any(value != expected_step for value in step_values)
        or not high_water_values
        or any(value != expected_high_water for value in high_water_values)
        or receipt.get("checkpoint_path") != str(checkpoint.resolve())
    ):
        raise ValueError("checkpoint identity or ledger high-water is inconsistent")
    binding = receipt.get("decision_ledger")
    if not isinstance(binding, dict):
        raise ValueError("checkpoint decision-ledger binding is absent")
    relative = binding.get("relative_path")
    if not isinstance(relative, str) or not relative:
        raise ValueError("checkpoint decision-ledger path is invalid")
    ledger_path = (checkpoint / relative).resolve()
    try:
        ledger_path.relative_to(checkpoint.resolve())
    except ValueError as error:
        raise ValueError(
            "checkpoint decision-ledger path escapes checkpoint"
        ) from error
    if not ledger_path.is_file():
        raise ValueError("checkpoint decision-ledger is absent")
    raw = ledger_path.read_bytes()
    if (
        binding.get("size_bytes") != len(raw)
        or not _is_sha256(binding.get("sha256"))
        or binding.get("sha256") != hashlib.sha256(raw).hexdigest()
    ):
        raise ValueError("checkpoint decision-ledger digest is invalid")
    rows = _read_ledger(ledger_path)
    expected_ids = list(range(1, expected_high_water + 1))
    if (
        binding.get("first_decision_id") != (1 if expected_ids else None)
        or binding.get("last_decision_id") != expected_high_water
        or binding.get("entry_count") != len(expected_ids)
        or [row.get("decision_id") for row in rows] != expected_ids
        or [row.get("global_step") for row in rows] != expected_ids
    ):
        raise ValueError("checkpoint decision-ledger prefix is not contiguous")
    tree_digest = receipt.get("checkpoint_tree_sha256")
    if not _is_sha256(tree_digest):
        raise ValueError("checkpoint tree digest is absent")
    if verify_tree and tree_digest != _checkpoint_tree_sha256(checkpoint):
        raise ValueError("checkpoint tree digest is invalid")


def validate_adaptive_decisions(rows: list[dict[str, Any]]) -> None:
    alpha = float(ADAPTIVE_SCHEDULE["ewma_alpha"])
    min_interval = int(ADAPTIVE_SCHEDULE["min_interval"])
    max_interval = int(ADAPTIVE_SCHEDULE["max_interval"])
    min_observations = int(ADAPTIVE_SCHEDULE["min_observations"])
    degradation = float(ADAPTIVE_SCHEDULE["degradation_threshold"])
    recovery = float(ADAPTIVE_SCHEDULE["recovery_threshold"])
    max_burst = int(ADAPTIVE_SCHEDULE["max_burst_updates"])
    acceptance_ewma: float | None = None
    reference_ewma: float | None = None
    valid_observations = 0
    phase = "monitoring"
    burst_updates = 0
    last_update_step: int | None = None
    applied_version = 0
    for expected_step, row in enumerate(rows, 1):
        observation = row.get("observed_acceptance")
        if (
            type(observation) not in (int, float)
            or not math.isfinite(float(observation))
            or not 0.0 <= float(observation) <= 1.0
        ):
            raise ValueError(f"adaptive observation is invalid at step {expected_step}")
        observation = float(observation)
        accepted = row.get("accepted_tokens")
        drafted = row.get("draft_tokens")
        if accepted is not None or drafted is not None:
            if (
                type(accepted) not in (int, float)
                or type(drafted) not in (int, float)
                or float(drafted) <= 0.0
                or not math.isclose(
                    observation, float(accepted) / float(drafted), abs_tol=1e-12
                )
            ):
                raise ValueError(
                    f"observed acceptance is not bound to rollout counts at step {expected_step}"
                )
        acceptance_ewma = (
            observation
            if acceptance_ewma is None
            else alpha * observation + (1.0 - alpha) * acceptance_ewma
        )
        valid_observations += 1
        if reference_ewma is None and valid_observations >= min_observations:
            reference_ewma = acceptance_ewma
        elif (
            phase == "monitoring"
            and reference_ewma is not None
            and acceptance_ewma > reference_ewma
        ):
            reference_ewma = acceptance_ewma

        update = False
        forced = False
        reason = "none"
        update_age = expected_step - (last_update_step or 0)
        if phase == "awaiting_post_refit_observation":
            if reference_ewma is not None:
                gap = reference_ewma - acceptance_ewma
                if gap <= recovery:
                    phase = "monitoring"
                    burst_updates = 0
                elif burst_updates >= max_burst:
                    raise ValueError(
                        "adaptive replay exhausted max_burst_updates without recovery"
                    )
                else:
                    update = True
                    reason = "adaptive_burst"
                    phase = "training_burst"
        elif update_age >= max_interval:
            update = True
            forced = True
            reason = "max_interval"
        elif (
            update_age >= min_interval
            and reference_ewma is not None
            and reference_ewma - acceptance_ewma >= degradation
        ):
            update = True
            reason = "adaptive_degradation"
            phase = "training_burst"

        expected = {
            "decision_id": expected_step,
            "global_step": expected_step,
            "update_requested": update,
            "draft_refit_requested": update,
            "reason": reason,
            "forced": forced,
            "applied_draft_version_before_step": applied_version,
        }
        if any(row.get(key) != value for key, value in expected.items()):
            raise ValueError(f"adaptive decision mismatch at step {expected_step}")
        if update:
            last_update_step = expected_step
            applied_version = expected_step
            burst_updates += 1
            phase = "awaiting_post_refit_observation"


def _expected_update_steps(arm: Arm, rows: list[dict[str, Any]]) -> set[int]:
    if arm.cadence != "adaptive":
        return set(arm.deterministic_update_steps())
    validate_adaptive_decisions(rows)
    steps = {
        int(row["global_step"]) for row in rows if row.get("update_requested") is True
    }
    if not steps:
        raise ValueError("adaptive run has no scheduled update")
    ordered = sorted(steps)
    if (
        ordered[0] > 20
        or any(right - left > 20 for left, right in zip(ordered, ordered[1:]))
        or arm.max_steps - ordered[-1] > 20
    ):
        raise ValueError("adaptive update gap exceeds max_interval=20")
    burst_length = 0
    for row in rows:
        requested = row.get("update_requested") is True
        reason = row.get("reason")
        if requested and reason not in {
            "adaptive_degradation",
            "adaptive_burst",
            "max_interval",
        }:
            raise ValueError("adaptive requested update lacks a scheduler reason")
        if not requested and reason != "none":
            raise ValueError("adaptive skipped update has a non-none reason")
        if row.get("forced") is not (reason == "max_interval"):
            raise ValueError("adaptive forced flag disagrees with scheduler reason")
        burst_length = burst_length + 1 if requested else 0
        if burst_length > 10:
            raise ValueError("adaptive update burst exceeds max_burst_updates=10")
    return steps


def validate_arm_receipts(root: Path, arm: Arm) -> dict[str, Any]:
    runtime = _read_json(root / "runtime-evidence.json")
    if (
        runtime.get("target_revision") != arm.target_revision
        or runtime.get("drafter_revision") != arm.drafter_revision
    ):
        raise ValueError("runtime model revision provenance does not match the arm")
    initial_refit = runtime.get("initial_draft_refit")
    if arm.drafter == "none":
        if initial_refit is not None:
            raise ValueError("baseline must not contain an initial draft refit")
    elif not isinstance(initial_refit, dict) or initial_refit != {
        "attempted": True,
        "successful": True,
        "serving_version": 0,
    }:
        raise ValueError("initial draft refit and version-0 publication are not proven")
    if runtime.get("cuda_graph_mode") != "PIECEWISE" or runtime.get(
        "cuda_graph_capture_sizes"
    ) != [
        1,
        2,
        4,
        6,
        8,
        10,
        12,
        16,
        18,
        20,
        24,
        28,
        30,
        32,
        36,
        40,
        42,
        48,
        50,
        56,
        60,
        64,
    ]:
        raise ValueError("CUDA Graph mode or bucket coverage is not proven")
    if (
        runtime.get("step_1_complete") is not True
        or runtime.get("step_2_complete") is not True
    ):
        raise ValueError("Step 1/Step 2 runtime evidence is incomplete")
    terminal = _read_json(root / "terminal.json")
    if (
        terminal.get("terminal") is not True
        or terminal.get("exit_code") != 0
        or terminal.get("requested_policy_steps") != arm.max_steps
        or terminal.get("completed_policy_steps") != arm.max_steps
    ):
        raise ValueError(f"{arm.name} is not terminal 200/200 success")
    rows = _read_ledger(root / "decision-ledger.jsonl")
    expected_decision_count = 0 if arm.cadence == "baseline" else arm.max_steps
    if len(rows) != expected_decision_count:
        raise ValueError(f"expected {arm.max_steps} decision rows, found {len(rows)}")
    expected_ids = list(range(1, expected_decision_count + 1))
    if [row.get("decision_id") for row in rows] != expected_ids:
        raise ValueError("decision ledger IDs are duplicate, gapped, or reordered")
    if [row.get("global_step") for row in rows] != expected_ids:
        raise ValueError("global steps are duplicate, gapped, or reordered")
    expected_updates = (
        set() if arm.cadence == "baseline" else _expected_update_steps(arm, rows)
    )
    applied_version = 0
    for row in rows:
        step = int(row["global_step"])
        if (
            row.get("target_refit_attempted") is not True
            or row.get("target_refit_successful") is not True
        ):
            raise ValueError(f"target refit is not successful at step {step}")
        if row.get("selected_rollout_draft_version") != row.get(
            "applied_draft_version_before_step"
        ):
            raise ValueError(
                f"selected rollout version provenance mismatch at step {step}"
            )
        if row.get("applied_draft_version_before_step") != applied_version:
            raise ValueError(f"applied draft version discontinuity at step {step}")
        requested = step in expected_updates
        if arm.cadence == "always" and row.get("reason") != "always":
            raise ValueError(f"always schedule reason mismatch at step {step}")
        if arm.cadence in {"static", "fixed-5", "fixed-10", "fixed-20"}:
            expected_reason = "fixed_interval" if requested else "none"
            if row.get("reason") != expected_reason:
                raise ValueError(f"fixed schedule reason mismatch at step {step}")
        for field in (
            "update_requested",
            "update_attempted",
            "update_successful",
            "draft_refit_requested",
            "draft_refit_attempted",
            "draft_refit_successful",
        ):
            if row.get(field) is not requested:
                raise ValueError(f"{field} disagrees with schedule at step {step}")
        if requested:
            applied_version = int(row["decision_id"])
        if row.get("applied_draft_version_after_step") != applied_version:
            raise ValueError(
                f"applied draft version publication mismatch at step {step}"
            )
        accepted = row.get("accepted_tokens")
        drafted = row.get("draft_tokens")
        if not isinstance(accepted, (int, float)) or not isinstance(
            drafted, (int, float)
        ):
            raise ValueError(f"acceptance counts are absent at step {step}")
        if accepted < 0 or drafted <= 0 or accepted > drafted:
            raise ValueError(f"acceptance counts are invalid at step {step}")
        if arm.cadence == "adaptive":
            observed = row.get("observed_acceptance")
            if (
                not isinstance(observed, (int, float))
                or not math.isfinite(float(observed))
                or not math.isclose(
                    float(observed), float(accepted) / float(drafted), abs_tol=1e-12
                )
            ):
                raise ValueError(
                    f"observed acceptance is not bound to rollout counts at step {step}"
                )
    for checkpoint_step in CHECKPOINT_STEPS:
        receipt = _read_json(
            root
            / "checkpoints"
            / f"step_{checkpoint_step}"
            / "cadence-checkpoint-receipt.json"
        )
        expected_high_water = 0 if arm.cadence == "baseline" else checkpoint_step
        try:
            _validate_checkpoint_receipt(
                root / "checkpoints" / f"step_{checkpoint_step}",
                receipt,
                expected_step=checkpoint_step,
                expected_high_water=expected_high_water,
                verify_tree=False,
            )
        except ValueError as error:
            raise ValueError(
                f"step_{checkpoint_step} checkpoint receipt is inconsistent"
            ) from error
    expected_count = len(expected_updates)
    reason_names = (
        "always",
        "fixed_interval",
        "none",
        "adaptive_degradation",
        "adaptive_burst",
        "max_interval",
    )
    reason_counts = {
        reason: sum(row.get("reason") == reason for row in rows)
        for reason in reason_names
    }
    required_terminal_counts = {
        "attempted_updates": expected_count,
        "successful_updates": expected_count,
        "attempted_draft_refits": expected_count,
        "successful_draft_refits": expected_count,
        "successful_target_refits": arm.max_steps,
        "decision_count": expected_decision_count,
        "skipped_updates": expected_decision_count - expected_count,
        "forced_updates": sum(row.get("forced") is True for row in rows),
    }
    for field, expected in required_terminal_counts.items():
        if terminal.get(field) != expected:
            raise ValueError(
                f"terminal {field} mismatch: {terminal.get(field)} != {expected}"
            )
    if terminal.get("decision_reason_counts") != reason_counts:
        raise ValueError("terminal decision reason counters disagree with ledger")
    return terminal


def validate_resume_ready(root: Path, arm: Arm, *, product_head: str) -> Path:
    identity = _read_json(root / "run-identity.json")
    if identity.get("arm") != arm.name or identity.get("product_head") != product_head:
        raise ValueError("resume identity does not match arm or product head")
    checkpoints = []
    for path in (root / "checkpoints").glob("step_*"):
        try:
            step = int(path.name.removeprefix("step_"))
        except ValueError:
            continue
        if path.is_dir():
            checkpoints.append((step, path))
    if not checkpoints:
        raise ValueError("resume requires a durable step checkpoint")
    step, latest = max(checkpoints)
    receipt = _read_json(latest / "cadence-checkpoint-receipt.json")
    expected_high_water = 0 if arm.cadence == "baseline" else step
    _validate_checkpoint_receipt(
        latest,
        receipt,
        expected_step=step,
        expected_high_water=expected_high_water,
        verify_tree=True,
    )
    return latest
