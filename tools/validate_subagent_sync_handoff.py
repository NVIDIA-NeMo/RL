#!/usr/bin/env python3
"""Validate an audited, reproducible sync step-20 donor for async training."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pickletools
import re
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from runtime_source_snapshot import verify_snapshot


REQUIRED_AUDIT_CHECKS = {
    "completed_optimizer_steps",
    "finite_training_metrics",
    "reward_learning_signal",
    "optimizer_update_applied",
    "clean_step_admission_metrics",
    "post_admission_role_training",
    "raw_tmpe_bound",
    "no_tmpe_masking",
    "diagnostic_step_alignment",
    "tmpe_diagnostics",
    "diagnostic_aggregate_correlation",
    "learner_evaluated_parent_and_child_roles",
    "rollout_parent_and_child_roles",
    "finite_episode_rewards",
    "episode_integrity",
    "episode_failures",
    "finalized_checkpoint",
    "tmpe_masking_config",
    "no_empty_rejection_poison_warnings",
}

SUBMODULE_PATHS = (
    "3rdparty/Gym-workspace/Gym",
    "3rdparty/Megatron-Bridge-workspace/Megatron-Bridge",
    "3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM",
)

BENCHMARK_CONTRACTS: dict[str, dict[str, Any]] = {
    "agentic_vbench": {
        "config": "examples/configs/recipes/vlm/"
        "vlm_grpo-nemotron-super-omni-120ba12b-16n8g-megatron-"
        "agentic-vbench-sc-256k.v1.yaml",
        "num_prompts": 4,
        "max_turns": 32,
        "context_tokens": 262_144,
        "output_tokens": 8_192,
        "tool_output_max_lines": 400,
        "tool_output_max_bytes": 8_192,
        "max_train_row_tokens": 65_536,
        "task_selection": "assembly",
        "train_limit": 4,
        "train_repeats": 4,
    },
    "claweval": {
        "config": "examples/configs/recipes/vlm/"
        "vlm_grpo-nemotron-super-omni-120ba12b-8n8g-megatron-"
        "claweval-sc.v2.yaml",
        "num_prompts": 8,
        "max_turns": 40,
        "context_tokens": 262_144,
        "output_tokens": 8_192,
        "tool_output_max_lines": 400,
        "tool_output_max_bytes": 8_192,
        "max_train_row_tokens": 65_536,
        "task_selection": "all",
        "train_limit": None,
        "train_repeats": 1,
    },
}


def _fail(message: str) -> None:
    raise ValueError(message)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _json_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        _fail(f"{label} is unreadable: {error}")
    if not isinstance(value, dict):
        _fail(f"{label} must be a JSON object")
    return value


def _nonempty(path: Path, label: str) -> None:
    if not path.is_file() or path.stat().st_size <= 0:
        _fail(f"missing or empty {label}: {path}")


def _training_command_tokens(script: str) -> set[str]:
    candidate_lines = [
        line
        for line in script.splitlines()
        if "grpo.max_num_steps=" in line and "logger.wandb.project=" in line
    ]
    if len(candidate_lines) != 1:
        _fail(
            "archived training command must contain exactly one rendered GRPO "
            f"invocation; found {len(candidate_lines)}"
        )
    try:
        tokens = set(shlex.split(candidate_lines[0]))
    except ValueError as error:
        _fail(f"archived training command is not valid shell syntax: {error}")
    if not {
        "run_grpo",
        "examples/run_grpo_single_controller.py",
    }.intersection(tokens):
        _fail("archived GRPO settings are not part of a training invocation")
    return tokens


def _exact_int(value: Any, expected: int, label: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value != expected:
        _fail(f"{label}={value!r}; expected {expected}")


def _git_commit(repo: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()


def _metadata_references(path: Path) -> list[str]:
    references: set[str] = set()
    saw_stop = False
    try:
        with path.open("rb") as handle:
            for opcode, argument, _position in pickletools.genops(handle):
                if opcode.name == "STOP":
                    saw_stop = True
                if not isinstance(argument, str) or not argument.endswith(".distcp"):
                    continue
                if (
                    re.fullmatch(r"[A-Za-z0-9_.-]+\.distcp", argument) is None
                    or Path(argument).name != argument
                ):
                    _fail(f"unsafe distributed payload reference: {argument!r}")
                references.add(argument)
            if handle.read(1):
                _fail("distributed metadata has trailing bytes after STOP")
    except (OSError, ValueError, TypeError, EOFError, UnicodeDecodeError) as error:
        _fail(f"cannot parse distributed metadata: {error}")
    if not saw_stop:
        _fail("distributed metadata pickle has no STOP opcode")
    if not references:
        _fail("distributed metadata references no .distcp payloads")
    return sorted(references)


def _fingerprint(paths: list[Path], root: Path) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        relative = path.relative_to(root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(path.stat().st_size).encode("ascii"))
        digest.update(b"\0")
        digest.update(bytes.fromhex(_sha256_file(path)))
        digest.update(b"\0")
    return digest.hexdigest()


def _validate_audit(
    audit: dict[str, Any], result_dir: Path, min_steps: int, tmpe_threshold: float
) -> None:
    if audit.get("schema_version") != 2 or audit.get("audit_mode") != "read_only":
        _fail("health audit must use read-only schema version 2")
    if audit.get("overall_status") != "PASS":
        _fail(f"health audit status is {audit.get('overall_status')!r}, not PASS")
    audit_root = Path(str(audit.get("result_directory", ""))).resolve(strict=True)
    if audit_root != result_dir:
        _fail(f"health audit covers {audit_root}, not {result_dir}")
    requirements = audit.get("requirements")
    if not isinstance(requirements, dict):
        _fail("health audit has no requirements object")
    _exact_int(
        requirements.get("minimum_completed_optimizer_steps"),
        min_steps,
        "audit minimum_completed_optimizer_steps",
    )
    threshold = requirements.get("raw_tmpe_upper_bound")
    if (
        not isinstance(threshold, (int, float))
        or isinstance(threshold, bool)
        or not math.isfinite(float(threshold))
        or not math.isclose(float(threshold), tmpe_threshold, abs_tol=1e-12)
    ):
        _fail(f"audit raw_tmpe_upper_bound={threshold!r}; expected {tmpe_threshold}")
    checks = audit.get("checks")
    if not isinstance(checks, list):
        _fail("health audit has no checks list")
    statuses = {
        check.get("name"): check.get("status")
        for check in checks
        if isinstance(check, dict) and isinstance(check.get("name"), str)
    }
    missing = sorted(REQUIRED_AUDIT_CHECKS - statuses.keys())
    failed = sorted(
        name for name in REQUIRED_AUDIT_CHECKS if statuses.get(name) != "PASS"
    )
    if missing:
        _fail(f"health audit omits required checks: {missing}")
    if failed:
        _fail(f"health audit has non-PASS required checks: {failed}")


def _validate_file_identity(
    identity: Any,
    expected_path: Path,
    label: str,
    *,
    require_hash: bool,
) -> None:
    if not isinstance(identity, dict):
        _fail(f"launch provenance has no {label} identity")
    expected_path = expected_path.resolve(strict=True)
    expected_values: dict[str, Any] = {
        "path": str(expected_path),
        "size_bytes": expected_path.stat().st_size,
    }
    if require_hash:
        expected_values["sha256"] = _sha256_file(expected_path)
    else:
        expected_values["mtime_ns"] = expected_path.stat().st_mtime_ns
    mismatches = [
        f"{key}={identity.get(key)!r}; expected {expected!r}"
        for key, expected in expected_values.items()
        if identity.get(key) != expected
    ]
    if mismatches:
        _fail(f"{label} identity mismatch: " + "; ".join(mismatches))


def _validate_provenance(
    provenance: dict[str, Any],
    *,
    result_dir: Path,
    benchmark: str,
    source_commit: str,
    repo_dir: Path,
    expected_model: Path,
    expected_wandb_project: str,
    expected_train_data: Path,
    expected_validation_data: Path,
    expected_container: Path,
    expected_delegation_prompt_sha256: str,
    expected_subagent_prompt_sha256: str,
    expected_generation_tensor_parallel_size: int,
) -> None:
    if provenance.get("schema_version") != 2:
        _fail("launch provenance must use schema version 2")
    contract = BENCHMARK_CONTRACTS[benchmark]
    expected_values: dict[str, Any] = {
        "benchmark": benchmark,
        "source_commit": source_commit,
        "result_directory": str(result_dir),
        "checkpoint_directory": str(result_dir / "checkpoints"),
        "model_path": str(expected_model),
        "config": contract["config"],
        "wandb_project": expected_wandb_project,
        "wandb_cluster_name": "nsc-svg-slurm-1",
        "num_nodes": 8,
        "num_generation_nodes": 4,
        "gpus_per_node": 8,
        "segment_size": 4,
        "num_prompts": contract["num_prompts"],
        "num_generations": 4,
        "generation_tensor_parallel_size": (expected_generation_tensor_parallel_size),
        "max_num_steps": 1_000_000,
        "max_num_epochs": None,
        "save_period": 20,
        "keep_top_k": None,
        "context_compaction": True,
        "opencode_compaction_auto": False,
        "opencode_compaction_prune": False,
        "subagents_enabled": True,
        "training_mode": "subagent",
        "immutable_source_snapshot": True,
        "max_turns": contract["max_turns"],
        "context_tokens": contract["context_tokens"],
        "output_tokens": contract["output_tokens"],
        "tool_output_max_lines": contract["tool_output_max_lines"],
        "tool_output_max_bytes": contract["tool_output_max_bytes"],
        "max_train_row_tokens": contract["max_train_row_tokens"],
        "task_selection": contract["task_selection"],
        "train_limit": contract["train_limit"],
        "train_repeats": contract["train_repeats"],
        "delegation_prompt_sha256": expected_delegation_prompt_sha256,
        "subagent_prompt_sha256": expected_subagent_prompt_sha256,
    }
    mismatches = [
        f"{key}={provenance.get(key)!r}; expected {expected!r}"
        for key, expected in expected_values.items()
        if provenance.get(key) != expected
    ]
    if mismatches:
        _fail("launch provenance mismatch: " + "; ".join(mismatches))

    expected_submodules = {
        relative: _git_commit(repo_dir / relative) for relative in SUBMODULE_PATHS
    }
    commit_values = {
        "gym_commit": expected_submodules[SUBMODULE_PATHS[0]],
        "bridge_commit": expected_submodules[SUBMODULE_PATHS[1]],
        "mcore_commit": expected_submodules[SUBMODULE_PATHS[2]],
    }
    commit_mismatches = [
        f"{key}={provenance.get(key)!r}; expected {expected!r}"
        for key, expected in commit_values.items()
        if provenance.get(key) != expected
    ]
    if commit_mismatches:
        _fail("submodule provenance mismatch: " + "; ".join(commit_mismatches))

    snapshot_root = Path(str(provenance.get("runtime_snapshot_directory", "")))
    manifest_path = Path(str(provenance.get("runtime_source_manifest", "")))
    expected_snapshot_root = result_dir / "runtime_source"
    expected_manifest_path = result_dir / "runtime_source_manifest.json"
    if snapshot_root.resolve(strict=True) != expected_snapshot_root.resolve(
        strict=True
    ):
        _fail(
            "runtime snapshot is not the donor's result-local immutable snapshot: "
            f"{snapshot_root}"
        )
    if manifest_path.resolve(strict=True) != expected_manifest_path.resolve(
        strict=True
    ):
        _fail(
            "runtime source manifest is not the donor's result-local manifest: "
            f"{manifest_path}"
        )
    snapshot_report = verify_snapshot(snapshot_root, manifest_path)
    if snapshot_report["runtime_manifest_sha256"] != provenance.get(
        "runtime_manifest_sha256"
    ):
        _fail("runtime snapshot digest differs from launch provenance")
    snapshot_manifest = _json_object(manifest_path, "runtime source manifest")
    if snapshot_manifest.get("source_commit") != source_commit:
        _fail("runtime source manifest has a different superproject commit")
    if snapshot_manifest.get("submodule_commits") != expected_submodules:
        _fail("runtime source manifest has different submodule commits")

    config_path = snapshot_root / str(contract["config"])
    _validate_file_identity(
        provenance.get("config_identity"), config_path, "config", require_hash=True
    )
    current_config = repo_dir / str(contract["config"])
    if _sha256_file(config_path) != _sha256_file(current_config):
        _fail("snapshot recipe differs from the current expected recipe")
    _validate_file_identity(
        provenance.get("train_data"),
        expected_train_data,
        "training data",
        require_hash=True,
    )
    _validate_file_identity(
        provenance.get("validation_data"),
        expected_validation_data,
        "validation data",
        require_hash=True,
    )
    _validate_file_identity(
        provenance.get("container"),
        expected_container,
        "container",
        require_hash=False,
    )


def validate(args: argparse.Namespace) -> dict[str, Any]:
    if args.expected_generation_tensor_parallel_size <= 0:
        _fail("expected generation tensor-parallel size must be positive")
    result_dir = args.result_dir.resolve(strict=True)
    weights_dir = args.weights_dir.resolve(strict=True)
    expected_weights = result_dir / "checkpoints" / "step_20" / "policy" / "weights"
    if weights_dir != expected_weights:
        _fail(f"donor is {weights_dir}, expected exact {expected_weights}")

    source_commit = args.expected_source_commit.lower()
    if re.fullmatch(r"[0-9a-f]{40}", source_commit) is None:
        _fail("expected source commit must be a full 40-character SHA")
    repo_dir = args.expected_repo_dir.resolve(strict=True)
    if _git_commit(repo_dir) != source_commit:
        _fail("expected source commit is not the current repository HEAD")
    expected_model = args.expected_model.resolve(strict=True)
    audit_path = args.audit_json.resolve(strict=True)

    audit = _json_object(audit_path, "health audit")
    _validate_audit(audit, result_dir, args.min_steps, args.tmpe_threshold)
    provenance_path = result_dir / "launch_provenance.json"
    provenance = _json_object(provenance_path, "launch provenance")
    _validate_provenance(
        provenance,
        result_dir=result_dir,
        benchmark=args.benchmark,
        source_commit=source_commit,
        repo_dir=repo_dir,
        expected_model=expected_model,
        expected_wandb_project=args.expected_wandb_project,
        expected_train_data=args.expected_train_data,
        expected_validation_data=args.expected_validation_data,
        expected_container=args.expected_container,
        expected_delegation_prompt_sha256=args.expected_delegation_prompt_sha256,
        expected_subagent_prompt_sha256=args.expected_subagent_prompt_sha256,
        expected_generation_tensor_parallel_size=(
            args.expected_generation_tensor_parallel_size
        ),
    )
    training_command_path = result_dir / "training_command.sh"
    _nonempty(training_command_path, "archived training command")
    training_command = training_command_path.read_text(encoding="utf-8")
    training_tokens = _training_command_tokens(training_command)
    for required_setting in (
        "grpo.max_num_steps=1000000",
        "grpo.max_num_epochs=null",
        f"logger.wandb.project={args.expected_wandb_project}",
        "checkpointing.save_period=20",
        "checkpointing.keep_top_k=null",
        "token_capture.context_compaction=true",
        "policy.generation.vllm_cfg.tensor_parallel_size="
        f"{args.expected_generation_tensor_parallel_size}",
    ):
        if required_setting not in training_tokens:
            _fail(f"archived training command omits {required_setting!r}")

    status = _json_object(
        result_dir / "checkpoints" / "latest_checkpoint_status.json",
        "latest checkpoint status",
    )
    latest_step = status.get("last_checkpoint_step")
    if (
        not isinstance(latest_step, int)
        or isinstance(latest_step, bool)
        or latest_step < 20
    ):
        _fail(f"latest finalized checkpoint step is {latest_step!r}; need at least 20")

    step_dir = result_dir / "checkpoints" / "step_20"
    training_info_path = step_dir / "training_info.json"
    training_info = _json_object(training_info_path, "step-20 training info")
    for key in ("current_step", "total_steps", "trainer_version"):
        _exact_int(training_info.get(key), 20, f"step-20 training_info.{key}")
    config_path = step_dir / "config.yaml"
    _nonempty(config_path, "step-20 config")

    tracker_path = weights_dir / "latest_checkpointed_iteration.txt"
    latest_train_state = weights_dir / "latest_train_state.pt"
    _nonempty(tracker_path, "policy iteration tracker")
    _nonempty(latest_train_state, "policy latest train state")
    marker = tracker_path.read_text(encoding="utf-8").strip()
    if marker == "release":
        iteration_dir = weights_dir / "release"
    elif re.fullmatch(r"\d+", marker):
        iteration_dir = weights_dir / f"iter_{int(marker):07d}"
    else:
        _fail(f"invalid policy iteration marker: {marker!r}")

    metadata_path = iteration_dir / ".metadata"
    policy_metadata_path = iteration_dir / "metadata.json"
    run_config_path = iteration_dir / "run_config.yaml"
    train_state_path = iteration_dir / "train_state.pt"
    for path, label in (
        (metadata_path, "distributed metadata"),
        (policy_metadata_path, "policy metadata"),
        (run_config_path, "policy run config"),
        (train_state_path, "policy train state"),
    ):
        _nonempty(path, label)
    policy_metadata = _json_object(policy_metadata_path, "policy metadata")
    if policy_metadata.get("sharded_backend") != "torch_dist":
        _fail("policy metadata sharded_backend must be torch_dist")
    if policy_metadata.get("sharded_backend_version") != 1:
        _fail("policy metadata sharded_backend_version must be 1")

    shard_paths = [iteration_dir / name for name in _metadata_references(metadata_path)]
    dcp_payloads: list[dict[str, Any]] = []
    for shard in shard_paths:
        _nonempty(shard, "distributed checkpoint payload")
        dcp_payloads.append(
            {
                "path": shard.relative_to(weights_dir).as_posix(),
                "size_bytes": shard.stat().st_size,
                "sha256": _sha256_file(shard),
            }
        )
    dcp_manifest_files = [
        {
            "path": ".metadata",
            "size_bytes": metadata_path.stat().st_size,
            "sha256": _sha256_file(metadata_path),
        },
        *[
            {
                "path": shard.name,
                "size_bytes": payload["size_bytes"],
                "sha256": payload["sha256"],
            }
            for shard, payload in zip(shard_paths, dcp_payloads, strict=True)
        ],
    ]
    canonical_dcp_manifest = json.dumps(
        dcp_manifest_files, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    dcp_manifest_sha256 = hashlib.sha256(canonical_dcp_manifest).hexdigest()
    fingerprint_paths = [
        provenance_path,
        training_command_path,
        training_info_path,
        config_path,
        tracker_path,
        latest_train_state,
        metadata_path,
        policy_metadata_path,
        run_config_path,
        train_state_path,
    ]
    control_fingerprint = _fingerprint(fingerprint_paths, result_dir)
    checkpoint_fingerprint = hashlib.sha256(
        f"{control_fingerprint}\0{dcp_manifest_sha256}".encode("ascii")
    ).hexdigest()
    return {
        "schema_version": 2,
        "status": "PASS",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "benchmark": args.benchmark,
        "source_commit": source_commit,
        "result_directory": str(result_dir),
        "policy_weights": str(weights_dir),
        "audited_minimum_steps": args.min_steps,
        "audited_tmpe_upper_bound": args.tmpe_threshold,
        "health_auditor_sha256": args.health_auditor_sha256,
        "latest_finalized_checkpoint_step": latest_step,
        "donor_step": 20,
        "dcp_payload_count": len(shard_paths),
        "dcp_payloads": dcp_payloads,
        "dcp_sha256_manifest": {
            "files": dcp_manifest_files,
            "manifest_sha256": dcp_manifest_sha256,
        },
        "checkpoint_fingerprint_sha256": checkpoint_fingerprint,
        "health_audit_sha256": hashlib.sha256(audit_path.read_bytes()).hexdigest(),
        "runtime_manifest_sha256": provenance["runtime_manifest_sha256"],
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--weights-dir", type=Path, required=True)
    parser.add_argument("--audit-json", type=Path, required=True)
    parser.add_argument(
        "--benchmark", choices=tuple(BENCHMARK_CONTRACTS), required=True
    )
    parser.add_argument("--expected-source-commit", required=True)
    parser.add_argument("--expected-repo-dir", type=Path, required=True)
    parser.add_argument("--expected-model", type=Path, required=True)
    parser.add_argument("--expected-train-data", type=Path, required=True)
    parser.add_argument("--expected-validation-data", type=Path, required=True)
    parser.add_argument("--expected-container", type=Path, required=True)
    parser.add_argument("--expected-delegation-prompt-sha256", required=True)
    parser.add_argument("--expected-subagent-prompt-sha256", required=True)
    parser.add_argument(
        "--expected-generation-tensor-parallel-size", type=int, required=True
    )
    parser.add_argument("--health-auditor-sha256", required=True)
    parser.add_argument(
        "--expected-wandb-project",
        default="nemorl-v2-subagent-comapction-debug",
    )
    parser.add_argument("--min-steps", type=int, default=20)
    parser.add_argument("--tmpe-threshold", type=float, default=1.03)
    return parser.parse_args()


def main() -> int:
    try:
        report = validate(_parse_args())
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"sync-to-async handoff validation failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
