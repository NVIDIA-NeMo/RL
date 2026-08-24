#!/usr/bin/env python3
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

"""Apply the scheduler test-only and exactly-once submission gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from benchmark import (
    ContractError,
    _validate_preflight_record,
    _write_exclusive_json,
    load_manifest,
    monitor_path,
    record_job,
    require_successful_canary,
    reserve_submission,
    validate_artifact_metadata,
    validate_canonical_plan,
    validate_output_root,
    validate_plan_runtime_files,
    verify_source,
)

Runner = Callable[..., subprocess.CompletedProcess[str]]
Sleeper = Callable[[float], object]
PreflightRevalidator = Callable[..., None]
OutputRootValidator = Callable[..., None]


def revalidate_preflight_state(
    *, plan: dict[str, Any], preflight_record: dict[str, Any], repo_root: Path
) -> None:
    """Revalidate fast source and canary state immediately before Slurm calls."""
    manifest = load_manifest()
    protected_paths_by_head = {
        manifest["pr_head"]: [
            Path(manifest["recipe"]),
            Path("examples/nemo_gym/grpo_qwen3_30ba3b_thinking_swe1.yaml"),
            Path("examples/nemo_gym/run_qwen3_swe_rollout_only.sh"),
            Path(manifest["entrypoint"]),
        ]
    }
    verify_source(
        repo_root=repo_root,
        source_commit=plan["source_commit"],
        required_ancestors=[manifest["pr_head"]],
        protected_paths_by_head=protected_paths_by_head,
        source_files_sha256={
            Path(path): digest
            for path, digest in manifest["source_files_sha256"].items()
        },
    )
    validate_artifact_metadata(artifacts=preflight_record["artifacts"])
    validate_plan_runtime_files(plan=plan)


def validate_scheduler_output_root(*, plan: dict[str, Any], repo_root: Path) -> None:
    """Bind scheduler writes to the user's approved Lustre experiment tree."""
    manifest = load_manifest()
    validate_output_root(
        output_root=Path(plan["output_root"]),
        repo_root=repo_root,
        approved_prefix=Path(manifest["output_root_prefix"]),
    )


def _run_for_arm(plan: dict[str, Any], arm_name: str) -> dict[str, Any]:
    matches = [item for item in plan["runs"] if item["arm"] == arm_name]
    if len(matches) != 1:
        raise ContractError(f"plan must contain exactly one run for arm {arm_name}")
    return matches[0]


def _completion_wrapped_command(
    *, plan: dict[str, Any], run: dict[str, Any], repo_root: Path, state_dir: Path
) -> str:
    benchmark_cli = (
        repo_root / "experiments/qwen3_30ba3b_swe_rollout_pr3733/benchmark.py"
    )
    completion = [
        "python3",
        str(benchmark_cli),
        "complete",
        "--state-dir",
        str(state_dir),
        "--campaign-id",
        plan["campaign_id"],
        "--profile",
        plan["profile"],
        "--arm",
        run["arm"],
        "--job-id",
        "${SLURM_JOB_ID}",
        "--exit-code",
        "${run_rc}",
    ]
    completion_shell = (
        shlex.join(completion)
        .replace("'${SLURM_JOB_ID}'", '"${SLURM_JOB_ID}"')
        .replace("'${run_rc}'", '"${run_rc}"')
    )
    command_shell = shlex.join(run["command"])
    exports = " ".join(
        f"{key}={shlex.quote(value)}" for key, value in run["environment"].items()
    )
    submission_record = state_dir / (
        f"{plan['campaign_id']}__{plan['profile']}__{run['arm']}.submission.json"
    )
    return (
        "set -euo pipefail; "
        f"cd {shlex.quote(str(repo_root))}; "
        f"export {exports}; "
        "run_rc=0; "
        f"{command_shell} || run_rc=$?; "
        "attempt=0; "
        f"while [[ ! -f {shlex.quote(str(submission_record))} && $attempt -lt 60 ]]; do "
        "attempt=$((attempt + 1)); sleep 1; done; "
        f"{completion_shell}; "
        'exit "${run_rc}"'
    )


def build_scheduler_contract(
    *,
    plan: dict[str, Any],
    run: dict[str, Any],
    repo_root: Path,
    account: str,
    partition: str,
    time_limit: str,
    state_dir: Path | None = None,
) -> dict[str, Any]:
    """Build the exact two-node PR #3733 trajectory-collection contract."""
    if not repo_root.is_absolute() or not Path(plan["container"]["path"]).is_absolute():
        raise ContractError("source and container paths must be absolute")
    common = plan["common"]
    if (
        common["num_nodes"] != 2
        or common["generation_num_nodes"] != 1
        or common["gpus_per_node"] != 4
    ):
        raise ContractError("scheduler topology drift from the authoritative recipe")
    if not account or not partition:
        raise ContractError("Slurm account and partition are required")
    run_output = Path(run["output_dir"])
    if state_dir is None:
        state_dir = run_output.parents[1] / "state"
    command = _completion_wrapped_command(
        plan=plan, run=run, repo_root=repo_root, state_dir=state_dir
    )
    environment = {
        "BASE_LOG_DIR": str(run_output / "ray"),
        "COMMAND": command,
        "CONTAINER": plan["container"]["path"],
        "GPUS_PER_NODE": str(common["gpus_per_node"]),
        "MOUNTS": f"/lustre:/lustre,{repo_root}:{repo_root}",
        "NEMO_RL_PY_EXECUTABLES_SYSTEM": "1",
        "NRL_FORCE_REBUILD_VENVS": "true",
        "WANDB_ENTITY": common["wandb_entity"],
        "WANDB_PROJECT": common["wandb_project"],
    }
    sbatch_args = [
        "sbatch",
        "--export=ALL",
        f"--nodes={common['num_nodes']}",
        "--segment=1",
        f"--account={account}",
        f"--job-name={account}.{run['environment']['WANDB_RUN_NAME']}",
        f"--partition={partition}",
        f"--time={time_limit}",
        f"--gres=gpu:{common['gpus_per_node']}",
        "--exclusive",
        "--mem=0",
        f"--output={run_output / 'slurm-%j.out'}",
        "--parsable",
        "ray.sub",
    ]
    fingerprint_payload = {
        "sbatch_args": sbatch_args,
        "environment": environment,
        "plan_source_commit": plan["source_commit"],
        "plan_container_sha256": plan["container"]["sha256"],
    }
    fingerprint = hashlib.sha256(
        json.dumps(fingerprint_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "sbatch_args": sbatch_args,
        "environment": environment,
        "fingerprint": fingerprint,
    }


def _test_only_path(
    state_dir: Path, campaign_id: str, profile: str, arm_name: str
) -> Path:
    return state_dir / f"{campaign_id}__{profile}__{arm_name}.test_only.json"


def run_scheduler_action(
    *,
    mode: str,
    plan: dict[str, Any],
    run: dict[str, Any],
    repo_root: Path,
    state_dir: Path,
    account: str,
    partition: str,
    time_limit: str,
    preflight_record: dict[str, Any],
    preflight_revalidator: PreflightRevalidator = revalidate_preflight_state,
    output_root_validator: OutputRootValidator = validate_scheduler_output_root,
    runner: Runner = subprocess.run,
) -> dict[str, Any]:
    """Run scheduler validation or submit once after an identical validation."""
    if mode not in {"test-only", "submit"}:
        raise ContractError(f"unsupported scheduler mode: {mode}")
    profile = plan["profile"]
    campaign_id = plan["campaign_id"]
    arm_name = run["arm"]
    manifest = load_manifest()
    _validate_preflight_record(
        preflight=preflight_record,
        manifest=manifest,
        source_commit=plan["source_commit"],
        container=Path(plan["container"]["path"]),
        container_sha256=plan["container"]["sha256"],
    )
    if plan.get("preflight_id") != preflight_record.get("preflight_id"):
        raise ContractError("plan is not bound to the verified preflight")
    validate_canonical_plan(plan=plan, preflight=preflight_record)
    canonical_run = _run_for_arm(plan, arm_name)
    if run != canonical_run:
        raise ContractError(f"run is not canonical for arm {arm_name}")
    output_root_validator(plan=plan, repo_root=repo_root)
    expected_state_dir = Path(plan["output_root"]).resolve(strict=False) / "state"
    if state_dir.resolve(strict=False) != expected_state_dir:
        raise ContractError(
            f"state directory is not the canonical state directory: {state_dir}"
        )
    preflight_revalidator(
        plan=plan, preflight_record=preflight_record, repo_root=repo_root
    )
    if profile == "full":
        require_successful_canary(state_dir=state_dir, campaign_id=campaign_id)
    contract = build_scheduler_contract(
        plan=plan,
        run=run,
        repo_root=repo_root,
        account=account,
        partition=partition,
        time_limit=time_limit,
        state_dir=state_dir,
    )
    test_only_path = _test_only_path(state_dir, campaign_id, profile, arm_name)
    Path(run["output_dir"]).mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.update(contract["environment"])

    if mode == "test-only":
        command = [
            *contract["sbatch_args"][:-1],
            "--test-only",
            contract["sbatch_args"][-1],
        ]
        result = runner(
            command,
            cwd=repo_root,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            raise ContractError(f"sbatch --test-only failed: {result.stderr.strip()}")
        record = {
            "status": "test-only-passed",
            "campaign_id": campaign_id,
            "profile": profile,
            "arm": arm_name,
            "fingerprint": contract["fingerprint"],
        }
        if test_only_path.exists():
            observed = json.loads(test_only_path.read_text(encoding="utf-8"))
            if observed != record:
                raise ContractError(
                    f"test-only contract drift for {profile}/{arm_name}"
                )
        else:
            _write_exclusive_json(test_only_path, record)
        return record

    if not test_only_path.is_file():
        raise ContractError(
            f"missing matching successful sbatch --test-only for {profile}/{arm_name}"
        )
    tested = json.loads(test_only_path.read_text(encoding="utf-8"))
    if tested.get("fingerprint") != contract["fingerprint"]:
        raise ContractError(
            f"missing matching successful sbatch --test-only for {profile}/{arm_name}"
        )
    reservation = reserve_submission(
        state_dir=state_dir,
        campaign_id=campaign_id,
        profile=profile,
        arm_name=arm_name,
    )
    result = runner(
        contract["sbatch_args"],
        cwd=repo_root,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise ContractError(
            f"sbatch failed after durable reservation; operator review required: "
            f"{result.stderr.strip()}"
        )
    job_id = result.stdout.strip().split(";", maxsplit=1)[0]
    submitted = record_job(
        state_dir=state_dir,
        campaign_id=campaign_id,
        profile=profile,
        arm_name=arm_name,
        reservation_id=reservation["reservation_id"],
        job_id=job_id,
    )
    return submitted


def monitor_jobs(
    *,
    job_ids: list[str],
    state_dir: Path,
    campaign_id: str,
    profile: str,
    passes: int = 6,
    interval_seconds: int = 60,
    runner: Runner = subprocess.run,
    sleeper: Sleeper = time.sleep,
) -> dict[str, Any]:
    """Monitor filtered job IDs for at least five minutes at a safe query rate."""
    if not job_ids or any(not job_id.isdigit() for job_id in job_ids):
        raise ContractError("monitor job IDs must be decimal Slurm IDs")
    if len(set(job_ids)) != len(job_ids):
        raise ContractError("monitor job IDs must be unique")
    record_path = monitor_path(
        state_dir=state_dir, campaign_id=campaign_id, profile=profile
    )
    if passes < 6 or interval_seconds < 60:
        raise ContractError("monitor window must span at least five minutes")
    observations: list[dict[str, Any]] = []
    command = [
        "squeue",
        "--jobs",
        ",".join(job_ids),
        "--noheader",
        "--format=%i|%T|%M|%R",
    ]
    for pass_index in range(passes):
        result = runner(command, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise ContractError(
                f"filtered squeue query failed: {result.stderr.strip()}"
            )
        observations.append(
            {
                "pass": pass_index + 1,
                "elapsed_seconds": pass_index * interval_seconds,
                "squeue": result.stdout.splitlines(),
            }
        )
        if pass_index + 1 < passes:
            sleeper(interval_seconds)
    record = {
        "status": "monitored",
        "campaign_id": campaign_id,
        "profile": profile,
        "job_ids": job_ids,
        "passes": passes,
        "interval_seconds": interval_seconds,
        "monitor_window_seconds": (passes - 1) * interval_seconds,
        "observations": observations,
    }
    _write_exclusive_json(record_path, record)
    return record


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("test-only", "submit", "monitor"))
    parser.add_argument("--plan", type=Path)
    parser.add_argument("--arm")
    parser.add_argument("--repo-root", type=Path)
    parser.add_argument("--state-dir", type=Path)
    parser.add_argument("--campaign-id")
    parser.add_argument("--profile", choices=("canary", "full"))
    parser.add_argument("--account")
    parser.add_argument("--partition", default="batch")
    parser.add_argument("--time", default="04:00:00")
    parser.add_argument("--job-id", action="append", default=[])
    parser.add_argument("--preflight-record", type=Path)
    return parser


def main() -> int:
    """Apply one explicit scheduler lifecycle transition."""
    args = _build_parser().parse_args()
    try:
        if args.mode == "monitor":
            if (
                args.state_dir is None
                or args.campaign_id is None
                or args.profile is None
            ):
                raise ContractError(
                    "monitor requires --state-dir, --campaign-id, and --profile"
                )
            result = monitor_jobs(
                job_ids=args.job_id,
                state_dir=args.state_dir,
                campaign_id=args.campaign_id,
                profile=args.profile,
            )
            print(json.dumps(result, indent=2, sort_keys=True))
            return 0
        required = {
            "--plan": args.plan,
            "--arm": args.arm,
            "--repo-root": args.repo_root,
            "--state-dir": args.state_dir,
            "--account": args.account,
            "--preflight-record": args.preflight_record,
        }
        missing = [name for name, value in required.items() if value is None]
        if missing:
            raise ContractError(f"{args.mode} requires: {', '.join(missing)}")
        plan = json.loads(args.plan.read_text(encoding="utf-8"))
        preflight_record = json.loads(args.preflight_record.read_text(encoding="utf-8"))
        run = _run_for_arm(plan, args.arm)
        result = run_scheduler_action(
            mode=args.mode,
            plan=plan,
            run=run,
            repo_root=args.repo_root,
            state_dir=args.state_dir,
            account=args.account,
            partition=args.partition,
            time_limit=args.time,
            preflight_record=preflight_record,
        )
    except (ContractError, OSError, json.JSONDecodeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
