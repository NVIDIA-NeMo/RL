from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess

from research.qwen3_8b_draft_cadence_200step.matrix import (
    CHECKPOINT_STEPS,
    CONTAINER,
    CONTAINER_SHA256,
    WINDOW,
    Arm,
    build_arms,
    render_hydra_overrides,
)
from research.qwen3_8b_draft_cadence_200step.receipts import (
    adapt_native_outputs,
    validate_arm_receipts,
    validate_resume_ready,
)


@dataclass(frozen=True, slots=True)
class Submission:
    argv: tuple[str, ...]
    environment: dict[str, str]
    arm: Arm
    remote_repo: Path
    expected_product_head: str


def _arm(name: str) -> Arm:
    try:
        return next(arm for arm in build_arms() if arm.name == name)
    except StopIteration as error:
        raise ValueError(f"unknown arm: {name}") from error


def _canonical_sha256(payload: object) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


def validate_checkpoint_paths(arm: Arm, *, target: Path, drafter: Path | None) -> None:
    if not target.is_dir() or target.name != arm.target_revision:
        raise ValueError(
            f"target revision path is absent or wrong: need {arm.target_revision}"
        )
    if not (target / "config.json").is_file():
        raise ValueError(f"checkpoint config.json is absent: {target}")
    if arm.drafter == "none":
        if drafter is not None:
            raise ValueError("baseline must not receive a drafter checkpoint")
        return
    if drafter is None or not drafter.is_dir() or drafter.name != arm.drafter_revision:
        raise ValueError(
            f"drafter revision path is absent or wrong: need {arm.drafter_revision}"
        )
    if not (drafter / "config.json").is_file():
        raise ValueError(f"checkpoint config.json is absent: {drafter}")


def validate_container(image: Path, *, expected_sha256: str = CONTAINER_SHA256) -> None:
    if not image.is_file():
        raise ValueError(f"container image is absent: {image}")
    metadata = Path(str(image) + ".metadata.txt")
    if not metadata.is_file() or f"sha256={expected_sha256}" not in {
        line.strip() for line in metadata.read_text().splitlines()
    }:
        raise ValueError("container digest does not match the pinned identity")
    digest = hashlib.sha256()
    with image.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    if digest.hexdigest() != expected_sha256:
        raise ValueError("container image bytes do not match the pinned digest")


def materialize_manifest(
    *, result_root: Path, product_head: str, harness_head: str
) -> Path:
    if len(product_head) != 40 or len(harness_head) != 40:
        raise ValueError("product and harness heads must be full 40-character SHAs")
    arms = build_arms()
    payload: dict[str, object] = {
        "schema_version": 1,
        "product_head": product_head,
        "harness_head": harness_head,
        "analysis_window": list(WINDOW),
        "required_checkpoint_steps": list(CHECKPOINT_STEPS),
        "container": CONTAINER,
        "container_sha256": CONTAINER_SHA256,
        "arms": [
            {
                **asdict(arm),
                "wandb_name": arm.wandb_name,
                "result_dir": str((result_root / arm.name).resolve()),
                "hydra_overrides": list(
                    render_hydra_overrides(
                        arm, result_dir=str((result_root / arm.name).resolve())
                    )
                ),
            }
            for arm in arms
        ],
    }
    payload["manifest_sha256"] = _canonical_sha256(payload)
    result_root.mkdir(parents=True, exist_ok=True)
    path = result_root / "manifest.json"
    with path.open("x") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
    (result_root / "scheduler-logs").mkdir(exist_ok=True)
    return path


def initialize_run_identity(
    *,
    result_dir: Path,
    arm: str,
    product_head: str,
    wandb_run_id: str,
    slurm_job_id: str | None,
) -> Path:
    _arm(arm)
    if len(product_head) != 40:
        raise ValueError("product head must be a full SHA")
    if not wandb_run_id:
        raise ValueError("W&B run ID must not be empty")
    result_dir.mkdir(exist_ok=True)
    identity = result_dir / "run-identity.json"
    payload = {
        "schema_version": 1,
        "arm": arm,
        "product_head": product_head,
        "wandb_run_id": wandb_run_id,
        "slurm_job_id": slurm_job_id,
    }
    with identity.open("x") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
    return identity


def build_submission(
    arm: Arm,
    *,
    remote_repo: Path,
    expected_product_head: str,
    result_root: Path,
    account: str,
    test_only: bool = True,
) -> Submission:
    if not str(remote_repo).startswith("/home/"):
        raise ValueError("source repository must live under /home on MARS clusters")
    if not str(result_root).startswith("/lustre/"):
        raise ValueError("durable experiment results must live under /lustre")
    if len(expected_product_head) != 40:
        raise ValueError("expected product head must be a full SHA")
    arm_result = result_root / arm.name
    run_command = shlex.join(
        (
            "bash",
            "research/qwen3_8b_draft_cadence_200step/run_arm.sh",
            "--arm",
            arm.name,
            "--result-dir",
            str(arm_result),
            "--expected-product-head",
            expected_product_head,
        )
    )
    argv = [
        "sbatch",
        "--nodes=1",
        f"--account={account}",
        f"--job-name={account}.q8c200-{arm.name}",
        "--partition=batch",
        "--time=04:00:00",
        "--gres=gpu:4",
        "--segment=16",
        f"--chdir={remote_repo}",
        f"--output={result_root}/scheduler-logs/q8c200-{arm.name}-%j.out",
    ]
    if test_only:
        argv.append("--test-only")
    argv.append(str(remote_repo / "ray.sub"))
    environment = {
        "COMMAND": run_command,
        "CONTAINER": CONTAINER,
        "HF_HOME": f"{Path(arm.target_snapshot).parents[3]}",
        "HF_DATASETS_CACHE": f"{Path(arm.target_snapshot).parents[3]}/cache",
        "MOUNTS": "/lustre:/lustre",
        "RAY_TMPDIR": "/tmp",
        "TMPDIR": "/tmp",
        "GPUS_PER_NODE": str(arm.gpus_per_node),
        "WANDB_PROJECT": arm.wandb_project,
        "WANDB_RUN_ID": f"q8c200-{arm.name}-{expected_product_head[:8]}",
        "WANDB_RESUME": "allow",
        "NRL_FORCE_REBUILD_VENVS": "true",
    }
    return Submission(tuple(argv), environment, arm, remote_repo, expected_product_head)


def validate_runtime_source_root(source_root: Path) -> None:
    location = str(source_root)
    if not source_root.is_absolute() or not location.startswith(
        ("/home/", "/raid/scratch/")
    ):
        raise ValueError(
            "source repository must live under /home or task-owned /raid/scratch"
        )


def validate_source(source_root: Path, expected_head: str) -> None:
    validate_runtime_source_root(source_root)
    head = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=source_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if head != expected_head:
        raise RuntimeError(f"product head mismatch: {head} != {expected_head}")
    porcelain = subprocess.run(
        ("git", "status", "--porcelain=v1", "--untracked-files=all"),
        cwd=source_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if porcelain:
        raise RuntimeError("product source is not recursively clean")
    submodules = subprocess.run(
        ("git", "submodule", "status", "--recursive"),
        cwd=source_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    if any(not line.startswith(" ") for line in submodules):
        raise RuntimeError("product submodules are missing, divergent, or conflicted")
    for line in submodules:
        submodule_path = line.split()[1]
        submodule_porcelain = subprocess.run(
            ("git", "status", "--porcelain=v1", "--untracked-files=all"),
            cwd=source_root / submodule_path,
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        if submodule_porcelain:
            raise RuntimeError(f"product submodule is dirty: {submodule_path}")
    subprocess.run(("git", "verify-commit", "HEAD"), cwd=source_root, check=True)
    _arm("dflash-adaptive").validate_product_source(source_root)


def run_submission(submission: Submission) -> str:
    validate_source(submission.remote_repo, submission.expected_product_head)
    validate_container(Path(CONTAINER))
    validate_checkpoint_paths(
        submission.arm,
        target=Path(submission.arm.target_snapshot),
        drafter=(
            None
            if submission.arm.drafter_snapshot is None
            else Path(submission.arm.drafter_snapshot)
        ),
    )
    environment = os.environ.copy()
    environment.update(submission.environment)
    result = subprocess.run(
        submission.argv,
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    return result.stdout.strip()


def _print_overrides(arm: Arm, result_dir: str) -> None:
    for override in render_hydra_overrides(arm, result_dir=result_dir):
        print(override)


def validate_all_config_compositions(result_root: Path) -> None:
    from omegaconf import OmegaConf

    from nemo_rl.algorithms.grpo import MasterConfig
    from nemo_rl.utils.config import (
        load_config,
        parse_hydra_overrides,
        register_omegaconf_resolvers,
    )

    register_omegaconf_resolvers()
    for arm in build_arms():
        config = load_config(arm.config_path)
        config = parse_hydra_overrides(
            config,
            list(render_hydra_overrides(arm, result_dir=str(result_root / arm.name))),
        )
        resolved = OmegaConf.to_container(config, resolve=True)
        if not isinstance(resolved, dict):
            raise TypeError(f"composed config for {arm.name} is not a mapping")
        MasterConfig(**resolved)
        print(f"CONFIG_COMPOSE_PASS {arm.name}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    overrides = subparsers.add_parser("overrides")
    overrides.add_argument("--arm", required=True)
    overrides.add_argument("--result-dir", required=True)
    preflight = subparsers.add_parser("preflight")
    preflight.add_argument("--arm", required=True)
    preflight.add_argument("--source-root", type=Path, required=True)
    preflight.add_argument("--expected-product-head", required=True)
    compose = subparsers.add_parser("compose-preflight")
    compose.add_argument("--result-root", type=Path, required=True)
    resume = subparsers.add_parser("resume-preflight")
    resume.add_argument("--arm", required=True)
    resume.add_argument("--result-dir", type=Path, required=True)
    resume.add_argument("--expected-product-head", required=True)
    terminal = subparsers.add_parser("terminal-preflight")
    terminal.add_argument("--arm", required=True)
    terminal.add_argument("--result-dir", type=Path, required=True)
    adapt = subparsers.add_parser("adapt-native")
    adapt.add_argument("--arm", required=True)
    adapt.add_argument("--result-dir", type=Path, required=True)
    adapt.add_argument("--expected-product-head", required=True)
    identity = subparsers.add_parser("init-identity")
    identity.add_argument("--arm", required=True)
    identity.add_argument("--result-dir", type=Path, required=True)
    identity.add_argument("--expected-product-head", required=True)
    identity.add_argument("--wandb-run-id", required=True)
    identity.add_argument("--slurm-job-id")
    args = parser.parse_args(argv)
    if args.command == "compose-preflight":
        validate_all_config_compositions(args.result_root)
        return
    arm = _arm(args.arm)
    if args.command == "overrides":
        _print_overrides(arm, args.result_dir)
    elif args.command == "preflight":
        validate_source(args.source_root, args.expected_product_head)
        validate_container(Path(CONTAINER))
        validate_checkpoint_paths(
            arm,
            target=Path(arm.target_snapshot),
            drafter=None
            if arm.drafter_snapshot is None
            else Path(arm.drafter_snapshot),
        )
    elif args.command == "resume-preflight":
        validate_resume_ready(
            args.result_dir, arm, product_head=args.expected_product_head
        )
    elif args.command == "adapt-native":
        adapt_native_outputs(
            args.result_dir, arm, product_head=args.expected_product_head
        )
    elif args.command == "init-identity":
        initialize_run_identity(
            result_dir=args.result_dir,
            arm=arm.name,
            product_head=args.expected_product_head,
            wandb_run_id=args.wandb_run_id,
            slurm_job_id=args.slurm_job_id,
        )
    else:
        validate_arm_receipts(args.result_dir, arm)


if __name__ == "__main__":
    main()
