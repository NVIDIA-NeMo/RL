#!/usr/bin/env python3
"""Create the formal Phase 2 experiment metadata for one Slurm engine launch."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def jsonl_record_count(path: Path) -> int:
    count = 0
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            if not isinstance(json.loads(line), dict):
                raise TypeError(f"{path}:{line_number}: expected a JSON object")
            count += 1
    return count


def jsonl_records(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise TypeError(f"{path}:{line_number}: expected a JSON object")
            records.append(value)
    return records


def command_output(command: list[str], *, cwd: Path | None = None) -> str:
    result = subprocess.run(
        command,
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    output = result.stdout.strip() or result.stderr.strip()
    if not output:
        raise RuntimeError(f"command produced no version output: {command!r}")
    return output


def clean_git_commit(path: Path, *, component: str) -> str:
    result = subprocess.run(
        ["git", "status", "--porcelain=v1", "--untracked-files=no"],
        cwd=path,
        check=True,
        capture_output=True,
        text=True,
    )
    status = result.stdout.strip()
    if status:
        raise RuntimeError(f"{component} source has tracked changes:\n{status}")
    return command_output(["git", "rev-parse", "HEAD"], cwd=path).splitlines()[0]


def environment_identity(path: Path, *, packages: list[str]) -> dict[str, object]:
    environment = path.expanduser().resolve(strict=True)
    python = environment / "bin/python"
    if not python.exists():
        raise FileNotFoundError(f"environment Python is missing: {python}")
    pyvenv_config = (environment / "pyvenv.cfg").read_text(encoding="utf-8")
    uv_match = re.search(r"^uv\s*=\s*(\S+)\s*$", pyvenv_config, flags=re.MULTILINE)
    creator_uv = uv_match.group(1) if uv_match else None
    if creator_uv != "0.11.28":
        raise RuntimeError(
            f"formal Phase 2 environment must be created by uv 0.11.28: {environment}"
        )
    package_code = (
        "import importlib.metadata as m, json, platform; "
        f"names={packages!r}; "
        "print(json.dumps({'python': platform.python_version(), "
        "'packages': {name: m.version(name) for name in names}}, sort_keys=True))"
    )
    versions = json.loads(command_output([str(python), "-c", package_code]))
    return {
        "path": str(environment),
        "created_by_uv": creator_uv,
        **versions,
    }


def runtime_verification(
    path: Path,
    *,
    environment: Path,
    uv_lock_sha256: str,
    pyproject_sha256: str,
) -> dict[str, object]:
    verification_path = path.expanduser().resolve(strict=True)
    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    if not isinstance(verification, dict):
        raise TypeError(f"runtime verification must be a JSON object: {path}")
    expected = {
        "schema_version": 1,
        "status": "passed",
        "environment": str(environment.expanduser().resolve(strict=True)),
        "uv_lock_sha256": uv_lock_sha256,
        "pyproject_sha256": pyproject_sha256,
    }
    for field, expected_value in expected.items():
        if verification.get(field) != expected_value:
            raise ValueError(
                f"{verification_path}: {field} differs from formal runtime: "
                f"{verification.get(field)!r} != {expected_value!r}"
            )
    verification_uv = verification.get("uv_version")
    if (
        not isinstance(verification_uv, str)
        or re.match(r"^uv 0\.11\.28(?:\s|$)", verification_uv) is None
    ):
        raise ValueError(
            f"{verification_path}: formal runtime requires uv 0.11.28, "
            f"got {verification_uv!r}"
        )
    expected_interpreter = str((environment / "bin/python").resolve(strict=True))
    if verification.get("interpreter") != expected_interpreter:
        raise ValueError(
            f"{verification_path}: interpreter differs from formal environment: "
            f"{verification.get('interpreter')!r} != {expected_interpreter!r}"
        )
    python_install_dir = verification.get("python_install_dir")
    if not isinstance(python_install_dir, str) or not Path(
        expected_interpreter
    ).is_relative_to(Path(python_install_dir).resolve(strict=True)):
        raise ValueError(
            f"{verification_path}: interpreter is outside persistent Python runtime"
        )
    if verification.get("build_environment") != {"DG_USE_LOCAL_VERSION": "0"}:
        raise ValueError(
            f"{verification_path}: deterministic DeepGEMM build setting is missing"
        )
    sync_check = verification.get("uv_sync_check")
    requirement_check = verification.get("requirement_check")
    if not isinstance(sync_check, dict) or sync_check.get("returncode") != 0:
        raise ValueError(f"{verification_path}: uv sync check did not pass")
    if (
        not isinstance(requirement_check, dict)
        or requirement_check.get("unexpected") != []
    ):
        raise ValueError(
            f"{verification_path}: unexpected dependency incompatibilities remain"
        )
    return {
        "artifact": str(verification_path),
        "artifact_sha256": sha256(verification_path),
        **verification,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workload", type=Path, required=True)
    parser.add_argument("--warmup", type=Path, required=True)
    parser.add_argument("--model-snapshot", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--model-repo-id", required=True)
    parser.add_argument("--model-revision", required=True)
    parser.add_argument("--container-digest", required=True)
    parser.add_argument("--rl-insight-source", type=Path, required=True)
    parser.add_argument("--prometheus-bin", type=Path, required=True)
    parser.add_argument("--uv-bin", type=Path, required=True)
    parser.add_argument("--runtime-env", type=Path, required=True)
    parser.add_argument("--nemo-gym-env", type=Path, required=True)
    parser.add_argument("--runtime-verification", type=Path, required=True)
    parser.add_argument("--nemo-gym-verification", type=Path, required=True)
    parser.add_argument("--launch-id", required=True)
    parser.add_argument(
        "--routing-policy",
        choices=("direct", "cache_aware", "consistent_hash"),
        required=True,
    )
    parser.add_argument("--seed", required=True)
    parser.add_argument("--num-prompts", type=int, required=True)
    parser.add_argument("--num-generations", type=int, required=True)
    parser.add_argument("--warmup-requests", type=int, required=True)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--data-parallel-size", type=int, default=8)
    parser.add_argument("--num-nodes", type=int, default=1)
    parser.add_argument("--gpus-per-node", type=int, default=8)
    parser.add_argument("--max-context-tokens", type=int, default=8192)
    parser.add_argument("--max-output-tokens", type=int, default=256)
    parser.add_argument("--max-num-seqs", type=int, default=256)
    parser.add_argument("--max-num-batched-tokens", type=int, default=8192)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if re.fullmatch(r"sha256:[0-9a-fA-F]{64}", args.container_digest) is None:
        raise ValueError("--container-digest must be sha256:<64 hex digits>")
    positive_counts = {
        "num_prompts": args.num_prompts,
        "num_generations": args.num_generations,
        "warmup_requests": args.warmup_requests,
    }
    for name, value in positive_counts.items():
        if value <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if jsonl_record_count(args.workload) != args.num_prompts:
        raise ValueError("measured workload record count differs from --num-prompts")
    if jsonl_record_count(args.warmup) != args.warmup_requests:
        raise ValueError("warmup workload record count differs from --warmup-requests")
    if (
        jsonl_records(args.warmup)
        != jsonl_records(args.workload)[: args.warmup_requests]
    ):
        raise ValueError(
            "warmup workload must match the measured workload prefix executed by "
            "the Phase 2 warmup hook"
        )

    model_snapshot = args.model_snapshot.expanduser().resolve(strict=True)
    if model_snapshot.name != args.model_revision:
        raise ValueError(
            "model snapshot directory does not match --model-revision: "
            f"{model_snapshot.name!r} != {args.model_revision!r}"
        )
    if not (model_snapshot / "config.json").is_file():
        raise FileNotFoundError(f"model config is missing from {model_snapshot}")
    tokenizer_config_path = model_snapshot / "tokenizer_config.json"
    tokenizer_config = json.loads(tokenizer_config_path.read_text(encoding="utf-8"))
    chat_template = tokenizer_config.get("chat_template")
    if not isinstance(chat_template, str) or not chat_template:
        raise ValueError(f"{tokenizer_config_path}: chat_template is missing")
    chat_template_sha256 = hashlib.sha256(chat_template.encode("utf-8")).hexdigest()
    nemo_rl_commit = clean_git_commit(args.repo, component="NeMo RL")
    nemo_gym_commit = clean_git_commit(
        args.repo / "3rdparty/Gym-workspace/Gym", component="NeMo Gym"
    )
    rl_insight_commit = clean_git_commit(args.rl_insight_source, component="RL-Insight")
    uv_version = command_output([str(args.uv_bin), "--version"]).splitlines()[0]
    if re.match(r"^uv 0\.11\.28(?:\s|$)", uv_version) is None:
        raise RuntimeError(f"formal Phase 2 runs require uv 0.11.28, got {uv_version}")
    prometheus_version = command_output(
        [str(args.prometheus_bin), "--version"]
    ).splitlines()[0]
    if "version 2.54.1" not in prometheus_version:
        raise RuntimeError(
            f"formal Phase 2 runs require Prometheus 2.54.1, got {prometheus_version}"
        )
    driver_environment = environment_identity(
        args.runtime_env,
        packages=["nemo-rl", "ray", "torch", "transformers", "vllm"],
    )
    gym_environment = environment_identity(
        args.nemo_gym_env,
        packages=["nemo-gym", "nemo-rl", "ray", "torch", "transformers", "vllm-router"],
    )
    driver_packages = driver_environment.get("packages")
    gym_packages = gym_environment.get("packages")
    if not isinstance(driver_packages, dict) or not isinstance(gym_packages, dict):
        raise RuntimeError("environment package inventory is invalid")
    if driver_packages.get("ray") != gym_packages.get("ray"):
        raise RuntimeError(
            "driver and NeMo Gym environments use different Ray versions"
        )
    uv_lock_sha256 = sha256(args.repo / "uv.lock")
    pyproject_sha256 = sha256(args.repo / "pyproject.toml")
    driver_verification = runtime_verification(
        args.runtime_verification,
        environment=args.runtime_env,
        uv_lock_sha256=uv_lock_sha256,
        pyproject_sha256=pyproject_sha256,
    )
    gym_verification = runtime_verification(
        args.nemo_gym_verification,
        environment=args.nemo_gym_env,
        uv_lock_sha256=uv_lock_sha256,
        pyproject_sha256=pyproject_sha256,
    )

    metadata = {
        "schema_version": 1,
        "engine": {"fresh": True, "launch_id": args.launch_id},
        "workload_replay": {
            "faithful": True,
            "workload_sha256": sha256(args.workload),
            "seed": args.seed,
            "num_prompts": args.num_prompts,
            "num_generations_per_prompt": args.num_generations,
        },
        "warmup": {
            "completed": True,
            "source": "measurement_workload_prefix",
            "workload_sha256": sha256(args.warmup),
            "requests": args.warmup_requests,
        },
        "software": {
            "nemo_rl_commit": nemo_rl_commit,
            "nemo_gym_commit": nemo_gym_commit,
            "rl_insight_commit": rl_insight_commit,
            "rl_insight_version": command_output(
                [
                    str(args.runtime_env / "bin/python"),
                    "-c",
                    "import rl_insight; print(rl_insight.__version__)",
                ]
            ),
            "prometheus_version": prometheus_version,
            "prometheus_binary_sha256": sha256(args.prometheus_bin),
            "uv_version": uv_version,
            "uv_lock_sha256": uv_lock_sha256,
            "pyproject_sha256": pyproject_sha256,
            "container_digest": args.container_digest,
            "driver_environment": driver_environment,
            "nemo_gym_environment": gym_environment,
            "runtime_verification": {
                "driver": driver_verification,
                "nemo_gym": gym_verification,
            },
        },
        "model": {
            "name": args.model_name,
            "repo_id": args.model_repo_id,
            "revision": args.model_revision,
            "snapshot_path": str(model_snapshot),
            "tokenizer": args.model_name,
            "tokenizer_repo_id": args.model_repo_id,
            "tokenizer_revision": args.model_revision,
            "chat_template_sha256": chat_template_sha256,
        },
        "topology": {
            "tensor_parallel_size": args.tensor_parallel_size,
            "data_parallel_size": args.data_parallel_size,
            "num_nodes": args.num_nodes,
            "gpus_per_node": args.gpus_per_node,
        },
        "generation": {
            "sampling_parameters": {
                "temperature": 1.0,
                "top_p": 1.0,
                "top_k": None,
                "seed": int(args.seed),
            },
            "concurrency": args.num_prompts * args.num_generations,
            "max_context_tokens": args.max_context_tokens,
            "max_output_tokens": args.max_output_tokens,
        },
        "backend": {
            "prefix_caching_enabled": True,
            "scheduler_parameters": {"scheduling_policy": "fcfs"},
            "batching_parameters": {
                "max_num_seqs": args.max_num_seqs,
                "max_num_batched_tokens": args.max_num_batched_tokens,
            },
        },
        "router": {
            "enabled": args.routing_policy != "direct",
            "policy": args.routing_policy,
            "session_affinity_header": "X-Session-ID",
            "cache_metrics_mode": (
                "debug_log_compat" if args.routing_policy == "cache_aware" else "native"
            ),
            "cache_threshold": 0.3,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite metadata: {args.output}")
    args.output.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(args.output)


if __name__ == "__main__":
    main()
