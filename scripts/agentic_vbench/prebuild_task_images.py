#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Warm the task backend's image cache for the tasks a training set uses.

Harbor builds each task image when its trial starts; on a fresh backend that is
several minutes per task during which the GPUs idle. This builds the same
adapted context the agent's RemoteEnvironment stages (apt runs as container
root in the rootless namespace) so Harbor's later build hits the layer cache.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

GYM_ROOT = Path(__file__).resolve().parents[2] / "3rdparty/Gym-workspace/Gym"
sys.path.insert(0, str(GYM_ROOT))

from responses_api_agents.agentic_vbench_agent.core import backend_environment, inventory  # noqa: E402


def build(task: dict, benchmark_root: Path, docker: Path, env: dict, log_dir: Path) -> bool:
    environment = benchmark_root / "tasks" / f"agentic_vbench_{task['family']}" / task["task_id"] / "environment"
    with tempfile.TemporaryDirectory(prefix="avb-prebuild-") as staging_root:
        staged = Path(staging_root) / "build-context"
        shutil.copytree(environment, staged)
        dockerfile = staged / "Dockerfile"
        # Identical to RemoteEnvironment's adaptation so the layer cache matches.
        dockerfile.write_bytes(dockerfile.read_bytes().replace(b"apt-get ", b"apt-get -o APT::Sandbox::User=root "))
        tag = "localhost/avb-prebuild-" + task["task_id"].lower()
        with (log_dir / f"{task['task_id']}.log").open("wb") as log:
            result = subprocess.run([str(docker), "build", "-t", tag, str(staged)], stdout=log, stderr=subprocess.STDOUT, env=env)
    print("READY" if result.returncode == 0 else "FAILED", task["task_id"], flush=True)
    return result.returncode == 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True, help="*_manifest.json written by build_train_dataset.py")
    parser.add_argument("--backend-dir", type=Path, required=True)
    parser.add_argument("--docker-cli-dir", type=Path, required=True)
    parser.add_argument("--log-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    tasks = inventory(args.benchmark_root)
    selected = [tasks[t] for t in json.loads(args.manifest.read_text())["tasks"]]
    env = dict(os.environ)
    env.update(backend_environment(args.backend_dir, wait_seconds=60))
    env["PATH"] = os.pathsep.join([str(args.docker_cli_dir), env.get("PATH", "")])
    env["DOCKER_CONFIG"] = str(args.docker_cli_dir)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    docker = args.docker_cli_dir / "docker"
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(lambda t: build(t, args.benchmark_root, docker, env, args.log_dir), selected))
    print(f"Built {sum(results)} of {len(results)} task images on {env['DOCKER_HOST']}", flush=True)
    raise SystemExit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
