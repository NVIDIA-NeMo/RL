# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Launch the real-input EP16 replay on two nodes in the diagnostic allocation."""

import os
import signal
import subprocess
from pathlib import Path

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy


@ray.remote(num_cpus=8, num_gpus=8)
def replay_node(index: int, master: str, source: str) -> int:  # pragma: no cover
    root = Path(os.environ["PR3_KL_ROOT"])
    output = root / "replay"
    output.mkdir(exist_ok=True)
    command = [
        "uv",
        "run",
        "--no-project",
        "--python",
        os.environ["PR3_PYTHON"],
        "python",
        "-m",
        "torch.distributed.run",
        "--nnodes=2",
        "--nproc-per-node=8",
        f"--node-rank={index}",
        f"--master-addr={master}",
        "--master-port=29675",
        "validation/pr3/kl_replay/replay.py",
        "--recipe",
        "validation/pr3/kl_replay/ep16.yaml",
        "--capture",
        str(root / "stream-capture/step_210"),
        "--checkpoint",
        os.environ["PR3_KL_CHECKPOINT"],
        "--output",
        str(output),
        "--source-sha",
        source,
    ]
    with (output / f"node-{index}.log").open("w") as stream:
        process = subprocess.Popen(
            command, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            return process.wait(timeout=9000)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()


if __name__ == "__main__":
    if os.environ["USER"] != "root" or not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Requires the diagnostic Slurm allocation")
    ray.init(address="auto")
    nodes = sorted(
        [
            node
            for node in ray.nodes()
            if node["Alive"] and node["Resources"].get("GPU", 0) >= 8
        ],
        key=lambda node: node["NodeManagerAddress"],
    )[:2]
    if len(nodes) != 2:
        raise RuntimeError("EP16 replay needs two 8-GPU nodes")
    names = (
        "PR3_REPO",
        "PR3_PYTHON",
        "PR3_KL_ROOT",
        "PR3_KL_CHECKPOINT",
        "PYTHONPATH",
        "PATH",
        "HF_HOME",
        "HF_DATASETS_CACHE",
        "HF_HUB_OFFLINE",
        "TRANSFORMERS_OFFLINE",
        "NEMO_RL_VENV_DIR",
        "NRL_IGNORE_VERSION_MISMATCH",
        "OMP_NUM_THREADS",
        "UV_NO_SYNC",
        "UV_NO_PROGRESS",
        "NCCL_DEBUG",
    )
    environment = {name: os.environ[name] for name in names}
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    refs = [
        replay_node.options(
            runtime_env={"env_vars": environment},
            scheduling_strategy=NodeAffinitySchedulingStrategy(
                node["NodeID"], soft=False
            ),
        ).remote(index, nodes[0]["NodeManagerAddress"], source)
        for index, node in enumerate(nodes)
    ]
    ray.wait(refs, num_returns=2, fetch_local=False)
    codes = ray.get(refs)
    if codes != [0, 0]:
        raise RuntimeError(f"Replay failed: {codes}")
