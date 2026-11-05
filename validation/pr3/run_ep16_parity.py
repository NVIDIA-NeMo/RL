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


"""Run the fixed-input worker harness on two exclusively allocated Ray nodes."""

import json
import os
import signal
import subprocess
from pathlib import Path

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy


@ray.remote(num_cpus=8, num_gpus=8)
def run_rank(node_rank: int, master_address: str, source_sha: str) -> int:
    repo = Path(os.environ["PR3_REPO"])
    output = repo / "validation/pr3/results/ep16-fixed-input"
    output.mkdir(parents=True, exist_ok=True)
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
        f"--node-rank={node_rank}",
        f"--master-addr={master_address}",
        "--master-port=29635",
        "tests/functional/automodel_sc_full_batch_parity.py",
        "--model",
        "/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/shuangy/models/Qwen/Qwen3.5-35B-A3B-Base",
        "--recipe",
        "validation/pr3/streaming_qwen35.yaml",
        "--recipe-optimizer",
        "--bf16-contract",
        "--optimizer-oracle",
        "--local-batch-size",
        "2",
        "--dispatch-sizes",
        "1,1",
        "--steps",
        "3",
        "--seq-len",
        "256",
        "--strict-determinism",
        "--rtol",
        "1e-5",
        "--atol",
        "1e-7",
        "--source-sha",
        source_sha,
        "--output",
        str(output),
    ]
    env = os.environ.copy()
    env.update(CUBLAS_WORKSPACE_CONFIG=":4096:8", NVTE_ALLOW_NONDETERMINISTIC_ALGO="0")
    with (output / f"node-{node_rank}.log").open("w") as log:
        process = subprocess.Popen(
            command,
            cwd=repo,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            return process.wait(timeout=10800)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()


if __name__ == "__main__":
    assert os.environ["USER"] == "root" and os.environ.get("SLURM_JOB_ID")
    ray.init(address="auto")
    nodes = sorted(
        (
            node
            for node in ray.nodes()
            if node["Alive"] and node["Resources"].get("GPU", 0) >= 8
        ),
        key=lambda node: node["NodeManagerAddress"],
    )
    assert len(nodes) == 2, nodes
    source = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    report = Path("validation/pr3/results/ep16-fixed-input/parity.json")
    report.unlink(missing_ok=True)
    futures = [
        run_rank.options(
            scheduling_strategy=NodeAffinitySchedulingStrategy(
                node["NodeID"], soft=False
            )
        ).remote(index, nodes[0]["NodeManagerAddress"], source)
        for index, node in enumerate(nodes)
    ]
    ray.wait(futures, num_returns=len(futures), fetch_local=False)
    codes = ray.get(futures)
    assert codes == [0, 0], codes
    assert json.loads(report.read_text())["passed"]
    print("EP16_FIXED_INPUT_PASSED", flush=True)
