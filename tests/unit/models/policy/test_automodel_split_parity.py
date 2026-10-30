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

"""Two-GPU V1/full-batch SC regression, paired with MCore's split parity suite."""

import json
import os
import signal
import subprocess
import sys
from pathlib import Path

import pytest
import torch

pytest.importorskip("nemo_automodel")
pytestmark = pytest.mark.automodel


@pytest.mark.hf_gated
@pytest.mark.timeout(660)
def test_full_batch_split_parity(tiny_llama_model_path, tmp_path):
    """Exercise real FSDP gradients, updates, metadata fetch, and abort on two ranks.

    Reuse the standalone numerical harness so CI and larger model qualification
    execute the same assertions. Two updates per loss normalization exercise
    optimizer/scheduler progression, with two microbatches per rank and a masked
    sample. Network TQ, Ray dispatch, and live refits are covered by L1.
    """
    if torch.cuda.device_count() < 2:
        pytest.skip("Requires two CUDA GPUs")
    root = Path(__file__).resolve().parents[4]
    env = os.environ.copy()
    env.update(
        CUBLAS_WORKSPACE_CONFIG=":4096:8",
        NVTE_ALLOW_NONDETERMINISTIC_ALGO="0",
    )
    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nproc-per-node=2",
        str(root / "tests/functional/automodel_sc_full_batch_parity.py"),
        "--model",
        str(tiny_llama_model_path),
        "--force-hf",
        "--recipe",
        "examples/configs/grpo_math_1B.yaml",
        "--strict-determinism",
        "--rtol",
        "0",
        "--atol",
        "0",
        "--output",
        str(tmp_path),
        "--source-sha",
        "pytest",
    ]
    # Kill the entire torchrun group on timeout, including its CUDA workers.
    log_path = tmp_path / "torchrun.log"
    with log_path.open("w") as log:
        process = subprocess.Popen(
            command,
            cwd=root,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        try:
            returncode = process.wait(timeout=600)
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
    assert returncode == 0, log_path.read_text()
    report = json.loads((tmp_path / "parity.json").read_text())
    assert report["passed"]
    assert {(row["token_level"], row["step"]) for row in report["records"]} == {
        (token_level, step) for token_level in (True, False) for step in range(2)
    }
