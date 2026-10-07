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

"""ray.sub's ``maybe_gres_arg``, executed against a stub ``sinfo``.

The function is cut out of ray.sub verbatim and run under ``set -eou pipefail``
with a ``sinfo`` on PATH that prints scripted ``%G`` lines, so the tests pin the
flag the srun steps receive and the two failure modes.
"""

import os
import re
import stat
import subprocess
from pathlib import Path

RAY_SUB = Path(__file__).resolve().parents[2] / "ray.sub"


def _function_text() -> str:
    match = re.search(
        r"^maybe_gres_arg\(\) \{\n.*?^\}\n", RAY_SUB.read_text(), re.S | re.M
    )
    assert match, "maybe_gres_arg not found in ray.sub"
    return match.group(0)


def _run(tmp_path, sinfo_lines, gpus_per_node="8"):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    stub = bin_dir / "sinfo"
    stub.write_text(
        "#!/bin/bash\n" + "".join(f"echo '{line}'\n" for line in sinfo_lines)
    )
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
    script = f"set -eou pipefail\n{_function_text()}\nmaybe_gres_arg\n"
    return subprocess.run(
        ["bash", "-c", script],
        capture_output=True,
        text=True,
        env={
            **os.environ,
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "SLURM_JOB_NODELIST": "node[1-2]",
            "SLURM_JOB_PARTITION": "batch",
            "GPUS_PER_NODE": gpus_per_node,
        },
    )


def test_no_gpu_gres_yields_no_flag(tmp_path):
    result = _run(tmp_path, ["(null)", "(null)"])
    assert result.returncode == 0
    assert result.stdout.strip() == ""


def test_uniform_allocation_yields_the_gres_flag(tmp_path):
    result = _run(tmp_path, ["gpu:8", "gpu:8"])
    assert result.returncode == 0
    assert result.stdout.strip() == "--gres=gpu:8"


def test_only_the_gpu_entry_is_read(tmp_path):
    """Socket suffixes, other GRES types, and typed GPU entries do not change
    the count that is compared."""
    for lines in (["gpu:8(S:0-3),shard:64"], ["gpu:h100:8"], ["shard:64,gpu:8"]):
        result = _run(tmp_path, lines)
        assert result.returncode == 0, result.stderr
        assert result.stdout.strip() == "--gres=gpu:8"


def test_a_count_mismatch_fails_naming_the_setting(tmp_path):
    result = _run(tmp_path, ["gpu:8", "gpu:8"], gpus_per_node="4")
    assert result.returncode == 1
    assert "GPUS_PER_NODE=4" in result.stderr
    assert "gpu:8" in result.stderr


def test_mixed_counts_inside_the_allocation_fail(tmp_path):
    """A step cannot be placed on a node with fewer GPUs than it asks for, so a
    mixed allocation is refused up front instead of idling to the deadline."""
    result = _run(tmp_path, ["gpu:8", "gpu:7"])
    assert result.returncode == 1
    assert "mixed GPU counts" in result.stderr
    assert "sbatch --gres=gpu:8" in result.stderr


def test_the_query_is_scoped_to_the_allocation():
    text = _function_text()
    assert 'sinfo -N -n "$SLURM_JOB_NODELIST"' in text
    assert "SLURM_JOB_PARTITION" not in text
