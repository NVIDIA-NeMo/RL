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

import os
import subprocess
from pathlib import Path


def test_vlm_mpo_chain_requires_targets_for_multiple_segments():
    repo_root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env.update({"CHAIN_SEGMENTS": "2", "CHAIN_STEP_TARGETS": ""})

    result = subprocess.run(
        ["bash", str(repo_root / "scripts" / "vlm_mpo_chain.sh")],
        capture_output=True,
        check=False,
        env=env,
        text=True,
    )

    assert result.returncode == 2
    assert (
        "CHAIN_STEP_TARGETS is required when CHAIN_SEGMENTS is greater than 1"
        in result.stderr
    )


def test_vlm_mpo_launcher_overrides_declared_stop_key(tmp_path):
    repo_root = Path(__file__).resolve().parents[2]
    fake_repo = tmp_path / "repo"
    fake_bin = tmp_path / "bin"
    fake_repo.mkdir()
    fake_bin.mkdir()
    (fake_repo / "config.yaml").write_text("{}\n")
    (fake_repo / "ray.sub").write_text("#!/usr/bin/env bash\n")
    fake_sbatch = fake_bin / "sbatch"
    fake_sbatch.write_text(
        "#!/usr/bin/env bash\n"
        "printf '%s\\n' \"$COMMAND\"\n"
        "printf 'Submitted batch job 123\\n'\n"
    )
    fake_sbatch.chmod(0o755)

    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{fake_bin}:/usr/bin:/bin",
            "HOME": str(tmp_path),
            "NEMORL": str(fake_repo),
            "CONFIG_PATH": "config.yaml",
            "CONTAINER": "test-container.sqsh",
            "SBATCH_ACCOUNT": "test-account",
            "MPO_DATA_PATH": "test-data.json",
            "MPO_MAX_NUM_STEPS": "5",
            "MPO_STOP_AFTER_STEP": "3",
        }
    )

    result = subprocess.run(
        ["bash", str(repo_root / "scripts" / "vlm_mpo.sh")],
        capture_output=True,
        check=False,
        env=env,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert "mpo.stop_after_step=3" in result.stdout
    assert "+mpo.stop_after_step" not in result.stdout
