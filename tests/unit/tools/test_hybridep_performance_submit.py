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

import os
import shutil
import subprocess
from pathlib import Path

SUBMIT_SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "experiments"
    / "h100-hybridep-performance-20260817"
    / "submit.sh"
)


def _run(command: list[str], cwd: Path) -> str:
    return subprocess.run(
        command,
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _build_test_repo(tmp_path: Path, model_family: str) -> tuple[Path, dict[str, str]]:
    repo = tmp_path / "repo"
    experiment_dir = repo / "experiments" / "h100-hybridep-performance-20260817"
    recipe_dir = repo / "examples" / "configs" / "recipes" / "llm" / "performance"
    fake_bin = tmp_path / "bin"
    experiment_dir.mkdir(parents=True)
    recipe_dir.mkdir(parents=True)
    fake_bin.mkdir()

    shutil.copy2(SUBMIT_SCRIPT, experiment_dir / "submit.sh")
    (experiment_dir / "matrix.tsv").write_text(
        "run_id\tmodel_family\trecipe\tnodes\tarm\tmax_steps\n"
        f"test-run\t{model_family}\trecipe.yaml\t1\thybridep\t20\n"
    )
    (recipe_dir / "recipe.yaml").write_text("cluster:\n  num_nodes: 1\n")
    (repo / "ray.sub").write_text("#!/bin/bash\n")

    fake_sbatch = fake_bin / "sbatch"
    fake_sbatch.write_text(
        "#!/bin/bash\n"
        "set -euo pipefail\n"
        "if [[ ${FAKE_SBATCH_MODE:-record} == mutate-and-run ]]; then\n"
        '  git -C "${FAKE_PROJECT_ROOT}" commit --allow-empty -m queued-change >/dev/null\n'
        '  SLURM_JOB_ID=123 bash -c "${COMMAND}"\n'
        "fi\n"
    )
    fake_sbatch.chmod(0o755)

    fake_uv = fake_bin / "uv"
    fake_uv.write_text('#!/bin/bash\nset -euo pipefail\ntouch "${FAKE_UV_MARKER}"\n')
    fake_uv.chmod(0o755)

    _run(["git", "init", "-q"], repo)
    _run(["git", "config", "user.name", "HybridEP test"], repo)
    _run(["git", "config", "user.email", "hybridep-test@example.com"], repo)
    _run(["git", "add", "."], repo)
    _run(["git", "commit", "-q", "-m", "fixture"], repo)
    baseline_commit = _run(["git", "rev-parse", "HEAD"], repo)

    container = tmp_path / "container.sqsh"
    container.touch()
    hf_home = tmp_path / "hf-home"
    hf_home.mkdir()
    run_root = tmp_path / "runs"

    environment = os.environ.copy()
    environment.update(
        {
            "BASELINE_COMMIT": baseline_commit,
            "CONTAINER": str(container),
            "FAKE_PROJECT_ROOT": str(repo),
            "FAKE_UV_MARKER": str(tmp_path / "uv-ran"),
            "HF_HOME": str(hf_home),
            "MATRIX": str(experiment_dir / "matrix.tsv"),
            "MOUNTS": f"{tmp_path}:{tmp_path}",
            "PATH": f"{fake_bin}:{environment['PATH']}",
            "RUN_ROOT": str(run_root),
            "SBATCH_ACCOUNT": "test-account",
            "WANDB_API_KEY": "test-key",
            "WANDB_PROJECT": "test-project",
        }
    )
    environment.pop("NRL_DEEPSEEK_V3_BF16_CKPT", None)
    return repo, environment


def _launch(
    repo: Path, environment: dict[str, str]
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            str(
                repo
                / "experiments"
                / "h100-hybridep-performance-20260817"
                / "submit.sh"
            ),
            "test-run",
        ],
        cwd=repo,
        env=environment,
        capture_output=True,
        text=True,
    )


def test_deepseek_submission_requires_bf16_checkpoint(tmp_path: Path) -> None:
    repo, environment = _build_test_repo(tmp_path, "DeepSeek-V3")

    result = _launch(repo, environment)

    assert result.returncode != 0
    assert "NRL_DEEPSEEK_V3_BF16_CKPT" in result.stderr


def test_submission_rejects_dirty_source_tree(tmp_path: Path) -> None:
    repo, environment = _build_test_repo(tmp_path, "Qwen3-30B-A3B")
    recipe = (
        repo
        / "examples"
        / "configs"
        / "recipes"
        / "llm"
        / "performance"
        / "recipe.yaml"
    )
    recipe.write_text("cluster:\n  num_nodes: 1\n# queued edit\n")

    result = _launch(repo, environment)

    assert result.returncode != 0
    assert "clean source tree" in result.stderr


def test_queued_job_rejects_source_revision_change(tmp_path: Path) -> None:
    repo, environment = _build_test_repo(tmp_path, "Qwen3-30B-A3B")
    environment["FAKE_SBATCH_MODE"] = "mutate-and-run"

    result = _launch(repo, environment)

    assert result.returncode != 0
    assert "source revision changed after submission" in result.stderr
    assert not Path(environment["FAKE_UV_MARKER"]).exists()
