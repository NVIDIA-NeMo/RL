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

import re
from pathlib import Path

EXPERIMENT_DIR = Path(__file__).parents[1] / "experiments/cutedsl_qwen3_30ba3b_oci_1n4g"
SCRIPT = (EXPERIMENT_DIR / "submit_oci_hsg.sh").read_text()
README = (EXPERIMENT_DIR / "README.md").read_text()


def test_wrapper_runs_complete_locked_linux_validation_gate() -> None:
    required_fragments = (
        "Copyright (c) 2026, NVIDIA CORPORATION.",
        "uv sync --locked --extra mcore --group test --group dev",
        "uv lock --check",
        "3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/tests/unit_tests/models/test_param_mapping.py",
        "3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/tests/unit_tests/training/test_checkpointing.py",
        "tests/unit/models/megatron/test_megatron_setup.py",
        "tests/unit/models/megatron/test_community_import.py",
        "tests/test_cutedsl_policy_recipe.py",
        "pyrefly check",
        "parent_precommit.log",
        "bridge_precommit.log",
    )
    for fragment in required_fragments:
        assert fragment in SCRIPT, fragment
    assert SCRIPT.index("uv lock --check") < SCRIPT.index("Running the four-GPU")


def test_wrapper_runs_transformer_engine_forward_backward_on_each_gpu() -> None:
    assert "te.Linear" in SCRIPT
    assert ".backward()" in SCRIPT
    assert "parameter.grad" in SCRIPT
    assert "torch.isfinite(output)" in SCRIPT


def test_wrapper_enforces_clean_pushed_source_and_safe_assignments() -> None:
    assert "status --porcelain" in SCRIPT
    assert "--untracked-files=normal" in SCRIPT
    assert "@{upstream}" in SCRIPT
    assert '"upstream_ref"' in SCRIPT
    assert '"upstream_sha"' in SCRIPT
    assert "submodule status --recursive" in SCRIPT
    assert "line:0:1" in SCRIPT or "${line:0:1}" in SCRIPT
    assert not re.search(r'^\s*readonly\s+\w+="\$\(', SCRIPT, re.MULTILINE)


def test_wrapper_discovers_repository_from_slurm_submit_directory() -> None:
    assert "${SLURM_SUBMIT_DIR:?" in SCRIPT
    assert 'git -C "${SLURM_SUBMIT_DIR}" rev-parse --show-toplevel' in SCRIPT
    assert (
        'EXPERIMENT_DIR="${REPO_ROOT}/experiments/cutedsl_qwen3_30ba3b_oci_1n4g"'
    ) in SCRIPT
    assert "BASH_SOURCE" not in SCRIPT


def test_wrapper_preserves_failure_artifacts_and_enforces_metrics() -> None:
    required_fragments = (
        "RAY_TMPDIR",
        "set +e",
        "PIPESTATUS[0]",
        "optimizer_update_successful",
        '"gen_kl_error_max": 0.05',
        '"token_mult_prob_error_max": 2.0',
        '.endswith(".mem_gb")',
        "assert policy_times",
        "assert tokens_per_second",
        "assert peak_memory_metrics",
    )
    for fragment in required_fragments:
        assert fragment in SCRIPT, fragment
    capture_index = SCRIPT.index("PIPESTATUS[0]")
    artifact_index = SCRIPT.index('find "${RAY_TMPDIR}"')
    exit_index = SCRIPT.index('exit "${grpo_exit_code}"')
    assert capture_index < artifact_index < exit_index


def test_readme_matches_enforced_gate_and_requeue_paths() -> None:
    required_fragments = (
        "$JOB_ID-r$SLURM_RESTART_COUNT",
        "train/optimizer_update_successful",
        "train/gen_kl_error",
        "< 0.05",
        "train/token_mult_prob_error",
        "< 2.0",
        "tests/functional/grpo_vllm_mxfp8_rollout_gb200.sh",
        "post-warmup",
        ".mem_gb",
    )
    for fragment in required_fragments:
        assert fragment in README, fragment
    assert "direct tensor equality" not in README.lower()
