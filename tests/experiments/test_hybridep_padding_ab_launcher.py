# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.

import os
import subprocess
from pathlib import Path

import pytest


ROOT = Path(__file__).parents[2]
LAUNCHER = (
    ROOT / "experiments" / "hybridep-padding-ab-q30" / "submit-cw-qwen30-matrix.sh"
)
RECIPE = "examples/configs/recipes/llm/performance/grpo-qwen3-30ba3b-4n8g.yaml"
BATCH_SCRIPT = ROOT / "experiments" / "hybridep-padding-ab-q30" / "ray-nonexclusive.sub"
BUILD_SCRIPT = (
    ROOT / "experiments" / "hybridep-padding-ab-q30" / "build-f725-wheel.sbatch"
)
VALIDATE_SCRIPT = (
    ROOT / "experiments" / "hybridep-padding-ab-q30" / "validate-legacy-cw.sbatch"
)


def _render(arm: str, *, test_only: bool = False) -> dict[str, str]:
    env = {
        **os.environ,
        "ARM": arm,
        "RENDER_ONLY": "1",
        "TEST_ONLY": str(int(test_only)),
    }
    result = subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    return dict(line.split("=", 1) for line in result.stdout.splitlines())


@pytest.mark.parametrize(
    (
        "arm",
        "dispatcher",
        "backend",
        "pad_uneven",
        "legacy_prepadding",
        "deepep_commit",
        "requires_deepep",
    ),
    [
        ("official-alltoall", "alltoall", "none", "0", "0", "none", "0"),
        (
            "official-pr5008-17cf",
            "flex",
            "hybridep",
            "1",
            "0",
            "17cfb817bccec3a9c247013360cc550c2bac441e",
            "1",
        ),
        (
            "official-pr5008-f725",
            "flex",
            "hybridep",
            "1",
            "0",
            "f725d29699f5bda9ba789456bb9579af69844685",
            "1",
        ),
        (
            "legacy-prepad-17cf",
            "flex",
            "hybridep",
            "0",
            "1",
            "17cfb817bccec3a9c247013360cc550c2bac441e",
            "1",
        ),
    ],
)
def test_rendered_arm_contract(
    arm: str,
    dispatcher: str,
    backend: str,
    pad_uneven: str,
    legacy_prepadding: str,
    deepep_commit: str,
    requires_deepep: str,
) -> None:
    rendered = _render(arm)

    assert rendered["arm"] == arm
    assert rendered["recipe"] == RECIPE
    assert rendered["nodes"] == "4"
    assert rendered["gpus_per_node"] == "8"
    assert rendered["segment"] == "4"
    assert rendered["max_steps"] == "20"
    assert rendered["sequence_packing"] == "1"
    assert rendered["dispatcher"] == dispatcher
    assert rendered["hybridep_backend"] == backend
    assert rendered["pad_uneven_dispatch_inputs"] == pad_uneven
    assert rendered["legacy_prepadding"] == legacy_prepadding
    assert rendered["deepep_commit"] == deepep_commit
    assert rendered["requires_deepep_artifact"] == requires_deepep
    assert rendered["source_profile"] in {"official", "legacy"}
    assert len(rendered["nemo_rl_commit"]) == 40
    assert len(rendered["bridge_commit"]) == 40
    assert len(rendered["mcore_commit"]) == 40
    assert rendered["source_branch"].startswith("sna/")
    assert Path(rendered["batch_script"]) == BATCH_SCRIPT
    assert rendered["container"].endswith("nemo_rl_nightly_20260805_15171871.sqsh")
    assert len(rendered["container_sha256"]) == 64
    assert len(rendered["preflight_manifest_sha256"]) == 64
    assert rendered["sbatch_environment_sanitized"] == "1"
    assert rendered["ray_environment_sanitized"] == "1"
    assert rendered["job_name"].endswith(arm)
    assert rendered["output_root"].endswith(f"/{arm}")
    assert "grpo.max_num_steps=20" in rendered["training_command"]
    assert "policy.sequence_packing.enabled=true" in rendered["training_command"]
    assert "--nodes=4" in rendered["sbatch_command"]
    assert "--gpus-per-node=8" in rendered["sbatch_command"]
    assert "--segment=4" in rendered["sbatch_command"]
    assert "--time=02:00:00" in rendered["sbatch_command"]
    assert "--exclusive" not in rendered["sbatch_command"]
    assert "--cpus" not in rendered["sbatch_command"]
    assert "--mem" not in rendered["sbatch_command"]
    assert str(BATCH_SCRIPT) in rendered["sbatch_command"]
    if legacy_prepadding == "1":
        assert (
            "++policy.megatron_cfg.moe_hybridep_prepad_packed_inputs=true"
            in rendered["training_command"]
        )
    else:
        assert "moe_hybridep_prepad_packed_inputs" not in rendered["training_command"]


def test_effective_batch_script_is_nonexclusive_and_uses_allocated_cpus() -> None:
    batch_script = BATCH_SCRIPT.read_text()

    assert "#SBATCH --exclusive" not in batch_script
    assert "SLURM_CPUS_ON_NODE" in batch_script
    assert 'export NRL_MATRIX_JOB_ID=${SLURM_JOB_ID:?SLURM_JOB_ID is required}' in batch_script
    assert 'exec bash "$SOURCE_PATH/ray.sub"' in batch_script


def test_driver_uses_job_id_captured_before_ray_clears_slurm_environment() -> None:
    launcher = LAUNCHER.read_text()
    driver = launcher.split("read -r -d '' COMMAND <<'DRIVER'", 1)[1].split(
        "\nDRIVER", 1
    )[0]

    assert ': "${NRL_MATRIX_JOB_ID:?NRL_MATRIX_JOB_ID is required}"' in driver
    assert "__SLURM_JOB_ID__/$NRL_MATRIX_JOB_ID" in driver
    assert 'training-$NRL_MATRIX_JOB_ID.log' in driver
    assert "$SLURM_JOB_ID" not in driver


def test_rendered_test_only_is_added_exactly_once() -> None:
    rendered = _render("official-pr5008-17cf", test_only=True)

    assert rendered["sbatch_command"].split().count("--test-only") == 1


def test_inherited_sbatch_options_are_sanitized() -> None:
    env = {
        **os.environ,
        "ARM": "official-alltoall",
        "RENDER_ONLY": "1",
        "SBATCH_EXCLUSIVE": "1",
        "SBATCH_CPUS_PER_GPU": "64",
        "SBATCH_MEM": "1T",
    }
    result = subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )
    rendered = dict(line.split("=", 1) for line in result.stdout.splitlines())

    assert rendered["sbatch_environment_sanitized"] == "1"
    assert "--exclusive" not in rendered["sbatch_command"]
    assert "--cpus" not in rendered["sbatch_command"]
    assert "--mem" not in rendered["sbatch_command"]


def test_inherited_ray_cluster_address_is_sanitized() -> None:
    launcher = LAUNCHER.read_text()
    validator = VALIDATE_SCRIPT.read_text()

    assert "unset RAY_ADDRESS RAY_NAMESPACE" in launcher
    assert "unset RAY_ADDRESS RAY_NAMESPACE" in validator


def test_f725_builder_has_explicit_gpu_allocation_contract() -> None:
    build_script = BUILD_SCRIPT.read_text()

    assert "#SBATCH --nodes=1" in build_script
    assert "#SBATCH --ntasks=1" in build_script
    assert "#SBATCH --gpus-per-node=1" in build_script
    assert "#SBATCH --time=01:00:00" in build_script


def test_runtime_and_validator_import_pinned_bridge_and_mcore_sources() -> None:
    launcher = LAUNCHER.read_text()
    validator = VALIDATE_SCRIPT.read_text()

    for script in (launcher, validator):
        assert "Megatron-Bridge/src" in script
        assert "Megatron-Bridge/3rdparty/Megatron-LM" in script
        assert (
            'PYTHONPATH="$SOURCE_PATH:$BRIDGE_SOURCE:$MCORE_SOURCE:$PREFLIGHT_SITE_PACKAGES'
            in script
        )


def test_ray_bootstrap_uses_the_pinned_preflight_site_packages() -> None:
    launcher = LAUNCHER.read_text()
    validator = VALIDATE_SCRIPT.read_text()

    assert "PREFLIGHT_SITE_PACKAGES=$PREFLIGHT_VENV/lib/python3.13/site-packages" in launcher
    assert "import ray, requests, urllib3.exceptions" in launcher
    assert "/opt/nemo_rl_venv/bin/ray --version" in validator
    assert "import ray, requests, urllib3, urllib3.exceptions" in validator
    assert 'LD_LIBRARY_PATH="$CUDNN_HOME/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"' in launcher
    assert (
        "CUDNN_CONTAINER_PATH=/opt/nemo_rl_venv/lib/python3.13/site-packages/nvidia/cudnn"
        in launcher
    )
    assert '$CUDNN_HOME:$CUDNN_CONTAINER_PATH' in launcher


def test_deepep_overlay_is_staged_once_on_lustre_and_validated_on_compute() -> None:
    launcher = LAUNCHER.read_text()
    setup = launcher.split("read -r -d '' SETUP_COMMAND <<'SETUP'", 1)[1].split(
        "\nSETUP", 1
    )[0]

    assert 'DEEPEP_OVERLAY_ROOT="$EXPERIMENT_ROOT/artifacts/deepep-overlays"' in launcher
    assert 'DEEPEP_OVERLAY_DIR="$DEEPEP_OVERLAY_ROOT/$DEEPEP_SHA256"' in launcher
    assert (
        'UV_NO_CONFIG=1 uv pip install --python "$PREFLIGHT_VENV/bin/python"'
        in launcher
    )
    assert '--target "$DEEPEP_OVERLAY_TEMP" --no-deps --reinstall' in launcher
    assert 'mv "$DEEPEP_OVERLAY_TEMP" "$DEEPEP_OVERLAY_DIR"' in launcher
    assert 'require_canonical_lustre_path DEEPEP_OVERLAY_DIR "$DEEPEP_OVERLAY_DIR"' in launcher
    assert 'directory_tree_sha256 "$DEEPEP_OVERLAY_TEMP"' in launcher
    assert 'directory_tree_sha256 "$DEEPEP_OVERLAY_DIR"' in launcher
    assert "export PYTHONDONTWRITEBYTECODE=1" in launcher
    assert '$DEEPEP_OVERLAY_DIR:$DEEPEP_OVERLAY_DIR' in launcher
    assert 'UV_NO_CONFIG=1 uv pip install --target "$DEEPEP_OVERLAY_DIR"' not in setup
    assert "import deep_ep, deep_ep_cpp, hybrid_ep_cpp" in setup


def test_validator_archives_pytest_results_outside_the_source_tree() -> None:
    validator = VALIDATE_SCRIPT.read_text()

    assert "archive_unit_results()" in validator
    assert "trap archive_unit_results EXIT" in validator
    assert '$OUTPUT_ROOT/generated-test-artifacts' in validator


def test_login_node_preflight_does_not_require_gpu_tools() -> None:
    launcher = LAUNCHER.read_text()

    assert "nvidia-smi" not in launcher.split("for command_name in ", 1)[1].split("; do", 1)[0]
    assert 'GPU_MODELS=$(nvidia-smi --query-gpu=name --format=csv,noheader)' in launcher


def test_container_checksum_cache_is_bound_to_file_identity() -> None:
    launcher = LAUNCHER.read_text()

    assert "stat --printf='%d:%i:%s:%Y:%Z'" in launcher
    assert "CONTAINER_CACHE_KEY=" in launcher
    assert "CONTAINER_CHECKSUM_MODE=cache-hit" in launcher
    assert "CONTAINER_CHECKSUM_MODE=cache-miss-verified" in launcher
    assert 'sha256sum "$CONTAINER"' in launcher
    assert "container_stat_fingerprint=" in launcher


def test_external_worktree_git_common_dir_is_mounted() -> None:
    launcher = LAUNCHER.read_text()

    assert "rev-parse --path-format=absolute --git-common-dir" in launcher
    assert 'require_canonical_lustre_path GIT_COMMON_DIR "$GIT_COMMON_DIR"' in launcher
    assert ',$GIT_COMMON_DIR:$GIT_COMMON_DIR' in launcher
    assert "git_common_dir=" in launcher


def test_unknown_arm_fails_before_side_effects() -> None:
    env = {**os.environ, "ARM": "typo", "RENDER_ONLY": "1"}

    result = subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "unknown experiment arm" in result.stderr
