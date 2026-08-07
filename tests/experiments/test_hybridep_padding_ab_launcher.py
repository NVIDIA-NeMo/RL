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
RUNTIME_VALIDATE_SCRIPT = (
    ROOT
    / "experiments"
    / "hybridep-padding-ab-q30"
    / "validate-container-runtime-cw.sbatch"
)
STAGE_NIGHTLY_SCRIPT = (
    ROOT
    / "experiments"
    / "hybridep-padding-ab-q30"
    / "stage-nightly-container-cw.sbatch"
)
OCI_STAGE_QWEN30_SCRIPT = (
    ROOT
    / "experiments"
    / "hybridep-padding-ab-q30"
    / "stage-qwen30-oci-nrt.sbatch"
)
LAUNCH_BIN = ROOT / "experiments" / "hybridep-padding-ab-q30" / "launch-bin"
SRUN_WRAPPER = LAUNCH_BIN / "srun"
UV_WRAPPER = LAUNCH_BIN / "uv"
PREFLIGHT_MANIFEST_SHA256 = (
    "ab6797d70d846ae8a9734947f1cac99e1b0184fa7f2ac6c0e2643e77700649da"
)


def _render(
    arm: str,
    *,
    test_only: bool = False,
    extra_env: dict[str, str] | None = None,
) -> dict[str, str]:
    env = {
        **os.environ,
        "ARM": arm,
        "RENDER_ONLY": "1",
        "TEST_ONLY": str(int(test_only)),
        **(extra_env or {}),
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
    assert rendered["padding_telemetry"] == "0"
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
    assert rendered["preflight_manifest_sha256"] == PREFLIGHT_MANIFEST_SHA256
    assert rendered["sbatch_environment_sanitized"] == "1"
    assert rendered["ray_environment_sanitized"] == "1"
    assert rendered["job_name"].endswith(arm)
    assert rendered["output_root"].endswith(f"/{arm}")
    assert "grpo.max_num_steps=20" in rendered["training_command"]
    assert "uv run --no-sync" not in rendered["training_command"]
    assert "/preflight-venv/bin/python examples/run_grpo.py" in rendered["training_command"]
    assert "--active" not in rendered["training_command"]
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


def test_padding_telemetry_is_opt_in_and_legacy_only() -> None:
    rendered = _render(
        "legacy-prepad-17cf",
        extra_env={"COLLECT_PADDING_TELEMETRY": "1"},
    )

    assert rendered["padding_telemetry"] == "1"

    env = {
        **os.environ,
        "ARM": "official-pr5008-17cf",
        "RENDER_ONLY": "1",
        "COLLECT_PADDING_TELEMETRY": "1",
    }
    result = subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=ROOT,
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "COLLECT_PADDING_TELEMETRY requires the legacy pre-padding arm" in result.stderr


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
    runtime_validator = RUNTIME_VALIDATE_SCRIPT.read_text()

    assert "unset RAY_ADDRESS RAY_NAMESPACE" in launcher
    assert "unset RAY_ADDRESS RAY_NAMESPACE" in validator
    assert "NRL_IGNORE_VERSION_MISMATCH" in launcher
    assert "NRL_FORCE_REBUILD_VENVS" in launcher
    assert "NRL_IGNORE_VERSION_MISMATCH" in runtime_validator
    assert "NRL_FORCE_REBUILD_VENVS" in runtime_validator


def test_f725_builder_has_explicit_gpu_allocation_contract() -> None:
    build_script = BUILD_SCRIPT.read_text()

    assert "#SBATCH --nodes=1" in build_script
    assert "#SBATCH --ntasks=1" in build_script
    assert "#SBATCH --gpus-per-node=1" in build_script
    assert "#SBATCH --time=01:00:00" in build_script


def test_container_runtime_probe_is_nonexclusive_and_one_gpu() -> None:
    runtime_probe = RUNTIME_VALIDATE_SCRIPT.read_text()

    assert "#SBATCH --nodes=1" in runtime_probe
    assert "#SBATCH --ntasks=1" in runtime_probe
    assert "#SBATCH --gpus-per-node=1" in runtime_probe
    assert "#SBATCH --segment=1" in runtime_probe
    assert "#SBATCH --exclusive" not in runtime_probe
    assert "#SBATCH --cpus" not in runtime_probe
    assert "#SBATCH --mem" not in runtime_probe
    assert "import deep_ep" in runtime_probe
    assert "import transformer_engine.pytorch as te" in runtime_probe
    assert ': "${SOURCE_PATH:?SOURCE_PATH is required}"' in runtime_probe
    assert ': "${PREFLIGHT_VENV:?PREFLIGHT_VENV is required}"' in runtime_probe
    assert '$SOURCE_PATH:$SOURCE_PATH' in runtime_probe
    assert '$PREFLIGHT_VENV:$PREFLIGHT_VENV' in runtime_probe
    assert "RUN_PYTHON=$PREFLIGHT_VENV/bin/python" in runtime_probe
    assert "export NRL_IGNORE_VERSION_MISMATCH=1" in runtime_probe
    assert "from nemo_rl.models.megatron.setup import _apply_moe_config" in runtime_probe
    assert "moe_hybridep_pad_uneven_dispatch_inputs is True" in runtime_probe
    assert "MANIFEST_OUTPUT" in runtime_probe
    assert 'pip freeze | LC_ALL=C sort' in runtime_probe
    assert 'sha256sum "$MANIFEST_OUTPUT"' in runtime_probe
    assert 'importlib.util.find_spec("uvloop")' in runtime_probe
    assert 'assert hasattr(uvloop, "install")' in runtime_probe
    assert "--container-env=" in runtime_probe


def test_nightly_staging_job_is_reproducible_and_does_not_strand_gpus() -> None:
    stage_script = STAGE_NIGHTLY_SCRIPT.read_text()

    assert "#SBATCH --nodes=1" in stage_script
    assert "#SBATCH --ntasks=1" in stage_script
    assert "#SBATCH --gpus-per-node=1" in stage_script
    assert "#SBATCH --segment=1" in stage_script
    assert "#SBATCH --exclusive" not in stage_script
    assert "#SBATCH --mem" not in stage_script
    assert "enroot import" in stage_script
    assert "sha256sum" in stage_script
    assert "metadata_file" in stage_script
    assert "source_commit=" in stage_script
    assert "mv -Tf" in stage_script


def test_oci_qwen30_staging_uses_cpu_and_persistent_lustre() -> None:
    stage_script = OCI_STAGE_QWEN30_SCRIPT.read_text()

    assert "#SBATCH --nodes=1" in stage_script
    assert "#SBATCH --ntasks=1" in stage_script
    assert "#SBATCH --gpus" not in stage_script
    assert "#SBATCH --gres" not in stage_script
    assert "#SBATCH --exclusive" not in stage_script
    assert "#SBATCH --cpus" not in stage_script
    assert "#SBATCH --mem" not in stage_script
    assert "Qwen/Qwen3-30B-A3B" in stage_script
    assert "snapshot_download" in stage_script
    assert "--no-container-mount-home" in stage_script
    assert "must be a canonical /lustre path" in stage_script
    assert "HF_TOKEN" not in stage_script
    assert "stage-metadata-$SLURM_JOB_ID.json" in stage_script


def test_runtime_and_validator_import_pinned_bridge_and_mcore_sources() -> None:
    launcher = LAUNCHER.read_text()
    validator = VALIDATE_SCRIPT.read_text()

    for script in (launcher, validator):
        assert "Megatron-Bridge/src" in script
        assert "Megatron-Bridge/3rdparty/Megatron-LM" in script
    assert (
        'PYTHONPATH="$SOURCE_PATH:$BRIDGE_SOURCE:$MCORE_SOURCE:$PREFLIGHT_SITE_PACKAGES'
        in launcher
    )
    assert (
        'PYTHONPATH="$SOURCE_PATH:$BRIDGE_SOURCE:$MCORE_SOURCE:$PREFLIGHT_SITE_PACKAGES'
        in validator
    )


def test_ray_bootstrap_uses_validated_frozen_preflight_runtime() -> None:
    launcher = LAUNCHER.read_text()
    validator = VALIDATE_SCRIPT.read_text()

    assert "PREFLIGHT_SITE_PACKAGES=$PREFLIGHT_VENV/lib/python3.13/site-packages" in launcher
    assert 'importlib.util.find_spec("uvloop")' in launcher
    assert "RUN_PYTHON=$PREFLIGHT_VENV/bin/python" in launcher
    assert 'env -u PYTHONPATH "$RUN_PYTHON" -m pip freeze' in launcher
    assert (
        "env -u PYTHONPATH \"$RUN_PYTHON\" -m pip freeze"
        in RUNTIME_VALIDATE_SCRIPT.read_text()
    )
    assert 'export PATH="$LAUNCH_BIN:$PREFLIGHT_VENV/bin:/opt/nemo_rl_venv/bin:$PATH"' in launcher
    assert "export VIRTUAL_ENV=$PREFLIGHT_VENV" in launcher
    assert "export UV_PROJECT_ENVIRONMENT=$PREFLIGHT_VENV" in launcher
    assert "export NRL_IGNORE_VERSION_MISMATCH=1" in launcher
    assert "version_mismatch_override=validated_frozen_preflight_venv" in launcher
    assert "force_rebuild_venvs=disabled" in launcher
    assert '$PREFLIGHT_VENV:$PREFLIGHT_VENV' in launcher
    assert '$LAUNCH_BIN:$LAUNCH_BIN' in launcher
    assert "/opt/nemo_rl_venv/bin/ray --version" in validator
    assert "import ray, requests, urllib3, urllib3.exceptions" in validator
    assert (
        'LD_LIBRARY_PATH="$CUDNN_HOME/lib:/usr/local/cuda/compat/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"'
        in launcher
    )
    assert "/usr/local/cuda/compat/lib:$CUDNN_HOME/lib" in RUNTIME_VALIDATE_SCRIPT.read_text()
    assert (
        "CUDNN_HOME=$PREFLIGHT_SITE_PACKAGES/nvidia/cudnn"
        in launcher
    )
    assert "CUDNN_CONTAINER_PATH" not in launcher
    assert "unset RAY_ADDRESS RAY_NAMESPACE UV_CACHE_DIR_OVERRIDE" in launcher
    assert "export UV_CACHE_DIR_OVERRIDE" not in launcher
    assert "UV_CACHE_DIR_OVERRIDE=${" not in launcher
    assert "/root/.cache/uv" not in launcher
    assert "URLLIB3_HOST_PATH" not in launcher
    assert "URLLIB3_CONTAINER_PATH" not in launcher
    assert "from ray.scripts.scripts import main as ray_cli_main" in launcher
    assert "assert click.ClickException" in launcher
    runtime_probe = RUNTIME_VALIDATE_SCRIPT.read_text()
    assert "URLLIB3_HOST_PATH" not in runtime_probe
    assert "URLLIB3_CONTAINER_PATH" not in runtime_probe
    assert "from ray.scripts.scripts import main as ray_cli_main" in runtime_probe
    assert "assert click.ClickException" in runtime_probe


def test_srun_wrapper_injects_frozen_runtime_without_requiring_deepep(
    tmp_path: Path,
) -> None:
    fake_srun = tmp_path / "srun-real"
    fake_srun.write_text('#!/usr/bin/env bash\nprintf \'%s\\n\' "$@"\n')
    fake_srun.chmod(0o755)
    audit_log = tmp_path / "srun-audit.log"
    env = {
        **os.environ,
        "SRUN_REAL_BIN": str(fake_srun),
        "SRUN_WRAPPER_AUDIT_LOG": str(audit_log),
        "UV_PROJECT_ENVIRONMENT": "/lustre/preflight-venv",
        "VIRTUAL_ENV": "/lustre/preflight-venv",
        "CUDNN_HOME": "/lustre/preflight-venv/cudnn",
        "CUDNN_PATH": "/lustre/preflight-venv/cudnn",
        "PREFLIGHT_SITE_PACKAGES": "/lustre/preflight-venv/site-packages",
        "RUN_PYTHON": "/lustre/preflight-venv/bin/python",
        "NRL_IGNORE_VERSION_MISMATCH": "1",
        "NEMO_RL_VENV_DIR": "/tmp/nemo-rl-venvs",
    }

    result = subprocess.run(
        [
            str(SRUN_WRAPPER),
            "--container-image=image.sqsh",
            "--container-name=ray-head",
            "bash",
            "-lc",
            "true",
        ],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    arguments = result.stdout.splitlines()
    container_env = next(arg for arg in arguments if arg.startswith("--container-env="))
    assert "UV_PROJECT_ENVIRONMENT" in container_env
    assert "PREFLIGHT_SITE_PACKAGES" in container_env
    assert "RUN_PYTHON" in container_env
    assert "NRL_IGNORE_VERSION_MISMATCH" in container_env
    assert "DEEPEP_OVERLAY_DIR" not in container_env
    assert "mode=containerized" in audit_log.read_text()


def test_srun_wrapper_does_not_inject_runtime_into_sandbox_container(
    tmp_path: Path,
) -> None:
    fake_srun = tmp_path / "srun-real"
    fake_srun.write_text('#!/usr/bin/env bash\nprintf \'%s\\n\' "$@"\n')
    fake_srun.chmod(0o755)
    audit_log = tmp_path / "srun-audit.log"
    env = {
        **os.environ,
        "SRUN_REAL_BIN": str(fake_srun),
        "SRUN_WRAPPER_AUDIT_LOG": str(audit_log),
    }

    result = subprocess.run(
        [
            str(SRUN_WRAPPER),
            "--container-image=sandbox.sqsh",
            "--container-mounts=/tmp/sandbox:/tmp/sandbox",
            "true",
        ],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    assert all(not arg.startswith("--container-env=") for arg in result.stdout.splitlines())
    assert "mode=delegate-other-container" in audit_log.read_text()


def test_srun_wrapper_rejects_an_existing_container_env(tmp_path: Path) -> None:
    fake_srun = tmp_path / "srun-real"
    fake_srun.write_text("#!/usr/bin/env bash\nexit 0\n")
    fake_srun.chmod(0o755)
    env = {
        **os.environ,
        "SRUN_REAL_BIN": str(fake_srun),
        "SRUN_WRAPPER_AUDIT_LOG": str(tmp_path / "srun-audit.log"),
    }

    result = subprocess.run(
        [
            str(SRUN_WRAPPER),
            "--container-image=image.sqsh",
            "--container-name=ray-head",
            "--container-env=PATH",
            "true",
        ],
        env=env,
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "container-env is managed by the wrapper" in result.stderr


def test_uv_wrapper_delegates_without_recursing(tmp_path: Path) -> None:
    fake_uv = tmp_path / "uv-real"
    fake_uv.write_text('#!/usr/bin/env bash\nprintf \'%s\\n\' "$@"\n')
    fake_uv.chmod(0o755)
    audit_log = tmp_path / "uv-audit.log"
    env = {
        **os.environ,
        "PATH": f"{LAUNCH_BIN}:{os.environ['PATH']}",
        "UV_REAL_BIN": str(fake_uv),
        "UV_WRAPPER_AUDIT_LOG": str(audit_log),
    }

    result = subprocess.run(
        [str(UV_WRAPPER), "run", "--no-sync", "python", "-V"],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout.splitlines() == ["run", "--no-sync", "python", "-V"]
    assert "mode=delegate" in audit_log.read_text()


def test_launcher_stages_and_mounts_a_pinned_uv_delegate() -> None:
    launcher = LAUNCHER.read_text()
    srun_wrapper = SRUN_WRAPPER.read_text()

    assert 'UV_DELEGATE_SOURCE=$(realpath -e -- "$(command -v uv)")' in launcher
    assert 'UV_DELEGATE_SHA256=$(sha256sum "$UV_DELEGATE_SOURCE"' in launcher
    assert 'UV_REAL_BIN="$UV_ARTIFACT_DIR/uv"' in launcher
    assert '[[ $(sha256sum "$UV_REAL_BIN"' in launcher
    assert "uv_delegate_sha256=%s" in launcher
    assert "$UV_ARTIFACT_DIR:$UV_ARTIFACT_DIR:ro" in launcher
    assert "export UV_REAL_BIN" in launcher
    assert "UV_REAL_BIN" in srun_wrapper.split("container_env=", 1)[1]


def test_deepep_overlay_is_staged_once_on_lustre_and_validated_on_compute() -> None:
    launcher = LAUNCHER.read_text()
    setup = launcher.split("read -r -d '' SETUP_COMMAND <<'SETUP'", 1)[1].split(
        "\nSETUP", 1
    )[0]

    assert 'DEEPEP_OVERLAY_ROOT="$EXPERIMENT_ROOT/artifacts/deepep-overlays"' in launcher
    assert (
        'DEEPEP_OVERLAY_DIR="$DEEPEP_OVERLAY_ROOT/$DEEPEP_SHA256-tree-v1"'
        in launcher
    )
    assert (
        "UV_NO_CONFIG=1 uv pip install --python-version 3.13" in launcher
    )
    assert "--python-platform x86_64-unknown-linux-gnu" in launcher
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
