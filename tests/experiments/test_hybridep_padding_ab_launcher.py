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
    assert rendered["job_name"].endswith(arm)
    assert rendered["output_root"].endswith(f"/{arm}")
    assert "grpo.max_num_steps=20" in rendered["training_command"]
    assert "policy.sequence_packing.enabled=true" in rendered["training_command"]
    assert "--nodes=4" in rendered["sbatch_command"]
    assert "--gpus-per-node=8" in rendered["sbatch_command"]
    assert "--segment=4" in rendered["sbatch_command"]
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
    assert 'exec bash "$SOURCE_PATH/ray.sub"' in batch_script


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


def test_f725_builder_has_explicit_gpu_allocation_contract() -> None:
    build_script = BUILD_SCRIPT.read_text()

    assert "#SBATCH --nodes=1" in build_script
    assert "#SBATCH --ntasks=1" in build_script
    assert "#SBATCH --gpus-per-node=1" in build_script
    assert "#SBATCH --time=01:00:00" in build_script


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
