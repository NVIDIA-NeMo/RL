# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
import glob
import os
import subprocess
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from nemo_rl.utils.config import load_config, register_omegaconf_resolvers

# All tests in this module should run first
pytestmark = pytest.mark.run_first

dir_path = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(dir_path, "..", ".."))
configs_dir = os.path.join(project_root, "examples", "configs")
recipes_dir = os.path.join(project_root, "examples", "configs", "recipes")
experiments_dir = os.path.join(project_root, "examples", "configs", "experiments")
test_suites_dir = os.path.join(project_root, "tests", "test_suites")

nightly_test_suite_path = os.path.join(test_suites_dir, "nightly.txt")
release_test_suite_path = os.path.join(test_suites_dir, "release.txt")
nightly_gb200_test_suite_path = os.path.join(test_suites_dir, "nightly_gb200.txt")
release_gb200_test_suite_path = os.path.join(test_suites_dir, "release_gb200.txt")
h100_performance_test_suite_path = os.path.join(test_suites_dir, "performance.txt")
gb200_performance_test_suite_path = os.path.join(
    test_suites_dir, "performance_gb200.txt"
)
disabled_test_suite_path = os.path.join(test_suites_dir, "disabled.txt")

# Relative to project root
ALGO_MAPPING_TO_BASE_YAML = {
    "sft": "examples/configs/sft.yaml",
    "dpo": "examples/configs/dpo.yaml",
    "grpo": "examples/configs/grpo_math_1B.yaml",
    "vlm_grpo": "examples/configs/vlm_grpo_3B.yaml",
    "vlm_sft": "examples/configs/sft_vlm_3B.yaml",
    "vlm_mpo": "examples/configs/vlm_mpo.yaml",
    "distillation": "examples/configs/distillation_math.yaml",
    "rm": "examples/configs/rm.yaml",
    "dapo": "examples/configs/grpo_math_1B.yaml",
    "prorlv2": "examples/configs/prorlv2.v2.yaml",
    "ppo": "examples/configs/ppo_math_1B_megatron.yaml",
    "mopd": "examples/configs/grpo_math_1B.yaml",
    "gdpo": "examples/configs/gdpo_math_1B.yaml",
    "flow_grpo": "examples/configs/flow_grpo_qwen_image_ocr.yaml",
}

# Configuration keys that are allowed to be added to base configs during testing
# These keys may exist in recipe configs but not in base configs, so we need to
# manually add them to avoid merge conflicts during config validation
ALLOWED_ADDITIONAL_CONFIG_KEYS = ["policy.draft", "policy.generation.vllm_kwargs"]


@pytest.fixture
def nightly_test_suite():
    nightly_suite = []
    with open(nightly_test_suite_path, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                nightly_suite.append(line)
    return nightly_suite


@pytest.fixture
def release_test_suite():
    release_suite = []
    with open(release_test_suite_path, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                release_suite.append(line)
    return release_suite


@pytest.fixture
def nightly_gb200_test_suite():
    nightly_gb200_suite = []
    with open(nightly_gb200_test_suite_path, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                nightly_gb200_suite.append(line)
    return nightly_gb200_suite


@pytest.fixture
def release_gb200_test_suite():
    release_gb200_suite = []
    with open(release_gb200_test_suite_path, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                release_gb200_suite.append(line)
    return release_gb200_suite


@pytest.fixture
def performance_test_suite():
    performance_suite = []
    with open(h100_performance_test_suite_path, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                performance_suite.append(line)
    with open(gb200_performance_test_suite_path, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                performance_suite.append(line)
    return performance_suite


@pytest.fixture
def disabled_test_suite():
    disabled_suite = []
    with open(disabled_test_suite_path, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                disabled_suite.append(line)
    return disabled_suite


@pytest.fixture
def all_test_suites(
    nightly_test_suite,
    release_test_suite,
    nightly_gb200_test_suite,
    release_gb200_test_suite,
    performance_test_suite,
    disabled_test_suite,
):
    return (
        nightly_test_suite
        + release_test_suite
        + nightly_gb200_test_suite
        + release_gb200_test_suite
        + performance_test_suite
        + disabled_test_suite
    )


def _test_suite_scripts(root: Path) -> set[str]:
    """Discover core and research suite scripts, relative to the repository."""
    scripts = list((root / "tests/test_suites").rglob("*.sh"))
    scripts.extend(root.glob("research/*/tests/test_suites/**/*.sh"))
    return {path.relative_to(root).as_posix() for path in scripts}


def _recipe_for_test_script(script: str) -> str:
    """Map a suite script to its recipe without losing the research project."""
    path = Path(script)
    if path.parts[:2] == ("tests", "test_suites"):
        return str(
            Path("examples/configs/recipes", *path.parts[2:]).with_suffix(".yaml")
        )
    if (
        len(path.parts) >= 5
        and path.parts[0] == "research"
        and path.parts[2:4] == ("tests", "test_suites")
    ):
        return str(
            Path(*path.parts[:2], "configs/recipes", *path.parts[4:]).with_suffix(
                ".yaml"
            )
        )
    raise ValueError(f"Unsupported test suite path: {script}")


@pytest.fixture
def all_recipe_yaml_rel_paths():
    root = Path(project_root)
    recipes = list((root / "examples/configs/recipes").rglob("*.yaml"))
    recipes.extend(root.glob("research/*/configs/recipes/**/*.yaml"))
    return [path.relative_to(root).as_posix() for path in recipes]


@pytest.fixture
def all_experiment_yaml_paths():
    return glob.glob(os.path.join(experiments_dir, "**", "*.yaml"), recursive=True)


def test_all_experiment_configs_resolve(all_experiment_yaml_paths):
    register_omegaconf_resolvers()
    for config_path in all_experiment_yaml_paths:
        with open(config_path) as config_file:
            config_text = config_file.read()
        for internal_prefix in ("/lustre/", "/scratch/", "/home/"):
            assert internal_prefix not in config_text, (
                f"Experiment config contains an internal path: {config_path}"
            )
        resolved = OmegaConf.to_container(load_config(config_path), resolve=True)
        assert isinstance(resolved, dict)
        wandb_config = resolved.get("logger", {}).get("wandb", {})
        assert "entity" not in wandb_config, (
            f"Experiment config pins a W&B entity: {config_path}"
        )


@pytest.mark.parametrize(
    "test_suite_path",
    [
        nightly_test_suite_path,
        release_test_suite_path,
        nightly_gb200_test_suite_path,
        release_gb200_test_suite_path,
        h100_performance_test_suite_path,
        gb200_performance_test_suite_path,
        disabled_test_suite_path,
    ],
    ids=[
        "nightly_test_suite",
        "release_test_suite",
        "nightly_gb200_test_suite",
        "release_gb200_test_suite",
        "h100_performance_test_suite",
        "gb200_performance_test_suite",
        "disabled_test_suite",
    ],
)
def test_test_suites_exist(test_suite_path):
    assert os.path.exists(test_suite_path), (
        f"Test suite {test_suite_path} does not exist"
    )


def test_no_overlap_across_test_suites(all_test_suites):
    all_tests = set(all_test_suites)
    assert len(all_tests) == len(all_test_suites), (
        f"Test suites have repeats {all_tests}"
    )


def test_nightly_suites_match_gpus_per_node(
    nightly_test_suite, nightly_gb200_test_suite
):
    for test_suite, expected_gpus_per_node in (
        (nightly_test_suite, 8),
        (nightly_gb200_test_suite, 4),
    ):
        for test_script in test_suite:
            gpus_per_node = 8
            with open(os.path.join(project_root, test_script)) as f:
                for line in f:
                    if line.startswith("GPUS_PER_NODE="):
                        gpus_per_node = int(line.split("=", 1)[1].split()[0])
                        break
            assert gpus_per_node == expected_gpus_per_node, (
                f"{test_script} requests {gpus_per_node} GPUs per node, but its "
                f"nightly suite requires {expected_gpus_per_node}"
            )


def test_all_test_scripts_accounted_for_in_test_suites(all_test_suites):
    discovered = _test_suite_scripts(Path(project_root))
    listed = set(all_test_suites)
    assert listed == discovered, (
        f"Unlisted test scripts: {sorted(discovered - listed)}; "
        f"Missing test scripts: {sorted(listed - discovered)}"
    )


def test_all_recipe_yamls_accounted_for_in_test_suites(
    all_recipe_yaml_rel_paths, all_test_suites
):
    """Require a matching recipe for every core and research suite script."""
    expected = {_recipe_for_test_script(script) for script in all_test_suites}
    recipes = set(all_recipe_yaml_rel_paths)
    assert expected == recipes, (
        f"Missing recipes: {sorted(expected - recipes)}; "
        f"Recipes without test scripts: {sorted(recipes - expected)}"
    )


def test_nightly_compute_stays_below_4700_hours(nightly_test_suite, tracker):
    command = f"DRYRUN=1 HF_HOME=... HF_DATASETS_CACHE=... CONTAINER= ACCOUNT= PARTITION= ./tools/launch {' '.join(nightly_test_suite)}"

    print(f"Running command: {command}")

    # Run the command from the project root directory
    result = subprocess.run(
        command,
        shell=True,
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,  # Don't raise exception on non-zero exit code
    )

    # Print stdout and stderr for debugging if the test fails
    print("STDOUT:")
    print(result.stdout)
    print("STDERR:")
    print(result.stderr)

    # Assert that the command exited successfully
    assert result.returncode == 0, f"Command failed with exit code {result.returncode}"

    # Assert that the last line of stdout contains the expected prefix
    stdout_lines = result.stdout.strip().splitlines()
    assert len(stdout_lines) > 0, "Command produced no output"
    last_line = stdout_lines[-1]
    assert last_line.startswith("[INFO]: Total GPU hours:"), (
        f"Last line of output was not as expected: '{last_line}'"
    )
    total_gpu_hours = float(last_line.split(":")[-1].strip())
    assert total_gpu_hours <= 4700, (
        f"Total GPU hours exceeded 4700: {last_line}. We should revisit the test suites to reduce the total GPU hours."
    )
    tracker.track("total_nightly_gpu_hours", total_gpu_hours)


def test_dry_run_does_not_fail_and_prints_total_gpu_hours(all_test_suites):
    command = f"DRYRUN=1 HF_HOME=... HF_DATASETS_CACHE=... CONTAINER= ACCOUNT= PARTITION= ./tools/launch {' '.join(all_test_suites)}"

    # Run the command from the project root directory
    result = subprocess.run(
        command,
        shell=True,
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,  # Don't raise exception on non-zero exit code
    )

    # Print stdout and stderr for debugging if the test fails
    print("STDOUT:")
    print(result.stdout)
    print("STDERR:")
    print(result.stderr)

    # Assert that the command exited successfully
    assert result.returncode == 0, f"Command failed with exit code {result.returncode}"

    # Assert that the last line of stdout contains the expected prefix
    stdout_lines = result.stdout.strip().splitlines()
    assert len(stdout_lines) > 0, "Command produced no output"
    last_line = stdout_lines[-1]
    assert last_line.startswith("[INFO]: Total GPU hours:"), (
        f"Last line of output was not as expected: '{last_line}'"
    )


def test_all_tests_can_find_config_if_dryrun(all_test_suites):
    for test_suite in all_test_suites:
        command = f"TEST_DRYRUN=1 {test_suite}"
        result = subprocess.run(
            command,
            shell=True,
            cwd=project_root,
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, (
            f"Command failed with exit code {result.returncode}"
        )


def test_all_recipes_start_with_algo_hyphen(all_recipe_yaml_rel_paths):
    expected_algos = set(ALGO_MAPPING_TO_BASE_YAML.keys())
    for recipe_yaml in all_recipe_yaml_rel_paths:
        # Research projects define their own algorithms and naming conventions.
        if recipe_yaml.startswith("research/"):
            continue
        basename = os.path.basename(recipe_yaml)
        algo = basename.split("-")[0]
        assert algo in expected_algos, (
            f"Recipe {recipe_yaml} has unexpected algo {algo}"
        )
