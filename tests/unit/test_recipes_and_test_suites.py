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

import pytest

# All tests in this module should run first
pytestmark = pytest.mark.run_first

dir_path = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(dir_path, "..", ".."))
configs_dir = os.path.join(project_root, "examples", "configs")
recipes_dir = os.path.join(project_root, "examples", "configs", "recipes")
test_suites_dir = os.path.join(project_root, "tests", "test_suites")

nightly_test_suite_path = os.path.join(test_suites_dir, "nightly.txt")
release_test_suite_path = os.path.join(test_suites_dir, "release.txt")
nightly_gb200_test_suite_path = os.path.join(test_suites_dir, "nightly_gb200.txt")
release_gb200_test_suite_path = os.path.join(test_suites_dir, "release_gb200.txt")
h100_performance_test_suite_path = os.path.join(test_suites_dir, "performance_h100.txt")
gb200_performance_test_suite_path = os.path.join(
    test_suites_dir, "performance_gb200.txt"
)

# Relative to project root
ALGO_MAPPING_TO_BASE_YAML = {
    "sft": "examples/configs/sft.yaml",
    "dpo": "examples/configs/dpo.yaml",
    "grpo": "examples/configs/grpo_math_1B.yaml",
    "vlm_grpo": "examples/configs/vlm_grpo_3B.yaml",
    "distillation": "examples/configs/distillation_math.yaml",
    "rm": "examples/configs/rm.yaml",
    "dapo": "examples/configs/grpo_math_1B.yaml",
}

# Configuration keys that are allowed to be added to base configs during testing
# These keys may exist in recipe configs but not in base configs, so we need to
# manually add them to avoid merge conflicts during config validation
ALLOWED_ADDITIONAL_CONFIG_KEYS = ["policy.generation.vllm_kwargs"]


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
def all_test_suites(
    nightly_test_suite,
    release_test_suite,
    nightly_gb200_test_suite,
    release_gb200_test_suite,
    performance_test_suite,
):
    return (
        nightly_test_suite
        + release_test_suite
        + nightly_gb200_test_suite
        + release_gb200_test_suite
        + performance_test_suite
    )


@pytest.fixture
def all_recipe_yaml_rel_paths():
    all_recipes = []
    for recipe_path in glob.glob(
        os.path.join(recipes_dir, "**", "*.yaml"), recursive=True
    ):
        all_recipes.append(recipe_path[len(recipes_dir) + 1 :])
    return all_recipes


@pytest.mark.parametrize(
    "test_suite_path",
    [
        nightly_test_suite_path,
        release_test_suite_path,
        nightly_gb200_test_suite_path,
        release_gb200_test_suite_path,
        h100_performance_test_suite_path,
        gb200_performance_test_suite_path,
    ],
    ids=[
        "nightly_test_suite",
        "release_test_suite",
        "nightly_gb200_test_suite",
        "release_gb200_test_suite",
        "h100_performance_test_suite",
        "gb200_performance_test_suite",
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


def test_all_test_scripts_accounted_for_in_test_suites(all_test_suites):
    all_test_scripts_in_test_suites = set(all_test_suites)

    all_tests_in_test_suites_dir = set()
    for recipe_path in glob.glob(
        os.path.join(test_suites_dir, "**", "*.sh"), recursive=True
    ):
        # Strip off the project root and leading slash
        recipe_name = recipe_path[len(project_root) + 1 :]
        all_tests_in_test_suites_dir.add(recipe_name)

    assert all_test_scripts_in_test_suites == all_tests_in_test_suites_dir, (
        "All test scripts are not accounted for in the test suites"
    )


def test_all_recipe_yamls_accounted_for_in_test_suites(
    all_recipe_yaml_rel_paths, all_test_suites
):
    """This test along with test_all_test_scripts_accounted_for_in_test_suites() ensures that all recipe yaml/test scripts/test_suite(txts) are in sync."""
    assert len(set(all_recipe_yaml_rel_paths)) == len(set(all_test_suites)), (
        "Recipe YAMLs should be accounted for in the test suites"
    )

    all_test_script_paths_in_test_suites = set()
    for test_script in all_test_suites:
        # Each test suite is relative from project root
        test_script_rel_to_test_suites_dir = test_script[
            len(os.path.join("tests", "test_suites")) + 1 :
        ]
        all_test_script_paths_in_test_suites.add(test_script_rel_to_test_suites_dir)

    # Since we're comparing yaml to sh, chop off the .sh/.yaml extensions for comparison
    all_test_script_paths_in_test_suites = {
        os.path.splitext(path)[0] for path in all_test_script_paths_in_test_suites
    }
    all_recipe_yaml_rel_paths = {
        os.path.splitext(path)[0] for path in all_recipe_yaml_rel_paths
    }

    assert all_test_script_paths_in_test_suites == set(all_recipe_yaml_rel_paths), (
        "All recipe YAMLs are not accounted for in the test suites"
    )


def test_nightly_compute_stays_below_1210_hours(nightly_test_suite, tracker):
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
    assert total_gpu_hours <= 1210, (
        f"Total GPU hours exceeded 1210: {last_line}. We should revisit the test suites to reduce the total GPU hours."
    )
    tracker.track("total_nightly_gpu_hours", total_gpu_hours)


def test_dry_run_does_not_fail_and_prints_total_gpu_hours():
    command = "DRYRUN=1 HF_HOME=... HF_DATASETS_CACHE=... CONTAINER= ACCOUNT= PARTITION= ./tools/launch ./tests/test_suites/**/*.sh"

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
        basename = os.path.basename(recipe_yaml)
        algo = basename.split("-")[0]
        assert algo in expected_algos, (
            f"Recipe {recipe_yaml} has unexpected algo {algo}"
        )
