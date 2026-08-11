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
import re
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
    "distillation": "examples/configs/distillation_math.yaml",
    "rm": "examples/configs/rm.yaml",
    "dapo": "examples/configs/grpo_math_1B.yaml",
    "prorlv2": "examples/configs/prorlv2.v2.yaml",
    "ppo": "examples/configs/ppo_math_1B_megatron.yaml",
    "mopd": "examples/configs/grpo_math_1B.yaml",
    "gdpo": "examples/configs/gdpo_math_1B.yaml",
}

# Configuration keys that are allowed to be added to base configs during testing
# These keys may exist in recipe configs but not in base configs, so we need to
# manually add them to avoid merge conflicts during config validation
ALLOWED_ADDITIONAL_CONFIG_KEYS = ["policy.draft", "policy.generation.vllm_kwargs"]

# Weekly GPU-hour budget per (suite, SKU).
#
# A lane's real cost is (GPU-hours per run) x (runs per week). Suites run at very
# different cadences, so comparing per-run numbers across lanes is misleading:
# nightly is the largest consumer of the fleet by a wide margin even though a
# single release run is far more expensive than a single nightly run.
#
# `runs_per_week` mirrors the nemo-ci pipeline schedules. Keep these in sync if
# a schedule changes:
#   "NeMo RL Nightly tests"          0 2 * * *  -> nightly, nightly_gb200
#   "NeMo RL Weekly Release Tests"   0 4 * * 6  -> release(_gb200)
#   "NeMo RL Weekly Perf Tests"      0 4 * * 6  -> performance(_gb200)
#
# The nightly_mcore lanes are deliberately absent. They are subsets of the
# nightly lanes, so bounding nightly bounds them transitively; giving them their
# own ceiling would mean two numbers to retune every time a nightly test lands.
#
# Budgets are per SKU because H100 and GB200 capacity are separate pools and
# cannot be traded against one another. Ceilings sit roughly 15% above current
# usage so that a legitimate new test has somewhere to land; if a lane is at its
# ceiling the right response is to retire a test, not to raise the number.
#
# The nightly, release and release_gb200 ceilings currently carry five recipes
# that exist to back user guides (the DAPO guide, the audio and audio-visual
# guides, and the README model table) rather than to catch regressions. Once
# those can be marked as documentation-only and stop running, roughly 1,300
# GPU-hours/week leaves these three lanes and their ceilings should come down.
SUITE_BUDGETS = {
    ("nightly", "h100"): {"runs_per_week": 7, "max_gpu_hours_per_week": 26_500},
    ("nightly", "gb200"): {"runs_per_week": 7, "max_gpu_hours_per_week": 2_600},
    ("release", "h100"): {"runs_per_week": 1, "max_gpu_hours_per_week": 7_400},
    ("release", "gb200"): {"runs_per_week": 1, "max_gpu_hours_per_week": 3_100},
    ("performance", "h100"): {"runs_per_week": 1, "max_gpu_hours_per_week": 13_800},
    ("performance", "gb200"): {"runs_per_week": 1, "max_gpu_hours_per_week": 4_800},
}


def _read_test_suite(path):
    """Return the test script paths listed in a test suite manifest."""
    entries = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                entries.append(line)
    return entries


def _suite_manifest_path(suite, sku):
    """Map a (suite, SKU) pair to its manifest. GB200 lanes use the _gb200 suffix."""
    name = suite if sku == "h100" else f"{suite}_gb200"
    return os.path.join(test_suites_dir, f"{name}.txt")


@pytest.fixture
def nightly_test_suite():
    return _read_test_suite(nightly_test_suite_path)


@pytest.fixture
def release_test_suite():
    return _read_test_suite(release_test_suite_path)


@pytest.fixture
def nightly_gb200_test_suite():
    return _read_test_suite(nightly_gb200_test_suite_path)


@pytest.fixture
def release_gb200_test_suite():
    return _read_test_suite(release_gb200_test_suite_path)


@pytest.fixture
def performance_test_suite():
    return _read_test_suite(h100_performance_test_suite_path) + _read_test_suite(
        gb200_performance_test_suite_path
    )


@pytest.fixture
def disabled_test_suite():
    return _read_test_suite(disabled_test_suite_path)


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


ALL_SUITE_NAMES = [
    "nightly",
    "nightly_gb200",
    "nightly_mcore",
    "nightly_mcore_gb200",
    "release",
    "release_gb200",
    "performance",
    "performance_gb200",
    "disabled",
]
VALID_SUITE_VALUES = {"nightly", "release", "performance", "disabled"}
VALID_SKU_VALUES = {"h100", "gb200"}


def _config_value(script_path, key):
    """Read one variable out of a driver's CONFIG block."""
    in_block = False
    with open(os.path.join(project_root, script_path)) as f:
        for line in f:
            if re.match(r"^# =+ BEGIN CONFIG =+", line):
                in_block = True
                continue
            if re.match(r"^# =+ END CONFIG =+", line):
                break
            if in_block and line.startswith(f"{key}="):
                value = line[len(key) + 1 :].split("#")[0].strip()
                return value.strip("\"'")
    return None


def test_every_driver_declares_a_valid_suite_and_sku():
    """Every driver must say which suite it belongs to and which SKU it targets."""
    problems = []
    for script_path in glob.glob(
        os.path.join(test_suites_dir, "**", "*.sh"), recursive=True
    ):
        rel = script_path[len(project_root) + 1 :]
        suite = _config_value(rel, "SUITE")
        sku = _config_value(rel, "SKU")
        if suite not in VALID_SUITE_VALUES:
            problems.append(
                f"{rel}: SUITE={suite!r}, expected one of {sorted(VALID_SUITE_VALUES)}"
            )
        if sku not in VALID_SKU_VALUES:
            problems.append(
                f"{rel}: SKU={sku!r}, expected one of {sorted(VALID_SKU_VALUES)}"
            )
        if suite == "disabled" and not _config_value(rel, "DISABLED_REASON"):
            problems.append(
                f"{rel}: SUITE=disabled requires a non-empty DISABLED_REASON"
            )
    assert not problems, "\n".join(problems)


def test_declared_sku_matches_gpus_per_node():
    """SKU is what the author declares; GPUS_PER_NODE is what Slurm is asked for.

    They must agree, or a test lands in a lane whose hardware it was not written
    for. This replaces the old nightly-only check with one that covers every
    suite.
    """
    expected_gpus = {"h100": 8, "gb200": 4}
    problems = []
    for script_path in glob.glob(
        os.path.join(test_suites_dir, "**", "*.sh"), recursive=True
    ):
        rel = script_path[len(project_root) + 1 :]
        sku = _config_value(rel, "SKU")
        gpus_per_node = int(_config_value(rel, "GPUS_PER_NODE") or 8)
        if expected_gpus.get(sku) != gpus_per_node:
            problems.append(
                f"{rel}: SKU={sku} implies {expected_gpus.get(sku)} GPUs per node, "
                f"but GPUS_PER_NODE={gpus_per_node}"
            )
    assert not problems, "\n".join(problems)


@pytest.mark.parametrize("suite", ALL_SUITE_NAMES)
def test_declarations_reproduce_the_manifest(suite):
    """`tools/list-suites <suite>` must return exactly what the manifest lists.

    The manifests are still authoritative; the declarations are a parallel
    source of truth until nemo-ci reads them instead. This test is what stops
    the two from drifting apart in the meantime, and it retires along with the
    manifests.
    """
    result = subprocess.run(
        [os.path.join(project_root, "tools", "list-suites"), suite],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"tools/list-suites {suite} failed:\n{result.stdout}\n{result.stderr}"
    )

    declared = set(result.stdout.split())
    listed = set(_read_test_suite(os.path.join(test_suites_dir, f"{suite}.txt")))

    only_declared = sorted(declared - listed)
    only_listed = sorted(listed - declared)
    assert not only_declared and not only_listed, (
        f"{suite}: declarations and manifest disagree.\n"
        f"  declared but not in {suite}.txt: {only_declared}\n"
        f"  in {suite}.txt but not declared: {only_listed}"
    )


@pytest.mark.parametrize(
    ("suite", "sku"),
    list(SUITE_BUDGETS),
    ids=[f"{suite}-{sku}" for suite, sku in SUITE_BUDGETS],
)
def test_suite_weekly_gpu_hours_within_budget(suite, sku, tracker):
    budget = SUITE_BUDGETS[(suite, sku)]
    scripts = _read_test_suite(_suite_manifest_path(suite, sku))
    assert scripts, f"Test suite {suite} ({sku}) is empty"

    command = f"DRYRUN=1 HF_HOME=... HF_DATASETS_CACHE=... CONTAINER= ACCOUNT= PARTITION= ./tools/launch {' '.join(scripts)}"
    result = subprocess.run(
        command,
        shell=True,
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, (
        f"Command failed with exit code {result.returncode}\n"
        f"STDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
    )

    stdout_lines = result.stdout.strip().splitlines()
    assert stdout_lines, "Command produced no output"
    last_line = stdout_lines[-1]
    assert last_line.startswith("[INFO]: Total GPU hours:"), (
        f"Last line of output was not as expected: '{last_line}'"
    )

    per_run = float(last_line.split(":")[-1].strip())
    per_week = per_run * budget["runs_per_week"]
    tracker.track(f"gpu_hours_per_week_{suite}_{sku}", per_week)

    if per_week > budget["max_gpu_hours_per_week"]:
        # Surface the most expensive tests so the author can see what to trade
        # away, rather than only being told the lane is full.
        costs = sorted(
            (
                (int(hours), script)
                for hours, script in re.findall(
                    r"^\[INFO\]: (\d+) GPUhrs to run (\S+)$", result.stdout, re.M
                )
            ),
            reverse=True,
        )
        worst = "\n".join(
            f"  {hours:>6} GPU-h/run  {script}" for hours, script in costs[:10]
        )
        raise AssertionError(
            f"{suite} ({sku}) needs {per_week:.0f} GPU-hours/week "
            f"({per_run:.0f} per run x {budget['runs_per_week']} runs/week), over its "
            f"{budget['max_gpu_hours_per_week']} budget.\n"
            f"Retire or shrink a test rather than raising the budget. "
            f"Most expensive tests in this lane:\n{worst}"
        )


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
