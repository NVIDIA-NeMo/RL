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
# `runs_per_week` mirrors the nemo-ci pipeline schedules (dl/JoC/nemo-ci). Keep
# these in sync if a schedule changes:
#   "NeMo RL Nightly tests"               0 2 * * *  -> nightly, nightly_gb200
#   "NeMo RL Nightly MBridge Bump tests"  0 2 * * *  -> nightly_mcore(_gb200)
#   "NeMo RL Weekly Release Tests"        0 4 * * 6  -> release(_gb200)
#   "NeMo RL Weekly Perf Tests"           0 4 * * 6  -> performance(_gb200)
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
    ("nightly_mcore", "h100"): {"runs_per_week": 7, "max_gpu_hours_per_week": 3_800},
    ("nightly_mcore", "gb200"): {"runs_per_week": 7, "max_gpu_hours_per_week": 400},
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
