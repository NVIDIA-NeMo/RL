#!/usr/bin/env python3
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

"""Generate GitLab CI pipeline YAML from pytest test classes.

This script discovers all test classes in tests/test_suites/ and generates
a GitLab CI pipeline where each job runs tests for a single test class.

Features:
- Automatic test discovery using pytest
- Filtering by test suites (e.g., --test-suites release,nightly)
- Filtering by algorithm (e.g., --algorithm grpo,dpo)
- Filtering by model class (e.g., --model-class llm,vlm)
- Configurable job timeout based on test config
- Supports pytest filters and markers
- support job dependencies and stages from test class job_dependencies (JobDependencies)
- support job dependencies using pytest-ordering markers

Example usage:
    # Generate pipeline for all tests
    python generate_ci_pipeline.py

    # Generate pipeline for only "release" test suite
    python generate_ci_pipeline.py --test-suites release

    # Generate pipeline for GRPO tests only
    python generate_ci_pipeline.py --algorithm grpo

    # Generate pipeline for LLM tests in release suite
    python generate_ci_pipeline.py --model-class llm --test-suites release

    # Output to file
    python generate_ci_pipeline.py --output .gitlab-ci.yml
"""

import argparse
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import pytest
import yaml


def format_yaml_with_spacing(yaml_str: str) -> str:
    """Add blank lines between top-level jobs in YAML output.

    Args:
        yaml_str: YAML string output from yaml.dump()

    Returns:
        Formatted YAML with blank lines between jobs
    """
    lines = yaml_str.split("\n")
    formatted_lines = []
    seen_first_job = False
    in_stages_list = False

    for line in lines:
        # Track if we're in the stages list
        if line.startswith("stages:"):
            in_stages_list = True
        elif (
            in_stages_list
            and line
            and not line.startswith("-")
            and not line.startswith(" ")
        ):
            # End of stages list, add blank line before first job
            in_stages_list = False
            formatted_lines.append("")

        # Detect top-level keys (jobs) - lines that start without indentation and contain ':'
        # But skip special keys like 'stages:'
        is_top_level_job = (
            line
            and not line.startswith(" ")
            and not line.startswith("-")
            and ":" in line
            and not line.startswith("stages:")
        )

        # Add blank line before job (except for first job)
        if is_top_level_job and not in_stages_list:
            if seen_first_job:
                formatted_lines.append("")
            seen_first_job = True

        formatted_lines.append(line)

    return "\n".join(formatted_lines)


def _build_pytest_command(
    test_mode: str,
    test_file: str,
    pytest_args: str,
    test_filter: Optional[str] = None,
    is_training: bool = False,
    train_mode: str = "local",
) -> str:
    """Build pytest command with filters.

    Args:
        test_mode: Test mode (local/slurm/all)
        test_file: Path to test file
        pytest_args: Additional pytest arguments
        test_filter: Test name filter for -k option
        is_training: Whether this is a training stage
        train_mode: Training mode (local/slurm/bash)

    Returns:
        Complete pytest command string
    """
    pytest_cmd_parts = ["pytest"]

    # Add test mode marker only for training stages
    if is_training:
        if train_mode == "local":
            pytest_cmd_parts.extend(["-m", "runner_local"])
        elif train_mode == "slurm":
            pytest_cmd_parts.extend(["-m", "runner_slurm"])
    # For non-training stages (validation, etc.), don't add runner markers

    # Add test filter
    if test_filter:
        pytest_cmd_parts.extend(["-k", test_filter])

    # Add test file
    pytest_cmd_parts.append(f"tests/test_suites/test_cases/{test_file}")

    # Add additional args
    if pytest_args:
        pytest_cmd_parts.append(pytest_args)

    return " ".join(pytest_cmd_parts)


def _build_pytest_commands(
    test_mode: str,
    test_file: str,
    pytest_args: str,
    test_names: List[str],
    is_training: bool = False,
    train_mode: str = "local",
) -> List[str]:
    """Build separate pytest commands for each test.

    Args:
        test_mode: Test mode (local/slurm/all)
        test_file: Path to test file
        pytest_args: Additional pytest arguments
        test_names: List of test names to run
        is_training: Whether this is a training stage
        train_mode: Training mode (local/slurm/bash)

    Returns:
        List of pytest command strings
    """
    commands = []
    for test_name in test_names:
        cmd = _build_pytest_command(
            test_mode, test_file, pytest_args, test_name, is_training, train_mode
        )
        commands.append(cmd)
    return commands


def _build_job(
    config: Any,
    commands: List[str],
    job_name: str,
    base_image: Optional[str],
    module: str,
    extends: Optional[str],
    stage: Optional[str] = None,
) -> Dict[str, Any]:
    """Build GitLab CI job configuration.

    Args:
        config: Test configuration object
        commands: List of commands to run
        job_name: Name for the job
        base_image: Docker image (optional)
        module: Module name for MODULE variable
        extends: Base job to extend (optional)
        stage: Stage name (optional)

    Returns:
        Job configuration dictionary
    """
    # Join commands with && for TEST_SCRIPT and add quotes
    test_script = " && ".join(commands)

    # Format TIME as "minutes:0"
    time_string = f"{config.time_limit_minutes}:0"

    job = {
        "variables": {
            "MODULE": module,
            "TEST_NAME": job_name,
            "TEST_SCRIPT": f'"{test_script}"',
            "TIME": time_string,
            "NUM_NODES": config.num_nodes,
        },
        "script": commands,
        "timeout": f"{config.time_limit_minutes}m",
    }

    if base_image:
        job["image"] = base_image

    if stage:
        job["stage"] = stage

    if extends:
        job["extends"] = extends

    return job


def discover_tests(tests_dir: Path, filters: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Discover tests using pytest and extract metadata.

    Args:
        tests_dir: Directory containing test files
        filters: Dictionary of filters to apply

    Returns:
        List of test metadata dictionaries
    """
    # Build pytest collection command
    pytest_args = [str(tests_dir), "--collect-only", "-q"]

    # Add filters
    if filters.get("test_suites"):
        suites = filters["test_suites"].split(",")
        # Collect for any of the suites
        pytest_args.extend(["-m", " or ".join([f"suite_{s}" for s in suites])])

    if filters.get("algorithm"):
        algos = filters["algorithm"].split(",")
        pytest_args.extend(["-m", " or ".join([f"algo_{a}" for a in algos])])

    if filters.get("model_class"):
        classes = filters["model_class"].split(",")
        pytest_args.extend(["-m", " or ".join([f"class_{c}" for c in classes])])

    # Collect tests
    class CollectPlugin:
        def __init__(self):
            self.collected_items = []

        def pytest_collection_finish(self, session):
            self.collected_items = session.items

    plugin = CollectPlugin()
    pytest.main(pytest_args, plugins=[plugin])

    # Extract metadata from collected items
    tests_by_class = defaultdict(
        lambda: {
            "test_class": None,
            "test_file": None,
            "config": None,
            "job_dependencies": None,
            "tests": [],
        }
    )

    for item in plugin.collected_items:
        if not hasattr(item, "cls") or item.cls is None:
            continue

        test_class = item.cls
        class_name = test_class.__name__

        # Get config and job_dependencies from test class
        if class_name not in tests_by_class:
            tests_by_class[class_name]["test_class"] = class_name
            tests_by_class[class_name]["test_file"] = str(
                Path(item.path).relative_to(tests_dir.absolute())
            )

            if hasattr(test_class, "config"):
                tests_by_class[class_name]["config"] = test_class.config

            if hasattr(test_class, "job_dependencies"):
                tests_by_class[class_name]["job_dependencies"] = (
                    test_class.job_dependencies
                )

        # Extract test metadata
        test_info = {
            "name": item.name,
            "nodeid": item.nodeid,
            "stage": None,
            "job_group": None,
            "order": None,
        }

        # Extract markers
        for marker in item.iter_markers():
            if marker.name == "stage" and marker.args:
                test_info["stage"] = marker.args[0]
            elif marker.name == "job_group" and marker.args:
                test_info["job_group"] = marker.args[0]
            elif marker.name == "order" and marker.args:
                test_info["order"] = marker.args[0]

        tests_by_class[class_name]["tests"].append(test_info)

    return list(tests_by_class.values())


def generate_flat_pipeline(
    tests_data: List[Dict[str, Any]],
    base_image: Optional[str],
    pytest_args: str,
    module: str,
    extends: Optional[str],
    test_mode: str,
) -> Dict[str, Any]:
    """Generate flat pipeline (one job per test class).

    Args:
        tests_data: List of test metadata
        base_image: Docker image to use (optional)
        pytest_args: Additional pytest arguments
        module: Module name for MODULE variable
        extends: Base job to extend
        test_mode: Test mode (local/slurm/all)

    Returns:
        GitLab CI pipeline dictionary
    """
    pipeline = {}

    for test_data in tests_data:
        config = test_data["config"]
        if not config:
            continue

        job_name = config.test_name
        test_file = test_data["test_file"]

        # Build pytest command (as a list for consistency)
        pytest_cmd = _build_pytest_command(test_mode, test_file, pytest_args)
        commands = [pytest_cmd]

        # Build job
        job = _build_job(config, commands, job_name, base_image, module, extends)

        pipeline[job_name] = job

    return pipeline


def generate_stages_pipeline(
    tests_data: List[Dict[str, Any]],
    base_image: Optional[str],
    pytest_args: str,
    module: str,
    extends: Optional[str],
    test_mode: str,
    train_mode: str = "local",
) -> Dict[str, Any]:
    """Generate pipeline with stages and dependencies.

    Args:
        tests_data: List of test metadata
        base_image: Docker image to use (optional)
        pytest_args: Additional pytest arguments
        module: Module name for MODULE variable
        extends: Base job to extend
        test_mode: Test mode (local/slurm/all)
        train_mode: Training mode (local/slurm/bash)

    Returns:
        GitLab CI pipeline dictionary
    """
    pipeline = {"stages": []}
    stages_seen: Set[str] = set()

    for test_data in tests_data:
        config = test_data["config"]
        job_deps = test_data["job_dependencies"]

        if not config:
            continue

        test_name = config.test_name
        test_file = test_data["test_file"]

        # Group tests by stage
        tests_by_stage = defaultdict(list)
        for test in test_data["tests"]:
            stage = test.get("stage") or "training"
            tests_by_stage[stage].append(test)

        # Create jobs for each stage
        for stage, stage_tests in tests_by_stage.items():
            if stage not in stages_seen:
                pipeline["stages"].append(stage)
                stages_seen.add(stage)

            # Group by job_group
            tests_by_job_group = defaultdict(list)
            for test in stage_tests:
                job_group = test.get("job_group") or "default"
                tests_by_job_group[job_group].append(test)

            # Create job for each job group
            for job_group, group_tests in tests_by_job_group.items():
                job_name = f"{test_name}:{stage}:{job_group}"

                # Sort tests by order marker
                sorted_tests = sorted(
                    group_tests,
                    key=lambda t: t.get("order") if t.get("order") is not None else 0,
                )

                # Build commands based on stage and train_mode
                if train_mode == "bash" and stage == "training":
                    # Use raw training command instead of pytest
                    commands = [config.build_command_string()]
                else:
                    # Build separate pytest commands for each test
                    test_names = [t["name"] for t in sorted_tests]
                    is_training = stage == "training"
                    commands = _build_pytest_commands(
                        test_mode,
                        test_file,
                        pytest_args,
                        test_names,
                        is_training,
                        train_mode,
                    )

                # Build job
                job = _build_job(
                    config, commands, job_name, base_image, module, extends, stage
                )

                # Add dependencies from JobDependencies
                if job_deps:
                    # Stage dependencies
                    if stage in job_deps.stages:
                        stage_deps = job_deps.stages[stage]
                        if stage_deps.get("depends_on"):
                            job["needs"] = [
                                f"{test_name}:{dep_stage}:*"
                                for dep_stage in stage_deps["depends_on"]
                            ]

                    # Job group dependencies
                    if job_group in job_deps.job_groups:
                        group_deps = job_deps.job_groups[job_group]
                        if group_deps.get("depends_on"):
                            if "needs" not in job:
                                job["needs"] = []
                            job["needs"].extend(
                                [
                                    f"{test_name}:{stage}:{dep_group}"
                                    for dep_group in group_deps["depends_on"]
                                ]
                            )

                pipeline[job_name] = job

    return pipeline


def generate_parent_child_pipeline(
    tests_data: List[Dict[str, Any]],
    base_image: Optional[str],
    pytest_args: str,
    module: str,
    extends: Optional[str],
    test_mode: str,
    train_mode: str = "local",
) -> Dict[str, Dict[str, Any]]:
    """Generate parent-child pipeline structure.

    Creates a parent pipeline with trigger jobs and separate child pipelines
    for each test class.

    Args:
        tests_data: List of test metadata
        base_image: Docker image to use (optional)
        pytest_args: Additional pytest arguments
        module: Module name for MODULE variable
        extends: Base job to extend
        test_mode: Test mode (local/slurm/all)
        train_mode: Training mode (local/slurm/bash)

    Returns:
        Dictionary with 'parent' and 'children' keys containing pipeline YAML
    """
    parent_pipeline = {}
    child_pipelines = {}

    for test_data in tests_data:
        config = test_data["config"]
        if not config:
            continue

        test_name = config.test_name
        child_file = f".gitlab-ci-{test_name}.yml"

        # Create parent trigger job
        parent_pipeline[test_name] = {
            "trigger": {
                "include": child_file,
                "strategy": "depend",
            }
        }

        # Generate child pipeline using stages mode
        child_tests_data = [test_data]  # Only this test class
        child_pipeline = generate_stages_pipeline(
            child_tests_data,
            base_image,
            pytest_args,
            module,
            extends,
            test_mode,
            train_mode,
        )

        child_pipelines[child_file] = child_pipeline

    return {"parent": parent_pipeline, "children": child_pipelines}


def main():
    parser = argparse.ArgumentParser(
        description="Generate GitLab CI pipeline from pytest test classes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--tests-dir",
        type=Path,
        default=Path("tests/test_suites/test_cases"),
        help="Directory containing test files (default: tests/test_suites/test_cases)",
    )

    parser.add_argument(
        "--output",
        type=Path,
        help="Output file path (default: stdout)",
    )

    parser.add_argument(
        "--test-suites",
        type=str,
        help="Comma-separated list of test suites to include (e.g., 'release,nightly')",
    )

    parser.add_argument(
        "--algorithm",
        type=str,
        help="Comma-separated list of algorithms to include (e.g., 'grpo,dpo,sft')",
    )

    parser.add_argument(
        "--model-class",
        type=str,
        help="Comma-separated list of model classes to include (e.g., 'llm,vlm')",
    )

    parser.add_argument(
        "--base-image",
        type=str,
        default=None,
        help="Docker image to use for jobs (optional, not set by default)",
    )

    parser.add_argument(
        "--pytest-args",
        type=str,
        default="",
        help="Additional pytest arguments (e.g., '--markers=slow -x')",
    )

    parser.add_argument(
        "--module",
        type=str,
        default="nemo_rl",
        help="Module name to set in MODULE variable (default: nemo_rl)",
    )

    parser.add_argument(
        "--extends",
        type=str,
        default=None,
        help="Add 'extends' field to each job (e.g., '.base_job')",
    )

    parser.add_argument(
        "--test-mode",
        type=str,
        choices=["local", "slurm", "all"],
        default="local",
        help="Test mode: 'local' (run test_train_local + other tests), "
        "'slurm' (run test_train_slurm only), "
        "'all' (run both test_train_local and test_train_slurm + other tests) "
        "(default: local)",
    )

    parser.add_argument(
        "--pipeline-mode",
        type=str,
        choices=["flat", "parent-child", "stages"],
        default="flat",
        help="Pipeline generation mode: "
        "'flat' (one job per test class, default), "
        "'parent-child' (parent pipeline + child pipelines per test class), "
        "'stages' (single pipeline with train/validate stages and dependencies)",
    )

    parser.add_argument(
        "--list-tests",
        action="store_true",
        help="List discovered tests and exit (don't generate pipeline)",
    )

    parser.add_argument(
        "--train-mode",
        type=str,
        choices=["local", "slurm", "bash"],
        default="local",
        help="Training mode: 'local' (pytest local test functions), "
        "'slurm' (pytest slurm test functions), "
        "'bash' (raw training command) "
        "(default: local)",
    )

    args = parser.parse_args()

    # Build filters
    filters = {}
    if args.test_suites:
        filters["test_suites"] = args.test_suites
    if args.algorithm:
        filters["algorithm"] = args.algorithm
    if args.model_class:
        filters["model_class"] = args.model_class

    # Discover tests
    tests_data = discover_tests(args.tests_dir, filters)

    if not tests_data:
        print("No tests found matching the specified filters.", file=sys.stderr)
        sys.exit(1)

    # List tests and exit if requested
    if args.list_tests:
        print(f"Found {len(tests_data)} test classes:")
        for test_data in tests_data:
            config = test_data["config"]
            if config:
                print(f"  - {config.test_name}")
                print(f"      Algorithm: {config.algorithm}")
                print(f"      Model class: {config.model_class}")
                print(f"      Test suites: {', '.join(config.test_suites)}")
                print(f"      Tests: {len(test_data['tests'])}")

                # Show stages and job groups
                stages = set()
                job_groups = set()
                for test in test_data["tests"]:
                    if test.get("stage"):
                        stages.add(test["stage"])
                    if test.get("job_group"):
                        job_groups.add(test["job_group"])

                if stages:
                    print(f"      Stages: {', '.join(sorted(stages))}")
                if job_groups:
                    print(f"      Job groups: {', '.join(sorted(job_groups))}")
                print()
        sys.exit(0)

    # Generate pipeline based on mode
    if args.pipeline_mode == "flat":
        pipeline = generate_flat_pipeline(
            tests_data,
            args.base_image,
            args.pytest_args,
            args.module,
            args.extends,
            args.test_mode,
        )
    elif args.pipeline_mode == "stages":
        pipeline = generate_stages_pipeline(
            tests_data,
            args.base_image,
            args.pytest_args,
            args.module,
            args.extends,
            args.test_mode,
            args.train_mode,
        )
    elif args.pipeline_mode == "parent-child":
        result = generate_parent_child_pipeline(
            tests_data,
            args.base_image,
            args.pytest_args,
            args.module,
            args.extends,
            args.test_mode,
            args.train_mode,
        )

        # For parent-child, we need to write multiple files
        parent_pipeline = result["parent"]
        child_pipelines = result["children"]

        # Determine output directory
        if args.output:
            output_dir = args.output.parent
            parent_file = args.output
        else:
            output_dir = Path(".")
            parent_file = output_dir / ".gitlab-ci.yml"

        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)

        # Write parent pipeline
        parent_yaml = yaml.dump(
            parent_pipeline, default_flow_style=False, sort_keys=False
        )
        parent_yaml = format_yaml_with_spacing(parent_yaml)
        parent_file.write_text(parent_yaml)
        print(f"Parent pipeline written to {parent_file}")

        # Write child pipelines
        for child_file_name, child_pipeline in child_pipelines.items():
            child_file = output_dir / child_file_name
            child_yaml = yaml.dump(
                child_pipeline, default_flow_style=False, sort_keys=False
            )
            child_yaml = format_yaml_with_spacing(child_yaml)
            child_file.write_text(child_yaml)
            print(f"  Child pipeline written to {child_file}")

        sys.exit(0)
    else:
        print(f"Unknown pipeline mode: {args.pipeline_mode}", file=sys.stderr)
        sys.exit(1)

    # Output pipeline (for flat and stages modes)
    yaml_output = yaml.dump(pipeline, default_flow_style=False, sort_keys=False)
    yaml_output = format_yaml_with_spacing(yaml_output)

    if args.output:
        # Create output directory if it doesn't exist
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(yaml_output)
        print(f"Pipeline written to {args.output}")
    else:
        print(yaml_output)


if __name__ == "__main__":
    main()
