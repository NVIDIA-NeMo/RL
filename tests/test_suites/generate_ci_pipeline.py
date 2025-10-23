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
- Automatic test discovery using AST parsing
- Filtering by test suites (e.g., --test-suites release,nightly)
- Filtering by algorithm (e.g., --algorithm grpo,dpo)
- Filtering by model class (e.g., --model-class llm,vlm)
- Each job runs tests for a single class
- Configurable job timeout based on test config
- Supports pytest filters and markers

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
import ast
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set

import yaml


@dataclass
class TestClassInfo:
    """Information about a discovered test class."""

    class_name: str
    file_path: Path
    test_name: str
    algorithm: str
    model_class: str
    test_suites: List[str]
    time_limit_minutes: int
    num_nodes: int
    num_gpus_per_node: int

    @property
    def relative_file_path(self) -> Path:
        """Get the file path relative to project root."""
        try:
            return self.file_path.relative_to(Path.cwd())
        except ValueError:
            # Already relative
            return self.file_path

    @property
    def pytest_path(self) -> str:
        """Get the pytest path to run this specific class."""
        return f"{self.relative_file_path}::{self.class_name}"

    @property
    def job_name(self) -> str:
        """Generate a unique job name for this test."""
        return f"{self.test_name}"

    @property
    def tags(self) -> List[str]:
        """Generate tags for this job based on metadata."""
        tags = []
        tags.append(self.algorithm)
        tags.append(self.model_class)
        tags.extend(self.test_suites)
        return tags


class TestClassDiscovery:
    """Discovers test classes by parsing Python files with AST."""

    def __init__(self, tests_dir: Path):
        self.tests_dir = tests_dir

    def discover_test_files(self) -> List[Path]:
        """Find all test_*.py files in the tests directory."""
        test_files = list(self.tests_dir.glob("**/test_*.py"))
        # Filter out __init__.py and other non-test files
        test_files = [f for f in test_files if f.name.startswith("test_")]
        return test_files

    def extract_config_dict(self, node: ast.AST) -> Optional[Dict]:
        """Extract configuration dictionary from AST Call node.

        Args:
            node: AST node representing the config assignment

        Returns:
            Dictionary with extracted config values, or None if extraction fails
        """
        if not isinstance(node, ast.Call):
            return None

        config = {}

        # Extract keyword arguments
        for keyword in node.keywords:
            if keyword.arg is None:
                continue

            key = keyword.arg
            value = keyword.value

            # Extract different value types
            if isinstance(value, ast.Constant):
                config[key] = value.value
            elif isinstance(value, ast.List):
                # Extract list values
                config[key] = [
                    elem.value for elem in value.elts if isinstance(elem, ast.Constant)
                ]
            elif isinstance(value, ast.Dict):
                # For now, just mark that overrides exist
                config[key] = "dict"
            elif isinstance(value, ast.Name):
                config[key] = value.id

        return config

    def parse_test_class(self, file_path: Path) -> List[TestClassInfo]:
        """Parse a test file and extract test class information.

        Args:
            file_path: Path to the test file

        Returns:
            List of TestClassInfo objects for each test class found
        """
        try:
            with open(file_path, "r") as f:
                tree = ast.parse(f.read(), filename=str(file_path))
        except Exception as e:
            print(f"Warning: Failed to parse {file_path}: {e}", file=sys.stderr)
            return []

        test_classes = []

        for node in ast.walk(tree):
            if not isinstance(node, ast.ClassDef):
                continue

            # Check if class has a 'config' attribute
            config_node = None
            for item in node.body:
                if isinstance(item, ast.Assign):
                    for target in item.targets:
                        if isinstance(target, ast.Name) and target.id == "config":
                            config_node = item.value
                            break

            if config_node is None:
                continue

            # Extract config information
            config_dict = self.extract_config_dict(config_node)
            if config_dict is None:
                continue

            # Create TestClassInfo if we have required fields
            required_fields = ["test_name", "algorithm", "model_class"]
            if not all(field in config_dict for field in required_fields):
                print(
                    f"Warning: Class {node.name} in {file_path} missing required config fields",
                    file=sys.stderr,
                )
                continue

            test_class_info = TestClassInfo(
                class_name=node.name,
                file_path=file_path,
                test_name=config_dict["test_name"],
                algorithm=config_dict["algorithm"],
                model_class=config_dict["model_class"],
                test_suites=config_dict.get("test_suites", ["nightly"]),
                time_limit_minutes=config_dict.get("time_limit_minutes", 120),
                num_nodes=config_dict.get("num_nodes", 1),
                num_gpus_per_node=config_dict.get("num_gpus_per_node", 8),
            )

            test_classes.append(test_class_info)

        return test_classes

    def discover_all_tests(self) -> List[TestClassInfo]:
        """Discover all test classes in the tests directory."""
        test_files = self.discover_test_files()
        all_tests = []

        for test_file in test_files:
            test_classes = self.parse_test_class(test_file)
            all_tests.extend(test_classes)

        return all_tests


class GitLabCIPipelineGenerator:
    """Generates GitLab CI pipeline YAML from test classes."""

    def __init__(
        self,
        tests: List[TestClassInfo],
        base_image: str = "python:3.10",
        pytest_args: Optional[List[str]] = None,
        module: str = "nemo_rl",
        extends: Optional[str] = None,
    ):
        self.tests = tests
        self.base_image = base_image
        self.pytest_args = pytest_args or []
        self.module = module
        self.extends = extends

    def generate_job(self, test: TestClassInfo) -> Dict:
        """Generate a GitLab CI job configuration for a single test class.

        Args:
            test: Test class information

        Returns:
            Dictionary with job configuration
        """
        # Convert time_limit_minutes to "MM:SS" format
        time_str = f"{test.time_limit_minutes}:00"

        job = {
            "stage": "test",
            "variables": {
                "MODULE": self.module,
                "TEST_NAME": test.class_name,
                "TIME": time_str,
                "NUM_NODES": test.num_nodes,
                "ALGORITHM": test.algorithm,
                "MODEL_CLASS": test.model_class,
                "NUM_GPUS": test.num_gpus_per_node,
            },
            "script": [
                "echo 'Running test: ${TEST_NAME}'",
                "echo 'Algorithm: ${ALGORITHM}'",
                "echo 'Model class: ${MODEL_CLASS}'",
                f"pytest {' '.join(self.pytest_args)} {test.pytest_path} -v",
            ],
        }

        # Add extends field if specified
        if self.extends:
            job["extends"] = self.extends

        return job

    def generate_pipeline(self) -> Dict:
        """Generate the complete GitLab CI pipeline configuration.

        Returns:
            Dictionary with complete pipeline configuration
        """
        pipeline = {
            "stages": ["test"],
            "default": {
                "image": self.base_image,
                "before_script": [
                    "pip install uv",
                    "uv sync --all-extras",
                ],
            },
        }

        # Generate a job for each test class
        for test in self.tests:
            job_name = test.job_name
            pipeline[job_name] = self.generate_job(test)

        return pipeline

    def generate_yaml(self) -> str:
        """Generate GitLab CI pipeline YAML string.

        Returns:
            YAML string representation of the pipeline
        """
        pipeline = self.generate_pipeline()

        # Convert to YAML with nice formatting
        yaml_str = yaml.dump(
            pipeline, default_flow_style=False, sort_keys=False, width=120
        )

        # Add 1 blank line between jobs
        # Jobs are top-level keys that are not 'stages' or 'default'
        lines = yaml_str.split("\n")
        result_lines = []
        special_keys = {"stages:", "default:"}

        for i, line in enumerate(lines):
            # Check if this line is a job name (starts at column 0 and not a special key)
            if (
                line
                and not line[0].isspace()
                and line not in special_keys
                and ":" in line
            ):
                # This is a job, add 1 blank line before it (unless it's the first line)
                if result_lines:
                    result_lines.append("")
            result_lines.append(line)

        yaml_str = "\n".join(result_lines)

        # Add header comment
        header = f"""# GitLab CI Pipeline
# Auto-generated from pytest test classes
# Total jobs: {len(self.tests)}

"""
        return header + yaml_str


def filter_tests(
    tests: List[TestClassInfo],
    test_suites: Optional[Set[str]] = None,
    algorithms: Optional[Set[str]] = None,
    model_classes: Optional[Set[str]] = None,
) -> List[TestClassInfo]:
    """Filter tests based on criteria.

    Args:
        tests: List of test classes to filter
        test_suites: Set of test suite names to include (e.g., {"release", "nightly"})
        algorithms: Set of algorithm names to include (e.g., {"grpo", "dpo"})
        model_classes: Set of model class names to include (e.g., {"llm", "vlm"})

    Returns:
        Filtered list of test classes
    """
    filtered = tests

    if test_suites:
        filtered = [
            t for t in filtered if any(suite in test_suites for suite in t.test_suites)
        ]

    if algorithms:
        filtered = [t for t in filtered if t.algorithm in algorithms]

    if model_classes:
        filtered = [t for t in filtered if t.model_class in model_classes]

    return filtered


def main():
    parser = argparse.ArgumentParser(
        description="Generate GitLab CI pipeline from pytest test classes",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--tests-dir",
        type=Path,
        default=Path("tests/test_suites/tests"),
        help="Directory containing test files (default: tests/test_suites/tests)",
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
        default="python:3.10",
        help="Docker image to use for jobs (default: python:3.10)",
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
        "--list-tests",
        action="store_true",
        help="List discovered tests and exit (don't generate pipeline)",
    )

    args = parser.parse_args()

    # Discover tests
    print(f"Discovering tests in {args.tests_dir}...", file=sys.stderr)
    discovery = TestClassDiscovery(args.tests_dir)
    all_tests = discovery.discover_all_tests()
    print(f"Found {len(all_tests)} test classes", file=sys.stderr)

    # Apply filters
    test_suites = set(args.test_suites.split(",")) if args.test_suites else None
    algorithms = set(args.algorithm.split(",")) if args.algorithm else None
    model_classes = set(args.model_class.split(",")) if args.model_class else None

    filtered_tests = filter_tests(
        all_tests,
        test_suites=test_suites,
        algorithms=algorithms,
        model_classes=model_classes,
    )

    print(f"After filtering: {len(filtered_tests)} test classes", file=sys.stderr)

    # List tests if requested
    if args.list_tests:
        print("\nDiscovered tests:", file=sys.stderr)
        for test in filtered_tests:
            print(
                f"  - {test.class_name} ({test.test_name}) "
                f"[{test.algorithm}, {test.model_class}, suites: {', '.join(test.test_suites)}]",
                file=sys.stderr,
            )
        return

    if not filtered_tests:
        print("No tests match the specified filters", file=sys.stderr)
        sys.exit(1)

    # Generate pipeline
    pytest_args = args.pytest_args.split() if args.pytest_args else []
    generator = GitLabCIPipelineGenerator(
        filtered_tests,
        base_image=args.base_image,
        pytest_args=pytest_args,
        module=args.module,
        extends=args.extends,
    )
    yaml_output = generator.generate_yaml()

    # Write output
    if args.output:
        args.output.write_text(yaml_output)
        print(f"Pipeline written to {args.output}", file=sys.stderr)
    else:
        print(yaml_output)


if __name__ == "__main__":
    main()
