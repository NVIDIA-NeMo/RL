#!/usr/bin/env python3
"""Apply the new TestConfig pattern to all test files.

Changes applied:
1. Remove manual marker decorators
2. Simplify TestConfig to use YAML + overrides
3. Split test into local and slurm methods
4. Update imports
"""

import re
from pathlib import Path
from typing import Any, Dict


def extract_test_metadata(content: str, file_path: Path) -> Dict[str, Any]:
    """Extract metadata from old TestConfig to build new one."""
    metadata = {}

    # Extract test_name from config
    match = re.search(r'test_name\s*=\s*["\']([^"\']+)["\']', content)
    if match:
        metadata["test_name"] = match.group(1)

    # Extract algorithm
    match = re.search(r'algorithm\s*=\s*["\']([^"\']+)["\']', content)
    if match:
        metadata["algorithm"] = match.group(1)

    # Extract test_suites
    match = re.search(r"test_suites\s*=\s*\[([^\]]+)\]", content)
    if match:
        suites_str = match.group(1)
        suites = [s.strip().strip('"').strip("'") for s in suites_str.split(",")]
        metadata["test_suites"] = suites
    else:
        metadata["test_suites"] = ["nightly"]

    # Extract time_limit_minutes
    match = re.search(r"time_limit_minutes\s*=\s*(\d+)", content)
    if match:
        metadata["time_limit_minutes"] = int(match.group(1))
    else:
        metadata["time_limit_minutes"] = 120

    # Extract max_steps to create override
    match = re.search(r"max_steps\s*=\s*(\d+)", content)
    if match:
        max_steps = int(match.group(1))
        metadata["max_steps"] = max_steps

    return metadata


def build_new_config(metadata: Dict[str, Any]) -> str:
    """Build the new TestConfig code."""
    test_name = metadata["test_name"]
    algorithm = metadata["algorithm"]
    test_suites = metadata["test_suites"]
    time_limit = metadata["time_limit_minutes"]

    # Build overrides if max_steps was specified
    overrides = {}
    if "max_steps" in metadata:
        overrides[f"{algorithm}.max_num_steps"] = metadata["max_steps"]

    config_lines = [
        "    # Test configuration",
        "    config = TestConfig(",
        "        # Test metadata",
        f'        test_name="{test_name}",',
        f'        algorithm="{algorithm}",',
        f"        test_suites={test_suites},",
        f"        time_limit_minutes={time_limit},",
    ]

    if overrides:
        config_lines.append("")
        config_lines.append("        # Configuration overrides")
        config_lines.append("        overrides={")
        for key, value in overrides.items():
            config_lines.append(f'            "{key}": {value},')
        config_lines.append("        }")

    config_lines.append("    )")

    return "\n".join(config_lines)


def update_test_file(file_path: Path):
    """Update a single test file to the new pattern."""
    print(f"\n{'=' * 80}")
    print(f"Updating: {file_path.name}")
    print(f"{'=' * 80}")

    content = file_path.read_text()

    # Check if already updated
    if "def test_training_runs_successfully_local(self):" in content:
        print("  ✓ Already has split local/slurm tests")
        return

    # Extract metadata from old config
    metadata = extract_test_metadata(content, file_path)
    print(f"  Extracted metadata: {metadata}")

    # Step 1: Remove marker decorators
    print("  1. Removing marker decorators...")
    # Find class definition
    class_match = re.search(r"((?:@pytest\.mark\.\w+\s*\n)+)(class \w+:)", content)
    if class_match:
        markers = class_match.group(1)
        class_def = class_match.group(2)
        # Replace markers + class with just class
        content = content.replace(markers + class_def, class_def)
        print(f"     Removed {len(markers.split(chr(10))) - 1} markers")

    # Step 2: Update imports
    print("  2. Updating imports...")
    if "from tests.test_suites.base_config import" in content:
        content = re.sub(
            r"from tests\.test_suites\.base_config import [^\n]+",
            "from tests.test_suites.base_config import TestConfig, run_command, run_via_slurm",
            content,
        )

    # Step 3: Replace TestConfig
    print("  3. Replacing TestConfig...")
    # Find the old TestConfig block
    config_pattern = r"(\s+)# Test configuration\n\s+config = TestConfig\([^)]+\)"
    config_match = re.search(config_pattern, content, re.DOTALL)

    if config_match:
        indent = config_match.group(1)
        old_config = config_match.group(0)
        new_config = build_new_config(metadata)
        content = content.replace(old_config, new_config)
        print("     Replaced TestConfig")
    else:
        print("     WARNING: Could not find TestConfig block")

    # Step 4: Split test method into local and slurm
    print("  4. Splitting test into local and slurm methods...")

    # Find the test method
    test_method_pattern = r"(\s+)def test_training_runs_successfully\(self, request, use_slurm\):\n(.*?)(?=\n\s+def |\n\s*\n\s*$|\Z)"
    test_match = re.search(test_method_pattern, content, re.DOTALL)

    if test_method_pattern and "use_slurm" in content:
        # Build new test methods
        new_methods = '''    def test_training_runs_successfully_local(self):
        """Test that training completes successfully when run locally."""
        cmd = self.config.build_command()
        return_code = run_command(cmd, self.config.run_log_path, cwd=self.config.project_root)

        assert return_code == 0, f"Training failed with return code {return_code}. Check {self.config.run_log_path}"

    def test_training_runs_successfully_slurm(self, request):
        """Test that training completes successfully when run via Slurm."""
        return_code = run_via_slurm(self.config, request.node.nodeid)

        assert return_code == 0, f"Training failed with return code {return_code}. Check {self.config.run_log_path}"
'''

        # Find and replace the old test method
        old_method_pattern = r"    def test_training_runs_successfully\(self, request, use_slurm\):.*?(?=\n\n|\Z)"
        content = re.sub(
            old_method_pattern, new_methods.rstrip(), content, flags=re.DOTALL
        )
        print("     Split into local and slurm methods")
    else:
        print("     WARNING: Could not find test method to split")

    # Step 5: Clean up docstring if needed
    content = re.sub(r'"""Test DPO[^"]*sh"""', "", content)

    # Write back
    file_path.write_text(content)
    print("  ✓ Updated successfully")


def main():
    """Update all test files."""
    test_suites_dir = Path("tests/test_suites/llm")

    # Find all test files
    test_files = sorted(test_suites_dir.glob("*/test_*.py"))

    print(f"Found {len(test_files)} test files to update")
    print("=" * 80)

    for test_file in test_files:
        try:
            update_test_file(test_file)
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            import traceback

            traceback.print_exc()

    print(f"\n{'=' * 80}")
    print(f"Completed! Updated {len(test_files)} test files")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
