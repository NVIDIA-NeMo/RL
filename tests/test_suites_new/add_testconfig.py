#!/usr/bin/env python3
"""Add TestConfig to test files that are missing it."""

import re
from pathlib import Path


def extract_test_name_from_path(file_path: Path) -> str:
    """Extract test name from file path."""
    # test name is the parent directory name
    return file_path.parent.name


def extract_algorithm_from_test_name(test_name: str) -> str:
    """Extract algorithm from test name."""
    # Algorithm is the first part before the first dash
    if test_name.startswith("sft-"):
        return "sft"
    elif test_name.startswith("dpo-"):
        return "dpo"
    elif test_name.startswith("grpo-"):
        return "grpo"
    else:
        return "unknown"


def infer_test_suites(test_name: str) -> list:
    """Infer test suites from test name."""
    if "quick" in test_name:
        return ["quick"]
    elif "long" in test_name:
        return ["nightly"]
    else:
        return ["nightly"]


def add_testconfig_to_file(file_path: Path):
    """Add TestConfig to a test file that's missing it."""
    print(f"Processing: {file_path.name}")

    content = file_path.read_text()

    # Check if already has config
    if "config = TestConfig(" in content:
        print("  ✓ Already has TestConfig")
        return

    # Extract test metadata
    test_name = extract_test_name_from_path(file_path)
    algorithm = extract_algorithm_from_test_name(test_name)
    test_suites = infer_test_suites(test_name)

    # Build the config block
    config_block = f'''    # Test configuration
    config = TestConfig(
        # Test metadata
        test_name="{test_name}",
        algorithm="{algorithm}",
        test_suites={test_suites},
        time_limit_minutes=120,
    )

'''

    # Find the class definition and insert after docstring
    class_pattern = r'(class \w+:.*?""".*?""")\n\n'
    match = re.search(class_pattern, content, re.DOTALL)

    if match:
        # Insert config block after class docstring
        before = match.group(1)
        after = content[match.end() :]

        new_content = before + "\n\n" + config_block + after

        # Also clean up any trailing junk
        new_content = re.sub(
            r"\n\n\s+assert return_code.*$", "", new_content, flags=re.DOTALL
        )

        file_path.write_text(new_content)
        print("  ✓ Added TestConfig")
    else:
        print("  ! Could not find class definition")


def main():
    """Add TestConfig to all test files missing it."""
    test_suites_dir = Path("tests/test_suites/llm")

    # Find all test files
    test_files = sorted(test_suites_dir.glob("*/test_*.py"))

    print(f"Found {len(test_files)} test files")
    print("=" * 80)

    for test_file in test_files:
        try:
            add_testconfig_to_file(test_file)
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            import traceback

            traceback.print_exc()

    print(f"\n{'=' * 80}")
    print("Completed!")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
