#!/usr/bin/env python3
"""Clean up duplicate code in test files after applying the new pattern."""

import re
from pathlib import Path


def cleanup_test_file(file_path: Path):
    """Remove duplicate if/else code from test file."""
    print(f"Cleaning: {file_path.name}")

    content = file_path.read_text()

    # Check if file has duplicate code
    if "if use_slurm:" not in content:
        print("  ✓ Already clean")
        return

    # Find the pattern: new method followed by old if/else code
    pattern = (
        r"(    def test_training_runs_successfully_slurm\(self, request\):\n"
        r'        """Test that training completes successfully when run via Slurm\."""\n'
        r"        return_code = run_via_slurm\(self\.config, request\.node\.nodeid\)\n"
        r"\n"
        r'        assert return_code == 0, f"Training failed with return code {return_code}\. Check {self\.config\.run_log_path}"\n'
        r"\n)"
        r"        # Run via Slurm.*?(?=\n\n|\Z)"
    )

    # Replace with just the new method (keeping the clean part)
    cleaned = re.sub(pattern, r"\1", content, flags=re.DOTALL)

    if cleaned != content:
        file_path.write_text(cleaned)
        print("  ✓ Cleaned up duplicate code")
    else:
        print("  ! Could not match pattern")


def main():
    """Clean up all test files."""
    test_suites_dir = Path("tests/test_suites/llm")

    # Find all test files
    test_files = sorted(test_suites_dir.glob("*/test_*.py"))

    print(f"Found {len(test_files)} test files")
    print("=" * 80)

    for test_file in test_files:
        try:
            cleanup_test_file(test_file)
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            import traceback

            traceback.print_exc()

    print(f"\n{'=' * 80}")
    print("Completed!")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
