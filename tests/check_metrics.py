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
import argparse
import json
import statistics
import sys

from rich.console import Console
from rich.table import Table


# Custom functions for working with dictionary values
def min(value):
    """Return the minimum value in a dictionary."""
    return __builtins__.min(float(v) for v in value.values())


def max(value):
    """Return the maximum value in a dictionary."""
    return __builtins__.max(float(v) for v in value.values())


def mean(value):
    """Return the mean of values in a dictionary."""
    return statistics.mean(float(v) for v in value.values())


def evaluate_check(data: dict, check: str) -> tuple[bool, str, object]:
    """Evaluate a check against the data.

    Returns:
        Tuple of (passed, message, value)
    """
    # Create a local context with our custom functions and the data
    local_context = {"data": data, "min": min, "max": max, "mean": mean}

    # Extract the value expression from the check
    value_expr = check.split(">")[0].split("<")[0].split("==")[0].strip()

    try:
        # Try to get the value first
        value = eval(value_expr, {"__builtins__": __builtins__}, local_context)

        # Then evaluate the check
        result = eval(check, {"__builtins__": __builtins__}, local_context)
        if result:
            return True, f"PASS: {check}", value
        else:
            return False, f"FAIL: {check} (condition evaluated to False)", value
    except KeyError as e:
        return False, f"FAIL: {check} (key not found: {e})", None
    except IndexError as e:
        return False, f"FAIL: {check} (index error: {e})", None
    except Exception as e:
        return False, f"FAIL: {check} (error: {e})", None


def main():
    parser = argparse.ArgumentParser(description="Check conditions against a JSON file")
    parser.add_argument("json_file", help="Path to the JSON file")
    parser.add_argument(
        "checks", nargs="+", help="Conditions to check, will be eval()'d"
    )

    # Add helpful usage examples
    parser.epilog = """
    Examples:
      # Check if a specific metric is above a threshold
      python check_metrics.py results.json "data['accuracy'] > 0.9"

      # Check multiple conditions
      python check_metrics.py results.json "data['precision'] > 0.8" "data['recall'] > 0.7"

      # Use helper functions
      python check_metrics.py results.json "min(data['class_f1']) > 0.6"
      python check_metrics.py results.json "mean(data['accuracies']) > 0.85"
    """
    parser.formatter_class = argparse.RawDescriptionHelpFormatter
    args = parser.parse_args()

    # Load the JSON data - simplified
    with open(args.json_file, "r") as f:
        data = json.load(f)

    # Initialize rich console
    console = Console()

    # Create a table
    table = Table(title="Metric Checks")
    table.add_column("Status", style="bold")
    table.add_column("Check", style="dim")
    table.add_column("Value", style="cyan")
    table.add_column("Message", style="italic")

    # Evaluate all checks
    success = True
    for check in args.checks:
        passed, message, value = evaluate_check(data, check)

        status = "[green]PASS[/green]" if passed else "[red]FAIL[/red]"
        value_str = str(value) if value is not None else "N/A"
        detail = "" if passed else message.split(": ", 1)[1]

        table.add_row(status, check, value_str, detail)

        if not passed:
            success = False

    # Display the table
    console.print(table)

    # Exit with appropriate status code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
