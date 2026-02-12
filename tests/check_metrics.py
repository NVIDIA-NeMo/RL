#!/usr/bin/env -S uv run --script -q
# /// script
# dependencies = [
#   "rich"
# ]
# ///
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
import builtins
import json
import statistics
import sys

from rich.console import Console
from rich.table import Table


# Custom functions for working with dictionary values
def min(value):
    """Return the minimum value in a dictionary."""
    return builtins.min(float(v) for v in value.values())


def max(value):
    """Return the maximum value in a dictionary."""
    return builtins.max(float(v) for v in value.values())


def ratio_above(value, threshold):
    """Return the ratio of values that are >= threshold.

    Args:
        value: Dictionary of step -> value
        threshold: Threshold value to compare against

    Returns:
        Float between 0.0 and 1.0 representing the proportion of values >= threshold
    """
    vals = [float(v) for v in value.values()]
    if len(vals) == 0:
        return 0.0
    count_above = sum(1 for v in vals if v >= threshold)
    return count_above / len(vals)


def mean(value, range_start=1, range_end=0, ignore_top_p=0.0):
    """Return the mean of values (or a range of values) in a dictionary.

    Note:
        step, and ranges, are 1 indexed. Range_end is exclusive.
        range_end=0 means to include until the last step in the run

    Args:
        value: Dictionary of step -> value
        range_start: Starting step (1-indexed, default=1)
        range_end: Ending step (1-indexed, exclusive, 0 means last step)
        ignore_top_p: Proportion of top outliers to ignore (0.0-1.0, default=0.0)
                     E.g., 0.05 ignores the top 5% of values
    """

    ## find potential offset that might arise from resuming from a checkpoint
    max_step_reached = builtins.max([int(s) for s in value.keys()])
    ## this is the number of steps that occurred prior to resuming
    offset = max_step_reached - len(value)

    num_elem = len(value)
    if range_start < 0:
        range_start += num_elem + 1 + offset
    if range_end <= 0:
        range_end += num_elem + 1 + offset

    vals = []
    for step, v in value.items():
        if range_start <= int(step) and int(step) < range_end:
            vals.append(float(v))

    # Validate ignore_top_p parameter
    if not 0.0 <= ignore_top_p <= 1.0:
        raise ValueError(
            f"ignore_top_p must be between 0.0 and 1.0, got {ignore_top_p}"
        )

    # Filter out top outliers if requested
    if ignore_top_p > 0.0 and len(vals) > 0:
        # Sort values and determine cutoff index
        sorted_vals = sorted(vals)
        cutoff_idx = int(len(sorted_vals) * (1.0 - ignore_top_p))
        # Take only values up to the cutoff (excluding top p%)
        vals = sorted_vals[:cutoff_idx] if cutoff_idx > 0 else sorted_vals[:1]

    return statistics.mean(vals)


def median(value, range_start=1, range_end=0):
    """Return the median of values (or a range of values) in a dictionary.

    Note:
        step, and ranges, are 1 indexed. Range_end is exclusive.
        range_end=0 means to include until the last step in the run

    Args:
        value: Dictionary of step -> value
        range_start: Starting step (1-indexed, default=1)
        range_end: Ending step (1-indexed, exclusive, 0 means last step)
    """

    ## find potential offset that might arise from resuming from a checkpoint
    max_step_reached = builtins.max([int(s) for s in value.keys()])
    ## this is the number of steps that occurred prior to resuming
    offset = max_step_reached - len(value)

    num_elem = len(value)
    if range_start < 0:
        range_start += num_elem + 1 + offset
    if range_end <= 0:
        range_end += num_elem + 1 + offset

    vals = []
    for step, v in value.items():
        if range_start <= int(step) and int(step) < range_end:
            vals.append(float(v))

    return statistics.median(vals)


def evaluate_check(data: dict, check: str) -> tuple[bool, str, object]:
    """Evaluate a check against the data.

    Returns:
        Tuple of (passed, message, value)
    """
    # Create a local context with our custom functions and the data
    local_context = {
        "data": data,
        "min": min,
        "max": max,
        "mean": mean,
        "median": median,
        "ratio_above": ratio_above,
    }

    # Extract the value expression from the check
    value_expr = check.split(">")[0].split("<")[0].split("==")[0].strip()

    try:
        # Try to get the value first
        value = eval(value_expr, {"__builtins__": builtins}, local_context)

        # Then evaluate the check
        result = eval(check, {"__builtins__": builtins}, local_context)
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
    parser.add_argument(
        "--table-width",
        type=int,
        default=None,
        help="Set the overall table width (columns will auto-size within this width)",
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
      python check_metrics.py results.json "median(data['accuracies']) > 0.85"
      python check_metrics.py results.json "mean(data['loss'], ignore_top_p=0.05) < 1.5"
      python check_metrics.py results.json "ratio_above(data['error'], 1.05) < 0.02"
    """
    parser.formatter_class = argparse.RawDescriptionHelpFormatter
    args = parser.parse_args()

    # Load the JSON data - simplified
    with open(args.json_file, "r") as f:
        data = json.load(f)

    # Initialize rich console
    console = Console()

    # Create a table
    table = Table(title="Metric Checks", min_width=150, width=args.table_width)
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
