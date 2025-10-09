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
import json
import os
import statistics
import sys

from rich.console import Console
from rich.table import Table


# Global flag to control whether range details should be collected
_show_range_details = True
_range_details = {}


# Custom functions for working with dictionary values
def min(value):
    """Return the minimum value in a dictionary."""
    min_val = __builtins__.min(float(v) for v in value.values())
    
    if _show_range_details:
        # Store all step-value pairs as a dict
        _range_details[id(value)] = json.dumps(
            {str(step): float(v) for step, v in value.items()}, 
            indent=2
        )
    
    return min_val


def max(value):
    """Return the maximum value in a dictionary."""
    max_val = __builtins__.max(float(v) for v in value.values())
    
    if _show_range_details:
        # Store all step-value pairs as a dict
        _range_details[id(value)] = json.dumps(
            {str(step): float(v) for step, v in value.items()}, 
            indent=2
        )
    
    return max_val


def mean(value, range_start=1, range_end=0):
    """Return the mean of values (or a range of values) in a dictionary.

    Note:
        step, and ranges, are 1 indexed. Range_end is exclusive.
        range_end=0 means to include until the last step in the run
    """

    ## find potential offset that might arise from resuming from a checkpoint
    max_step_reached = __builtins__.max([int(s) for s in value.keys()])
    ## this is the number of steps that occurred prior to resuming
    offset = max_step_reached - len(value)

    num_elem = len(value)
    if range_start < 0:
        range_start += num_elem + 1 + offset
    if range_end <= 0:
        range_end += num_elem + 1 + offset

    vals = []
    range_dict = {}
    for step, v in value.items():
        if range_start <= int(step) and int(step) < range_end:
            vals.append(float(v))
            range_dict[str(step)] = float(v)

    mean_val = statistics.mean(vals)
    
    if _show_range_details:
        # Store the filtered step-value pairs as a dict
        _range_details[id(value)] = json.dumps(range_dict, indent=2)
    
    return mean_val


def evaluate_check(data: dict, check: str) -> tuple[bool, str, object, str]:
    """Evaluate a check against the data.

    Returns:
        Tuple of (passed, message, value, range_details)
    """
    global _range_details
    
    # Create a local context with our custom functions and the data
    local_context = {"data": data, "min": min, "max": max, "mean": mean}

    # Extract the value expression from the check
    value_expr = check.split(">")[0].split("<")[0].split("==")[0].strip()

    try:
        # Clear any previous range details
        _range_details.clear()
        
        # Try to get the value first
        value = eval(value_expr, {"__builtins__": __builtins__}, local_context)

        # Capture range details if they were collected
        range_detail = list(_range_details.values())[0] if _range_details else ""
        
        # Then evaluate the check
        result = eval(check, {"__builtins__": __builtins__}, local_context)
        if result:
            return True, f"PASS: {check}", value, range_detail
        else:
            return False, f"FAIL: {check} (condition evaluated to False)", value, range_detail
    except KeyError as e:
        return False, f"FAIL: {check} (key not found: {e})", None, ""
    except IndexError as e:
        return False, f"FAIL: {check} (index error: {e})", None, ""
    except Exception as e:
        return False, f"FAIL: {check} (error: {e})", None, ""


def main():
    global _show_range_details
    
    parser = argparse.ArgumentParser(description="Check conditions against a JSON file")
    parser.add_argument("json_file", help="Path to the JSON file")
    parser.add_argument(
        "checks", nargs="+", help="Conditions to check, will be eval()'d"
    )
    parser.add_argument(
        "--table-width",
        type=int,
        default=None,
        help="Set the overall table width (columns will auto-size within this width. Minimum is 150)"
    )
    parser.add_argument(
        "--hide-range-details",
        action="store_true",
        help="Hide step and value details when using min/max/mean on ranges (shown by default)"
    )

    # Add helpful usage examples
    parser.epilog = """
    Examples:
      # Check if a specific metric is above a threshold
      ./check_metrics.py results.json "data['accuracy'] > 0.9"

      # Check multiple conditions
      ./check_metrics.py results.json "data['precision'] > 0.8" "data['recall'] > 0.7"

      # Use helper functions
      ./check_metrics.py results.json "min(data['class_f1']) > 0.6"
      ./check_metrics.py results.json "mean(data['accuracies']) > 0.85"
      
      # Set table width (range details shown by default)
      ./check_metrics.py --table-width 100 results.json "mean(data['loss']) < 0.5"
      
      # Hide range details if not needed
      ./check_metrics.py --hide-range-details results.json "mean(data['loss']) < 0.5"
    """
    parser.formatter_class = argparse.RawDescriptionHelpFormatter
    args = parser.parse_args()
    
    # Set global flag for range details (invert the hide flag)
    _show_range_details = not args.hide_range_details
    
    # Determine table width: use specified value, or fall back to COLUMNS env var (default 150)
    table_width = args.table_width

    # Load the JSON data - simplified
    with open(args.json_file, "r") as f:
        data = json.load(f)

    # Initialize rich console
    console = Console()

    # Create a table with optional width setting
    table = Table(title="Metric Checks", min_width=150, width=table_width)
    table.add_column("Status", style="bold")
    table.add_column("Check", style="dim")
    table.add_column("Value", style="cyan")
    table.add_column("Message", style="italic")
    
    # Add Range Details column by default (unless hidden)
    if not args.hide_range_details:
        table.add_column("Range Details", style="yellow")

    # Evaluate all checks
    success = True
    for check in args.checks:
        passed, message, value, range_detail = evaluate_check(data, check)

        status = "[green]PASS[/green]" if passed else "[red]FAIL[/red]"
        value_str = str(value) if value is not None else "N/A"
        detail = "" if passed else message.split(": ", 1)[1]

        if not args.hide_range_details:
            table.add_row(status, check, value_str, detail, range_detail)
        else:
            table.add_row(status, check, value_str, detail)

        if not passed:
            success = False

    # Display the table
    console.print(table)

    # Exit with appropriate status code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
