#!/usr/bin/env python3
"""
Collect and analyze benchmark results from multiple runs.

Usage:
    python collect_results.py                    # Collect all results from vllm_standalone_perf_exp/
    python collect_results.py 1234567 1234568   # Collect specific job IDs
    python collect_results.py --output results.csv
"""

import argparse
import json
import glob
import os
from pathlib import Path


def find_result_files(job_ids=None, base_dir="."):
    """Find all results.json files."""
    base_path = Path(base_dir)
    
    if job_ids:
        # Specific job IDs
        files = []
        for job_id in job_ids:
            pattern = base_path / f"{job_id}-logs" / "results.json"
            if pattern.exists():
                files.append(pattern)
            else:
                print(f"Warning: {pattern} not found")
    else:
        # Find all
        files = list(base_path.glob("*-logs/results.json"))
    
    return sorted(files)


def load_results(files):
    """Load results from JSON files."""
    results = []
    
    for f in files:
        try:
            with open(f, 'r') as fp:
                data = json.load(fp)
                # Add file path for reference
                data['_file'] = str(f)
                data['_job_id'] = f.parent.name.replace('-logs', '')
                results.append(data)
        except Exception as e:
            print(f"Error loading {f}: {e}")
    
    return results


def print_table(results):
    """Print results as a formatted table."""
    if not results:
        print("No results found.")
        return
    
    # Header
    print("\n" + "=" * 120)
    print(f"{'Model':<40} {'TP':>3} {'PP':>3} {'DP':>3} {'GPUs':>5} {'Requests':>8} {'Tokens/s':>12} {'Tok/s/GPU':>12} {'Time(s)':>10}")
    print("=" * 120)
    
    # Sort by throughput
    results_sorted = sorted(results, key=lambda x: x.get('generation_throughput_tokens_per_sec', 0), reverse=True)
    
    for r in results_sorted:
        model = r.get('model', 'unknown')
        # Shorten model name
        if '/' in model:
            model = model.split('/')[-1]
        if len(model) > 38:
            model = model[:35] + "..."
        
        print(f"{model:<40} "
              f"{r.get('tp_size', '?'):>3} "
              f"{r.get('pp_size', '?'):>3} "
              f"{r.get('dp_size', '?'):>3} "
              f"{r.get('total_gpus', '?'):>5} "
              f"{r.get('total_requests', '?'):>8} "
              f"{r.get('generation_throughput_tokens_per_sec', 0):>12,.0f} "
              f"{r.get('tokens_per_sec_per_gpu', 0):>12,.0f} "
              f"{r.get('total_time_sec', 0):>10.1f}")
    
    print("=" * 120)


def save_csv(results, output_path):
    """Save results to CSV."""
    import csv
    
    if not results:
        print("No results to save.")
        return
    
    # Get all keys
    keys = set()
    for r in results:
        keys.update(r.keys())
    keys = sorted(keys)
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)
    
    print(f"Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Collect benchmark results")
    parser.add_argument("job_ids", nargs="*", help="Job IDs to collect (default: all)")
    parser.add_argument("--base-dir", default="vllm_standalone_perf_exp", help="Base directory (default: vllm_standalone_perf_exp)")
    parser.add_argument("--output", "-o", help="Output CSV file")
    parser.add_argument("--json", help="Output JSON file")
    
    args = parser.parse_args()
    
    # Find and load results
    files = find_result_files(args.job_ids or None, args.base_dir)
    print(f"Found {len(files)} result files")
    
    results = load_results(files)
    print(f"Loaded {len(results)} results")
    
    # Print table
    print_table(results)
    
    # Save if requested
    if args.output:
        save_csv(results, args.output)
    
    if args.json:
        with open(args.json, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.json}")


if __name__ == "__main__":
    main()

