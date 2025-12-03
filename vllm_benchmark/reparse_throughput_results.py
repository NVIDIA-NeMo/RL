#!/usr/bin/env python3
"""
Re-parse throughput results from log files to fix parsing errors.

Usage:
    python reparse_throughput_results.py               # All jobs
    python reparse_throughput_results.py 1141527       # Specific job
    python reparse_throughput_results.py --dry-run     # Just show what would be fixed
"""

import json
import re
import sys
from pathlib import Path


def parse_throughput_from_log(log_file):
    """Parse throughput values from vllm bench throughput output."""
    content = log_file.read_text()
    
    # Look for: "Throughput: 23.94 requests/s, 30641.62 total tokens/s, 24513.30 output tokens/s"
    match = re.search(
        r'Throughput:\s*([\d.]+)\s*requests/s,\s*([\d.]+)\s*total tokens/s',
        content
    )
    
    if match:
        return {
            'throughput_requests_per_sec': float(match.group(1)),
            'throughput_tokens_per_sec': float(match.group(2)),
        }
    
    return None


def fix_results_json(job_dir, dry_run=False):
    """Fix results.json by re-parsing from log files."""
    results_file = job_dir / 'results.json'
    if not results_file.exists():
        print(f"  No results.json in {job_dir.name}")
        return False
    
    # Load existing results
    try:
        with open(results_file) as f:
            data = json.load(f)
    except Exception as e:
        print(f"  Error loading {results_file}: {e}")
        return False
    
    if not isinstance(data, list):
        data = [data]
    
    fixed = False
    for item in data:
        isl = item.get('input_len', 0)
        osl = item.get('output_len', 0)
        
        # Find corresponding log file
        log_file = job_dir / f'result_ISL{isl}_OSL{osl}.txt'
        if not log_file.exists():
            continue
        
        # Parse from log
        parsed = parse_throughput_from_log(log_file)
        if not parsed:
            print(f"  Could not parse log for ISL={isl}, OSL={osl}")
            continue
        
        old_tok = item.get('throughput_tokens_per_sec', 0)
        new_tok = parsed['throughput_tokens_per_sec']
        
        if abs(old_tok - new_tok) > 1:  # Significant difference
            print(f"  ISL={isl} OSL={osl}: {old_tok:.2f} -> {new_tok:.2f} tok/s")
            if not dry_run:
                item['throughput_tokens_per_sec'] = new_tok
                item['throughput_requests_per_sec'] = parsed['throughput_requests_per_sec']
            fixed = True
    
    # Save fixed results
    if fixed and not dry_run:
        with open(results_file, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"  -> Saved fixed results")
    
    return fixed


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Re-parse throughput results")
    parser.add_argument("job_ids", nargs="*", help="Job IDs to fix (default: all)")
    parser.add_argument("--dry-run", action="store_true", help="Just show what would be fixed")
    args = parser.parse_args()
    
    # Find throughput results directory
    script_dir = Path(__file__).parent.resolve()
    exp_base = script_dir.parent / "vllm_standalone_perf_exp" / "throughput"
    
    if not exp_base.exists():
        print(f"No throughput results found at {exp_base}")
        return
    
    # Find job directories
    if args.job_ids:
        job_dirs = [exp_base / f"{jid}-logs" for jid in args.job_ids]
        job_dirs = [d for d in job_dirs if d.exists()]
    else:
        job_dirs = sorted(exp_base.glob("*-logs"))
    
    print(f"Checking {len(job_dirs)} job directories...")
    if args.dry_run:
        print("(Dry run - no changes will be made)")
    print()
    
    fixed_count = 0
    for job_dir in job_dirs:
        print(f"Job {job_dir.name}:")
        if fix_results_json(job_dir, dry_run=args.dry_run):
            fixed_count += 1
    
    print()
    print(f"Fixed {fixed_count} jobs" + (" (dry run)" if args.dry_run else ""))


if __name__ == "__main__":
    main()
