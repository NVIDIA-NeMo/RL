#!/usr/bin/env python3
"""
Re-parse online benchmark results from log files to fix parsing errors.

Usage:
    python reparse_online_results.py               # All jobs
    python reparse_online_results.py 1058012       # Specific job
    python reparse_online_results.py --dry-run     # Just show what would be fixed
"""

import re
import sys
from pathlib import Path


def parse_online_from_log(log_file):
    """Parse throughput values from vllm bench serve output."""
    content = log_file.read_text()
    
    def extract(pattern, default='0'):
        m = re.search(pattern, content)
        return m.group(1) if m else default
    
    req_tp = extract(r'Request throughput[^:]*:\s*([0-9.]+)')
    out_tp = extract(r'Output token throughput[^:]*:\s*([0-9.]+)')
    ttft = extract(r'Mean TTFT \(ms\):\s*([0-9.]+)')
    itl = extract(r'Mean ITL \(ms\):\s*([0-9.]+)')
    
    return {
        'request_throughput': float(req_tp),
        'output_token_throughput': float(out_tp),
        'mean_ttft_ms': float(ttft),
        'mean_itl_ms': float(itl),
    }


def fix_results_summary(job_dir, dry_run=False):
    """Fix results_summary.txt by re-parsing from slurm log files."""
    summary_file = job_dir / 'results_summary.txt'
    if not summary_file.exists():
        print(f"  No results_summary.txt in {job_dir.name}")
        return False
    
    # Find slurm output file
    slurm_out = list(job_dir.glob("slurm-*.out"))
    if not slurm_out:
        print(f"  No slurm output file in {job_dir.name}")
        return False
    
    slurm_file = slurm_out[0]
    
    # Parse from slurm log
    parsed = parse_online_from_log(slurm_file)
    
    # Read current summary
    with open(summary_file) as f:
        lines = f.readlines()
    
    if len(lines) < 2:
        print(f"  Empty summary file")
        return False
    
    # Check if throughput is 0 (needs fixing)
    fixed = False
    new_lines = [lines[0]]  # Keep header
    
    for line in lines[1:]:
        parts = line.strip().split(',')
        if len(parts) >= 8:
            isl, osl, conc, prompts, req_tp, tok_tp, ttft, itl = parts[:8]
            
            if float(req_tp) == 0 and parsed['request_throughput'] > 0:
                # Fix the values
                print(f"  ISL={isl} OSL={osl}: req_tp 0 -> {parsed['request_throughput']:.2f}, tok_tp 0 -> {parsed['output_token_throughput']:.2f}")
                if not dry_run:
                    new_line = f"{isl},{osl},{conc},{prompts},{parsed['request_throughput']:.2f},{parsed['output_token_throughput']:.2f},{ttft},{itl}\n"
                    new_lines.append(new_line)
                    fixed = True
                else:
                    new_lines.append(line)
                    fixed = True
            else:
                new_lines.append(line)
    
    if fixed and not dry_run:
        with open(summary_file, 'w') as f:
            f.writelines(new_lines)
        print(f"  -> Saved fixed results")
    
    return fixed


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Re-parse online benchmark results")
    parser.add_argument("job_ids", nargs="*", help="Job IDs to fix (default: all)")
    parser.add_argument("--dry-run", action="store_true", help="Just show what would be fixed")
    args = parser.parse_args()
    
    # Find online results directory
    script_dir = Path(__file__).parent.resolve()
    exp_base = script_dir.parent / "vllm_standalone_perf_exp" / "online"
    
    if not exp_base.exists():
        print(f"No online results found at {exp_base}")
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
        if fix_results_summary(job_dir, dry_run=args.dry_run):
            fixed_count += 1
    
    print()
    print(f"Fixed {fixed_count} jobs" + (" (dry run)" if args.dry_run else ""))


if __name__ == "__main__":
    main()
