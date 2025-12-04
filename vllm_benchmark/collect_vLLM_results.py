#!/usr/bin/env python3
"""
Collect and analyze benchmark results from multiple runs.

Usage:
    python collect_results.py                    # Show offline results
    python collect_results.py --online           # Show online results
    python collect_results.py --throughput       # Show throughput results (vllm bench throughput)
    python collect_results.py --group            # Group by config, show mean±std
    python collect_results.py --all              # Include running/failed jobs
    python collect_results.py 1234567 1234568   # Collect specific job IDs
    python collect_results.py --output results.csv
"""

import argparse
import json
import glob
import os
import subprocess
import re
from pathlib import Path

# ============================================================
# GRPO Configuration definitions (matching grpo_benchmark_sweep.sh)
# ============================================================
# Format: (model_pattern, tp, pp, ep, nodes, isl, osl, num_prompts)
# ISL:OSL = 1:4 ratio (OpenMathInstruct-2 dataset characteristics)
GRPO_CONFIGS = [
    # qwen32b: Gen(TP=1, PP=1, DP=16), 4 nodes, ISL=640, OSL=2560
    ("Qwen3-32B", 1, 1, 1, 4, 640, 2560, 2048),
    # qwen30b: Gen(TP=1, PP=1, DP=16), 4 nodes, ISL=640, OSL=2560
    ("Qwen3-30B-A3B", 1, 1, 1, 4, 640, 2560, 2048),
    # llama8b: Gen(TP=1, PP=1, DP=8), 2 nodes, ISL=256, OSL=1024
    ("Llama-3.1-8B", 1, 1, 1, 2, 256, 1024, 2048),
    ("Llama-3.1-8B-Instruct", 1, 1, 1, 2, 256, 1024, 2048),
    # llama70b: Gen(TP=2, PP=1, DP=8), 4 nodes, ISL=128, OSL=512
    ("Llama-3.1-70B", 2, 1, 1, 4, 128, 512, 2048),
    ("Llama-3.1-70B-Instruct", 2, 1, 1, 4, 128, 512, 2048),
    # llama70b-lowgbs: Gen(TP=2, PP=1, DP=8), 4 nodes, ISL=128, OSL=512, GBS=512
    ("Llama-3.1-70B", 2, 1, 1, 4, 128, 512, 512),
    ("Llama-3.1-70B-Instruct", 2, 1, 1, 4, 128, 512, 512),
    # llama70b-highseq: Gen(TP=2, PP=1, DP=8), 4 nodes, ISL=256, OSL=1024
    ("Llama-3.1-70B", 2, 1, 1, 4, 256, 1024, 2048),
    ("Llama-3.1-70B-Instruct", 2, 1, 1, 4, 256, 1024, 2048),
]


def is_grpo_config(result):
    """Check if a result matches any GRPO configuration."""
    model = result.get('model', '')
    tp = result.get('tp_size', 0)
    pp = result.get('pp_size', 0)
    nodes = result.get('num_nodes', 0)
    isl = result.get('input_len', 0)
    osl = result.get('output_len', 0)
    nprompts = result.get('num_prompts', 0)
    
    for cfg in GRPO_CONFIGS:
        model_pattern, cfg_tp, cfg_pp, cfg_ep, cfg_nodes, cfg_isl, cfg_osl, cfg_nprompts = cfg
        
        # Check if model matches (partial match)
        if model_pattern not in model:
            continue
        
        # Check parallelism
        if tp != cfg_tp or pp != cfg_pp:
            continue
            
        # Check nodes
        if nodes != cfg_nodes:
            continue
            
        # Check ISL/OSL (exact match)
        if isl != cfg_isl or osl != cfg_osl:
            continue
            
        # Check batch size
        if nprompts != cfg_nprompts:
            continue
            
        return True
    
    return False


def filter_grpo_results(results):
    """Filter results to only include GRPO configurations."""
    return [r for r in results if is_grpo_config(r)]



def get_running_jobs():
    """
    Get SLURM job states for current user.
    Returns: dict mapping job_id -> state ('R'=Running, 'PD'=Pending, etc.)
    """
    try:
        # Format: job_id|state
        result = subprocess.run(
            ['squeue', '-u', os.environ.get('USER', ''), '-h', '-o', '%i|%t'],
            capture_output=True, text=True, timeout=10
        )
        jobs = {}
        if result.stdout.strip():
            for line in result.stdout.strip().split('\n'):
                if '|' in line:
                    job_id, state = line.split('|', 1)
                    jobs[job_id.strip()] = state.strip()
        return jobs
    except Exception:
        return {}


def check_job_status(job_dir, running_jobs):
    """
    Check job status based on files and content.
    Returns: 'Completed', 'Running', 'Pending', 'Failed', 'Failed (OOM)', 'Failed (Timeout)', 'Cancelled', or 'Unknown'
    """
    job_id = job_dir.name.replace('-logs', '')
    results_file = job_dir / 'results.json'
    
    # Check if job is in SLURM queue (running or pending)
    if job_id in running_jobs:
        state = running_jobs[job_id]
        if state == 'R':
            return 'Running'
        elif state == 'PD':
            return 'Pending'
        elif state == 'CG':  # Completing
            return 'Running'
        else:
            return f'Queued ({state})'
    
    # Check if results.json exists and is valid
    # This should take priority - if we have valid results, job completed
    if results_file.exists():
        try:
            with open(results_file) as f:
                data = json.load(f)
                # Handle both dict (offline) and list (throughput) formats
                if isinstance(data, list) and len(data) > 0:
                    # Throughput benchmark: list of results
                    return 'Completed'
                elif isinstance(data, dict):
                    # Offline benchmark: single dict
                    if data.get('generation_throughput_tokens_per_sec', 0) > 0:
                        return 'Completed'
                    elif data.get('throughput_tokens_per_sec', 0) > 0:
                        return 'Completed'
        except Exception:
            pass  # Invalid JSON, continue to error checking
    
    # Specific error patterns with priority (check these first)
    specific_errors = [
        (r'CUDA out of memory|OutOfMemoryError|OOM|torch\.cuda\.OutOfMemoryError', 'Failed (OOM)'),
        (r'DUE TO TIME LIMIT|timeout|TIMEOUT', 'Failed (Timeout)'),
        (r'CANCELLED', 'Cancelled'),
    ]
    
    # More strict error patterns - avoid false positives from INFO/WARNING messages
    general_error_patterns = [
        r'Traceback \(most recent call last\)',  # Python traceback
        r'RuntimeError:',
        r'ValueError:',
        r'AssertionError:',
        r'srun: error:',
        r'Killed',
    ]
    
    def check_log_content(content):
        """Check content for errors, return specific status if found."""
        # Check specific errors first
        for pattern, status in specific_errors:
            if re.search(pattern, content, re.IGNORECASE):
                return status
        # Check general errors (case-sensitive for stricter matching)
        for pattern in general_error_patterns:
            if re.search(pattern, content):
                return 'Failed'
        return None
    
    for log_file in job_dir.glob('slurm-*.out'):
        try:
            with open(log_file, 'r', errors='ignore') as f:
                content = f.read()
                status = check_log_content(content)
                if status:
                    return status
        except Exception:
            pass
    
    for log_file in job_dir.glob('slurm-*.err'):
        try:
            with open(log_file, 'r', errors='ignore') as f:
                content = f.read()
                # Filter out set -x trace output (lines starting with +)
                lines = [l for l in content.split('\n') 
                         if l.strip() and not l.startswith('+')]
                if lines:
                    status = check_log_content('\n'.join(lines))
                    if status:
                        return status
        except Exception:
            pass
    
    # If results.json exists but wasn't valid, check if it has data
    if results_file.exists():
        return 'Completed'  # Has results file, assume completed
    
    return 'Failed'  # No results file and not running


def find_all_job_dirs(job_ids=None, base_dir="."):
    """Find all job directories (including those without results.json)."""
    base_path = Path(base_dir)
    
    if job_ids:
        dirs = []
        for job_id in job_ids:
            job_dir = base_path / f"{job_id}-logs"
            if job_dir.exists():
                dirs.append(job_dir)
    else:
        dirs = [d for d in base_path.glob("*-logs") if d.is_dir()]
    
    return sorted(dirs)


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


def load_results(files, running_jobs=None):
    """Load results from JSON files."""
    if running_jobs is None:
        running_jobs = get_running_jobs()
    
    results = []
    
    for f in files:
        try:
            with open(f, 'r') as fp:
                data = json.load(fp)
                job_id = f.parent.name.replace('-logs', '')
                status = check_job_status(f.parent, running_jobs)
                
                # Handle both single result (dict) and multiple results (list)
                if isinstance(data, list):
                    # Throughput benchmark: list of results
                    for item in data:
                        item['_file'] = str(f)
                        item['_job_id'] = job_id
                        item['_status'] = status
                        results.append(item)
                else:
                    # Single result (offline/online benchmark)
                    data['_file'] = str(f)
                    data['_job_id'] = job_id
                    data['_status'] = status
                    results.append(data)
        except Exception as e:
            print(f"Error loading {f}: {e}")
    
    return results


def load_all_jobs(job_dirs, running_jobs):
    """Load all jobs including running/failed ones."""
    results = []
    
    for job_dir in job_dirs:
        job_id = job_dir.name.replace('-logs', '')
        status = check_job_status(job_dir, running_jobs)
        results_file = job_dir / 'results.json'
        
        if results_file.exists():
            try:
                with open(results_file, 'r') as fp:
                    data = json.load(fp)
                    data['_file'] = str(results_file)
                    data['_job_id'] = job_id
                    data['_status'] = status
                    results.append(data)
                    continue
            except Exception:
                pass
        
        # For running/failed jobs without valid results.json
        # Try to extract info from slurm output
        data = {
            '_file': str(job_dir),
            '_job_id': job_id,
            '_status': status,
            'model': 'unknown',
            'num_nodes': 0,
            'tp_size': 0,
            'pp_size': 0,
            'ep_size': 1,
            'dp_size': 0,
            'total_gpus': 0,
            'generation_throughput_tokens_per_sec': 0,
            'tokens_per_sec_per_gpu': 0,
            'total_time_sec': 0,
            'total_requests': 0,
        }
        
        # Try to parse model info from log
        for log_file in job_dir.glob('slurm-*.out'):
            try:
                with open(log_file, 'r', errors='ignore') as f:
                    content = f.read()
                    # Extract model name
                    m = re.search(r'Model:\s*(\S+)', content)
                    if m:
                        data['model'] = m.group(1)
                    # Extract parallelism
                    m = re.search(r'TP=(\d+)', content)
                    if m:
                        data['tp_size'] = int(m.group(1))
                    m = re.search(r'PP=(\d+)', content)
                    if m:
                        data['pp_size'] = int(m.group(1))
                    m = re.search(r'EP=(\d+)', content)
                    if m:
                        data['ep_size'] = int(m.group(1))
                    m = re.search(r'DP=(\d+)', content)
                    if m:
                        data['dp_size'] = int(m.group(1))
                    m = re.search(r'Nodes:\s*(\d+)', content)
                    if m:
                        data['num_nodes'] = int(m.group(1))
                    m = re.search(r'Total GPUs:\s*(\d+)', content)
                    if m:
                        data['total_gpus'] = int(m.group(1))
                    break
            except Exception:
                pass
        
        results.append(data)
    
    return results


def get_gpu_model(r):
    """Get GPU model, defaulting to GB200 for older results without this field."""
    gpu = r.get('gpu_model', 'GB200')
    if gpu in ('unknown', '', None):
        gpu = 'GB200'
    
    # Fix truncated GPU names like "NVIDIAGB20" -> "GB200"
    import re
    if gpu.startswith('NVIDIA'):
        # Try to extract known GPU models
        match = re.search(r'(GB200|GB20|H100|H200|A100|A10|L40|RTX\d+)', gpu)
        if match:
            gpu = match.group(1)
            # Fix truncated names
            if gpu == 'GB20':
                gpu = 'GB200'
        else:
            gpu = gpu.replace('NVIDIA', '')
    
    return gpu

def get_gpus_per_node(r):
    """Get GPUs per node, defaulting to 4 for GB200."""
    gpn = r.get('gpus_per_node')
    if gpn is None or gpn == '?':
        gpn = 4  # Default for GB200
    return gpn

def get_config_key(r):
    """Get a unique key for a configuration."""
    model = r.get('model', 'unknown')
    if '/' in model:
        model = model.split('/')[-1]
    gpu_model = get_gpu_model(r)
    gpus_per_node = get_gpus_per_node(r)
    return (gpu_model, model, r.get('num_nodes', 1), gpus_per_node, r.get('tp_size', 0), r.get('pp_size', 0), r.get('dp_size', 0), r.get('ep_size', 1))


def print_table(results, group_by_config=False, bench_type="offline"):
    """Print results as a formatted table."""
    if not results:
        print("No results found.")
        return
    
    # Check if this is throughput benchmark (has input_len/output_len)
    is_throughput = bench_type == "throughput" or any('input_len' in r for r in results)
    
    if is_throughput:
        print_throughput_table(results)
    elif group_by_config:
        print_grouped_table(results)
    else:
        print_individual_table(results)


def get_status_display(status):
    """Get colored status display."""
    # ANSI color codes
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    GRAY = '\033[90m'
    MAGENTA = '\033[95m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    
    status_colors = {
        'Completed': f'{GREEN}Completed{RESET}',
        'Running': f'{YELLOW}Running{RESET}',
        'Pending': f'{BLUE}Pending{RESET}',
        'Failed': f'{RED}Failed{RESET}',
        'Failed (OOM)': f'{RED}Failed (OOM){RESET}',
        'Failed (Timeout)': f'{RED}Failed (Timeout){RESET}',
        'Cancelled': f'{MAGENTA}Cancelled{RESET}',
        'Unknown': f'{GRAY}Unknown{RESET}',
    }
    # Handle Queued states
    if status.startswith('Queued'):
        return f'{BLUE}{status}{RESET}'
    return status_colors.get(status, f'{RED}{status}{RESET}')


def print_individual_table(results):
    """Print all results individually."""
    # Header
    print("\n" + "=" * 190)
    print(f"{'Job ID':<10} {'Status':<18} {'GPU':<8} {'Model':<20} {'N':>2} {'G/N':>3} {'TP':>3} {'PP':>3} {'EP':>3} {'DP':>3} {'GPUs':>5} {'Requests':>8} {'Tokens/s':>12} {'Tok/s/GPU':>12} {'Time(s)':>10}")
    print("=" * 190)
    
    # Sort by status first (Completed first), then by throughput
    def sort_key(x):
        status = x.get('_status', 'Unknown')
        # Group all Failed statuses together
        if status.startswith('Failed'):
            status_order_val = 2
        else:
            status_order = {'Completed': 0, 'Running': 1, 'Cancelled': 3, 'Unknown': 4}
            status_order_val = status_order.get(status, 4)
        return (status_order_val, -x.get('generation_throughput_tokens_per_sec', 0))
    
    results_sorted = sorted(results, key=sort_key)
    
    for r in results_sorted:
        model = r.get('model', 'unknown')
        # Shorten model name
        if '/' in model:
            model = model.split('/')[-1]
        if len(model) > 18:
            model = model[:15] + "..."
        
        job_id = r.get('_job_id', '?')
        gpu_model = get_gpu_model(r)[:7]  # Truncate GPU model name
        status = r.get('_status', 'Completed')
        
        print(f"{job_id:<10} "
              f"{get_status_display(status):<27} "  # Extra space for ANSI codes
              f"{gpu_model:<8} "
              f"{model:<20} "
              f"{r.get('num_nodes', 1):>2} "
              f"{get_gpus_per_node(r):>3} "
              f"{r.get('tp_size', '?'):>3} "
              f"{r.get('pp_size', '?'):>3} "
              f"{r.get('ep_size', 1):>3} "
              f"{r.get('dp_size', '?'):>3} "
              f"{r.get('total_gpus', '?'):>5} "
              f"{r.get('total_requests', '?'):>8} "
              f"{r.get('generation_throughput_tokens_per_sec', 0):>12,.0f} "
              f"{r.get('tokens_per_sec_per_gpu', 0):>12,.0f} "
              f"{r.get('total_time_sec', 0):>10.1f}")
    
    print("=" * 190)
    # Print summary with colors
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    MAGENTA = '\033[95m'
    BLUE = '\033[94m'
    RESET = '\033[0m'
    
    completed = sum(1 for r in results if r.get('_status') == 'Completed')
    running = sum(1 for r in results if r.get('_status') == 'Running')
    pending = sum(1 for r in results if r.get('_status') == 'Pending' or r.get('_status', '').startswith('Queued'))
    failed = sum(1 for r in results if r.get('_status', '').startswith('Failed') and 'OOM' not in r.get('_status', '') and 'Timeout' not in r.get('_status', ''))
    oom = sum(1 for r in results if 'OOM' in r.get('_status', ''))
    timeout = sum(1 for r in results if 'Timeout' in r.get('_status', ''))
    cancelled = sum(1 for r in results if r.get('_status') == 'Cancelled')
    
    summary_parts = [f"{GREEN}{completed} Completed{RESET}"]
    if running > 0:
        summary_parts.append(f"{YELLOW}{running} Running{RESET}")
    if pending > 0:
        summary_parts.append(f"{BLUE}{pending} Pending{RESET}")
    if failed > 0:
        summary_parts.append(f"{RED}{failed} Failed{RESET}")
    if oom > 0:
        summary_parts.append(f"{RED}{oom} OOM{RESET}")
    if timeout > 0:
        summary_parts.append(f"{RED}{timeout} Timeout{RESET}")
    if cancelled > 0:
        summary_parts.append(f"{MAGENTA}{cancelled} Cancelled{RESET}")
    
    print(f"\nSummary: {', '.join(summary_parts)}")


def print_throughput_table(results):
    """Print throughput benchmark results (vllm bench throughput) grouped by model config."""
    from collections import defaultdict
    
    # ANSI color codes
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    CYAN = '\033[96m'
    MAGENTA = '\033[95m'
    BOLD = '\033[1m'
    RESET = '\033[0m'
    
    # Group results by model configuration (without job_id to consolidate)
    groups = defaultdict(list)
    for r in results:
        model = r.get('model', 'unknown')
        if '/' in model:
            model = model.split('/')[-1]
        
        key = (
            model,
            get_gpu_model(r),
            r.get('num_nodes', 1),
            get_gpus_per_node(r),
            r.get('tp_size', 1),
            r.get('pp_size', 1),
            r.get('ep_size', 1),
        )
        groups[key].append(r)
    
    # Print each group
    for (model, gpu, nodes, gpn, tp, pp, ep), items in sorted(groups.items()):
        # Collect unique job IDs and statuses
        job_ids = sorted(set(r.get('_job_id', '?') for r in items))
        statuses = [r.get('_status', 'Unknown') for r in items]
        completed = sum(1 for s in statuses if s == 'Completed')
        total_jobs = len(job_ids)
        
        # Group header
        model_short = model[:25] + "..." if len(model) > 25 else model
        print(f"\n{BOLD}{CYAN}{'─' * 100}{RESET}")
        print(f"{BOLD}Model: {model_short}{RESET}")
        print(f"  GPU: {gpu}  |  {nodes}N × {gpn}G  |  TP={tp}, PP={pp}, EP={ep}  |  Jobs: {total_jobs} ({GREEN}{completed} completed{RESET})")
        print(f"  Job IDs: {', '.join(job_ids)}")
        print(f"{CYAN}{'─' * 100}{RESET}")
        
        # Group by ISL/OSL/Prompts and calculate averages
        from collections import defaultdict
        import statistics
        
        isl_osl_groups = defaultdict(list)
        for r in items:
            key = (r.get('input_len', 0), r.get('output_len', 0), r.get('num_prompts', 0))
            isl_osl_groups[key].append(r)
        
        # Table header
        print(f"  {'ISL':>6} {'OSL':>6} {'Prompts':>8} {'Runs':>5} {'Req/s':>12} {'Tok/s (mean±std)':>22} {'Tok/s/GPU':>12}")
        print(f"  {'-'*6} {'-'*6} {'-'*8} {'-'*5} {'-'*12} {'-'*22} {'-'*12}")
        
        total_gpus = nodes * gpn
        current_isl = None
        
        for (isl, osl, prompts), runs in sorted(isl_osl_groups.items()):
            req_values = [float(r.get('throughput_requests_per_sec', 0)) for r in runs]
            tok_values = [float(r.get('throughput_tokens_per_sec', 0)) for r in runs]
            
            req_mean = statistics.mean(req_values)
            tok_mean = statistics.mean(tok_values)
            tok_std = statistics.stdev(tok_values) if len(tok_values) > 1 else 0
            tok_gpu = tok_mean / total_gpus if total_gpus > 0 else 0
            
            # Add visual separator between different ISL values
            if current_isl is not None and isl != current_isl:
                print()
            current_isl = isl
            
            if tok_std > 0:
                tok_str = f"{tok_mean:>12,.0f} ± {tok_std:>6,.0f}"
            else:
                tok_str = f"{tok_mean:>12,.0f}         "
            
            print(f"  {isl:>6} {osl:>6} {prompts:>8} {len(runs):>5} {req_mean:>12.2f} {tok_str:>22} {tok_gpu:>12,.0f}")
    
    # Overall summary
    print(f"\n{BOLD}{'=' * 100}{RESET}")
    
    completed = sum(1 for r in results if r.get('_status') == 'Completed')
    running = sum(1 for r in results if r.get('_status') == 'Running')
    failed = sum(1 for r in results if r.get('_status', '').startswith('Failed'))
    cancelled = sum(1 for r in results if r.get('_status') == 'Cancelled')
    
    # Unique configurations
    unique_configs = set()
    for r in results:
        unique_configs.add((r.get('input_len'), r.get('output_len')))
    
    summary_parts = [f"{GREEN}{completed} Completed{RESET}"]
    if running > 0:
        summary_parts.append(f"{YELLOW}{running} Running{RESET}")
    if failed > 0:
        summary_parts.append(f"{RED}{failed} Failed{RESET}")
    if cancelled > 0:
        summary_parts.append(f"{MAGENTA}{cancelled} Cancelled{RESET}")
    
    print(f"Summary: {', '.join(summary_parts)}")
    print(f"Total: {len(results)} results across {len(groups)} model configurations")
    print(f"ISL/OSL combinations: {len(unique_configs)}")


def print_grouped_table(results):
    """Print results grouped by configuration with mean/std."""
    from collections import defaultdict
    import statistics
    
    # Group by config
    groups = defaultdict(list)
    for r in results:
        key = get_config_key(r)
        groups[key].append(r)
    
    # Header
    print("\n" + "=" * 170)
    print(f"{'GPU':<8} {'Model':<22} {'N':>2} {'G/N':>3} {'TP':>3} {'PP':>3} {'EP':>3} {'DP':>3} {'GPUs':>5} {'Runs':>5} {'Tokens/s (mean±std)':>25} {'Tok/s/GPU':>12} {'Time(s)':>12}")
    print("=" * 170)
    
    # Calculate stats and sort by mean throughput
    stats = []
    for (gpu_model, model, nodes, gpus_per_node, tp, pp, dp, ep), group in groups.items():
        throughputs = [r.get('generation_throughput_tokens_per_sec', 0) for r in group]
        per_gpu = [r.get('tokens_per_sec_per_gpu', 0) for r in group]
        times = [r.get('total_time_sec', 0) for r in group]
        gpus = group[0].get('total_gpus', 0)
        
        mean_tp = statistics.mean(throughputs)
        std_tp = statistics.stdev(throughputs) if len(throughputs) > 1 else 0
        mean_per_gpu = statistics.mean(per_gpu)
        mean_time = statistics.mean(times)
        
        stats.append({
            'gpu_model': gpu_model,
            'model': model,
            'nodes': nodes,
            'gpus_per_node': gpus_per_node,
            'tp': tp,
            'pp': pp,
            'dp': dp,
            'ep': ep,
            'gpus': gpus,
            'runs': len(group),
            'mean_tp': mean_tp,
            'std_tp': std_tp,
            'mean_per_gpu': mean_per_gpu,
            'mean_time': mean_time,
            'job_ids': [r.get('_job_id', '?') for r in group]
        })
    
    # Sort by mean throughput
    stats_sorted = sorted(stats, key=lambda x: x['mean_tp'], reverse=True)
    
    for s in stats_sorted:
        model = s['model']
        if len(model) > 20:
            model = model[:17] + "..."
        gpu_model = (s['gpu_model'] or 'GB200')[:7]
        
        if s['std_tp'] > 0:
            tp_str = f"{s['mean_tp']:,.0f} ± {s['std_tp']:,.0f}"
        else:
            tp_str = f"{s['mean_tp']:,.0f}"
        
        print(f"{gpu_model:<8} "
              f"{model:<22} "
              f"{s['nodes']:>2} "
              f"{s['gpus_per_node']:>3} "
              f"{s['tp']:>3} "
              f"{s['pp']:>3} "
              f"{s['ep']:>3} "
              f"{s['dp']:>3} "
              f"{s['gpus']:>5} "
              f"{s['runs']:>5} "
              f"{tp_str:>25} "
              f"{s['mean_per_gpu']:>12,.0f} "
              f"{s['mean_time']:>12.1f}")
    
    print("=" * 170)
    
    # Print job IDs for reference
    print("\nJob IDs per configuration:")
    for s in stats_sorted:
        ep_str = f" EP={s['ep']}" if s['ep'] > 1 else ""
        print(f"  [{s['gpu_model'][:6]}] {s['model'][:18]} N={s['nodes']} G/N={s['gpus_per_node']} TP={s['tp']} PP={s['pp']}{ep_str} DP={s['dp']}: {', '.join(s['job_ids'])}")


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
    
    # Benchmark type (mutually exclusive)
    bench_type = parser.add_mutually_exclusive_group()
    bench_type.add_argument("--offline", action="store_true", help="Collect offline benchmark results (default)")
    bench_type.add_argument("--online", action="store_true", help="Collect online benchmark results")
    bench_type.add_argument("--throughput", "-t", action="store_true", help="Collect throughput benchmark results (vllm bench throughput)")
    
    parser.add_argument("--base-dir", help="Override base directory")
    parser.add_argument("--group", "-g", action="store_true", help="Group results by configuration and show mean±std")
    parser.add_argument("--all", "-a", action="store_true", help="Include running/failed jobs with status column")
    parser.add_argument("--grpo", action="store_true", help="Filter to only show GRPO configurations (from grpo_benchmark_sweep.sh)")
    parser.add_argument("--output", "-o", help="Output CSV file")
    parser.add_argument("--json", help="Output JSON file")
    
    args = parser.parse_args()
    
    # Determine base directory and benchmark type
    # Results are stored in parent directory (nemo-rl/vllm_standalone_perf_exp/)
    script_dir = Path(__file__).parent.resolve()
    parent_dir = script_dir.parent
    exp_base = parent_dir / "vllm_standalone_perf_exp"
    
    if args.base_dir:
        base_dir = args.base_dir
        bench_type_str = "custom"
    elif args.online:
        base_dir = str(exp_base / "online")
        bench_type_str = "online"
        print("Collecting ONLINE benchmark results...")
    elif args.throughput:
        base_dir = str(exp_base / "throughput")
        bench_type_str = "throughput"
        print("Collecting THROUGHPUT benchmark results (vllm bench throughput)...")
    else:
        base_dir = str(exp_base / "offline")
        bench_type_str = "offline"
        print("Collecting OFFLINE benchmark results...")
    
    # Get running jobs for status check
    running_jobs = get_running_jobs()
    
    # Find and load results
    if args.all:
        # Include running/failed jobs (jobs without results.json)
        job_dirs = find_all_job_dirs(args.job_ids or None, base_dir)
        print(f"Found {len(job_dirs)} job directories")
        results = load_all_jobs(job_dirs, running_jobs)
        print(f"Loaded {len(results)} jobs (including running/failed)")
    else:
        # Only completed jobs (with results.json)
        files = find_result_files(args.job_ids or None, base_dir)
        print(f"Found {len(files)} result files")
        results = load_results(files, running_jobs)
        print(f"Loaded {len(results)} results")
    
    # Filter to GRPO configs if requested
    if args.grpo:
        original_count = len(results)
        results = filter_grpo_results(results)
        print(f"Filtered to {len(results)} GRPO results (from {original_count} total)")
    
    # Print table
    print_table(results, group_by_config=args.group, bench_type=bench_type_str)
    
    # Save if requested
    if args.output:
        save_csv(results, args.output)
    
    if args.json:
        with open(args.json, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {args.json}")


if __name__ == "__main__":
    main()

