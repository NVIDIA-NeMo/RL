#!/usr/bin/env python3
"""Parse vllm bench serve output and extract metrics."""

import re
import sys

def parse_result(filepath):
    """Parse vllm bench serve output file."""
    try:
        with open(filepath) as f:
            content = f.read()
        
        def extract(pattern, default='0'):
            m = re.search(pattern, content)
            return m.group(1) if m else default
        
        # Updated patterns to match actual vllm bench serve output format:
        # "Request throughput (req/s):              3.42"
        # "Output token throughput (tok/s):         6842.11"
        # Also support older format without "(req/s)" and "(tok/s)"
        req_tp = extract(r'Request throughput[^:]*:\s*([0-9.]+)')
        out_tp = extract(r'Output token throughput[^:]*:\s*([0-9.]+)')
        ttft = extract(r'Mean TTFT \(ms\):\s*([0-9.]+)')
        itl = extract(r'Mean ITL \(ms\):\s*([0-9.]+)')
        
        return req_tp, out_tp, ttft, itl
    except Exception as e:
        print(f"Error parsing {filepath}: {e}", file=sys.stderr)
        return '0', '0', '0', '0'

if __name__ == "__main__":
    if len(sys.argv) < 11:
        print("Usage: parse_online_result.py <result_file> <isl> <osl> <concurrency> <num_prompts> <gpu_model> <nodes> <gpus_per_node> <total_gpus> <tp> <pp>")
        sys.exit(1)
    
    filepath = sys.argv[1]
    isl = sys.argv[2]
    osl = sys.argv[3]
    concurrency = sys.argv[4]
    num_prompts = sys.argv[5]
    gpu_model = sys.argv[6]
    nodes = sys.argv[7]
    gpus_per_node = sys.argv[8]
    total_gpus = sys.argv[9]
    tp = sys.argv[10]
    pp = sys.argv[11] if len(sys.argv) > 11 else "1"
    
    req_tp, out_tp, ttft, itl = parse_result(filepath)
    print(f"{gpu_model},{nodes},{gpus_per_node},{total_gpus},{tp},{pp},{isl},{osl},{concurrency},{num_prompts},{req_tp},{out_tp},{ttft},{itl}")
