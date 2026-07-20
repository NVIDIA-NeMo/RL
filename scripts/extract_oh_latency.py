#!/usr/bin/env python3
"""Extract REAL per-request LLM latency + per-action execution latency from retained
OpenHands output.jsonl trajectories (metrics.response_latencies / action_execution_latencies).
No gym patch — these are already logged. Uses a structural regex to skip the huge history.
Usage: python3 extract_oh_latency.py <outdir> <run>=<swebench_results_dir> [...]
"""
import csv
import glob
import re
import sys
import time

# response_latencies entry: {"model": "...", "latency": N, "response_id": "..."}
RESP = re.compile(r'"model":\s*"[^"]*",\s*"latency":\s*([0-9.eE+-]+),\s*"response_id"')
# action_execution_latencies entry: {"observation_type": "X", "observation_id": "..", "latency": N, ...}
ACT = re.compile(r'"observation_type":\s*"([^"]+)",\s*"observation_id":\s*"[^"]*",\s*"latency":\s*([0-9.eE+-]+)')


def main(outdir, args):
    fr = csv.writer(open(f"{outdir}/per_request_latency.csv", "w", newline="")); fr.writerow(["run", "latency_s"])
    fa = csv.writer(open(f"{outdir}/per_action_latency.csv", "w", newline="")); fa.writerow(["run", "obs_type", "latency_s"])
    t0 = time.time()
    for a in args:
        run, d = a.split("=", 1)
        files = glob.glob(d + "/*/trajectories/*/output.jsonl")
        nr = na = 0
        for f in files:
            txt = open(f, errors="ignore").read()
            for m in RESP.finditer(txt):
                fr.writerow([run, m.group(1)]); nr += 1
            for m in ACT.finditer(txt):
                fa.writerow([run, m.group(1), m.group(2)]); na += 1
        print(f"  {run}: {len(files)} trajs -> {nr} requests, {na} actions ({time.time()-t0:.0f}s)", flush=True)
    print(f"DONE {time.time()-t0:.0f}s -> {outdir}/per_request_latency.csv, per_action_latency.csv")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
