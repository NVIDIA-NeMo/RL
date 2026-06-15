#!/usr/bin/env python3
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

"""Reset all worker state across a Ray cluster.

Kills all Ray actors (GPU and CPU-only), removes placement groups, kills
GPU-resident processes on every node, and then sweeps for orphan Ray worker
processes. Intended for use between interactive training runs on a persistent
Slurm allocation.

Usage (from inside the head node container):
    python reset_ray_cluster.py

    # With a custom number of GPUs to verify (default: auto-detect from Ray)
    python reset_ray_cluster.py --num-gpus 256

    # Limit sweep duration
    python reset_ray_cluster.py --timeout 30

    # Skip GPU verification (useful if you only care about killing workers)
    python reset_ray_cluster.py --skip-gpu-check

Typical interactive workflow:
    1. Kill or Ctrl+C the current training run
    2. python reset_ray_cluster.py
    3. source <jobid>-run-cmd.sh   # re-run training
"""

import argparse
import time

import ray


def _kill_all_actors_and_placement_groups():
    """Use Ray APIs to tear down all live actors and placement groups."""
    from ray.util.placement_group import remove_placement_group
    from ray.util.state import list_actors, list_placement_groups

    # --- Placement groups ---
    pgs = list_placement_groups(filters=[("state", "=", "CREATED")])
    if pgs:
        print(f"  Removing {len(pgs)} placement group(s)...")
        for pg_info in pgs:
            try:
                pg = ray.util.placement_group.get_placement_group(pg_info["name"])
                remove_placement_group(pg)
            except Exception:
                pass

    # --- Actors ---
    actors = list_actors(filters=[("state", "=", "ALIVE")])
    if actors:
        print(f"  Killing {len(actors)} actor(s)...")
        for actor_info in actors:
            try:
                handle = ray.get_actor(actor_info["name"])
                ray.kill(handle, no_restart=True)
            except Exception:
                pass

    killed_count = len(actors) if actors else 0
    pg_count = len(pgs) if pgs else 0
    return killed_count, pg_count


def main():
    parser = argparse.ArgumentParser(
        description="Reset all worker state (GPU + CPU) across a Ray cluster"
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=None,
        help="Number of GPUs to verify (default: auto-detect from Ray cluster resources)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=60,
        help="Max seconds to spend sweeping for stale processes (default: 60)",
    )
    parser.add_argument(
        "--skip-gpu-check",
        action="store_true",
        help="Skip the GPU memory verification phase",
    )
    args = parser.parse_args()

    ray.init()

    cluster_gpus = int(ray.cluster_resources().get("GPU", 0))
    cluster_cpus = int(ray.cluster_resources().get("CPU", 0))
    num_gpus = args.num_gpus or cluster_gpus
    print(f"Cluster has {cluster_cpus} CPUs, {cluster_gpus} GPUs; will verify {num_gpus} GPUs")

    # -------------------------------------------------------------------------
    # Phase 1: Kill all Ray actors and placement groups via the Ray API.
    # This catches CPU-only workers (e.g. NemoGym) that don't appear in
    # nvidia-smi.
    # -------------------------------------------------------------------------
    print("\n=== Phase 1: Killing Ray actors & placement groups ===")
    actors_killed, pgs_removed = _kill_all_actors_and_placement_groups()
    print(f"  Actors killed: {actors_killed}, placement groups removed: {pgs_removed}")
    if actors_killed > 0:
        print("  Waiting 5s for processes to exit...")
        time.sleep(5)

    # -------------------------------------------------------------------------
    # Phase 2: Kill stale GPU processes on every node via nvidia-smi.
    # Uses SPREAD scheduling with minimal CPU to land on every node. We launch
    # more tasks than nodes to ensure full coverage even with scheduling jitter.
    # -------------------------------------------------------------------------
    print("\n=== Phase 2: Sweeping GPU processes (nvidia-smi) ===")

    @ray.remote(num_cpus=0.01, scheduling_strategy="SPREAD")
    def kill_gpu_processes():
        import os
        import subprocess

        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            capture_output=True,
            text=True,
        )
        pids = [p.strip() for p in result.stdout.strip().split("\n") if p.strip()]
        my_pid = os.getpid()
        my_ppid = os.getppid()
        killed = []
        for pid in pids:
            pid_int = int(pid)
            if pid_int not in (my_pid, my_ppid):
                try:
                    os.kill(pid_int, 9)
                    killed.append(pid)
                except Exception:
                    pass
        return f"{os.uname().nodename}: killed GPU PIDs {killed}"

    num_sweep_tasks = max(num_gpus, 256)
    end_time = time.time() + args.timeout
    sweep = 0
    while time.time() < end_time:
        sweep += 1
        print(f"\n--- Sweep {sweep} ---")
        futures = [kill_gpu_processes.remote() for _ in range(num_sweep_tasks)]
        results = ray.get(futures)
        found_any = False
        for r in results:
            if "killed GPU PIDs []" not in r:
                print(r)
                found_any = True
        if not found_any:
            print("All GPUs clean!")
            break
        remaining = int(end_time - time.time())
        print(f"Sleeping 10s... ({remaining}s remaining)")
        time.sleep(10)

    # -------------------------------------------------------------------------
    # Phase 3: Kill orphan Ray worker processes on every node.
    # Some CPU-only workers (or leaked subprocesses) may survive actor teardown.
    # We look for processes whose cmdline contains "ray::".
    # -------------------------------------------------------------------------
    print("\n=== Phase 3: Sweeping orphan Ray worker processes ===")

    @ray.remote(num_cpus=0.01, scheduling_strategy="SPREAD")
    def kill_orphan_ray_workers():
        import os
        import signal

        my_pid = os.getpid()
        killed = []
        for entry in os.listdir("/proc"):
            if not entry.isdigit():
                continue
            pid = int(entry)
            if pid == my_pid:
                continue
            try:
                with open(f"/proc/{pid}/cmdline", "rb") as f:
                    cmdline = f.read().decode("utf-8", errors="replace")
                if "ray::" in cmdline and "reset_cluster" not in cmdline:
                    os.kill(pid, signal.SIGKILL)
                    killed.append(pid)
            except (ProcessLookupError, PermissionError, FileNotFoundError):
                pass
        return f"{os.uname().nodename}: killed {len(killed)} orphan ray worker(s)"

    futures = [kill_orphan_ray_workers.remote() for _ in range(num_sweep_tasks)]
    results = ray.get(futures)
    for r in results:
        if "killed 0 orphan" not in r:
            print(r)

    # -------------------------------------------------------------------------
    # Phase 4: Verify GPU memory is actually freed on all GPUs.
    # Requests one GPU per task so Ray schedules exactly one per device.
    # -------------------------------------------------------------------------
    if not args.skip_gpu_check and num_gpus > 0:
        print(f"\n=== Phase 4: Verifying memory on {num_gpus} GPUs ===")

        @ray.remote(num_gpus=1)
        def check_gpu_memory():
            import os
            import subprocess

            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
            )
            return f"{os.uname().nodename}: {result.stdout.strip()} MiB used"

        checks = ray.get([check_gpu_memory.remote() for _ in range(num_gpus)])
        for c in checks:
            print(c)

    print("\nCluster reset complete.")


if __name__ == "__main__":
    main()
