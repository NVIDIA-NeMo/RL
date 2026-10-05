# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check one experiment every two hours and resume only a completed healthy job."""

import argparse
import fcntl
import json
import os
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


def check_once(run: Path, launcher: Path) -> dict[str, str]:
    """Inspect Slurm and checkpoint evidence before calling the guarded launcher."""
    job = (run / "job-id.txt").read_text().strip()
    if not job.isdecimal():
        raise ValueError("The recorded Slurm job ID is invalid")
    # Query the user's queue: squeue -j returns an error once a completed job
    # ages out of its live table, before sacct is consulted.
    queue = subprocess.check_output(
        ["squeue", "-h", "-u", str(os.getuid()), "-o", "%A|%T"], text=True
    )
    active = " ".join(
        row.split("|")[1] for row in queue.splitlines() if row.split("|")[0] == job
    )
    if active:
        return {"action": "wait", "job_id": job, "state": active}
    records = subprocess.check_output(
        ["sacct", "-X", "-n", "-P", "-j", job, "--format=JobIDRaw,State"],
        text=True,
    )
    states = [
        row.split("|")[1] for row in records.splitlines() if row.split("|")[0] == job
    ]
    if states != ["COMPLETED"]:
        raise RuntimeError(f"Inspect job {job} before continuing: {states}")
    attempt = run / "attempts" / job
    if (attempt / "driver-status.txt").read_text().strip() != "completed":
        raise RuntimeError("Training or checkpoint verification did not complete")
    report = json.loads((attempt / "checkpoint-report.json").read_text())
    if report["status"] != "metadata-passed":
        raise RuntimeError("Checkpoint metadata verification failed")
    if report["training_complete"]:
        return {"action": "finished", "job_id": job, "state": "COMPLETED"}
    # The launcher independently rechecks all jobs, image/data/config identity,
    # checkpoint files and the single-submission lock immediately before sbatch.
    subprocess.run(["bash", str(launcher), "resume", str(run)], check=True)
    next_job = (run / "job-id.txt").read_text().strip()
    if not next_job.isdecimal() or next_job == job:
        raise RuntimeError("The launcher did not record a new Slurm job ID")
    return {"action": "submitted", "previous_job_id": job, "job_id": next_job}


def monitor(run: Path, *, once: bool) -> None:
    """Hold one monitor lock and stop on failures or an explicit stop file."""
    if not run.is_absolute() or not run.is_dir():
        raise ValueError("Pass the absolute existing experiment directory")
    launcher = Path(__file__).with_name("launch_e2e.sh")
    with (run / ".monitor.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (run / "monitor.pid").write_text(f"{os.getpid()}\n")
        while not (run / "monitor.stop").exists():
            try:
                result = check_once(run, launcher)
            except (
                OSError,
                ValueError,
                KeyError,
                RuntimeError,
                subprocess.CalledProcessError,
            ) as error:
                result = {"action": "stopped_on_error", "error": str(error)}
            result["checked_at_utc"] = datetime.now(timezone.utc).isoformat()
            with (run / "monitor-events.jsonl").open("a") as output:
                output.write(json.dumps(result) + "\n")
            print(json.dumps(result), flush=True)
            if once or result["action"] in ("finished", "stopped_on_error"):
                return
            time.sleep(7200)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    monitor(args.run_dir, once=args.once)
