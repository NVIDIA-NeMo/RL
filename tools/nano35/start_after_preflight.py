# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check a preflight every two hours, then start fresh training and continuations."""

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def preflight_state(job: str, report_path: Path) -> str:
    """Require scheduler completion and a successful real GPU/SWE report."""
    if not job.isdecimal():
        raise ValueError("Invalid preflight job ID")
    queue = subprocess.check_output(
        ["squeue", "-h", "-u", str(os.getuid()), "-o", "%A|%T"], text=True
    )
    if any(row.split("|")[0] == job for row in queue.splitlines()):
        return "wait"
    records = subprocess.check_output(
        ["sacct", "-X", "-n", "-P", "-j", job, "--format=JobIDRaw,State"],
        text=True,
    )
    states = [
        row.split("|")[1] for row in records.splitlines() if row.split("|")[0] == job
    ]
    if states != ["COMPLETED"]:
        raise RuntimeError(f"Inspect preflight job {job}: {states}")
    report = json.loads(report_path.read_text())
    required = (
        "tp4_generation",
        "mnnvl_breakable",
        "conversation_reuse",
        "cache_reset",
        "swe_agent_and_evaluation",
    )
    if report.get("status") != "passed" or not all(
        report.get("checks", {}).get(key) == "passed" for key in required
    ):
        raise RuntimeError("GPU/SWE preflight has not passed all required checks")
    return "ready"


def bootstrap(job: str, report_path: Path, run: Path, *, once: bool = False) -> None:
    """Start only after validation; never queue a second job alongside it."""
    if not run.is_absolute() or not report_path.is_absolute():
        raise ValueError("Use absolute run and report paths")
    run.mkdir(parents=True, exist_ok=True)
    with (run / ".bootstrap.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        (run / "bootstrap.pid").write_text(f"{os.getpid()}\n")
        while not (run / "monitor.stop").exists():
            try:
                state = preflight_state(job, report_path)
            except (
                OSError,
                ValueError,
                KeyError,
                RuntimeError,
                subprocess.CalledProcessError,
            ) as error:
                state = "stopped_on_error"
                result = {"action": state, "error": str(error)}
            else:
                result = {"action": state, "preflight_job_id": job}
            result["checked_at_utc"] = datetime.now(timezone.utc).isoformat()
            with (run / "bootstrap-events.jsonl").open("a") as output:
                output.write(json.dumps(result) + "\n")
            print(json.dumps(result), flush=True)
            if state == "stopped_on_error":
                return
            if state == "ready":
                os.environ["NANO35_PREFLIGHT_REPORT"] = str(report_path)
                if not (run / "job-id.txt").exists():
                    # The launcher verifies image hash, assets and the whole
                    # Nano queue again, then takes its submission lock.
                    subprocess.run(
                        [
                            "bash",
                            str(Path(__file__).with_name("launch_e2e.sh")),
                            "fresh",
                            str(run),
                        ],
                        check=True,
                    )
                subprocess.run(
                    [
                        sys.executable,
                        str(Path(__file__).with_name("continue_training.py")),
                        str(run),
                        *(["--once"] if once else []),
                    ],
                    check=True,
                )
                return
            if once:
                return
            time.sleep(7200)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight-job", required=True)
    parser.add_argument("--report", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    bootstrap(args.preflight_job, args.report, args.run_dir, once=args.once)
