# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stream existing Nano TensorBoard scalars without changing the training job."""

import argparse
import fcntl
import json
import os
import re
import signal
import subprocess
import time
import types
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import wandb
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ACTIVE_STATES = {
    "PENDING",
    "RUNNING",
    "CONFIGURING",
    "COMPLETING",
    "SUSPENDED",
    "REQUEUED",
    "RESIZING",
    "SIGNALING",
    "STAGE_OUT",
}
G_STOP = False


def tensorboard_dir(run_dir: Path, job: str) -> Path | None:
    """Locate the experiment directory actually printed by this job's driver."""
    candidates = [run_dir / f"{job}-logs" / "ray-driver.log"]
    candidates.extend(run_dir.glob(f"{job}-*-logs/ray-driver.log"))
    found = set()
    for log in candidates:
        if not log.is_file():
            continue
        with log.open(errors="replace") as source:
            for line in source:
                if "Using log directory:" not in line:
                    continue
                name = line.split("Using log directory:", 1)[1].strip()
                name = re.sub(r"\x1b\[[0-9;]*m", "", name)
                path = (Path(name) / "tensorboard").resolve()
                if not path.is_relative_to(run_dir / "nemo"):
                    raise ValueError("Driver reported a log directory outside this run")
                found.add(path)
    if len(found) > 1:
        raise ValueError(f"Job {job} reported multiple experiment directories")
    return next(iter(found), None)


def request_stop(_signal: int, _frame: types.FrameType | None) -> None:
    """Let the exporter finish its current polling operation before stopping."""
    global G_STOP
    G_STOP = True


def save_state(path: Path, state: dict[str, Any]) -> None:
    """Replace the local import cursor file atomically."""
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n")
    temporary.replace(path)


def slurm_state(job: str) -> str:
    """Read the allocation's accounting state without changing Slurm jobs."""
    result = subprocess.run(
        ["sacct", "-X", "-j", job, "-n", "-P", "-o", "JobIDRaw,State"],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    for line in result.stdout.splitlines():
        fields = line.split("|")
        if fields[0] == job:
            return fields[1].split()[0].rstrip("+")
    raise ValueError(f"No Slurm state returned for {job}")


def submitted_jobs(run_dir: Path) -> list[str]:
    """Read numeric job IDs from the guarded launcher's submission history."""
    return [
        line.split("\t")[1]
        for line in (run_dir / "submissions.tsv").read_text().splitlines()
        if len(line.split("\t")) >= 2 and line.split("\t")[1].isdecimal()
    ]


def scalar_rows(
    accumulator: EventAccumulator, cursors: dict[str, int]
) -> tuple[list[dict[str, int | float]], dict[str, int]]:
    """Preserve source steps, including late tags and repeated same-step values."""
    rows = {}
    next_cursors = dict(cursors)
    for tag in accumulator.Tags().get("scalars", []):
        events = accumulator.Scalars(tag)
        start = cursors.get(tag, 0)
        if start > len(events):
            raise ValueError(f"TensorBoard scalar history shrank: {tag}")
        occurrences = defaultdict(int)
        axis = "ray_step" if tag.startswith("ray/") else "global_step"
        for event in events[start:]:
            ordinal = occurrences[event.step]
            occurrences[event.step] += 1
            key = (axis, event.step, ordinal)
            row = rows.setdefault(
                key, {axis: event.step, "_timestamp": event.wall_time}
            )
            row[tag] = event.value
            row["_timestamp"] = max(row["_timestamp"], event.wall_time)
        next_cursors[tag] = len(events)
    return sorted(rows.values(), key=lambda row: row["_timestamp"]), next_cursors


def driver_progress(run_dir: Path, job: str) -> dict[str, int | float | str]:
    """Report driver progress as summary metadata, separate from scalar history."""
    path = run_dir / f"{job}-logs/ray-driver.log"
    if not path.exists():
        return {}
    text = re.sub(r"\x1b\[[0-9;]*m", "", path.read_text(errors="replace"))
    result = {}
    steps = re.findall(r"=+ Step (\d+)/(\d+) =+", text)
    if steps:
        result["progress/started_training_step"] = int(steps[-1][0])
        result["progress/planned_training_steps"] = int(steps[-1][1])
    phases = re.findall(r"▶ ([^\r\n]+)", text)
    if phases:
        result["progress/driver_phase"] = phases[-1]
    rewards = re.findall(r"Rewards stats:.*?mean=([0-9.]+)", text)
    if rewards:
        result["progress/latest_batch_reward_mean_from_driver"] = float(rewards[-1])
    return result


def stream_job(
    run_dir: Path,
    job: str,
    state: dict[str, Any],
    output: Path,
    state_path: Path,
    interval: int,
    *,
    entity: str,
    project: str,
) -> None:
    """Export one job's events and drain the final flush after termination."""
    source = tensorboard_dir(run_dir, job)
    if source is None or not source.is_dir():
        return
    record = state["jobs"].setdefault(job, {"cursors": {}, "history_rows": 0})
    config = json.loads((run_dir / "run-manifest.json").read_text())
    config.pop("container_fingerprint", None)
    config.update(slurm_job_id=job, metrics_source="existing TensorBoard event files")
    accumulator = EventAccumulator(str(source), size_guidance={"scalars": 0})
    with wandb.init(
        entity=entity,
        project=project,
        id=f"nano35-{job}",
        resume="allow",
        name=f"{run_dir.name}/job-{job}",
        group=run_dir.name,
        job_type="e2e-training",
        tags=["live-tensorboard-import"],
        dir=str(output),
        config=config,
        mode="online",
        settings=wandb.Settings(
            disable_git=True, disable_code=True, x_disable_stats=True
        ),
    ) as remote:
        remote.define_metric("global_step")
        remote.define_metric("ray_step")
        remote.define_metric("*", step_metric="global_step", step_sync=False)
        remote.define_metric("ray/*", step_metric="ray_step", step_sync=False)
        remote.summary.update(
            {
                "source/sync_scope": f"live TensorBoard scalars; {interval}-second polling",
                "source/slurm_job_id": job,
                "source/tensorboard_dir": str(source),
                "source/gpu_step_unit": "seconds since NeMo GPU monitor started",
                "source/training_step_semantics": "original TensorBoard training step",
            }
        )
        record.update(status="streaming", url=remote.url, tensorboard_dir=str(source))
        (output / "current-run-url.txt").write_text(remote.url + "\n")
        terminal_seen = False
        while not G_STOP and not (output / "stop").exists():
            job_state = slurm_state(job)
            accumulator.Reload()
            rows, cursors = scalar_rows(accumulator, record["cursors"])
            for row in rows:
                remote.log(row)
            record["cursors"] = cursors
            record["history_rows"] += len(rows)
            now = datetime.now(timezone.utc).isoformat()
            train_steps = [
                event.step
                for tag in accumulator.Tags().get("scalars", [])
                if tag.startswith("train/")
                for event in accumulator.Scalars(tag)
            ]
            marker = run_dir / "attempts" / job / "driver-status.txt"
            driver_completed = (
                marker.exists() and marker.read_text().strip() == "completed"
            )
            remote.summary.update(
                {
                    **driver_progress(run_dir, job),
                    "source/slurm_state": job_state,
                    "source/last_sync_utc": now,
                    "source/scalar_points_imported": sum(cursors.values()),
                    "source/history_rows_imported": record["history_rows"],
                    "source/driver_and_checkpoint_checks_completed": driver_completed,
                    "progress/last_logged_training_step": max(train_steps, default=0),
                }
            )
            record.update(slurm_state=job_state, checked_at_utc=now)
            state["phase"] = "streaming_live_results"
            state.pop("last_error", None)
            save_state(state_path, state)
            print(
                json.dumps(
                    {
                        "job": job,
                        "slurm_state": job_state,
                        "new_history_rows": len(rows),
                        "scalar_points": sum(cursors.values()),
                        "url": remote.url,
                        "checked_at_utc": now,
                    }
                ),
                flush=True,
            )
            # Read once more after Slurm terminates to include the final file flush.
            if job_state not in ACTIVE_STATES:
                if terminal_seen:
                    remote.summary["source/sync_complete"] = True
                    remote.finish(exit_code=0 if job_state == "COMPLETED" else 1)
                    record["status"] = "synced"
                    save_state(state_path, state)
                    return
                terminal_seen = True
            else:
                terminal_seen = False
            for _ in range(interval):
                if G_STOP or (output / "stop").exists():
                    break
                time.sleep(1)


def load_state(path: Path, *, entity: str, project: str) -> dict[str, Any]:
    """Keep import cursors bound to one W&B destination across restarts."""
    destination = {"entity": entity, "project": project}
    if not path.exists():
        return {"destination": destination, "jobs": {}}
    state = json.loads(path.read_text())
    if state.get("destination") != destination:
        raise ValueError(
            "Saved W&B destination is missing or differs; use a separate run copy "
            "for a different destination. Do not reuse another exporter's cursors."
        )
    return state


def main() -> None:
    """Run one exporter, or inspect local events without uploading anything."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--entity", help="W&B team or user; required for uploads")
    parser.add_argument("--project", default="nano3.5-e2e")
    parser.add_argument("--interval", type=int, default=60)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    if not (run_dir / "submissions.tsv").is_file():
        parser.error("Pass an existing Nano experiment with submission records")
    if args.interval < 1:
        parser.error("--interval must be positive")
    output = run_dir / "wandb-sync"
    if args.dry_run:
        for job in submitted_jobs(run_dir):
            source = tensorboard_dir(run_dir, job)
            if source is not None and source.is_dir():
                accumulator = EventAccumulator(
                    str(source), size_guidance={"scalars": 0}
                ).Reload()
                rows, cursors = scalar_rows(accumulator, {})
                print(
                    json.dumps(
                        {
                            "job": job,
                            "tensorboard_dir": str(source),
                            "history_rows": len(rows),
                            "scalar_points": sum(cursors.values()),
                            "tags": len(cursors),
                            "training_rows": sum("global_step" in row for row in rows),
                            "progress": driver_progress(run_dir, job),
                        }
                    )
                )
        return
    if not args.entity:
        parser.error("--entity is required for uploads")
    output.mkdir(exist_ok=True)
    for name, value in {
        "WANDB_MODE": "online",
        "WANDB_CACHE_DIR": str(output / "cache"),
        "WANDB_DATA_DIR": str(output / "staging"),
    }.items():
        os.environ[name] = value
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    state_path = output / "live-state.json"
    with (output / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state = load_state(state_path, entity=args.entity, project=args.project)
        save_state(state_path, state)
        (output / "pid.txt").write_text(f"{os.getpid()}\n")
        while not G_STOP and not (output / "stop").exists():
            try:
                for job in submitted_jobs(run_dir):
                    if state["jobs"].get(job, {}).get("status") != "synced":
                        stream_job(
                            run_dir,
                            job,
                            state,
                            output,
                            state_path,
                            args.interval,
                            entity=args.entity,
                            project=args.project,
                        )
                    if G_STOP or (output / "stop").exists():
                        break
            except (
                OSError,
                ValueError,
                subprocess.SubprocessError,
                wandb.errors.Error,
            ) as error:
                state["last_error"] = f"{type(error).__name__}: {error}"
                save_state(state_path, state)
                print(json.dumps({"error": state["last_error"]}), flush=True)
            for _ in range(args.interval):
                if G_STOP or (output / "stop").exists():
                    break
                time.sleep(1)


if __name__ == "__main__":
    main()
