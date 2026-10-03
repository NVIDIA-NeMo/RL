#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.13"
# dependencies = ["ray==2.58.0", "matplotlib==3.11.2"]
# ///
"""Temporary RL-1610 actor/process-tree measurements. Remove after validation."""

import argparse
import asyncio
import json
import os
import signal
import socket
import statistics
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path


def read_process(path: Path) -> dict:
    fields = (path / "stat").read_text().rpartition(")")[2].split()
    memory = {}
    try:
        for line in (path / "smaps_rollup").read_text().splitlines():
            key, _, value = line.partition(":")
            if key in {"Rss", "Pss"}:
                memory[key] = int(value.split()[0]) * 1024
    except (FileNotFoundError, PermissionError):
        pass
    return {
        "pid": int(path.name),
        "ppid": int(fields[1]),
        "cpu_ticks": int(fields[11]) + int(fields[12]),
        "start_ticks": int(fields[19]),
        "rss_bytes": memory.get("Rss"),
        "pss_bytes": memory.get("Pss"),
    }


def assign_processes(
    processes: dict[int, dict], roots: dict[int, str]
) -> dict[int, str]:
    owners = {}
    for pid in processes:
        ancestor = pid
        visited = set()
        while ancestor in processes and ancestor not in visited:
            if ancestor in roots:
                owners[pid] = roots[ancestor]
                break
            visited.add(ancestor)
            ancestor = processes[ancestor]["ppid"]
    return owners


class NodeSampler:
    def __init__(self) -> None:
        self.previous: dict[tuple[int, int], tuple[int, float]] = {}
        self.previous_node_cpu: tuple[int, float] | None = None

    def sample(self, actors: dict[str, dict]) -> dict:
        started = time.monotonic()
        timestamp = time.time()
        hz = os.sysconf("SC_CLK_TCK")
        cpu_fields = [
            int(value)
            for value in Path("/proc/stat").read_text().splitlines()[0].split()[1:9]
        ]
        active_ticks = sum(cpu_fields) - cpu_fields[3] - cpu_fields[4]
        node_cpu_cores = None
        if self.previous_node_cpu is not None:
            prior_ticks, prior_time = self.previous_node_cpu
            node_cpu_cores = (active_ticks - prior_ticks) / hz / (started - prior_time)
        self.previous_node_cpu = active_ticks, started
        boot_time = next(
            int(line.split()[1])
            for line in Path("/proc/stat").read_text().splitlines()
            if line.startswith("btime ")
        )
        processes = {}
        inaccessible = []
        for path in Path("/proc").glob("[0-9]*"):
            try:
                processes[int(path.name)] = read_process(path)
            except (FileNotFoundError, ProcessLookupError):
                continue
            except (PermissionError, ValueError, IndexError):
                inaccessible.append(int(path.name))
        valid_roots = {
            record["pid"]: actor_id
            for actor_id, record in actors.items()
            if record["pid"] in processes
            and (boot_time + processes[record["pid"]]["start_ticks"] / hz) * 1000
            <= record["start_time_ms"]
        }
        owners = assign_processes(processes, valid_roots)
        samples = []
        for actor_id, actor in actors.items():
            owned = [
                processes[pid] for pid, owner in owners.items() if owner == actor_id
            ]
            row = {**actor, "actor_id": actor_id, "processes": owned, "cpu_cores": 0.0}
            row["root_available"] = actor["pid"] in valid_roots
            memory_complete = bool(owned) and all(
                process["pss_bytes"] is not None for process in owned
            )
            row["pss_bytes"] = (
                sum(process["pss_bytes"] for process in owned)
                if memory_complete
                else None
            )
            row["rss_bytes"] = (
                sum(process["rss_bytes"] for process in owned)
                if memory_complete
                else None
            )
            known_cpu = bool(owned)
            for process in owned:
                key = (process["pid"], process["start_ticks"])
                previous = self.previous.get(key)
                if previous is None:
                    known_cpu = False
                else:
                    ticks, prior_time = previous
                    row["cpu_cores"] += (
                        (process["cpu_ticks"] - ticks) / hz / (started - prior_time)
                    )
            if not known_cpu:
                row["cpu_cores"] = None
            samples.append(row)
        self.previous = {
            (pid, process["start_ticks"]): (process["cpu_ticks"], started)
            for pid, process in processes.items()
        }
        memory = {
            key.removesuffix(":"): int(value.split()[0]) * 1024
            for key, value in (
                line.split(":", 1)
                for line in Path("/proc/meminfo").read_text().splitlines()
            )
        }
        own = processes.get(os.getpid(), {})
        return {
            "timestamp": timestamp,
            "hostname": socket.gethostname(),
            "actors": samples,
            "node_os_used_bytes": memory["MemTotal"] - memory["MemAvailable"],
            "node_os_total_bytes": memory["MemTotal"],
            "node_cpu_cores": node_cpu_cores,
            "inaccessible_pids": inaccessible,
            "sampler_pss_bytes": own.get("pss_bytes"),
            "sample_elapsed_s": time.monotonic() - started,
        }


def prepare_output(path: str) -> Path:
    output = Path(path).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError("Use an empty output directory for each measurement run")
    return output


@contextmanager
def running_driver(command: list[str], **kwargs):
    def terminate(signum, _frame):
        raise SystemExit(128 + signum)

    previous = signal.signal(signal.SIGTERM, terminate)
    child = None
    try:
        child = subprocess.Popen(command, start_new_session=True, **kwargs)
        yield child
    finally:
        try:
            if child is not None and child.poll() is None:
                os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid, signal.SIGKILL)
                    child.wait()
        finally:
            signal.signal(signal.SIGTERM, previous)


def record(args: argparse.Namespace) -> int:
    # Ray is needed only for live Linux cluster collection.
    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    output = prepare_output(args.output)
    ray.init(address="auto", log_to_driver=False)
    own_job_id = ray.get_runtime_context().get_job_id()
    samplers = {
        node["NodeID"]: ray.remote(num_cpus=0)(NodeSampler)
        .options(
            scheduling_strategy=NodeAffinitySchedulingStrategy(
                node["NodeID"], soft=False
            )
        )
        .remote()
        for node in ray.nodes()
        if node["Alive"]
    }

    async def actor_inventory() -> dict:
        return (
            await ray._private.worker.global_worker.gcs_client.async_get_all_actor_info(
                actor_state_name="ALIVE", timeout=30
            )
        )

    env = {**os.environ, "NEMO_RL_RL1610_PROFILE_DIR": str(output)}
    with (
        (output / "driver.log").open("w") as driver_log,
        (output / "actors.jsonl").open("w") as samples,
    ):
        driver = running_driver(
            args.command,
            stdout=driver_log,
            stderr=subprocess.STDOUT,
            env=env,
        )
        try:
            with driver as child:
                return collect(
                    args, child, samplers, own_job_id, actor_inventory, samples
                )
        finally:
            for sampler in samplers.values():
                ray.kill(sampler)
            ray.shutdown()


def collect(args, child, samplers, own_job_id, actor_inventory, samples):
    import ray

    while child.poll() is None:
        started = time.monotonic()
        try:
            per_node = {node_id: {} for node_id in samplers}
            for actor in asyncio.run(actor_inventory()).values():
                if actor.job_id.hex() == own_job_id:
                    continue
                node_id = actor.node_id.hex()
                if node_id in per_node:
                    per_node[node_id][actor.actor_id.hex()] = {
                        "pid": actor.pid,
                        "name": actor.name,
                        "class_name": actor.class_name,
                        "job_id": actor.job_id.hex(),
                        "start_time_ms": actor.start_time,
                        "resources": dict(actor.required_resources),
                    }
            rows = ray.get(
                [
                    sampler.sample.remote(per_node[node_id])
                    for node_id, sampler in samplers.items()
                ],
                timeout=30,
            )
            elapsed = time.monotonic() - started
            for row in rows:
                row["collection_elapsed_s"] = elapsed
                samples.write(json.dumps(row) + "\n")
            samples.flush()
        except Exception as error:
            samples.write(
                json.dumps({"timestamp": time.time(), "collection_error": str(error)})
                + "\n"
            )
            samples.flush()
        time.sleep(max(0, args.interval - (time.monotonic() - started)))
    return child.wait()


def plot(args: argparse.Namespace) -> int:
    # Plotting is separate from collection so it never runs in the measurement loop.
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    samples = [
        json.loads(line)
        for line in (Path(args.output) / "actors.jsonl").read_text().splitlines()
    ]
    steps = [
        json.loads(line)
        for line in (Path(args.output) / "steps.jsonl").read_text().splitlines()
    ]
    steady = [step for step in steps if step["step"] > args.warmup_steps]
    if len(steady) < 2:
        raise ValueError("Need at least two post-warmup steps")
    start = steady[0]["timestamp"] - steady[0]["total_step_time_s"]
    end = steady[-1]["timestamp"]
    samples = [sample for sample in samples if start <= sample["timestamp"] <= end]
    times = [step["total_step_time_s"] for step in steady]
    summary = {
        "steady_steps": len(times),
        "step_mean_s": statistics.mean(times),
        "step_stdev_s": statistics.stdev(times),
        "step_min_s": min(times),
        "step_max_s": max(times),
        "warmup_steps": args.warmup_steps,
        "collection_errors": sum("collection_error" in sample for sample in samples),
    }
    figure, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    series = {}
    nodes = {}
    for sample in samples:
        if "collection_error" in sample:
            continue
        host = sample["hostname"]
        for actor in sample["actors"]:
            label = (
                f"{host}/{actor['name'] or actor['class_name']}/{actor['actor_id'][:8]}"
            )
            series.setdefault(label, []).append(
                (sample["timestamp"] - start, actor["pss_bytes"], actor["cpu_cores"])
            )
        nodes.setdefault(host, []).append(
            (sample["timestamp"] - start, sample["node_os_used_bytes"] / 2**30)
        )
    for host, points in nodes.items():
        axes[2].plot(*zip(*points), label=host)
    for label, points in series.items():
        axes[0].plot(
            [point[0] for point in points],
            [
                float("nan") if point[1] is None else point[1] / 2**30
                for point in points
            ],
            label=label,
        )
        axes[1].plot(
            [point[0] for point in points],
            [float("nan") if point[2] is None else point[2] for point in points],
            label=label,
        )
    axes[0].set_ylabel("Actor tree PSS (GiB)")
    axes[1].set_ylabel("Actor tree CPU cores")
    axes[2].set_ylabel("Node OS used (GiB)")
    axes[2].set_xlabel("Seconds in measured window")
    axes[0].legend(fontsize=6, bbox_to_anchor=(1.02, 1), loc="upper left")
    axes[2].legend(fontsize=6, bbox_to_anchor=(1.02, 1), loc="upper left")
    for axis in axes:
        axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(Path(args.output) / "actor-resources.png", dpi=150)
    (Path(args.output) / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="mode", required=True)
    collector = subcommands.add_parser("record")
    collector.add_argument("--output", required=True)
    collector.add_argument("--interval", type=float, default=1.0)
    collector.add_argument("command", nargs=argparse.REMAINDER)
    plotting = subcommands.add_parser("plot")
    plotting.add_argument("--output", required=True)
    plotting.add_argument("--warmup-steps", type=int, default=5)
    args = parser.parse_args()
    if args.mode == "record":
        if args.command and args.command[0] == "--":
            args.command.pop(0)
        if not args.command or args.interval <= 0:
            parser.error("record requires a command and a positive interval")
        return record(args)
    return plot(args)


if __name__ == "__main__":
    raise SystemExit(main())
