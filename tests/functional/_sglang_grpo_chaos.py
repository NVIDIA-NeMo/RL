# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Real GRPO fault injection; no replacement of the trainer or production methods.

Two colocated GPUs host two TP=1 SGLang engines and the DTensor trainer. The
observer attaches to whichever Ray cluster the training joined or started.
The six engine-only fault-tolerance pytest cases remain in L0.
"""

import argparse
import json
import os
import re
import signal
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

# Match nemo_rl's driver setup before Ray caches this flag at import time.
# The observer needs no remote runtime environment or repository upload.
os.environ["RAY_ENABLE_UV_RUN_RUNTIME_ENV"] = "0"

import psutil
import ray
import requests

from tests.functional._find_generation_actors import _address_from_session


@dataclass(frozen=True)
class Engine:
    actor_pid: int
    actor_created: float
    url: str


@dataclass(frozen=True)
class KillReceipt:
    engine: Engine
    weight_version: int
    running_requests: int
    completed_step: int
    timestamp: float


@dataclass(frozen=True)
class ReplacementReceipt:
    engine: Engine
    weight_version: int
    running_requests: int
    completed_step: int
    timestamp: float


def completed_steps(log: str) -> list[int]:
    """Count actual training results, not the step banner before generation."""
    current = 0
    completed = []
    for line in log.splitlines():
        banner = re.search(r"={5,} Step (\d+)/\d+ ={5,}", line)
        if banner:
            current = int(banner[1])
        if "📊 Training Results:" in line and current:
            completed.append(current)
    return completed


def engine_endpoints(log: str) -> dict[int, str]:
    """Read endpoints advertised by the actual actors, not guessed port ranges."""
    clean = re.sub(r"\x1b\[[0-9;]*m", "", log)
    return {
        int(pid): f"http://{address}"
        for pid, address in re.findall(
            r"SGLangGenerationWorker pid=(\d+)[^\n]*"
            r"Launch HttpServerEngineAdapter at: (\S+)",
            clean,
        )
    }


def needs_actor_observation(
    expect: str,
    kill: KillReceipt | None,
    replacement: ReplacementReceipt | None,
) -> bool:
    return kill is None or (expect == "survival" and replacement is None)


def validate_outcome(
    *,
    expect: str,
    returncode: int,
    log_after_kill: str,
    completed: list[int],
    max_steps: int,
    kill: KillReceipt,
    replacement: ReplacementReceipt | None,
) -> None:
    assert kill.completed_step >= 1, "Fault preceded the first completed train step"
    assert kill.running_requests >= 2, "Victim was not observably serving training"
    assert kill.weight_version > 0, "Victim had no trainer-transferred weights"
    if expect == "bounded_failure":
        assert returncode != 0, "Restart-budget exhaustion unexpectedly succeeded"
        assert re.search(
            r"SGLang engines \[[0-9, ]+\] exhausted "
            r"rollout_max_restart_attempts=0; aborting refit\.",
            log_after_kill,
        ), "Run failed for a reason other than the expected restart-budget error"
        assert "Restarting SGLang engine" not in log_after_kill
        return

    assert expect == "survival", f"Unsupported expectation: {expect}"
    assert returncode == 0, f"Training failed with exit code {returncode}"
    assert replacement is not None, "No replacement received weights and served"
    assert replacement.engine != kill.engine, "Victim was mistaken for replacement"
    assert replacement.weight_version > kill.weight_version, "Stale replacement weights"
    assert replacement.running_requests >= 2, "Replacement never served training"
    assert "Restarting SGLang engine" in log_after_kill, "No recovery was recorded"
    assert completed == list(range(1, max_steps + 1)), (
        "Training did not finish all steps"
    )
    later = [step for step in completed if step > replacement.completed_step]
    assert len(later) >= 2, "Fewer than two train steps completed after replacement"


def live_engines(log: str) -> list[Engine]:
    # Heavy/optional Ray is needed only by the GPU harness, not its parser tests.
    import ray._private.state as ray_state

    endpoints = engine_endpoints(log)
    engines = []
    for actor in ray_state.actors().values():
        pid = actor.get("Pid", 0)
        if (
            actor.get("ActorClassName") == "SGLangGenerationWorker"
            and actor.get("State") == "ALIVE"
            and pid in endpoints
        ):
            try:
                engines.append(
                    Engine(pid, psutil.Process(pid).create_time(), endpoints[pid])
                )
            except psutil.NoSuchProcess:
                continue
    return engines


def serving_state(http: requests.Session, engine: Engine) -> tuple[int, int]:
    # Pinned SGLang 3003d70f: /model_info exposes the trainer-supplied
    # weight_version; /get_load reports running + waiting requests per DP rank.
    info = http.get(f"{engine.url}/model_info", timeout=2)
    info.raise_for_status()
    raw_version = info.json()["weight_version"]
    version = int(raw_version) if str(raw_version).isdigit() else 0
    load = http.get(f"{engine.url}/get_load", timeout=2)
    load.raise_for_status()
    running = sum(row["num_reqs"] - row["num_waiting_reqs"] for row in load.json())
    return version, running


def capture_children(
    processes: dict[tuple[int, float], psutil.Process], pid: int
) -> None:
    try:
        parent = psutil.Process(pid)
        for process in [parent, *parent.children(recursive=True)]:
            processes[(process.pid, process.create_time())] = process
    except psutil.NoSuchProcess:
        pass


def cleanup(processes: dict[tuple[int, float], psutil.Process]) -> None:
    """Reap only captured descendants of this invocation, guarding PID reuse."""
    for process in list(processes.values()):
        if process.is_running():
            capture_children(processes, process.pid)
    live = [process for process in processes.values() if process.is_running()]
    for process in live:
        try:
            process.terminate()
        except psutil.NoSuchProcess:
            pass
    _, remaining = psutil.wait_procs(live, timeout=15)
    for process in remaining:
        try:
            process.kill()
        except psutil.NoSuchProcess:
            pass
    _, remaining = psutil.wait_procs(remaining, timeout=15)
    surviving_pids = []
    for process in remaining:
        try:
            if process.is_running() and process.status() != psutil.STATUS_ZOMBIE:
                surviving_pids.append(process.pid)
        except psutil.NoSuchProcess:
            pass
    assert not surviving_pids, (
        f"Training descendants survived cleanup: {surviving_pids}"
    )


def connect_ray() -> None:
    """Attach like _find_generation_actors.py: RAY_ADDRESS or auto, then the session dir."""
    try:
        ray.init(
            address=os.environ.get("RAY_ADDRESS") or "auto",
            log_to_driver=False,
            include_dashboard=False,
        )
    except ConnectionError:
        address = _address_from_session()
        assert address, "Completed training but its Ray address is missing"
        ray.init(address=address, log_to_driver=False, include_dashboard=False)


def run(args: argparse.Namespace) -> None:
    project = Path(__file__).resolve().parents[2]
    exp_dir = args.exp_dir.resolve()
    exp_dir.mkdir(parents=True, exist_ok=True)
    log_path = exp_dir / "run.log"
    assert not log_path.exists(), f"Refusing to overwrite previous evidence: {log_path}"
    env = dict(os.environ, RAY_DEDUP_LOGS="0", PYTHONUNBUFFERED="1")
    command = args.command
    (exp_dir / "command.json").write_text(json.dumps(command, indent=2) + "\n")
    print(f"[chaos] Launching real GRPO: {command}", flush=True)
    processes: dict[tuple[int, float], psutil.Process] = {}
    kill = None
    replacement = None
    original_engines: set[Engine] = set()
    log_offset = 0
    deadline = time.monotonic() + args.startup_timeout
    http = requests.Session()
    http.trust_env = False
    with log_path.open("w") as output:
        training = subprocess.Popen(
            command, cwd=project, env=env, stdout=output, stderr=subprocess.STDOUT
        )
        capture_children(processes, training.pid)
        try:
            while training.poll() is None:
                capture_children(processes, training.pid)
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        "GRPO startup/fault/recovery exceeded its bounded deadline"
                    )
                log = log_path.read_text(errors="replace")
                steps = completed_steps(log)
                if not steps:
                    time.sleep(0.2)
                    continue
                if not needs_actor_observation(args.expect, kill, replacement):
                    # The driver owns the GCS and may shut it down before it
                    # exits. After the last receipt, observe only its log/PID.
                    time.sleep(0.1)
                    continue
                if not ray.is_initialized():
                    connect_ray()
                    deadline = time.monotonic() + args.fault_timeout
                engines = live_engines(log)
                if kill is None and len(engines) == 2:
                    original_engines = set(engines)
                    for engine in engines:
                        try:
                            version, running = serving_state(http, engine)
                        except requests.RequestException:
                            continue
                        # One /health_generate sentinel is not training traffic.
                        if running < 2 or version < 1:
                            continue
                        assert steps[-1] <= args.max_steps - 3, "Fault landed too late"
                        actor = psutil.Process(engine.actor_pid)
                        assert actor.create_time() == engine.actor_created, (
                            "Actor PID was reused"
                        )
                        capture_children(processes, engine.actor_pid)
                        log_offset = len(log)
                        kill = KillReceipt(
                            engine, version, running, steps[-1], time.time()
                        )
                        actor.kill()
                        (exp_dir / "kill.json").write_text(
                            json.dumps(asdict(kill), indent=2) + "\n"
                        )
                        print(
                            f"[chaos] Killed observed-busy training actor: {kill}",
                            flush=True,
                        )
                        deadline = time.monotonic() + args.completion_timeout
                        break
                elif (
                    kill is not None
                    and args.expect == "survival"
                    and replacement is None
                ):
                    for engine in engines:
                        if engine in original_engines:
                            continue
                        try:
                            version, running = serving_state(http, engine)
                        except requests.RequestException:
                            continue
                        if version <= kill.weight_version or running < 2:
                            continue
                        replacement = ReplacementReceipt(
                            engine, version, running, steps[-1], time.time()
                        )
                        (exp_dir / "replacement.json").write_text(
                            json.dumps(asdict(replacement), indent=2) + "\n"
                        )
                        print(
                            f"[chaos] Replacement received newer trainer weights and serves: {replacement}",
                            flush=True,
                        )
                        break
                if not needs_actor_observation(args.expect, kill, replacement):
                    # Disconnect while the driver is still serving, before its
                    # teardown can terminate an attached observer's core worker.
                    ray.shutdown()
                time.sleep(0.1)
            returncode = training.wait()
            ray.shutdown()
            log = log_path.read_text(errors="replace")
            assert kill is not None, "Training exited before fault injection"
            validate_outcome(
                expect=args.expect,
                returncode=returncode,
                log_after_kill=log[log_offset:],
                completed=completed_steps(log),
                max_steps=args.max_steps,
                kill=kill,
                replacement=replacement,
            )
            summary: dict[str, Any] = {
                "expect": args.expect,
                "returncode": returncode,
                "completed_steps": completed_steps(log),
                "kill": asdict(kill),
                "replacement": asdict(replacement) if replacement else None,
            }
            (exp_dir / "result.json").write_text(json.dumps(summary, indent=2) + "\n")
            print(f"[chaos] PASS: real GRPO {args.expect}", flush=True)
        except BaseException:
            print("[chaos] FAIL; final training log lines:", flush=True)
            print(
                "\n".join(log_path.read_text(errors="replace").splitlines()[-100:]),
                flush=True,
            )
            raise
        finally:
            ray.shutdown()
            http.close()
            cleanup(processes)
            if training.poll() is None:
                training.wait(timeout=10)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--expect", choices=["survival", "bounded_failure"], required=True
    )
    parser.add_argument("--exp-dir", type=Path, required=True)
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--startup-timeout", type=int, default=1200)
    parser.add_argument("--fault-timeout", type=int, default=300)
    parser.add_argument("--completion-timeout", type=int, default=1200)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    assert args.max_steps >= 4
    assert args.command, "Missing the training command"

    # Keep native Ray/GCS calls bounded too; this is a harness timeout, never an
    # accepted bounded-failure result from the training process.
    def timeout_handler(signum: int, frame: Any) -> None:
        raise TimeoutError("Whole GRPO chaos harness exceeded its deadline")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(
        args.startup_timeout + args.fault_timeout + args.completion_timeout + 180
    )
    try:
        run(args)
    finally:
        signal.alarm(0)


if __name__ == "__main__":
    main()
