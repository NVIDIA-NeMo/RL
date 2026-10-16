# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

"""Gym servers must not outlive NemoGym.shutdown().

With ``nemo_gym_log_dir`` set, Gym runs each server behind a ``( ... ) | tee``
bash wrapper, and ``RunHelper.shutdown()`` signals only the wrapper, so the
server survives it, and Gym waits out its grace period on each wrapper in turn.
These tests start real processes the way Gym does and check that
``_shut_down_gym_servers`` interrupts the servers before Gym's shutdown, each
exactly once, stops what is left afterwards, fits the shutdown budget, and
touches nothing else.
"""

import shlex
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

import psutil
import pytest

from nemo_rl.environments import nemo_gym as nemo_gym_mod
from nemo_rl.environments.utils import DEFAULT_ENV_SHUTDOWN_TIMEOUT_SECONDS

# A stand-in Gym server. It records its pid once its SIGINT handler is in place,
# logs each SIGINT it gets, and leaves a marker when SIGINT stops it, so a test
# can tell an interrupted server from a killed one. A "slow" server takes a
# moment to stop, long enough to see a second SIGINT.
_STUB_SERVER = """
import os, signal, sys, time
name, mode, run_dir = sys.argv[1:4]
interrupted_at = []

def on_sigint(signum, frame):
    with open(os.path.join(run_dir, name + ".sigints"), "a") as f:
        f.write("SIGINT\\n")
    interrupted_at.append(time.monotonic())

signal.signal(signal.SIGINT, signal.SIG_IGN if mode == "ignore" else on_sigint)
print(name + " serving", flush=True)
with open(os.path.join(run_dir, name + ".pid.tmp"), "w") as f:
    f.write(str(os.getpid()))
os.replace(os.path.join(run_dir, name + ".pid.tmp"), os.path.join(run_dir, name + ".pid"))
stop_after_s = 0.3 if mode == "slow" else 0.0
while not interrupted_at or time.monotonic() - interrupted_at[0] < stop_after_s:
    time.sleep(0.02)
open(os.path.join(run_dir, name + ".stopped"), "w").close()
print(name + " stopped on SIGINT", flush=True)
"""


@pytest.fixture
def run_dir(tmp_path: Path):
    """A directory holding the stub server; kills any stub a failed test left behind."""
    (tmp_path / "stub_server.py").write_text(_STUB_SERVER)
    yield tmp_path
    for pid_file in tmp_path.glob("*.pid"):
        try:
            psutil.Process(int(pid_file.read_text())).kill()
        except psutil.NoSuchProcess:
            pass


def _server_command(run_dir: Path, name: str, mode: str) -> str:
    # The shape of Gym's server command: cd, setup, then the server as the last
    # simple command.
    return (
        f"cd {shlex.quote(str(run_dir))} && true && FOO=bar "
        f"{shlex.quote(sys.executable)} stub_server.py {name} {mode} "
        f"{shlex.quote(str(run_dir))}"
    )


def _wait_for_pid(pid_file: Path, timeout_s: float = 30.0) -> int:
    deadline = time.monotonic() + timeout_s
    while not pid_file.exists():
        assert time.monotonic() < deadline, f"{pid_file.name} never appeared"
        time.sleep(0.05)
    return int(pid_file.read_text())


def _alive(pid: int) -> bool:
    try:
        return psutil.Process(pid).status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def _gone(pid: int, timeout_s: float = 5.0) -> bool:
    """Whether the process has exited, allowing a killed one a moment to go."""
    deadline = time.monotonic() + timeout_s
    while _alive(pid):
        if time.monotonic() > deadline:
            return False
        time.sleep(0.05)
    return True


def _tee_running_for(log_dir: Path, timeout_s: float = 5.0) -> bool:
    """Whether a tee writing into log_dir is still running after timeout_s.

    A killed tee can take a moment to leave the process table, so this polls.
    """
    deadline = time.monotonic() + timeout_s
    while True:
        running = [
            process
            for process in psutil.process_iter(["name", "cmdline", "status"])
            if process.info["name"] == "tee"
            and process.info["status"] != psutil.STATUS_ZOMBIE
            and str(log_dir) in " ".join(process.info["cmdline"] or [])
        ]
        if not running or time.monotonic() > deadline:
            return bool(running)
        time.sleep(0.05)


class _GymLikeRunHelper:
    """Starts and stops servers the way Gym's RunHelper does.

    With a log dir each server runs behind Gym's ``( ... ) | tee`` wrapper.
    shutdown() signals only the processes it launched, waits up to ``grace_s``
    on each in turn before killing it, and clears ``_processes``.
    """

    grace_s = 0.5

    def __init__(self, run_dir: Path, log_dir: Optional[Path]) -> None:
        self._run_dir = run_dir
        self._log_dir = log_dir
        self._processes: dict[str, subprocess.Popen] = {}
        self._server_pids: dict[str, int] = {}
        self.servers_alive_after_shutdown: Optional[bool] = None

    def launch(self, name: str, mode: str = "normal") -> int:
        """Start a server and return the pid of the server process itself."""
        command = _server_command(self._run_dir, name, mode)
        if self._log_dir is not None:
            self._log_dir.mkdir(exist_ok=True)
            log_path = self._log_dir / f"{name}.log"
            command = f"set -o pipefail; ({command}) 2>&1 | tee -a {log_path}"
        self._processes[name] = subprocess.Popen(
            command, executable="/bin/bash", shell=True
        )
        self._server_pids[name] = _wait_for_pid(self._run_dir / f"{name}.pid")
        return self._server_pids[name]

    def shutdown(self) -> None:
        for process in self._processes.values():
            process.send_signal(signal.SIGINT)
        for process in self._processes.values():
            try:
                process.wait(timeout=self.grace_s)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=2)
        self._processes = {}
        self.servers_alive_after_shutdown = any(
            _alive(pid) for pid in self._server_pids.values()
        )


def test_stops_a_server_behind_the_tee_wrapper(run_dir):
    log_dir = run_dir / "logs"
    run_helper = _GymLikeRunHelper(run_dir, log_dir=log_dir)
    server_pid = run_helper.launch("server")
    # Gym's SIGINT goes to the wrapping shell, not to the server.
    assert run_helper._processes["server"].pid != server_pid

    nemo_gym_mod._shut_down_gym_servers(run_helper)

    assert _gone(server_pid)
    # Interrupted rather than killed, and tee outlived it to log the shutdown.
    assert (run_dir / "server.stopped").exists()
    assert "server stopped on SIGINT" in (log_dir / "server.log").read_text()
    assert not _tee_running_for(log_dir)


def test_without_a_log_dir_gym_stops_the_server_itself(run_dir):
    run_helper = _GymLikeRunHelper(run_dir, log_dir=None)
    server_pid = run_helper.launch("server")

    nemo_gym_mod._shut_down_gym_servers(run_helper)

    assert run_helper.servers_alive_after_shutdown is False
    assert _gone(server_pid)
    assert (run_dir / "server.stopped").exists()


@pytest.mark.parametrize("with_log_dir", [False, True])
def test_each_server_gets_a_single_sigint(run_dir, with_log_dir):
    """A second SIGINT makes uvicorn skip the application's shutdown."""
    log_dir = run_dir / "logs" if with_log_dir else None
    run_helper = _GymLikeRunHelper(run_dir, log_dir=log_dir)
    server_pid = run_helper.launch("server", mode="slow")

    nemo_gym_mod._shut_down_gym_servers(run_helper)

    assert _gone(server_pid)
    assert (run_dir / "server.sigints").read_text() == "SIGINT\n"
    assert (run_dir / "server.stopped").exists()


def test_kills_a_leftover_server_that_ignores_sigint(run_dir, monkeypatch):
    monkeypatch.setattr(nemo_gym_mod, "NEMO_GYM_LEFTOVER_SERVER_TIMEOUT_S", 0.5)
    run_helper = _GymLikeRunHelper(run_dir, log_dir=run_dir / "logs")
    server_pid = run_helper.launch("server", mode="ignore")

    nemo_gym_mod._shut_down_gym_servers(run_helper)

    assert _gone(server_pid)
    assert not (run_dir / "server.stopped").exists()
    assert not _tee_running_for(run_dir / "logs")


def test_stops_leftover_servers_when_gym_shutdown_raises(run_dir):
    run_helper = _GymLikeRunHelper(run_dir, log_dir=run_dir / "logs")
    server_pid = run_helper.launch("server")
    gym_shutdown = run_helper.shutdown

    def shutdown_then_fail() -> None:
        gym_shutdown()
        raise RuntimeError("telemetry flush failed")

    run_helper.shutdown = shutdown_then_fail

    with pytest.raises(RuntimeError, match="telemetry flush failed"):
        nemo_gym_mod._shut_down_gym_servers(run_helper)

    assert _gone(server_pid)
    assert (run_dir / "server.stopped").exists()


def test_interrupts_the_servers_when_gym_shutdown_fails_before_signalling(run_dir):
    """Gym flushes telemetry before it signals anything, so that can fail first.

    Without a log dir only Gym's own SIGINT would reach the server.
    """
    run_helper = _GymLikeRunHelper(run_dir, log_dir=None)
    server_pid = run_helper.launch("server")

    def fail_before_signalling() -> None:
        raise RuntimeError("telemetry flush failed")

    run_helper.shutdown = fail_before_signalling

    with pytest.raises(RuntimeError, match="telemetry flush failed"):
        nemo_gym_mod._shut_down_gym_servers(run_helper)

    assert _gone(server_pid)
    assert (run_dir / "server.stopped").exists()


def test_leaves_processes_gym_did_not_launch_alone(run_dir):
    """Only Gym's servers are stopped, never other children of this process.

    In a unit test that is, for example, the Ray cluster the session started.
    """
    bystander = subprocess.Popen(
        [sys.executable, "stub_server.py", "bystander", "normal", str(run_dir)],
        cwd=run_dir,
    )
    try:
        _wait_for_pid(run_dir / "bystander.pid")
        run_helper = _GymLikeRunHelper(run_dir, log_dir=run_dir / "logs")
        server_pid = run_helper.launch("server")

        nemo_gym_mod._shut_down_gym_servers(run_helper)

        assert _gone(server_pid)
        assert bystander.poll() is None
        assert not (run_dir / "bystander.stopped").exists()
    finally:
        bystander.kill()
        bystander.wait()


def test_skips_a_command_that_has_exited(run_dir):
    """Its pid may since have gone to an unrelated process, here a bystander.

    Gym's poll(), which NemoGym.health_check() runs, reaps a command that died.
    """
    bystander = subprocess.Popen(
        [sys.executable, "stub_server.py", "bystander", "normal", str(run_dir)],
        cwd=run_dir,
    )
    try:
        bystander_pid = _wait_for_pid(run_dir / "bystander.pid")
        run_helper = _GymLikeRunHelper(run_dir, log_dir=run_dir / "logs")
        server_pid = run_helper.launch("server")
        crashed = subprocess.Popen(["true"])
        crashed.wait()
        crashed.pid = bystander_pid  # Reaped, and its pid reused.
        run_helper._processes["crashed"] = crashed

        nemo_gym_mod._shut_down_gym_servers(run_helper)

        assert _gone(server_pid)
        assert bystander.poll() is None
        assert not (run_dir / "bystander.sigints").exists()
    finally:
        bystander.kill()
        bystander.wait()


def test_nemo_gym_shutdown_stops_many_wrapped_servers_within_the_env_budget(run_dir):
    """At the end of a run the Gym actor gets DEFAULT_ENV_SHUTDOWN_TIMEOUT_SECONDS.

    Gym waits up to a second on each command in turn, and a wrapper whose server
    never saw the signal uses all of it. More wrapped servers than the budget
    has seconds must still all stop, gracefully, before the caller gives up on
    the actor.
    """
    run_helper = _GymLikeRunHelper(run_dir, log_dir=run_dir / "logs")
    run_helper.grace_s = 1.0  # Gym's _GRACEFUL_SHUTDOWN_TIMEOUT_SEC
    num_servers = int(DEFAULT_ENV_SHUTDOWN_TIMEOUT_SECONDS) + 2
    server_pids = [run_helper.launch(f"server{i}") for i in range(num_servers)]
    actor = nemo_gym_mod.NemoGym.__ray_metadata__.modified_class({})
    actor.rh = run_helper

    start = time.monotonic()
    actor.shutdown()
    elapsed_s = time.monotonic() - start

    assert elapsed_s < DEFAULT_ENV_SHUTDOWN_TIMEOUT_SECONDS
    assert all(_gone(pid, timeout_s=0) for pid in server_pids)
    assert all((run_dir / f"server{i}.stopped").exists() for i in range(num_servers))


@pytest.mark.nemo_gym
def test_stops_servers_that_gym_run_command_launches(run_dir):
    """The same, against Gym's own launch and shutdown code.

    _shut_down_gym_servers reads RunHelper._processes, which Gym does not offer
    as API, so this also fails if a Gym bump renames it or changes what it
    holds. Beyond _processes, RunHelper.shutdown() needs only the head-server
    handles that start() would have created. There are more servers than the
    shutdown budget has seconds, as Gym's own shutdown waits on each in turn.
    """
    from nemo_gym.cli.env import RunHelper
    from nemo_gym.cli.setup_command import run_command
    from omegaconf import DictConfig

    log_dir = run_dir / "logs"
    global_config_dict = DictConfig(
        {"uv_cache_dir": str(run_dir / "uv_cache"), "nemo_gym_log_dir": str(log_dir)}
    )
    names = [f"server{i}" for i in range(int(DEFAULT_ENV_SHUTDOWN_TIMEOUT_SECONDS) + 2)]
    run_helper = RunHelper()
    run_helper._processes = {
        name: run_command(
            _server_command(run_dir, name, "normal"),
            run_dir,
            server_name=name,
            global_config_dict=global_config_dict,
        )
        for name in names
    }
    server_pids = [_wait_for_pid(run_dir / f"{name}.pid") for name in names]
    run_helper._head_server = SimpleNamespace(should_exit=False)
    run_helper._head_server_thread = threading.Thread(target=lambda: None)
    run_helper._head_server_thread.start()

    start = time.monotonic()
    nemo_gym_mod._shut_down_gym_servers(run_helper)
    elapsed_s = time.monotonic() - start

    assert elapsed_s < DEFAULT_ENV_SHUTDOWN_TIMEOUT_SECONDS
    for name, server_pid in zip(names, server_pids, strict=True):
        assert _gone(server_pid)
        assert (run_dir / f"{name}.stopped").exists()
        assert f"{name} stopped on SIGINT" in (log_dir / f"{name}.log").read_text()
