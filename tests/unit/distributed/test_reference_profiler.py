# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import tools.rl1610_profile as profiler
import os
import signal
import sys

import pytest
from tools.rl1610_profile import assign_processes, read_process


def test_repeated_output_is_rejected(tmp_path):
    output = profiler.prepare_output(str(tmp_path / "run"))
    (output / "steps.jsonl").write_text("previous run\n")
    with pytest.raises(ValueError, match="empty output directory"):
        profiler.prepare_output(str(output))
    assert (output / "steps.jsonl").read_text() == "previous run\n"


def test_sigterm_stops_driver_and_restores_handler(tmp_path):
    previous = signal.getsignal(signal.SIGTERM)
    with pytest.raises(SystemExit) as error:
        with profiler.running_driver(
            [sys.executable, "-c", "import time; time.sleep(60)"]
        ) as child:
            os.kill(os.getpid(), signal.SIGTERM)
    assert error.value.code == 143
    assert child.poll() is not None
    assert signal.getsignal(signal.SIGTERM) == previous


def test_nested_actor_roots_own_descendants_once():
    processes = {
        pid: {"ppid": parent}
        for pid, parent in [(1, 0), (2, 1), (3, 2), (4, 1), (5, 4), (6, 99)]
    }
    owners = assign_processes(processes, {1: "policy", 4: "reference"})
    assert owners == {
        1: "policy",
        2: "policy",
        3: "policy",
        4: "reference",
        5: "reference",
    }


def test_process_stat_handles_spaces_and_reports_missing_pss(tmp_path):
    process_dir = tmp_path / "123"
    process_dir.mkdir()
    fields = ["S"] + ["0"] * 49
    fields[1], fields[11], fields[12], fields[19] = "42", "150", "25", "1000"
    (process_dir / "stat").write_text("123 (worker with spaces) " + " ".join(fields))
    missing = read_process(process_dir)
    assert missing["cpu_ticks"] == 175 and missing["ppid"] == 42
    assert missing["start_ticks"] == 1000
    assert missing["pss_bytes"] is None
    (process_dir / "smaps_rollup").write_text("Rss: 2048 kB\nPss: 1024 kB\n")
    memory = read_process(process_dir)
    assert memory["pss_bytes"] == 1024 * 1024
    assert memory["rss_bytes"] == 2048 * 1024


def test_sampler_measures_cpu_differences_and_rejects_reused_actor_pid(
    tmp_path, monkeypatch
):
    process_dir = tmp_path / "proc" / "123"
    process_dir.mkdir(parents=True)
    fields = ["S"] + ["0"] * 49
    fields[11], fields[12], fields[19] = "100", "0", "100"
    (process_dir / "stat").write_text("123 (worker) " + " ".join(fields))
    (process_dir / "smaps_rollup").write_text("Rss: 2048 kB\nPss: 1024 kB\n")
    (tmp_path / "proc" / "stat").write_text("cpu 100 0 100 100 0 0 0 0\nbtime 1000\n")
    (tmp_path / "proc" / "meminfo").write_text(
        "MemTotal: 1000 kB\nMemAvailable: 400 kB\n"
    )
    monkeypatch.setattr(profiler, "Path", lambda path: tmp_path / path.lstrip("/"))
    clock = iter([0.0, 0.01, 1.0, 1.01, 2.0, 2.01])
    monkeypatch.setattr(profiler.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(profiler.os, "sysconf", lambda _: 100)
    sampler = profiler.NodeSampler()
    actors = {"policy": {"pid": 123, "start_time_ms": 1002000}}
    first = sampler.sample(actors)
    assert first["actors"][0]["cpu_cores"] is None
    fields[11] = "200"
    (process_dir / "stat").write_text("123 (worker) " + " ".join(fields))
    second = sampler.sample(actors)
    assert second["actors"][0]["cpu_cores"] == 1.0
    assert second["actors"][0]["pss_bytes"] == 1024 * 1024
    assert second["node_os_used_bytes"] == 600 * 1024
    fields[19] = "1000"
    (process_dir / "stat").write_text("123 (reused) " + " ".join(fields))
    reused = sampler.sample(actors)
    assert reused["actors"][0]["root_available"] is False
    assert reused["actors"][0]["pss_bytes"] is None
