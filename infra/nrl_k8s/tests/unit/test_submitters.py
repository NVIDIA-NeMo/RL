# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Tests for :mod:`nrl_k8s.submitters`.

Covers:

* Handle cache round-trip.
* ``build_submitter`` dispatches on :class:`SubmitterMode`.
* :class:`ExecSubmitter` generates a launcher that ``nohup``s + ``disown``s
  and writes a pidfile + exitcode sentinel.
* Env-var values survive shell quoting (spaces, dollars, single quotes).
* ``kubectl cp`` is invoked for the launcher (simpler contract than a
  length-threshold branch).
* Head-pod-not-found surfaces a cluster-name/namespace-bearing error.
* ``follow`` uses ``tail -F`` (not ``-f``), preserving through log rotation.
"""

from __future__ import annotations

import types
from pathlib import Path

import pytest
from nrl_k8s.submitters import (
    SubmissionHandle,
    build_submitter,
    load_handle,
    save_handle,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture(autouse=True)
def _tmp_cache_dir(tmp_path, monkeypatch):
    """Redirect ~/.cache/nrl-k8s/runs for each test.

    ``_cache_root()`` re-reads NRL_K8S_CACHE_DIR on every call, so setting
    the env var is enough — no reload needed.
    """
    monkeypatch.setenv("NRL_K8S_CACHE_DIR", str(tmp_path))
    yield


@pytest.fixture
def fake_head_pod(monkeypatch):
    """Make ``k8s.get_head_pod`` return a stub pod without contacting the API."""
    pod = types.SimpleNamespace(
        metadata=types.SimpleNamespace(name="ray-head-abc12"),
        status=types.SimpleNamespace(phase="Running"),
    )
    monkeypatch.setattr("nrl_k8s.k8s.get_head_pod", lambda *a, **kw: pod)
    return pod


# =============================================================================
# SubmissionHandle + cache
# =============================================================================


class TestSubmissionHandle:
    def test_roundtrip_ray(self, tmp_path):
        h = SubmissionHandle(
            kind="ray",
            run_id="raysubmit_abc",
            cluster_name="rc-train",
            namespace="ns",
        )
        p = save_handle(h)
        assert p.exists()
        loaded = load_handle("raysubmit_abc")
        assert loaded == h

    def test_roundtrip_exec(self, tmp_path):
        h = SubmissionHandle(
            kind="exec",
            run_id="train-1699999999",
            cluster_name="rc-train",
            namespace="ns",
            pod="ray-head-xyz",
            tmp_dir="/tmp/nrl-train-1699999999",
        )
        save_handle(h)
        loaded = load_handle("train-1699999999")
        assert loaded == h

    def test_missing_handle_returns_none(self, tmp_path):
        assert load_handle("never-submitted") is None


# =============================================================================
# Factory
# =============================================================================


class TestBuildSubmitter:
    def test_port_forward_default(self):
        from nrl_k8s.schema import InfraConfig

        infra = InfraConfig.model_validate({"namespace": "ns", "image": "img:tag"})
        sub = build_submitter(infra)
        from nrl_k8s.submitters.portforward import PortForwardSubmitter

        assert isinstance(sub, PortForwardSubmitter)

    def test_exec_when_selected(self):
        from nrl_k8s.schema import InfraConfig

        infra = InfraConfig.model_validate(
            {
                "namespace": "ns",
                "image": "img:tag",
                "submit": {"submitter": "exec", "execTmpDir": "/scratch"},
            }
        )
        sub = build_submitter(infra)
        from nrl_k8s.submitters.exec_ import ExecSubmitter

        assert isinstance(sub, ExecSubmitter)
        # /scratch, not /tmp — execTmpDir plumbed through.
        assert sub._tmp_root == "/scratch"


# =============================================================================
# ExecSubmitter — launcher script composition
# =============================================================================


def _capture_runs(monkeypatch):
    """Replace the helpers that shell out so we can inspect the cmdlines.

    - ``_run`` covers mkdir + kubectl cp + pidfile-cat probes.
    - ``subprocess.Popen`` covers the detached launch exec.

    Fake ``kubectl cp`` snarfs the launcher content from the local
    tempfile so tests can assert on its body. Fake ``cat <pidfile>``
    returns a believable PID so the poll loop in submit() completes
    on the first try.
    """
    calls: list[list[str]] = []
    captured: dict[str, str] = {}
    popen_calls: list[list[str]] = []

    def fake_run(cmd, *, capture=False, capture_stderr=False):
        calls.append(list(cmd))
        if len(cmd) >= 4 and cmd[0] == "kubectl" and cmd[1] == "cp":
            local = Path(cmd[2])
            if local.exists():
                captured["launcher"] = local.read_text()
                captured["dest"] = cmd[3]
        # pidfile-cat probe -> return a PID so the poll loop succeeds.
        if (
            len(cmd) >= 7
            and cmd[0] == "kubectl"
            and cmd[1] == "exec"
            and cmd[-2] == "cat"
        ):
            return "12345\n"
        return ""

    class FakePopen:
        def __init__(self, cmd, **kwargs):
            popen_calls.append(list(cmd))
            # Return code doesn't matter — submit() only reads
            # ``poll()`` to decide whether to terminate. Mark the
            # subprocess as having exited cleanly so neither terminate()
            # nor kill() need to run.
            self._rc = 0

        def poll(self):
            return self._rc

        def terminate(self):
            pass

        def wait(self, timeout=None):
            return self._rc

        def kill(self):
            pass

        def communicate(self, timeout=None):
            return ("", "")

    monkeypatch.setattr("nrl_k8s.submitters.exec_._run", fake_run)
    monkeypatch.setattr("nrl_k8s.submitters.exec_.subprocess.Popen", FakePopen)
    captured["popen_calls"] = popen_calls
    return calls, captured


class TestExecSubmitterLauncher:
    def test_submit_invokes_mkdir_cp_exec(self, fake_head_pod, monkeypatch):
        calls, captured = _capture_runs(monkeypatch)
        from nrl_k8s.submitters.exec_ import ExecSubmitter

        ExecSubmitter(exec_tmp_dir="/tmp").submit(
            "rc-train",
            "ns",
            entrypoint="python run_grpo.py",
            run_id="rid-1",
            env_vars={"FOO": "bar"},
        )

        # _run calls: mkdir exec + kubectl cp + (one or more) cat pidfile.
        assert any(c[:2] == ["kubectl", "exec"] and "mkdir" in c for c in calls)
        assert any(c[:2] == ["kubectl", "cp"] for c in calls)
        # The detached launch goes through Popen, not _run.
        assert any("nohup" in " ".join(c) for c in captured["popen_calls"])

    def test_launcher_contains_nohup_disown_and_pid(self, fake_head_pod, monkeypatch):
        calls, captured = _capture_runs(monkeypatch)
        from nrl_k8s.submitters.exec_ import ExecSubmitter

        ExecSubmitter(exec_tmp_dir="/tmp").submit(
            "rc-train",
            "ns",
            entrypoint="python run_grpo.py",
            run_id="rid-2",
        )

        # The detached launch lives under Popen.
        assert captured["popen_calls"], "no Popen invocation captured"
        bg_cmd = captured["popen_calls"][0]
        assert bg_cmd[:4] == ["kubectl", "exec", "-n", "ns"]
        assert bg_cmd[5:7] == ["--", "bash"]
        joined = " ".join(bg_cmd)
        assert "nohup" in joined
        assert "disown" in joined
        assert "</dev/null" in joined
        assert "stdout.log" in joined
        assert "/tmp/nrl-rid-2/pid" in joined

    def test_launcher_exports_env_vars(self, fake_head_pod, monkeypatch):
        _, captured = _capture_runs(monkeypatch)
        from nrl_k8s.submitters.exec_ import ExecSubmitter

        ExecSubmitter(exec_tmp_dir="/tmp").submit(
            "rc-train",
            "ns",
            entrypoint="echo hi",
            run_id="rid-3",
            env_vars={
                "SIMPLE": "value",
                "HAS_SPACES": "foo bar",
                "HAS_DOLLAR": "$PATH",
                "HAS_QUOTES": "it's fine",
            },
        )

        launcher = captured["launcher"]
        assert "export SIMPLE=value" in launcher
        # shlex.quote wraps things with spaces / special chars.
        assert "'foo bar'" in launcher
        assert "'$PATH'" in launcher
        assert "'it'\"'\"'s fine'" in launcher  # standard POSIX escape

    def test_launcher_writes_exitcode(self, fake_head_pod, monkeypatch):
        _, captured = _capture_runs(monkeypatch)
        from nrl_k8s.submitters.exec_ import ExecSubmitter

        ExecSubmitter(exec_tmp_dir="/tmp").submit(
            "rc-train",
            "ns",
            entrypoint="python run_grpo.py",
            run_id="rid-4",
        )
        launcher = captured["launcher"]
        # shlex.quote doesn't wrap paths without shell metacharacters.
        assert "echo $ec > /tmp/nrl-rid-4/exitcode" in launcher
        assert launcher.rstrip().endswith("exit $ec")

    def test_working_dir_rejected_on_exec(self, fake_head_pod, monkeypatch):
        _capture_runs(monkeypatch)
        from nrl_k8s.submitters.exec_ import ExecSubmitter

        with pytest.raises(ValueError, match="working_dir upload"):
            ExecSubmitter().submit(
                "rc-train",
                "ns",
                entrypoint="echo",
                run_id="rid-5",
                working_dir=Path("/some/dir"),
            )

    def test_invalid_run_id_rejected(self, fake_head_pod, monkeypatch):
        _capture_runs(monkeypatch)
        from nrl_k8s.submitters.exec_ import ExecSubmitter

        with pytest.raises(ValueError, match="run_id"):
            ExecSubmitter().submit(
                "rc-train",
                "ns",
                entrypoint="echo",
                run_id="has spaces",
            )

    def test_returned_handle_populated(self, fake_head_pod, monkeypatch):
        _capture_runs(monkeypatch)
        from nrl_k8s.submitters.exec_ import ExecSubmitter

        handle = ExecSubmitter(exec_tmp_dir="/scratch").submit(
            "rc-train",
            "ns",
            entrypoint="echo",
            run_id="rid-7",
        )
        assert handle.kind == "exec"
        assert handle.run_id == "rid-7"
        assert handle.cluster_name == "rc-train"
        assert handle.namespace == "ns"
        assert handle.pod == "ray-head-abc12"
        assert handle.tmp_dir == "/scratch/nrl-rid-7"


# =============================================================================
# ExecSubmitter.follow / status / stop
# =============================================================================


class TestExecSubmitterObservability:
    def test_follow_uses_tail_F(self, monkeypatch):
        """Tail -F (capital) survives log rotation; tail -f does not."""
        from nrl_k8s.submitters.exec_ import ExecSubmitter

        captured_cmd: list[list[str]] = []

        import io

        class FakePopen:
            def __init__(self, cmd, **kwargs):
                captured_cmd.append(list(cmd))
                # Empty stream — `readline()` returns "" immediately, which
                # terminates the iter() in follow().
                self.stdout = io.StringIO("")

            def terminate(self):
                pass

            def wait(self, timeout=None):
                return 0

            def kill(self):
                pass

        monkeypatch.setattr("nrl_k8s.submitters.exec_.subprocess.Popen", FakePopen)

        handle = SubmissionHandle(
            kind="exec",
            run_id="r",
            cluster_name="rc",
            namespace="ns",
            pod="head-1",
            tmp_dir="/tmp/nrl-r",
        )
        list(ExecSubmitter().follow(handle))
        cmd = captured_cmd[0]
        assert "tail" in cmd
        assert "-F" in cmd
        assert "-f" not in cmd
        assert "/tmp/nrl-r/stdout.log" in cmd

    def test_status_dispatches_on_probe_output(self, monkeypatch):
        from nrl_k8s.submitters.exec_ import ExecSubmitter

        for probe_out, expected in [
            ("running", "running"),
            ("succeeded", "succeeded"),
            ("failed", "failed"),
            ("stopped", "stopped"),
            ("garbage", "unknown"),
        ]:
            monkeypatch.setattr(
                "nrl_k8s.submitters.exec_._run",
                lambda cmd, *, capture=False, out=probe_out: out + "\n",
            )
            handle = SubmissionHandle(
                kind="exec",
                run_id="r",
                cluster_name="rc",
                namespace="ns",
                pod="head-1",
                tmp_dir="/tmp/nrl-r",
            )
            assert ExecSubmitter().status(handle) == expected

    def test_stop_signals_term_then_kill(self, monkeypatch):
        from nrl_k8s.submitters.exec_ import ExecSubmitter

        calls: list[list[str]] = []
        monkeypatch.setattr(
            "nrl_k8s.submitters.exec_._run",
            lambda cmd, *, capture=False: calls.append(list(cmd)) or "",
        )
        handle = SubmissionHandle(
            kind="exec",
            run_id="r",
            cluster_name="rc",
            namespace="ns",
            pod="head-1",
            tmp_dir="/tmp/nrl-r",
        )
        ExecSubmitter().stop(handle)
        ExecSubmitter().stop(handle, force=True)
        # Both calls should be kubectl exec, and the second should pass KILL.
        assert "kill -s TERM" in " ".join(calls[0])
        assert "kill -s KILL" in " ".join(calls[1])


# =============================================================================
# Head-pod lookup failure
# =============================================================================


class TestHeadPodLookup:
    def test_no_running_head_pod_surfaces_cluster_and_ns(self, monkeypatch):
        from nrl_k8s import k8s

        def raise_(*a, **kw):
            raise RuntimeError(
                "no Running head pod found for RayCluster 'rc-train' in namespace 'ns'"
            )

        monkeypatch.setattr(k8s, "get_head_pod", raise_)

        from nrl_k8s.submitters.exec_ import ExecSubmitter

        with pytest.raises(RuntimeError) as excinfo:
            ExecSubmitter().submit(
                "rc-train",
                "ns",
                entrypoint="echo",
                run_id="r",
            )
        assert "rc-train" in str(excinfo.value)
        assert "ns" in str(excinfo.value)
