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
"""``kubectl exec`` submitter — runs the training entrypoint as a raw
backgrounded process on the training head pod.

Same shape as a Slurm driver on a login node. The user's entrypoint
(``python -u run_grpo.py …``) is wrapped into a launcher script that
``source``s env vars the user declared, runs the command under
``nohup`` with redirected stdio, and captures ``$!`` as a pidfile +
``$?`` as an exitcode file. ``nrl-k8s job logs / stop`` then talk to
the same directory via additional ``kubectl exec`` calls.

Why not ``ray job submit``? Two reasons:

1. The training driver calls ``ray.init(address="auto")`` anyway; there
   is no meaningful difference between "Ray Job" and "regular driver" —
   the dashboard only adds a status row and a log buffer that duplicates
   the file we're already writing.
2. Ray's ``runtime_env.env_vars`` doesn't merge with ``ray.init``'s
   captured env — setting both is a recurring source of "Failed to
   merge" crashes in prior versions of these examples. Using shell
   ``export`` avoids the class entirely.
"""

from __future__ import annotations

import re
import shlex
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Iterator

from .. import k8s
from . import JobStatusStr, SubmissionHandle


_VALID_ID = re.compile(r"^[a-zA-Z0-9_.-]+$")
# Upper bound for how long we wait for the pidfile to appear on the pod
# after the launch exec fires. In normal conditions the pidfile lands in
# well under a second; a flaky control plane can push it to a few
# seconds. 30 s is generous headroom without blocking the caller for
# the full websocket teardown (which can be 60+ seconds on EKS).
_PIDFILE_POLL_SECONDS = 30


class ExecSubmitter:
    """Back-grounded-process transport."""

    def __init__(self, *, exec_tmp_dir: str = "/tmp") -> None:
        self._tmp_root = exec_tmp_dir.rstrip("/") or "/tmp"

    # ---------- submit --------------------------------------------------

    def submit(
        self,
        cluster_name: str,
        namespace: str,
        *,
        entrypoint: str,
        run_id: str,
        env_vars: dict[str, str] | None = None,
        working_dir: Path | None = None,
    ) -> SubmissionHandle:
        if working_dir is not None:
            # Exec submitter never uploads. Callers who want uploads should
            # pick PortForwardSubmitter.
            raise ValueError(
                "ExecSubmitter does not support working_dir upload; set "
                "infra.launch.codeSource to image or lustre (or switch "
                "submitter to portForward)."
            )

        _validate_run_id(run_id)
        pod = k8s.get_head_pod(cluster_name, namespace)
        pod_name = pod.metadata.name

        tmp_dir = f"{self._tmp_root}/nrl-{run_id}"
        entry_path = f"{tmp_dir}/entry.sh"
        log_path = f"{tmp_dir}/stdout.log"
        pid_path = f"{tmp_dir}/pid"
        exitcode_path = f"{tmp_dir}/exitcode"

        launcher = _render_launcher(
            run_id=run_id,
            env_vars=env_vars or {},
            user_entrypoint=entrypoint,
            exitcode_path=exitcode_path,
        )

        # mkdir + cp happen as two separate exec calls because `kubectl cp`
        # uses `tar` under the hood which fails if the destination
        # directory doesn't exist on the pod side.
        _run(["kubectl", "exec", "-n", namespace, pod_name, "--",
              "mkdir", "-p", tmp_dir])

        with tempfile.NamedTemporaryFile("w", suffix=".sh", delete=False) as f:
            f.write(launcher)
            local_entry = Path(f.name)
        try:
            _run([
                "kubectl", "cp",
                str(local_entry),
                f"{namespace}/{pod_name}:{entry_path}",
            ])
        finally:
            local_entry.unlink(missing_ok=True)

        # The launcher is backgrounded with ``nohup`` + ``disown`` +
        # redirected stdio + ``</dev/null`` so it fully detaches from
        # the ``kubectl exec`` websocket. The exec call itself should
        # return in milliseconds, but EKS / flaky control planes
        # sometimes hold the websocket open for tens of seconds during
        # teardown, even after the foreground shell has exited.
        #
        # We don't wait for it. Once the bg process is started, the
        # ``pid`` file appears on the pod near-instantly. We poll for
        # it and return as soon as it's there, killing the (possibly
        # still-open) exec subprocess without waiting.
        bg = (
            f"chmod +x {shlex.quote(entry_path)} && "
            f"nohup bash {shlex.quote(entry_path)} "
            f"  > {shlex.quote(log_path)} 2>&1 </dev/null & "
            f"echo $! > {shlex.quote(pid_path)}; "
            f"disown; "
            f"exit 0"
        )
        exec_cmd = ["kubectl", "exec", "-n", namespace, pod_name, "--",
                    "bash", "-c", bg]
        exec_proc = subprocess.Popen(
            exec_cmd,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # Poll the pidfile directly — independent of the exec
        # subprocess's lifecycle. A fresh ``kubectl exec cat`` call per
        # poll is cheap (< 1 s) and has its own short-lived websocket,
        # so the flaky long-lived one from the bg launch doesn't block
        # anything.
        deadline = time.monotonic() + _PIDFILE_POLL_SECONDS
        pid = ""
        while time.monotonic() < deadline:
            try:
                pid = _run(
                    ["kubectl", "exec", "-n", namespace, pod_name, "--",
                     "cat", pid_path],
                    capture=True, capture_stderr=True,
                ).strip()
            except _ExecFailed:
                time.sleep(0.5)
                continue
            if pid.isdigit():
                break
            time.sleep(0.5)

        # Either we have a pid (success) or we exhausted the budget
        # (fail). Either way the bg-launch exec subprocess is no longer
        # useful to us; kill it so we return promptly.
        if exec_proc.poll() is None:
            exec_proc.terminate()
            try:
                exec_proc.wait(timeout=1)
            except subprocess.TimeoutExpired:
                exec_proc.kill()

        if not pid.isdigit():
            stderr = ""
            try:
                _, stderr = exec_proc.communicate(timeout=1)
            except subprocess.TimeoutExpired:
                pass
            raise RuntimeError(
                f"exec submit: pidfile {pid_path} did not appear on "
                f"{pod_name} within {_PIDFILE_POLL_SECONDS}s. Stderr "
                f"from launch exec was:\n{stderr or '<none>'}"
            )

        return SubmissionHandle(
            kind="exec",
            run_id=run_id,
            cluster_name=cluster_name,
            namespace=namespace,
            pod=pod_name,
            tmp_dir=tmp_dir,
        )

    # ---------- follow / status / stop ---------------------------------

    def follow(self, handle: SubmissionHandle) -> Iterator[str]:
        tmp = _require_tmp(handle)
        cmd = [
            "kubectl", "exec", "-n", handle.namespace, _require_pod(handle), "--",
            "tail", "-F", "-n", "500", f"{tmp}/stdout.log",
        ]
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert proc.stdout is not None
        try:
            for line in iter(proc.stdout.readline, ""):
                yield line
        finally:
            proc.terminate()
            try:
                proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                proc.kill()

    def status(self, handle: SubmissionHandle) -> JobStatusStr:
        tmp = _require_tmp(handle)
        pod = _require_pod(handle)
        # `kill -0 $pid` returns 0 if the process exists. The exitcode file
        # only appears after the launcher finishes; so:
        #   - kill -0 ok     → running (exitcode file may or may not exist)
        #   - kill -0 fail + exitcode=0  → succeeded
        #   - kill -0 fail + exitcode!=0 → failed
        #   - kill -0 fail + no exitcode → stopped (killed externally / pod restarted)
        probe = (
            f'if [ -f {shlex.quote(tmp)}/pid ] && kill -0 "$(cat {shlex.quote(tmp)}/pid)" 2>/dev/null; then '
            f'  echo running; '
            f'elif [ -f {shlex.quote(tmp)}/exitcode ]; then '
            f'  ec=$(cat {shlex.quote(tmp)}/exitcode); '
            f'  if [ "$ec" = "0" ]; then echo succeeded; else echo failed; fi; '
            f'else '
            f'  echo stopped; '
            f'fi'
        )
        try:
            out = _run(
                ["kubectl", "exec", "-n", handle.namespace, pod, "--", "bash", "-c", probe],
                capture=True,
            ).strip()
        except subprocess.CalledProcessError:
            return "unknown"
        return out if out in ("running", "succeeded", "failed", "stopped") else "unknown"

    def stop(self, handle: SubmissionHandle, *, force: bool = False) -> None:
        tmp = _require_tmp(handle)
        pod = _require_pod(handle)
        sig = "KILL" if force else "TERM"
        kill = (
            f'if [ -f {shlex.quote(tmp)}/pid ]; then '
            f'  kill -s {sig} "$(cat {shlex.quote(tmp)}/pid)" 2>/dev/null || true; '
            f'fi'
        )
        _run(["kubectl", "exec", "-n", handle.namespace, pod, "--",
              "bash", "-c", kill])


# =============================================================================
# Internals
# =============================================================================

def _validate_run_id(run_id: str) -> None:
    if not run_id:
        raise ValueError("run_id is required for exec submissions")
    if not _VALID_ID.match(run_id):
        raise ValueError(
            f"invalid run_id {run_id!r}: must match [a-zA-Z0-9_.-]+ "
            "(used in a pod path)"
        )


def _render_launcher(
    *,
    run_id: str,
    env_vars: dict[str, str],
    user_entrypoint: str,
    exitcode_path: str,
) -> str:
    """Produce the bash script we kubectl-cp onto the head pod.

    The script ``export``s every declared env var, runs the user's
    entrypoint, and writes the trailing exit code to a sentinel file so
    ``status()`` can tell succeeded from killed-mid-run.
    """
    lines = [
        "#!/bin/bash",
        "# Generated by nrl-k8s ExecSubmitter — do not edit by hand.",
        # `set -e` would make a nonzero exit in the user's entrypoint skip
        # our exitcode writer; we don't want that. Just propagate the
        # exit code explicitly.
        "set -u",
        "set -o pipefail",
        f"export NRL_K8S_RUN_ID={shlex.quote(run_id)}",
    ]
    for k, v in env_vars.items():
        _validate_env_key(k)
        lines.append(f"export {k}={shlex.quote(v)}")
    lines.append("")
    lines.append("# ---- user entrypoint (verbatim) ----")
    lines.append(user_entrypoint.rstrip())
    lines.append("ec=$?")
    lines.append(f"echo $ec > {shlex.quote(exitcode_path)}")
    lines.append("exit $ec")
    return "\n".join(lines) + "\n"


_ENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def _validate_env_key(k: str) -> None:
    if not _ENV_KEY_RE.match(k):
        raise ValueError(f"invalid env var name {k!r}")


def _require_pod(h: SubmissionHandle) -> str:
    if not h.pod:
        raise ValueError(f"exec handle for {h.run_id} is missing pod")
    return h.pod


def _require_tmp(h: SubmissionHandle) -> str:
    if not h.tmp_dir:
        raise ValueError(f"exec handle for {h.run_id} is missing tmp_dir")
    return h.tmp_dir


class _ExecFailed(Exception):
    """Wraps a failed kubectl exec/cp with its stderr payload.

    Callers that want to reconcile the error against side-effects on the
    pod (e.g. a pidfile written before the websocket dropped) can catch
    this explicitly and re-raise only if the side-effect is missing.
    """

    def __init__(self, returncode: int, stderr: str) -> None:
        super().__init__(f"kubectl exited {returncode}: {stderr}")
        self.returncode = returncode
        self.stderr = stderr


def _run(cmd: list[str], *, capture: bool = False, capture_stderr: bool = False) -> str:
    """Thin ``subprocess.run`` wrapper.

    - ``capture=True`` returns stdout as str; stderr is inherited.
    - ``capture_stderr=True`` captures stderr so transient "connection
      reset" lines from kubectl don't pollute the CLI when the caller
      has a stronger success signal (e.g. a pidfile on the pod). On a
      non-zero exit we raise :class:`_ExecFailed` with the captured
      stderr body.
    - No stdin: ``</dev/null`` on kubectl exec so it doesn't keep the
      websocket open waiting for input.
    """
    stdout = subprocess.PIPE if capture else None
    stderr = subprocess.PIPE if capture_stderr else None
    res = subprocess.run(
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=stdout,
        stderr=stderr,
        text=True,
    )
    if res.returncode != 0:
        if capture_stderr:
            raise _ExecFailed(res.returncode, res.stderr or "")
        raise subprocess.CalledProcessError(res.returncode, cmd)
    return (res.stdout or "") if capture else ""


__all__ = ["ExecSubmitter"]
