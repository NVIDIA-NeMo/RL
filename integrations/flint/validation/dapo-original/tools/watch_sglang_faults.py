#!/usr/bin/env python3
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
"""Terminate only the supplied driver when its log records a fatal CUDA access.

Linux /proc birth identity and command-line hash are checked before signalling.
A pidfd pins the original process, eliminating a PID-reuse race at the signal.
Only SIGTERM is sent; this tool never addresses process groups, nodes, or jobs.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import os
import platform
import re
import signal
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

READ_BYTES = 64 * 1024
MAX_LINE_BYTES = 1024 * 1024
ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
WORKER_PREFIX = re.compile(r"^(\([^()\n]*pid=\d+[^()\n]*\))\s*")
REPEATED = re.compile(r"\s*\[repeated \d+x across cluster\]\s*$")
FATAL = re.compile(
    r"^(torch\.AcceleratorError|RuntimeError): CUDA error: "
    r"an illegal memory access was encountered\s*$"
)
SGLANG_CONTEXT = re.compile(r"^\(SGLangGenerationWorker pid=\d+(?:, ip=[^()]*)?\)$")
SCHEDULER_FATAL = re.compile(
    r"^(?:\[\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}(?:[.,]\d+)?\]\s*)?"
    r"Scheduler hit an exception: Traceback \(most recent call last\):$"
)


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def validate_pid(pid):
    if pid <= 1 or pid in (os.getpid(), os.getppid()):
        raise ValueError(
            "Driver PID must exceed one and cannot be the watcher or its parent"
        )


def _native_pidfd(operation, first, second=0):
    """Use libc when the image's Python build omits Linux pidfd bindings."""
    if operation not in {"pidfd_open", "pidfd_send_signal"}:
        raise ValueError("Unknown pidfd operation")
    if sys.platform != "linux":
        raise RuntimeError(
            "Linux pidfd support is required; refusing a PID-only signal fallback"
        )
    libc = ctypes.CDLL(None, use_errno=True)
    function = getattr(libc, operation, None)
    ctypes.set_errno(0)
    if function is not None:
        function.restype = ctypes.c_int
        if operation == "pidfd_open":
            function.argtypes = [ctypes.c_int, ctypes.c_uint]
            result = function(first, 0)
        else:
            function.argtypes = [
                ctypes.c_int,
                ctypes.c_int,
                ctypes.c_void_p,
                ctypes.c_uint,
            ]
            result = function(first, second, None, 0)
    else:
        # These numbers are shared by Linux's verified aarch64 and x86_64 ABIs.
        # Never guess an architecture or substitute kill(pid, signal).
        machine = platform.machine().lower()
        if machine not in {"aarch64", "arm64", "x86_64", "amd64"}:
            raise RuntimeError(f"No verified Linux pidfd syscall ABI for {machine}")
        syscall = getattr(libc, "syscall", None)
        if syscall is None:
            raise RuntimeError("libc has no pidfd symbols or syscall entry point")
        syscall.restype = ctypes.c_long
        syscall.argtypes = [ctypes.c_long]
        if operation == "pidfd_open":
            result = syscall(ctypes.c_long(434), ctypes.c_int(first), ctypes.c_uint(0))
        else:
            result = syscall(
                ctypes.c_long(424),
                ctypes.c_int(first),
                ctypes.c_int(second),
                ctypes.c_void_p(),
                ctypes.c_uint(0),
            )
    if result < 0:
        error = ctypes.get_errno()
        raise OSError(error, f"{operation}: {os.strerror(error)}")
    return int(result)


def open_pidfd(pid):
    function = getattr(os, "pidfd_open", None)
    return function(pid, 0) if callable(function) else _native_pidfd("pidfd_open", pid)


def send_pidfd_signal(descriptor, number):
    function = getattr(signal, "pidfd_send_signal", None)
    if callable(function):
        function(descriptor, number, None, 0)
    else:
        _native_pidfd("pidfd_send_signal", descriptor, number)


def _stat_identity(payload):
    # comm is enclosed in parentheses and may itself contain spaces or ')'.
    closing = payload.rfind(")")
    if closing < 0:
        raise ValueError("Malformed Linux process stat")
    fields = payload[closing + 2 :].split()
    if len(fields) < 20:
        raise ValueError("Incomplete Linux process stat")
    return fields[0], int(fields[19])  # state (field 3), starttime (field 22).


def read_identity(pid, proc_root=Path("/proc")):
    directory = proc_root / str(pid)
    try:
        state, born = _stat_identity((directory / "stat").read_text())
        if state in ("Z", "X", "x"):
            return None
        command = (directory / "cmdline").read_bytes()
        state_again, born_again = _stat_identity((directory / "stat").read_text())
    except (FileNotFoundError, ProcessLookupError):
        return None
    if state_again in ("Z", "X", "x") or born_again != born or not command:
        return None
    return {
        "pid": pid,
        "start_time_ticks": born,
        "cmdline_sha256": hashlib.sha256(command).hexdigest(),
    }


class ProcessGuard:
    def __init__(self, pid, identity, descriptor, proc_root=Path("/proc")):
        self.pid, self.identity, self.descriptor = pid, identity, descriptor
        self.proc_root = proc_root

    @classmethod
    def capture(cls, pid):
        validate_pid(pid)
        identity = read_identity(pid)
        if identity is None:
            return None
        try:
            descriptor = open_pidfd(pid)
        except ProcessLookupError:
            return None
        try:
            if read_identity(pid) != identity:
                os.close(descriptor)
                return None
            # Signal zero checks the send binding and permission without
            # delivering a signal. A watching receipt therefore proves both.
            send_pidfd_signal(descriptor, 0)
            return cls(pid, identity, descriptor)
        except BaseException:
            os.close(descriptor)
            raise

    def current_state(self):
        current = read_identity(self.pid, self.proc_root)
        if current is None:
            return "gone"
        return "same" if current == self.identity else "identity_changed"

    def terminate(self):
        state = self.current_state()
        if state != "same":
            return "refused_" + state
        try:
            send_pidfd_signal(self.descriptor, signal.SIGTERM)
        except ProcessLookupError:
            return "refused_gone"
        return "SIGTERM_sent"

    def close(self):
        os.close(self.descriptor)


class FatalDetector:
    def __init__(self):
        self.tracebacks = {}

    def inspect(self, raw_line, line_number):
        content = ANSI.sub("", raw_line.decode("utf-8", errors="replace")).strip()
        content = REPEATED.sub("", content)
        match = WORKER_PREFIX.match(content)
        context = match.group(1) if match else "driver"
        if match:
            content = content[match.end() :]
        if "Traceback (most recent call last):" in content:
            self.tracebacks[context] = line_number
        scheduler_fatal = bool(
            SGLANG_CONTEXT.fullmatch(context) and SCHEDULER_FATAL.fullmatch(content)
        )
        fatal = FATAL.fullmatch(content)
        if fatal is None and not scheduler_fatal:
            return None
        if fatal is not None and fatal.group(1) == "RuntimeError":
            first = self.tracebacks.get(context)
            if first is None or line_number - first > 4096:
                return None
        return {
            "exception_line": content,
            "worker_context": context,
            "fatal_kind": "sglang_scheduler_exception"
            if scheduler_fatal
            else "cuda_illegal_memory_access",
            "line_number": line_number,
            "raw_line_sha256": hashlib.sha256(raw_line).hexdigest(),
        }


class LogTail:
    """Read fixed-size chunks, preserving lines split across read boundaries."""

    def __init__(self, path):
        self.path, self.stream = path, None
        self.pending, self.pending_offset = b"", 0
        self.line_number, self.discarding_long_line = 0, False

    def read(self):
        if self.stream is None:
            try:
                self.stream = self.path.open("rb")
            except FileNotFoundError:
                return [], False
        chunk = self.stream.read(READ_BYTES)
        if not chunk:
            return [], False
        self.pending += chunk
        lines = []
        while b"\n" in self.pending:
            raw, self.pending = self.pending.split(b"\n", 1)
            self.line_number += 1
            offset = self.pending_offset
            self.pending_offset += len(raw) + 1
            if not self.discarding_long_line and len(raw) <= MAX_LINE_BYTES:
                lines.append((raw, self.line_number, offset))
            self.discarding_long_line = False
        if len(self.pending) > MAX_LINE_BYTES:
            self.pending_offset += len(self.pending)
            self.pending = b""
            self.discarding_long_line = True
        return lines, True

    def file_identity(self):
        if self.stream is None:
            return None
        stat = os.fstat(self.stream.fileno())
        return {"device": stat.st_dev, "inode": stat.st_ino}

    def close(self):
        if self.stream is not None:
            self.stream.close()


def watch(pid, log, output, *, interval=2.0, guard_factory=None, sleeper=None):
    validate_pid(pid)
    guard_factory = ProcessGuard.capture if guard_factory is None else guard_factory
    sleeper = time.sleep if sleeper is None else sleeper
    receipt = {
        "schema_version": 1,
        "status": "watching",
        "driver_pid": pid,
        "log_path": str(log),
        "started_at_utc": utc_now(),
        "signal_scope": "Only supplied driver process through its pinned Linux pidfd",
        "signal": "SIGTERM",
        "poll_interval_seconds": interval,
    }
    guard, tail, fatal_seen = None, LogTail(log), False
    try:
        guard = guard_factory(pid)
        if guard is None:
            receipt.update(status="driver_gone_at_start", signal_action="none")
            return 0
        receipt["driver_birth_identity"] = guard.identity
        atomic_json(output, receipt)
        detector = FatalDetector()
        while True:
            state = guard.current_state()
            if state != "same":
                receipt.update(
                    status="fatal_detected_driver_ended"
                    if fatal_seen
                    else "driver_ended",
                    terminal_driver_state=state,
                )
                return 0
            lines, read_data = tail.read()
            if not fatal_seen:
                for raw, line_number, offset in lines:
                    match = detector.inspect(raw, line_number)
                    if match is None:
                        continue
                    fatal_seen = True
                    receipt.update(
                        status="fatal_detected",
                        detected_at_utc=utc_now(),
                        detected_monotonic_ns=time.monotonic_ns(),
                        source={
                            **match,
                            "byte_offset": offset,
                            "file_identity": tail.file_identity(),
                        },
                        signal_action="pending_identity_check",
                    )
                    atomic_json(output, receipt)
                    receipt["signal_action"] = guard.terminate()
                    receipt["signal_checked_at_utc"] = utc_now()
                    atomic_json(output, receipt)
                    break
            if fatal_seen or not read_data:
                sleeper(interval)
    except Exception as error:
        receipt.update(
            status="watcher_failed",
            error={"type": type(error).__name__, "message": str(error)},
        )
        return 1
    finally:
        if guard is not None:
            guard.close()
        tail.close()
        receipt["finished_at_utc"] = utc_now()
        atomic_json(output, receipt)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pid", type=int, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.umask(0o077)
    return watch(args.pid, args.log, args.output)


if __name__ == "__main__":
    sys.exit(main())
