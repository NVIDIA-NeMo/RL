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
"""CPU guard tests; Linux pidfd calls are emulated on non-Linux test hosts."""

import errno
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import watch_sglang_faults as watcher

FATAL = b"torch.AcceleratorError: CUDA error: an illegal memory access was encountered"


def write_process(
    root, pid, *, born=123456, command=b"python\x00driver.py\x00", state="S"
):
    directory = root / str(pid)
    directory.mkdir(exist_ok=True)
    fields = [state] + ["0"] * 18 + [str(born)] + ["0"] * 5
    (directory / "stat").write_text(
        f"{pid} (driver with ) parentheses) " + " ".join(fields)
    )
    (directory / "cmdline").write_bytes(command)


class DetectorTests(unittest.TestCase):
    def test_exact_torch_exception_with_ray_ansi(self):
        detector = watcher.FatalDetector()
        line = b"\x1b[36m(SGLangGenerationWorker pid=123, ip=10.0.0.1)\x1b[0m " + FATAL
        self.assertIsNotNone(detector.inspect(line, 8))

    def test_benign_mentions_and_bare_runtime_error_do_not_match(self):
        detector = watcher.FatalDetector()
        self.assertIsNone(detector.inspect(b"print('" + FATAL + b"')", 1))
        self.assertIsNone(detector.inspect(b"INFO example: " + FATAL, 2))
        self.assertIsNone(
            detector.inspect(
                FATAL.replace(b"torch.AcceleratorError", b"RuntimeError"), 3
            )
        )

    def test_runtime_error_requires_same_worker_traceback(self):
        detector = watcher.FatalDetector()
        runtime = FATAL.replace(b"torch.AcceleratorError", b"RuntimeError")
        detector.inspect(
            b"(SGLangGenerationWorker pid=10) Scheduler hit an exception: Traceback (most recent call last):",
            1,
        )
        self.assertIsNone(
            detector.inspect(b"(SGLangGenerationWorker pid=11) " + runtime, 2)
        )
        self.assertIsNotNone(
            detector.inspect(b"(SGLangGenerationWorker pid=10) " + runtime, 3)
        )

    def test_exact_sglang_scheduler_fatal_matches_without_exception_class(self):
        line = (
            b"(SGLangGenerationWorker pid=3010712, ip=10.109.18.221) "
            b"[2026-10-20 05:31:05] Scheduler hit an exception: Traceback (most recent call last):"
        )
        match = watcher.FatalDetector().inspect(line, 42)
        self.assertEqual(match["fatal_kind"], "sglang_scheduler_exception")
        self.assertEqual(
            match["worker_context"],
            "(SGLangGenerationWorker pid=3010712, ip=10.109.18.221)",
        )

    def test_scheduler_marker_from_other_worker_and_health_noise_do_not_match(self):
        detector = watcher.FatalDetector()
        marker = b"[2026-10-20 05:31:05] Scheduler hit an exception: Traceback (most recent call last):"
        lines = [
            marker,
            b"(AutomodelPolicyWorker pid=123) " + marker,
            b"(SGLangGenerationWorker pid=123) INFO example: " + marker,
            b"(SGLangGenerationWorker pid=123) Task exception was never retrieved",
            b"(SGLangGenerationWorker pid=123) [2026-10-20 05:31:05] Health check failed.",
            b"(SGLangGenerationWorker pid=123) TimeoutError",
        ]
        for index, line in enumerate(lines):
            self.assertIsNone(detector.inspect(line, index))


class ProcessTests(unittest.TestCase):
    def test_pid_safety(self):
        for pid in (-1, 0, 1, os.getpid(), os.getppid()):
            with self.assertRaises(ValueError):
                watcher.validate_pid(pid)

    def test_linux_stat_birth_cmdline_and_zombie(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_process(root, 43210)
            identity = watcher.read_identity(43210, root)
            self.assertEqual(identity["start_time_ticks"], 123456)
            write_process(root, 43210, state="Z")
            self.assertIsNone(watcher.read_identity(43210, root))
            self.assertIsNone(watcher.read_identity(99999, root))

    def test_reused_pid_and_changed_command_are_never_signalled(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_process(root, 43210)
            identity = watcher.read_identity(43210, root)
            guard = watcher.ProcessGuard(43210, identity, 999, root)
            with patch.object(signal, "pidfd_send_signal", create=True) as send:
                write_process(root, 43210, born=123457)
                self.assertEqual(guard.terminate(), "refused_identity_changed")
                write_process(root, 43210, command=b"python\x00unrelated.py\x00")
                self.assertEqual(guard.terminate(), "refused_identity_changed")
                send.assert_not_called()

    def test_gone_pid_is_never_signalled(self):
        with tempfile.TemporaryDirectory() as directory:
            guard = watcher.ProcessGuard(43210, {"pid": 43210}, 999, Path(directory))
            with patch.object(signal, "pidfd_send_signal", create=True) as send:
                self.assertEqual(guard.terminate(), "refused_gone")
                send.assert_not_called()

    def test_birth_changes_during_pidfd_capture_refuses_process(self):
        before = {"pid": 43210, "start_time_ticks": 1, "cmdline_sha256": "old"}
        after = {**before, "start_time_ticks": 2}
        with (
            patch.object(watcher, "read_identity", side_effect=[before, after]),
            patch.object(os, "pidfd_open", return_value=999, create=True),
            patch.object(signal, "pidfd_send_signal", create=True) as send,
            patch.object(os, "close") as close,
        ):
            self.assertIsNone(watcher.ProcessGuard.capture(43210))
            send.assert_not_called()
            close.assert_called_once_with(999)

    def test_capture_probes_signal_zero_before_becoming_ready(self):
        identity = {"pid": 43210, "start_time_ticks": 1, "cmdline_sha256": "same"}
        with (
            patch.object(watcher, "read_identity", return_value=identity),
            patch.object(watcher, "open_pidfd", return_value=999),
            patch.object(watcher, "send_pidfd_signal") as send,
        ):
            guard = watcher.ProcessGuard.capture(43210)
            self.assertIsNotNone(guard)
            send.assert_called_once_with(999, 0)

    def test_capture_probe_failure_closes_descriptor_and_refuses_readiness(self):
        identity = {"pid": 43210, "start_time_ticks": 1, "cmdline_sha256": "same"}
        with (
            patch.object(watcher, "read_identity", return_value=identity),
            patch.object(watcher, "open_pidfd", return_value=999),
            patch.object(
                watcher, "send_pidfd_signal", side_effect=PermissionError("denied")
            ),
            patch.object(os, "close") as close,
        ):
            with self.assertRaises(PermissionError):
                watcher.ProcessGuard.capture(43210)
            close.assert_called_once_with(999)


class NativeCompatibilityTests(unittest.TestCase):
    def test_missing_python_bindings_use_libc_pidfds(self):
        open_function, send_function = Mock(return_value=18), Mock(return_value=0)
        libc = SimpleNamespace(
            pidfd_open=open_function, pidfd_send_signal=send_function
        )
        with (
            patch.object(os, "pidfd_open", None, create=True),
            patch.object(signal, "pidfd_send_signal", None, create=True),
            patch.object(watcher.sys, "platform", "linux"),
            patch.object(watcher.ctypes, "CDLL", return_value=libc),
            patch.object(os, "kill") as pid_kill,
        ):
            self.assertEqual(watcher.open_pidfd(43210), 18)
            watcher.send_pidfd_signal(18, signal.SIGTERM)
            open_function.assert_called_once_with(43210, 0)
            send_function.assert_called_once_with(18, signal.SIGTERM, None, 0)
            pid_kill.assert_not_called()

    def test_missing_libc_symbols_use_only_verified_syscalls(self):
        syscall = Mock(side_effect=[19, 0])
        libc = SimpleNamespace(syscall=syscall)
        with (
            patch.object(watcher.sys, "platform", "linux"),
            patch.object(watcher.platform, "machine", return_value="aarch64"),
            patch.object(watcher.ctypes, "CDLL", return_value=libc),
            patch.object(os, "kill") as pid_kill,
        ):
            self.assertEqual(watcher._native_pidfd("pidfd_open", 43210), 19)
            watcher._native_pidfd("pidfd_send_signal", 19, signal.SIGTERM)
            calls = [
                [argument.value for argument in call.args]
                for call in syscall.call_args_list
            ]
            self.assertEqual(
                calls, [[434, 43210, 0], [424, 19, signal.SIGTERM, None, 0]]
            )
            pid_kill.assert_not_called()

    def test_unverified_syscall_architecture_is_rejected(self):
        syscall = Mock(return_value=19)
        with (
            patch.object(watcher.sys, "platform", "linux"),
            patch.object(watcher.platform, "machine", return_value="s390x"),
            patch.object(
                watcher.ctypes, "CDLL", return_value=SimpleNamespace(syscall=syscall)
            ),
        ):
            with self.assertRaisesRegex(
                RuntimeError, "No verified Linux pidfd syscall ABI"
            ):
                watcher._native_pidfd("pidfd_open", 43210)
            syscall.assert_not_called()

    def test_libc_errno_preserves_process_gone_exception(self):
        libc = SimpleNamespace(pidfd_send_signal=Mock(return_value=-1))
        with (
            patch.object(watcher.sys, "platform", "linux"),
            patch.object(watcher.ctypes, "CDLL", return_value=libc),
            patch.object(watcher.ctypes, "get_errno", return_value=errno.ESRCH),
        ):
            with self.assertRaises(ProcessLookupError):
                watcher._native_pidfd("pidfd_send_signal", 19, signal.SIGTERM)

    def test_libc_failure_never_falls_back_to_pid_signal(self):
        libc = SimpleNamespace(pidfd_open=Mock(return_value=-1))
        with (
            patch.object(watcher.sys, "platform", "linux"),
            patch.object(watcher.ctypes, "CDLL", return_value=libc),
            patch.object(watcher.ctypes, "get_errno", return_value=errno.ENOSYS),
            patch.object(os, "kill") as pid_kill,
        ):
            with self.assertRaises(OSError):
                watcher._native_pidfd("pidfd_open", 43210)
            pid_kill.assert_not_called()


class RealChildGuard(watcher.ProcessGuard):
    """Use real child liveness with a fixture Linux /proc identity."""

    def __init__(self, child, root):
        super().__init__(child.pid, watcher.read_identity(child.pid, root), 999, root)
        self.child = child

    def current_state(self):
        return "gone" if self.child.poll() is not None else super().current_state()

    def close(self):
        pass  # Descriptor 999 is the mocked pidfd, not an OS descriptor.


class WatchTests(unittest.TestCase):
    def test_benign_log_never_signals_real_child(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            child = subprocess.Popen(
                [sys.executable, "-c", "import time; time.sleep(0.15)"]
            )
            try:
                write_process(root, child.pid)
                log, output = root / "training.log", root / "watch.json"
                log.write_bytes(b"ordinary progress\nINFO example: " + FATAL + b"\n")
                with patch.object(signal, "pidfd_send_signal", create=True) as send:
                    code = watcher.watch(
                        child.pid,
                        log,
                        output,
                        interval=0.01,
                        guard_factory=lambda _: RealChildGuard(child, root),
                    )
                    send.assert_not_called()
                self.assertEqual(code, 0)
                self.assertEqual(child.wait(timeout=2), 0)
                self.assertEqual(
                    json.loads(output.read_text())["status"], "driver_ended"
                )
            finally:
                if child.poll() is None:
                    child.terminate()
                child.wait(timeout=2)

    def test_fatal_append_split_across_chunks_terminates_only_passed_child(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            child = subprocess.Popen(
                [sys.executable, "-c", "import time; time.sleep(60)"]
            )
            untouched = subprocess.Popen(
                [sys.executable, "-c", "import time; time.sleep(60)"]
            )
            try:
                write_process(root, child.pid)
                log, output = root / "training.log", root / "watch.json"
                log.write_bytes(b"ordinary progress\n")
                fragments = [
                    b"(SGLangGenerationWorker pid=999) " + FATAL[:20],
                    FATAL[20:] + b"\n",
                ]
                signals = []

                def send(descriptor, number, siginfo, flags):
                    self.assertEqual(
                        (descriptor, number, siginfo, flags),
                        (999, signal.SIGTERM, None, 0),
                    )
                    signals.append(child.pid)
                    os.kill(child.pid, number)

                def sleep(interval):
                    if fragments:
                        with log.open("ab") as stream:
                            stream.write(fragments.pop(0))
                    time.sleep(interval)

                with (
                    patch.object(watcher, "READ_BYTES", 7),
                    patch.object(
                        signal, "pidfd_send_signal", side_effect=send, create=True
                    ),
                ):
                    code = watcher.watch(
                        child.pid,
                        log,
                        output,
                        interval=0.01,
                        guard_factory=lambda _: RealChildGuard(child, root),
                        sleeper=sleep,
                    )
                receipt = json.loads(output.read_text())
                self.assertEqual(code, 0)
                self.assertEqual(signals, [child.pid])
                self.assertEqual(child.wait(timeout=2), -signal.SIGTERM)
                self.assertIsNone(untouched.poll())
                self.assertEqual(receipt["signal_action"], "SIGTERM_sent")
                self.assertEqual(receipt["source"]["line_number"], 2)
                self.assertEqual(
                    receipt["source"]["byte_offset"], len(b"ordinary progress\n")
                )
                self.assertEqual(receipt["status"], "fatal_detected_driver_ended")
            finally:
                for process in (child, untouched):
                    if process.poll() is None:
                        process.terminate()
                    process.wait(timeout=2)


if __name__ == "__main__":
    unittest.main()
