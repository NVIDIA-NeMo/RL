# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Submission boundaries for the optional two-hour training monitor."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from nano35.continue_training import check_once


class ContinuationTest(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.run_dir = Path(self.directory.name)
        self.launcher = Path("/candidate/tools/nano35/launch_e2e.sh")
        (self.run_dir / "job-id.txt").write_text("123\n")

    def make_checkpoint_report(self, *, complete: bool) -> None:
        attempt = self.run_dir / "attempts/123"
        attempt.mkdir(parents=True)
        (attempt / "driver-status.txt").write_text("completed\n")
        (attempt / "checkpoint-report.json").write_text(
            json.dumps({"status": "metadata-passed", "training_complete": complete})
        )

    def test_live_job_never_submits_or_reads_checkpoint(self) -> None:
        for state in ("PENDING", "RUNNING", "COMPLETING"):
            with self.subTest(state=state):
                with (
                    patch(
                        "nano35.continue_training.subprocess.check_output",
                        return_value=f"123|{state}\n",
                    ) as query,
                    patch("nano35.continue_training.subprocess.run") as submit,
                ):
                    result = check_once(self.run_dir, self.launcher)
                self.assertEqual(result["action"], "wait")
                query.assert_called_once()
                submit.assert_not_called()

    def test_failure_or_missing_accounting_never_submits(self) -> None:
        for records in ("123|FAILED|\n", "123|TIMEOUT|\n", "123|CANCELLED|\n", ""):
            with self.subTest(records=records):
                with (
                    patch(
                        "nano35.continue_training.subprocess.check_output",
                        side_effect=["", records],
                    ),
                    patch("nano35.continue_training.subprocess.run") as submit,
                    self.assertRaises(RuntimeError),
                ):
                    check_once(self.run_dir, self.launcher)
                submit.assert_not_called()

    def test_completed_job_requires_driver_and_checkpoint_evidence(self) -> None:
        with (
            patch(
                "nano35.continue_training.subprocess.check_output",
                side_effect=["", "123|COMPLETED|\n"],
            ),
            patch("nano35.continue_training.subprocess.run") as submit,
            self.assertRaises(FileNotFoundError),
        ):
            check_once(self.run_dir, self.launcher)
        submit.assert_not_called()

    def test_final_training_checkpoint_stops_continuation(self) -> None:
        self.make_checkpoint_report(complete=True)
        with (
            patch(
                "nano35.continue_training.subprocess.check_output",
                side_effect=["", "123|COMPLETED|\n"],
            ),
            patch("nano35.continue_training.subprocess.run") as submit,
        ):
            result = check_once(self.run_dir, self.launcher)
        self.assertEqual(result["action"], "finished")
        submit.assert_not_called()

    def test_aged_out_completed_job_calls_guarded_resume_once(self) -> None:
        self.make_checkpoint_report(complete=False)

        def record_next_job(*args, **kwargs) -> None:
            (self.run_dir / "job-id.txt").write_text("124\n")

        with (
            patch(
                "nano35.continue_training.subprocess.check_output",
                side_effect=["", "123|COMPLETED|\n"],
            ) as query,
            patch(
                "nano35.continue_training.subprocess.run",
                side_effect=record_next_job,
            ) as submit,
        ):
            result = check_once(self.run_dir, self.launcher)
        self.assertNotIn("-j", query.call_args_list[0].args[0])
        submit.assert_called_once_with(
            ["bash", str(self.launcher), "resume", str(self.run_dir)], check=True
        )
        self.assertEqual(result["job_id"], "124")

    def test_launcher_supports_ray_sockets_with_long_inherited_temp_paths(self) -> None:
        """Run the real launcher's early setup without querying or submitting jobs."""
        launcher = Path(__file__).resolve().parents[2] / "tools/nano35/launch_e2e.sh"
        probe = self.run_dir / "socket_probe.py"
        probe.write_text(
            "import json, os, socket, tempfile\n"
            "from pathlib import Path\n"
            "with tempfile.TemporaryDirectory(dir=os.environ['RAY_TMPDIR']) as root:\n"
            "    path = Path(root) / 'session_2026-10-04_00-00-00_000000_12345/sockets'\n"
            "    path.mkdir(parents=True)\n"
            "    with socket.socket(socket.AF_UNIX) as server:\n"
            "        server.bind(str(path / 'plasma_store'))\n"
            "    print(json.dumps({key: os.environ[key] for key in "
            "('TMPDIR', 'TMP', 'TEMP', 'RAY_TMPDIR')}))\n"
        )
        inherited = str(self.run_dir / ("long-login-temp-path-" * 8))
        result = subprocess.run(
            [
                "bash",
                "-c",
                'trap \'"$2" "$3"\' EXIT\nsource "$1" invalid "$4"',
                "bash",
                str(launcher),
                sys.executable,
                str(probe),
                str(self.run_dir),
            ],
            env={
                **os.environ,
                **dict.fromkeys(("TMPDIR", "TMP", "TEMP", "RAY_TMPDIR"), inherited),
            },
            text=True,
            capture_output=True,
            check=False,
        )
        self.assertEqual(result.returncode, 2, result.stderr)
        self.assertEqual(
            json.loads(result.stdout),
            dict.fromkeys(("TMPDIR", "TMP", "TEMP", "RAY_TMPDIR"), "/tmp"),
        )


if __name__ == "__main__":
    unittest.main()
