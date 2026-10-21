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
"""Check continuation decisions against small metadata-only fixtures."""

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location("checkpoint_window", Path(__file__).with_name("checkpoint_window.py"))
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class CheckpointWindowTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.log = self.root / "training.log"
        self.log.write_text("Timeout has been reached, stopping training early\n")

    def checkpoint(self, step):
        path = self.root / "results/dapo50" / f"step_{step}"
        for relative, value in [
            ("config.yaml", "{}\n"),
            ("training_info.json", json.dumps({"total_steps": step})),
            ("policy/optimizer/optim/.metadata", "fixture"),
            ("policy/weights/model/model.safetensors", "fixture"),
        ]:
            target = path / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(value)
        return path

    def test_fresh_run_has_no_resume_checkpoint(self):
        self.assertEqual(MODULE.latest_checkpoint(self.root), {"step": 0, "path": None})

    def test_unfinalized_checkpoint_is_ignored(self):
        path = self.checkpoint(5)
        (path.parent / "tmp_step_9").mkdir()
        self.assertEqual(MODULE.latest_checkpoint(self.root)["step"], 5)

    def test_clean_timeout_with_progress_continues(self):
        self.checkpoint(8)
        result = MODULE.finish_window(self.root, {"step": 5}, self.log)
        self.assertEqual(result["disposition"], "continue")
        self.assertEqual(result["checkpoint_step"], 8)

    def test_no_progress_does_not_submit_another_window(self):
        self.checkpoint(5)
        with self.assertRaisesRegex(RuntimeError, "did not advance"):
            MODULE.finish_window(self.root, {"step": 5}, self.log)

    def test_unexpected_early_exit_does_not_continue(self):
        self.checkpoint(5)
        self.log.write_text("ordinary early return\n")
        with self.assertRaisesRegex(RuntimeError, "not the configured"):
            MODULE.finish_window(self.root, {"step": 0}, self.log)

    def test_final_checkpoint_completes(self):
        self.checkpoint(50)
        result = MODULE.finish_window(self.root, {"step": 45}, self.log)
        self.assertEqual(result["disposition"], "complete")

    def test_missing_optimizer_prevents_resume(self):
        checkpoint = self.checkpoint(5)
        (checkpoint / "policy/optimizer/optim/.metadata").unlink()
        with self.assertRaisesRegex(RuntimeError, "missing"):
            MODULE.latest_checkpoint(self.root)

    def test_saved_step_must_match_directory(self):
        checkpoint = self.checkpoint(5)
        (checkpoint / "training_info.json").write_text('{"total_steps":4}')
        with self.assertRaisesRegex(RuntimeError, "disagree"):
            MODULE.latest_checkpoint(self.root)


if __name__ == "__main__":
    unittest.main()
