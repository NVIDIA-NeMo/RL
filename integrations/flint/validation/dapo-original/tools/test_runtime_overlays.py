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
"""Exercise the new wrapper with toy sources, without copying private sources."""

import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


SPEC = importlib.util.spec_from_file_location("runtime_overlay_wrapper", Path(__file__).with_name("apply_runtime_overlays.py"))
WRAPPER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(WRAPPER)


def digest(value):
    return hashlib.sha256(value).hexdigest()


class RuntimeOverlayTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.base = self.root / "base"
        self.baseline = b"value = 1\n"
        self.candidate = b"value = 2\n"
        relative = Path("package/layers.py")
        for path, content in [
            (self.base / "evidence/runtime-source/flint_layers.py", self.baseline),
            (self.base / "automodel-overlay" / relative, self.candidate),
        ]:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
        self.target = self.root / "runtime.py"
        self.target.write_bytes(self.baseline)
        self.sg_target = self.root / "sg_runtime.py"
        self.sg_target.write_bytes(self.baseline)
        self.sg_second_target = self.root / "sg_second_runtime.py"
        self.sg_second_target.write_bytes(self.baseline)
        self.router_target = self.root / "router_args.py"
        self.router_baseline = (
            b"class RouterArgs:\n"
            b"    def __init__(self, policy='cache_aware'):\n"
            b"        self.policy = policy\n"
        )
        self.router_candidate = self.router_baseline + WRAPPER.ROUTER_ARGS_SUFFIX
        self.router_target.write_bytes(self.router_baseline)
        self.auto = SimpleNamespace(
            RELATIVE_SOURCE=relative,
            TARGET=self.target,
            BASELINE_SHA256=digest(self.baseline),
            CANDIDATE_SHA256=digest(self.candidate),
            digest=digest,
        )
        self.writes = []

        def atomic_write(path, content, mode=0o600):
            self.writes.append(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
            path.chmod(mode)

        self.sg = SimpleNamespace(digest=digest, atomic_write=atomic_write, prepare=self.prepare_sg)

    def prepare_sg(self, base):
        plans = []
        for target in (self.sg_target, self.sg_second_target):
            before = digest(target.read_bytes())
            if before not in (digest(self.baseline), digest(self.candidate)):
                raise RuntimeError("Unknown toy SGLang source")
            plans.append({
                "target": target,
                "relative_path": target.name,
                "candidate": self.candidate,
                "baseline_sha256": digest(self.baseline),
                "candidate_sha256": digest(self.candidate),
                "before_sha256": before,
                "mode": 0o600,
            })
        return plans, b"toy review patch\n"

    def install(self):
        with (
            patch.object(WRAPPER, "load_module", side_effect=[self.auto, self.sg]),
            patch.multiple(
                WRAPPER,
                ROUTER_ARGS_TARGET=self.router_target,
                ROUTER_ARGS_BASELINE_SHA256=digest(self.router_baseline),
                ROUTER_ARGS_CANDIDATE_SHA256=digest(self.router_candidate),
            ),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            return WRAPPER.install(self.base, self.root / "new/evidence/receipt.json")

    def test_unknown_sglang_preflight_does_not_apply_automodel(self):
        self.sg_target.write_bytes(b"unexpected\n")
        with self.assertRaisesRegex(RuntimeError, "Unknown toy"):
            self.install()
        self.assertEqual(self.target.read_bytes(), self.baseline)
        self.assertEqual(self.writes, [])

    def test_known_overlays_apply_and_repeat_without_runtime_rewrites(self):
        receipt = self.install()
        self.assertEqual(receipt["status"], "pass")
        self.assertEqual(len(receipt["files"]), 4)
        self.assertEqual(self.target.read_bytes(), self.candidate)
        self.assertEqual(self.sg_target.read_bytes(), self.candidate)
        self.assertEqual(self.sg_second_target.read_bytes(), self.candidate)
        self.assertEqual(self.router_target.read_bytes(), self.router_candidate)
        self.writes.clear()
        repeated = self.install()
        self.assertTrue(all(record["status"] == "already_applied" for record in repeated["files"]))
        self.assertEqual(self.writes, [self.root / "new/evidence/receipt.json"])
        stored = json.loads(self.writes[0].read_text())
        self.assertFalse(any("candidate" in item for item in stored["files"]))
        self.assertNotIn("Compatibility for NeMo RL", self.writes[0].read_text())

    def test_redirected_automodel_path_is_rejected(self):
        link = self.root / "redirect.py"
        link.symlink_to(self.target)
        self.auto.TARGET = link
        with self.assertRaisesRegex(RuntimeError, "redirected"):
            self.install()
        self.assertEqual(self.target.read_bytes(), self.baseline)
        self.assertEqual(self.writes, [])

    def test_router_alias_updates_policy_without_an_extra_router_keyword(self):
        self.install()
        namespace = {}
        exec(compile(self.router_target.read_bytes(), str(self.router_target), "exec"), namespace)
        arguments = namespace["RouterArgs"]()
        self.assertEqual(arguments.router_policy, "cache_aware")
        arguments.router_policy = "round_robin"
        self.assertEqual(arguments.policy, "round_robin")
        self.assertEqual(vars(arguments), {"policy": "round_robin"})
        self.assertNotIn("router_policy", vars(arguments))

    def test_unknown_router_preflight_does_not_apply_any_other_overlay(self):
        self.router_target.write_bytes(b"unexpected router\n")
        with self.assertRaisesRegex(RuntimeError, "Unknown router"):
            self.install()
        self.assertEqual(self.target.read_bytes(), self.baseline)
        self.assertEqual(self.sg_target.read_bytes(), self.baseline)
        self.assertEqual(self.sg_second_target.read_bytes(), self.baseline)
        self.assertEqual(self.router_target.read_bytes(), b"unexpected router\n")
        self.assertEqual(self.writes, [])

    def test_redirected_router_path_is_rejected_before_any_write(self):
        original = self.root / "actual_router.py"
        self.router_target.rename(original)
        self.router_target.symlink_to(original)
        with self.assertRaisesRegex(RuntimeError, "redirected router"):
            self.install()
        self.assertEqual(original.read_bytes(), self.router_baseline)
        self.assertEqual(self.target.read_bytes(), self.baseline)
        self.assertEqual(self.sg_target.read_bytes(), self.baseline)
        self.assertEqual(self.sg_second_target.read_bytes(), self.baseline)
        self.assertEqual(self.writes, [])

    def test_known_candidate_still_requires_its_reviewed_baseline(self):
        forged = b"class RouterArgs: pass\n" + WRAPPER.ROUTER_ARGS_SUFFIX
        self.router_target.write_bytes(forged)
        self.router_candidate = forged
        with self.assertRaisesRegex(RuntimeError, "Router baseline differs"):
            self.install()
        self.assertEqual(self.writes, [])

    def test_known_candidate_still_requires_its_reviewed_suffix(self):
        forged = self.router_baseline + b"# unreviewed append\n"
        self.router_target.write_bytes(forged)
        self.router_candidate = forged
        with self.assertRaisesRegex(RuntimeError, "lacks its reviewed suffix"):
            self.install()
        self.assertEqual(self.writes, [])

    def test_all_targets_are_rechecked_before_the_first_mutation(self):
        original = WRAPPER.prepare_router_args

        def change_prior_target_after_preparation():
            plan = original()
            self.sg_second_target.write_bytes(b"changed during preparation\n")
            return plan

        with patch.object(WRAPPER, "prepare_router_args", side_effect=change_prior_target_after_preparation):
            with self.assertRaisesRegex(RuntimeError, "changed after preflight"):
                self.install()
        self.assertEqual(self.target.read_bytes(), self.baseline)
        self.assertEqual(self.sg_target.read_bytes(), self.baseline)
        self.assertEqual(self.router_target.read_bytes(), self.router_baseline)
        self.assertEqual(self.writes, [])


if __name__ == "__main__":
    unittest.main()
