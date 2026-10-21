# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise the production router child lifecycle without Ray or SGLang."""

from __future__ import annotations

import ast
import sys
import unittest
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock, patch

SOURCE = (
    Path(__file__).resolve().parents[2]
    / "nemo_rl/models/generation/sglang/sglang_router.py"
)


class RouterParentGuardTests(unittest.TestCase):
    def setUp(self) -> None:
        tree = ast.parse(SOURCE.read_text())
        router_class = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "RouterActor"
        )
        functions = [
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "run_router"
        ] + [
            node
            for node in router_class.body
            if isinstance(node, ast.FunctionDef) and node.name in ("start", "stop")
        ]
        self.process = Mock()
        self.process.is_alive.return_value = True
        self.namespace = {
            "os": SimpleNamespace(
                getpid=Mock(return_value=123), getppid=Mock(return_value=123)
            ),
            "multiprocessing": SimpleNamespace(Process=Mock(return_value=self.process)),
            "time": SimpleNamespace(sleep=Mock()),
        }
        module = ast.Module(
            body=[
                ast.ImportFrom(
                    module="__future__", names=[ast.alias(name="annotations")], level=0
                ),
                *functions,
            ],
            type_ignores=[],
        )
        exec(
            compile(ast.fix_missing_locations(module), str(SOURCE), "exec"),
            self.namespace,
        )
        self.guard = Mock()
        self.launch = Mock()
        utils = ModuleType("sglang.srt.utils")
        utils.kill_itself_when_parent_died = self.guard
        router = ModuleType("sglang_router.launch_router")
        router.launch_router = self.launch
        self.modules = patch.dict(
            sys.modules,
            {"sglang.srt.utils": utils, "sglang_router.launch_router": router},
        )
        self.modules.start()
        self.addCleanup(self.modules.stop)

    def test_guard_is_armed_then_parent_checked_before_router_launch(self) -> None:
        events = []
        self.guard.side_effect = lambda: events.append("guard")
        self.namespace["os"].getppid.side_effect = (
            lambda: events.append("parent") or 123
        )
        self.launch.side_effect = lambda args: events.append(("launch", args))
        args = object()
        self.namespace["run_router"](args, 123)
        self.assertEqual(events, ["guard", "parent", ("launch", args)])

    def test_parent_died_before_guard_does_not_launch_router(self) -> None:
        self.namespace["os"].getppid.return_value = 1
        self.namespace["run_router"](object(), 123)
        self.guard.assert_called_once_with()
        self.launch.assert_not_called()

    def test_guard_failure_does_not_launch_unguarded_router(self) -> None:
        self.guard.side_effect = RuntimeError("guard installation failed")
        with self.assertRaisesRegex(RuntimeError, "guard installation failed"):
            self.namespace["run_router"](object(), 123)
        self.launch.assert_not_called()

    def test_start_threads_actual_actor_pid_and_router_args(self) -> None:
        actor = SimpleNamespace()
        args = object()
        self.namespace["start"](actor, args)
        self.namespace["multiprocessing"].Process.assert_called_once_with(
            target=self.namespace["run_router"], args=(args, 123)
        )
        self.assertIs(actor._process, self.process)
        self.assertTrue(self.process.daemon)
        self.process.start.assert_called_once_with()
        self.namespace["time"].sleep.assert_called_once_with(3)

    def test_normal_stop_still_terminates_and_joins(self) -> None:
        actor = SimpleNamespace(_process=self.process)
        self.process.is_alive.side_effect = [True, False]
        self.namespace["stop"](actor, timeout=0.25)
        self.process.terminate.assert_called_once_with()
        self.process.join.assert_called_once_with(timeout=0.25)
        self.process.kill.assert_not_called()

    def test_stop_keeps_forced_kill_fallback(self) -> None:
        actor = SimpleNamespace(_process=self.process)
        self.process.is_alive.return_value = True
        self.namespace["stop"](actor, timeout=0.25)
        self.process.terminate.assert_called_once_with()
        self.process.kill.assert_called_once_with()
        self.assertEqual(self.process.join.call_count, 2)


if __name__ == "__main__":
    unittest.main()
