"""Exercise the actual patched methods without importing GPU/Ray dependencies."""

import ast
import __future__
import asyncio
import contextlib
from functools import partial
from pathlib import Path
from types import SimpleNamespace as NS
import threading
import time
import unittest


class RefitAborted(Exception):
    pass


class RayActorError(Exception):
    pass


def should_use_nemo_gym(config):
    return True


def is_refit_context_lost(error):
    return False


REPO_ROOT = Path(__file__).resolve().parents[3]
source = (REPO_ROOT / "nemo_rl/algorithms/single_controller.py").read_text()
module = ast.parse(source)
controller = next(n for n in module.body if isinstance(n, ast.ClassDef) and n.name == "SingleControllerActor")
methods = [
    n
    for n in controller.body
    if isinstance(n, ast.AsyncFunctionDef) and n.name in {"_sync_weights", "_sync_weights_within"}
]
compiled_class = ast.ClassDef(name="Controller", bases=[], keywords=[], body=methods, decorator_list=[])
namespace = dict(globals(), partial=partial, threading=threading)
exec(
    compile(
        ast.fix_missing_locations(ast.Module(body=[compiled_class], type_ignores=[])),
        "<actual-controller-methods>",
        "exec",
        flags=__future__.annotations.compiler_flag,
    ),
    namespace,
)

Controller = namespace["Controller"]


class PauseTests(unittest.IsolatedAsyncioTestCase):
    def make(
        self,
        *,
        backend="vllm",
        async_engine=True,
        colocated=False,
        recompute=True,
        failure=None,
        pause=True,
        resume=True,
    ):
        c = Controller()
        events = []
        gate = asyncio.Event()
        gate.set()
        c._rollout_permitted = gate
        c._master_config = NS(
            policy={"generation": {"backend": backend, "vllm_cfg": {"async_engine": async_engine}}},
            token_capture=NS(enabled=True),
        )
        c._async_cfg = NS(recompute_kv_cache_after_weight_updates=recompute)

        def record(name, value=None):
            events.append(name)
            self.assertFalse(gate.is_set(), name)
            return value

        c._gen = NS(
            blocks_training=lambda: colocated,
            requires_kv_scale_sync=False,
            pause_generation_for_refit=lambda **kwargs: record(("pause", kwargs["clear_cache"]), pause),
            resume_generation_after_refit=lambda: record("resume", resume),
            invalidate_kv_cache=lambda: record("invalidate", True),
            set_rollout_weight_version=lambda v: record(("version", v)),
        )
        c._rollout_manager = NS(
            suspend_request_deadlines=lambda: record("suspend"),
            resume_request_deadlines=lambda: events.append("deadlines-resumed"),
        )
        c._trainer = NS(sync_params_before_refit=lambda: record("params"))
        c._trainer_version = 2
        c._timer = NS(time=lambda name: contextlib.nullcontext())
        c._refit_await_budget_s = lambda: 0.1
        attempts = []

        def sync(**kwargs):
            record("transfer")
            attempts.append(1)
            if failure and len(attempts) == 1:
                raise failure

        c._weight_synchronizer = NS(sync_weights=sync)

        async def reconcile():
            record("reconcile")

        async def recover(error):
            record("recover")

        c._reconcile_refit_membership = reconcile
        c._recover_from_failed_refit = recover
        c._refit_participants = lambda: []
        c._record_refit_landed = lambda participants: record("landed")
        c._recovery_window = contextlib.nullcontext
        return c, events

    async def test_pause_transfer_resume_order(self):
        c, events = self.make()
        self.assertEqual(await c._sync_weights(), 0)
        self.assertEqual(
            events,
            [
                "suspend",
                ("pause", True),
                "reconcile",
                "reconcile",
                "params",
                "transfer",
                "landed",
                ("version", 2),
                "resume",
                "deadlines-resumed",
            ],
        )
        self.assertTrue(c._rollout_permitted.is_set())

    async def test_keep_cache(self):
        c, events = self.make(recompute=False)
        await c._sync_weights()
        self.assertIn(("pause", False), events)
        self.assertNotIn("invalidate", events)

    async def test_other_backends_and_colocation(self):
        for options in [dict(backend="trtllm"), dict(async_engine=False), dict(colocated=True)]:
            with self.subTest(options=options):
                c, events = self.make(**options)
                await c._sync_weights()
                self.assertNotIn("suspend", events)
                self.assertNotIn("resume", events)
                self.assertIn("invalidate", events)

    async def test_recovery_repauses(self):
        c, events = self.make(failure=RefitAborted("retry"))
        await c._sync_weights()
        self.assertEqual(events.count(("pause", True)), 2)
        self.assertEqual(events.count("transfer"), 2)
        self.assertLess(events.index("recover"), events.index(("pause", True), events.index("recover")))

    async def test_failed_pause_does_not_transfer_or_resume(self):
        c, events = self.make(pause=False)
        with self.assertRaisesRegex(RuntimeError, "pause"):
            await c._sync_weights()
        self.assertNotIn("transfer", events)
        self.assertNotIn("resume", events)
        self.assertFalse(c._rollout_permitted.is_set())

    async def test_failed_transfer_does_not_resume(self):
        c, events = self.make(failure=ValueError("transfer failed"))
        with self.assertRaises(ValueError):
            await c._sync_weights()
        self.assertNotIn("resume", events)
        self.assertFalse(c._rollout_permitted.is_set())

    async def test_failed_resume_keeps_gate_closed(self):
        c, events = self.make(resume=False)
        with self.assertRaisesRegex(RuntimeError, "resume"):
            await c._sync_weights()
        self.assertFalse(c._rollout_permitted.is_set())
        self.assertNotIn("deadlines-resumed", events)

    async def test_callback_timeout(self):
        c, events = self.make()
        c._refit_await_budget_s = lambda: 0.005
        with self.assertRaises(RefitAborted):
            await c._sync_weights_within(None, "pause", operation=lambda: time.sleep(0.02))
        await asyncio.sleep(0.03)

    async def test_unbounded_callback_return(self):
        c, events = self.make()
        c._refit_await_budget_s = lambda: None
        self.assertTrue(await c._sync_weights_within(None, "pause", operation=lambda: True))


if __name__ == "__main__":
    unittest.main()
