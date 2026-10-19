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

"""Exercise the actual controller method with deterministic backend failures.

AST loading avoids initializing Ray/CUDA; the method body is taken verbatim from
this checkout. This suite tests orchestration, not GPU cache numerics.
"""

import ast
import asyncio
import time
import unittest
from collections.abc import Callable
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import AsyncMock, Mock

ROOT = Path(__file__).resolve().parents[3]


class RefitAborted(RuntimeError):
    pass


class RayActorError(RuntimeError):
    pass


def method(path: str, name: str, extra: dict | None = None) -> Callable:
    tree = ast.parse((ROOT / path).read_text())
    node = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            node,
        ],
        type_ignores=[],
    )
    env = dict(
        asyncio=asyncio,
        time=time,
        RefitAborted=RefitAborted,
        RayActorError=RayActorError,
        should_use_nemo_gym=lambda _: True,
        is_refit_context_lost=lambda _: False,
    )
    env.update(extra or {})
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), env)
    return env[name]


class CacheRefitTests(unittest.TestCase):
    def controller(self, enabled: bool = True) -> tuple[NS, list]:
        events = []
        c = NS(
            _async_cfg=NS(
                recompute_kv_cache_after_weight_updates=enabled,
                generation_fleet_health=NS(refit_timeout_s=1),
            ),
            _master_config=NS(
                policy={
                    "generation": {
                        "backend": "vllm",
                        "vllm_cfg": {"async_engine": True},
                    }
                },
                token_capture=NS(enabled=True),
            ),
            _rollout_permitted=asyncio.Event(),
            _trainer_version=9,
            _timer=NS(time=lambda _: nullcontext()),
        )
        c._rollout_permitted.set()
        c._reconcile_refit_membership = AsyncMock()
        c._refit_participants = lambda: [0, 1]
        c._trainer = NS(sync_params_before_refit=lambda: events.append("prepare"))

        def pause(**kwargs) -> bool:
            assert kwargs == {"clear_cache": True}
            assert not c._rollout_permitted.is_set()
            events.append("pause")
            return True

        def resume() -> bool:
            assert not c._rollout_permitted.is_set()
            events.append("resume")
            return True

        c._gen = NS(
            pause_generation_for_refit=Mock(side_effect=pause),
            resume_generation_after_refit=Mock(side_effect=resume),
            invalidate_kv_cache=Mock(return_value=True),
            set_rollout_weight_version=lambda v: events.append(("version", v)),
        )

        async def sync(*args) -> None:
            events.append("refit")

        c._sync_weights_within = AsyncMock(side_effect=sync)
        c._record_refit_landed = lambda _: events.append("landed")
        c._recover_from_failed_refit = AsyncMock()
        c._recovery_window = nullcontext
        c._rollout_manager = NS(
            resume_request_deadlines=lambda: events.append("deadlines")
        )
        return c, events

    def run_refit(self, c: NS) -> int:
        return asyncio.run(
            method("nemo_rl/algorithms/single_controller.py", "_sync_weights")(c)
        )

    def test_pause_clear_before_refit_stamp_before_resume(self) -> None:
        c, e = self.controller()
        self.run_refit(c)
        self.assertEqual(
            e,
            [
                "prepare",
                "pause",
                "refit",
                "landed",
                ("version", 9),
                "resume",
                "deadlines",
            ],
        )
        self.assertTrue(c._rollout_permitted.is_set())
        c._gen.invalidate_kv_cache.assert_not_called()

    def test_disabled_keeps_old_path(self) -> None:
        c, e = self.controller(False)
        self.run_refit(c)
        c._gen.pause_generation_for_refit.assert_not_called()
        c._gen.resume_generation_after_refit.assert_not_called()
        self.assertTrue(c._rollout_permitted.is_set())

    def test_pause_false_or_none_never_updates_weights(self) -> None:
        for result in [False, None]:
            with self.subTest(result=result):
                c, e = self.controller()
                c._gen.pause_generation_for_refit = Mock(return_value=result)
                with self.assertRaisesRegex(RuntimeError, "confirmed pause"):
                    self.run_refit(c)
                c._sync_weights_within.assert_not_called()
                self.assertFalse(c._rollout_permitted.is_set())

    def test_pause_exception_never_updates_weights(self) -> None:
        c, e = self.controller()
        c._gen.pause_generation_for_refit = Mock(
            side_effect=RuntimeError("pause failed")
        )
        with self.assertRaisesRegex(RuntimeError, "pause failed"):
            self.run_refit(c)
        c._sync_weights_within.assert_not_called()
        c._gen.resume_generation_after_refit.assert_not_called()

    def test_refit_failures_do_not_resume_or_recover_partial_weights(self) -> None:
        for error in [
            RefitAborted("partial"),
            RayActorError("dead"),
            RuntimeError("bad weights"),
        ]:
            with self.subTest(error=type(error)):
                c, e = self.controller()
                c._sync_weights_within = AsyncMock(side_effect=error)
                with self.assertRaises(type(error)):
                    self.run_refit(c)
                c._gen.resume_generation_after_refit.assert_not_called()
                c._recover_from_failed_refit.assert_not_called()
                self.assertFalse(c._rollout_permitted.is_set())
                self.assertNotIn(("version", 9), e)

    def test_version_failure_keeps_generation_paused(self) -> None:
        c, e = self.controller()
        c._gen.set_rollout_weight_version = Mock(
            side_effect=RuntimeError("stamp failed")
        )
        with self.assertRaisesRegex(RuntimeError, "stamp failed"):
            self.run_refit(c)
        c._gen.resume_generation_after_refit.assert_not_called()
        self.assertFalse(c._rollout_permitted.is_set())

    def test_resume_failure_keeps_dispatch_closed(self) -> None:
        for result in [False, None]:
            with self.subTest(result=result):
                c, e = self.controller()
                c._gen.resume_generation_after_refit = Mock(return_value=result)
                with self.assertRaisesRegex(RuntimeError, "resume every"):
                    self.run_refit(c)
                self.assertFalse(c._rollout_permitted.is_set())
                self.assertNotIn("deadlines", e)

    def test_fallback_invalidation_failure_is_not_ignored(self) -> None:
        c, e = self.controller()
        c._master_config.policy["generation"]["backend"] = "other"
        c._gen.invalidate_kv_cache.return_value = False
        with self.assertRaisesRegex(RuntimeError, "invalidation failed"):
            self.run_refit(c)
        self.assertFalse(c._rollout_permitted.is_set())

    def test_worker_preserves_reset_boolean(self) -> None:
        fn = method(
            "nemo_rl/models/generation/vllm/vllm_worker_async.py",
            "reset_prefix_cache_async",
            {
                "gc": NS(collect=lambda: None),
                "torch": NS(cuda=NS(empty_cache=lambda: None)),
            },
        )
        for result in [True, False]:
            c = NS(
                llm=NS(reset_prefix_cache=AsyncMock(return_value=result)),
                cfg={"vllm_cfg": {"async_engine": True}},
            )
            self.assertIs(asyncio.run(fn(c)), result)

    def test_reset_aggregation_requires_positive_ack(self) -> None:
        fn = method(
            "nemo_rl/models/generation/vllm/vllm_generation.py",
            "invalidate_kv_cache",
            {"ray": NS(get=lambda x: x)},
        )
        for results, expected in [
            ([], False),
            ([None], False),
            ([False], False),
            ([True, False], False),
            ([True, None], False),
            ([True, True], True),
        ]:
            c = NS(
                cfg={"vllm_cfg": {"async_engine": True}},
                worker_group=NS(run_all_workers_single_data=lambda *a, **k: results),
            )
            self.assertIs(fn(c), expected)

    def test_pause_resume_aggregation_requires_positive_ack(self) -> None:
        for name in ["pause_generation_for_refit", "resume_generation_after_refit"]:
            fn = method(
                "nemo_rl/models/generation/vllm/vllm_generation.py",
                name,
                {"ray": NS(get=lambda x: x)},
            )
            for results in [[], [None], [True, None], [True, False], [True, True]]:
                with self.subTest(method=name, results=results):
                    c = NS(
                        cfg={"vllm_cfg": {"async_engine": True}},
                        worker_group=NS(
                            workers=[object(), object()],
                            run_all_workers_single_data=lambda *a, **k: results,
                        ),
                    )
                    kwargs = {"clear_cache": True} if name.startswith("pause") else {}
                    if results == [True, True]:
                        self.assertIs(fn(c, **kwargs), True)
                    else:
                        with self.assertRaises(RuntimeError):
                            fn(c, **kwargs)

    def test_worker_pause_preserves_requests_and_clears_cache(self) -> None:
        fn = method(
            "nemo_rl/models/generation/vllm/vllm_worker_async.py",
            "pause_generation_async",
        )
        c = NS(
            cfg={"vllm_cfg": {"async_engine": True}},
            llm=NS(pause_generation=AsyncMock()),
        )
        self.assertIs(asyncio.run(fn(c, clear_cache=True)), True)
        c.llm.pause_generation.assert_awaited_once_with(mode="keep", clear_cache=True)


if __name__ == "__main__":
    unittest.main()
