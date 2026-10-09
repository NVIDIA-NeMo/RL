"""Exercise actual controller selection and sampler ownership without GPU imports."""

import ast
import asyncio
from pathlib import Path
from types import SimpleNamespace
import unittest

SOURCE = Path(__file__).resolve().parents[3] / "nemo_rl"
ROLLOUT_METRICS = "rollout_metrics"


def load_methods(path, source_class, methods, name, bases=()):
    tree = ast.parse((SOURCE / path).read_text())
    original = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == source_class)
    selected = [
        n for n in original.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in methods
    ]
    assert len(selected) == len(methods)
    for method in selected:
        method.decorator_list = [d for d in method.decorator_list if isinstance(d, ast.Name) and d.id == "staticmethod"]
    cls = ast.ClassDef(
        name=name, bases=[ast.Name(id=b, ctx=ast.Load()) for b in bases], keywords=[], body=selected, decorator_list=[]
    )
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), cls], type_ignores=[]
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), globals())
    return globals()[name]


Base = load_methods(
    Path("algorithms/async_utils/staleness_sampler.py"),
    "BaseSampler",
    {"_validate_group_bounds", "_finalize_selection"},
    "Base",
)
Sampler = load_methods(
    Path("algorithms/async_utils/staleness_sampler.py"), "InOrderSampler", {"select"}, "Sampler", ("Base",)
)
Controller = load_methods(
    Path("algorithms/single_controller.py"), "SingleControllerActor", {"_select_training_chunk"}, "Controller"
)
AsyncConfig = load_methods(
    Path("algorithms/single_controller_utils/config.py"),
    "AsyncRLConfig",
    {"_validate_streaming_tail_coalescing"},
    "AsyncConfig",
)
# The real validator decorator is unnecessary outside Pydantic.


class Meta:
    def __init__(self, ids):
        self.ids = list(ids)
        self.extra_info = {ROLLOUT_METRICS: [{"id": i} for i in ids]}

    def concat(self, *others):
        return Meta([i for meta in (self, *others) for i in meta.ids])


class Buffer:
    def __init__(self, groups, ready, future=3):
        self.target_step_list = [0] * groups + [1] * future
        self.ready_list = [i < ready for i in range(groups)] + [True] * future
        self.meta_list = [Meta([i]) for i in range(groups + future)]
        self.claims = set()
        self.claim_calls = 0

    async def claim_for_training(self, indices):
        ids = [i for index in indices for i in self.meta_list[index].ids]
        assert not self.claims.intersection(ids)
        self.claims.update(ids)
        self.claim_calls += 1
        for index in sorted(indices, reverse=True):
            for values in (self.target_step_list, self.ready_list, self.meta_list):
                values.pop(index)

    def finish_current(self):
        for index, target in enumerate(self.target_step_list):
            if target == 0:
                self.ready_list[index] = True

    def drop(self, group):
        index = next(i for i, meta in enumerate(self.meta_list) if meta.ids == [group])
        for values in (self.target_step_list, self.ready_list, self.meta_list):
            values.pop(index)


class TailTests(unittest.IsolatedAsyncioTestCase):
    def make(self, ready, groups=32, enabled=True):
        buffer = Buffer(groups, ready)
        sampler = Sampler()
        sampler._buffer = buffer
        controller = Controller()
        controller._async_cfg = SimpleNamespace(coalesce_streaming_tails=enabled)
        controller._sampler = sampler
        controller._trainer_version = 0
        return controller, buffer

    async def select(self, controller, remaining, floor=8):
        return await controller._select_training_chunk(
            min_prompt_groups=min(floor, remaining), max_prompt_groups=remaining
        )

    async def test_all_ready_is_one_chunk(self):
        c, b = self.make(32)
        meta, count = await self.select(c, 32)
        self.assertEqual((meta.ids, count), (list(range(32)), 32))
        self.assertEqual(b.claims, set(range(32)))
        self.assertEqual(b.claim_calls, 1)
        self.assertEqual(b.target_step_list, [1] * 3)

    async def test_large_partial_leaves_floor_and_claims_once(self):
        c, b = self.make(31)
        first, count = await self.select(c, 32)
        self.assertEqual((first.ids, count), (list(range(24)), 24))
        self.assertEqual(b.claims, set(range(24)))
        self.assertEqual(await self.select(c, 8), (None, 0))
        self.assertEqual(b.claim_calls, 1)
        b.finish_current()
        second, count = await self.select(c, 8)
        self.assertEqual((second.ids, count), (list(range(24, 32)), 8))
        self.assertEqual(b.claims, set(range(32)))
        self.assertEqual(
            first.extra_info[ROLLOUT_METRICS] + second.extra_info[ROLLOUT_METRICS], [{"id": i} for i in range(32)]
        )

    async def test_disabled_retains_greedy_selection(self):
        c, b = self.make(31, enabled=False)
        meta, count = await self.select(c, 32)
        self.assertEqual((meta.ids, count), (list(range(31)), 31))
        b.finish_current()
        tail, count = await self.select(c, 1)
        self.assertEqual((tail.ids, count), ([31], 1))

    async def test_below_floor_does_not_claim_or_yield(self):
        c, b = self.make(7)
        progress = []

        async def marker():
            progress.append("scheduled")

        task = asyncio.create_task(marker())
        self.assertEqual(await self.select(c, 32), (None, 0))
        self.assertEqual(b.claims, set())
        self.assertEqual(b.claim_calls, 0)
        self.assertEqual(progress, [])
        await task

    async def test_remainder_less_than_two_floors_waits_for_whole(self):
        for remaining in (8, 9, 15):
            c, b = self.make(remaining - 1, groups=remaining)
            self.assertEqual(await self.select(c, remaining), (None, 0))
            self.assertEqual(b.claims, set())
            b.finish_current()
            meta, count = await self.select(c, remaining)
            self.assertEqual((meta.ids, count), (list(range(remaining)), remaining))

    async def test_mid_size_coalesces_without_exceeding_bound(self):
        c, b = self.make(15, groups=16)
        meta, count = await self.select(c, 16)
        self.assertEqual((meta.ids, count), (list(range(8)), 8))
        b.finish_current()
        meta, count = await self.select(c, 8)
        self.assertEqual((meta.ids, count), (list(range(8, 16)), 8))

    async def test_drop_during_poll_shrinks_remainder(self):
        c, b = self.make(7, groups=8)
        self.assertEqual(await self.select(c, 8), (None, 0))
        b.drop(7)
        meta, count = await self.select(c, 7)
        self.assertEqual((meta.ids, count), (list(range(7)), 7))

    async def test_drop_after_partial_releases_surviving_tail(self):
        c, b = self.make(31)
        _, count = await self.select(c, 32)
        self.assertEqual(count, 24)
        b.drop(31)
        meta, count = await self.select(c, 7)
        self.assertEqual((meta.ids, count), (list(range(24, 31)), 7))
        self.assertEqual(b.claims, set(range(31)))

    async def test_noncontiguous_ready_preserves_sampler_order(self):
        c, b = self.make(0)
        ready_ids = [i for i in range(32) if i not in (2, 13, 30)]
        for i in ready_ids:
            b.ready_list[i] = True
        meta, count = await self.select(c, 32)
        self.assertEqual((meta.ids, count), (ready_ids[:24], 24))
        self.assertEqual(b.claims, set(ready_ids[:24]))

    async def test_other_sampler_modes_rejected_only_when_enabled(self):
        for enabled in (False, True):
            for name in ("in_order", "ready_first", "custom"):
                cfg = AsyncConfig()
                cfg.coalesce_streaming_tails = enabled
                cfg.sampler = SimpleNamespace(name=name)
                if enabled and name != "in_order":
                    with self.assertRaisesRegex(ValueError, "requires sampler.name=in_order"):
                        cfg._validate_streaming_tail_coalescing()
                else:
                    self.assertIs(cfg._validate_streaming_tail_coalescing(), cfg)


if __name__ == "__main__":
    unittest.main()
