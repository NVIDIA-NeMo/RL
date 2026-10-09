#!/usr/bin/env python3
"""Test the actual worker context and frozen-input TE accumulation in the image."""

import argparse
import ast
from contextlib import contextmanager, nullcontext
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Iterator
import unittest


def load_context(source):
    tree = ast.parse(Path(source).read_text())
    method = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "_preserve_streaming_microbatch_flags"
    )
    namespace = {"contextmanager": contextmanager, "Iterator": Iterator}
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec"), namespace)
    return namespace[method.name]


REPO_ROOT = Path(__file__).resolve().parents[4]
WORKER_SOURCE = REPO_ROOT / "nemo_rl/models/policy/workers/megatron_policy_worker.py"
CONTEXT = load_context(WORKER_SOURCE)


class FlagModel:
    def __init__(self):
        self.calls = 0
        self.first = False

    def set_is_first_microbatch(self):
        self.calls += 1
        self.first = True


class ContextTests(unittest.TestCase):
    def worker(self, model=None):
        worker = SimpleNamespace(model=model or FlagModel())
        worker.context = lambda continuing: CONTEXT(worker, continuing)
        return worker

    def test_first_chunk_and_next_step(self):
        worker = self.worker()
        for _ in range(2):
            for continuing in (False, True, True):
                with worker.context(continuing):
                    worker.model.set_is_first_microbatch()
                    worker.model.first = False
        self.assertEqual(worker.model.calls, 2)
        self.assertNotIn("set_is_first_microbatch", worker.model.__dict__)

    def test_exception_restores_inherited_method(self):
        worker = self.worker()
        with self.assertRaisesRegex(ValueError, "test"):
            with worker.context(True):
                worker.model.set_is_first_microbatch()
                raise ValueError("test")
        self.assertNotIn("set_is_first_microbatch", worker.model.__dict__)
        worker.model.set_is_first_microbatch()
        self.assertEqual(worker.model.calls, 1)

    def test_instance_override_restored(self):
        worker = self.worker()
        override = lambda: "original"
        worker.model.set_is_first_microbatch = override
        with worker.context(True):
            self.assertIsNone(worker.model.set_is_first_microbatch())
        self.assertIs(worker.model.set_is_first_microbatch, override)

    def test_delegated_method_restored(self):
        class Delegated:
            def __init__(self):
                self.inner = FlagModel()

            def __getattr__(self, name):
                return getattr(self.inner, name)

        worker = self.worker(Delegated())
        with worker.context(True):
            worker.model.set_is_first_microbatch()
        self.assertNotIn("set_is_first_microbatch", worker.model.__dict__)
        worker.model.set_is_first_microbatch()
        self.assertEqual(worker.model.inner.calls, 1)

    def test_nested_contexts_restore_exactly(self):
        worker = self.worker()
        with worker.context(True):
            outer = worker.model.set_is_first_microbatch
            with worker.context(True):
                worker.model.set_is_first_microbatch()
            self.assertIs(worker.model.set_is_first_microbatch, outer)
        self.assertNotIn("set_is_first_microbatch", worker.model.__dict__)

    def test_frozen_scalar_gradients(self):
        def run(chunks, patched):
            worker = self.worker()
            grad = 0.0
            index = 0
            for chunk, size in enumerate(chunks):
                with worker.context(chunk > 0) if patched else nullcontext():
                    worker.model.set_is_first_microbatch()
                    for group in range(index, index + size):
                        contribution = (group + 1) / 512
                        grad = contribution if worker.model.first else grad + contribution
                        worker.model.first = False
                    index += size
            return grad

        expected = run([32], False)
        for chunks in ([31, 1], [24, 8], [8, 8, 8, 8]):
            self.assertNotAlmostEqual(run(chunks, False), expected)
            self.assertAlmostEqual(run(chunks, True), expected)


def gpu_tests():
    import torch
    import transformer_engine.pytorch as te
    from transformer_engine.common.recipe import MXFP8BlockScaling
    from megatron.core.transformer.module import MegatronModule
    from torch.utils.checkpoint import checkpoint

    torch.manual_seed(42)
    results = []
    for grouped in (False, True):
        gemms = 2 if grouped else 1
        layer = (
            te.GroupedLinear(gemms, 128, 128, bias=False, params_dtype=torch.bfloat16, fuse_wgrad_accumulation=True)
            if grouped
            else te.Linear(128, 128, bias=False, params_dtype=torch.bfloat16, fuse_wgrad_accumulation=True)
        )
        parameters = list(layer.parameters())
        for parameter in parameters:
            parameter.main_grad = torch.zeros_like(parameter, dtype=torch.float32)
        inputs = [torch.randn(32, 512, 128, device="cuda", dtype=torch.bfloat16) * 0.125 for _ in range(gemms)]
        cotangents = [torch.randn_like(x) * 0.125 for x in inputs]

        class Wrapped(torch.nn.Module):
            def __init__(self, layer=layer):
                super().__init__()
                self.layer = layer
                self.is_first_microbatch = True

            def forward(self, x):
                first = self.is_first_microbatch
                self.is_first_microbatch = False
                with te.fp8_autocast(enabled=True, fp8_recipe=MXFP8BlockScaling()):
                    return (
                        self.layer(x, [x.shape[0] // gemms] * gemms, is_first_microbatch=first)
                        if grouped
                        else self.layer(x, is_first_microbatch=first)
                    )

        model = MegatronModule(SimpleNamespace(fp8="e4m3", fp4=None))
        model.add_module("wrapped", Wrapped())
        worker = SimpleNamespace(model=model)

        def run(chunks, patched, recompute, *, parameters=parameters, worker=worker, model=model):
            for parameter in parameters:
                parameter.main_grad.zero_()
                parameter.grad = None
                parameter.grad_added_to_main_grad = False
            offset = 0
            for chunk, size in enumerate(chunks):
                with CONTEXT(worker, chunk > 0) if patched else nullcontext():
                    # The exact MCore method invoked by each fresh schedule.
                    model.set_is_first_microbatch()
                    x = torch.cat([v[offset : offset + size].reshape(-1, 128) for v in inputs])
                    dy = torch.cat([v[offset : offset + size].reshape(-1, 128) for v in cotangents])
                    x.requires_grad_(True)
                    output = checkpoint(model.wrapped, x, use_reentrant=False) if recompute else model.wrapped(x)
                    (output.float() * dy.float()).sum().backward()
                offset += size
            torch.cuda.synchronize()
            return [p.main_grad.detach().clone() / (32 * 512 * gemms) for p in parameters]

        def errors(candidate, reference):
            return [float((a - b).norm() / b.norm().clamp(min=1e-12)) for a, b in zip(candidate, reference)]

        for recompute in (False, True):
            reference = run([32], False, recompute)
            for chunks in ([31, 1], [24, 8], [8, 8, 8, 8]):
                broken = errors(run(chunks, False, recompute), reference)
                fixed = errors(run(chunks, True, recompute), reference)
                row = {
                    "layer": "GroupedLinear" if grouped else "Linear",
                    "activation_checkpoint": recompute,
                    "chunks": chunks,
                    "unpatched_relative_l2": broken,
                    "patched_relative_l2": fixed,
                }
                print(json.dumps(row), flush=True)
                assert max(broken) > 0.05, row
                assert max(fixed) < 0.005, row
                results.append(row)
        del model, worker, layer, parameters
        torch.cuda.empty_cache()
    print(json.dumps({"gpu_tests": "PASS", "cases": len(results)}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default=str(WORKER_SOURCE))
    parser.add_argument("--gpu", action="store_true")
    args = parser.parse_args()
    CONTEXT = load_context(args.source)
    if args.gpu:
        gpu_tests()
    else:
        unittest.main(argv=["test_streaming_wgrad.py"])
