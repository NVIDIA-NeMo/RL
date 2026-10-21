# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU tests of snapshot lifecycle and the actual Flint loader's accounting.

The loader method is compiled from its source AST, with only the SGLang tensor
copy implementation replaced. This exercises streaming/name/coverage behavior
without pretending to validate CUDA kernels or fused tensor packing.
"""

from __future__ import annotations

import ast
import importlib.util
import logging
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest

SOURCE = Path(__file__).resolve().parents[1] / "src/flint_sglang"
spec = importlib.util.spec_from_file_location("flint_refit_tested", SOURCE / "refit.py")
refit = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = refit
spec.loader.exec_module(refit)


class Tensor:
    def __init__(self, value, dtype="bfloat16"):
        self.value = value
        self.dtype = dtype
        self.shape = (4,)

    def chunk(self, parts, dim):
        assert parts == 1 and dim == 0
        return (self,)


class ParentLoader:
    @staticmethod
    def load_weights(model, weights, is_mtp=False):
        for name, tensor in weights:
            if name == model.fail_on_name:
                raise RuntimeError("synthetic tensor copy failure")
            model.values[name] = tensor


tree = ast.parse((SOURCE / "model.py").read_text())
model_class = next(
    node
    for node in tree.body
    if isinstance(node, ast.ClassDef) and node.name == "FlintForCausalLM"
)
load_method = next(
    node
    for node in model_class.body
    if isinstance(node, ast.FunctionDef) and node.name == "load_weights"
)
helpers = [
    node
    for node in tree.body
    if isinstance(node, ast.FunctionDef)
    and node.name in ("_target_param_name", "_check_load_coverage", "_shard_heads")
]
method_module = ast.Module(
    body=[
        ast.ImportFrom(
            module="__future__", names=[ast.alias(name="annotations")], level=0
        ),
        *helpers,
        load_method,
    ],
    type_ignores=[],
)
namespace = {
    "Qwen3MoeForCausalLM": ParentLoader,
    "get_parallel": lambda: SimpleNamespace(attn_tp_size=1, attn_tp_rank=0),
    "PPMissingLayer": type("PPMissingLayer", (), {}),
    "CHECKPOINT_EXPERT_BIAS_SUFFIX": ".mlp.gate.e_score_correction_bias",
    "EXPERT_BIAS_SUFFIX": ".mlp.gate.expert_bias",
    "SINK_SUFFIX": ".self_attn.softmax_offset",
    "logger": logging.getLogger(__name__),
}
exec(
    compile(ast.fix_missing_locations(method_module), str(SOURCE / "model.py"), "exec"),
    namespace,
)


class Model:
    load_weights = namespace["load_weights"]

    def __init__(self):
        self._refit_coverage = refit.RefitCoverage()
        self.quant_config = None
        self.values = {}
        self.fail_on_name = None
        self.stacked_params_mapping = [
            ("qkv_proj", name + "_proj", name) for name in ("q", "k", "v")
        ]
        self.expert_params_mapping = [
            (
                "experts.w13_weight",
                f"experts.{expert}.{projection}_proj.weight",
                expert,
                projection,
            )
            for expert in range(2)
            for projection in ("gate", "up")
        ]
        self.names = (
            [
                f"model.layers.{layer}.self_attn.{projection}_proj.weight"
                for layer in range(2)
                for projection in ("q", "k", "v")
            ]
            + [f"model.layers.{layer}.self_attn.softmax_offset" for layer in range(2)]
            + [
                f"model.layers.1.mlp.experts.{expert}.{projection}_proj.weight"
                for expert in range(2)
                for projection in ("gate", "up")
            ]
            + ["lm_head.weight"]
        )

    def weights(self, names=None):
        return [
            (
                name,
                Tensor(
                    index + 0.000001,
                    "float32"
                    if name.endswith("softmax_offset") or name == "lm_head.weight"
                    else "bfloat16",
                ),
            )
            for index, name in enumerate(self.names if names is None else names)
        ]

    def named_parameters(self):
        targets = {
            namespace["_target_param_name"](
                name, self.stacked_params_mapping, self.expert_params_mapping
            )
            for name in self.names
        }
        return [(name, object()) for name in targets]

    def named_modules(self):
        return []

    def _skipped_by_parent_loader(self, *args, **kwargs):
        return False


def runner_type():
    class Runner:
        def __init__(self, model):
            self.model = model
            self.calls = []
            self.fail_postprocess = False

        def begin_weight_update(self):
            self.calls.append("restore")

        def end_weight_update(self, run_post_load):
            self.calls.append(("postprocess", run_post_load))
            if self.fail_postprocess:
                raise RuntimeError("postprocess failed")

    refit.install_runner_refit_hooks(Model, Runner)
    return Runner


class RefitTests(unittest.TestCase):
    def setUp(self):
        self.model = Model()
        self.model.load_weights(self.model.weights())
        self.runner = runner_type()(self.model)

    def test_initial_load_requires_all_native_parameters(self):
        model = Model()
        with self.assertRaisesRegex(RuntimeError, "parameters were not loaded"):
            model.load_weights(model.weights(model.names[:-1]))
        self.assertIsNone(model._refit_coverage.expected)

    def test_partial_load_without_session_remains_strict(self):
        with self.assertRaisesRegex(RuntimeError, "parameters were not loaded"):
            self.model.load_weights(self.model.weights(self.model.names[:1]))

    def test_multiple_buckets_layers_and_expert_slices(self):
        self.runner.begin_weight_update()
        for offset in range(0, len(self.model.names), 3):
            self.model.load_weights(
                self.model.weights(self.model.names[offset : offset + 3])
            )
        self.runner.end_weight_update(run_post_load=False)
        self.assertEqual(self.runner.calls, ["restore", ("postprocess", False)])
        self.assertFalse(self.model._refit_coverage.active)
        # A subsequent session starts from empty coverage.
        self.runner.begin_weight_update()
        with self.assertRaisesRegex(RuntimeError, "tensors were not loaded"):
            self.runner.end_weight_update(run_post_load=False)

    def test_missing_fused_projection_slice_is_detected(self):
        self.runner.begin_weight_update()
        names = [name for name in self.model.names if "k_proj" not in name]
        self.model.load_weights(self.model.weights(names))
        with self.assertRaisesRegex(RuntimeError, "k_proj"):
            self.runner.end_weight_update(run_post_load=False)
        self.assertEqual(self.runner.calls, ["restore"])
        self.assertTrue(self.model._refit_coverage.active)

    def test_missing_individual_expert_slice_is_detected(self):
        self.runner.begin_weight_update()
        names = [name for name in self.model.names if "experts.1.up_proj" not in name]
        self.model.load_weights(self.model.weights(names))
        with self.assertRaisesRegex(RuntimeError, "experts.1.up_proj"):
            self.runner.end_weight_update(run_post_load=False)
        self.assertEqual(self.runner.calls, ["restore"])

    def test_duplicate_across_buckets_fails_closed(self):
        self.runner.begin_weight_update()
        weights = self.model.weights(self.model.names[:1])
        self.model.load_weights(weights)
        with self.assertRaisesRegex(RuntimeError, "Duplicate Flint refit"):
            self.model.load_weights(weights)
        with self.assertRaisesRegex(RuntimeError, "session failed"):
            self.runner.end_weight_update(run_post_load=False)
        self.assertEqual(self.runner.calls, ["restore"])

    def test_duplicate_in_one_bucket_fails_closed(self):
        self.runner.begin_weight_update()
        weights = self.model.weights(self.model.names[:1])
        with self.assertRaisesRegex(RuntimeError, "Duplicate Flint checkpoint"):
            self.model.load_weights(weights + weights)
        self.assertTrue(self.model._refit_coverage.failed)

    def test_unknown_tensor_fails_closed(self):
        self.runner.begin_weight_update()
        with self.assertRaisesRegex(RuntimeError, "match no Flint parameter"):
            self.model.load_weights([("unknown.weight", Tensor(1.0))])
        self.assertTrue(self.model._refit_coverage.failed)
        with self.assertRaisesRegex(RuntimeError, "session failed"):
            self.runner.end_weight_update(run_post_load=False)

    def test_tensor_copy_failure_cannot_finalize(self):
        self.runner.begin_weight_update()
        self.model.fail_on_name = self.model.names[2]
        with self.assertRaisesRegex(RuntimeError, "tensor copy failure"):
            self.model.load_weights(self.model.weights())
        with self.assertRaisesRegex(RuntimeError, "session failed"):
            self.runner.end_weight_update(run_post_load=False)
        self.assertEqual(self.runner.calls, ["restore"])

    def test_direct_bypass_cannot_satisfy_coverage(self):
        self.runner.begin_weight_update()
        with self.assertRaisesRegex(RuntimeError, "tensors were not loaded"):
            self.runner.end_weight_update(run_post_load=True)
        self.assertEqual(self.runner.calls, ["restore"])

    def test_refit_preserves_incoming_fp32_tensors(self):
        self.runner.begin_weight_update()
        weights = self.model.weights()
        self.model.load_weights(weights)
        self.runner.end_weight_update(run_post_load=False)
        for name, value in weights:
            self.assertIs(self.model.values[name], value)
            if name.endswith("softmax_offset") or name == "lm_head.weight":
                self.assertEqual(value.dtype, "float32")

    def test_postprocess_failure_leaves_failed_open_session(self):
        self.runner.begin_weight_update()
        self.model.load_weights(self.model.weights())
        self.runner.fail_postprocess = True
        with self.assertRaisesRegex(RuntimeError, "postprocess failed"):
            self.runner.end_weight_update(run_post_load=False)
        self.assertTrue(self.model._refit_coverage.failed)
        self.assertTrue(self.model._refit_coverage.active)

    def test_begin_requires_initialized_model_and_no_nested_session(self):
        fresh = runner_type()(Model())
        with self.assertRaisesRegex(RuntimeError, "full initial"):
            fresh.begin_weight_update()
        self.assertEqual(fresh.calls, [])
        self.runner.begin_weight_update()
        with self.assertRaisesRegex(RuntimeError, "already open"):
            self.runner.begin_weight_update()
        self.assertEqual(self.runner.calls, ["restore"])

    def test_end_without_begin_fails(self):
        with self.assertRaisesRegex(RuntimeError, "without an open session"):
            self.runner.end_weight_update(run_post_load=False)
        self.assertEqual(self.runner.calls, [])

    def test_non_flint_runner_is_unchanged(self):
        runner = runner_type()(object())
        runner.begin_weight_update()
        runner.end_weight_update(run_post_load=True)
        self.assertEqual(runner.calls, ["restore", ("postprocess", True)])

    def test_hook_is_idempotent(self):
        cls = runner_type()
        before = cls.begin_weight_update
        refit.install_runner_refit_hooks(Model, cls)
        self.assertIs(cls.begin_weight_update, before)

    def test_wrong_runner_signature_rejected(self):
        class Unsupported:
            def begin_weight_update(self, unexpected):
                pass

            def end_weight_update(self, run_post_load):
                pass

        with self.assertRaisesRegex(RuntimeError, "Unsupported SGLang"):
            refit.install_runner_refit_hooks(Model, Unsupported)


if __name__ == "__main__":
    unittest.main()
