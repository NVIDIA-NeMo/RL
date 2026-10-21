# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise the pending constructor/storage change and optional small audit.

The scalar storage double applies real IEEE FP32/BF16 rounding. CUDA kernels and
SGLang packing remain a separate live validation scope.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import logging
import struct
import sys
import tempfile
import unittest
from importlib.machinery import SourceFileLoader
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

SOURCE = Path(__file__).resolve().parents[1] / "src/flint_sglang"
CANDIDATE = SOURCE / "model.py.next"
if not CANDIDATE.exists():
    CANDIDATE = SOURCE / "model.py"
REFIT_CANDIDATE = SOURCE / "refit.py.next"
if not REFIT_CANDIDATE.exists():
    REFIT_CANDIDATE = SOURCE / "refit.py"
loader = SourceFileLoader("flint_refit_candidate", str(REFIT_CANDIDATE))
spec = importlib.util.spec_from_loader(loader.name, loader)
refit = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = refit
loader.exec_module(refit)


def rounded(value, dtype):
    bits = struct.unpack("I", struct.pack("f", value))[0]
    if dtype == "bfloat16":
        bits = ((bits + 0x7FFF + ((bits >> 16) & 1)) >> 16) << 16
    return struct.unpack("f", struct.pack("I", bits))[0]


class Tensor:
    def __init__(self, values, dtype="float32"):
        self.dtype = dtype
        if isinstance(values[0], list):
            self.values = [[rounded(value, dtype) for value in row] for row in values]
            self.shape = (len(values), len(values[0]))
        else:
            self.values = [rounded(value, dtype) for value in values]
            self.shape = (len(values),)

    def copy_(self, source):
        value = Tensor(source.values, self.dtype)
        self.values, self.shape = value.values, value.shape

    def detach(self):
        return self

    def __getitem__(self, slices):
        rows, columns = slices
        return Tensor([row[columns] for row in self.values[rows]], self.dtype)

    def to(self, *, device, dtype):
        assert device == "cpu" and dtype == "float32"
        return Tensor(self.values, dtype)

    def contiguous(self):
        return self

    def numpy(self):
        flat = (
            [value for row in self.values for value in row]
            if len(self.shape) == 2
            else self.values
        )
        return SimpleNamespace(tobytes=lambda: struct.pack("f" * len(flat), *flat))

    def tolist(self):
        return self.values


class ParallelLMHead:
    def __init__(self, vocab, hidden, *, params_dtype=None, **kwargs):
        self.weight = Tensor(
            [[0.0] * hidden for _ in range(vocab)], params_dtype or "bfloat16"
        )


class BaseModule:
    def __init__(self):
        pass


class ParentLoader:
    @staticmethod
    def load_weights(model, weights, is_mtp=False):
        for name, tensor in weights:
            assert name == "lm_head.weight"
            model.lm_head.weight.copy_(tensor)


class HeadStorageTests(unittest.TestCase):
    def test_runner_audits_success_before_clearing_session_coverage(self):
        class Model:
            def __init__(self):
                self._refit_coverage = refit.RefitCoverage()
                self._refit_coverage.initial_load({"lm_head.weight"})

        class Runner:
            def __init__(self):
                self.model = Model()

            def begin_weight_update(self):
                pass

            def end_weight_update(self, run_post_load):
                return "postprocessed"

        refit.install_runner_refit_hooks(Model, Runner)
        runner = Runner()
        runner.begin_weight_update()
        runner.model._refit_coverage.record("lm_head.weight")
        observed = []
        audit = Mock(
            side_effect=lambda model, count: observed.append(
                (count, model._refit_coverage.active)
            )
        )
        with patch.object(refit, "write_refit_audit", audit):
            self.assertEqual(runner.end_weight_update(False), "postprocessed")
        audit.assert_called_once_with(runner.model, 1)
        self.assertEqual(observed, [(1, True)])
        self.assertFalse(runner.model._refit_coverage.active)

    def make_model(self, *, enable_fp32=True, quant_config=None):
        tree = ast.parse(CANDIDATE.read_text())
        cls = next(
            node
            for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "FlintForCausalLM"
        )
        methods = [
            node
            for node in cls.body
            if isinstance(node, ast.FunctionDef)
            and node.name in ("__init__", "load_weights")
        ]
        helpers = [
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef)
            and node.name in ("_target_param_name", "_check_load_coverage")
        ]
        parallel = SimpleNamespace(
            attn_cp_size=1,
            attn_cp_rank=0,
            moe_dp_size=1,
            attn_tp_size=1,
            attn_tp_rank=0,
        )
        namespace = {
            "nn": SimpleNamespace(Module=BaseModule),
            "torch": SimpleNamespace(float32="float32"),
            "get_parallel": lambda: parallel,
            "_check_runtime": lambda _: None,
            "get_pp_group": lambda: None,
            "FlintModel": lambda *args, **kwargs: None,
            "add_prefix": lambda name, prefix: prefix + name,
            "ParallelLMHead": ParallelLMHead,
            "LogitsProcessor": lambda _: None,
            "get_global_server_args": lambda: SimpleNamespace(
                enable_fp32_lm_head=enable_fp32, enable_dp_lm_head=False
            ),
            "FusedMoE": SimpleNamespace(make_expert_params_mapping=lambda **kwargs: []),
            "RefitCoverage": refit.RefitCoverage,
            "install_runner_refit_hooks": lambda _: None,
            "Qwen3MoeForCausalLM": ParentLoader,
            "PPMissingLayer": type("Missing", (), {}),
            "CHECKPOINT_EXPERT_BIAS_SUFFIX": ".mlp.gate.e_score_correction_bias",
            "EXPERT_BIAS_SUFFIX": ".mlp.gate.expert_bias",
            "SINK_SUFFIX": ".self_attn.softmax_offset",
            "logger": logging.getLogger(__name__),
        }
        module = ast.Module(
            body=[
                ast.ImportFrom(
                    module="__future__", names=[ast.alias(name="annotations")], level=0
                ),
                *helpers,
                *methods,
            ],
            type_ignores=[],
        )
        exec(
            compile(ast.fix_missing_locations(module), str(CANDIDATE), "exec"),
            namespace,
        )
        model_type = type(
            "CandidateFlint",
            (BaseModule,),
            {
                "__init__": namespace["__init__"],
                "load_weights": namespace["load_weights"],
                "named_parameters": lambda self: [
                    ("lm_head.weight", self.lm_head.weight)
                ],
                "named_modules": lambda self: [],
                "_skipped_by_parent_loader": lambda *args, **kwargs: False,
            },
        )
        return model_type(
            SimpleNamespace(vocab_size=64, hidden_size=64, num_experts=2),
            quant_config=quant_config,
        )

    def test_unquantized_head_retains_sub_bf16_update_after_real_candidate_refit(self):
        model = self.make_model()
        self.assertEqual(model.lm_head.weight.dtype, "float32")
        initial = Tensor([[1.0] * 64 for _ in range(64)])
        updated = Tensor([[1.00001] * 64 for _ in range(64)])
        self.assertEqual(Tensor(updated.values, "bfloat16").values, initial.values)
        model.load_weights([("lm_head.weight", initial)])
        model._refit_coverage.begin()
        model.load_weights([("lm_head.weight", updated)])
        model._refit_coverage.finish()
        self.assertEqual(model.lm_head.weight.values, updated.values)
        self.assertNotEqual(model.lm_head.weight.values, initial.values)

    def test_disabled_flag_keeps_existing_default_storage(self):
        self.assertEqual(
            self.make_model(enable_fp32=False).lm_head.weight.dtype, "bfloat16"
        )

    def test_quantized_head_keeps_quantization_owned_default(self):
        self.assertEqual(
            self.make_model(quant_config=object()).lm_head.weight.dtype, "bfloat16"
        )

    def test_unset_audit_does_no_io_or_model_access(self):
        with (
            patch.dict(refit.os.environ, {}, clear=True),
            patch.object(refit, "Path", side_effect=AssertionError("unexpected IO")),
        ):
            refit.write_refit_audit(object(), 123)

    def test_audit_hashes_only_small_fp32_samples_and_preserves_resident_dtypes(self):
        head = Tensor([[1.00001] * 64 for _ in range(64)])
        sink = Tensor([0.5, 0.25], "bfloat16")
        model = SimpleNamespace(
            lm_head=SimpleNamespace(weight=head),
            named_parameters=lambda: [
                ("model.layers.0.self_attn.softmax_offset", sink)
            ],
        )
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(
                refit.os.environ, {"FLINT_REFIT_AUDIT_DIR": directory}, clear=True
            ),
            patch.dict(sys.modules, {"torch": SimpleNamespace(float32="float32")}),
        ):
            refit.write_refit_audit(model, 1589)
            paths = list(Path(directory).glob("*.jsonl"))
            self.assertEqual(len(paths), 1)
            report = json.loads(paths[0].read_text())
            self.assertEqual(report["loaded_hf_name_count"], 1589)
            self.assertEqual(report["head"]["dtype"], "float32")
            self.assertEqual(report["head"]["shape"], [64, 64])
            self.assertEqual(report["head"]["sample_shape"], [32, 32])
            self.assertEqual(report["head"]["sample_row_start"], 0)
            expected_bytes = head[:32, :32].numpy().tobytes()
            self.assertEqual(
                report["head"]["sample_sha256"],
                hashlib.sha256(expected_bytes).hexdigest(),
            )
            self.assertEqual(report["sinks"][0]["dtype"], "bfloat16")
            self.assertEqual(report["sinks"][0]["sample_values"], [0.5, 0.25])

    def test_audit_configurable_head_row_start_samples_requested_rows(self):
        head = Tensor([[float(row)] * 64 for row in range(64)])
        model = SimpleNamespace(
            lm_head=SimpleNamespace(weight=head), named_parameters=lambda: []
        )
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict(
                refit.os.environ,
                {
                    "FLINT_REFIT_AUDIT_DIR": directory,
                    "FLINT_REFIT_AUDIT_HEAD_ROW_START": "32",
                },
                clear=True,
            ),
            patch.dict(sys.modules, {"torch": SimpleNamespace(float32="float32")}),
        ):
            refit.write_refit_audit(model, 1589)
            report = json.loads(next(Path(directory).glob("*.jsonl")).read_text())
            self.assertEqual(report["head"]["sample_row_start"], 32)
            self.assertEqual(report["head"]["sample_values"][0][0], 32.0)
            self.assertEqual(report["head"]["sample_values"][-1][0], 63.0)
            expected = hashlib.sha256(head[32:64, :32].numpy().tobytes()).hexdigest()
            self.assertEqual(report["head"]["sample_sha256"], expected)

    def test_audit_rejects_invalid_or_empty_head_row_selection(self):
        model = SimpleNamespace(
            lm_head=SimpleNamespace(weight=Tensor([[0.0] * 64 for _ in range(64)])),
            named_parameters=lambda: [],
        )
        for row in ("-1", "1.5", "bad", "", "１", "64"):
            with (
                self.subTest(row=row),
                tempfile.TemporaryDirectory() as directory,
                patch.dict(
                    refit.os.environ,
                    {
                        "FLINT_REFIT_AUDIT_DIR": directory,
                        "FLINT_REFIT_AUDIT_HEAD_ROW_START": row,
                    },
                    clear=True,
                ),
                patch.dict(sys.modules, {"torch": SimpleNamespace(float32="float32")}),
            ):
                with self.assertRaisesRegex(
                    ValueError, "FLINT_REFIT_AUDIT_HEAD_ROW_START"
                ):
                    refit.write_refit_audit(model, 1589)
                self.assertEqual(list(Path(directory).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
