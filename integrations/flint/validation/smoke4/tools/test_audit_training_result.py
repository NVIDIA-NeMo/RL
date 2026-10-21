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
"""Small, actual safetensors fixtures for the training-result audit."""

import json
import tempfile
import unittest
from pathlib import Path

import audit_training_result as auditor
import torch
from create_four_layer_proxy import expected_shapes
from safetensors.torch import save_file


def make_source(root):
    source = root / "source"
    source.mkdir()
    config = {
        "num_hidden_layers": 4,
        "hidden_size": 4,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 2,
        "vocab_size": 8,
        "intermediate_size": 6,
        "num_experts": 3,
        "moe_intermediate_size": 2,
        "shared_expert_intermediate_size": 4,
        "mlp_only_layers": [0, 1],
        "no_rope_layer_indices": [0, 1],
        "layer_types": [
            "full_attention",
            "full_attention",
            "sliding_attention",
            "sliding_attention",
        ],
    }
    values = {
        name: (
            torch.arange(torch.tensor(shape).prod().item()).reshape(shape).float() / 100
            + 1
        ).to(torch.bfloat16)
        for name, shape in expected_shapes(config, 4).items()
    }
    save_file(values, source / "model.safetensors")
    (source / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": dict.fromkeys(values, "model.safetensors")})
    )
    (source / "config.json").write_text(json.dumps(config))
    recipe = root / "recipe.yaml"
    recipe.write_text("policy:\n  optimizer:\n    kwargs:\n      weight_decay: 0.0\n")
    trained = {name: tensor.float() for name, tensor in values.items()}
    trained["lm_head.weight"][0, 0] += 0.001
    trained["model.layers.0.self_attn.softmax_offset"][0] += 0.001
    trained["model.layers.2.mlp.experts.1.gate_proj.weight"][0, 1] += 0.001
    return source, config, trained, recipe


class AuditTests(unittest.TestCase):
    def test_hf_sharded_offsets_and_fp32_updates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, config, trained, recipe = make_source(root)
            model = root / "step1" / "policy" / "model"
            model.mkdir(parents=True)
            for shard in (0, 1):
                payload, metadata = {}, {}
                for name, tensor in trained.items():
                    middle = tensor.shape[0] // 2
                    start, end = (
                        (0, middle) if shard == 0 else (middle, tensor.shape[0])
                    )
                    if start == end:
                        continue
                    payload[name] = tensor[start:end].clone()
                    metadata[name] = {
                        "saved_offsets": [start] + [0] * (tensor.ndim - 1)
                    }
                save_file(
                    payload,
                    model / f"rank{shard}.safetensors",
                    metadata={"DCP_SHARDING_INFO": json.dumps(metadata)},
                )
            (model / "config.json").write_text(json.dumps(config))
            report = auditor.audit(source, root / "step1", recipe, "float32", 7, True)
            self.assertEqual(report["status"], "pass", report["errors"])
            self.assertEqual(report["summary"]["changed_elements"], 3)
            self.assertEqual(report["summary"]["head_changed_elements"], 1)
            self.assertEqual(report["summary"]["sinks_changed_elements"], 1)
            self.assertTrue(report["summary"]["all_source_parameters_covered"])

    def test_native_experts_transposes_and_parameter_aliases(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, config, trained, recipe = make_source(root)
            model = root / "model"
            model.mkdir()
            native = {}
            for name, tensor in trained.items():
                if ".mlp.experts." in name:
                    continue
                name = name.replace(".mlp.shared_expert.", ".mlp.shared_experts.")
                name = name.replace(
                    ".self_attn.softmax_offset",
                    ".self_attn._fp32_params.softmax_offset",
                )
                native[name] = tensor
            for layer in (2, 3):
                prefix = f"model.layers.{layer}.mlp.experts."
                native[prefix + "gate_and_up_projs"] = torch.stack(
                    [
                        torch.cat(
                            [
                                trained[prefix + f"{expert}.gate_proj.weight"].T,
                                trained[prefix + f"{expert}.up_proj.weight"].T,
                            ],
                            dim=1,
                        )
                        for expert in range(config["num_experts"])
                    ]
                )
                native[prefix + "down_projs"] = torch.stack(
                    [
                        trained[prefix + f"{expert}.down_proj.weight"].T
                        for expert in range(config["num_experts"])
                    ]
                )
            save_file(native, model / "native.safetensors")
            report = auditor.audit(source, model, recipe, "float32", 7, True)
            self.assertEqual(report["status"], "pass", report["errors"])
            self.assertEqual(report["summary"]["changed_elements"], 3)
            self.assertEqual(
                report["summary"]["source_tensor_count"],
                report["summary"]["audited_tensor_count"],
            )

    def test_nonfinite_and_protected_dtype_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, config, trained, recipe = make_source(root)
            model = root / "model"
            model.mkdir()
            trained["lm_head.weight"] = trained["lm_head.weight"].to(torch.bfloat16)
            trained["model.norm.weight"][0] = float("nan")
            save_file(trained, model / "bad.safetensors")
            report = auditor.audit(source, model, recipe, "float32", 7, False)
            self.assertEqual(report["status"], "fail")
            self.assertFalse(report["summary"]["all_saved_parameters_finite"])
            self.assertFalse(report["summary"]["dtype_protection_passed"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
