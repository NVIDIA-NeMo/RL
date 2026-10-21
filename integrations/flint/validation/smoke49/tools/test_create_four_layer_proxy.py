#!/usr/bin/env python3
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
"""Exercise byte preservation, source selection, and no-overwrite behavior."""

import contextlib
import hashlib
import io
import json
import math
import struct
import tempfile
import unittest
from pathlib import Path

import create_four_layer_proxy as builder


def write_safetensors(path, entries):
    offset, header, chunks = 0, {"__metadata__": {"format": "pt"}}, []
    for name, shape, dtype, payload in entries:
        header[name] = {
            "shape": shape,
            "dtype": dtype,
            "data_offsets": [offset, offset + len(payload)],
        }
        chunks.append(payload)
        offset += len(payload)
    raw = json.dumps(header, separators=(",", ":")).encode()
    raw += b" " * (-len(raw) % 8)
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + b"".join(chunks))


def fixture(source):
    source.mkdir()
    config = {
        "architectures": ["FlintForCausalLM"],
        "model_type": "flint",
        "auto_map": {"AutoConfig": "configuration_flint.FlintConfig"},
        "num_hidden_layers": 5,
        "layer_types": [
            "full_attention",
            "full_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention",
        ],
        "no_rope_layer_indices": [0, 1, 4],
        "mlp_only_layers": [0, 1],
        "hidden_size": 4,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 2,
        "vocab_size": 8,
        "intermediate_size": 6,
        "num_experts": 3,
        "moe_intermediate_size": 2,
        "shared_expert_intermediate_size": 4,
        "router_aux_loss_coef": 0.0005,
        "softmax_type": "learnable",
        "torch_dtype": "bfloat16",
        "sliding_window": 513,
    }
    (source / "config.json").write_text(json.dumps(config))
    (source / "configuration_flint.py").write_text(
        "# Remote config fixture copied byte for byte.\n"
    )
    (source / "tokenizer.json").write_text('{"version":"fixture"}\n')
    (source / "chat_template.jinja").write_text("{{ messages }}\n")
    expected, mapping, groups = (
        {},
        {},
        {"source-a.safetensors": [], "source-b.safetensors": []},
    )
    for number, (name, shape) in enumerate(builder.expected_shapes(config, 5).items()):
        if name.startswith("model.layers.4."):
            mapping[name] = "unselected-payload-must-never-be-opened.safetensors"
            continue
        dtype = (
            "F32"
            if name == "lm_head.weight" or name.endswith("softmax_offset")
            else "BF16"
        )
        size = math.prod(shape) * builder.DTYPE_BITS[dtype] // 8
        payload = bytes((number + offset) % 256 for offset in range(size))
        filename = "source-a.safetensors" if number % 2 else "source-b.safetensors"
        mapping[name] = filename
        groups[filename].append((name, shape, dtype, payload))
        expected[name] = {"shape": shape, "dtype": dtype, "bytes": payload}
    for filename, entries in groups.items():
        write_safetensors(source / filename, entries)
    (source / "model.safetensors.index.json").write_text(
        json.dumps({"metadata": {}, "weight_map": mapping})
    )
    (source / "SHA256SUMS").write_text(
        "".join(
            hashlib.sha256((source / name).read_bytes()).hexdigest()
            + "  "
            + name
            + "\n"
            for name in groups
        )
    )
    return config, expected


class ProxyTests(unittest.TestCase):
    def test_exact_bytes_multishard_config_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, destination = root / "source", root / "proxy"
            config, expected = fixture(source)
            before = {path.name: path.read_bytes() for path in source.iterdir()}
            with contextlib.redirect_stdout(io.StringIO()):
                manifest = builder.build_proxy(
                    source, destination, max_shard_bytes=400, chunk_bytes=3
                )
            self.assertEqual(
                before, {path.name: path.read_bytes() for path in source.iterdir()}
            )
            self.assertFalse((destination / ".PROXY_BUILD_INCOMPLETE").exists())
            self.assertTrue(
                all(not path.is_symlink() for path in destination.rglob("*"))
            )
            self.assertEqual(destination.stat().st_mode & 0o777, 0o700)
            self.assertTrue(
                all(
                    path.stat().st_mode & 0o777 == 0o600
                    for path in destination.rglob("*")
                    if path.is_file()
                )
            )
            derived = json.loads((destination / "config.json").read_text())
            self.assertEqual(derived["num_hidden_layers"], 4)
            self.assertEqual(derived["layer_types"], config["layer_types"][:4])
            self.assertEqual(derived["no_rope_layer_indices"], [0, 1])
            self.assertEqual(derived["mlp_only_layers"], [0, 1])
            self.assertEqual(derived["router_aux_loss_coef"], 0)
            for key in config.keys() - {
                "num_hidden_layers",
                "layer_types",
                "no_rope_layer_indices",
                "router_aux_loss_coef",
            }:
                self.assertEqual(config[key], derived[key])
            mapping = json.loads(
                (destination / "model.safetensors.index.json").read_text()
            )["weight_map"]
            self.assertEqual(set(mapping), set(expected))
            self.assertGreater(len(set(mapping.values())), 1)
            received = {}
            for filename in set(mapping.values()):
                path = destination / filename
                header, base, _ = builder.read_header(path)
                data = path.read_bytes()
                for name, spec in header.items():
                    if name == "__metadata__":
                        continue
                    begin, end = spec["data_offsets"]
                    received[name] = {
                        "dtype": spec["dtype"],
                        "shape": spec["shape"],
                        "bytes": data[base + begin : base + end],
                    }
                self.assertEqual(
                    manifest["files"][filename]["sha256"],
                    hashlib.sha256(data).hexdigest(),
                )
            self.assertEqual(expected, received)
            self.assertEqual(manifest["tensor_count"], len(expected))
            self.assertEqual(
                manifest["raw_tensor_bytes"],
                sum(len(value["bytes"]) for value in expected.values()),
            )
            for entry in manifest["tensors"]:
                self.assertEqual(
                    entry["sha256"],
                    hashlib.sha256(expected[entry["name"]]["bytes"]).hexdigest(),
                )
            self.assertEqual(manifest["source_num_hidden_layers"], 5)
            self.assertEqual(len(manifest["source_weight_files"]), 2)
            for line in (destination / "SHA256SUMS").read_text().splitlines():
                digest, filename = line.split("  ", 1)
                self.assertEqual(
                    digest,
                    hashlib.sha256((destination / filename).read_bytes()).hexdigest(),
                )
            for filename in (
                "configuration_flint.py",
                "tokenizer.json",
                "chat_template.jinja",
            ):
                self.assertEqual(
                    (source / filename).read_bytes(),
                    (destination / filename).read_bytes(),
                )
            with self.assertRaises(FileExistsError):
                builder.build_proxy(source, destination, 400)

    def test_missing_selected_shape_rejected_before_output_creation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, destination = root / "source", root / "proxy"
            fixture(source)
            path = source / "model.safetensors.index.json"
            index = json.loads(path.read_text())
            del index["weight_map"]["model.layers.2.mlp.experts.0.up_proj.weight"]
            path.write_text(json.dumps(index))
            with self.assertRaisesRegex(ValueError, "complete Flint schema"):
                builder.build_proxy(source, destination, 400)
            self.assertFalse(destination.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
