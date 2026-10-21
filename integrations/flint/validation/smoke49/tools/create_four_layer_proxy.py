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
"""Create a physical four-layer Flint checkpoint with stdlib-only byte copies.

Run this on an allocated compute node, not on a shared login node. No torch,
numpy, or safetensors dependency is required. Only selected tensor payloads are
read; all tensor names, shapes, dtypes, and bytes are preserved exactly. The
default output refuses to overwrite anything and contains no weight symlinks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import socket
import struct
from collections import Counter, defaultdict
from contextlib import ExitStack
from datetime import datetime, timezone
from pathlib import Path

DEFAULT_SOURCE = Path(
    "/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/wenweng/Automodel/cedargrove-model_vv1"
)
DEFAULT_DESTINATION = Path(
    "/lustre/fs1/portfolios/coreai/projects/coreai_dlalgo_nemorl/users/wenweng/Automodel/flint-grpo-4layer-20261019/checkpoint"  # pragma: allowlist secret - filesystem path
)
LAYER_COUNT = 4
DTYPE_BITS = {
    "BOOL": 8,
    "U8": 8,
    "I8": 8,
    "I16": 16,
    "U16": 16,
    "I32": 32,
    "U32": 32,
    "I64": 64,
    "U64": 64,
    "F16": 16,
    "BF16": 16,
    "F32": 32,
    "F64": 64,
    "F8_E4M3": 8,
    "F8_E4M3FN": 8,
    "F8_E5M2": 8,
}
METADATA_FILES = {
    "README.md",
    "LICENSE",
    "LICENSE.md",
    "NOTICE",
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "vocab.txt",
    "merges.txt",
    "tokenizer.model",
    "spiece.model",
    "chat_template.jinja",
}


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    ).encode()


def unique_object(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key {key!r}")
        result[key] = value
    return result


def read_json(path: Path) -> tuple[dict, bytes]:
    raw = path.read_bytes()
    value = json.loads(raw, object_pairs_hook=unique_object)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}")
    return value, raw


def file_stat(path: Path) -> dict:
    value = path.stat()
    return {
        "bytes": value.st_size,
        "mtime_ns": value.st_mtime_ns,
        "inode": value.st_ino,
    }


def local_source_file(source: Path, filename: str) -> Path:
    if (
        not isinstance(filename, str)
        or Path(filename).name != filename
        or filename in (".", "..")
    ):
        raise ValueError(
            f"Source weight filename must be a local basename: {filename!r}"
        )
    path = source / filename
    if not path.is_file() or path.resolve().parent != source:
        raise ValueError(
            f"Source file is missing or resolves outside the source directory: {path}"
        )
    return path


def expected_shapes(config: dict, depth: int) -> dict[str, list[int]]:
    """Enumerate split-expert checkpoint shapes independently of file headers."""
    hidden = int(config["hidden_size"])
    query = int(config["num_attention_heads"]) * int(config["head_dim"])
    kv = int(config["num_key_value_heads"]) * int(config["head_dim"])
    shapes = {
        "model.embed_tokens.weight": [int(config["vocab_size"]), hidden],
        "model.norm.weight": [hidden],
        "lm_head.weight": [int(config["vocab_size"]), hidden],
    }
    for layer in range(depth):
        prefix = f"model.layers.{layer}."
        shapes.update(
            {
                prefix + "input_layernorm.weight": [hidden],
                prefix + "post_attention_layernorm.weight": [hidden],
                prefix + "self_attn.q_proj.weight": [query, hidden],
                prefix + "self_attn.k_proj.weight": [kv, hidden],
                prefix + "self_attn.v_proj.weight": [kv, hidden],
                prefix + "self_attn.o_proj.weight": [hidden, query],
                prefix + "self_attn.q_norm.weight": [int(config["head_dim"])],
                prefix + "self_attn.k_norm.weight": [int(config["head_dim"])],
                prefix + "self_attn.softmax_offset": [
                    int(config["num_attention_heads"])
                ],
            }
        )
        mlps = [("mlp", int(config["intermediate_size"]))]
        if layer not in config["mlp_only_layers"]:
            shapes[prefix + "mlp.gate.weight"] = [int(config["num_experts"]), hidden]
            mlps = [
                ("mlp.shared_expert", int(config["shared_expert_intermediate_size"]))
            ]
            mlps += [
                (f"mlp.experts.{expert}", int(config["moe_intermediate_size"]))
                for expert in range(int(config["num_experts"]))
            ]
        for module, width in mlps:
            shapes[prefix + module + ".gate_proj.weight"] = [width, hidden]
            shapes[prefix + module + ".up_proj.weight"] = [width, hidden]
            shapes[prefix + module + ".down_proj.weight"] = [hidden, width]
    return shapes


def proxy_config(
    original: dict, router_aux_loss_coef: float | None
) -> tuple[dict, dict]:
    if int(original["num_hidden_layers"]) < LAYER_COUNT:
        raise ValueError("Source checkpoint has fewer than four layers")
    if original.get("layer_types", [])[:4] != [
        "full_attention",
        "full_attention",
        "sliding_attention",
        "sliding_attention",
    ]:
        raise ValueError(
            "Expected source first four layers to be full/full/sliding/sliding"
        )
    if original.get("mlp_only_layers") != [0, 1]:
        raise ValueError(
            "Expected source dense MLP layers 0 and 1, followed by MoE layers"
        )
    if len(original["layer_types"]) != int(original["num_hidden_layers"]):
        raise ValueError("Source config layer_types does not match source depth")
    output = dict(original)
    output["num_hidden_layers"] = LAYER_COUNT
    output["layer_types"] = original["layer_types"][:LAYER_COUNT]
    for key in ("no_rope_layer_indices", "mlp_only_layers"):
        output[key] = [index for index in original[key] if index < LAYER_COUNT]
    # Native Automodel validates schedule depth against the physical arrays.
    # The released remote configuration also validates physical array length.
    # Keeping 49 here with four entries fails native config validation.
    if "schedule_num_hidden_layers" in output:
        output["schedule_num_hidden_layers"] = LAYER_COUNT
    if "hybrid_layer_pattern" in output:
        output["hybrid_layer_pattern"] = [
            int(kind == "sliding_attention") for kind in output["layer_types"]
        ]
    if router_aux_loss_coef is not None:
        output["router_aux_loss_coef"] = router_aux_loss_coef
    modifications = {
        key: {"source": original.get(key), "proxy": output[key]}
        for key in output
        if output[key] != original.get(key)
    }
    return output, modifications


def read_header(path: Path) -> tuple[dict, int, dict]:
    before = file_stat(path)
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError(f"Truncated safetensors prefix: {path}")
        length = struct.unpack("<Q", prefix)[0]
        if not 2 <= length <= 100_000_000 or length + 8 > before["bytes"]:
            raise ValueError(f"Invalid safetensors header length: {path}")
        raw = stream.read(length)
    header = json.loads(raw, object_pairs_hook=unique_object)
    if not isinstance(header, dict):
        raise ValueError(f"Expected a safetensors JSON object: {path}")
    info = dict(before)
    info["header_sha256"] = sha256_bytes(prefix + raw)
    info["header_bytes"] = length + 8
    return header, length + 8, info


def source_declared_hashes(path: Path) -> dict[str, str]:
    if not path.is_file():
        return {}
    values = {}
    for line in path.read_text().splitlines():
        match = re.fullmatch(r"([0-9a-fA-F]{64})\s+\*?(.+)", line)
        if match:
            values[match[2].removeprefix("./")] = match[1].lower()
    return values


def build_proxy(
    source: Path,
    destination: Path,
    max_shard_bytes: int,
    router_aux_loss_coef: float | None = 0.0,
    chunk_bytes: int = 8 * 1024 * 1024,
) -> dict:
    source = source.resolve(strict=True)
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(
            f"Refusing to overwrite existing destination: {destination}"
        )
    if destination.resolve() == source or source in destination.resolve().parents:
        raise ValueError(
            "Destination must not be inside the read-only source checkpoint"
        )
    if max_shard_bytes <= 0 or chunk_bytes <= 0:
        raise ValueError("Shard size and chunk size must be positive")
    config, config_raw = read_json(source / "config.json")
    index, index_raw = read_json(source / "model.safetensors.index.json")
    derived_config, modifications = proxy_config(config, router_aux_loss_coef)
    mapping = index["weight_map"]
    complete_shapes = expected_shapes(config, int(config["num_hidden_layers"]))
    if set(mapping) != set(complete_shapes):
        missing = sorted(set(complete_shapes) - set(mapping))[:10]
        extra = sorted(set(mapping) - set(complete_shapes))[:10]
        raise ValueError(
            f"Source index differs from complete Flint schema: missing={missing}, extra={extra}"
        )
    selected_shapes = expected_shapes(config, LAYER_COUNT)
    groups = defaultdict(list)
    for name in selected_shapes:
        groups[mapping[name]].append(name)
    declared_hashes = source_declared_hashes(source / "SHA256SUMS")
    sources, tensors = {}, []
    for filename, names in sorted(groups.items()):
        source_path = local_source_file(source, filename)
        header, data_base, info = read_header(source_path)
        if filename in declared_hashes:
            info["publisher_declared_sha256_not_recomputed"] = declared_hashes[filename]
        sources[filename] = info
        for name in names:
            item = header[name]
            shape, dtype = item["shape"], item["dtype"]
            if shape != selected_shapes[name]:
                raise ValueError(
                    f"Source shape mismatch for {name}: {shape} vs {selected_shapes[name]}"
                )
            if dtype not in DTYPE_BITS:
                raise ValueError(f"Unsupported source dtype {dtype} for {name}")
            begin, end = item["data_offsets"]
            nbytes = math.prod(shape) * DTYPE_BITS[dtype] // 8
            if (
                not 0 <= begin <= end
                or end - begin != nbytes
                or data_base + end > info["bytes"]
            ):
                raise ValueError(f"Invalid tensor byte range for {name}")
            tensors.append(
                {
                    "name": name,
                    "shape": shape,
                    "dtype": dtype,
                    "parameter_count": math.prod(shape),
                    "bytes": nbytes,
                    "source_shard": filename,
                    "source_data_offset": data_base + begin,
                }
            )
    tensors.sort(key=lambda value: (value["source_shard"], value["source_data_offset"]))
    for left, right in zip(tensors, tensors[1:]):
        if (
            left["source_shard"] == right["source_shard"]
            and left["source_data_offset"] + left["bytes"] > right["source_data_offset"]
        ):
            raise ValueError("Selected source tensor byte ranges overlap")
    shards, active, size = [], [], 0
    for item in tensors:
        if active and size + item["bytes"] > max_shard_bytes:
            shards.append(active)
            active, size = [], 0
        active.append(item)
        size += item["bytes"]
    if active:
        shards.append(active)
    # mkdir(exist_ok=False) is the final race-safe no-overwrite check.
    destination.mkdir(parents=True, exist_ok=False, mode=0o700)
    os.chmod(destination, 0o700)
    (destination / ".PROXY_BUILD_INCOMPLETE").write_text(
        "A completed checkpoint has proxy_manifest.json and no incomplete marker.\n"
    )
    metadata = {}
    candidates = [
        path
        for path in source.iterdir()
        if path.is_file() and (path.name in METADATA_FILES or path.suffix == ".py")
    ]
    for path in sorted(candidates):
        target = destination / path.name
        shutil.copyfile(path, target, follow_symlinks=True)
        raw = target.read_bytes()
        metadata[path.name] = {
            "bytes": len(raw),
            "sha256": sha256_bytes(raw),
            "copied_unchanged": True,
        }
    if not (destination / "configuration_flint.py").is_file():
        raise ValueError(
            "Required remote configuration code configuration_flint.py was not copied"
        )
    for path in (source / "chat_templates",):
        if path.is_dir():
            for template in sorted(path.rglob("*")):
                if template.is_file():
                    relative = template.relative_to(source)
                    target = destination / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(template, target, follow_symlinks=True)
                    raw = target.read_bytes()
                    metadata[str(relative)] = {
                        "bytes": len(raw),
                        "sha256": sha256_bytes(raw),
                        "copied_unchanged": True,
                    }
    if (source / "SHA256SUMS").is_file():
        shutil.copyfile(
            source / "SHA256SUMS", destination / "original-source-SHA256SUMS.txt"
        )
    config_output = json_bytes(derived_config)
    (destination / "config.json").write_bytes(config_output)
    files = {
        name: {"bytes": item["bytes"], "sha256": item["sha256"]}
        for name, item in metadata.items()
    }
    files["config.json"] = {
        "bytes": len(config_output),
        "sha256": sha256_bytes(config_output),
    }
    output_mapping = {}
    with ExitStack() as stack:
        handles = {
            filename: stack.enter_context((source / filename).open("rb"))
            for filename in sources
        }
        for shard_id, values in enumerate(shards, 1):
            filename = f"model-{shard_id:05d}-of-{len(shards):05d}.safetensors"
            offset, header = 0, {"__metadata__": {"format": "pt"}}
            for item in values:
                header[item["name"]] = {
                    "dtype": item["dtype"],
                    "shape": item["shape"],
                    "data_offsets": [offset, offset + item["bytes"]],
                }
                offset += item["bytes"]
            header_raw = json.dumps(header, separators=(",", ":")).encode()
            header_raw += b" " * (-len(header_raw) % 8)
            prefix = struct.pack("<Q", len(header_raw)) + header_raw
            output_hash = hashlib.sha256(prefix)
            with (destination / filename).open("xb") as output:
                output.write(prefix)
                for item in values:
                    handle = handles[item["source_shard"]]
                    handle.seek(item["source_data_offset"])
                    remaining, tensor_hash = item["bytes"], hashlib.sha256()
                    while remaining:
                        chunk = handle.read(min(chunk_bytes, remaining))
                        if not chunk:
                            raise EOFError(
                                f"Source payload truncated while copying {item['name']}"
                            )
                        output.write(chunk)
                        output_hash.update(chunk)
                        tensor_hash.update(chunk)
                        remaining -= len(chunk)
                    item["sha256"] = tensor_hash.hexdigest()
                    item["output_shard"] = filename
                    output_mapping[item["name"]] = filename
            files[filename] = {
                "bytes": len(prefix) + offset,
                "sha256": output_hash.hexdigest(),
                "tensor_count": len(values),
                "raw_tensor_bytes": offset,
            }
            print(
                json.dumps(
                    {"event": "shard_complete", "file": filename, **files[filename]}
                ),
                flush=True,
            )
    for filename, previous in sources.items():
        if file_stat(source / filename) != {
            key: previous[key] for key in ("bytes", "mtime_ns", "inode")
        }:
            raise RuntimeError(
                f"Source changed while proxy was being created: {filename}"
            )
    if (source / "config.json").read_bytes() != config_raw or (
        source / "model.safetensors.index.json"
    ).read_bytes() != index_raw:
        raise RuntimeError("Source config/index changed while proxy was being created")
    total_bytes = sum(item["bytes"] for item in tensors)
    output_index = json_bytes(
        {"metadata": {"total_size": total_bytes}, "weight_map": output_mapping}
    )
    (destination / "model.safetensors.index.json").write_bytes(output_index)
    files["model.safetensors.index.json"] = {
        "bytes": len(output_index),
        "sha256": sha256_bytes(output_index),
    }
    for path in destination.rglob("*"):
        os.chmod(path, 0o700 if path.is_dir() else 0o600)
    for filename, info in files.items():
        info["mode"] = f"{(destination / filename).stat().st_mode & 0o777:04o}"
    dtypes = {}
    for dtype in sorted({item["dtype"] for item in tensors}):
        subset = [item for item in tensors if item["dtype"] == dtype]
        dtypes[dtype] = {
            "tensor_count": len(subset),
            "parameter_count": sum(item["parameter_count"] for item in subset),
            "bytes": sum(item["bytes"] for item in subset),
        }
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "host": socket.gethostname(),
        "source": str(source),
        "destination": str(destination.resolve()),
        "source_num_hidden_layers": config["num_hidden_layers"],
        "num_hidden_layers": LAYER_COUNT,
        "source_layer_indices": [0, 1, 2, 3],
        "layer_types": derived_config["layer_types"],
        "mlp_only_layers": derived_config["mlp_only_layers"],
        "no_rope_layer_indices": derived_config["no_rope_layer_indices"],
        "config_modifications": modifications,
        "config_schedule_note": "Physical arrays contain four entries; Automodel schedule depth defaults to four. Original depth49 and source config SHA are recorded here. Explicit arrays preserve layers2,3 as sliding attention.",
        "copy_policy": "Only selected tensor payloads read; raw bytes copied without casting or tensor libraries; no source weight symlinks.",
        "output_permissions": {
            "directory": "0700",
            "files": "0600",
            "source_files_modified": False,
        },
        "source_config_sha256": sha256_bytes(config_raw),
        "source_index_sha256": sha256_bytes(index_raw),
        "source_metadata": {name: item for name, item in metadata.items()},
        "source_weight_files": sources,
        "tensor_count": len(tensors),
        "parameter_count": sum(item["parameter_count"] for item in tensors),
        "raw_tensor_bytes": total_bytes,
        "dtype_summary": dtypes,
        "tensor_counts_by_layer": dict(
            Counter(
                re.match(r"model\.layers\.(\d+)\.", item["name"])[1]
                if re.match(r"model\.layers\.(\d+)\.", item["name"])
                else "global"
                for item in tensors
            )
        ),
        "files": files,
        "tensors": tensors,
        "builder_sha256": sha256_bytes(Path(__file__).read_bytes()),
    }
    manifest_raw = json_bytes(manifest)
    (destination / "proxy_manifest.json").write_bytes(manifest_raw)
    os.chmod(destination / "proxy_manifest.json", 0o600)
    checksums = {name: item["sha256"] for name, item in files.items()}
    checksums["proxy_manifest.json"] = sha256_bytes(manifest_raw)
    original_sums = destination / "original-source-SHA256SUMS.txt"
    if original_sums.is_file():
        checksums[original_sums.name] = sha256_bytes(original_sums.read_bytes())
    (destination / "SHA256SUMS").write_text(
        "".join(f"{value}  {name}\n" for name, value in sorted(checksums.items()))
    )
    os.chmod(destination / "SHA256SUMS", 0o600)
    (destination / ".PROXY_BUILD_INCOMPLETE").unlink()
    print(
        json.dumps(
            {
                "event": "proxy_complete",
                "destination": str(destination),
                "tensor_count": len(tensors),
                "parameter_count": manifest["parameter_count"],
                "raw_tensor_bytes": total_bytes,
                "dtype_summary": dtypes,
            }
        ),
        flush=True,
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--destination", type=Path, default=DEFAULT_DESTINATION)
    parser.add_argument("--max-shard-bytes", type=int, default=2 * 1024**3)
    parser.add_argument(
        "--keep-router-aux-loss",
        action="store_true",
        help="Keep source auxiliary-loss coefficient instead of recording an override to zero",
    )
    args = parser.parse_args()
    build_proxy(
        args.source,
        args.destination,
        args.max_shard_bytes,
        None if args.keep_router_aux_loss else 0.0,
    )


if __name__ == "__main__":
    main()
