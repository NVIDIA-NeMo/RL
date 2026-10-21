#!/usr/bin/env python3
"""Read-only Flint source inventory using metadata and safetensors headers only.

No model code is executed and no tensor payload is read. This establishes shape,
name, dtype and file-layout consistency, not payload integrity or finiteness.
The default expected counts describe the original 49-layer source. Overrides
permit small fixtures or another Flint checkpoint without editing this script.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import socket

from create_four_layer_proxy import DTYPE_BITS, expected_shapes, local_source_file, read_header, read_json


def inspect_source(source: Path, expected_layers: int = 49, expected_tensors: int = 36734,
                   expected_parameters: int = 39785287456, expected_dtype: str = "BF16") -> dict:
    source = source.resolve(strict=True)
    config, config_bytes = read_json(source / "config.json")
    index, index_bytes = read_json(source / "model.safetensors.index.json")
    mapping = index["weight_map"]
    depth = int(config["num_hidden_layers"])
    errors = []
    if depth != expected_layers:
        errors.append(f"Expected {expected_layers} layers, found {depth}")
    schedule = config.get("layer_types", [])
    if len(schedule) != depth or any(kind not in ("full_attention", "sliding_attention") for kind in schedule):
        errors.append("layer_types must contain exactly one valid entry per layer")
    for field in ("mlp_only_layers", "no_rope_layer_indices"):
        values = config.get(field, [])
        if len(values) != len(set(values)) or any(not isinstance(value, int) or not 0 <= value < depth for value in values):
            errors.append(f"Invalid layer indices in {field}")
    shapes = expected_shapes(config, depth)
    if set(mapping) != set(shapes):
        errors.append(f"Index differs from independent shape enumeration: missing={len(set(shapes) - set(mapping))}, extra={len(set(mapping) - set(shapes))}")
    records, names, dtypes = [], set(), Counter()
    parameters, payload_bytes, bytes_read = 0, 0, 0
    protected = []
    for filename in sorted(set(mapping.values())):
        path = local_source_file(source, filename)
        header, offset, info = read_header(path)
        bytes_read += offset
        ranges = []
        count = 0
        for name, spec in header.items():
            if name == "__metadata__":
                continue
            count += 1
            if name in names:
                errors.append(f"Tensor duplicated across source shards: {name}")
            names.add(name)
            if mapping.get(name) != filename:
                errors.append(f"Header/index location mismatch: {name}")
            if shapes.get(name) != spec["shape"]:
                errors.append(f"Shape mismatch: {name}")
            dtype, shape = spec["dtype"], spec["shape"]
            dtypes[dtype] += 1
            if expected_dtype != "any" and dtype != expected_dtype:
                errors.append(f"Unexpected source dtype {dtype} for {name}")
            if dtype not in DTYPE_BITS or any(not isinstance(size, int) or size <= 0 for size in shape):
                raise ValueError(f"Unsupported dtype or invalid source shape for {name}")
            elements = math.prod(shape)
            lower, upper = spec["data_offsets"]
            if not 0 <= lower <= upper or upper - lower != elements * DTYPE_BITS[dtype] // 8:
                errors.append(f"Invalid tensor byte range: {name}")
            ranges.append((lower, upper, name))
            parameters += elements
            payload_bytes += upper - lower
            if name == "lm_head.weight" or name.endswith("self_attn.softmax_offset"):
                protected.append({"name": name, "shape": shape, "source_dtype": dtype})
        cursor = 0
        for lower, upper, name in sorted(ranges):
            if lower != cursor:
                errors.append(f"Gap or overlap in safetensors payload before {name}")
            cursor = upper
        if offset + cursor != info["bytes"]:
            errors.append(f"Safetensors payload end differs from file end: {filename}")
        records.append({"file": filename, "tensor_count": count, **info})
    if names != set(mapping):
        errors.append(f"Header/index names differ: missing={len(set(mapping) - names)}, extra={len(names - set(mapping))}")
    if len(names) != expected_tensors:
        errors.append(f"Expected {expected_tensors} source tensors, found {len(names)}")
    if parameters != expected_parameters:
        errors.append(f"Expected {expected_parameters} parameters, found {parameters}")
    declared_total = index.get("metadata", {}).get("total_size")
    if declared_total is not None and int(declared_total) != payload_bytes:
        errors.append(f"Index total_size={declared_total} differs from payload bytes={payload_bytes}")
    if len(protected) != depth + 1:
        errors.append(f"Expected one LM head and {depth} sinks, found {len(protected)} protected tensors")
    return {"schema_version": 1, "status": "pass" if not errors else "fail",
            "created_at_utc": datetime.now(timezone.utc).isoformat(), "host": socket.gethostname(), "source": str(source),
            "verification_scope": "Metadata and structural header audit only; tensor payload integrity and finiteness are unverified",
            "payload_bytes_read": 0, "weight_file_bytes_read": bytes_read,
            "expected": {"layers": expected_layers, "tensors": expected_tensors, "parameters": expected_parameters, "dtype": expected_dtype},
            "observed": {"num_hidden_layers": depth, "tensor_count": len(names), "parameter_count": parameters,
                         "payload_bytes": payload_bytes, "shard_count": len(records), "dtype_tensor_counts": dict(dtypes),
                         "layer_type_counts": dict(Counter(schedule)), "protected_tensor_count": len(protected)},
            "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
            "index_sha256": hashlib.sha256(index_bytes).hexdigest(), "config": config,
            "files": records, "protected_tensors": sorted(protected, key=lambda item: item["name"]), "errors": errors}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("/flint-model"))
    parser.add_argument("--expected-layers", type=int, default=49)
    parser.add_argument("--expected-tensors", type=int, default=36734)
    parser.add_argument("--expected-parameters", type=int, default=39785287456)
    parser.add_argument("--expected-dtype", default="BF16", help="Safetensors dtype code or any")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if any(value <= 0 for value in (args.expected_layers, args.expected_tensors, args.expected_parameters)):
        parser.error("Expected counts must be positive")
    source, output = args.source.resolve(), args.output.resolve()
    if output == source or source in output.parents:
        raise ValueError("Write the inventory outside the read-only source directory")
    try:
        report = inspect_source(source, args.expected_layers, args.expected_tensors, args.expected_parameters, args.expected_dtype)
    except Exception as error:
        report = {"status": "error", "error": str(error), "source": str(source), "payload_bytes_read": 0}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    args.output.chmod(0o600)
    print(json.dumps({"status": report["status"], "observed": report.get("observed"), "errors": report.get("errors", report.get("error")), "report": str(output)}), flush=True)
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
