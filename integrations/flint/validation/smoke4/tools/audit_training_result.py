#!/usr/bin/env python3
"""Read-only, chunked audit of a saved Flint training checkpoint.

Prefer the resumable model/ shards, whose dtypes reflect training storage. HF
consolidation can cast ordinary tensors to the original checkpoint dtype. Input
may be a step directory, its policy/weights directory, or model/ itself. No model
is instantiated and no input file is modified. Torch/safetensors/PyYAML must be
available in the training environment. Only the explicitly requested report is
written, with private permissions.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct
import sys


@dataclass
class View:
    file: Path
    saved_name: str
    source_name: str
    shape: list[int]
    offsets: list[int]
    dtype: str
    kind: str = "ordinary"
    local_expert: int = 0
    native_column_start: int = 0

    def read_rows(self, handle, begin: int, end: int):
        value = handle.get_slice(self.saved_name)
        if self.kind == "ordinary":
            return value[(slice(begin, end),) + (slice(None),) * (len(self.shape) - 1)]
        return value[self.local_expert, :, self.native_column_start + begin:self.native_column_start + end].transpose(0, 1)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_header(path: Path) -> dict:
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            raise ValueError(f"Truncated safetensors prefix: {path}")
        length = struct.unpack("<Q", prefix)[0]
        if not 2 <= length <= 100_000_000:
            raise ValueError(f"Invalid safetensors header size: {path}")
        value = json.loads(stream.read(length))
    return value


def canonical_name(name: str) -> str:
    for prefix in ("module.", "_orig_mod.", "base_model.model."):
        while name.startswith(prefix):
            name = name[len(prefix):]
    return name.replace(".self_attn._fp32_params.softmax_offset", ".self_attn.softmax_offset").replace(
        ".mlp.shared_experts.", ".mlp.shared_expert."
    )


def locate_model_directory(checkpoint: Path) -> Path:
    candidates = [checkpoint, checkpoint / "model", checkpoint / "policy" / "model", checkpoint / "weights" / "model",
                  checkpoint / "policy" / "weights" / "model"]
    found = [path for path in candidates if path.is_dir() and any(path.glob("*.safetensors"))]
    if found:
        # A supplied consolidated path is honored; otherwise the raw model/
        # directory wins before searching its consolidated/ child.
        return found[0]
    found = [path for path in checkpoint.rglob("model") if path.is_dir() and any(path.glob("*.safetensors"))]
    if len(found) == 1:
        return found[0]
    if len(found) > 1:
        raise ValueError(f"Several checkpoints found; specify one model directory: {found}")
    consolidated = [path / "consolidated" for path in candidates if (path / "consolidated").is_dir()
                    and any((path / "consolidated").glob("*.safetensors"))]
    if len(consolidated) == 1:
        return consolidated[0]
    raise FileNotFoundError(f"No unambiguous safetensors checkpoint under {checkpoint}")


def describe_views(path: Path, config: dict) -> tuple[list[View], dict]:
    """Map physical HF/native shard fragments into source HF tensor regions."""
    header = read_header(path)
    custom = header.get("__metadata__", {}).get("DCP_SHARDING_INFO", {})
    custom = json.loads(custom) if isinstance(custom, str) else custom
    views = []
    skipped = []
    for saved_name, spec in header.items():
        if saved_name == "__metadata__":
            continue
        name = canonical_name(saved_name)
        if name.endswith("_extra_state"):
            skipped.append(saved_name)
            continue
        shape = spec["shape"]
        offsets = list(custom.get(saved_name, {}).get("saved_offsets", [0] * len(shape)))
        if len(offsets) != len(shape) or any(value < 0 for value in offsets):
            raise ValueError(f"Invalid DCP saved_offsets for {saved_name} in {path}")
        group = re.fullmatch(r"(model\.layers\.\d+\.mlp\.experts)\.(gate_and_up_projs|down_projs)(?:\.weight)?", name)
        if not group:
            if not shape:
                raise ValueError(f"Unexpected scalar model parameter {saved_name}")
            views.append(View(path, saved_name, name, shape, offsets, spec["dtype"]))
            continue
        if len(shape) != 3:
            raise ValueError(f"Native grouped expert tensor must have three dimensions: {saved_name}")
        width = int(config["moe_intermediate_size"])
        for expert in range(shape[0]):
            prefix = f"{group[1]}.{offsets[0] + expert}."
            if group[2] == "down_projs":
                views.append(View(path, saved_name, prefix + "down_proj.weight", [shape[2], shape[1]],
                                  [offsets[2], offsets[1]], spec["dtype"], "native", expert))
            else:
                for suffix, lower, upper in (("gate_proj.weight", 0, width), ("up_proj.weight", width, 2 * width)):
                    start, stop = max(offsets[2], lower), min(offsets[2] + shape[2], upper)
                    if start < stop:
                        views.append(View(path, saved_name, prefix + suffix, [stop - start, shape[1]],
                                          [start - lower, offsets[1]], spec["dtype"], "native", expert, start - offsets[2]))
    return views, {"file": str(path), "bytes": path.stat().st_size,
                   "physical_tensor_count": len(header) - int("__metadata__" in header),
                   "dcp_sharding_metadata": bool(custom), "skipped_extra_state": skipped}


def region_overlaps(left: tuple, right: tuple) -> bool:
    return all(a < d and c < b for (a, b), (c, d) in zip(left, right))


def audit(source: Path, checkpoint: Path, training_config: Path, expected_ordinary_dtype: str,
          chunk_elements: int, require_head_and_sink_update: bool) -> dict:
    import torch
    from safetensors import safe_open
    import yaml

    torch.set_num_threads(min(4, torch.get_num_threads()))
    if chunk_elements <= 0:
        raise ValueError("chunk_elements must be positive")
    source, checkpoint = source.resolve(strict=True), checkpoint.resolve(strict=True)
    source_config = json.loads((source / "config.json").read_text())
    source_index = json.loads((source / "model.safetensors.index.json").read_text())["weight_map"]
    source_shapes, source_dtypes = {}, {}
    source_files = {}
    for filename in sorted(set(source_index.values())):
        if Path(filename).name != filename:
            raise ValueError(f"Source index must use local basenames: {filename}")
        source_files[filename] = source / filename
        header = read_header(source / filename)
        for name, spec in header.items():
            if name in source_index and source_index[name] == filename:
                source_shapes[name] = spec["shape"]
                source_dtypes[name] = spec["dtype"]
    if set(source_shapes) != set(source_index):
        raise ValueError("Source index names missing from source safetensors headers")
    training = yaml.safe_load(training_config.read_text())
    weight_decay = training.get("policy", {}).get("optimizer", {}).get("kwargs", {}).get("weight_decay")
    directory = locate_model_directory(checkpoint)
    files, views = [], []
    for path in sorted(directory.glob("*.safetensors")):
        found, info = describe_views(path, source_config)
        views.extend(found)
        files.append(info)
    report = {
        "schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_proxy": str(source), "checkpoint": str(checkpoint), "model_directory": str(directory),
        "training_config": str(training_config.resolve()), "training_config_sha256": sha256_file(training_config),
        "source_config_sha256": sha256_file(source / "config.json"),
        "source_index_sha256": sha256_file(source / "model.safetensors.index.json"),
        "weight_decay": weight_decay, "expected_ordinary_dtype": expected_ordinary_dtype,
        "protected_parameters": "lm_head.weight and every self_attn.softmax_offset must remain F32 on disk",
        "saved_files": files, "tensors": [], "errors": [], "checkpoint_config": None,
        "comparison": "Every saved parameter region compared exactly against the source proxy promoted to its saved dtype; no full model loaded.",
    }
    if weight_decay is None or float(weight_decay) != 0.0:
        report["errors"].append(f"Training config must explicitly evidence weight_decay=0; found {weight_decay!r}")
    config_paths = [directory / "config.json", directory / ".hf_metadata" / "config.json", directory.parent / "config.json"]
    for path in config_paths:
        if path.is_file():
            saved_config = json.loads(path.read_text())
            report["checkpoint_config"] = {"path": str(path), "sha256": sha256_file(path),
                                           "num_hidden_layers": saved_config.get("num_hidden_layers"),
                                           "layer_types": saved_config.get("layer_types"),
                                           "dtype": saved_config.get("torch_dtype", saved_config.get("dtype"))}
            for key in ("num_hidden_layers", "hidden_size", "vocab_size", "num_experts", "layer_types", "no_rope_layer_indices", "mlp_only_layers"):
                if saved_config.get(key) != source_config.get(key):
                    report["errors"].append(f"Checkpoint config changed {key}: {saved_config.get(key)!r} vs {source_config.get(key)!r}")
            break
    coverage = defaultdict(list)
    aggregate = {}
    bad_finite, bad_dtype, unexpected = [], [], []
    dtype_names = {"float32": "F32", "bfloat16": "BF16", "float16": "F16", "any": None}
    with ExitStack() as stack:
        sources = {name: stack.enter_context(safe_open(path, framework="pt", device="cpu")) for name, path in source_files.items()}
        # Keep source mmap handles open, and only one saved-file mmap at a time.
        by_file = defaultdict(list)
        for view in views:
            by_file[view.file].append(view)
        completed = 0
        for path, items in by_file.items():
            with safe_open(path, framework="pt", device="cpu") as saved:
                for view in items:
                    name = view.source_name
                    if name not in source_shapes:
                        unexpected.append({"saved_name": view.saved_name, "source_name": name, "file": str(path)})
                        # Unknown parameter payloads are still checked for finiteness.
                        shape = view.shape
                        finite = True
                        step = max(1, chunk_elements // max(1, math.prod(shape[1:])))
                        for begin in range(0, shape[0], step):
                            finite &= bool(torch.isfinite(view.read_rows(saved, begin, min(begin + step, shape[0]))).all())
                        if not finite:
                            bad_finite.append(name)
                        continue
                    if len(view.shape) != len(source_shapes[name]) or any(offset + size > full for offset, size, full in zip(view.offsets, view.shape, source_shapes[name])):
                        raise ValueError(f"Saved region falls outside source shape: {name} {view.offsets}/{view.shape} vs {source_shapes[name]}")
                    protected = name == "lm_head.weight" or name.endswith("self_attn.softmax_offset")
                    expected = "F32" if protected else dtype_names[expected_ordinary_dtype]
                    if expected is not None and view.dtype != expected:
                        bad_dtype.append({"name": name, "saved_dtype": view.dtype, "expected_dtype": expected, "file": str(path)})
                    changed, finite, max_diff, sum_abs = 0, True, 0.0, 0.0
                    digest = hashlib.sha256()
                    rows = max(1, chunk_elements // max(1, math.prod(view.shape[1:])))
                    reference = sources[source_index[name]].get_slice(name)
                    for begin in range(0, view.shape[0], rows):
                        stop = min(view.shape[0], begin + rows)
                        current = view.read_rows(saved, begin, stop).contiguous()
                        slicing = tuple(slice(offset + (begin if axis == 0 else 0), offset + (stop if axis == 0 else size))
                                        for axis, (offset, size) in enumerate(zip(view.offsets, view.shape)))
                        original = reference[slicing].to(current.dtype)
                        finite &= bool(torch.isfinite(current).all())
                        changed += int(torch.count_nonzero(current != original))
                        delta = (current.float() - original.float()).abs()
                        max_diff = max(max_diff, float(delta.max()))
                        sum_abs += float(delta.sum(dtype=torch.float64))
                        digest.update(memoryview(current.view(torch.uint8).numpy()))
                    region = tuple((offset, offset + size) for offset, size in zip(view.offsets, view.shape))
                    saved_hash = digest.hexdigest()
                    duplicate = False
                    for prior_region, prior_hash in coverage[name]:
                        if region == prior_region and saved_hash == prior_hash:
                            duplicate = True
                            break
                        if region_overlaps(region, prior_region):
                            report["errors"].append(f"Overlapping or unequal replicated saved regions for {name}")
                    if not duplicate:
                        coverage[name].append((region, saved_hash))
                        item = aggregate.setdefault(name, {"name": name, "source_dtype": source_dtypes[name], "source_shape": source_shapes[name],
                                                           "saved_dtypes": set(), "parameter_count": math.prod(source_shapes[name]),
                                                           "covered_elements": 0, "changed_elements": 0, "finite": True,
                                                           "max_abs_change": 0.0, "sum_abs_change": 0.0, "saved_regions": []})
                        item["saved_dtypes"].add(view.dtype)
                        item["covered_elements"] += math.prod(view.shape)
                        item["changed_elements"] += changed
                        item["finite"] &= finite
                        item["max_abs_change"] = max(item["max_abs_change"], max_diff)
                        item["sum_abs_change"] += sum_abs
                        item["saved_regions"].append({"file": path.name, "saved_name": view.saved_name, "shape": view.shape,
                                                     "offsets": view.offsets, "sha256": saved_hash, "transform": view.kind})
                    if not finite:
                        bad_finite.append(name)
                    completed += 1
                    if completed % 256 == 0:
                        print(json.dumps({"event": "audit_progress", "parameter_regions": completed, "total_regions": len(views)}), flush=True)
    missing = sorted(set(source_shapes) - set(aggregate))
    incomplete = []
    for name, item in sorted(aggregate.items()):
        item["saved_dtypes"] = sorted(item["saved_dtypes"])
        if item["covered_elements"] != item["parameter_count"]:
            incomplete.append({"name": name, "covered_elements": item["covered_elements"], "parameter_count": item["parameter_count"]})
        report["tensors"].append(item)
    changed = [item for item in report["tensors"] if item["changed_elements"]]
    head = aggregate.get("lm_head.weight", {})
    sinks = [item for name, item in aggregate.items() if name.endswith("self_attn.softmax_offset")]
    report["summary"] = {
        "source_tensor_count": len(source_shapes), "audited_tensor_count": len(aggregate),
        "physical_parameter_regions": len(views), "parameter_count": sum(item["parameter_count"] for item in aggregate.values()),
        "changed_tensor_count": len(changed), "changed_elements": sum(item["changed_elements"] for item in changed),
        "max_abs_change": max((item["max_abs_change"] for item in changed), default=0.0),
        "all_saved_parameters_finite": not bad_finite, "dtype_protection_passed": not bad_dtype,
        "all_source_parameters_covered": not missing and not incomplete,
        "head_changed_elements": head.get("changed_elements", 0),
        "sinks_changed_elements": sum(item["changed_elements"] for item in sinks),
        "sinks_with_updates": sum(bool(item["changed_elements"]) for item in sinks),
        "sink_count": len(sinks), "weight_decay_zero_verified": weight_decay is not None and float(weight_decay) == 0,
    }
    for label, values in (("nonfinite_saved_parameters", sorted(set(bad_finite))), ("dtype_failures", bad_dtype),
                          ("missing_parameters", missing), ("incomplete_parameters", incomplete), ("unexpected_parameters", unexpected)):
        report[label] = values
        if values:
            report["errors"].append(f"{label}: {len(values)}")
    if not changed:
        report["errors"].append("No saved parameter changed from its source-proxy value")
    if require_head_and_sink_update and not (head.get("changed_elements") and any(item["changed_elements"] for item in sinks)):
        report["errors"].append("Expected at least one changed LM-head element and one changed attention-sink element")
    report["status"] = "pass" if not report["errors"] else "fail"
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Four-layer source proxy checkpoint")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Saved NeMo RL step or model directory")
    parser.add_argument("--training-config", type=Path, required=True, help="Recipe/resolved config explicitly containing policy.optimizer.kwargs.weight_decay")
    parser.add_argument("--output", type=Path, required=True, help="JSON report; checkpoint/source are read-only")
    parser.add_argument("--expected-ordinary-dtype", choices=("float32", "bfloat16", "float16", "any"), default="float32")
    parser.add_argument("--chunk-elements", type=int, default=1024 * 1024)
    parser.add_argument("--require-head-and-sink-update", action="store_true")
    args = parser.parse_args()
    source = args.source.resolve()
    checkpoint = args.checkpoint.resolve()
    output = args.output.resolve()
    if source in output.parents or checkpoint in output.parents or output in (source, checkpoint):
        raise ValueError("Write the audit report outside the read-only source and checkpoint directories")
    try:
        report = audit(args.source, args.checkpoint, args.training_config, args.expected_ordinary_dtype,
                       args.chunk_elements, args.require_head_and_sink_update)
    except Exception as exc:
        import traceback
        report = {"status": "error", "error": str(exc), "traceback": traceback.format_exc()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    os.chmod(args.output, 0o600)
    print(json.dumps({"status": report["status"], "summary": report.get("summary"), "errors": report.get("errors", report.get("error")),
                      "report": str(args.output)}), flush=True)
    if report["status"] != "pass":
        sys.exit(1)


if __name__ == "__main__":
    main()
