"""Tensor comparisons for the MXFP8 numerical and rollout boundary probes."""

import hashlib
import json
import math
import inspect
from pathlib import Path

import torch


def mxfp8_storage_snapshot(shape, metadata):
    """Hash compact, unswizzled TE storage, keeping allocation padding separate."""
    assert len(shape) >= 2
    rows, cols = math.prod(shape[:-1]), shape[-1]
    semantic, raw = {}, {}
    for key, tensor in metadata.items():
        if not isinstance(tensor, torch.Tensor):
            continue
        raw[key] = {"shape": list(tensor.shape), "sha256": tensor_hash(tensor)}
        logical = tensor
        if key in ["rowwise_scale_inv", "columnwise_scale_inv"]:
            assert metadata.get("with_gemm_swizzled_scales") is False, (
                "Cannot crop swizzled or unknown MXFP8 scale layout"
            )
            m, n = (
                (rows, (cols + 31) // 32)
                if key == "rowwise_scale_inv"
                else ((rows + 31) // 32, cols)
            )
            assert tensor.ndim == 2 and tensor.shape[0] >= m and tensor.shape[1] >= n
            logical = tensor[:m, :n]
        semantic[key] = {"shape": list(logical.shape), "sha256": tensor_hash(logical)}
    return {
        "logical_shape": list(shape),
        "semantic": semantic,
        "raw_storage": raw,
        "scale_layout": "compact_unswizzled",
    }


def semantic_model_state(value):
    """Exclude only explicitly tagged raw allocations; keep all active caches."""
    if isinstance(value, dict):
        return {
            k: semantic_model_state(v) for k, v in value.items() if k != "raw_storage"
        }
    if isinstance(value, list):
        return [semantic_model_state(v) for v in value]
    return value


def canonical_request_records(records):
    """Compare request-keyed multisets, preserving multiplicity and raw order on disk."""
    if records and all(record.get("request_key") is not None for record in records):
        return sorted(
            records,
            key=lambda record: (
                record["request_key"],
                json.dumps(record, sort_keys=True),
            ),
        )
    return records


async def collect_indexed_outputs(generator, expected_rows):
    rows = {}
    async for index, output in generator:
        assert index not in rows, f"Duplicate generated row {index}"
        rows[index] = output
    assert set(rows) == set(range(expected_rows)), (
        "Missing or unexpected generated rows"
    )
    return [rows[index] for index in range(expected_rows)]


async def generate_single_sample_rows(generate_async, inputs, *, greedy):
    """vLLM async generation accepts one row; its output index is local to that call."""
    assert inputs.size > 0, "No rollout inputs"
    rows = []
    for prompt_index in range(inputs.size):
        sample = inputs.slice(prompt_index, prompt_index + 1)
        assert sample.size == 1
        local_rows = await collect_indexed_outputs(
            generate_async(sample, greedy=greedy), 1
        )
        assert local_rows[0].size == 1, "Generation returned more than one sample"
        rows.append(local_rows[0])
    return rows


def order_logit_captures(logits, row_ids):
    assert len(logits) == len(row_ids) and len(set(row_ids)) == len(row_ids), (
        "Missing or duplicate teacher-forced logit captures"
    )
    return [
        logit for _, logit in sorted(zip(row_ids, logits), key=lambda item: item[0])
    ]


def ordered_single_sequence_plan(row_ids, bounds, lengths):
    n = len(row_ids)
    assert len(set(row_ids)) == n and len(bounds) == n and len(lengths) == n
    assert bounds == [[i, i + 1] for i in range(n)], (
        "Expected one sequence per packed microbatch"
    )
    order = sorted(range(n), key=lambda i: row_ids[i])
    return order, [[i, i + 1] for i in range(n)], [lengths[i] for i in order]


def forward_input_fingerprint(
    forward, args, kwargs, *, fields=("input_ids", "positions"), include_values=True
):
    try:
        bound = inspect.signature(forward).bind_partial(*args, **kwargs).arguments
    except (TypeError, ValueError):
        bound = {}
    inputs = {**bound, **kwargs}
    result = {}
    for key in fields:
        value = inputs.get(key)
        if isinstance(value, torch.Tensor):
            result[key] = {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "sha256": tensor_hash(value),
            }
            if include_values:
                result[key]["values"] = value.cpu().tolist()
    return result


def tensor_hash(tensor):
    value = tensor.detach().contiguous().cpu()
    digest = hashlib.sha256()
    digest.update(str((tuple(value.shape), value.dtype)).encode())
    digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def tensor_error(actual, expected, *, atol, rtol):
    if actual.shape != expected.shape:
        return {
            "pass": False,
            "reason": "shape",
            "actual": list(actual.shape),
            "expected": list(expected.shape),
        }
    a, b = actual.detach().double().cpu(), expected.detach().double().cpu()
    finite = bool(torch.isfinite(a).all() and torch.isfinite(b).all())
    if not finite:
        return {"pass": False, "reason": "nonfinite"}
    delta = a - b
    norm_a, norm_b = a.norm(), b.norm()
    relative_l2 = float(delta.norm() / norm_b.clamp_min(1e-300))
    return {
        "pass": bool((delta.abs() <= atol + rtol * b.abs()).all()),
        "numel": a.numel(),
        "max_abs": float(delta.abs().max()) if a.numel() else 0.0,
        "relative_l2": relative_l2,
        "cosine": float(
            (a.flatten() @ b.flatten()) / (norm_a * norm_b).clamp_min(1e-300)
        ),
        "different": int((a != b).sum()),
        "atol": atol,
        "rtol": rtol,
    }


def compare_gradient_prefixes(actual, expected, *, atol, rtol):
    if len(actual) != len(expected):
        return {"pass": False, "reason": "missing_prefix_snapshots"}
    for index, (a, b) in enumerate(zip(actual, expected)):
        if a["next_row"] != b["next_row"]:
            return {
                "pass": False,
                "reason": "different_accumulation_order",
                "snapshot": index,
            }
        if set(a["gradients"]) != set(b["gradients"]):
            return {
                "pass": False,
                "reason": "different_gradient_targets",
                "snapshot": index,
            }
        for name in a["gradients"]:
            error = tensor_error(
                a["gradients"][name], b["gradients"][name], atol=atol, rtol=rtol
            )
            if not error["pass"]:
                return {
                    "pass": False,
                    "reason": "different_prefix_gradient",
                    "snapshot": index,
                    "before_row": a["next_row"],
                    "parameter": name,
                    "error": error,
                }
    return {"pass": True, "snapshots": len(actual)}


def name_key(name):
    return hashlib.sha256(name.encode()).hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def compare_tensor_directory(actual_dir, reference_dir, *, atol, rtol):
    actual_dir, reference_dir = Path(actual_dir), Path(reference_dir)
    manifest = json.loads((actual_dir / "manifest.json").read_text())
    reference = json.loads((reference_dir / "manifest.json").read_text())
    if set(manifest) != set(reference):
        return {
            "pass": False,
            "reason": "parameter_set",
            "missing": sorted(set(reference) - set(manifest)),
            "extra": sorted(set(manifest) - set(reference)),
        }
    results = {}
    for name in sorted(manifest):
        a = torch.load(
            actual_dir / manifest[name], map_location="cpu", weights_only=True
        )
        b = torch.load(
            reference_dir / reference[name], map_location="cpu", weights_only=True
        )
        results[name] = tensor_error(a, b, atol=atol, rtol=rtol)
    return {
        "pass": bool(results) and all(r["pass"] for r in results.values()),
        "tensors": results,
    }


def first_boundary_difference(left, right, ordered_stages):
    """Missing evidence is a failure, never evidence for equivalence."""
    for stage in ordered_stages:
        if stage not in left or stage not in right:
            return {"stage": stage, "reason": "missing"}
        if left[stage] != right[stage]:
            return {"stage": stage, "reason": "different"}
    return None
