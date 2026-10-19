# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare selected native call fingerprints to a canonical TMPE dump on CPU.

The lineage proof must already authenticate each selected model call's token
prefix and chain against this same raw dump. Missing traces remain unverified.
This tool neither runs a model nor changes existing training data or artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import torch


def _digest(tensor: torch.Tensor, dtype: str) -> str:
    if tensor.device.type != "cpu":
        raise ValueError("Prediction evidence comparison requires CPU tensors")
    return hashlib.sha256(
        tensor.detach()
        .contiguous()
        .numpy()
        .astype(dtype, copy=False)
        .tobytes(order="C")
    ).hexdigest()


def compare_predictions(
    raw: dict[str, Any],
    selected_calls: list[dict[str, Any]],
    evidence: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    """Compare only original call IDs authenticated by the supplied lineage."""
    for key in ("input_ids", "input_lengths", "generation_logprobs", "routed_experts"):
        value = raw.get(key)
        if isinstance(value, torch.Tensor) and value.device.type != "cpu":
            raise ValueError("Prediction evidence comparison requires CPU tensors")
    valid_length = int(raw["input_lengths"])
    if valid_length != raw["sequence_length"]:
        raise ValueError("Raw TMPE dump has inconsistent valid length")
    by_call = defaultdict(list)
    for event in evidence:
        if event.get("event") == "captured_prediction_evidence":
            by_call[event.get("model_call_id")].append(event)
    results = []
    for call in selected_calls:
        call_id = call["model_call_id"]
        if not (call["token_prefix_sha_matches"] and call["chain_hash_matches"]):
            raise ValueError("Selected call lineage is not authenticated")
        candidates = by_call[call_id]
        item = {"model_call_id": call_id, "status": "missing_trace"}
        if len(candidates) > 1:
            # Duplicate/replayed records are acceptable only when identical.
            unique = {json.dumps(c, sort_keys=True) for c in candidates}
            if len(unique) != 1:
                item.update(status="ambiguous_trace", candidate_count=len(candidates))
                results.append(item)
                continue
        if candidates:
            event = candidates[0]
            item["status"] = event.get("status", "unavailable")
            if event.get("status") == "available":
                p, g = event["prompt_tokens"], event["generation_tokens"]
                total = p + g
                if not (
                    event.get("digest_encoding")
                    == "contiguous-little-endian-i64-f32-i16-v1"
                    and 0 <= call["prev_len"] <= p
                    and p > 0
                    and g >= 0
                    and total == call["cum_len"] <= valid_length
                    and event["predictor_start"] == p - 1
                    and event["predictor_end"] == total - 1
                ):
                    item.update(status="unaligned_trace")
                    results.append(item)
                    continue
                routes = raw.get("routed_experts")
                item.update(
                    prompt_tokens=p,
                    generation_tokens=g,
                    admission_weight_version=event["admission_weight_version"],
                    response_weight_version=event["response_weight_version"],
                    prompt_token_ids_match=_digest(
                        raw["input_ids"][:p].to(torch.int64), "<i8"
                    )
                    == event["prompt_token_ids_sha256"],
                    generation_token_ids_match=_digest(
                        raw["input_ids"][p:total].to(torch.int64), "<i8"
                    )
                    == event["generation_token_ids_sha256"],
                    generation_logprobs_match=_digest(
                        raw["generation_logprobs"][p:total].float(), "<f4"
                    )
                    == event["generation_logprobs_sha256"],
                    generation_predictor_routes_match=(
                        list(routes[p - 1 : total - 1].shape)
                        == event["predictor_shape"]
                        and _digest(routes[p - 1 : total - 1].to(torch.int16), "<i2")
                        == event["generation_predictor_routes_sha256"]
                    )
                    if routes is not None
                    else None,
                )
        results.append(item)
    complete = bool(results) and all(r["status"] == "available" for r in results)
    return {
        "selected_calls": len(results),
        "complete_call_evidence": complete,
        "all_original_token_ids_and_logprobs_match": all(
            r["prompt_token_ids_match"]
            and r["generation_token_ids_match"]
            and r["generation_logprobs_match"]
            for r in results
        )
        if complete
        else None,
        "all_generation_predictor_routes_match": all(
            r["generation_predictor_routes_match"] is True for r in results
        )
        if complete
        and all(r["generation_predictor_routes_match"] is not None for r in results)
        else None,
        "calls": results,
        "new_GPU_forwards": 0,
        "training_modified": False,
        "scope": "Original CPU prediction fingerprints vs canonical dump. Does not establish per-token GPU weight provenance or replay trainer weights.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lineage-proof", type=Path, required=True)
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    lineage = json.loads(args.lineage_proof.read_text())
    raw_path = Path(lineage["outlier"]["path"])
    raw = torch.load(raw_path, map_location="cpu", weights_only=False, mmap=True)

    def events():
        for path in args.trace_dir.glob("*.jsonl"):
            with path.open() as stream:
                for line in stream:
                    if '"captured_prediction_evidence"' in line:
                        yield json.loads(line)

    result = compare_predictions(raw, lineage["matching_calls"], events())
    result.update(raw_dump=str(raw_path), lineage_proof=str(args.lineage_proof))
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "calls"}))


if __name__ == "__main__":
    main()
