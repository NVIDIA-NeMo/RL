#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compare the first resumed Flint refit with an informative saved head sample.

Read-only inputs, standard library only. Exactly one serving-replica receipt is
expected. Only its first nonblank JSONL record is checked: later records can
legitimately contain another training update. The comparison establishes exact
FP32 bytes for rows 1:33, columns 0:32, name-count coverage, resident head dtype,
and all 49 sink names/dtypes. It does not establish full-model value equality.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct


EXPECTED = {"replicas": 1, "loaded_hf_name_count": 36734, "sinks": 49,
            "head_dtype": "torch.float32", "sink_dtype": "torch.bfloat16",
            "sample_row_start": 1, "sample_shape": [32, 32], "sample_dtype": "torch.float32"}


def read_object(path: Path) -> dict:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def valid_hash(value) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


def first_record(path: Path) -> tuple[dict, int, str]:
    with path.open() as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected a JSON object at {path}:{number}")
            return value, number, hashlib.sha256(line.encode()).hexdigest()
    raise ValueError(f"Missing first refit record: {path} is empty")


def compare(head_sample: Path, refit_dir: Path) -> dict:
    errors = []
    report = {"schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "status": "fail", "head_sample_evidence": str(head_sample), "refit_directory": str(refit_dir),
              "expected": EXPECTED, "replicas": [], "errors": errors,
              "scope": "First resumed refit only: exact saved FP32 head sample at rows 1:33, columns 0:32; full49 tensor-name count and resident head/sink dtype checks. Full-model and sink value equality are not verified.",
              "full_model_value_equality_verified": False, "sink_value_equality_verified": False,
              "later_refit_records_checked": False}
    try:
        saved = read_object(head_sample)
        report["head_sample_evidence_sha256"] = hashlib.sha256(head_sample.read_bytes()).hexdigest()
        sample = saved.get("informative_sample")
        if saved.get("status") != "pass" or not isinstance(sample, dict):
            raise ValueError("Head evidence must be a passing changed-head report with an informative_sample")
        source_hash, checkpoint_hash = sample.get("source_fp32_sha256"), sample.get("checkpoint_fp32_sha256")
        if not valid_hash(source_hash) or not valid_hash(checkpoint_hash) or source_hash == checkpoint_hash:
            raise ValueError("Saved informative sample must have distinct valid source and learned-checkpoint hashes")
        if (sample.get("row_start"), sample.get("row_stop"), sample.get("column_start"), sample.get("column_stop"),
                sample.get("shape"), sample.get("dtype")) != (1, 33, 0, 32, [32, 32], "float32"):
            raise ValueError("Expected saved informative FP32 sample at rows 1:33, columns 0:32")
        changed = sample.get("changed_elements")
        if isinstance(changed, bool) or not isinstance(changed, int) or changed <= 0:
            raise ValueError("Saved informative sample must contain changed elements")
        report["saved_checkpoint"] = saved.get("checkpoint")
        report["source_fp32_sha256"] = source_hash
        report["checkpoint_fp32_sha256"] = checkpoint_hash
        report["sample_changed_elements"] = changed
        if not refit_dir.is_dir():
            raise ValueError(f"Missing refit directory: {refit_dir}")
        files = sorted(refit_dir.glob("flint-refit*.jsonl"))
        report["observed_replica_receipt_files"] = len(files)
        if len(files) != EXPECTED["replicas"]:
            errors.append(f"Expected exactly one serving-replica PID receipt file, found {len(files)}")
        expected_sink_names = {f"model.layers.{layer}.self_attn.softmax_offset" for layer in range(49)}
        for path in files:
            replica = {"evidence": str(path), "first_record_only": True, "errors": []}
            report["replicas"].append(replica)
            try:
                value, line_number, line_hash = first_record(path)
                replica.update(record=1, line_number=line_number, first_record_sha256=line_hash,
                               hostname=value.get("hostname"), pid=value.get("pid"), timestamp_utc=value.get("timestamp_utc"),
                               loaded_hf_name_count=value.get("loaded_hf_name_count"))
                if value.get("loaded_hf_name_count") != EXPECTED["loaded_hf_name_count"]:
                    replica["errors"].append("First refit did not report all 36,734 HF tensor names")
                hostname, pid = value.get("hostname"), value.get("pid")
                if (not isinstance(hostname, str) or not hostname or isinstance(pid, bool) or not isinstance(pid, int)
                        or pid <= 0 or path.name != f"flint-refit-{hostname}-{pid}.jsonl"):
                    replica["errors"].append("Receipt filename does not match its valid hostname/PID identity")
                head = value.get("head")
                if not isinstance(head, dict):
                    raise ValueError("Missing resident head receipt")
                replica["head"] = {key: head.get(key) for key in ("name", "dtype", "shape", "sample_dtype", "sample_row_start", "sample_shape", "sample_sha256")}
                for key, expected in (("name", "lm_head.weight"), ("dtype", EXPECTED["head_dtype"]),
                                      ("shape", saved.get("head_shape")), ("sample_dtype", EXPECTED["sample_dtype"]),
                                      ("sample_row_start", 1), ("sample_shape", [32, 32])):
                    if head.get(key) != expected:
                        replica["errors"].append(f"Head {key}: expected {expected!r}, found {head.get(key)!r}")
                serving_hash = head.get("sample_sha256")
                replica["matches_saved_checkpoint_sample"] = serving_hash == checkpoint_hash
                replica["differs_from_source_sample"] = valid_hash(serving_hash) and serving_hash != source_hash
                if not replica["matches_saved_checkpoint_sample"] or not replica["differs_from_source_sample"]:
                    replica["errors"].append("First refit head sample does not match the learned checkpoint distinctly from the original source")
                # Recompute the declared sample hash from the actual finite JSON
                # sample values; the pinned ARM serving runtime is little-endian.
                values = head.get("sample_values")
                if not isinstance(values, list) or len(values) != 32 or any(not isinstance(row, list) or len(row) != 32 for row in values):
                    raise ValueError("Expected the actual 32 by 32 head sample values")
                flattened = [element for row in values for element in row]
                if any(isinstance(element, bool) or not isinstance(element, (int, float)) or not math.isfinite(element) for element in flattened):
                    raise ValueError("Head sample values must be finite numbers")
                recomputed = hashlib.sha256(struct.pack("<1024f", *flattened)).hexdigest()
                replica["recomputed_head_sample_sha256"] = recomputed
                if recomputed != serving_hash:
                    replica["errors"].append("Actual JSON sample values do not reproduce the declared FP32 head sample hash")
                sinks = value.get("sinks")
                if not isinstance(sinks, list) or any(not isinstance(sink, dict) for sink in sinks):
                    raise ValueError("Missing resident sink tensor receipts")
                names = [sink.get("name") for sink in sinks]
                replica["sink_count"] = len(sinks)
                replica["sink_dtype_counts"] = {dtype: sum(sink.get("dtype") == dtype for sink in sinks)
                                                for dtype in sorted({str(sink.get("dtype")) for sink in sinks})}
                if len(sinks) != 49 or set(names) != expected_sink_names:
                    replica["errors"].append("Expected exactly the 49 unique full-model attention-sink tensor names")
                if any(sink.get("dtype") != EXPECTED["sink_dtype"] for sink in sinks):
                    replica["errors"].append("Every resident FA4 serving sink must be torch.bfloat16")
            except Exception as error:
                replica["errors"].append(str(error))
            replica["status"] = "pass" if not replica["errors"] else "fail"
            errors.extend(f"{path.name}: {error}" for error in replica["errors"])
        report["all_expected_replicas_matched"] = len(files) == 1 and bool(report["replicas"]) and all(item["status"] == "pass" for item in report["replicas"])
    except Exception as error:
        errors.append(str(error))
        report["all_expected_replicas_matched"] = False
    report["status"] = "pass" if not errors and report["all_expected_replicas_matched"] else "fail"
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--head-sample", type=Path, required=True)
    parser.add_argument("--refit-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output == args.head_sample.resolve() or output == args.refit_dir.resolve() or args.refit_dir.resolve() in output.parents:
        raise ValueError("Write comparison output outside the read-only refit directory and head evidence file")
    report = compare(args.head_sample.resolve(), args.refit_dir.resolve())
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(output, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    output.chmod(0o600)
    print(json.dumps({"status": report["status"], "replicas": report.get("observed_replica_receipt_files"),
                      "all_expected_replicas_matched": report["all_expected_replicas_matched"], "errors": report["errors"],
                      "report": str(output)}), flush=True)
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
