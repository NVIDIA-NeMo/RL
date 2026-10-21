#!/usr/bin/env python3
"""Read small saved Adam step tensors without loading optimizer moments.

Pass one --checkpoint and --expected-step pair per checkpoint. A checkpoint may
be a NeMo RL step directory or the policy/optimizer/optim DCP directory itself.
The input remains read-only and only the explicitly requested JSON is written.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path


def locate_optimizer_directory(path: Path) -> Path:
    candidates = [path, path / "policy" / "optimizer" / "optim", path / "optimizer" / "optim", path / "optim"]
    found = [candidate for candidate in candidates if (candidate / ".metadata").is_file()]
    if len(found) != 1:
        raise ValueError(f"Expected one DCP optimizer directory under {path}; found {found}")
    return found[0]


def optimizer_step(path: Path, parameter_name: str = "lm_head") -> dict:
    import torch
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner

    directory = locate_optimizer_directory(path.resolve(strict=True))
    reader = dcp.FileSystemReader(directory)
    metadata = reader.read_metadata()
    keys = sorted(key for key in metadata.state_dict_metadata if parameter_name in key and key.endswith(".step"))
    if not keys:
        raise ValueError(f"No saved Adam step tensor matching {parameter_name!r}")
    state = {}
    for key in keys:
        spec = metadata.state_dict_metadata[key]
        if not hasattr(spec, "size") or math.prod(spec.size) != 1:
            raise ValueError(f"Expected a scalar Adam step tensor: {key}")
        state[key] = torch.empty(spec.size, dtype=spec.properties.dtype)
    dcp.load(state, storage_reader=reader, planner=DefaultLoadPlanner(allow_partial_load=True))
    values = {key: value.item() for key, value in state.items()}
    if len(set(values.values())) != 1:
        raise ValueError(f"Saved Adam step tensors disagree: {values}")
    return {"checkpoint": str(path), "optimizer_directory": str(directory), "value": next(iter(values.values())),
            "parameter_name_filter": parameter_name, "step_tensors": values,
            "payload_elements_read": sum(value.numel() for value in state.values())}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--expected-step", type=int, action="append", required=True)
    parser.add_argument("--parameter-name", default="lm_head")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.checkpoint) != len(args.expected_step):
        parser.error("Supply one --expected-step per --checkpoint in the same order")
    if not args.parameter_name or any(step < 0 for step in args.expected_step):
        parser.error("Parameter filter must be nonempty and expected steps nonnegative")
    output = args.output.resolve()
    if any(path.resolve() == output or path.resolve() in output.parents for path in args.checkpoint):
        raise ValueError("Write the audit outside the read-only checkpoint directories")
    report = {"created_at_utc": datetime.now(timezone.utc).isoformat(), "status": "passed",
              "scope": "Only selected Adam step tensors; optimizer moments and all other state are not read", "records": []}
    try:
        for path, expected in zip(args.checkpoint, args.expected_step):
            record = optimizer_step(path, args.parameter_name)
            record["expected_step"] = expected
            record["passed"] = record["value"] == expected
            report["records"].append(record)
        if not all(record["passed"] for record in report["records"]):
            report["status"] = "failed"
    except Exception as error:
        report.update(status="error", error=str(error))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(descriptor, "w") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    args.output.chmod(0o600)
    print(json.dumps(report), flush=True)
    if report["status"] != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
