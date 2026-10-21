#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
"""Verify a finalized model/optimizer checkpoint before another batch window."""

import argparse
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path

TARGET_STEP = 50


def latest_checkpoint(root):
    checkpoints = root / "results/dapo50"
    paths = [
        path
        for path in checkpoints.glob("step_*")
        if re.fullmatch(r"step_[0-9]+", path.name)
    ]
    if not paths:
        return {"step": 0, "path": None}
    latest = max(paths, key=lambda path: int(path.name.split("_")[1]))
    if latest.is_symlink() or not latest.is_dir():
        raise RuntimeError("Refusing a redirected or non-directory checkpoint")
    step = int(latest.name.split("_")[1])
    if not 1 <= step <= TARGET_STEP:
        raise RuntimeError(f"Unexpected finalized checkpoint step: {step}")
    for relative in [
        "config.yaml",
        "training_info.json",
        "policy/optimizer/optim/.metadata",
    ]:
        if not (latest / relative).is_file():
            raise RuntimeError(f"Finalized checkpoint is missing {relative}")
    info = json.loads((latest / "training_info.json").read_text())
    if info.get("total_steps") != step:
        raise RuntimeError("Checkpoint directory and saved training step disagree")
    if not any((latest / "policy/weights/model").glob("*.safetensors")):
        raise RuntimeError("Finalized checkpoint has no raw model weights")
    return {"step": step, "path": str(latest)}


def finish_window(root, starting, log_path):
    checkpoint = latest_checkpoint(root)
    if checkpoint["step"] == TARGET_STEP:
        disposition = "complete"
    else:
        if checkpoint["step"] <= starting["step"]:
            raise RuntimeError(
                "A clean window did not advance its finalized checkpoint"
            )
        if (
            "Timeout has been reached, stopping training early"
            not in log_path.read_text()
        ):
            raise RuntimeError(
                "Early success was not the configured graceful checkpoint timeout"
            )
        disposition = "continue"
    return {
        "status": "pass",
        "disposition": disposition,
        "starting_step": starting["step"],
        "checkpoint_step": checkpoint["step"],
        "checkpoint_path": checkpoint["path"],
        "target_step": TARGET_STEP,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Atomic finalized checkpoint directory and model/optimizer files; full payload audit runs at step 50",
    }


def write_json(path, record):
    descriptor, temporary = tempfile.mkstemp(prefix=".window-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(record, stream, indent=2)
            stream.write("\n")
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["start", "finish"])
    parser.add_argument("--run-root", type=Path, default=Path("/flint-run"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--starting", type=Path)
    parser.add_argument("--log", type=Path)
    args = parser.parse_args()
    if args.mode == "start":
        record = latest_checkpoint(args.run_root)
        minimum = int(os.environ.get("FLINT_EXPECTED_RESUME_STEP", "0"))
        if record["step"] < minimum:
            raise RuntimeError(
                "Expected continuation checkpoint is absent or older than the parent window"
            )
        record.update(
            status="pass", timestamp_utc=datetime.now(timezone.utc).isoformat()
        )
        write_json(args.output, record)
        print(record["step"])
    else:
        if args.starting is None or args.log is None:
            parser.error("finish requires --starting and --log")
        record = finish_window(
            args.run_root, json.loads(args.starting.read_text()), args.log
        )
        write_json(args.output, record)
        print(record["disposition"])


if __name__ == "__main__":
    main()
