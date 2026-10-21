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
"""Apply the pinned Flint sink fix only inside the writable task container."""

from __future__ import annotations

import argparse
import ast
import difflib
import hashlib
import json
import os
import stat
import tempfile
from datetime import datetime, timezone
from pathlib import Path

BASELINE_SHA256 = "dbe9b8eb7585cd01c28feaf3172ef886b76d90bcd95f4593f7bae08051a55593"
CANDIDATE_SHA256 = "e57d036664e889d4ac639d1aeec198c5e127d2926de00c0656ede4b233ed4a4c"
RELATIVE_SOURCE = Path("nemo_automodel/components/models/flint/layers.py")
TARGET = Path("/opt/flint/Automodel") / RELATIVE_SOURCE


def digest(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def atomic_write(path: Path, content: bytes, mode: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, mode)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    workspace = Path(__file__).resolve().parents[1]
    baseline = (workspace / "evidence/runtime-source/flint_layers.py").read_bytes()
    candidate = (workspace / "automodel-overlay" / RELATIVE_SOURCE).read_bytes()
    if digest(baseline) != BASELINE_SHA256 or digest(candidate) != CANDIDATE_SHA256:
        raise RuntimeError(
            "Captured baseline or proposed overlay differs from the reviewed hashes"
        )
    ast.parse(candidate, filename=str(TARGET))
    if TARGET.is_symlink() or TARGET.resolve() != TARGET:
        raise RuntimeError("Refusing to replace a redirected Automodel source path")
    current = TARGET.read_bytes()
    before = digest(current)
    if before not in (BASELINE_SHA256, CANDIDATE_SHA256):
        raise RuntimeError(f"Refusing unknown Automodel source hash: {before}")
    status = "already_applied" if before == CANDIDATE_SHA256 else "baseline_verified"
    if not args.check_only and before == BASELINE_SHA256:
        atomic_write(TARGET, candidate, stat.S_IMODE(TARGET.stat().st_mode))
        status = "applied"
    after = digest(TARGET.read_bytes())
    if not args.check_only and after != CANDIDATE_SHA256:
        raise RuntimeError(
            "Automodel source did not match the overlay after replacement"
        )
    patch = "".join(
        difflib.unified_diff(
            baseline.decode().splitlines(keepends=True),
            candidate.decode().splitlines(keepends=True),
            fromfile=f"a/{RELATIVE_SOURCE}",
            tofile=f"b/{RELATIVE_SOURCE}",
        )
    )
    evidence = workspace / "evidence"
    atomic_write(evidence / "automodel-sink-checkpoint.patch", patch.encode(), 0o600)
    receipt = {
        "schema_version": 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "check_only": args.check_only,
        "target": str(TARGET),
        "baseline_sha256": BASELINE_SHA256,
        "candidate_sha256": CANDIDATE_SHA256,
        "before_sha256": before,
        "after_sha256": after,
        "scope": "Flint sink call input only; original model weights and Torch source are unchanged",
    }
    atomic_write(
        evidence / "runtime-overlay.json",
        (json.dumps(receipt, indent=2) + "\n").encode(),
        0o600,
    )
    print(
        json.dumps(
            {"status": status, "after_sha256": after, "check_only": args.check_only}
        )
    )


if __name__ == "__main__":
    main()
