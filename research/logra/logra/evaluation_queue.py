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


"""Request independent checkpoint evaluations without changing the trainer.

The external evaluator owns its protocol file and model-export implementation.
A request identifies a saved policy and pins that protocol by content hash; it
does not report an accuracy or claim the evaluation has completed.
"""

import hashlib
import json
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class EvaluationRequest:
    step: int
    checkpoint: str
    protocol_file: str
    protocol_sha256: str
    requesting_run: str


def protocol_digest(root: Path) -> str:
    """Pin the external evaluator's opaque JSON protocol descriptor."""
    data = (root / "protocol.json").read_bytes()
    json.loads(data)
    return hashlib.sha256(data).hexdigest()


def enqueue_checkpoint(
    root: Path,
    checkpoint: Path,
    *,
    step: int,
    requesting_run: Path,
    resume: Path | None,
) -> Path:
    """Atomically request a finalized checkpoint; never overwrite another point."""
    # At the initial resume boundary, weights still live in the previous run.
    # Later missing checkpoints must fail rather than reuse an older policy.
    if not checkpoint.exists() and resume is not None:
        restored = resume.resolve()
        if restored.name == f"step_{step}":
            checkpoint = restored
    checkpoint = checkpoint.resolve()
    if step <= 0 or checkpoint.name != f"step_{step}":
        raise ValueError(
            "External evaluation requires a finalized positive-step checkpoint"
        )
    info = json.loads((checkpoint / "training_info.json").read_text())
    if info["step"] != step or not (checkpoint / "cua.pt").is_file():
        raise ValueError("Checkpoint step or CUA state is incomplete")
    model = checkpoint / "policy/weights/model"
    if not model.is_dir() or not any(model.glob("*.safetensors")):
        raise ValueError("Checkpoint has no saved policy weights")
    request = EvaluationRequest(
        step=step,
        checkpoint=str(checkpoint),
        protocol_file=str((root / "protocol.json").resolve()),
        protocol_sha256=protocol_digest(root),
        requesting_run=str(requesting_run.resolve()),
    )
    pending = root / "requests"
    pending.mkdir(parents=True, exist_ok=True)
    destination = pending / f"step-{step:06d}.json"
    with tempfile.NamedTemporaryFile(mode="w", dir=pending, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(asdict(request), stream, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    try:
        try:
            # link publishes a complete file and fails rather than clobbering
            # an existing request if a resumed driver reports the same step.
            os.link(temporary, destination)
        except FileExistsError:
            previous = json.loads(destination.read_text())
            if any(
                previous[key] != asdict(request)[key]
                for key in ("step", "checkpoint", "protocol_file", "protocol_sha256")
            ):
                raise ValueError(f"Conflicting evaluation request: {destination}")
    finally:
        temporary.unlink()
    return destination
