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
"""Golden values: check a training path against a Megatron-Core reference run.

A golden file records, for a fixed token batch and a few optimizer steps, the
loss and grad norm of each step and a fingerprint of every weight before the
first step and after each one. ``tools/golden_values/record_mcore_goldens.py``
writes it from Megatron-Bridge and Megatron-Core alone;
``tools/golden_values/check_nemo_rl_goldens.py`` replays the batch through
NeMo-RL's Megatron policy and compares.

Fingerprints are taken on full tensors under their HuggingFace names (the
Megatron-Bridge export), so a file recorded at one parallel layout can be
checked against a run at another. Only pure-torch code lives here; both sides
import it.
"""

import gzip
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional

import torch

GOLDEN_FORMAT_VERSION = 1
# Signed random projections per tensor. One 32-bit hash per element supplies
# all of them, 4 bits each.
NUM_PROJECTIONS = 8
_CHUNK_NUMEL = 1 << 22

# Defaults for comparing a check run with its golden file. Layout changes reorder
# bf16 reductions, so these sit well above that noise; a missing gradient
# reduction moves the grad norm and the weight updates by far more.
# weight_update is the size-weighted error over all tensors in a step;
# weight_update_per_tensor bounds each tensor, which is noisier (an MoE expert's
# gradient moves when bf16 noise flips a near-tie in routing).
# initial_weights compares each tensor before the first step: both runs load the
# same checkpoint, so anything above float rounding of the fingerprint means the
# check run started from a different model.
DEFAULT_TOLERANCES = {
    "initial_weights": 1e-6,
    "loss": 5e-3,
    "grad_norm": 5e-2,
    "weight_update": 5e-2,
    "weight_update_per_tensor": 0.5,
}


def synthetic_token_batch(
    *,
    num_rows: int,
    seq_length: int,
    seed: int,
    token_low: int,
    token_high: int,
    excluded_token_ids: Iterable[int] = (),
) -> torch.Tensor:
    """Seeded [num_rows, seq_length] int64 token ids in [token_low, token_high).

    Excluded ids (for example media placeholders) are remapped to token_low so
    that every row is plain text.
    """
    excluded_token_ids = set(excluded_token_ids)
    if not 0 <= token_low < token_high:
        raise ValueError(f"Bad token range [{token_low}, {token_high})")
    if token_low in excluded_token_ids:
        raise ValueError(f"token_low {token_low} is itself an excluded token id")
    generator = torch.Generator().manual_seed(seed)
    tokens = torch.randint(
        token_low, token_high, (num_rows, seq_length), generator=generator
    )
    for token_id in excluded_token_ids:
        tokens[tokens == token_id] = token_low
    return tokens


def _projection_weights(start: int, count: int, device: torch.device) -> torch.Tensor:
    """[NUM_PROJECTIONS, count] pseudo-random weights in [-0.5, 0.5] for flat indices.

    Integer hashing instead of torch's RNG, so the weights are identical on every
    device and torch version. Values stay below 2**63 throughout.
    """
    index = torch.arange(start, start + count, device=device, dtype=torch.int64)
    h = (index * 0x9E3779B1 + 0x85EBCA77) & 0xFFFFFFFF
    h = h ^ (h >> 15)
    h = (h * 0x2C1B3C6D) & 0xFFFFFFFF
    h = h ^ (h >> 12)
    shifts = torch.arange(0, 4 * NUM_PROJECTIONS, 4, device=device).unsqueeze(1)
    return ((h.unsqueeze(0) >> shifts) & 15).to(torch.float64) / 15.0 - 0.5


def tensor_fingerprint(tensor: torch.Tensor) -> dict[str, Any]:
    """Sum, sum of squares and NUM_PROJECTIONS signed projections, in float64.

    Sum and projections are linear, so the fingerprint of a weight update is the
    difference of two fingerprints. The projections are signed: unlike a norm,
    they move when an update flips sign, as Adam's first step does when it sees
    the wrong gradient.
    """
    flat = tensor.detach().reshape(-1)
    device = flat.device
    total = torch.zeros((), dtype=torch.float64, device=device)
    squares = torch.zeros((), dtype=torch.float64, device=device)
    projections = torch.zeros(NUM_PROJECTIONS, dtype=torch.float64, device=device)
    for start in range(0, flat.numel(), _CHUNK_NUMEL):
        part = flat[start : start + _CHUNK_NUMEL].to(torch.float64)
        total += part.sum()
        squares += part.square().sum()
        projections += _projection_weights(start, part.numel(), device) @ part
    return {
        "shape": list(tensor.shape),
        "sum": total.item(),
        "sumsq": squares.item(),
        "proj": projections.tolist(),
    }


def fingerprint_named_tensors(
    named_tensors: Iterable[tuple[str, torch.Tensor]],
) -> dict[str, dict[str, Any]]:
    fingerprints = {}
    for name, tensor in named_tensors:
        if name in fingerprints:
            raise ValueError(f"Duplicate exported weight name {name!r}")
        fingerprints[name] = tensor_fingerprint(tensor)
    return fingerprints


def scalar_config_fields(config: Any) -> dict[str, Any]:
    """JSON-friendly scalar fields of a model config, to diff two runs' settings."""
    fields = {}
    for name, value in sorted(vars(config).items()):
        if name.startswith("_"):
            continue
        if isinstance(value, torch.dtype):
            fields[name] = str(value)
        elif value is None or isinstance(value, (bool, int, float, str)):
            fields[name] = value
    return fields


def diff_config_fields(
    expected: dict[str, Any], actual: dict[str, Any]
) -> dict[str, tuple[Any, Any]]:
    """Fields present in both with different values: {name: (expected, actual)}."""
    return {
        name: (expected[name], actual[name])
        for name in sorted(expected.keys() & actual.keys())
        if expected[name] != actual[name]
    }


def write_golden_file(path: str | Path, golden: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(golden, indent=1, sort_keys=True).encode()
    if path.suffix == ".gz":
        path.write_bytes(gzip.compress(payload))
    else:
        path.write_bytes(payload)


def read_golden_file(path: str | Path) -> dict[str, Any]:
    raw = Path(path).read_bytes()
    if Path(path).suffix == ".gz":
        raw = gzip.decompress(raw)
    golden = json.loads(raw)
    if golden.get("format_version") != GOLDEN_FORMAT_VERSION:
        raise ValueError(
            f"{path}: golden format {golden.get('format_version')!r}, expected "
            f"{GOLDEN_FORMAT_VERSION}"
        )
    return golden


@dataclass
class Mismatch:
    metric: str
    step: int
    name: Optional[str]
    expected: float
    actual: float
    relative_error: float

    def __str__(self) -> str:
        where = f"step {self.step}" + (f" {self.name}" if self.name else "")
        if self.metric == "initial_weights":
            values = "weights before the first step differ from the golden's"
        elif self.metric.startswith("weight_update"):
            # For weights, expected is the golden update's size and actual the
            # size of the difference between the two runs' updates.
            values = f"update size {self.expected:.6g}, difference {self.actual:.6g}"
        else:
            values = f"expected {self.expected:.6g}, got {self.actual:.6g}"
        return f"{self.metric} {where}: {values} (relative error {self.relative_error:.3g})"


def _relative_error(expected: float, actual: float) -> float:
    if expected == actual:
        return 0.0
    return abs(actual - expected) / max(abs(expected), 1e-12)


def _update_vector(
    fingerprints: list[dict[str, dict[str, Any]]], step: int, name: str
) -> list[float]:
    """Projections of the update a tensor received from step - 1 to step."""
    before, after = fingerprints[step - 1][name], fingerprints[step][name]
    return [a - b for a, b in zip(after["proj"], before["proj"])]


def compare_goldens(
    golden: dict[str, Any],
    actual: dict[str, Any],
    tolerances: Optional[dict[str, float]] = None,
) -> tuple[list[Mismatch], dict[str, float]]:
    """Compare a check run with its golden file.

    Returns the mismatches and, per metric, the largest relative error seen, so
    a passing run still shows how close it came.

    Weights before the first step must match the golden's
    (``initial_weights``), or the two runs didn't train the same model.
    Each step's weight update is compared through its signed projections, as
    the size-weighted error over all tensors (``weight_update``) and per tensor
    (``weight_update_per_tensor``). A tensor's size is floored at 1% of the
    largest update: bf16 rounding decides whether a near-ulp update shows at
    all, so a tensor that barely moved is judged on the scale of the others.
    """
    tolerances = {**DEFAULT_TOLERANCES, **(tolerances or {})}
    mismatches: list[Mismatch] = []
    worst = {metric: 0.0 for metric in DEFAULT_TOLERANCES}

    def check(metric, step, name, expected, got, error):
        worst[metric] = max(worst[metric], error)
        if error > tolerances[metric]:
            mismatches.append(Mismatch(metric, step, name, expected, got, error))

    golden_steps, actual_steps = golden["steps"], actual["steps"]
    if len(golden_steps) != len(actual_steps):
        raise ValueError(
            f"Golden file has {len(golden_steps)} steps, check run has {len(actual_steps)}"
        )
    for step, (expected_step, actual_step) in enumerate(
        zip(golden_steps, actual_steps), start=1
    ):
        for metric in ("loss", "grad_norm"):
            expected, got = expected_step[metric], actual_step[metric]
            check(metric, step, None, expected, got, _relative_error(expected, got))

    golden_weights, actual_weights = golden["weights"], actual["weights"]
    missing = set(golden_weights[0]) ^ set(actual_weights[0])
    if missing:
        raise ValueError(
            f"Exported weight names differ between golden and check run: {sorted(missing)[:10]}"
        )
    for name, fingerprint in golden_weights[0].items():
        if fingerprint["shape"] != actual_weights[0][name]["shape"]:
            raise ValueError(
                f"{name}: golden shape {fingerprint['shape']}, check run shape "
                f"{actual_weights[0][name]['shape']}"
            )
        got = actual_weights[0][name]
        projection_error = math.hypot(
            *(a - e for a, e in zip(got["proj"], fingerprint["proj"]))
        ) / max(math.hypot(*fingerprint["proj"]), 1e-30)
        check(
            "initial_weights",
            0,
            name,
            fingerprint["sumsq"],
            got["sumsq"],
            max(
                projection_error,
                _relative_error(fingerprint["sumsq"], got["sumsq"]),
            ),
        )
    for step in range(1, len(golden_weights)):
        sizes, differences = {}, {}
        for name in golden_weights[0]:
            expected = _update_vector(golden_weights, step, name)
            got = _update_vector(actual_weights, step, name)
            sizes[name] = math.hypot(*expected)
            differences[name] = math.hypot(*(a - e for a, e in zip(got, expected)))
        total_size = math.hypot(*sizes.values())
        total_difference = math.hypot(*differences.values())
        check(
            "weight_update",
            step,
            None,
            total_size,
            total_difference,
            total_difference / max(total_size, 1e-30),
        )
        largest = max(sizes.values(), default=0.0)
        for name, difference in differences.items():
            floor = max(sizes[name], 1e-2 * largest, 1e-30)
            check(
                "weight_update_per_tensor",
                step,
                name,
                sizes[name],
                difference,
                difference / floor,
            )
    return mismatches, worst
