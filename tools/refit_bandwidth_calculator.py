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

"""Estimate when measured sparse refit beats NCCL over Ethernet.

This is a benchmark-specific estimator. Sparse latency is fitted from the July
2026 S3 and ZeroMQ benchmarks, and NCCL latency is interpolated from measured
H100 reshard results on 400 Gbps/rank InfiniBand using hierarchical API.
``--candidate-ethernet-gbps`` projects that NCCL reference onto a raw
per-rank Ethernet rate.
"""

import argparse
import json
import math
from dataclasses import asdict, dataclass
from itertools import pairwise
from typing import Literal

Transport = Literal["s3", "zmq"]
Compression = Literal["raw", "zstd"]

_REFERENCE_IB_GBPS = 400.0
_CALIBRATED_DENSITIES = (3.0, 5.0)
_CALIBRATED_MODEL_SIZE_GB = (63.2, 1121.0)


@dataclass(frozen=True)
class _SparseFit:
    intercept_s: float
    seconds_per_tb: float


# T(S) = intercept_s + seconds_per_tb * S for decimal TB of indexed BF16.
_SPARSE_FITS: dict[tuple[Transport, Compression, float], _SparseFit] = {
    ("s3", "raw", 3.0): _SparseFit(2.370, 116.138),
    ("s3", "zstd", 3.0): _SparseFit(0.000, 84.746),
    ("s3", "raw", 5.0): _SparseFit(2.646, 183.431),
    ("s3", "zstd", 5.0): _SparseFit(0.000, 134.041),
    ("zmq", "raw", 3.0): _SparseFit(0.000, 264.753),
    ("zmq", "zstd", 3.0): _SparseFit(6.517, 73.621),
    ("zmq", "raw", 5.0): _SparseFit(0.000, 428.008),
    ("zmq", "zstd", 5.0): _SparseFit(8.165, 157.339),
}

# (indexed model GB, low seconds, high seconds), with generation EP enabled.
_NCCL_ANCHORS = (
    (63.2, 0.84, 1.60),
    (247.2, 1.46, 1.74),
    (470.2, 2.31, 2.73),
    (1342.0, 3.27, 3.46),
)

# Approximate serialized bytes / changed BF16 bytes.
_WIRE_MULTIPLIER: dict[Compression, float] = {"raw": 2.0, "zstd": 0.74}


@dataclass(frozen=True)
class Estimate:
    """One sparse transport compared with the NCCL reference."""

    transport: Transport
    compression: Compression
    model_size_gb: float
    changed_pct: float
    sparse_seconds: float
    approximate_wire_gb: float
    nccl_ib_low_s: float
    nccl_ib_high_s: float
    candidate_ethernet_gbps: float | None
    nccl_ethernet_low_s: float | None
    nccl_ethernet_high_s: float | None
    break_even_ethernet_low_gbps: float
    break_even_ethernet_high_gbps: float
    candidate_winner: str | None
    model_size_extrapolated: bool
    density_extrapolated: bool


def predict_sparse_seconds(
    model_size_gb: float,
    changed_pct: float,
    *,
    transport: Transport,
    compression: Compression,
) -> float:
    """Interpolate or extrapolate sparse latency from the 3% and 5% fits."""
    if model_size_gb <= 0 or changed_pct <= 0:
        raise ValueError("model_size_gb and changed_pct must be positive")

    size_tb = model_size_gb / 1000.0
    density_low, density_high = _CALIBRATED_DENSITIES
    latency_low = _fit_latency(size_tb, transport, compression, density_low)
    latency_high = _fit_latency(size_tb, transport, compression, density_high)
    exponent = math.log(latency_high / latency_low) / math.log(
        density_high / density_low
    )
    return latency_low * (changed_pct / density_low) ** exponent


def _fit_latency(
    size_tb: float,
    transport: Transport,
    compression: Compression,
    changed_pct: float,
) -> float:
    fit = _SPARSE_FITS[(transport, compression, changed_pct)]
    return fit.intercept_s + fit.seconds_per_tb * size_tb


def _nccl_reference(model_size_gb: float) -> tuple[float, float, bool]:
    if model_size_gb <= 0:
        raise ValueError("model_size_gb must be positive")

    extrapolated = not _NCCL_ANCHORS[0][0] <= model_size_gb <= _NCCL_ANCHORS[-1][0]
    if model_size_gb <= _NCCL_ANCHORS[0][0]:
        left, right = _NCCL_ANCHORS[:2]
    elif model_size_gb >= _NCCL_ANCHORS[-1][0]:
        left, right = _NCCL_ANCHORS[-2:]
    else:
        left, right = _NCCL_ANCHORS[:2]
        for candidate_left, candidate_right in pairwise(_NCCL_ANCHORS):
            if candidate_left[0] <= model_size_gb <= candidate_right[0]:
                left, right = candidate_left, candidate_right
                break

    position = math.log(model_size_gb / left[0]) / math.log(right[0] / left[0])
    low_s = left[1] + position * (right[1] - left[1])
    high_s = left[2] + position * (right[2] - left[2])
    return max(0.001, low_s), max(0.001, high_s), extrapolated


def estimate(
    *,
    model_size_gb: float,
    changed_pct: float,
    transport: Transport,
    compression: Compression = "zstd",
    candidate_ethernet_gbps: float | None = None,
) -> Estimate:
    """Estimate sparse latency and the NCCL-over-Ethernet crossover."""
    if candidate_ethernet_gbps is not None and candidate_ethernet_gbps <= 0:
        raise ValueError("candidate_ethernet_gbps must be positive")

    sparse_seconds = predict_sparse_seconds(
        model_size_gb,
        changed_pct,
        transport=transport,
        compression=compression,
    )
    nccl_low_s, nccl_high_s, model_size_extrapolated = _nccl_reference(model_size_gb)
    break_even_low = _REFERENCE_IB_GBPS * nccl_low_s / sparse_seconds
    break_even_high = _REFERENCE_IB_GBPS * nccl_high_s / sparse_seconds

    candidate_low = candidate_high = None
    winner = None
    if candidate_ethernet_gbps is not None:
        scale = _REFERENCE_IB_GBPS / candidate_ethernet_gbps
        candidate_low = nccl_low_s * scale
        candidate_high = nccl_high_s * scale
        if sparse_seconds < candidate_low and not math.isclose(
            sparse_seconds, candidate_low
        ):
            winner = transport
        elif sparse_seconds > candidate_high and not math.isclose(
            sparse_seconds, candidate_high
        ):
            winner = "nccl"
        else:
            winner = "depends"

    return Estimate(
        transport=transport,
        compression=compression,
        model_size_gb=model_size_gb,
        changed_pct=changed_pct,
        sparse_seconds=sparse_seconds,
        approximate_wire_gb=(
            model_size_gb * changed_pct / 100.0 * _WIRE_MULTIPLIER[compression]
        ),
        nccl_ib_low_s=nccl_low_s,
        nccl_ib_high_s=nccl_high_s,
        candidate_ethernet_gbps=candidate_ethernet_gbps,
        nccl_ethernet_low_s=candidate_low,
        nccl_ethernet_high_s=candidate_high,
        break_even_ethernet_low_gbps=break_even_low,
        break_even_ethernet_high_gbps=break_even_high,
        candidate_winner=winner,
        model_size_extrapolated=(
            model_size_extrapolated
            or not _CALIBRATED_MODEL_SIZE_GB[0]
            <= model_size_gb
            <= _CALIBRATED_MODEL_SIZE_GB[1]
        ),
        density_extrapolated=(
            not _CALIBRATED_DENSITIES[0] <= changed_pct <= _CALIBRATED_DENSITIES[1]
        ),
    )


def _positive_float(value: str) -> float:
    number = float(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-size-gb", type=_positive_float, required=True)
    parser.add_argument(
        "--changed-pct",
        "--sparsity-pct",
        type=_positive_float,
        required=True,
        help="Any positive percentage of changed weight elements.",
    )
    parser.add_argument("--transport", choices=("all", "s3", "zmq"), default="all")
    parser.add_argument("--compression", choices=("raw", "zstd"), default="zstd")
    parser.add_argument(
        "--candidate-ethernet-gbps",
        type=_positive_float,
        help="Raw per-rank Ethernet rate used by the projected NCCL refit.",
    )
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def _seconds_range(low: float, high: float) -> str:
    return f"{low:.3f}-{high:.3f}s"


def _print_results(results: list[Estimate]) -> None:
    first = results[0]
    print(
        f"Model: {first.model_size_gb:g} GB indexed BF16; "
        f"changed: {first.changed_pct:g}%; compression: {first.compression}"
    )
    print(
        "Measured NCCL on 400 Gbps/rank IB: "
        f"{_seconds_range(first.nccl_ib_low_s, first.nccl_ib_high_s)}"
    )

    if first.candidate_ethernet_gbps is not None:
        assert first.nccl_ethernet_low_s is not None
        assert first.nccl_ethernet_high_s is not None
        print(
            f"Projected NCCL on {first.candidate_ethernet_gbps:g} Gbps/rank "
            f"Ethernet: {_seconds_range(first.nccl_ethernet_low_s, first.nccl_ethernet_high_s)}"
        )

    print()
    winner_header = "Winner@candidate" if first.candidate_ethernet_gbps else ""
    print(
        f"{'Path':<6} {'Sparse':>10} {'Wire':>10} "
        f"{'Ethernet crossover':>26} {winner_header:>18}"
    )
    for result in results:
        crossover = (
            f"{result.break_even_ethernet_low_gbps:.2f}-"
            f"{result.break_even_ethernet_high_gbps:.2f} Gbps/rank"
        )
        winner = result.candidate_winner or ""
        print(
            f"{result.transport.upper():<6} {result.sparse_seconds:>9.3f}s "
            f"{result.approximate_wire_gb:>8.3f} GB {crossover:>26} {winner:>18}"
        )

    print()
    print(
        "Below the lower crossover, sparse refit beats the full NCCL envelope; "
        "above the upper crossover, NCCL wins."
    )
    print(
        "Projection: T_ethernet = T_H100_IB * 400 / candidate_gbps, with all "
        "bandwidth values expressed per rank."
    )
    if first.model_size_extrapolated:
        print("Note: model size is outside the measured calibration range.")
    if first.density_extrapolated:
        print("Note: changed density is extrapolated from measured 3% and 5% arms.")


def main() -> None:
    """Run the estimator."""
    args = parse_args()
    transports: tuple[Transport, ...] = (
        ("s3", "zmq") if args.transport == "all" else (args.transport,)
    )
    results = [
        estimate(
            model_size_gb=args.model_size_gb,
            changed_pct=args.changed_pct,
            transport=transport,
            compression=args.compression,
            candidate_ethernet_gbps=args.candidate_ethernet_gbps,
        )
        for transport in transports
    ]
    if args.json:
        print(json.dumps([asdict(result) for result in results], indent=2))
    else:
        _print_results(results)


if __name__ == "__main__":
    main()
