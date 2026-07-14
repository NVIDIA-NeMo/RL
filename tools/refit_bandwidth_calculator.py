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

"""Compare measured zstd sparse refit with NCCL projected to Ethernet."""

import argparse
import json
import math
from bisect import bisect_right
from dataclasses import asdict, dataclass
from typing import Literal

Transport = Literal["s3", "zmq"]

_REFERENCE_IB_GBPS = 400.0
_DENSITIES = (3.0, 5.0)
_SPARSE_SIZE_RANGE_GB = (63.2, 1121.0)
# Each pair is (fixed seconds, seconds per 1,000 GB) at 3% and 5% changed.
_SPARSE_LATENCY_FITS: dict[
    Transport, tuple[tuple[float, float], tuple[float, float]]
] = {
    "s3": ((2.271898, 90.539029), (9.132526, 164.284601)),
    "zmq": ((5.290113, 78.016171), (10.586558, 124.416925)),
}
# Unique compressed wire GB per 1,000 GB of indexed BF16 weights at 3% and 5%.
_SPARSE_WIRE_GB_PER_TB = (22.430807, 37.362273)
_NCCL_ANCHORS = (
    (63.2, 0.84, 1.60),
    (247.2, 1.46, 1.74),
    (470.2, 2.31, 2.73),
    (1342.0, 3.27, 3.46),
)


@dataclass(frozen=True)
class Estimate:
    transport: Transport
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


def predict_sparse_seconds(
    model_size_gb: float,
    changed_pct: float,
    *,
    transport: Transport,
) -> float:
    """Evaluate the campaign fit at any positive changed density."""
    if model_size_gb <= 0 or changed_pct <= 0:
        raise ValueError("model_size_gb and changed_pct must be positive")
    size_tb = model_size_gb / 1000.0
    (a3, b3), (a5, b5) = _SPARSE_LATENCY_FITS[transport]
    latency_3, latency_5 = a3 + b3 * size_tb, a5 + b5 * size_tb
    exponent = math.log(latency_5 / latency_3) / math.log(5.0 / 3.0)
    return latency_3 * (changed_pct / 3.0) ** exponent


def predict_sparse_wire_gb(model_size_gb: float, changed_pct: float) -> float:
    """Scale the measured zstd wire fit to model size and changed density."""
    if model_size_gb <= 0 or changed_pct <= 0:
        raise ValueError("model_size_gb and changed_pct must be positive")
    wire_3, wire_5 = _SPARSE_WIRE_GB_PER_TB
    exponent = math.log(wire_5 / wire_3) / math.log(5.0 / 3.0)
    return model_size_gb / 1000.0 * wire_3 * (changed_pct / 3.0) ** exponent


def _nccl_reference(model_size_gb: float) -> tuple[float, float]:
    sizes = tuple(anchor[0] for anchor in _NCCL_ANCHORS)
    index = min(max(bisect_right(sizes, model_size_gb) - 1, 0), len(sizes) - 2)
    left, right = _NCCL_ANCHORS[index : index + 2]
    position = math.log(model_size_gb / left[0]) / math.log(right[0] / left[0])
    low = left[1] + position * (right[1] - left[1])
    high = left[2] + position * (right[2] - left[2])
    return max(0.001, low), max(0.001, high)


def estimate(
    *,
    model_size_gb: float,
    changed_pct: float,
    transport: Transport,
    candidate_ethernet_gbps: float | None = None,
) -> Estimate:
    """Estimate sparse latency and the NCCL-over-Ethernet crossover."""
    if candidate_ethernet_gbps is not None and candidate_ethernet_gbps <= 0:
        raise ValueError("candidate_ethernet_gbps must be positive")
    sparse_seconds = predict_sparse_seconds(
        model_size_gb,
        changed_pct,
        transport=transport,
    )
    nccl_low, nccl_high = _nccl_reference(model_size_gb)
    crossover_low = _REFERENCE_IB_GBPS * nccl_low / sparse_seconds
    crossover_high = _REFERENCE_IB_GBPS * nccl_high / sparse_seconds

    projected_low = projected_high = None
    winner = None
    if candidate_ethernet_gbps is not None:
        scale = _REFERENCE_IB_GBPS / candidate_ethernet_gbps
        projected_low, projected_high = nccl_low * scale, nccl_high * scale
        if sparse_seconds < projected_low and not math.isclose(
            sparse_seconds, projected_low
        ):
            winner = transport
        elif sparse_seconds > projected_high and not math.isclose(
            sparse_seconds, projected_high
        ):
            winner = "nccl"
        else:
            winner = "depends"

    return Estimate(
        transport,
        model_size_gb,
        changed_pct,
        sparse_seconds,
        predict_sparse_wire_gb(model_size_gb, changed_pct),
        nccl_low,
        nccl_high,
        candidate_ethernet_gbps,
        projected_low,
        projected_high,
        crossover_low,
        crossover_high,
        winner,
    )


def _positive(value: str) -> float:
    number = float(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return number


def _seconds(low: float, high: float) -> str:
    return f"{low:.3f}-{high:.3f}s"


def _print_results(results: list[Estimate]) -> None:
    first = results[0]
    print(
        f"Model: {first.model_size_gb:g} GB indexed BF16; "
        f"changed: {first.changed_pct:g}%; compression: zstd"
    )
    print(
        "Measured NCCL on 400 Gbps/rank H100 IB: "
        f"{_seconds(first.nccl_ib_low_s, first.nccl_ib_high_s)}"
    )
    if first.candidate_ethernet_gbps is not None:
        assert first.nccl_ethernet_low_s is not None
        assert first.nccl_ethernet_high_s is not None
        print(
            f"Projected NCCL on {first.candidate_ethernet_gbps:g} Gbps/rank "
            f"Ethernet: {_seconds(first.nccl_ethernet_low_s, first.nccl_ethernet_high_s)}"
        )

    print(
        "\nPath       Sparse       Wire         Ethernet crossover   Winner@candidate"
    )
    for result in results:
        crossover = (
            f"{result.break_even_ethernet_low_gbps:.2f}-"
            f"{result.break_even_ethernet_high_gbps:.2f} Gbps/rank"
        )
        print(
            f"{result.transport.upper():<6} {result.sparse_seconds:>9.3f}s "
            f"{result.approximate_wire_gb:>8.3f} GB {crossover:>26} "
            f"{result.candidate_winner or '':>18}"
        )
    print(
        "\nBelow the lower crossover sparse wins across the NCCL envelope; "
        "above the upper crossover NCCL wins."
    )
    if not _SPARSE_SIZE_RANGE_GB[0] <= first.model_size_gb <= _SPARSE_SIZE_RANGE_GB[1]:
        print("Note: model size is outside the measured sparse calibration range.")
    if not _DENSITIES[0] <= first.changed_pct <= _DENSITIES[1]:
        print("Note: changed density is extrapolated from measured 3% and 5% arms.")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-size-gb", type=_positive, required=True)
    parser.add_argument(
        "--changed-pct", "--sparsity-pct", type=_positive, required=True
    )
    parser.add_argument("--transport", choices=("all", "s3", "zmq"), default="all")
    parser.add_argument("--candidate-ethernet-gbps", type=_positive)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()
    transports: tuple[Transport, ...] = (
        ("s3", "zmq") if args.transport == "all" else (args.transport,)
    )
    results = [
        estimate(
            model_size_gb=args.model_size_gb,
            changed_pct=args.changed_pct,
            transport=transport,
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
