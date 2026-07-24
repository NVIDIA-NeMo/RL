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

"""CPU-only preflight and evidence checks for SGLang refit test drivers."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


class CheckError(RuntimeError):
    """Raised when a refit preflight or evidence check fails."""


GROUP_READY_PATTERN = re.compile(
    r"\bNRL_SGLANG_REFIT_GROUP_READY "
    r"world_size=(?P<world_size>\d+) engines=(?P<engines>\d+)\b"
)
FATAL_LOG_PATTERNS = {
    "SGLang refit failure marker": re.compile(r"\bNRL_SGLANG_REFIT_FAILURE\b"),
    "KV-cache invalidation failure": re.compile(
        r"\bKV[- ]cache\b.*\b(?:invalidate|invalidation)\w*\b.*\bfail\w*\b",
        re.IGNORECASE,
    ),
    "NCCL system or remote error": re.compile(
        r"\bNCCL\b.*\b(?:system|remote) error\b", re.IGNORECASE
    ),
    "collective watchdog timeout": re.compile(
        r"(?:\bcollective\b.*\bwatchdog\b|\bwatchdog\b.*\bcollective\b)"
        r".*\btime(?:d out|out)\b",
        re.IGNORECASE,
    ),
    "failed receive": re.compile(r"\bFailed to recv\b", re.IGNORECASE),
    "connection reset": re.compile(r"\bconnection reset\b", re.IGNORECASE),
}


def _whole_number(value: float) -> int | float:
    return int(value) if value.is_integer() else value


def check_cluster_capacity(
    nodes: Sequence[Mapping[str, Any]],
    *,
    required_nodes: int,
    gpus_per_node: int,
    available_gpus: float | None = None,
) -> dict[str, Any]:
    """Validate per-node GPU capacity without scheduling a Ray task."""
    if required_nodes <= 0 or gpus_per_node <= 0:
        raise ValueError("required_nodes and gpus_per_node must be positive")

    alive_nodes = [node for node in nodes if node.get("Alive") is True]
    gpu_capacities = sorted(
        float(node.get("Resources", {}).get("GPU", 0)) for node in alive_nodes
    )
    gpu_capacities = [capacity for capacity in gpu_capacities if capacity > 0]
    eligible_nodes = [
        capacity for capacity in gpu_capacities if capacity >= gpus_per_node
    ]
    required_gpus = required_nodes * gpus_per_node
    total_gpus = sum(gpu_capacities)

    observed = {
        "alive_nodes": len(alive_nodes),
        "alive_gpu_nodes": len(gpu_capacities),
        "eligible_gpu_nodes": len(eligible_nodes),
        "gpu_capacities": [_whole_number(value) for value in gpu_capacities],
        "total_gpus": _whole_number(total_gpus),
    }
    if available_gpus is not None:
        observed["available_gpus"] = _whole_number(available_gpus)
    required = {
        "nodes": required_nodes,
        "gpus_per_node": gpus_per_node,
        "total_gpus": required_gpus,
    }

    insufficient_available_gpus = (
        available_gpus is not None and available_gpus < required_gpus
    )
    if (
        len(eligible_nodes) < required_nodes
        or total_gpus < required_gpus
        or insufficient_available_gpus
    ):
        available_detail = (
            f", {_whole_number(available_gpus)} currently available"
            if available_gpus is not None
            else ""
        )
        raise CheckError(
            "Ray cluster capacity is insufficient: "
            f"required {required_nodes} nodes with at least {gpus_per_node} GPUs "
            f"each ({required_gpus} GPUs total), observed "
            f"{len(eligible_nodes)} eligible nodes and "
            f"{_whole_number(total_gpus)} GPUs across alive GPU nodes "
            f"{observed['gpu_capacities']}{available_detail}."
        )

    return {"status": "passed", "required": required, "observed": observed}


def check_log_markers(
    log_text: str,
    *,
    expected_world_size: int,
    expected_engines: int,
    min_refit_successes: int,
) -> dict[str, Any]:
    """Validate exact communicator topology and fatal/success markers."""
    observed_topologies = [
        (int(match["world_size"]), int(match["engines"]))
        for match in GROUP_READY_PATTERN.finditer(log_text)
    ]
    expected_topology = (expected_world_size, expected_engines)
    if not observed_topologies:
        raise CheckError("No NRL_SGLANG_REFIT_GROUP_READY marker was found.")

    unexpected_topologies = sorted(
        topology
        for topology in set(observed_topologies)
        if topology != expected_topology
    )
    if expected_topology not in observed_topologies or unexpected_topologies:
        raise CheckError(
            "Unexpected SGLang refit topology: "
            f"expected only world_size={expected_world_size}, "
            f"engines={expected_engines}; observed {sorted(set(observed_topologies))}."
        )

    fatal_matches = [
        description
        for description, pattern in FATAL_LOG_PATTERNS.items()
        if pattern.search(log_text)
    ]
    if fatal_matches:
        raise CheckError(
            "Fatal refit evidence was found in the run log: "
            + ", ".join(fatal_matches)
            + "."
        )

    success_count = log_text.count("NRL_SGLANG_REFIT_SUCCESS")
    if success_count < min_refit_successes:
        raise CheckError(
            f"Expected at least {min_refit_successes} completed SGLang refits, "
            f"found {success_count}."
        )

    return {
        "group_ready_count": len(observed_topologies),
        "refit_success_count": success_count,
        "world_size": expected_world_size,
        "engines": expected_engines,
    }


def check_training_metrics(
    metrics: Mapping[str, Any], *, expected_max_step: int
) -> dict[str, int]:
    """Require train/loss to include the requested terminal step."""
    train_loss = metrics.get("train/loss")
    if not isinstance(train_loss, Mapping) or not train_loss:
        raise CheckError("metrics.json has no non-empty train/loss series.")

    try:
        recorded_steps = [int(step) for step in train_loss]
    except (TypeError, ValueError) as exc:
        raise CheckError("train/loss contains a non-integer step key.") from exc

    max_recorded_step = max(recorded_steps)
    if max_recorded_step < expected_max_step:
        raise CheckError(
            f"Expected train/loss through step {expected_max_step}, "
            f"found step {max_recorded_step}."
        )
    return {
        "expected_max_step": expected_max_step,
        "max_recorded_step": max_recorded_step,
    }


def find_nonempty_gym_artifacts(log_dir: Path) -> list[Path]:
    """Return non-empty NeMo-Gym training rollout artifacts."""
    if not log_dir.is_dir():
        return []
    return sorted(
        path
        for path in log_dir.rglob("train_data_step*.jsonl")
        if path.is_file() and path.stat().st_size > 0
    )


def validate_run_evidence(
    *,
    log_text: str,
    metrics: Mapping[str, Any],
    expected_max_step: int,
    expected_world_size: int,
    expected_engines: int,
    min_refit_successes: int,
    gym_log_dir: Path | None = None,
) -> dict[str, Any]:
    """Combine marker, metric, and optional Gym artifact validation."""
    summary: dict[str, Any] = {
        "status": "passed",
        "markers": check_log_markers(
            log_text,
            expected_world_size=expected_world_size,
            expected_engines=expected_engines,
            min_refit_successes=min_refit_successes,
        ),
        "metrics": check_training_metrics(metrics, expected_max_step=expected_max_step),
    }
    if gym_log_dir is not None:
        gym_artifacts = find_nonempty_gym_artifacts(gym_log_dir)
        if not gym_artifacts:
            raise CheckError(
                "No non-empty NeMo-Gym train_data_step*.jsonl artifact was produced."
            )
        summary["gym"] = {
            "artifact_count": len(gym_artifacts),
            "artifacts": [str(path.relative_to(gym_log_dir)) for path in gym_artifacts],
        }
    return summary


def _write_summary(summary: Mapping[str, Any], output: Path | None) -> None:
    rendered = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered)
    print(rendered, end="")


def _run_preflight(args: argparse.Namespace) -> dict[str, Any]:
    try:
        import ray

        ray.init(
            address=args.ray_address,
            include_dashboard=False,
            log_to_driver=False,
        )
    except Exception as exc:
        raise CheckError(
            "Could not attach to an existing Ray cluster. This preflight never "
            "starts a local cluster; allocate and bootstrap Ray before running "
            "the reproducer."
        ) from exc

    try:
        nodes = ray.nodes()
        available_gpus = float(ray.available_resources().get("GPU", 0))
    finally:
        ray.shutdown()

    return check_cluster_capacity(
        nodes,
        required_nodes=args.num_nodes,
        gpus_per_node=args.gpus_per_node,
        available_gpus=available_gpus,
    )


def _run_evidence_validation(args: argparse.Namespace) -> dict[str, Any]:
    try:
        log_text = args.run_log.read_text(errors="replace")
    except OSError as exc:
        raise CheckError(f"Could not read run log {args.run_log}: {exc}") from exc
    try:
        metrics = json.loads(args.metrics_json.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise CheckError(
            f"Could not read metrics JSON {args.metrics_json}: {exc}"
        ) from exc
    if not isinstance(metrics, Mapping):
        raise CheckError("metrics.json must contain a JSON object.")

    return validate_run_evidence(
        log_text=log_text,
        metrics=metrics,
        expected_max_step=args.max_step,
        expected_world_size=args.world_size,
        expected_engines=args.engines,
        min_refit_successes=args.min_refit_successes,
        gym_log_dir=args.gym_log_dir,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    preflight = subparsers.add_parser(
        "preflight", help="Check an already-running Ray cluster without using a GPU."
    )
    preflight.add_argument("--num-nodes", type=int, required=True)
    preflight.add_argument("--gpus-per-node", type=int, required=True)
    preflight.add_argument("--ray-address", default="auto")
    preflight.add_argument("--output", type=Path)
    preflight.set_defaults(handler=_run_preflight)

    validate = subparsers.add_parser(
        "validate", help="Validate fresh run metrics, topology, and refit markers."
    )
    validate.add_argument("--run-log", type=Path, required=True)
    validate.add_argument("--metrics-json", type=Path, required=True)
    validate.add_argument("--max-step", type=int, required=True)
    validate.add_argument("--world-size", type=int, required=True)
    validate.add_argument("--engines", type=int, required=True)
    validate.add_argument("--min-refit-successes", type=int, default=2)
    validate.add_argument("--gym-log-dir", type=Path)
    validate.add_argument("--output", type=Path)
    validate.set_defaults(handler=_run_evidence_validation)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        summary = args.handler(args)
    except (CheckError, ValueError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        if exc.__cause__ is not None:
            print(f"[ERROR] cause: {exc.__cause__!r}", file=sys.stderr)
        return 1
    _write_summary(summary, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
