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

"""Benchmark DFlash FlexAttention with explicit provenance and metric scope."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import time
import warnings
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from nemo_rl.models.megatron.draft.block_attention import (
    _block_visibility,
    _grouped_masked_attention,
    _trunk_visibility,
    dflash_block_attention,
)
from nemo_rl.models.megatron.draft.block_plan import (
    DFlashBatchPlan,
    build_dflash_batch_plan,
)


_DEFAULT_CASES = ((1024, 4, 3), (4096, 8, 9), (1024, 512, 9))


def _parse_case(value: str) -> tuple[int, int, int]:
    try:
        sequence_length, anchors_per_sample, block_size = (
            int(item) for item in value.split(":")
        )
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "cases must use SEQUENCE_LENGTH:ANCHORS_PER_SAMPLE:BLOCK_SIZE",
        ) from error
    if sequence_length < 1 or anchors_per_sample < 1 or block_size < 2:
        raise argparse.ArgumentTypeError(
            "sequence length and anchors must be positive; block size must be >= 2",
        )
    return sequence_length, anchors_per_sample, block_size


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    index = quantile * (len(ordered) - 1)
    lower = math.floor(index)
    upper = math.ceil(index)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)


def _git_head(repo_root: Path) -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build_plan(
    *,
    batch_size: int,
    sequence_length: int,
    anchors_per_sample: int,
    block_size: int,
    device: torch.device,
) -> DFlashBatchPlan:
    return build_dflash_batch_plan(
        torch.ones(
            (batch_size, sequence_length),
            dtype=torch.bool,
            device=device,
        ),
        torch.arange(batch_size, dtype=torch.int64, device=device),
        anchors_per_sample=anchors_per_sample,
        gamma=block_size - 1,
        optimizer_step=17,
        seed=2026,
    )


def _make_inputs(
    plan: DFlashBatchPlan,
    *,
    device: torch.device,
) -> tuple[Tensor, ...]:
    generator = torch.Generator(device=device).manual_seed(
        plan.sequence_length * 1000 + plan.anchors_per_sample * 10 + plan.block_size
    )
    num_blocks = plan.batch_size * plan.anchors_per_sample
    shapes = (
        (plan.batch_size, plan.sequence_length, 8, 64),
        (plan.batch_size, plan.sequence_length, 2, 64),
        (plan.batch_size, plan.sequence_length, 2, 64),
        (num_blocks, plan.block_size, 8, 64),
        (num_blocks, plan.block_size, 2, 64),
        (num_blocks, plan.block_size, 2, 64),
    )
    return tuple(
        torch.randn(
            shape,
            dtype=torch.bfloat16,
            device=device,
            generator=generator,
        ).requires_grad_(True)
        for shape in shapes
    )


def _train_step(plan: DFlashBatchPlan, inputs: tuple[Tensor, ...]) -> None:
    for tensor in inputs:
        tensor.grad = None
    trunk_output, block_output = dflash_block_attention(
        plan=plan,
        trunk_q=inputs[0],
        trunk_k=inputs[1],
        trunk_v=inputs[2],
        block_q=inputs[3],
        block_k=inputs[4],
        block_v=inputs[5],
    )
    loss = trunk_output.float().square().mean() + block_output.float().square().mean()
    loss.backward()


def _time_call(device: torch.device, operation: Any) -> float:
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    operation()
    torch.cuda.synchronize(device)
    return (time.perf_counter() - start) * 1000


def _benchmark_case(
    *,
    batch_size: int,
    sequence_length: int,
    anchors_per_sample: int,
    block_size: int,
    warmup: int,
    iterations: int,
    device: torch.device,
) -> dict[str, Any]:
    torch.compiler.reset()
    plan = _build_plan(
        batch_size=batch_size,
        sequence_length=sequence_length,
        anchors_per_sample=anchors_per_sample,
        block_size=block_size,
        device=device,
    )
    inputs = _make_inputs(plan, device=device)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    first_train_step_ms = _time_call(device, lambda: _train_step(plan, inputs))
    first_train_step_peak_bytes = torch.cuda.max_memory_allocated(device)

    for _ in range(warmup):
        _train_step(plan, inputs)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)
    durations_ms = [
        _time_call(device, lambda: _train_step(plan, inputs)) for _ in range(iterations)
    ]
    return {
        "kind": "flex_synchronized_train_step",
        "batch_size": batch_size,
        "sequence_length": sequence_length,
        "anchors_per_sample": anchors_per_sample,
        "block_size": block_size,
        "dtype": "bfloat16",
        "num_query_heads": 8,
        "num_kv_heads": 2,
        "head_dim": 64,
        "first_synchronized_train_step_ms": first_train_step_ms,
        "steady_synchronized_train_step_p50_ms": _percentile(durations_ms, 0.50),
        "steady_synchronized_train_step_p95_ms": _percentile(durations_ms, 0.95),
        "first_train_step_peak_bytes": first_train_step_peak_bytes,
        "steady_train_step_peak_bytes": torch.cuda.max_memory_allocated(device),
        "additional_warmup_steps": warmup,
        "measurement_iterations": iterations,
    }


@torch.no_grad()
def _correctness_comparison(
    *,
    device: torch.device,
    iterations: int,
) -> dict[str, Any]:
    torch.compiler.reset()
    plan = _build_plan(
        batch_size=2,
        sequence_length=1024,
        anchors_per_sample=4,
        block_size=3,
        device=device,
    )
    inputs = tuple(tensor.detach() for tensor in _make_inputs(plan, device=device))

    def flex_forward() -> tuple[Tensor, Tensor]:
        return dflash_block_attention(
            plan=plan,
            trunk_q=inputs[0],
            trunk_k=inputs[1],
            trunk_v=inputs[2],
            block_q=inputs[3],
            block_k=inputs[4],
            block_v=inputs[5],
        )

    global_key = torch.cat(
        (inputs[1].reshape(1, -1, 2, 64), inputs[4].reshape(1, -1, 2, 64)),
        dim=1,
    )
    global_value = torch.cat(
        (inputs[2].reshape(1, -1, 2, 64), inputs[5].reshape(1, -1, 2, 64)),
        dim=1,
    )

    def dense_forward() -> tuple[Tensor, Tensor]:
        return (
            _grouped_masked_attention(
                inputs[0],
                inputs[1],
                inputs[2],
                _trunk_visibility(plan),
                scale=0.125,
            ),
            _grouped_masked_attention(
                inputs[3],
                global_key,
                global_value,
                _block_visibility(plan),
                scale=0.125,
            ),
        )

    flex_outputs = flex_forward()
    dense_outputs = dense_forward()
    flex_durations_ms = [_time_call(device, flex_forward) for _ in range(iterations)]
    dense_durations_ms = [_time_call(device, dense_forward) for _ in range(iterations)]
    errors = [
        (flex.float() - dense.float()).abs().max().item()
        for flex, dense in zip(flex_outputs, dense_outputs, strict=True)
    ]
    return {
        "kind": "forward_correctness_comparison",
        "scope": "correctness; timings are same-shape forward-only observations",
        "batch_size": 2,
        "sequence_length": 1024,
        "anchors_per_sample": 4,
        "block_size": 3,
        "flex_forward_only_p50_ms": _percentile(flex_durations_ms, 0.50),
        "dense_forward_only_p50_ms": _percentile(dense_durations_ms, 0.50),
        "trunk_max_abs_error": errors[0],
        "block_max_abs_error": errors[1],
        "measurement_iterations": iterations,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--case",
        action="append",
        type=_parse_case,
        help="SEQUENCE_LENGTH:ANCHORS_PER_SAMPLE:BLOCK_SIZE; may be repeated",
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    args = parser.parse_args()
    if args.batch_size < 1 or args.warmup < 0 or args.iterations < 1:
        parser.error("batch size and iterations must be positive; warmup must be >= 0")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 is required")

    warnings.filterwarnings(
        "error",
        message=r"flex_attention called without torch\.compile",
    )
    script_path = Path(__file__).resolve()
    repo_root = script_path.parents[1]
    device = torch.device("cuda", 0)
    records: list[dict[str, Any]] = [
        {
            "kind": "provenance",
            "source_head": _git_head(repo_root),
            "benchmark_script": str(script_path.relative_to(repo_root)),
            "benchmark_script_sha256": _sha256(script_path),
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "device_name": torch.cuda.get_device_name(device),
            "device_count": torch.cuda.device_count(),
        }
    ]
    cases = args.case if args.case is not None else _DEFAULT_CASES
    for sequence_length, anchors_per_sample, block_size in cases:
        record = _benchmark_case(
            batch_size=args.batch_size,
            sequence_length=sequence_length,
            anchors_per_sample=anchors_per_sample,
            block_size=block_size,
            warmup=args.warmup,
            iterations=args.iterations,
            device=device,
        )
        records.append(record)
        print(json.dumps(record), flush=True)
    correctness_record = _correctness_comparison(
        device=device,
        iterations=args.iterations,
    )
    records.append(correctness_record)
    print(json.dumps(correctness_record), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "\n".join(json.dumps(record) for record in records) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
