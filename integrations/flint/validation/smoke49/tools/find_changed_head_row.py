#!/usr/bin/env python3
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
"""Find an informative changed LM-head row in raw saved DCP safetensors.

Run on the compute allocation with the training Python. Only LM-head slices
are read. Hashes describe contiguous FP32 sample bytes, matching SGLang's FP32
LM head. Both the earliest changed row anywhere and the earliest row changed
within the sampled columns are reported; the latter is suitable for refit
receipts that sample the first 32 columns.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from contextlib import ExitStack
from pathlib import Path

from audit_training_result import describe_views, locate_model_directory, read_header


def tensor_sha256(value):
    import torch

    return hashlib.sha256(
        memoryview(value.contiguous().view(torch.uint8).numpy())
    ).hexdigest()


def find_changed_head(
    source: Path,
    checkpoint: Path,
    sample_rows: int = 32,
    sample_columns: int = 32,
    chunk_rows: int = 256,
) -> dict:
    import torch
    from safetensors import safe_open

    torch.set_num_threads(min(4, torch.get_num_threads()))
    source, checkpoint = source.resolve(strict=True), checkpoint.resolve(strict=True)
    config = json.loads((source / "config.json").read_text())
    mapping = json.loads((source / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    source_filename = mapping["lm_head.weight"]
    if Path(source_filename).name != source_filename:
        raise ValueError("Source head shard must be a local basename")
    source_head_spec = read_header(source / source_filename)["lm_head.weight"]
    directory = locate_model_directory(checkpoint)
    views = []
    for path in sorted(directory.glob("*.safetensors")):
        values, _ = describe_views(path, config)
        views.extend(value for value in values if value.source_name == "lm_head.weight")
    if not views:
        raise ValueError("No LM-head regions found in saved checkpoint")
    if any(value.dtype != "F32" or len(value.shape) != 2 for value in views):
        raise ValueError("Expected saved head to consist of FP32 matrix regions")
    views.sort(key=lambda value: (value.offsets[0], value.offsets[1]))
    with ExitStack() as stack:
        original_handle = stack.enter_context(
            safe_open(source / source_filename, framework="pt", device="cpu")
        )
        original = original_handle.get_slice("lm_head.weight")
        shape = original.get_shape()
        sample_rows, sample_columns = (
            min(sample_rows, shape[0]),
            min(sample_columns, shape[1]),
        )
        handles = {
            path: stack.enter_context(safe_open(path, framework="pt", device="cpu"))
            for path in {value.file for value in views}
        }
        first_row, first_column, first_sample_row = None, None, None
        for view in views:
            start, width = view.offsets[0], view.shape[1]
            if (
                first_row is not None
                and first_sample_row is not None
                and start > max(first_row, first_sample_row)
            ):
                break
            saved = handles[view.file].get_slice(view.saved_name)
            for begin in range(0, view.shape[0], chunk_rows):
                global_begin = start + begin
                if (
                    first_row is not None
                    and first_sample_row is not None
                    and global_begin > max(first_row, first_sample_row)
                ):
                    break
                stop = min(begin + chunk_rows, view.shape[0])
                current = saved[begin:stop, :]
                baseline = original[
                    global_begin : start + stop,
                    view.offsets[1] : view.offsets[1] + width,
                ].float()
                different = current != baseline
                changed = torch.nonzero(different, as_tuple=False)
                if changed.numel():
                    row, column = (
                        int(changed[0, 0]) + global_begin,
                        int(changed[0, 1]) + view.offsets[1],
                    )
                    if (
                        first_row is None
                        or row < first_row
                        or (row == first_row and column < first_column)
                    ):
                        first_row, first_column = row, column
                prefix_width = min(width, max(0, sample_columns - view.offsets[1]))
                if prefix_width:
                    selected = torch.nonzero(
                        different[:, :prefix_width], as_tuple=False
                    )
                    if selected.numel():
                        row = int(selected[0, 0]) + global_begin
                        if first_sample_row is None or row < first_sample_row:
                            first_sample_row = row

        def sample(row):
            stop = min(row + sample_rows, shape[0])
            expected = original[row:stop, :sample_columns].float().contiguous()
            actual = torch.empty_like(expected)
            covered = torch.zeros_like(expected, dtype=torch.bool)
            for view in views:
                low, high = (
                    max(row, view.offsets[0]),
                    min(stop, view.offsets[0] + view.shape[0]),
                )
                left, right = (
                    max(0, view.offsets[1]),
                    min(sample_columns, view.offsets[1] + view.shape[1]),
                )
                if low >= high or left >= right:
                    continue
                tile = handles[view.file].get_slice(view.saved_name)[
                    low - view.offsets[0] : high - view.offsets[0],
                    left - view.offsets[1] : right - view.offsets[1],
                ]
                prior_coverage = covered[low - row : high - row, left:right]
                prior_values = actual[low - row : high - row, left:right]
                if bool(prior_coverage.any()) and not torch.equal(
                    prior_values[prior_coverage], tile[prior_coverage]
                ):
                    raise ValueError(
                        "Replicated saved head regions disagree in the requested sample"
                    )
                actual[low - row : high - row, left:right] = tile
                covered[low - row : high - row, left:right] = True
            if not bool(covered.all()):
                raise ValueError(
                    "Saved DCP regions do not completely cover the requested head sample"
                )
            if not bool(torch.isfinite(actual).all()):
                raise ValueError("Saved head sample contains nonfinite values")
            return {
                "row_start": row,
                "row_stop": stop,
                "column_start": 0,
                "column_stop": sample_columns,
                "shape": list(actual.shape),
                "dtype": "float32",
                "source_fp32_sha256": tensor_sha256(expected),
                "checkpoint_fp32_sha256": tensor_sha256(actual),
                "changed_elements": int(torch.count_nonzero(actual != expected)),
                "max_abs_change": float((actual - expected).abs().max()),
                "source_first_row_values": expected[0, :8].tolist(),
                "checkpoint_first_row_values": actual[0, :8].tolist(),
            }

        report = {
            "status": "pass" if first_row is not None else "unchanged",
            "source": str(source),
            "checkpoint": str(checkpoint),
            "model_directory": str(directory),
            "head_shape": shape,
            "source_dtype": source_head_spec["dtype"],
            "checkpoint_dtype": "F32",
            "first_changed_global_row": first_row,
            "first_changed_global_column": first_column,
            "first_changed_row_in_sample_columns": first_sample_row,
            "head_regions": [
                {
                    "file": value.file.name,
                    "shape": value.shape,
                    "offsets": value.offsets,
                }
                for value in views
            ],
        }
        if first_row is not None:
            report["sample_at_first_changed_row"] = sample(first_row)
        if first_sample_row is not None:
            report["informative_sample"] = sample(first_sample_row)
            report["recommended_refit_sample_row_start"] = first_sample_row
        return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--sample-rows", type=int, default=32)
    parser.add_argument("--sample-columns", type=int, default=32)
    parser.add_argument("--chunk-rows", type=int, default=256)
    args = parser.parse_args()
    if min(args.sample_rows, args.sample_columns, args.chunk_rows) <= 0:
        raise ValueError("Sample and chunk dimensions must be positive")
    if args.output:
        output = args.output.resolve()
        if (
            args.source.resolve() in output.parents
            or args.checkpoint.resolve() in output.parents
        ):
            raise ValueError("Report must be outside read-only input directories")
    report = find_changed_head(
        args.source,
        args.checkpoint,
        args.sample_rows,
        args.sample_columns,
        args.chunk_rows,
    )
    text = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(args.output, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        with os.fdopen(descriptor, "w") as stream:
            stream.write(text)
        os.chmod(args.output, 0o600)
    print(text, end="", flush=True)


if __name__ == "__main__":
    main()
