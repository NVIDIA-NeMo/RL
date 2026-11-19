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
"""Fail if a Megatron checkpoint repeats the previous checkpoint's training state.

Usage: check_checkpoint_freshness.py <previous weights dir> <current weights dir>
"""

import argparse
import hashlib
import sys
from collections import defaultdict
from pathlib import Path

import torch
from torch.distributed.checkpoint.filesystem import FileSystemReader
from torch.distributed.checkpoint.metadata import TensorStorageMetadata


def hash_checkpoint_tensors(weights_path: str) -> dict[tuple, str]:
    """Hash every tensor chunk of a torch_dist checkpoint by its byte range."""
    (iter_dir,) = Path(weights_path).glob("iter_*")
    metadata = FileSystemReader(str(iter_dir)).read_metadata()
    hashes = {}
    for index, info in metadata.storage_data.items():
        if not isinstance(
            metadata.state_dict_metadata[index.fqn], TensorStorageMetadata
        ):
            continue
        with open(iter_dir / info.relative_path, "rb") as f:
            f.seek(info.offset)
            chunk = f.read(info.length)
        hashes[(index.fqn, str(index.offset), index.index)] = hashlib.sha256(
            chunk
        ).hexdigest()
    return hashes


def _weight_dtypes(weights_path: str) -> set[torch.dtype]:
    """Dtypes of the model weights (non-optimizer tensors) in a torch_dist checkpoint."""
    (iter_dir,) = Path(weights_path).glob("iter_*")
    metadata = FileSystemReader(str(iter_dir)).read_metadata()
    return {
        md.properties.dtype
        for fqn, md in metadata.state_dict_metadata.items()
        if isinstance(md, TensorStorageMetadata) and not fqn.startswith("optimizer.")
    }


def stale_tensors(previous: str, current: str) -> list[str]:
    """Tensors that show the current checkpoint repeats the previous one."""
    first = hash_checkpoint_tensors(previous)
    second = hash_checkpoint_tensors(current)
    low_precision = any(
        dtype in (torch.bfloat16, torch.float16) for dtype in _weight_dtypes(previous)
    )
    # fp32 params never update TE FusedAdam's master `param`; with bf16/fp16 params it is the master weight.
    checked = (".exp_avg", ".param") if low_precision else (".exp_avg",)
    # A tensor counts as changed once any of its chunks changes.
    changed: dict[str, bool] = defaultdict(bool)
    for key, digest in first.items():
        if key[0].startswith("optimizer.") and any(s in key[0] for s in checked):
            changed[key[0]] |= digest != second[key]
    assert changed, f"no optimizer state in {previous}"
    stale = sorted(fqn for fqn, is_changed in changed.items() if not is_changed)
    # A small lr leaves many bf16 weights unchanged, so only an all-repeated model is stale.
    weight_keys = [key for key in first if not key[0].startswith("optimizer.")]
    if weight_keys and all(first[key] == second[key] for key in weight_keys):
        stale.append("model weights")
    return stale


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("previous", help="weights dir of the earlier checkpoint")
    parser.add_argument("current", help="weights dir of the later checkpoint")
    args = parser.parse_args()
    stale = stale_tensors(args.previous, args.current)
    if stale:
        sys.exit(f"{len(stale)} tensors repeat {args.previous}: {stale}")
    print(f"{args.current}: training state differs from {args.previous}")


if __name__ == "__main__":
    main()
