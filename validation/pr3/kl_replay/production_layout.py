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

"""Repeat the seven controls with identical production DP placement throughout.

Only the diagnostic module's metadata sharder is replaced. For full-input calls,
each rank receives the concatenation of its two production chunk shards. Thus
full, numerator-reference and streaming execute the same two global microbatches.
The worker, loss, original replay implementation and parity helpers stay intact.
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any
from unittest.mock import patch

import torch

from nemo_rl.data_plane.interfaces import KVBatchMeta
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from validation.pr3.kl_replay import replay


def run(args: argparse.Namespace) -> None:
    """Validate actual fetched inputs and replay one production microbatch layout."""
    if os.environ["USER"] != "root" or not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Requires an allocated container")
    rank = int(os.environ["RANK"])
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    packets = [
        torch.load(path, map_location="cpu", weights_only=False)
        for path in sorted(Path(args.capture).glob("chunk-*.pt"))
    ]
    expected = BatchedDataDict.from_batches(
        [BatchedDataDict(packet["data"]) for packet in packets]
    )
    ids = [sample for packet in packets for sample in packet["ids"]]
    index = {sample: position for position, sample in enumerate(ids)}
    source_shard = replay.shard_meta_for_dp
    source_fetch = replay.AutomodelPolicyWorkerImpl._fetch
    source_raw = replay.parity.snapshot_raw_optimizer
    checked: set[tuple[str, tuple[str, ...]]] = set()
    initial_written = False

    def shard(
        meta: KVBatchMeta, *, dp_world: int, batch_size: int | None = None
    ) -> tuple[list[KVBatchMeta], list[int] | None]:
        if len(meta.sample_ids) != 32:
            return source_shard(meta, dp_world=dp_world, batch_size=batch_size)
        halves = [
            source_shard(
                meta.slice(start, start + 16), dp_world=dp_world, batch_size=16
            )[0]
            for start in (0, 16)
        ]
        positions = {
            sample: position for position, sample in enumerate(meta.sample_ids)
        }
        shards = [
            meta.subset(
                [positions[sample] for half in halves for sample in half[r].sample_ids]
            )
            for r in range(dp_world)
        ]
        flat = [sample for part in shards for sample in part.sample_ids]
        if len(flat) != 32 or set(flat) != set(ids):
            raise ValueError("Production layout lost or duplicated a response")
        inverse = [0] * 32
        for new, sample in enumerate(flat):
            inverse[positions[sample]] = new
        (output / f"layout-rank{rank}.json").write_text(
            json.dumps(
                {"all_rank_sample_ids": [part.sample_ids for part in shards]}, indent=2
            )
            + "\n"
        )
        return shards, inverse

    def fetch(worker: Any, meta: KVBatchMeta, **kwargs: Any) -> Any:
        data = source_fetch(worker, meta, **kwargs)
        key = (meta.partition_id, tuple(meta.sample_ids))
        if key not in checked:
            for row, sample in enumerate(meta.sample_ids):
                position = index[sample]
                length = int(expected["input_lengths"][position])
                for field, original in expected.items():
                    actual = data[field][row].detach().cpu()
                    wanted = original[position]
                    # The adapter regenerates padding using tokenizer defaults;
                    # require exact values/dtypes on every valid token column.
                    if (
                        original.ndim >= 2
                        and original.shape[1] == expected["input_ids"].shape[1]
                    ):
                        actual, wanted = actual[:length], wanted[:length]
                    if actual.dtype != wanted.dtype or not torch.equal(actual, wanted):
                        raise ValueError(f"Fetched {field} differs for {sample}")
            with (output / f"input-check-rank{rank}.jsonl").open("a") as stream:
                stream.write(
                    json.dumps(
                        {
                            "partition": meta.partition_id,
                            "sample_ids": meta.sample_ids,
                            "forward_pad": data["input_ids"].shape[1],
                            "validated_fields": list(expected),
                        }
                    )
                    + "\n"
                )
            checked.add(key)
        return data

    def snapshot_raw(worker: Any) -> dict[str, Any]:
        nonlocal initial_written
        raw = source_raw(worker)
        if not initial_written:
            initial_written = True
            dtypes: dict[str, set[str]] = {}
            for values in raw["state"].values():
                for field, value in values.items():
                    if torch.is_tensor(value):
                        dtypes.setdefault(field, set()).add(str(value.dtype))
            groups = [
                {
                    key: value.item() if torch.is_tensor(value) else value
                    for key, value in group.items()
                    if key in {"step", "lr", "initial_lr"}
                }
                for group in worker.optimizer.param_groups
            ]
            scheduler = worker.scheduler.state_dict()
            (output / f"initial-state-rank{rank}.json").write_text(
                json.dumps(
                    {
                        "scheduler": scheduler,
                        "optimizer_groups": groups,
                        "raw_state_dtypes": {k: sorted(v) for k, v in dtypes.items()},
                        "raw_scales_present": bool(raw["scales"]),
                    },
                    indent=2,
                )
                + "\n"
            )
            if args.checkpoint and scheduler["last_epoch"] != 210:
                raise ValueError("Checkpoint210 scheduler was not restored")
            if args.checkpoint and any(
                group.get("step", 210) != 210 for group in groups
            ):
                raise ValueError("Checkpoint210 optimizer counter was not restored")
        return raw

    with (
        patch.object(replay, "shard_meta_for_dp", side_effect=shard),
        patch.object(replay.AutomodelPolicyWorkerImpl, "_fetch", fetch),
        patch.object(replay.parity, "snapshot_raw_optimizer", side_effect=snapshot_raw),
    ):
        replay.run(args)
    if rank == 0:
        path = output / "diagnostic.json"
        report = json.loads(path.read_text())
        report["layout_control"] = "Identical production DP microbatches in every case"
        report["input_validation"] = (
            "Exact valid-token columns/dtypes checked once per metadata fetch; no tensor validation in timing repeats"
        )
        path.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--capture", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--output", required=True)
    parser.add_argument("--source-sha", required=True)
    parser.add_argument("--repeats", type=int, default=3)
    run(parser.parse_args())
