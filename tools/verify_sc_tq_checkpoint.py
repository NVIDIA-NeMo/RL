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

"""Verify SingleController-driven TQ save/load across a full process restart.

This is a CPU-only integration probe. It deliberately avoids constructing
generation and policy actors, but invokes the real async checkpoint hooks on
``SingleControllerActor`` from a Ray actor with the same connect-only
``TQDataPlaneClient`` used by production SingleController. Save and load run in
separate Python/TQ/Ray processes so success cannot come from retained
in-process state.

Example:
    uv run --no-sync python tools/verify_sc_tq_checkpoint.py \
        --checkpoint-dir /lustre/.../sc-tq-checkpoint-smoke
"""

from __future__ import annotations

import argparse
import asyncio
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

import ray
import torch
from tensordict import TensorDict

from nemo_rl.algorithms.single_controller import SingleControllerActor
from nemo_rl.data_plane import DataPlaneConfig, build_data_plane_client

PARTITION_ID = "sc_checkpoint_smoke"
TASK_NAME = "train"
SAMPLE_IDS = [f"prompt-0:generation-{index}" for index in range(4)]
SEQ_LEN = 16
FIELDS = ["token_ids", "token_mask", "generation_logprobs"]


def _data_plane_config(num_storage_units: int) -> DataPlaneConfig:
    return cast(
        DataPlaneConfig,
        {
            "enabled": True,
            "impl": "transfer_queue",
            "backend": "simple",
            "storage_capacity": 1024,
            "num_storage_units": num_storage_units,
            "claim_meta_poll_interval_s": 0.05,
            "global_segment_size": 8 * 1024**3,
            "local_buffer_size": 1024**3,
        },
    )


def _controller_for_checkpoint(dp_client: Any) -> Any:
    """Construct only the state used by the real async lifecycle hooks."""
    controller_cls = SingleControllerActor.__ray_metadata__.modified_class
    controller = object.__new__(controller_cls)
    controller._dp_client = dp_client
    controller._data_plane_checkpoint_lock = asyncio.Lock()
    controller._run_started = False
    controller._train_steps = 0
    controller._trainer_version = 3
    controller._current_epoch = 1
    return controller


@ray.remote(num_cpus=1)
class _CheckpointProbe:
    """Host the real SingleController checkpoint hooks in a Ray process."""

    def __init__(self, data_plane_config: DataPlaneConfig) -> None:
        self._dp_client = build_data_plane_client(
            data_plane_config,
            bootstrap=False,
        )
        self._controller = _controller_for_checkpoint(self._dp_client)

    async def save(
        self,
        checkpoint_dir: str,
        metadata: dict[str, Any],
    ) -> None:
        await self._controller.save_data_plane_checkpoint(
            checkpoint_dir,
            metadata=metadata,
        )

    async def load(self, checkpoint_dir: str) -> dict[str, Any]:
        return await self._controller.load_data_plane_checkpoint(checkpoint_dir)

    def close(self) -> None:
        self._dp_client.close()


def _expected_fields() -> TensorDict:
    token_ids = torch.arange(len(SAMPLE_IDS) * SEQ_LEN, dtype=torch.int64).reshape(
        len(SAMPLE_IDS),
        SEQ_LEN,
    )
    return TensorDict(
        {
            "token_ids": token_ids,
            "token_mask": torch.ones_like(token_ids),
            "generation_logprobs": -token_ids.to(torch.float32) / 100.0,
        },
        batch_size=[len(SAMPLE_IDS)],
    )


def _close_data_plane(dp_client: Any, probe: Any | None) -> None:
    try:
        if probe is not None:
            ray.get(probe.close.remote())
    finally:
        try:
            dp_client.close()
        finally:
            ray.shutdown()


def _save(checkpoint_dir: Path, num_storage_units: int) -> None:
    config = _data_plane_config(num_storage_units)
    dp_client = build_data_plane_client(
        config,
        bootstrap=True,
    )
    probe = None
    try:
        dp_client.register_partition(
            partition_id=PARTITION_ID,
            fields=FIELDS,
            num_samples=len(SAMPLE_IDS),
            consumer_tasks=[TASK_NAME],
        )
        dp_client.put_samples(
            sample_ids=SAMPLE_IDS,
            partition_id=PARTITION_ID,
            fields=_expected_fields(),
            tags=[{"policy_version": 3, "prompt_id": "prompt-0"} for _ in SAMPLE_IDS],
        )

        consumed = dp_client.claim_meta(
            partition_id=PARTITION_ID,
            task_name=TASK_NAME,
            required_fields=FIELDS,
            batch_size=1,
            timeout_s=30.0,
        )
        if consumed.size != 1:
            raise AssertionError(f"Expected one consumed row, got {consumed.size}")

        probe = _CheckpointProbe.remote(config)
        ray.get(
            probe.save.remote(
                str(checkpoint_dir),
                {"expected_consumed_ids": consumed.sample_ids},
            )
        )
    finally:
        _close_data_plane(dp_client, probe)


def _load(checkpoint_dir: Path, num_storage_units: int) -> None:
    config = _data_plane_config(num_storage_units)
    dp_client = build_data_plane_client(
        config,
        bootstrap=True,
    )
    probe = None
    try:
        probe = _CheckpointProbe.remote(config)
        metadata = ray.get(probe.load.remote(str(checkpoint_dir)))

        restored = dp_client.get_samples(
            sample_ids=SAMPLE_IDS,
            partition_id=PARTITION_ID,
            select_fields=FIELDS,
        )
        expected = _expected_fields()
        for field in FIELDS:
            if not torch.equal(restored[field], expected[field]):
                raise AssertionError(f"Restored field differs: {field}")

        consumed_ids = set(metadata["expected_consumed_ids"])
        if metadata["data_plane_checkpoint_schema_version"] != 1:
            raise AssertionError("Unexpected data-plane checkpoint schema")
        if metadata["single_controller_trainer_version"] != 3:
            raise AssertionError("SingleController recovery metadata was not saved")

        remaining = dp_client.claim_meta(
            partition_id=PARTITION_ID,
            task_name=TASK_NAME,
            required_fields=FIELDS,
            batch_size=len(SAMPLE_IDS),
            timeout_s=30.0,
        )
        if consumed_ids.intersection(remaining.sample_ids):
            raise AssertionError("A previously consumed row was claimed after restore")
        if set(remaining.sample_ids).union(consumed_ids) != set(SAMPLE_IDS):
            raise AssertionError("Restored consumption state lost or added rows")
        if not dp_client.check_consumption_status(PARTITION_ID, [TASK_NAME]):
            raise AssertionError("Restored consumer cursor did not reach completion")
    finally:
        _close_data_plane(dp_client, probe)


def _run_child(
    phase: str,
    checkpoint_dir: Path,
    num_storage_units: int,
) -> None:
    subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve()),
            "--phase",
            phase,
            "--checkpoint-dir",
            str(checkpoint_dir),
            "--num-storage-units",
            str(num_storage_units),
        ],
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=("round-trip", "save", "load"),
        default="round-trip",
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--num-storage-units", type=int, default=4)
    args = parser.parse_args()

    checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    if args.phase == "save":
        _save(checkpoint_dir, args.num_storage_units)
        return
    if args.phase == "load":
        _load(checkpoint_dir, args.num_storage_units)
        return

    _run_child("save", checkpoint_dir, args.num_storage_units)
    _run_child("load", checkpoint_dir, args.num_storage_units)
    print(
        "PASS: SingleController TQ checkpoint survived a fresh Python/TQ/Ray process",
        flush=True,
    )


if __name__ == "__main__":
    main()
