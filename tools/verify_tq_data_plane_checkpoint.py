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

"""Verify TQ data-plane save/load across a fresh process and parent rename.

The save phase writes sample tensors, tags, and partial consumer progress to
TQ under ``tmp_<bundle>/data_plane``, then renames the parent bundle to its
final path just like ``CheckpointManager``. The load phase starts a fresh TQ
instance, restores from ``<bundle>/data_plane`` before any partition operations,
and verifies both the tensors and the consumer cursor.

Example:
    uv run --no-sync python tools/verify_tq_data_plane_checkpoint.py \
        --checkpoint-dir /lustre/.../tq-data-plane-checkpoint-smoke
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

import torch
from tensordict import TensorDict

from nemo_rl.algorithms.async_utils.replay_buffer import (
    DataPlaneCheckpointMetadata,
)
from nemo_rl.data_plane import (
    DATA_PLANE_CHECKPOINT_SCHEMA_VERSION,
    DataPlaneConfig,
    build_data_plane_client,
)

PARTITION_ID = "tq_checkpoint_smoke"
TASK_NAME = "train"
SAMPLE_IDS = [f"prompt-0:generation-{index}" for index in range(4)]
SEQ_LEN = 16
FIELDS = ["token_ids", "token_mask", "generation_logprobs"]
DATA_PLANE_DIR = "data_plane"


def _checkpoint_metadata(
    expected_consumed_ids: list[str],
    *,
    schema_version: int = DATA_PLANE_CHECKPOINT_SCHEMA_VERSION,
) -> dict[str, Any]:
    """Build the typed SC envelope plus smoke-test-only cursor metadata."""
    envelope: DataPlaneCheckpointMetadata = {
        "data_plane_checkpoint_schema_version": schema_version,
        "single_controller_train_steps": 0,
        "single_controller_trainer_version": 0,
        "single_controller_epoch": 0,
        "partition_id": PARTITION_ID,
        "sampler_name": "checkpoint_smoke",
        "mode": "shadow",
    }
    return {**envelope, "expected_consumed_ids": expected_consumed_ids}


def _data_plane_config(num_storage_units: int) -> DataPlaneConfig:
    return cast(
        DataPlaneConfig,
        {
            "enabled": True,
            "impl": "transfer_queue",
            "backend": "simple",
            "claim_meta_poll_interval_s": 0.05,
            "simple": {
                "storage_capacity": 1024,
                "num_storage_units": num_storage_units,
            },
        },
    )


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


def _save(checkpoint_dir: Path, num_storage_units: int) -> None:
    dp_client = build_data_plane_client(
        _data_plane_config(num_storage_units),
        bootstrap=True,
    )
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

        dp_client.save_checkpoint(
            checkpoint_dir,
            metadata=_checkpoint_metadata(consumed.sample_ids),
        )
    finally:
        dp_client.close()


def _load(checkpoint_dir: Path, num_storage_units: int) -> None:
    dp_client = build_data_plane_client(
        _data_plane_config(num_storage_units),
        bootstrap=True,
    )
    try:
        metadata = dp_client.load_checkpoint(checkpoint_dir)
        checkpoint_metadata = cast(DataPlaneCheckpointMetadata, metadata)

        restored = dp_client.get_samples(
            sample_ids=SAMPLE_IDS,
            partition_id=PARTITION_ID,
            select_fields=FIELDS,
        )
        expected = _expected_fields()
        for field in FIELDS:
            restored_value = restored[field]
            expected_value = expected[field]
            assert isinstance(restored_value, torch.Tensor)
            assert isinstance(expected_value, torch.Tensor)
            if not torch.equal(restored_value, expected_value):
                raise AssertionError(f"Restored field differs: {field}")

        if (
            checkpoint_metadata["data_plane_checkpoint_schema_version"]
            != DATA_PLANE_CHECKPOINT_SCHEMA_VERSION
        ):
            raise AssertionError("Unexpected data-plane checkpoint schema")
        consumed_ids = set(metadata["expected_consumed_ids"])
        expected_remaining_ids = set(SAMPLE_IDS) - consumed_ids

        if dp_client.check_consumption_status(PARTITION_ID, [TASK_NAME]):
            raise AssertionError(
                "Restored consumer cursor marked every row consumed before "
                "the expected remaining rows were claimed"
            )
        remaining = dp_client.claim_meta(
            partition_id=PARTITION_ID,
            task_name=TASK_NAME,
            required_fields=FIELDS,
            batch_size=len(expected_remaining_ids),
            timeout_s=30.0,
        )
        if consumed_ids.intersection(remaining.sample_ids):
            raise AssertionError("A previously consumed row was claimed after restore")
        if set(remaining.sample_ids) != expected_remaining_ids:
            raise AssertionError("Restored consumption state lost or added rows")
        if not dp_client.check_consumption_status(PARTITION_ID, [TASK_NAME]):
            raise AssertionError("Restored consumer cursor did not reach completion")
    finally:
        dp_client.close()


def _save_and_finalize_bundle(
    bundle_dir: Path,
    num_storage_units: int,
) -> None:
    """Save below a temporary parent, then rename it to ``bundle_dir``."""
    staging_dir = bundle_dir.with_name(f"tmp_{bundle_dir.name}")
    if bundle_dir.exists():
        raise FileExistsError(f"Final checkpoint bundle already exists: {bundle_dir}")
    if staging_dir.exists():
        raise FileExistsError(
            f"Staging checkpoint bundle already exists: {staging_dir}"
        )

    # CheckpointManager creates tmp_step_N before component writers run.
    # Mirror that precondition instead of relying on TQ to create the parent.
    staging_dir.mkdir(parents=True)
    try:
        _save(staging_dir / DATA_PLANE_DIR, num_storage_units)
        staging_dir.rename(bundle_dir)
    except Exception:
        if staging_dir.exists():
            shutil.rmtree(staging_dir)
        raise


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
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        required=True,
        help="Final SC-like checkpoint bundle directory.",
    )
    parser.add_argument("--num-storage-units", type=int, default=4)
    args = parser.parse_args()

    checkpoint_dir = args.checkpoint_dir.expanduser().resolve()
    if args.phase == "save":
        _save_and_finalize_bundle(checkpoint_dir, args.num_storage_units)
        return
    if args.phase == "load":
        _load(checkpoint_dir / DATA_PLANE_DIR, args.num_storage_units)
        return

    _run_child("save", checkpoint_dir, args.num_storage_units)
    _run_child("load", checkpoint_dir, args.num_storage_units)
    print(
        "PASS: TQ checkpoint survived a parent rename and fresh process",
        flush=True,
    )


if __name__ == "__main__":
    main()
