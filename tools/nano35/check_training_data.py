# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check the resolved recipe and read all SWE rows through the training loader."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Any

from omegaconf import OmegaConf

from nemo_rl.utils.config import load_config, register_omegaconf_resolvers

if TYPE_CHECKING:
    from nemo_rl.data import DataConfig


def sha256(path: Path) -> str:
    """Hash one recorded input without loading it into memory."""
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def validate_data_config(data: DataConfig, *, expected_path: Path) -> None:
    """Reject inherited math datasets before any dataset download or GPU setup."""
    if data["shuffle"] or data["use_multiple_dataloader"]:
        raise ValueError("The SWE recipe requires one ordered training dataset")
    for split in ("train", "validation"):
        entries = data[split]
        if isinstance(entries, dict):
            entries = [entries]
        if not isinstance(entries, list) or len(entries) != 1:
            raise ValueError(f"Expected one SWE {split} dataset")
        effective = {**data["default"], **entries[0]}
        for key, expected in (
            ("dataset_name", "NemoGymDataset"),
            ("processor", "nemo_gym_data_processor"),
            ("env_name", "nemo_gym"),
        ):
            if effective.get(key) != expected:
                raise ValueError(
                    f"data.{split}.{key} must be {expected}; got {effective.get(key)!r}"
                )
        if Path(effective["data_path"]).resolve() != expected_path.resolve():
            raise ValueError(f"data.{split}.data_path differs from the recorded input")
        if effective.get("split_validation_size") not in (None, 0):
            raise ValueError(f"data.{split} must not split the 403 SWE rows")


def check(config_path: Path, *, data_path: Path, image_sha256: str) -> dict[str, Any]:
    """Use the real tokenizer, processors, collator and worker-based dataloader."""
    register_omegaconf_resolvers()
    config_dict = OmegaConf.to_container(load_config(config_path), resolve=True)
    validate_data_config(config_dict["data"], expected_path=data_path)

    # Heavy runtime dependencies are needed only inside the validated image;
    # the configuration guard can also run in a small host-side test environment.
    from torchdata.stateful_dataloader import StatefulDataLoader

    from nemo_rl.algorithms.grpo import MasterConfig
    from nemo_rl.algorithms.utils import get_tokenizer
    from nemo_rl.data.collate_fn import rl_collate_fn
    from nemo_rl.data.utils import setup_response_data

    config = MasterConfig(**config_dict)
    raw_rows = [
        json.loads(line) for line in data_path.read_text().splitlines() if line.strip()
    ]
    if len(raw_rows) != 403:
        raise ValueError(f"Expected 403 SWE rows, got {len(raw_rows)}")
    if any(row["agent_ref"]["name"] != "swe_agents_train" for row in raw_rows):
        raise ValueError("Every input row must retain the SWE training agent route")
    tokenizer = get_tokenizer(config.policy["tokenizer"])
    train, validation = setup_response_data(
        tokenizer, copy.deepcopy(config.data), env_configs=None
    )
    report: dict[str, Any] = {
        "status": "passed",
        "image_sha256": image_sha256,
        "recipe_sha256": sha256(config_path),
        "data_sha256": sha256(data_path),
        "data_path": str(data_path.resolve()),
        "data_rows": len(raw_rows),
        "num_workers": config.data["num_workers"],
        "batch_size": config.grpo.num_prompts_per_step,
        "splits": {},
    }
    for name, dataset in (("train", train), ("validation", validation)):
        if (
            dataset is None
            or isinstance(dataset, dict)
            or len(dataset) != len(raw_rows)
        ):
            raise ValueError(f"{name} loader did not retain all 403 SWE rows")
        loader = StatefulDataLoader(
            dataset,
            batch_size=config.grpo.num_prompts_per_step,
            shuffle=False,
            num_workers=config.data["num_workers"],
            collate_fn=rl_collate_fn,
            # Include the final partial batch so every source row is checked.
            drop_last=False,
        )
        cursor = 0
        batches = 0
        for batch in loader:
            for index, row in zip(batch["idx"], batch["extra_env_info"], strict=True):
                if index != cursor or row != raw_rows[cursor]:
                    raise ValueError(
                        f"{name} row {cursor} changed content, order or index"
                    )
                cursor += 1
            batches += 1
        if cursor != len(raw_rows):
            raise ValueError(f"{name} dataloader yielded only {cursor} rows")
        report["splits"][name] = {
            "rows": cursor,
            "batches": batches,
            "raw_payload_and_order_preserved": True,
        }
    return report


def main() -> None:
    """Write a success report only after both complete data passes succeed."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = check(
        args.config,
        data_path=Path(os.environ["NANO35_DATA"]),
        image_sha256=os.environ["NANO35_IMAGE_SHA256"],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as output:
        output.write(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
