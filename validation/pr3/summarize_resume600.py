# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Audit the completed exploratory continuation on an allocated CPU node."""

import hashlib
import io
import json
import math
import os
import re
import statistics

import torch
from plot_comparison import REPO, STEP, committed_log_rows
from torch.distributed.checkpoint import FileSystemReader

OUT = REPO / "validation/pr3/results/resume600"
JOBS = (21331883, 21331884, 21331885, 21331930, 21509383, 21509384, 21509385)
METRICS = (
    "gen_kl_error",
    "policy_kl_error",
    "reward",
    "mean_gen_tokens_per_sample",
    "grad_norm",
    "loss",
    "approx_entropy",
)


def describe(rows: list[dict]) -> dict:
    keys = {key: f"train/{key}" for key in METRICS}
    keys["step_time"] = "timing/train/total_step_time"
    return {
        key: {
            "mean": statistics.mean(row[source] for row in rows),
            "min": min(row[source] for row in rows),
            "max": max(row[source] for row in rows),
        }
        for key, source in keys.items()
    }


def main() -> None:
    assert os.environ["USER"] == "root" and os.environ.get("SLURM_JOB_ID")
    OUT.mkdir(parents=True, exist_ok=True)
    hashes = {
        "nemo_rl/models/policy/workers/automodel_policy_worker.py": "7f59824258aafb13f5540810776dfa18c86a623856c9c89e4c6e26881259f74d",
        "tests/functional/automodel_sc_full_batch_parity.py": "ccb26bda2ad5a1408cbbd125cf2aec5a119d232ecaa1777bebbce0071c0dc399",
    }
    assert all(
        hashlib.sha256((REPO / path).read_bytes()).hexdigest() == expected
        for path, expected in hashes.items()
    )
    rows, sources, segments = [], {}, {}
    for job in JOBS:
        path = REPO / f"validation/pr3/slurm/{job}-logs/ray-driver.log"
        data = path.read_bytes()
        receipts = re.findall(
            r"train step (\d+)/(?:300|600)\s+trainer_v=(\d+)\s+lag=(\d+)",
            data.decode(),
        )
        part = committed_log_rows(path)
        assert len(part) == len(receipts)
        assert all(step == version and lag == "0" for step, version, lag in receipts)
        sources[str(path.relative_to(REPO))] = hashlib.sha256(data).hexdigest()
        rows.extend(part)
        segments[str(job)] = [int(part[0][STEP]), int(part[-1][STEP])]
    assert [row[STEP] for row in rows] == list(range(1, 601))
    assert all(row["train/weight_version_lag"] == 0 for row in rows)
    assert all(row["train/policy_train_dispatches"] == 2 for row in rows)
    assert all(row["train/global_valid_seqs"] == 32 for row in rows)
    assert all(row["train/lr"] == 1e-6 for row in rows[300:])
    assert all(math.isfinite(value) for row in rows for value in row.values())
    summary = {
        "analyzed_inside_job": int(os.environ["SLURM_JOB_ID"]),
        "source_log_sha256": sources,
        "unchanged_training_source_sha256": hashes,
        "segments": segments,
        "committed_updates": 600,
        "all_steps_unique_and_contiguous": True,
        "all_lag_zero": True,
        "all_two_dispatches": True,
        "all_32_valid_sequences": True,
        "all_continuation_lr_1e_6": True,
        "all_metrics_finite": True,
        "windows": {},
        "latest": rows[-1],
        "qualification_passed": False,
        "scope": "Exploratory single-seed continuation; original BF16 1% gate remains FAILED.",
    }
    for first, last in [
        (1, 100),
        (101, 200),
        (201, 300),
        (301, 400),
        (401, 500),
        (501, 600),
        (301, 320),
        (481, 500),
        (581, 600),
    ]:
        summary["windows"][f"{first}..{last}"] = describe(rows[first - 1 : last])
    (OUT / "final-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    segment3 = {
        "segment_job": 21509385,
        "analyzed_inside_job": int(os.environ["SLURM_JOB_ID"]),
        "first_step": 504,
        "last_step": 600,
        "committed_updates": 97,
        "windows": {
            "first20": describe(rows[503:523]),
            "last20": describe(rows[-20:]),
            "segment": describe(rows[503:]),
        },
    }
    (OUT / "segment3-summary.json").write_text(json.dumps(segment3, indent=2) + "\n")

    checkpoint = (
        REPO
        / "validation/pr3/results/yuekaiz-qwen35-streaming-la0-300-exploratory/checkpoints/step_600"
    )
    base = checkpoint / "policy/optimizer/optim"
    metadata = FileSystemReader(base).read_metadata()
    steps, rates, scheduler = [], [], {}
    for index, storage in metadata.storage_data.items():
        key = index.fqn
        selected = key.startswith("sched.") or (
            key.startswith("optim.param_groups.") and key.endswith((".step", ".lr"))
        )
        if (
            not selected
            or type(metadata.state_dict_metadata[key]).__name__
            != "BytesStorageMetadata"
        ):
            continue
        assert storage.length < 65536
        with (base / storage.relative_path).open("rb") as file:
            file.seek(storage.offset)
            value = torch.load(
                io.BytesIO(file.read(storage.length)),
                map_location="cpu",
                weights_only=False,
            )
        if key.startswith("sched."):
            scheduler[key] = value
        elif key.endswith(".step"):
            steps.append(value)
        else:
            rates.append(value)
    info = json.loads((checkpoint / "training_info.json").read_text())
    assert (
        info["total_steps"]
        == info["trainer_version"]
        == info["sampler_dispatch_index"]
        == 600
    )
    assert info["consumed_samples"] == 4800 and info["current_epoch"] == 0
    assert len(steps) == len(rates) == 693
    assert all(value == 600 for value in steps) and all(
        value == 1e-6 for value in rates
    )
    assert scheduler["sched.last_epoch"] == 600
    assert len(list(base.glob("*.distcp"))) == 16
    proof = {
        "training_info": info,
        "optimizer_step_entries": len(steps),
        "all_saved_optimizer_steps_600": True,
        "optimizer_lr_entries": len(rates),
        "all_saved_optimizer_lrs_1e_6": True,
        "optimizer_shards": 16,
        "scheduler": scheduler,
        "metadata_sha256": hashlib.sha256(
            (base / ".metadata").read_bytes()
        ).hexdigest(),
        "analyzed_inside_job": int(os.environ["SLURM_JOB_ID"]),
        "scope": "Saved scalar state only; not a live per-rank raw optimizer-buffer comparison.",
    }
    (OUT / "checkpoint600-state.json").write_text(json.dumps(proof, indent=2) + "\n")
    print(
        json.dumps(
            {
                "updates": 600,
                "last100": summary["windows"]["501..600"],
                "last20": summary["windows"]["581..600"],
            }
        )
    )


if __name__ == "__main__":
    main()
