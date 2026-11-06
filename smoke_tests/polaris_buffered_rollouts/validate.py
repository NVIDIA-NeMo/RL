# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate native checkpoints, consumed responses and metrics from the GPU smoke."""

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch
from omegaconf import OmegaConf
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from nemo_rl.algorithms.rollout_buffer import BUFFER_FILENAME, RolloutBuffer
from nemo_rl.models.megatron.alignment import get_parallel_token_alignment
from nemo_rl.utils.checkpoint import CheckpointManager


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.output.resolve()
    config_path = root / "checkpoints/step_5/config.yaml"
    config = OmegaConf.load(config_path)
    assert config.grpo.rollout_batch_multiplier == 3
    assert config.grpo.omega_loss.form == "per_sequence"
    assert config.policy.train_global_batch_size == 64
    assert (
        config.policy.sequence_packing.enabled
        and config.policy.sequence_packing.fuse_loss
    )
    assert not config.policy.megatron_cfg.use_fused_linear_logprobs
    alignment = config.policy.make_sequence_length_divisible_by
    assert (
        alignment > 0
        and alignment % get_parallel_token_alignment(config.policy.megatron_cfg) == 0
    )
    assert not config.logger.wandb_enabled
    assert config.policy.generation.temperature == 1.0
    assert not config.policy.generation.vllm_cfg.enforce_eager
    assert (
        config.policy.generation.vllm_kwargs.compilation_config.cudagraph_mode
        == "FULL_AND_PIECEWISE"
    )
    texts = {
        phase: (root / f"{phase}.log").read_text() for phase in ("fresh", "resume")
    }
    assert "Timeout has been reached" in texts["fresh"]
    assert "Restored 2 buffered optimizer batches" in texts["resume"]
    assert "Max number of steps has been reached" in texts["resume"]
    assert "Step 2/" not in texts["fresh"] and "Step 1/" not in texts["resume"]
    assert texts["fresh"].count("Generating responses for batch of size 192") == 1
    assert texts["resume"].count("Generating responses for batch of size 128") == 1
    assert "Generating responses for batch of size 192" not in texts["resume"]
    assert texts["resume"].index("Step 4/") < texts["resume"].index(
        "Generating responses for batch of size 128"
    )
    for phase, text in texts.items():
        # Actual runtime evidence, not merely config flag strings.
        assert "Capturing CUDA graphs" in text or "Graph capturing finished" in text, (
            phase
        )
        assert "torch.compile" in text or "Compiling" in text, phase
    checkpoint_info = []
    for step in range(1, 6):
        checkpoint = root / "checkpoints" / f"step_{step}"
        info = json.loads((checkpoint / "training_info.json").read_text())
        assert info["total_steps"] == step and info["consumed_samples"] == 8 * step
        assert info["rollout_buffer_version"] == 1
        assert (checkpoint / "train_dataloader.pt").is_file()
        weights, optimizer = CheckpointManager.get_resume_paths(checkpoint)
        assert weights is not None and optimizer is not None, checkpoint
        # Megatron embeds optimizer shards under weights; optimizer_path is a
        # restore flag, not a separately materialized directory.
        assert any(weights.rglob("*.distcp")), checkpoint
        buffer = RolloutBuffer(8, 8, 3)
        buffer.restore(checkpoint, completed_steps=step)
        assert buffer.remaining_updates == [2, 1, 0, 1, 0][step - 1]
        checkpoint_info.append(info)
    # The resumed updates must use exactly the completions saved after step1.
    pending = torch.load(
        root / "checkpoints/step_1" / BUFFER_FILENAME, weights_only=False
    )
    data_files = {}
    for path in (root / "logs").rglob("train_data_step*.jsonl"):
        step = int(path.stem.removeprefix("train_data_step"))
        assert step not in data_files, f"Duplicate training step {step}"
        data_files[step] = path
    assert sorted(data_files) == list(range(1, 6))
    for step, start in ((2, 0), (3, 64)):
        records = [
            json.loads(line) for line in data_files[step].read_text().splitlines()
        ]
        assert len(records) == 64
        for offset, record in enumerate(records):
            row = start + offset
            tokens = torch.cat(
                [
                    message["token_ids"]
                    for message in pending["batch"]["message_log"][row]
                ]
            ).tolist()
            length = record["input_lengths"][0]
            assert record["token_ids"][0][:length] == tokens
            assert record["rewards"][0] == pending["batch"]["total_reward"][row].item()
        print(
            f"Verified all 64 checkpointed responses and rewards at update {step}",
            flush=True,
        )
    scalar_series = {}
    for path in (root / "logs").rglob("events.out.tfevents.*"):
        accumulator = EventAccumulator(str(path), size_guidance={"scalars": 0})
        accumulator.Reload()
        for tag in accumulator.Tags()["scalars"]:
            series = scalar_series.setdefault(tag, {})
            for event in accumulator.Scalars(tag):
                series[event.step] = event.value
    selected = {}
    for tag in [
        "train/loss",
        "train/grad_norm",
        "train/omega/objective",
        "train/rollout/generated_sequences",
        "train/rollout/age_updates",
    ]:
        series = scalar_series[tag]
        assert sorted(series) == list(range(1, 6)), (tag, series)
        values = [series[step] for step in range(1, 6)]
        assert all(math.isfinite(value) for value in values), tag
        selected[tag] = values
    assert all(value > 0 for value in selected["train/grad_norm"])
    assert selected["train/rollout/generated_sequences"] == [192, 0, 0, 128, 0]
    assert selected["train/rollout/age_updates"] == [0, 1, 2, 0, 1]
    receipt = {
        "passed": True,
        "updates": 5,
        "generation_calls": 2,
        "pending_responses_verified": 128,
        "metrics": selected,
        "checkpoint_info": checkpoint_info,
        "config_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
    }
    (root / "final").mkdir(exist_ok=True)
    with (root / "final/validation.json").open("x") as stream:
        json.dump(receipt, stream, indent=2)
    print(
        "Validated buffered POLARIS smoke, including native mid-buffer resume",
        flush=True,
    )


if __name__ == "__main__":
    main()
