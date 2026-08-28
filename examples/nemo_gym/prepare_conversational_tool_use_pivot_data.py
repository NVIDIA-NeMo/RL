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
"""Prepare train/val splits of the conversational tool-use pivot dataset.

Downloads ``nvidia/Nemotron-RL-Agentic-Conversational-Tool-Use-Pivot-v1`` (or reads a
local JSONL), sanity-checks the pivot row contract expected by the NeMo-Gym
``single_step_tool_use_with_argument_comparison`` environment, optionally filters by
profiled difficulty (``qwen_235b_info.reward_mean``), then shuffles and writes
``train.jsonl`` / ``val.jsonl`` for use with
``examples/nemo_gym/grpo_qwen3_30ba3b_thinking_stage1.yaml``.

Example:
    uv run examples/nemo_gym/prepare_conversational_tool_use_pivot_data.py \
        --output-dir datasets/conversational_tool_use_pivot --val-size 512
"""

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from huggingface_hub import hf_hub_download

EXPECTED_AGENT_REF_NAME = "single_step_tool_use_with_argument_comparison_agent"
EXPECTED_ACTION_TYPES = ("message", "function_call", "function_call_batch")
REQUIRED_KEYS = ("responses_create_params", "expected_action", "agent_ref")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-id",
        type=str,
        default="nvidia/Nemotron-RL-Agentic-Conversational-Tool-Use-Pivot-v1",
        help="HF dataset repo to download from.",
    )
    parser.add_argument(
        "--artifact",
        type=str,
        default="train.jsonl",
        help="File inside the HF dataset repo.",
    )
    parser.add_argument(
        "--local-jsonl",
        type=str,
        default=None,
        help="Skip the HF download and read this JSONL instead.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="datasets/conversational_tool_use_pivot",
        help="Directory to write train.jsonl and val.jsonl into.",
    )
    parser.add_argument(
        "--val-size",
        type=int,
        default=512,
        help="Number of rows held out for validation.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Shuffle seed.")
    parser.add_argument(
        "--reward-mean-max",
        type=float,
        default=None,
        help=(
            "Keep only rows with qwen_235b_info.reward_mean <= this value "
            "(the original tau run trained on reward_mean <= 0.6). Default keeps all rows."
        ),
    )
    parser.add_argument(
        "--max-message-ratio",
        type=float,
        default=None,
        help=(
            "Downsample expected_action.type=message rows in the TRAIN split so they "
            "make up at most this fraction of it (guards against the reward exploit "
            "where any text scores 1.0 on message-expected prompts). The val split is "
            "not touched, so val metrics stay comparable across runs."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Download, sanity-check, filter, shuffle, and split the pivot dataset."""
    args = parse_args()

    if args.local_jsonl is not None:
        source_path = Path(args.local_jsonl)
    else:
        source_path = Path(
            hf_hub_download(
                repo_id=args.repo_id, filename=args.artifact, repo_type="dataset"
            )
        )
    print(f"Reading pivot rows from {source_path}")

    kept_lines: list[str] = []
    stats: Counter = Counter()
    reward_mean_sum = 0.0
    with source_path.open() as source:
        for line_number, line in enumerate(source, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            missing_keys = [key for key in REQUIRED_KEYS if key not in row]
            if missing_keys:
                raise ValueError(
                    f"line {line_number}: missing required keys {missing_keys}"
                )

            agent_ref_name = row["agent_ref"].get("name")
            if agent_ref_name != EXPECTED_AGENT_REF_NAME:
                stats["unexpected_agent_ref", agent_ref_name] += 1

            action_type = row["expected_action"].get("type")
            if action_type not in EXPECTED_ACTION_TYPES:
                raise ValueError(
                    f"line {line_number}: unexpected expected_action type {action_type!r}"
                )
            stats["expected_action", action_type] += 1

            reward_mean = (row.get("qwen_235b_info") or {}).get("reward_mean")
            if reward_mean is not None:
                reward_mean_sum += reward_mean
                if (
                    args.reward_mean_max is not None
                    and reward_mean > args.reward_mean_max
                ):
                    stats["filtered_by_reward_mean"] += 1
                    continue
            kept_lines.append((line, action_type))

    total_kept = len(kept_lines)
    if total_kept <= args.val_size:
        raise ValueError(
            f"Only {total_kept} rows kept; cannot hold out {args.val_size} for validation"
        )

    random.Random(args.seed).shuffle(kept_lines)
    val_lines = [line for line, _ in kept_lines[: args.val_size]]
    train_pairs = kept_lines[args.val_size :]

    if args.max_message_ratio is not None:
        rng = random.Random(args.seed + 1)
        fc_pairs = [p for p in train_pairs if p[1] != "message"]
        msg_pairs = [p for p in train_pairs if p[1] == "message"]
        max_msgs = int(len(fc_pairs) * args.max_message_ratio / (1.0 - args.max_message_ratio))
        if len(msg_pairs) > max_msgs:
            msg_pairs = rng.sample(msg_pairs, max_msgs)
        train_pairs = fc_pairs + msg_pairs
        rng.shuffle(train_pairs)
        print(
            f"message downsampling: kept {len(msg_pairs)} message rows "
            f"({len(msg_pairs) / max(len(train_pairs), 1):.1%} of train)"
        )
    train_lines = [line for line, _ in train_pairs]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train.jsonl"
    val_path = output_dir / "val.jsonl"
    train_path.write_text("\n".join(train_lines) + "\n")
    val_path.write_text("\n".join(val_lines) + "\n")

    total_seen = total_kept + stats["filtered_by_reward_mean"]
    print(
        f"Rows read: {total_seen}, kept: {total_kept} "
        f"(filtered by reward_mean: {stats['filtered_by_reward_mean']})"
    )
    print(
        f"Mean profiled reward_mean over all rows: {reward_mean_sum / max(total_seen, 1):.4f}"
    )
    for stat_key, count in sorted(stats.items(), key=str):
        if not isinstance(stat_key, tuple):
            continue
        key, value = stat_key
        if key == "expected_action":
            print(f"expected_action={value}: {count}")
        elif key == "unexpected_agent_ref":
            print(f"WARNING: unexpected agent_ref name {value!r}: {count} rows")
    print(f"Wrote {len(train_lines)} train rows -> {train_path}")
    print(f"Wrote {len(val_lines)} val rows -> {val_path}")
    print(
        "Full contract validation (from the Gym checkout):\n"
        f"  uv run --no-sync python 3rdparty/Gym-workspace/Gym/.claude/skills/nemo-gym-pivot-datasets/"
        f"scripts/validate_pivot_dataset.py --path {train_path} "
        f"--agent-ref {EXPECTED_AGENT_REF_NAME} --gym-repo 3rdparty/Gym-workspace/Gym"
    )


if __name__ == "__main__":
    main()
