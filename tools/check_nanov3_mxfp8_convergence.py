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

"""Check token-level generation/policy agreement in a short GRPO run."""

import argparse
import json
import math
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("log_dir", type=Path)
    parser.add_argument("--max-avg-prob-mult-error", type=float, default=1.10)
    parser.add_argument("--expected-steps", type=int, required=True)
    parser.add_argument("--require-learning-signal", action="store_true")
    return parser.parse_args()


def load_step(
    path: Path,
) -> tuple[int, float, float, float, float, float, float, float, float]:
    abs_diffs: list[float] = []
    signed_diffs: list[float] = []
    rewards: list[float] = []
    abs_advantages: list[float] = []
    for line in path.read_text().splitlines():
        row = json.loads(line)
        rewards.extend(float(value) for value in row["rewards"])
        for advantages, mask in zip(row["advantages"], row["token_loss_mask"]):
            abs_advantages.extend(
                abs(float(value)) for value, valid in zip(advantages, mask) if valid
            )
        generation = row["generation_logprobs"][0]
        policy = row["prev_logprobs"][0]
        mask = row["token_loss_mask"][0]
        abs_diffs.extend(
            abs(float(gen) - float(pol))
            for gen, pol, valid in zip(generation, policy, mask)
            if valid
        )
        signed_diffs.extend(
            float(pol) - float(gen)
            for gen, pol, valid in zip(generation, policy, mask)
            if valid
        )
    if not abs_diffs:
        raise RuntimeError(f"No scored tokens found in {path}.")
    return (
        len(abs_diffs),
        sum(abs_diffs) / len(abs_diffs),
        max(abs_diffs),
        sum(math.exp(value) for value in abs_diffs) / len(abs_diffs),
        math.exp(max(abs_diffs)),
        sum(signed_diffs) / len(signed_diffs),
        sum(math.exp(value) for value in signed_diffs) / len(signed_diffs),
        sum(rewards) / len(rewards),
        max(abs_advantages, default=0.0),
    )


def main() -> None:
    args = parse_args()
    step_files = sorted(args.log_dir.glob("train_data_step*.jsonl"))
    if len(step_files) != args.expected_steps:
        raise RuntimeError(
            f"Expected {args.expected_steps} step files in {args.log_dir}, "
            f"found {len(step_files)}."
        )

    for step, path in enumerate(step_files, start=1):
        (
            tokens,
            mean_abs_diff,
            max_abs_diff,
            avg_multiplier,
            max_multiplier,
            mean_signed_diff,
            mean_importance_ratio,
            mean_reward,
            max_abs_advantage,
        ) = load_step(path)
        status = "PASS" if avg_multiplier <= args.max_avg_prob_mult_error else "FAIL"
        print(
            f"NRL_NANOV3_MXFP8_CONVERGENCE_LOGPROB: {status} step={step} "
            f"tokens={tokens} mean_abs_diff={mean_abs_diff:.6f} "
            f"max_abs_diff={max_abs_diff:.6f} "
            f"avg_prob_mult_error={avg_multiplier:.6f} "
            f"max_prob_mult_error={max_multiplier:.6f} "
            f"mean_signed_logprob_diff={mean_signed_diff:.6f} "
            f"mean_importance_ratio={mean_importance_ratio:.6f} "
            f"mean_reward={mean_reward:.6f} "
            f"max_abs_advantage={max_abs_advantage:.6f}",
            flush=True,
        )
        if status == "FAIL":
            raise RuntimeError(
                f"Step {step} average probability multiplier "
                f"{avg_multiplier:.6f} exceeds "
                f"{args.max_avg_prob_mult_error:.6f}."
            )
        if args.require_learning_signal and max_abs_advantage == 0.0:
            raise RuntimeError(
                f"Step {step} has no nonzero GRPO advantage, so it cannot "
                "validate a production optimizer update."
            )


if __name__ == "__main__":
    main()
