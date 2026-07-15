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

import argparse
import os

import pandas as pd
import wandb


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--org", type=str, default="nvidia")
    parser.add_argument("--project", type=str, default="nemo-rl")
    parser.add_argument("--uid", type=str, required=True)
    parser.add_argument("--at-step", type=int, default=5)
    parser.add_argument("--average-steps", type=int, default=5)
    return parser.parse_args()


def main():
    args = parse_args()
    assert os.environ.get("WANDB_API_KEY") is not None, "WANDB_API_KEY is not set"

    keys = [
        "train/mean_total_tokens_per_sample",
        "timing/train/total_step_time",
        "timing/train/generation",
        "timing/train/exposed_generation",
        "timing/train/policy_training",
        "timing/train/policy_and_reference_logprobs",
        "timing/train/weight_sync",
        "timing/train/prepare_for_generation/total",
        "timing/train/prepare_for_generation/transfer_and_update_weights",
        "performance/tokens_per_sec_per_gpu",
        "performance/generation_tokens_per_sec_per_gpu",
        "performance/training_worker_group_tokens_per_sec_per_gpu",
        "performance/policy_training_tokens_per_sec_per_gpu",
        "performance/policy_and_reference_logprobs_tokens_per_sec_per_gpu",
        "performance/train_flops_per_gpu",
        "performance/train_fp_utilization",
        "timing/train/checkpointing",
    ]
    api = wandb.Api()
    run = api.run(f"{args.org}/{args.project}/{args.uid}")
    min_step = args.at_step - args.average_steps // 2
    max_step = args.at_step + args.average_steps // 2 - 1 + (args.average_steps % 2)
    raw_history = run.history()

    # Newer wandb versions (or environments without pandas at import time) may
    # return a list of per-step dicts instead of a DataFrame.
    if not isinstance(raw_history, pd.DataFrame):
        raw_history = pd.DataFrame(list(raw_history))
    missing_keys = [key for key in keys if key not in raw_history.columns]
    if missing_keys:
        print(f"Warning: skipping keys not found in history: {missing_keys}")
    keys = [key for key in keys if key in raw_history.columns]
    history = raw_history.loc[min_step:max_step, keys]

    # get average of the history
    average_history = history.mean(axis=0)
    history_at_step = history.loc[args.at_step]
    print("Average history:")
    print(average_history)
    print("History at step:")
    print(history_at_step)


if __name__ == "__main__":
    main()
