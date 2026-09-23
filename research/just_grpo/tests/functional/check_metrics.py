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
"""Verify native GRPO TensorBoard metrics from a three-step Sudoku smoke."""

import math
import sys
from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def check(path: Path) -> None:
    events = EventAccumulator(str(path), size_guidance={"scalars": 0}).Reload()

    def values(tag):
        return {point.step: point.value for point in events.Scalars(tag)}

    rewards = values("train/reward")
    assert len(rewards) >= 3, "Require at least three train steps"
    for tag in (
        "train/reward",
        "train/grad_norm",
        "train/gen_kl_error",
        "train/kl_penalty",
        "train/loss",
        "train/lr",
        "train/global_valid_toks",
    ):
        metric = values(tag)
        assert set(rewards) <= set(metric), tag
        assert all(math.isfinite(metric[step]) for step in rewards), tag
    assert all(0 <= value <= 1 for value in rewards.values())
    # The k3 estimate can round just below zero for identical distributions.
    assert all(-1e-7 <= value < 0.05 for value in values("train/gen_kl_error").values())
    # Equal rewards produce zero advantages and can legitimately give no update.
    gradients = values("train/grad_norm")
    assert all(value >= 0 for value in gradients.values())
    assert any(value > 0 for value in gradients.values()), (
        "No nonzero training gradient"
    )
    for tag in (
        "train/lr",
        "train/global_valid_toks",
        "timing/train/total_step_time",
        "timing/train/policy_training",
    ):
        assert all(value > 0 for value in values(tag).values()), tag
    validation = values("validation/accuracy")
    assert 0 in validation and max(rewards) in validation
    assert all(0 <= value <= 1 for value in validation.values())
    print(
        {
            "reward": rewards,
            "gen_kl_error": values("train/gen_kl_error"),
            "kl_penalty": values("train/kl_penalty"),
            "validation": validation,
        }
    )
    print(
        "PASS: upstream GRPO metrics, three updates, finite gradients, matched distributions, Sudoku validation"
    )


if __name__ == "__main__":
    check(Path(sys.argv[1]))
