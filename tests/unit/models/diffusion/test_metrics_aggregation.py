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
import torch

from nemo_rl.models.diffusion.flow_grpo_policy import aggregate_worker_metrics
from nemo_rl.models.diffusion.workers.flow_grpo_worker import accumulate_metrics


def test_accumulate_metrics_min_max_and_weighted_mean():
    acc: dict[str, float] = {}
    accumulate_metrics(
        acc, {"loss": torch.tensor(2.0), "ratio_min": 0.9, "ratio_max": 1.1}, 0.5
    )
    accumulate_metrics(
        acc, {"loss": torch.tensor(4.0), "ratio_min": 0.8, "ratio_max": 1.05}, 0.5
    )
    assert acc["loss"] == 3.0  # weighted mean
    assert acc["ratio_min"] == 0.8  # min, not a weighted sum
    assert acc["ratio_max"] == 1.1  # max


def test_aggregate_worker_metrics_min_max_and_mean():
    out = aggregate_worker_metrics(
        [
            {"loss": 1.0, "ratio_min": 0.9, "ratio_max": 1.2},
            {"loss": 3.0, "ratio_min": 0.7, "ratio_max": 1.0},
        ]
    )
    assert out == {"loss": 2.0, "ratio_min": 0.7, "ratio_max": 1.2}
