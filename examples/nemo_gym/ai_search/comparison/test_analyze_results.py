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

from pathlib import Path

import pytest

from analyze_results import parse_nemo_log, parse_verl_log


def test_parse_verl_log_uses_model_tokens_and_excludes_checkpoint(
    tmp_path: Path,
) -> None:
    log_path = tmp_path / "verl.log"
    log_path.write_text(
        "step:2 - state_tokens/total:120.000 - response_length/mean:50.000 - "
        "critic/score/mean:0.500 - env/search_calls:8.000 - timing_s/search:1.000 - "
        "timing_s/gen:11.000 - timing_s/policy_logprob:2.000 - timing_s/ref:3.000 - "
        "timing_s/reward:0.100 - timing_s/advantage:0.200 - "
        "timing_s/update_actor:4.000 - timing_s/save_checkpoint:5.000 - "
        "timing_s/step:30.000\n",
        encoding="utf-8",
    )

    [step] = parse_verl_log(log_path, samples_per_step=4)

    assert step["core_step_s"] == 25.0
    assert step["generated_tokens"] == 120.0
    assert step["generated_tokens_per_sample"] == 30.0
    assert step["response_tokens_including_observations"] == 200.0
    assert step["stages_s"]["generation"] == 10.0
    assert step["checkpoint_s"] == 5.0


def test_parse_nemo_log_ignores_nested_logprob_aggregate(tmp_path: Path) -> None:
    log_path = tmp_path / "nemo.log"
    log_path.write_text(
        "========================= Step 2/4 =========================\n"
        "  • Avg Reward: 0.7500\n"
        "  • Mean Generation Length: 10.0000\n"
        "  • Total step time: 20.00s\n"
        "  • checkpointing: 2.00s (10.0%)\n"
        "  • prepare_for_generation/total: 3.00s (15.0%)\n"
        "  • generation: 4.00s (20.0%)\n"
        "  • logprob_inference_prep: 1.00s (5.0%)\n"
        "  • policy_and_reference_logprobs: 5.00s (25.0%)\n"
        "  • policy_logprobs: 2.00s (10.0%)\n"
        "  • reference_logprobs: 3.00s (15.0%)\n"
        "  • training_prep: 1.00s (5.0%)\n"
        "  • policy_training: 2.00s (10.0%)\n"
        "  • reward_calculation: 0.10s (0.5%)\n"
        "  • advantage_calculation: 0.20s (1.0%)\n",
        encoding="utf-8",
    )

    [step] = parse_nemo_log(log_path, samples_per_step=4)

    assert step["core_step_s"] == 18.0
    assert step["generated_tokens"] == 40.0
    assert step["generated_tokens_per_sample"] == 10.0
    assert step["stages_s"]["policy_logprob"] == 2.0
    assert step["stages_s"]["reference_logprob"] == 3.0
    assert step["stages_s"]["other"] == pytest.approx(1.7)
