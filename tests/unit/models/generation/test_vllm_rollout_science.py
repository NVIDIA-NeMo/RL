# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from unittest.mock import MagicMock

import pytest

from nemo_rl.models.generation.vllm.vllm_generation import VllmGeneration


def _generation_with_counter_snapshots(
    *snapshots: dict[str | tuple[str, int], float],
) -> VllmGeneration:
    generation = object.__new__(VllmGeneration)
    generation._rollout_science_snapshot = None
    generation._get_raw_spec_counters = MagicMock(side_effect=snapshots)
    return generation


def test_rollout_science_reports_one_batch_delta_without_touching_step_snapshot() -> (
    None
):
    generation = _generation_with_counter_snapshots(
        {
            "vllm:spec_decode_num_drafts": 10.0,
            "vllm:spec_decode_num_draft_tokens": 50.0,
            "vllm:spec_decode_num_accepted_tokens": 30.0,
        },
        {
            "vllm:spec_decode_num_drafts": 12.0,
            "vllm:spec_decode_num_draft_tokens": 60.0,
            "vllm:spec_decode_num_accepted_tokens": 36.0,
        },
    )
    generation._step_metrics_snapshot = {"sentinel": 1.0}

    generation.begin_rollout_science()
    metrics = generation.finish_rollout_science()

    assert metrics["vllm/spec_num_draft_tokens"] == 10.0
    assert metrics["vllm/spec_num_accepted_tokens"] == 6.0
    assert generation._step_metrics_snapshot == {"sentinel": 1.0}
    assert generation._rollout_science_snapshot is None


def test_rollout_science_rejects_overlapping_or_unopened_capture() -> None:
    generation = _generation_with_counter_snapshots({})
    generation.begin_rollout_science()

    with pytest.raises(RuntimeError, match="already open"):
        generation.begin_rollout_science()

    generation.cancel_rollout_science()
    with pytest.raises(RuntimeError, match="not open"):
        generation.finish_rollout_science()


def test_rollout_science_cancel_clears_failed_batch_snapshot() -> None:
    generation = _generation_with_counter_snapshots({})
    generation.begin_rollout_science()

    generation.cancel_rollout_science()

    assert generation._rollout_science_snapshot is None
