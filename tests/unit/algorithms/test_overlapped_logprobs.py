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

import copy
from unittest.mock import MagicMock, patch

import pytest
import torch

from nemo_rl.algorithms.async_utils import OverlappedLogprobs
from nemo_rl.algorithms.async_utils.replay_buffer import ReplayBufferImpl
from nemo_rl.algorithms.grpo import add_grpo_token_loss_masks_and_generation_logprobs
from nemo_rl.data.llm_message_utils import batched_message_log_to_flat_message
from nemo_rl.distributed.batched_data_dict import BatchedDataDict


class FakePolicy:
    """Logprobs are a function of each token, so any batching gives the same rows."""

    def __init__(self, dp_size: int):
        self.sharding_annotations = MagicMock(get_axis_size=lambda axis: dp_size)
        self.dp_size = dp_size
        self.calls: list[int] = []

    def prepare_for_lp_inference(self):
        pass

    def get_logprobs(self, data, timer=None):
        assert data.size % self.dp_size == 0
        self.calls.append(data.size)
        return {"logprobs": data["input_ids"].float() / 10}


def _build_batch(rows: BatchedDataDict) -> BatchedDataDict:
    add_grpo_token_loss_masks_and_generation_logprobs(rows["message_log"])
    flat_messages, input_lengths = batched_message_log_to_flat_message(
        rows["message_log"],
        pad_value_dict={"token_ids": 0},
        make_sequence_length_divisible_by=4,
    )
    return BatchedDataDict(
        {
            "input_ids": flat_messages["token_ids"],
            "input_lengths": input_lengths,
            "generation_logprobs": flat_messages["generation_logprobs"],
        }
    )


def _group(lengths: list[int], offset: int = 0) -> dict:
    message_logs = [
        [{"role": "user", "token_ids": torch.arange(1, length + 1) + offset}]
        for length in lengths
    ]
    return {"batch": BatchedDataDict({"message_log": message_logs})}


@pytest.fixture
def buffer():
    return ReplayBufferImpl(max_size=10, drop_incomplete_targets_on_restore=False)


@pytest.fixture(autouse=True)
def ray_get_copies():
    # Like a Ray actor call, peek hands back copies of the buffered groups.
    with patch(
        "nemo_rl.algorithms.async_utils.overlapped_logprobs.ray.get",
        side_effect=copy.deepcopy,
    ):
        yield


def _overlapped(policy, buffer) -> OverlappedLogprobs:
    remote_buffer = MagicMock()
    remote_buffer.peek.remote = buffer.peek
    return OverlappedLogprobs(policy, remote_buffer, _build_batch)


def _train_data(buffer, reverse: bool = False) -> BatchedDataDict:
    trajectories = buffer.sample(
        num_prompt_groups=3, current_weight_version=0, max_age_steps=1
    )["trajectories"]
    rows = BatchedDataDict.from_batches([t["batch"] for t in trajectories])
    if reverse:
        rows["message_log"].reverse()
    return _build_batch(rows)


@pytest.mark.parametrize("case", ["in_order", "reordered", "hash_collision"])
def test_overlapped_logprobs_match_full_batch(buffer, case):
    """Logprobs computed during the wait plus the rest equal one full-batch call."""
    policy = FakePolicy(dp_size=2)
    overlapped = _overlapped(policy, buffer)

    buffer.add(_group([3, 5, 2]), weight_version=0, target_weight_version=0)
    buffer.add(_group([4, 1]), weight_version=0, target_weight_version=0)
    overlapped.compute_arrived(weight_version=0)
    assert policy.calls == [4]  # 5 rows arrived; one is held for dp_size=2
    buffer.add(_group([9, 6, 7]), weight_version=0, target_weight_version=0)

    train_data = _train_data(buffer, reverse=case == "reordered")
    expected = policy.get_logprobs(train_data)["logprobs"]
    if case == "hash_collision":
        # Same hash, different tokens: nothing is reused.
        overlapped._computed = {
            key: (ids + 1, row_logprobs)
            for key, (ids, row_logprobs) in overlapped._computed.items()
        }
    logprobs, num_overlapped_rows = overlapped.get_logprobs(train_data, timer=None)

    assert torch.equal(logprobs, expected)
    assert num_overlapped_rows == (0 if case == "hash_collision" else 4)


def test_rest_is_rounded_up_to_dp_size(buffer):
    """Rows left for after the wait are padded to a dp-divisible call."""
    policy = FakePolicy(dp_size=2)
    overlapped = _overlapped(policy, buffer)

    # Identical responses are common in a GRPO group; one computed row covers
    # every copy, so an odd number of rows can be left to compute.
    buffer.add(_group([2, 2]), weight_version=0, target_weight_version=0)
    overlapped.compute_arrived(weight_version=0)
    buffer.add(_group([2, 5, 6, 7]), weight_version=0, target_weight_version=0)
    buffer.add(_group([8, 9]), weight_version=0, target_weight_version=0)

    train_data = _train_data(buffer)
    policy.calls.clear()
    logprobs, num_overlapped_rows = overlapped.get_logprobs(train_data, timer=None)

    assert num_overlapped_rows == 3
    assert policy.calls == [6]  # 5 missing rows plus one reused row
    assert torch.equal(logprobs, train_data["input_ids"].float() / 10)


def test_only_the_awaited_step_is_computed(buffer):
    """Groups for other steps are skipped, and a new step starts fresh."""
    policy = FakePolicy(dp_size=1)
    overlapped = _overlapped(policy, buffer)

    buffer.add(_group([2, 3]), weight_version=0, target_weight_version=1)
    overlapped.compute_arrived(weight_version=0)
    assert policy.calls == []

    overlapped.compute_arrived(weight_version=1)
    assert policy.calls == [2]
    overlapped.compute_arrived(weight_version=1)  # nothing new arrived
    assert policy.calls == [2]
