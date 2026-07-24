# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import pytest
import torch

from nemo_rl.distributed.model_utils import (
    ChunkedDistributedEntropy,
    ChunkedDistributedGatherLogprob,
    ChunkedDistributedHiddenStatesToLogprobs,
    ChunkedDistributedLogprob,
    DistributedCrossEntropy,
    DistributedLogprob,
    _compute_distributed_log_softmax_with_grad,
    _compute_distributed_softmax,
    _get_world_size_or_1,
)


def _mock_distributed_world_size(monkeypatch: pytest.MonkeyPatch, world_size: int):
    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(
        torch.distributed, "get_world_size", lambda group=None: world_size
    )


def _raise_if_called(*args, **kwargs):
    raise AssertionError("collective should not be called for world size 1")


def _forbid_collectives(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(torch.distributed, "all_reduce", _raise_if_called)
    monkeypatch.setattr(torch.distributed, "reduce_scatter", _raise_if_called)
    monkeypatch.setattr(torch.distributed, "all_gather", _raise_if_called)
    monkeypatch.setattr(torch.distributed.nn.functional, "all_reduce", _raise_if_called)


def test_world_size_helper_passes_none_to_default_group(monkeypatch):
    seen_groups = []

    monkeypatch.setattr(torch.distributed, "is_available", lambda: True)
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)

    def get_world_size(group=None):
        seen_groups.append(group)
        return 1

    monkeypatch.setattr(torch.distributed, "get_world_size", get_world_size)

    assert _get_world_size_or_1(None) == 1
    assert seen_groups == [None]


def test_distributed_log_softmax_skips_collectives_for_world_size_one(monkeypatch):
    _mock_distributed_world_size(monkeypatch, world_size=1)
    _forbid_collectives(monkeypatch)

    logits = torch.randn(2, 3, 5, requires_grad=True)

    actual = _compute_distributed_log_softmax_with_grad(logits, group=None)
    expected = torch.log_softmax(logits, dim=-1)

    torch.testing.assert_close(actual, expected)
    actual.sum().backward()
    assert logits.grad is not None


def test_distributed_log_softmax_keeps_collectives_for_larger_world_size(
    monkeypatch,
):
    _mock_distributed_world_size(monkeypatch, world_size=2)
    calls = {"all_reduce": 0, "functional_all_reduce": 0}

    def all_reduce(tensor, op=None, group=None):
        calls["all_reduce"] += 1

    def functional_all_reduce(tensor, op=None, group=None):
        calls["functional_all_reduce"] += 1
        return tensor

    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)
    monkeypatch.setattr(
        torch.distributed.nn.functional, "all_reduce", functional_all_reduce
    )

    logits = torch.randn(2, 3, 5, requires_grad=True)
    _compute_distributed_log_softmax_with_grad(logits, group=object())

    assert calls == {"all_reduce": 1, "functional_all_reduce": 1}


def test_distributed_softmax_skips_collectives_for_world_size_one(monkeypatch):
    _mock_distributed_world_size(monkeypatch, world_size=1)
    _forbid_collectives(monkeypatch)

    logits = torch.randn(2, 3, 5)

    actual = _compute_distributed_softmax(logits.clone(), group=None)
    expected = torch.softmax(logits.float(), dim=-1)

    torch.testing.assert_close(actual, expected)


def test_selected_logprob_and_cross_entropy_skip_collectives_for_world_size_one(
    monkeypatch,
):
    _mock_distributed_world_size(monkeypatch, world_size=1)
    _forbid_collectives(monkeypatch)

    logits = torch.randn(2, 3, 5)
    target = torch.tensor([[0, 2, 4], [1, 3, 0]])

    actual_logprobs = DistributedLogprob.apply(
        logits, target, 0, logits.shape[-1], None, True
    )
    expected_logprobs = torch.gather(
        torch.log_softmax(logits.float(), dim=-1), -1, target.unsqueeze(-1)
    ).squeeze(-1)
    torch.testing.assert_close(actual_logprobs, expected_logprobs)

    student_logits = torch.randn(2, 3, 5)
    target_logits = torch.randn(2, 3, 5)
    actual_ce = DistributedCrossEntropy.apply(student_logits, target_logits, None, True)
    expected_ce = -(
        torch.softmax(target_logits.float(), dim=-1)
        * torch.log_softmax(student_logits.float(), dim=-1)
    ).sum(dim=-1)
    torch.testing.assert_close(actual_ce, expected_ce)


def test_chunked_distributed_logprob_skips_collectives_for_world_size_one(
    monkeypatch,
):
    _mock_distributed_world_size(monkeypatch, world_size=1)
    _forbid_collectives(monkeypatch)

    logits = torch.randn(2, 4, 5, requires_grad=True)
    target = torch.tensor([[0, 2, 4, 1], [1, 3, 0, 2]])

    actual = ChunkedDistributedLogprob.apply(
        logits, target, 0, logits.shape[-1], 2, None, False
    )
    expected = torch.gather(
        torch.log_softmax(logits.float(), dim=-1), -1, target.unsqueeze(-1)
    ).squeeze(-1)
    torch.testing.assert_close(actual, expected)

    actual.sum().backward()
    assert logits.grad is not None


def test_chunked_gather_and_entropy_skip_collectives_for_world_size_one(monkeypatch):
    _mock_distributed_world_size(monkeypatch, world_size=1)
    _forbid_collectives(monkeypatch)

    logits = torch.randn(2, 4, 6, requires_grad=True)
    global_indices = torch.tensor(
        [
            [[0, 2], [1, 3], [4, 5], [0, 1]],
            [[5, 4], [3, 2], [1, 0], [2, 4]],
        ]
    )

    gathered = ChunkedDistributedGatherLogprob.apply(
        logits, global_indices, 0, logits.shape[-1], 2, None, True
    )
    expected_gathered = torch.gather(
        torch.log_softmax(logits.float(), dim=-1), -1, global_indices
    )
    torch.testing.assert_close(gathered, expected_gathered)

    entropy = ChunkedDistributedEntropy.apply(logits, 2, None, False)
    log_probs = torch.log_softmax(logits.float(), dim=-1)
    expected_entropy = (log_probs.exp() * log_probs).sum(dim=-1)
    torch.testing.assert_close(entropy, expected_entropy)

    entropy.sum().backward()
    assert logits.grad is not None


def test_hidden_state_logprob_skips_all_gather_all_reduce_and_reduce_scatter_tp1(
    monkeypatch,
):
    _mock_distributed_world_size(monkeypatch, world_size=1)
    _forbid_collectives(monkeypatch)

    hidden_states = torch.randn(3, 2, 4, requires_grad=True)
    output_weight = torch.randn(5, 4, requires_grad=True)
    target = torch.tensor([[0, 2, 4], [1, 3, 0]])

    actual = ChunkedDistributedHiddenStatesToLogprobs.apply(
        hidden_states,
        target,
        output_weight,
        0,
        output_weight.shape[0],
        2,
        None,
        False,
    )
    logits = torch.matmul(hidden_states, output_weight.T).float().transpose(0, 1)
    expected = torch.gather(
        torch.log_softmax(logits, dim=-1), -1, target.unsqueeze(-1)
    ).squeeze(-1)
    torch.testing.assert_close(actual, expected)

    actual.sum().backward()

    assert hidden_states.grad is not None
    assert output_weight.grad is not None


def test_hidden_state_logprob_keeps_collectives_for_world_size_two(monkeypatch):
    _mock_distributed_world_size(monkeypatch, world_size=2)
    calls = {
        "all_gather": 0,
        "all_reduce": 0,
        "functional_all_reduce": 0,
        "reduce_scatter": 0,
    }

    def all_gather(output_tensors, tensor, group=None):
        for output_tensor in output_tensors:
            output_tensor.copy_(tensor)
        calls["all_gather"] += 1

    def all_reduce(tensor, op=None, group=None):
        calls["all_reduce"] += 1

    def functional_all_reduce(tensor, op=None, group=None):
        calls["functional_all_reduce"] += 1
        return tensor

    def reduce_scatter(output, input_list, op=None, group=None):
        output.copy_(input_list[0])
        calls["reduce_scatter"] += 1

    monkeypatch.setattr(torch.distributed, "all_gather", all_gather)
    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)
    monkeypatch.setattr(torch.distributed, "reduce_scatter", reduce_scatter)
    monkeypatch.setattr(
        torch.distributed.nn.functional, "all_reduce", functional_all_reduce
    )

    hidden_states = torch.randn(2, 2, 4, requires_grad=True)
    output_weight = torch.randn(6, 4, requires_grad=True)
    target = torch.tensor([[0, 2, 4, 1], [1, 3, 0, 2]])

    actual = ChunkedDistributedHiddenStatesToLogprobs.apply(
        hidden_states,
        target,
        output_weight,
        0,
        output_weight.shape[0],
        2,
        object(),
        False,
    )

    actual.sum().backward()

    assert hidden_states.grad is not None
    assert output_weight.grad is not None
    assert calls["all_gather"] == 2
    assert calls["all_reduce"] >= 1
    assert calls["functional_all_reduce"] >= 1
    assert calls["reduce_scatter"] == 1
