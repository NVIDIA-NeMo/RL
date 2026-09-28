# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio

import pytest
import torch

from tests.mock_stack.components import CopyRefit, Generation, Policy, Turn


def batch():
    return {
        "input_ids": torch.tensor([[1, 2, 3], [4, 5, 0]]),
        "input_lengths": torch.tensor([3, 2]),
        "token_mask": torch.tensor([[0, 1, 1], [0, 1, 0]]),
        "total_reward": torch.tensor([1.0, 2.0]),
    }


@pytest.mark.asyncio
async def test_refit_changes_logprobs_but_not_tokens():
    policy, generation = Policy(), Generation()
    refit = CopyRefit(policy, generation)
    assert refit.is_stale

    refit.sync_weights()
    turn = Turn("p1", 0, 1, 0)
    before = await generation.generate(turn)
    policy.train(batch())
    assert await generation.generate(turn) == before
    refit.sync_weights()
    after = await generation.generate(turn)
    assert (
        list(after.logprobs) == policy.logprobs(torch.tensor(after.token_ids)).tolist()
    )
    assert after.token_ids == before.token_ids
    assert after.text == before.text
    assert after.logprobs != before.logprobs
    assert after.weight_digest != before.weight_digest


@pytest.mark.parametrize(
    "policy,generation", [(object(), Generation()), (Policy(), object())]
)
def test_refit_rejects_incompatible_components(policy, generation):
    with pytest.raises(TypeError, match="export_weights and accept_weights"):
        CopyRefit(policy, generation)


@pytest.mark.asyncio
async def test_inflight_call_keeps_its_starting_weights():
    policy, generation = Policy(), Generation()
    refit = CopyRefit(policy, generation)
    refit.sync_weights()
    before = await generation.generate(Turn("p1", 0, 1, 0))
    pending = asyncio.create_task(generation.generate(Turn("p1", 0, 1, 0.1)))
    await asyncio.sleep(0.02)
    assert not pending.done(), "Timing precondition missed: generation already finished"
    policy.train(batch())
    refit.sync_weights()
    assert await pending == before


def test_checkpoint_restores_the_next_update(tmp_path):
    policy = Policy()
    policy.train(batch())
    policy.save_checkpoint(str(tmp_path), is_final_checkpoint=False)
    restored = Policy()
    restored.load_checkpoint(tmp_path)
    assert restored.steps == 1
    policy.train(batch())
    restored.train(batch())
    assert restored.steps == policy.steps == 2
    assert torch.equal(restored.export_weights(), policy.export_weights())


@pytest.mark.parametrize(
    "field", ["input_ids", "input_lengths", "token_mask", "total_reward"]
)
def test_updates_depend_on_actual_training_data(field):
    original, changed = Policy(), Policy()
    original.train(batch())
    data = batch()
    data[field].flatten()[0] += 1
    changed.train(data)
    assert not torch.equal(original.export_weights(), changed.export_weights())


def test_row_order_changes_the_update():
    original, reversed_rows = Policy(), Policy()
    original.train(batch())
    reversed_rows.train({key: value.flip(0) for key, value in batch().items()})
    assert not torch.equal(original.export_weights(), reversed_rows.export_weights())


def test_weight_exports_do_not_alias_policy_state():
    policy = Policy()
    before = policy.export_weights()
    policy.export_weights().zero_()
    assert torch.equal(before, policy.export_weights())


def test_policy_logprobs_form_a_distribution():
    torch.testing.assert_close(
        Policy().logprobs(torch.arange(257)).exp().sum(), torch.tensor(1.0)
    )


def test_refit_requires_acknowledgement():
    class RejectingGeneration(Generation):
        def accept_weights(self, weights):
            return "wrong digest"

    refit = CopyRefit(Policy(), RejectingGeneration())
    with pytest.raises(RuntimeError, match="acknowledge"):
        refit.sync_weights()
    assert refit.is_stale
