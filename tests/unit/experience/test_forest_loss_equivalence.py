# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Same-evidence per-call versus segment loss/gradient conservation on CPU.

Uses real staging, finalization, controller advantages, policy-gradient loss,
and the PPO critic's value loss.
Generation/storage are doubles; the order-sensitive toy model does not qualify
Megatron, multimodal/router execution, or compaction versus no-compaction parity.
"""

import asyncio
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence

pytest.importorskip("nemo_gym", reason="requires the paired Gym checkout")

from nemo_gym.token_id_capture.staging.capture import ActiveCall, RolloutTokenCapture
from nemo_gym.token_id_capture.staging.records import CommitCoords

from nemo_rl.algorithms.advantage_estimator import (
    GAEConfig,
    GeneralizedAdvantageEstimator,
)
from nemo_rl.algorithms.loss import ClippedPGLossConfig, ClippedPGLossFn
from nemo_rl.algorithms.loss.loss_functions import MseValueLossConfig, MseValueLossFn
from nemo_rl.data_plane.tq_token_sink import TQTokenSink, TQTokenSource
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.experience.rollout_reassembler import RolloutReassembler
from tests.unit.experience.test_logical_owner_finalization import (
    PublicationDataPlane,
    capture_segment,
    decide_capture_input,
    finalize,
    gym_harness,
)
from tests.unit.single_controller.test_logical_advantage import (
    _batch,
    _controller,
    _ppo_controller,
)

pytestmark = pytest.mark.nemo_gym


class _CausalModel(nn.Module):
    """A tiny recurrent model: token order and segment-local state both matter."""

    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(1024, 8)
        self.recurrent = nn.GRU(8, 8, batch_first=True)
        self.head = nn.Linear(8, 1024)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        hidden, _ = self.recurrent(self.embedding(tokens[:, :-1]))
        return (
            self.head(hidden)
            .log_softmax(-1)
            .gather(-1, tokens[:, 1:].unsqueeze(-1))
            .squeeze(-1)
        )


def _actions(data: BatchedDataDict) -> dict[int, tuple[list[int], float, float]]:
    """Compare full conditioning independently of the model and loss reduction."""
    actions = {}
    for row in range(data.size):
        for position in (data["token_mask"][row] * data["sample_mask"][row]).nonzero():
            index = position.item()
            token = data["input_ids"][row, index].item()
            assert token not in actions, "sampled token owned twice in this fixture"
            actions[token] = (
                data["input_ids"][row, :index].tolist(),
                data["advantages"][row, index].item(),
                data["generation_logprobs"][row, index].item(),
            )
    return actions


def _loss_gradient(
    model: nn.Module, data: BatchedDataDict, *, denominator: int = 5
) -> tuple[torch.Tensor, torch.Tensor]:
    model = deepcopy(model)
    loss, _ = ClippedPGLossFn(
        ClippedPGLossConfig(
            reference_policy_kl_penalty=0.0, use_importance_sampling_correction=True
        )
    )(
        model(data["input_ids"]),
        data,
        global_valid_seqs=data["sample_mask"].sum(),
        global_valid_toks=torch.tensor(float(denominator)),
    )
    loss.backward()
    gradients = torch.cat(
        [parameter.grad.flatten() for parameter in model.parameters()]
    )
    assert torch.isfinite(loss) and torch.isfinite(gradients).all()
    assert gradients.norm() > 1e-8, "zero gradients cannot establish conservation"
    return loss.detach(), gradients


@pytest.mark.parametrize("dp_size", [1, 2])
@pytest.mark.parametrize("filter_owner", [False, True])
def test_packing_bins_preserve_forest_loss_accounting(
    dp_size: int, filter_owner: bool
) -> None:
    """Real bin planner + row loss; packed attention requires the GPU gate."""
    meta, data = _batch([("a", [(0.0, 2), (1.0, 2), (3.0, 1)])], padding=True)
    lengths = torch.tensor([3, 3, 2, 3, 2, 3])
    data["token_mask"][lengths == 2, 2] = 0
    if filter_owner:
        data["mask_sample"][1] = True
    ctrl = _controller(meta, data)
    asyncio.run(ctrl._advantage_stage(meta))
    batch = BatchedDataDict(
        {
            "input_ids": torch.arange(10, 28).reshape(6, 3),
            "input_lengths": lengths,
            "generation_logprobs": torch.full((6, 3), -2.0),
            "advantages": data["advantages"],
            "token_mask": data["token_mask"],
            "sample_mask": data["sample_mask"],
            "row_index": torch.arange(6),
        }
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(11)
        model = _CausalModel().double()
    with torch.no_grad():
        batch["prev_logprobs"] = nn.functional.pad(
            model(batch["input_ids"]) - 0.05, (1, 0)
        )
    shards, _ = batch.shard_by_batch_size(
        dp_size,
        batch_size=6,
        sequence_packing_args={
            "algorithm": "modified_first_fit_decreasing",
            "input_key": "input_ids",
            "input_lengths_key": "input_lengths",
            "sequence_length_pad_multiple": 1,
            "max_tokens_per_microbatch": 8,
        },
    )
    microbatches = [
        mb
        for shard in shards
        for mb in shard.make_microbatch_iterator_for_packable_sequences()
    ]
    seen = []
    for mb in microbatches:
        indices = mb["row_index"]
        seen.extend(indices.tolist())
        for field in batch:
            torch.testing.assert_close(mb[field], batch[field][indices])
    assert sorted(seen) == list(range(6)), "packing must retain each row exactly once"
    assert any(mb.size > 1 for mb in microbatches)
    denominator = int(
        (batch["token_mask"][:, 1:] * batch["sample_mask"][:, None]).sum()
    )
    reference_loss, reference_grad = _loss_gradient(
        model, batch, denominator=denominator
    )
    packed_loss, packed_grad = (
        torch.zeros_like(reference_loss),
        torch.zeros_like(reference_grad),
    )
    for mb in microbatches:
        if not (mb["token_mask"][:, 1:] * mb["sample_mask"][:, None]).any():
            continue
        loss, grad = _loss_gradient(model, mb, denominator=denominator)
        packed_loss += loss
        packed_grad += grad
    torch.testing.assert_close(packed_loss, reference_loss)
    torch.testing.assert_close(packed_grad, reference_grad)


@pytest.mark.parametrize(
    "corruption",
    [None, "reordered_history", "duplicate_loss", "owner_weight", "denominator"],
)
def test_same_calls_as_segments_preserve_loss_and_gradients(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, corruption: str | None
) -> None:
    calls = []
    complete_call = RolloutTokenCapture.complete_call

    def record(
        capture: RolloutTokenCapture, call: ActiveCall, **kwargs: Any
    ) -> CommitCoords:
        # Observe full worker inputs BEFORE staging/materialization. The oracle
        # must not reconstruct its prompts by splitting the finalizer's output.
        calls.append((call.admission.capture_key, deepcopy(kwargs)))
        return complete_call(capture, call, **kwargs)

    monkeypatch.setattr(RolloutTokenCapture, "complete_call", record)
    plane = PublicationDataPlane()
    source = TQTokenSource(plane, staging_partition="staged")
    root_prompt = [10, 11]
    harness = gym_harness.make_capture_harness(
        monkeypatch,
        tmp_path,
        sink=TQTokenSink(plane, staging_partition="staged"),
        fetch_prefix=lambda keys: source.fetch_prefix_chains(keys).expanded,
        root_prompt=root_prompt,
        decide_input=decide_capture_input,
    )
    try:
        initial = capture_segment(harness, "group_g0", child=True)
        root_prompt[:] = [12, 13]  # A rewritten root, not a continuation of s0.
        rewritten = capture_segment(harness, "group_g0", child=True)
        root_prompt[:] = [10, 11]  # Same original grouping prompt for both owners.
        sibling = capture_segment(harness, "group_g1")
        finalizer = RolloutReassembler(
            plane,
            partition_id="canonical",
            staging_partition="staged",
            pad_token_id=0,
            max_seq_len=1024,
        )
        result = finalize(finalizer, [[initial, rewritten], [sibling]])
        assert result.valid_row_count == 3 and len(calls) == 5
        meta = result.meta
        assert meta is not None and meta.sequence_lengths == [6, 6, 3]
        canonical = plane.get_samples(
            sample_ids=meta.sample_ids,
            partition_id="canonical",
            select_fields=meta.fields,
        )
        ctrl = _controller(meta, canonical)
        _, valid = asyncio.run(ctrl._advantage_stage(meta))
        assert valid
    finally:
        harness.client.close()
        asyncio.run(harness.ledger.close())

    reference = []
    for scope, evidence in calls:
        prompt, generated = (
            evidence["prompt_token_ids"],
            evidence["generated_token_ids"],
        )
        # Rewards 1 and 2 give logical advantages -0.5 and +0.5, irrespective
        # of the 2:1 segment count or 4:1 action count. Do not reuse SC's fanout.
        advantage = -0.5 if scope == "group_g0" else 0.5
        reference.append(
            {
                "input_ids": torch.tensor(prompt + generated),
                "token_mask": torch.tensor(
                    [0.0] * len(prompt) + [1.0] * len(generated)
                ),
                "generation_logprobs": torch.tensor(
                    [0.0] * len(prompt) + evidence["generated_logprobs"]
                ),
                "advantages": torch.full((len(prompt) + len(generated),), advantage),
            }
        )
    fields = tuple(reference[0])
    batches = [
        BatchedDataDict(
            {
                **{
                    key: pad_sequence([row[key] for row in reference], batch_first=True)
                    for key in fields
                },
                "sample_mask": torch.ones(len(reference)),
            }
        ),
        BatchedDataDict(
            {
                **{
                    key: pad_sequence(list(canonical[key].unbind()), batch_first=True)
                    for key in fields
                },
                "sample_mask": canonical["sample_mask"],
            }
        ),
    ]
    assert _actions(batches[0]) == _actions(batches[1])
    assert set(_actions(batches[0])) == set(range(1001, 1006))
    assert batches[1]["input_ids"][1, :2].tolist() == [12, 13]
    for batch in batches:
        assert (batch["token_mask"][:, 1:] * batch["sample_mask"][:, None]).sum() == 5

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(11)
        model = _CausalModel().double()
    for batch in batches:
        with torch.no_grad():
            # Freeze old-policy evidence BEFORE applying corruption; otherwise
            # recomputing it could hide a damaged conditioning prefix at ratio 1.
            batch["prev_logprobs"] = nn.functional.pad(
                model(batch["input_ids"]) - 0.05, (1, 0)
            )
    expected = _loss_gradient(model, batches[0])
    merged = batches[1]
    denominator = 5
    if corruption == "reordered_history":
        merged["input_ids"][0, :2] = merged["input_ids"][0, :2].flip(0)
    elif corruption == "duplicate_loss":
        # Duplicate the first whole row: normalization must not hide double ownership.
        merged = BatchedDataDict(
            {key: torch.cat([value, value[:1]]) for key, value in merged.items()}
        )
        with pytest.raises(AssertionError, match="owned twice"):
            _actions(merged)
    elif corruption == "owner_weight":
        merged["advantages"][:2] /= 2  # Incorrect inverse-segment weighting.
    elif corruption == "denominator":
        denominator = 3  # Incorrect physical-row normalization.
    actual = _loss_gradient(model, merged, denominator=denominator)
    if corruption is None:
        for observed, wanted in zip(actual, expected, strict=True):
            torch.testing.assert_close(observed, wanted, rtol=1e-9, atol=1e-12)
    else:
        assert not torch.allclose(actual[0], expected[0], rtol=1e-6, atol=1e-12)
        assert not torch.allclose(actual[1], expected[1], rtol=1e-6, atol=1e-12)


class _CausalValueModel(nn.Module):
    """The recurrent toy above with a scalar value head at every position."""

    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(1024, 8)
        self.recurrent = nn.GRU(8, 8, batch_first=True)
        self.head = nn.Linear(8, 1)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        hidden, _ = self.recurrent(self.embedding(tokens))
        return self.head(hidden)  # [B, S, 1]; the value loss squeezes it.


def _value_loss_gradient(
    model: nn.Module,
    data: BatchedDataDict,
    *,
    denominator: int,
    cliprange: float | None,
) -> tuple[torch.Tensor, torch.Tensor]:
    model = deepcopy(model)
    loss, _ = MseValueLossFn(MseValueLossConfig(cliprange=cliprange))(
        model(data["input_ids"]),
        data,
        global_valid_seqs=data["sample_mask"].sum(),
        global_valid_toks=torch.tensor(float(denominator)),
    )
    loss.backward()
    gradients = torch.cat(
        [parameter.grad.flatten() for parameter in model.parameters()]
    )
    assert torch.isfinite(loss) and torch.isfinite(gradients).all()
    assert gradients.norm() > 1e-8, "zero gradients cannot establish conservation"
    return loss.detach(), gradients


def _old_values(model: nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
    """Frozen critic evidence; predictions stay inside a 0.2 clip range of it."""
    with torch.no_grad():
        return model(input_ids).squeeze(-1) - 0.05


@pytest.mark.parametrize("dp_size", [1, 2])
@pytest.mark.parametrize("filter_owner", [False, True])
@pytest.mark.parametrize("cliprange", [None, 0.2])
def test_packing_bins_preserve_forest_value_loss_accounting(
    dp_size: int, filter_owner: bool, cliprange: float | None
) -> None:
    """Critic counterpart of the policy bin test: real PPO stage, real value loss."""
    meta, data = _batch([("a", [(0.0, 2), (1.0, 2), (3.0, 1)])], padding=True)
    lengths = torch.tensor([3, 3, 2, 3, 2, 3])
    data["token_mask"][lengths == 2, 2] = 0
    if filter_owner:
        data["mask_sample"][1] = True
    input_ids = torch.arange(10, 28).reshape(6, 3)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(11)
        model = _CausalValueModel().double()
    # Per-row GAE runs over the same old values the clipped loss reads.
    data["values"] = _old_values(model, input_ids).float()
    ctrl = _ppo_controller(meta, data)
    _, valid = asyncio.run(ctrl._advantage_stage(meta))
    assert valid
    # A flagged segment drops every row of its owner.
    assert data["sample_mask"][:2].sum() == (0 if filter_owner else 2)
    batch = BatchedDataDict(
        {
            "input_ids": input_ids,
            "input_lengths": lengths,
            "values": data["values"].double(),
            "returns": data["returns"].double(),
            "token_mask": data["token_mask"],
            "sample_mask": data["sample_mask"],
            "row_index": torch.arange(6),
        }
    )
    shards, _ = batch.shard_by_batch_size(
        dp_size,
        batch_size=6,
        sequence_packing_args={
            "algorithm": "modified_first_fit_decreasing",
            "input_key": "input_ids",
            "input_lengths_key": "input_lengths",
            "sequence_length_pad_multiple": 1,
            "max_tokens_per_microbatch": 8,
        },
    )
    microbatches = [
        mb
        for shard in shards
        for mb in shard.make_microbatch_iterator_for_packable_sequences()
    ]
    seen = []
    for mb in microbatches:
        indices = mb["row_index"]
        seen.extend(indices.tolist())
        for field in batch:
            torch.testing.assert_close(mb[field], batch[field][indices])
    assert sorted(seen) == list(range(6)), "packing must retain each row exactly once"
    assert any(mb.size > 1 for mb in microbatches)
    denominator = int((batch["token_mask"] * batch["sample_mask"][:, None]).sum())
    reference_loss, reference_grad = _value_loss_gradient(
        model, batch, denominator=denominator, cliprange=cliprange
    )
    packed_loss, packed_grad = (
        torch.zeros_like(reference_loss),
        torch.zeros_like(reference_grad),
    )
    for mb in microbatches:
        if not (mb["token_mask"] * mb["sample_mask"][:, None]).any():
            continue
        loss, grad = _value_loss_gradient(
            model, mb, denominator=denominator, cliprange=cliprange
        )
        packed_loss += loss
        packed_grad += grad
    torch.testing.assert_close(packed_loss, reference_loss)
    torch.testing.assert_close(packed_grad, reference_grad)


# (owner, prompt, generated). The second call continues the first; the third is
# a rewritten root of the same owner; the fourth is a sibling owner.
_CRITIC_CALLS = [
    ("g0", [10, 11], [1001, 1002]),
    ("g0", [10, 11, 1001, 1002, 20], [1003]),
    ("g0", [12, 13], [1004]),
    ("g1", [10, 11], [1005]),
]
_CRITIC_REWARDS = {"g0": 1.0, "g1": 2.0}


def _critic_rows(segments: list[list[int]]) -> BatchedDataDict:
    """One row per segment: the last call's full input, every call's tokens trained."""
    input_ids, token_masks, rewards = [], [], []
    for calls in segments:
        owner, prompt, generated = _CRITIC_CALLS[calls[-1]]
        tokens = prompt + generated
        mask = [0.0] * len(tokens)
        for call in calls:
            call_owner, call_prompt, call_generated = _CRITIC_CALLS[call]
            end = len(call_prompt) + len(call_generated)
            assert call_owner == owner and tokens[:end] == call_prompt + call_generated
            mask[len(call_prompt) : end] = [1.0] * len(call_generated)
        input_ids.append(torch.tensor(tokens))
        token_masks.append(torch.tensor(mask, dtype=torch.float64))
        rewards.append(_CRITIC_REWARDS[owner])
    return BatchedDataDict(
        {
            "input_ids": pad_sequence(input_ids, batch_first=True),
            "token_mask": pad_sequence(token_masks, batch_first=True),
            "sample_mask": torch.ones(len(segments), dtype=torch.float64),
            "total_reward": torch.tensor(rewards, dtype=torch.float64),
        }
    )


def _critic_targets(data: BatchedDataDict) -> dict[int, tuple[list[int], float]]:
    """Each trained token's full conditioning and return target."""
    targets = {}
    for row in range(data.size):
        for position in (data["token_mask"][row] * data["sample_mask"][row]).nonzero():
            index = position.item()
            token = data["input_ids"][row, index].item()
            assert token not in targets, "sampled token owned twice in this fixture"
            targets[token] = (
                data["input_ids"][row, :index].tolist(),
                round(data["returns"][row, index].item(), 9),
            )
    return targets


@pytest.mark.parametrize("cliprange", [None, 0.2])
@pytest.mark.parametrize(
    "corruption", [None, "reordered_history", "duplicate_loss", "denominator"]
)
def test_same_calls_as_segments_preserve_value_loss_and_gradients(
    cliprange: float | None, corruption: str | None
) -> None:
    """Per-call rows and Forest segments of the same calls train the critic alike.

    GAE runs per physical row; with lambda = gamma = 1 every trained token's
    return is its owner's reward however the calls are split into rows.
    """
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(11)
        model = _CausalValueModel().double()
    estimator = GeneralizedAdvantageEstimator(
        GAEConfig(gae_lambda=1.0, gae_gamma=1.0, normalize_advantages=False),
        ClippedPGLossConfig(),
    )
    batches = [
        _critic_rows([[0], [1], [2], [3]]),
        _critic_rows([[0, 1], [2], [3]]),
    ]
    for batch in batches:
        batch["values"] = _old_values(model, batch["input_ids"])
        _, batch["returns"] = estimator.compute_advantage(
            prompt_ids=None,
            rewards=batch["total_reward"],
            mask=batch["token_mask"],
            values=batch["values"],
        )
    assert _critic_targets(batches[0]) == _critic_targets(batches[1])
    assert {
        token: target for token, (_, target) in _critic_targets(batches[1]).items()
    } == {1001: 1.0, 1002: 1.0, 1003: 1.0, 1004: 1.0, 1005: 2.0}
    for batch in batches:
        assert (batch["token_mask"] * batch["sample_mask"][:, None]).sum() == 5

    expected = _value_loss_gradient(
        model, batches[0], denominator=5, cliprange=cliprange
    )
    merged = batches[1]
    denominator = 5
    if corruption == "reordered_history":
        merged["input_ids"][0, :2] = merged["input_ids"][0, :2].flip(0)
    elif corruption == "duplicate_loss":
        merged = BatchedDataDict(
            {key: torch.cat([value, value[:1]]) for key, value in merged.items()}
        )
        with pytest.raises(AssertionError, match="owned twice"):
            _critic_targets(merged)
    elif corruption == "denominator":
        denominator = 3
    actual = _value_loss_gradient(
        model, merged, denominator=denominator, cliprange=cliprange
    )
    if corruption is None:
        for observed, wanted in zip(actual, expected, strict=True):
            torch.testing.assert_close(observed, wanted, rtol=1e-9, atol=1e-12)
    else:
        assert not torch.allclose(actual[0], expected[0], rtol=1e-6, atol=1e-12)
        assert not torch.allclose(actual[1], expected[1], rtol=1e-6, atol=1e-12)
