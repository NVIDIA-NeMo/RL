# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Same-evidence per-call versus segment loss/gradient conservation on CPU.

Uses real staging, finalization, controller advantages, and policy-gradient loss.
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

from nemo_rl.algorithms.loss import ClippedPGLossConfig, ClippedPGLossFn
from nemo_rl.data_plane.tq_token_sink import TQTokenSink, TQTokenSource
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.experience.rollout_reassembler import RolloutReassembler
from tests.unit.experience.test_logical_owner_finalization import (
    PublicationDataPlane,
    capture_segment,
    finalize,
    gym_harness,
)
from tests.unit.single_controller.test_logical_advantage import _controller

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
        calls.append((call.admission.rollout_id, deepcopy(kwargs)))
        return complete_call(capture, call, **kwargs)

    monkeypatch.setattr(RolloutTokenCapture, "complete_call", record)
    plane = PublicationDataPlane()
    source = TQTokenSource(plane, staging_partition="staged")
    root_prompt = [10, 11]
    harness = gym_harness.make_capture_harness(
        monkeypatch,
        tmp_path,
        sink=TQTokenSink(plane, staging_partition="staged"),
        fetch_prefix=source.fetch_prefix_token_ids,
        root_prompt=root_prompt,
    )
    try:
        initial = capture_segment(harness, "group_g0_s0", child=True)
        root_prompt[:] = [12, 13]  # A rewritten root, not a continuation of s0.
        rewritten = capture_segment(harness, "group_g0_s1", child=True)
        root_prompt[:] = [10, 11]  # Same original grouping prompt for both owners.
        sibling = capture_segment(harness, "group_g1_s0")
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
        ctrl = _controller(meta, canonical, grpo={"baseline_population": "all_owners"})
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
        advantage = -0.5 if scope.startswith("group_g0_") else 0.5
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
