# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Call provenance survives later authenticated prefix and boundary changes."""

import json
from copy import deepcopy
from dataclasses import replace

import pytest
import torch
from torch import nn

pytest.importorskip("nemo_gym.token_id_capture.staging")

from nemo_gym.token_id_capture.staging.digest import (  # noqa: E402
    EXTRAS_DIGEST_VERSION,
    compute_extras_digest,
)

from nemo_rl.algorithms.loss import ClippedPGLossConfig, ClippedPGLossFn  # noqa: E402
from nemo_rl.data_plane.schema import ROUTE_ENCODING_ENVELOPE  # noqa: E402
from nemo_rl.distributed.batched_data_dict import BatchedDataDict  # noqa: E402
from nemo_rl.experience.route_assembly import (  # noqa: E402
    RouteFragment,
    execute_route_plan,
    execute_route_plan_with_call_views,
)
from nemo_rl.experience.route_plan import (  # noqa: E402
    ROUTE_PLAN_SCHEMA_VERSION,
    RouteAssemblyPlan,
    RouteSpan,
)
from nemo_rl.utils.routed_experts_codec import encode_routed_experts  # noqa: E402

pytestmark = pytest.mark.nemo_gym


def _case() -> tuple[RouteAssemblyPlan, dict[str, RouteFragment]]:
    spans = []
    fragments = {}
    # A later prefill changes every earlier route, including the previous
    # call's terminal token. The last decode token itself has no forward route.
    for index, (carry, generation, end) in enumerate(((2, 2, 4), (1, 2, 7), (0, 1, 8))):
        key = f"owner/c{index}"
        routes = torch.arange(index * 10, index * 10 + end, dtype=torch.int16)[
            :, None, None
        ]
        routes[-1] = -1
        metadata = {"routed_experts_prefix_len": (0, 4, 7)[index]}
        if index:
            metadata["predecessor_tail_route"] = [[70 + index]]
        fragments[key] = RouteFragment(
            routes=routes,
            encoding=ROUTE_ENCODING_ENVELOPE,
            extras_metadata_json=json.dumps(metadata).encode(),
        )
        digest = compute_extras_digest(
            {**metadata, "routed_experts": encode_routed_experts(routes)}
        )
        spans.append(
            RouteSpan(key, carry, generation, end, EXTRAS_DIGEST_VERSION, digest)
        )
    return (
        RouteAssemblyPlan(
            ROUTE_PLAN_SCHEMA_VERSION, "staged", tuple(spans), tuple(fragments), 8
        ),
        fragments,
    )


def test_views_keep_original_prefixes_and_every_target_exactly_once() -> None:
    plan, fragments = _case()
    result, failure = execute_route_plan_with_call_views(
        plan, fragments, dims=(1, 1), canonical_len=8, backpatch_predecessor_tail=True
    )
    assert failure is None and result is not None
    assert [view.routed_experts[:, 0, 0].tolist() for view in result.call_views] == [
        [0, 1, 2, -1],
        [10, 11, 12, 71, 14, 15, -1],
        [20, 21, 22, 23, 24, 25, 72, -1],
    ]
    # Canonical training keeps each original generation predictor. The latest
    # prefill remains an independent diagnostic view, not the training map.
    canonical, failure = execute_route_plan(
        plan, fragments, dims=(1, 1), canonical_len=8, backpatch_predecessor_tail=True
    )
    assert failure is None and torch.equal(canonical, result.routed_experts)
    assert canonical[:, 0, 0].tolist() == [20, 1, 2, 23, 14, 15, 72, -1]
    assert not torch.equal(result.call_views[-1].routed_experts, canonical)
    targets = [
        position
        for view in result.call_views
        for position in range(view.generation_start, view.generation_end)
    ]
    assert targets == [2, 3, 5, 6, 7] and len(set(targets)) == len(targets)
    for view in result.call_views:
        assert view.routed_experts.shape[0] == view.generation_end
        assert view.routed_experts.data_ptr() != canonical.data_ptr()
    # Returned tensors also own their bytes independently of storage payloads.
    fragments["owner/c0"].routes.fill_(99)
    assert result.call_views[0].routed_experts[:, 0, 0].tolist() == [0, 1, 2, -1]


@pytest.mark.parametrize("corruption", ["digest", "shape", "prefix", "missing", "tail"])
def test_late_failure_never_exposes_partially_verified_views(corruption: str) -> None:
    plan, fragments = _case()
    last = fragments["owner/c2"]
    if corruption == "missing":
        del fragments["owner/c2"]
    elif corruption == "digest":
        last.routes[0] = 99
    else:
        metadata = json.loads(last.extras_metadata_json)
        routes = last.routes
        if corruption == "shape":
            routes = routes.expand(-1, 2, -1).clone()
        elif corruption == "prefix":
            metadata["routed_experts_prefix_len"] = 6
        elif corruption == "tail":
            metadata["predecessor_tail_route"] = [[-1]]
        fragments["owner/c2"] = replace(
            last, routes=routes, extras_metadata_json=json.dumps(metadata).encode()
        )
        span = replace(
            plan.spans[-1],
            extras_digest=compute_extras_digest(
                {**metadata, "routed_experts": encode_routed_experts(routes)}
            ),
        )
        plan = replace(plan, spans=(*plan.spans[:-1], span))
    result, failure = execute_route_plan_with_call_views(
        plan, fragments, dims=(1, 1), canonical_len=8, backpatch_predecessor_tail=True
    )
    assert result is None and failure is not None


def test_zero_generation_call_keeps_context_without_adding_loss_targets() -> None:
    plan, fragments = _case()
    plan = replace(
        plan,
        spans=(replace(plan.spans[0], carry_len=4, generation_len=0), *plan.spans[1:]),
    )
    result, failure = execute_route_plan_with_call_views(
        plan, fragments, dims=(1, 1), canonical_len=8, backpatch_predecessor_tail=True
    )
    assert failure is None and result is not None
    assert (
        result.call_views[0].generation_start
        == result.call_views[0].generation_end
        == 4
    )
    assert result.call_views[1].routed_experts.shape[0] == 7
    # A prompt-only call owns no generation predictors. Later authenticated
    # prefill routes may replace all of its conditioning positions.
    assert result.routed_experts[:, 0, 0].tolist() == [20, 21, 22, 23, 14, 15, 72, -1]


def test_legacy_tail_missing_prompt_route_can_be_repaired_by_later_prefix() -> None:
    spans = []
    fragments = {}
    for key, carry, generation, values, metadata in (
        ("owner/legacy", 2, 2, [11, -1], {}),
        (
            "owner/continuation",
            2,
            1,
            [20, 21, 22, 23, 24, 25, -1],
            {"routed_experts_prefix_len": 4},
        ),
    ):
        routes = torch.tensor(values, dtype=torch.int16)[:, None, None]
        digest = compute_extras_digest(
            {**metadata, "routed_experts": encode_routed_experts(routes)}
        )
        spans.append(
            RouteSpan(
                key, carry, generation, len(values), EXTRAS_DIGEST_VERSION, digest
            )
        )
        fragments[key] = RouteFragment(
            routes, ROUTE_ENCODING_ENVELOPE, json.dumps(metadata).encode()
        )
    plan = RouteAssemblyPlan(
        ROUTE_PLAN_SCHEMA_VERSION, "staged", tuple(spans), tuple(fragments), 7
    )
    result, failure = execute_route_plan_with_call_views(
        plan, fragments, dims=(1, 1), canonical_len=7, backpatch_predecessor_tail=False
    )
    assert failure is None and result is not None
    assert result.routed_experts[:, 0, 0].tolist() == [20, 21, 11, 23, 24, 25, -1]
    assert result.call_views[0].routed_experts[:, 0, 0].tolist() == [-1, -1, 11, -1]
    assert result.call_views[1].routed_experts[:, 0, 0].tolist() == [
        20,
        21,
        22,
        23,
        24,
        25,
        -1,
    ]


class _RouteConditionedModel(nn.Module):
    """A recurrent oracle whose conditional logprobs depend on prefix routes.

    This checks the replay representation and real GRPO loss on CPU; it does
    not qualify Megatron routing, multimodal execution, or GPU gradients.
    """

    def __init__(self) -> None:
        super().__init__()
        self.tokens = nn.Embedding(16, 8)
        self.experts = nn.Embedding(100, 8)
        self.recurrent = nn.GRU(8, 8, batch_first=True)
        self.head = nn.Linear(8, 16)

    def forward(self, tokens: torch.Tensor, routes: torch.Tensor) -> torch.Tensor:
        hidden = self.tokens(tokens[:, :-1]) + self.experts(
            routes[:-1, 0, 0].long().clamp_min(0)
        )
        output, _ = self.recurrent(hidden)
        return (
            self.head(output)
            .log_softmax(-1)
            .gather(-1, tokens[:, 1:, None])
            .squeeze(-1)
        )


@pytest.mark.parametrize(
    "corruption",
    [None, "latest_prefix", "duplicate_targets", "physical_row_denominator"],
)
def test_call_views_conserve_route_conditioned_grpo_loss_and_gradient(
    corruption: str | None,
) -> None:
    plan, fragments = _case()
    result, failure = execute_route_plan_with_call_views(
        plan, fragments, dims=(1, 1), canonical_len=8, backpatch_predecessor_tail=True
    )
    assert result is not None and failure is None
    # The oracle specifies the actual call inputs independently of the
    # assembled views. In particular, a later prefix is not evidence of an
    # earlier call's route choices.
    expected = [
        (2, [0, 1, 2, -1]),
        (5, [10, 11, 12, 71, 14, 15, -1]),
        (7, [20, 21, 22, 23, 24, 25, 72, -1]),
    ]
    oracle_rows = [
        (start, torch.tensor(routes, dtype=torch.int16)[:, None, None])
        for start, routes in expected
    ]
    actual_rows = [
        (
            view.generation_start,
            result.routed_experts[: view.generation_end]
            if corruption == "latest_prefix"
            else view.routed_experts,
        )
        for view in result.call_views
    ]
    if corruption == "duplicate_targets":
        actual_rows.append(actual_rows[0])
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(29)
        model = _RouteConditionedModel().double()
    loss_fn = ClippedPGLossFn(
        ClippedPGLossConfig(
            reference_policy_kl_penalty=0.0,
            use_importance_sampling_correction=True,
            truncated_importance_sampling_type="tis",
            truncated_importance_sampling_ratio=2.0,
            truncated_importance_sampling_ratio_min=0.5,
        )
    )
    # Frozen previous-policy evidence is captured with the oracle's routes.
    previous = []
    with torch.no_grad():
        for _, routes in oracle_rows:
            tokens = torch.arange(1, len(routes) + 1)[None, :]
            previous.append(nn.functional.pad(model(tokens, routes) - 0.05, (1, 0)))

    def evaluate(
        rows: list[tuple[int, torch.Tensor]], *, denominator: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        instance = deepcopy(model)
        losses = []
        for index, (start, routes) in enumerate(rows):
            length = len(routes)
            tokens = torch.arange(1, length + 1)[None, :]
            mask = torch.zeros(1, length, dtype=torch.float64)
            mask[:, start:] = 1
            old = previous[index if index < len(previous) else 0]
            batch = BatchedDataDict(
                {
                    "input_ids": tokens,
                    "token_mask": mask,
                    "sample_mask": torch.ones(1, dtype=torch.float64),
                    "advantages": torch.full((1, length), 0.7, dtype=torch.float64),
                    "prev_logprobs": old,
                    "generation_logprobs": old + 0.1,
                }
            )
            loss, _ = loss_fn(
                instance(tokens, routes),
                batch,
                global_valid_seqs=torch.tensor(1.0),
                global_valid_toks=torch.tensor(float(denominator)),
            )
            losses.append(loss)
        total = torch.stack(losses).sum()
        total.backward()
        gradient = torch.cat(
            [parameter.grad.flatten() for parameter in instance.parameters()]
        )
        assert torch.isfinite(total) and torch.isfinite(gradient).all()
        assert gradient.norm() > 1e-8
        return total.detach(), gradient

    wanted = evaluate(oracle_rows, denominator=5)
    actual = evaluate(
        actual_rows, denominator=3 if corruption == "physical_row_denominator" else 5
    )
    if corruption is None:
        for observed, reference in zip(actual, wanted, strict=True):
            torch.testing.assert_close(observed, reference, rtol=1e-10, atol=1e-12)
    else:
        assert not torch.allclose(actual[0], wanted[0], rtol=1e-6, atol=1e-12)
        assert not torch.allclose(actual[1], wanted[1], rtol=1e-6, atol=1e-12)
