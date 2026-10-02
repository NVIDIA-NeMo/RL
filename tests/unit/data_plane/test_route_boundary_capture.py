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
"""Captured multi-turn routes must restore the prior padded decode boundary."""

import json

import pytest
import torch

pytest.importorskip("nemo_gym.token_id_capture.staging")

from nemo_gym.token_id_capture.adapters.vllm import VLLMCaptureAdapter
from nemo_gym.token_id_capture.staging.digest import compute_extras_digest
from nemo_rl.data_plane.schema import ROUTE_ENCODING_ENVELOPE
from nemo_rl.experience.route_assembly import RouteFragment, execute_route_plan
from nemo_rl.experience.route_plan import (
    ROUTE_PLAN_SCHEMA_VERSION,
    RouteAssemblyPlan,
    RouteSpan,
)
from nemo_rl.models.generation.vllm.vllm_worker_async import (
    VllmAsyncGenerationWorkerImpl,
)
from nemo_rl.utils.routed_experts_codec import (
    decode_routed_experts,
    encode_routed_experts,
)

pytestmark = pytest.mark.nemo_gym


def test_capture_restores_real_route_at_multi_turn_boundary():
    first = torch.tensor([[[10, 11]], [[20, 21]], [[0, 1]]], dtype=torch.int16)
    second = torch.tensor(
        [[[10, 11]], [[20, 21]], [[30, 31]], [[40, 41]], [[0, 1]]], dtype=torch.int16
    )
    fragments = {}
    spans = []
    for key, routes, previous, prompt, generated in [
        ("first", first, 0, 2, 1),
        ("second", second, 3, 4, 1),
    ]:
        payload = {
            "choices": [{"message": {"routed_experts": encode_routed_experts(routes)}}]
        }
        VllmAsyncGenerationWorkerImpl._delta_align_routed_experts(
            payload, prev_len=previous, prompt_len=prompt, generated_len=generated
        )
        extras = VLLMCaptureAdapter().extract_extras(payload)
        delta = decode_routed_experts(extras["routed_experts"], torch.int16)
        metadata = {
            key: value for key, value in extras.items() if key != "routed_experts"
        }
        if previous == 0:
            assert "routed_experts_prefix_boundary" not in extras
        else:
            assert (
                decode_routed_experts(
                    extras["routed_experts_prefix_boundary"], torch.int16
                ).tolist()
                == second[2:3].tolist()
            )
        fragments[key] = RouteFragment(
            delta, ROUTE_ENCODING_ENVELOPE, json.dumps(metadata).encode()
        )
        spans.append(
            RouteSpan(
                key,
                prompt - previous,
                generated,
                len(delta),
                1,
                compute_extras_digest(extras),
            )
        )
    plan = RouteAssemblyPlan(
        ROUTE_PLAN_SCHEMA_VERSION, "staging", tuple(spans), ("first", "second"), 5
    )
    assembled, error = execute_route_plan(plan, fragments, dims=(1, 2), canonical_len=5)
    assert error is None
    expected = torch.cat([first, second[3:]])
    expected[2] = second[2]
    assert torch.equal(assembled, expected)


@pytest.mark.parametrize("boundary_shape", [(2, 1, 2), (1, 2, 2), (1, 1, 3)])
def test_rejects_invalid_prefix_boundary_shape(boundary_shape):
    first = torch.tensor([[[0, 1]]], dtype=torch.int16)
    boundary = encode_routed_experts(torch.zeros(boundary_shape, dtype=torch.int16))
    second = first.clone()
    extras = {
        "routed_experts": encode_routed_experts(second),
        "routed_experts_prefix_boundary": boundary,
    }
    first_extras = {"routed_experts": encode_routed_experts(first)}
    fragments = {
        "first": RouteFragment(first, ROUTE_ENCODING_ENVELOPE, b"{}"),
        "second": RouteFragment(
            second,
            ROUTE_ENCODING_ENVELOPE,
            json.dumps({"routed_experts_prefix_boundary": boundary}).encode(),
        ),
    }
    spans = (
        RouteSpan("first", 0, 1, 1, 1, compute_extras_digest(first_extras)),
        RouteSpan("second", 0, 1, 1, 1, compute_extras_digest(extras)),
    )
    plan = RouteAssemblyPlan(
        ROUTE_PLAN_SCHEMA_VERSION, "staging", spans, ("first", "second"), 2
    )
    _, error = execute_route_plan(plan, fragments, dims=(1, 2), canonical_len=2)
    assert error == "prefix_boundary_mismatch"


def test_boundary_route_is_covered_by_fragment_digest():
    routes = torch.tensor([[[0, 1]]], dtype=torch.int16)
    extras = {"routed_experts": encode_routed_experts(routes)}
    boundary = encode_routed_experts(torch.tensor([[[2, 3]]], dtype=torch.int16))
    fragment = RouteFragment(
        routes,
        ROUTE_ENCODING_ENVELOPE,
        json.dumps({"routed_experts_prefix_boundary": boundary}).encode(),
    )
    span = RouteSpan("first", 0, 1, 1, 1, compute_extras_digest(extras))
    plan = RouteAssemblyPlan(
        ROUTE_PLAN_SCHEMA_VERSION, "staging", (span,), ("first",), 1
    )
    _, error = execute_route_plan(
        plan, {"first": fragment}, dims=(1, 2), canonical_len=1
    )
    assert error == "fragment_integrity"
