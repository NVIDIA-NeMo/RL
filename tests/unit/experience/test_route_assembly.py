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
"""Token-capture router replay across calls: worker staging + route executor.

vLLM never routes a call's last sampled token; that row is the -1 sentinel.
The next call routes it during prefill. For a linear chain of calls, the
training routes must therefore equal the last call's full route tensor.
"""

from __future__ import annotations

import pytest
import torch

pytest.importorskip("nemo_gym.token_id_capture.staging")

from nemo_gym.token_id_capture.staging.digest import (  # noqa: E402
    EXTRAS_DIGEST_VERSION,
    compute_extras_digest,
)

from nemo_rl.data_plane.schema import ROUTE_ENCODING_TENSOR  # noqa: E402
from nemo_rl.experience.route_assembly import (  # noqa: E402
    RouteFragment,
    RouteLayout,
    boundary_route_rows,
    execute_route_plan,
)
from nemo_rl.experience.route_plan import (  # noqa: E402
    ROUTE_PLAN_SCHEMA_VERSION,
    RouteAssemblyPlan,
    RouteSpan,
)
from nemo_rl.models.generation.vllm.vllm_worker_async import (  # noqa: E402
    VllmAsyncGenerationWorkerImpl,
)

pytestmark = pytest.mark.nemo_gym

NO_ROUTE = [-1, -1]  # vLLM never routes a call's last sampled token (top_k=2)


def _routes(*rows: list[int]) -> torch.Tensor:
    """One MoE layer, top_k=2: ``[len(rows), 1, 2]``."""
    return torch.tensor([[row] for row in rows], dtype=torch.int16)


def _assemble(calls: list[tuple[torch.Tensor, int, int]]) -> torch.Tensor:
    """Stage each call like the vLLM worker, then run the route executor.

    Each call is ``(vllm_routes, prev_len, prompt_len)``.
    """
    spans, fragments = [], {}
    for index, (vllm_routes, prev_len, prompt_len) in enumerate(calls):
        payload = {"choices": [{"message": {"routed_experts": vllm_routes}}]}
        staged = VllmAsyncGenerationWorkerImpl._delta_align_routed_experts(
            payload,
            prev_len=prev_len,
            prompt_len=prompt_len,
            generated_len=len(vllm_routes) - prompt_len,
        )
        assert staged is not None
        route_metadata = payload["choices"][0]["message"]["routed_experts"]
        key = f"rollout/call{index}"
        spans.append(
            RouteSpan(
                key,
                carry_len=prompt_len - prev_len,
                generation_len=len(vllm_routes) - prompt_len,
                staged_route_len=len(staged),
                extras_digest_version=EXTRAS_DIGEST_VERSION,
                extras_digest=compute_extras_digest({"routed_experts": route_metadata}),
                boundary_rows=boundary_route_rows(prev_len),
            )
        )
        fragments[key] = RouteFragment(staged, ROUTE_ENCODING_TENSOR, b"null")

    total_len = len(calls[-1][0])
    plan = RouteAssemblyPlan(
        schema_version=ROUTE_PLAN_SCHEMA_VERSION,
        staging_partition="staging",
        spans=tuple(spans),
        cleanup_staging_keys=tuple(fragments),
        expected_token_length=total_len,
    )
    routed, reason = execute_route_plan(
        plan,
        fragments,
        layout=RouteLayout.compressed(num_moe_layers=1, top_k=2),
        canonical_len=total_len,
    )
    assert reason is None, reason
    return routed


def test_single_call_keeps_vllm_routes() -> None:
    # Prompt: 2 tokens. Generation: 2 tokens.
    call = _routes([10, 11], [12, 13], [14, 15], NO_ROUTE)
    assert torch.equal(_assemble([(call, 0, 2)]), call)


def test_turn_boundary_gets_the_next_calls_route(monkeypatch) -> None:
    def reject_codec(*args, **kwargs):
        pytest.fail("native route capture must not encode or decode route tensors")

    monkeypatch.setattr(
        "nemo_rl.utils.routed_experts_codec.encode_routed_experts", reject_codec
    )
    monkeypatch.setattr(
        "nemo_rl.utils.routed_experts_codec.decode_routed_experts", reject_codec
    )
    monkeypatch.setattr(
        "nemo_rl.experience.route_assembly.encode_routed_experts", reject_codec
    )

    # Call 1: prompt 2 tokens, generates 2 (positions 2-3).
    call1 = _routes([10, 11], [12, 13], [14, 15], NO_ROUTE)
    # Call 2: prompt = call 1's 4 tokens + 1 tool token, generates 1.
    # Its prefill computed the real route of position 3 (call 1's last token).
    call2 = _routes([10, 11], [12, 13], [14, 15], [16, 17], [18, 19], NO_ROUTE)

    routed = _assemble([(call1, 0, 2), (call2, 4, 5)])

    assert routed[3].tolist() == [[16, 17]], "boundary token has no route"
    assert torch.equal(routed, call2)
