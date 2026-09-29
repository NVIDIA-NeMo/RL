# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native route attachments through Gym capture, TQ staging, and verification."""

from __future__ import annotations

from dataclasses import replace

import pytest
import torch

pytest.importorskip("nemo_gym.token_id_capture.staging")

from nemo_gym.token_id_capture.staging import (  # noqa: E402
    CaptureAdmission,
    RolloutTokenCapture,
)

from nemo_rl.data_plane.schema import (  # noqa: E402
    ROUTED_EXPERTS_FIELD,
)
from nemo_rl.data_plane.tq_token_sink import TQTokenSink, TQTokenSource  # noqa: E402
from nemo_rl.experience.route_assembly import (  # noqa: E402
    execute_route_plan,
    verify_route_fragment_integrity,
)
from nemo_rl.experience.route_plan import RouteAssemblyPlan, RouteSpan  # noqa: E402
from nemo_rl.utils.routed_experts_codec import (  # noqa: E402
    routed_experts_tensor_metadata,
)

pytestmark = pytest.mark.nemo_gym


class _StagingClient:
    def __init__(self):
        self.fields = None
        self.puts = 0

    def put_samples(self, *, fields, **kwargs):
        self.fields = fields
        self.puts += 1

    def get_samples(self, *, select_fields, **kwargs):
        return self.fields.select(*select_fields)


@pytest.mark.parametrize("dtype", [torch.int8, torch.int16, torch.int32])
def test_capture_stages_and_verifies_native_routes_without_codec(dtype, monkeypatch):
    def forbid_codec(*args, **kwargs):
        pytest.fail("native capture must not encode or decode routes")

    monkeypatch.setattr(
        "nemo_rl.utils.routed_experts_codec.encode_routed_experts", forbid_codec
    )
    monkeypatch.setattr(
        "nemo_rl.utils.routed_experts_codec.decode_routed_experts", forbid_codec
    )
    monkeypatch.setattr(
        "nemo_rl.experience.route_assembly.encode_routed_experts", forbid_codec
    )
    client = _StagingClient()
    capture = RolloutTokenCapture(
        sink=TQTokenSink(client, staging_partition="staging"),
        weight_version_fn=lambda: 7,
    )
    routes = torch.tensor([[[1, -1]], [[3, 4]], [[5, 6]]], dtype=dtype)
    coords = capture.complete_call(
        capture.begin_call(
            CaptureAdmission(rollout_id="r", model_call_id="c", mode="text")
        ),
        prompt_token_ids=[10, 11],
        generated_token_ids=[12],
        generated_logprobs=[-0.5],
        extras={
            "routed_experts": routed_experts_tensor_metadata(routes),
            "other_extra": 42,
        },
        attachments={"routed_experts": routes},
    )
    assert coords.disposition == "staged"
    assert client.puts == 1
    assert client.fields[ROUTED_EXPERTS_FIELD].data_ptr() == routes.data_ptr()

    source = TQTokenSource(client, staging_partition="staging")
    deferred = source.fetch_for_finalization([coords.staging_key])[0]
    assert deferred.fragment is None
    assert deferred.routed_len == 3
    direct = source.fetch_for_finalization(
        [coords.staging_key], include_route_fragments=True
    )[0]
    fragment = direct.fragment
    assert fragment.routes.dtype == dtype
    assert torch.equal(fragment.routes, routes)
    span = RouteSpan(
        coords.staging_key,
        2,
        1,
        3,
        extras_digest_version=1,
        extras_digest=coords.extras_digest,
    )
    plan = RouteAssemblyPlan(
        schema_version=1,
        staging_partition="staging",
        spans=(span,),
        cleanup_staging_keys=(coords.staging_key,),
        expected_token_length=3,
    )
    assembled, error = execute_route_plan(
        plan, {coords.staging_key: fragment}, dims=(1, 2), canonical_len=3
    )
    assert error is None
    assert torch.equal(assembled, routes.to(torch.int16))

    def verifies(candidate):
        return verify_route_fragment_integrity(
            replace(fragment, routes=candidate),
            extras_digest_version=1,
            expected_extras_digest=coords.extras_digest,
        )

    assert not verifies(routes.reshape(1, 3, 2))
    assert not verifies(routes.to(torch.int32 if dtype != torch.int32 else torch.int16))
    tampered = routes.clone()
    tampered[0, 0, 0] += 1
    assert not verifies(tampered)


@pytest.mark.parametrize(
    "attachment", [None, {"routed_experts": None}, {"unknown": torch.ones(1)}]
)
def test_capture_rejects_missing_or_unsupported_attachments(attachment):
    client = _StagingClient()
    capture = RolloutTokenCapture(
        sink=TQTokenSink(client, staging_partition="staging"),
        weight_version_fn=lambda: 7,
    )
    coords = capture.complete_call(
        capture.begin_call(
            CaptureAdmission(rollout_id="r", model_call_id="c", mode="text")
        ),
        prompt_token_ids=[10],
        generated_token_ids=[11],
        generated_logprobs=[-0.5],
        extras={
            "routed_experts": routed_experts_tensor_metadata(
                torch.ones(2, 1, 1, dtype=torch.int16)
            )
        },
        attachments=attachment,
    )
    assert coords.disposition == "capture_failed"
    assert client.puts == 0
