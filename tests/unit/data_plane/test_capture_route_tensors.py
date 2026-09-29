# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Native route attachments through Gym capture, TQ staging, and verification."""

from __future__ import annotations

import json
from collections import Counter
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from pydantic import BaseModel, ConfigDict

pytest.importorskip("nemo_gym.token_id_capture.staging")

from nemo_gym.token_id_capture.adapters.vllm import VLLMCaptureAdapter  # noqa: E402
from nemo_gym.token_id_capture.staging import (  # noqa: E402
    CaptureAdmission,
    RolloutTokenCapture,
)

from nemo_rl.data_plane import KVBatchMeta  # noqa: E402
from nemo_rl.data_plane.schema import (  # noqa: E402
    ROUTE_PASSTHROUGH_FLAG,
    ROUTE_PLAN_TAG,
    ROUTED_EXPERTS_FIELD,
)
from nemo_rl.data_plane.tq_token_sink import TQTokenSink, TQTokenSource  # noqa: E402
from nemo_rl.data_plane.worker_mixin import TQWorkerMixin  # noqa: E402
from nemo_rl.distributed.batched_data_dict import BatchedDataDict  # noqa: E402
from nemo_rl.experience.route_assembly import (  # noqa: E402
    execute_route_plan,
    verify_route_fragment_integrity,
)
from nemo_rl.experience.route_plan import (  # noqa: E402
    ROUTE_PLAN_SCHEMA_VERSION,
    RouteAssemblyPlan,
    RouteSpan,
    encode_route_plan,
)
from nemo_rl.models.generation.vllm import utils as vllm_utils  # noqa: E402
from nemo_rl.models.generation.vllm.vllm_worker_async import (  # noqa: E402
    VllmAsyncGenerationWorkerImpl,
)
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


class _Message(BaseModel):
    model_config = ConfigDict(extra="allow")
    role: str = "assistant"
    content: str = "answer"


class _Choice(BaseModel):
    index: int = 0
    message: _Message


class _Response(BaseModel):
    choices: list[_Choice]


class _PolicyWorker(TQWorkerMixin):
    def __init__(self, client):
        self._dp_client = client
        self._route_fallback_counts = Counter()

    def _routed_experts_dimensions(self):
        return 1, 2


@pytest.fixture
def forbid_route_codec(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("native capture must not encode or decode routes")

    for target in (
        "nemo_rl.utils.routed_experts_codec.encode_routed_experts",
        "nemo_rl.utils.routed_experts_codec.decode_routed_experts",
        "nemo_rl.models.generation.vllm.utils.encode_routed_experts",
        "nemo_rl.experience.route_assembly.encode_routed_experts",
    ):
        monkeypatch.setattr(target, forbidden)


@pytest.mark.parametrize("dtype", [torch.int8, torch.int16, torch.int32])
@pytest.mark.parametrize("prev_len", [0, 3], ids=["root", "continuation"])
@pytest.mark.parametrize("has_routes", [False, True], ids=["dense", "moe"])
def test_served_capture_bypasses_codec_through_route_consumers(
    dtype, prev_len, has_routes, forbid_route_codec
):
    client = _StagingClient()
    worker = SimpleNamespace(_capture_calls={})
    worker.token_capture = RolloutTokenCapture(
        sink=TQTokenSink(client, staging_partition="staging"),
        weight_version_fn=lambda: 7,
        adapter=VLLMCaptureAdapter(),
    )
    worker._capture_admission = (
        VllmAsyncGenerationWorkerImpl._capture_admission.__get__(worker)
    )
    worker._delta_align_routed_experts = (
        VllmAsyncGenerationWorkerImpl._delta_align_routed_experts
    )
    admission = {"rollout_id": "r", "model_call_id": "c", "mode": "text"}
    prompt = [10, 11]
    if prev_len:
        prompt = [10, 11, 12, 20, 21]
        admission.update(
            mode="token_in",
            parent_call_id="parent",
            prev_len=prev_len,
            required_prefix_token_ids=prompt[:prev_len],
            parent_chain_hash="1" * 64,
        )
    request = SimpleNamespace(ng_capture=admission, stream=False)
    VllmAsyncGenerationWorkerImpl._begin_request_capture(worker, request, prompt)
    prompt_routes = torch.arange((len(prompt) - 1) * 2, dtype=dtype).reshape(-1, 1, 2)
    prompt_routes[0, 0, 0] = -1
    generated_routes = torch.tensor([[[8, 9]]], dtype=dtype)
    output = SimpleNamespace(
        prompt_token_ids=prompt,
        prompt_routed_experts=prompt_routes,
        outputs=[
            SimpleNamespace(
                index=0,
                token_ids=[22],
                logprobs=[{22: SimpleNamespace(logprob=-0.25)}],
                routed_experts=generated_routes,
            )
        ],
    )
    response = _Response(choices=[_Choice(message=_Message())])
    vllm_utils.attach_token_information_to_chat_response_choices(response, output)
    if has_routes:
        vllm_utils.attach_routed_experts_to_chat_response_choices(
            response,
            output,
            device=torch.device("cpu"),
            routed_experts_dtype=dtype,
            encode_for_wire=id(request) not in worker._capture_calls,
        )
    content = VllmAsyncGenerationWorkerImpl._finish_request_capture(
        worker,
        request,
        vllm_utils.model_dump_chat_response_with_dynamic_message_fields(response),
    )
    # This is the actual JSON response boundary; no tensor or token arrays remain.
    json.dumps(content)
    assert content["choices"][0]["message"] == {
        "role": "assistant",
        "content": "answer",
    }
    coords = content["ng_commit_coords"]
    assert coords["disposition"] == "staged"
    assert client.puts == 1
    assert client.fields["token_ids_delta"].tolist() == [prompt[prev_len:] + [22]]
    source = TQTokenSource(client, staging_partition="staging")
    key = coords["staging_key"]
    deferred = source.fetch_for_finalization([key])[0]
    direct = source.fetch_for_finalization([key], include_route_fragments=True)[0]
    assert deferred.fragment is None
    if not has_routes:
        assert deferred.routed_len == 0
        assert direct.fragment is None
        return

    expected = torch.cat(
        (prompt_routes, generated_routes, torch.tensor([[[0, 1]]], dtype=dtype))
    )[prev_len:]
    fragment = direct.fragment
    assert fragment.routes.dtype == dtype
    assert torch.equal(fragment.routes, expected)
    # The sink receives the native slice, without a second route-buffer copy.
    assert (
        fragment.routes.data_ptr()
        == response.choices[0].message.routed_experts[prev_len:].data_ptr()
    )
    length = len(prompt) + 1 - prev_len
    plan = RouteAssemblyPlan(
        schema_version=ROUTE_PLAN_SCHEMA_VERSION,
        staging_partition="staging",
        spans=(RouteSpan(key, length - 1, 1, length, 1, coords["extras_digest"]),),
        cleanup_staging_keys=(key,),
        expected_token_length=length,
    )
    assembled, error = execute_route_plan(
        plan, {key: fragment}, dims=(1, 2), canonical_len=length
    )
    assert error is None
    assert torch.equal(assembled, expected.to(torch.int16))
    policy = _PolicyWorker(client)
    meta = KVBatchMeta(
        partition_id="canonical",
        task_name="train",
        sample_ids=["row"],
        fields=["input_ids", "input_lengths"],
        sequence_lengths=[length],
        tags=[{ROUTE_PLAN_TAG: encode_route_plan(plan)}],
        extra_info={ROUTE_PASSTHROUGH_FLAG: True},
    )
    batch = BatchedDataDict(
        {
            "input_ids": torch.zeros((1, length), dtype=torch.long),
            "input_lengths": torch.tensor([length]),
        }
    )
    replay = policy._maybe_assemble_routed_experts(meta, batch)[ROUTED_EXPERTS_FIELD]
    assert torch.equal(replay[0], assembled)
    assert not policy._route_fallback_counts


@pytest.mark.parametrize("dtype", [torch.int8, torch.int16, torch.int32])
def test_capture_stages_and_verifies_native_routes_without_codec(
    dtype, forbid_route_codec
):
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
        schema_version=ROUTE_PLAN_SCHEMA_VERSION,
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
