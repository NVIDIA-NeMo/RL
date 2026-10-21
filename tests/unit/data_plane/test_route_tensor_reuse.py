"""CPU regressions for record-bound route tensors and canonical terminal staging."""

from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch

pytest.importorskip("nemo_gym.token_id_capture.staging.capture")

from nemo_gym.token_id_capture.adapters.vllm import VLLMCaptureAdapter  # noqa: E402
from nemo_gym.token_id_capture.staging.capture import RolloutTokenCapture  # noqa: E402
from nemo_gym.token_id_capture.staging.records import (  # noqa: E402
    CaptureAdmission,
    StageResult,
)

from nemo_rl.data_plane.tq_token_sink import (  # noqa: E402
    TQTokenSink,
    _row_to_base_snapshot,
    _row_to_route_fragment,
)
from nemo_rl.experience.route_assembly import verify_route_fragment_integrity  # noqa: E402
from nemo_rl.models.generation.vllm.vllm_worker_async import (  # noqa: E402
    VllmAsyncGenerationWorkerImpl,
)
from nemo_rl.utils import routed_experts_codec  # noqa: E402
from nemo_rl.utils.routed_experts_codec import (  # noqa: E402
    ROUTED_EXPERTS_STAGING_ATTACHMENT,
    RoutedExpertsStagingPayload,
    encode_routed_experts,
)

pytestmark = pytest.mark.nemo_gym


class _Client:
    def __init__(self):
        self.puts = []
        self.clears = []
        self.fail = False

    def put_samples(self, **kwargs):
        self.puts.append(kwargs)
        if self.fail:
            raise RuntimeError("injected put failure")

    def clear_samples(self, **kwargs):
        self.clears.extend(kwargs["sample_ids"])


class _RecordSink:
    def stage(self, record, **kwargs):
        self.record = record
        return StageResult(ok=True, staging_key=record.staging_key)


def _record(*, dtype=torch.int16, route_length=4, routed=True, legacy_list=False):
    recorder = _RecordSink()
    capture = RolloutTokenCapture(sink=recorder, weight_version_fn=lambda: 7)
    call = capture.begin_call(
        CaptureAdmission(rollout_id="rollout", model_call_id="call", mode="text")
    )
    payload = RoutedExpertsStagingPayload(
        torch.arange(route_length * 4).reshape(route_length, 2, 2).to(dtype)
    )
    coords = capture.complete_call(
        call,
        prompt_token_ids=[1, 2],
        generated_token_ids=[3, 4],
        generated_logprobs=[-0.25, -0.5],
        extras={
            "routed_experts": payload.tensor_for_payload(payload.encoded).tolist()
            if legacy_list
            else payload.encoded
        }
        if routed
        else None,
    )
    assert coords.disposition == "staged"
    return recorder.record, payload


def _unexpected_decode(*args, **kwargs):
    pytest.fail("staging must reuse the tensor instead of decoding the envelope")


@pytest.mark.parametrize("dtype", [torch.int8, torch.int16, torch.int32])
@pytest.mark.parametrize("media", [False, True])
def test_reused_route_fields_and_digests_match_legacy_staging(
    monkeypatch, dtype, media
):
    client = _Client()
    sink = TQTokenSink(
        client,
        staging_partition="test",
        capture_media=media,
        media_pixel_dtype=torch.bfloat16 if media else None,
    )
    record, payload = _record(dtype=dtype)
    attachments = (
        {
            "imgs": torch.zeros(1, 1, 12, dtype=torch.bfloat16),
            "imgs_sizes": torch.tensor([[2, 2]], dtype=torch.int32),
        }
        if media
        else {}
    )
    assert sink.stage(record, attachments=attachments or None).ok
    expected = client.puts[-1]
    monkeypatch.setattr(
        routed_experts_codec, "decode_routed_experts", _unexpected_decode
    )
    assert sink.stage(
        record, attachments={**attachments, ROUTED_EXPERTS_STAGING_ATTACHMENT: payload}
    ).ok
    actual = client.puts[-1]
    assert actual["tags"] == expected["tags"]
    assert actual["sample_ids"] == expected["sample_ids"]
    assert actual["fields"].keys() == expected["fields"].keys()
    assert all(
        torch.equal(actual["fields"][key], expected["fields"][key])
        for key in actual["fields"].keys()
    )
    row = actual["fields"]
    assert _row_to_base_snapshot(row).digest == record.digest
    assert verify_route_fragment_integrity(
        _row_to_route_fragment(row),
        extras_digest_version=record.extras_digest_version,
        expected_extras_digest=record.extras_digest,
    )


@pytest.mark.parametrize(
    "failure", ["mismatch", "mutated", "missing", "shape", "type", "unknown_media"]
)
def test_invalid_attachment_fails_before_put(failure):
    client = _Client()
    sink = TQTokenSink(client, staging_partition="test")
    record, payload = _record(route_length=3 if failure == "shape" else 4)
    if failure == "mismatch":
        payload = RoutedExpertsStagingPayload(torch.ones(4, 2, 2, dtype=torch.int16))
    elif failure == "mutated":
        payload.tensor_for_payload(payload.encoded).add_(1)
    elif failure == "missing":
        record = record.model_copy(update={"extras": None})
    elif failure == "type":
        payload = torch.zeros(4, 2, 2)
    attachments = {ROUTED_EXPERTS_STAGING_ATTACHMENT: payload}
    if failure == "unknown_media":
        attachments["not_media"] = torch.zeros(1)
    assert not sink.stage(record, attachments=attachments).ok
    assert not client.puts


def test_failed_put_keeps_existing_cleanup_and_attachment_can_retry(monkeypatch):
    client = _Client()
    sink = TQTokenSink(client, staging_partition="test")
    record, payload = _record()
    monkeypatch.setattr(
        routed_experts_codec, "decode_routed_experts", _unexpected_decode
    )
    attachments = {ROUTED_EXPERTS_STAGING_ATTACHMENT: payload}
    client.fail = True
    assert not sink.stage(record, attachments=attachments).ok
    assert client.clears == [record.staging_key]
    client.fail = False
    assert sink.stage(record, attachments=attachments).ok


@pytest.mark.parametrize("legacy_list", [False, True])
def test_no_attachment_keeps_legacy_staging(legacy_list):
    client = _Client()
    sink = TQTokenSink(client, staging_partition="test")
    record, _ = _record(legacy_list=legacy_list)
    assert sink.stage(record).ok
    assert verify_route_fragment_integrity(
        _row_to_route_fragment(client.puts[-1]["fields"]),
        extras_digest_version=record.extras_digest_version,
        expected_extras_digest=record.extras_digest,
    )


@pytest.mark.parametrize("prev_len", [0, 1, 2])
def test_delta_alignment_preserves_boundary_and_wire_dtype(prev_len):
    routes = torch.arange(4 * 4, dtype=torch.int8).reshape(4, 2, 2)
    content = {
        "choices": [{"message": {"routed_experts": encode_routed_experts(routes)}}]
    }
    payload = VllmAsyncGenerationWorkerImpl._delta_align_routed_experts(
        content,
        prev_len=prev_len,
        prompt_len=2,
        generated_len=2,
    )
    encoded = content["choices"][0]["message"]["routed_experts"]
    start = max(0, prev_len - 1)
    assert payload.encoded == encoded == encode_routed_experts(routes[start:])
    assert payload.tensor_for_payload(encoded).dtype == torch.int8


@pytest.mark.parametrize("reuse", [False, True])
def test_terminal_capture_forwards_tensor_through_real_gym(monkeypatch, reuse):
    client = _Client()
    sink = TQTokenSink(client, staging_partition="test")
    capture = RolloutTokenCapture(
        sink=sink, weight_version_fn=lambda: 7, adapter=VLLMCaptureAdapter()
    )
    call = capture.begin_call(
        CaptureAdmission(rollout_id="rollout", model_call_id="call", mode="text")
    )
    request = object()
    state = SimpleNamespace(call=call, prompt_token_ids=[1, 2], media=None)
    worker = object.__new__(VllmAsyncGenerationWorkerImpl)
    worker.token_capture = capture
    worker._capture_calls = {id(request): state}
    decodes = []
    decode = routed_experts_codec.decode_routed_experts

    def counted(*args, **kwargs):
        decodes.append(1)
        return decode(*args, **kwargs)

    monkeypatch.setattr(routed_experts_codec, "decode_routed_experts", counted)
    if not reuse:

        def legacy(payload, **kwargs):
            VllmAsyncGenerationWorkerImpl._delta_align_routed_experts(payload, **kwargs)

        worker._delta_align_routed_experts = legacy
    content = {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {
                    "generation_token_ids": [3, 4],
                    "generation_log_probs": [-0.25, -0.5],
                    "routed_experts": encode_routed_experts(
                        torch.zeros(4, 2, 2, dtype=torch.int16)
                    ),
                },
            }
        ]
    }
    result = worker._finish_request_capture(request, deepcopy(content))
    assert result["ng_commit_coords"]["disposition"] == "staged"
    assert len(decodes) == (1 if reuse else 2)
    assert len(client.puts) == 1
    assert "routed_experts" not in result["choices"][0]["message"]


@pytest.mark.parametrize("routed", [False, True])
def test_terminal_preserves_media_and_route_free_attachment_identity(routed):
    media = {"imgs": torch.zeros(1), "imgs_sizes": torch.zeros(1)}
    received = []

    def complete(call, payload, *, attachments):
        received.append(attachments)
        return SimpleNamespace(model_dump=lambda: {})

    request = object()
    worker = object.__new__(VllmAsyncGenerationWorkerImpl)
    worker.token_capture = SimpleNamespace(
        adapter=SimpleNamespace(extract_generation=lambda payload: ([3, 4], [])),
        complete_call_from_response=complete,
    )
    worker._capture_calls = {
        id(request): SimpleNamespace(
            call=SimpleNamespace(admission=SimpleNamespace(prev_len=0)),
            prompt_token_ids=[1, 2],
            media=SimpleNamespace(tensors=media, items=[]),
        )
    }
    message = (
        {
            "routed_experts": encode_routed_experts(
                torch.zeros(4, 2, 2, dtype=torch.int16)
            )
        }
        if routed
        else {}
    )
    worker._finish_request_capture(request, {"choices": [{"message": message}]})
    if routed:
        assert received[0]["imgs"] is media["imgs"]
        assert isinstance(
            received[0][ROUTED_EXPERTS_STAGING_ATTACHMENT], RoutedExpertsStagingPayload
        )
    else:
        assert received[0] is media
