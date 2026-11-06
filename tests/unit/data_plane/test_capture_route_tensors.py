# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native route capture adapted to the prefix branch's existing source APIs."""

from dataclasses import replace

import pytest
import torch

from nemo_rl.data_plane.schema import ROUTE_ENCODING_ENVELOPE, ROUTE_ENCODING_TENSOR
from nemo_rl.data_plane.tq_token_sink import TQTokenSink, TQTokenSource
from nemo_rl.experience.route_assembly import verify_route_fragment_integrity
from nemo_rl.utils.routed_experts_codec import (
    ROUTED_EXPERTS_STAGING_ATTACHMENT,
    RoutedExpertsStagingPayload,
    routed_experts_tensor_metadata,
)
from tests.unit.data_plane.test_direct_prefix_batch import MemoryClient, record, stage


@pytest.mark.parametrize("dtype", [torch.int8, torch.int16, torch.int32])
@pytest.mark.parametrize("prev_len", [0, 3])
def test_terminal_delta_keeps_native_tensor_and_boundary(
    dtype: torch.dtype, prev_len: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Load optional generation dependencies only for terminal-worker coverage.
    from nemo_rl.models.generation.vllm.vllm_worker_async import (
        VllmAsyncGenerationWorkerImpl,
    )

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("native terminal alignment must not use the route codec")

    monkeypatch.setattr(
        "nemo_rl.utils.routed_experts_codec.encode_routed_experts", forbidden
    )
    monkeypatch.setattr(
        "nemo_rl.models.generation.vllm.vllm_worker_async.decode_routed_experts",
        forbidden,
    )
    prompt_len = prev_len + 2
    routes = torch.arange((prompt_len + 2) * 4, dtype=dtype).reshape(
        prompt_len + 2, 2, 2
    )
    payload = {"choices": [{"message": {"routed_experts": routes}}]}
    attached = VllmAsyncGenerationWorkerImpl._delta_align_routed_experts(
        payload, prev_len=prev_len, prompt_len=prompt_len, generated_len=2
    )
    metadata = payload["choices"][0]["message"]["routed_experts"]
    expected = routes[max(0, prev_len - 1) :]
    assert metadata == routed_experts_tensor_metadata(expected)
    assert torch.equal(attached.tensor_for_payload(metadata), expected)


@pytest.mark.parametrize("dtype", [torch.int8, torch.int16, torch.int32])
@pytest.mark.parametrize("direct", [False, True])
@pytest.mark.parametrize("native", [False, True])
def test_native_prefix_roundtrip_and_tamper_detection(
    dtype: torch.dtype, direct: bool, native: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    if native:

        def forbidden(*args: object, **kwargs: object) -> None:
            raise AssertionError(
                "native routes must not encode or decode a wire envelope"
            )

        for target in [
            "nemo_rl.utils.routed_experts_codec.encode_routed_experts",
            "nemo_rl.utils.routed_experts_codec.decode_routed_experts",
            "nemo_rl.data_plane.tq_token_sink.decode_routed_experts",
            "nemo_rl.data_plane.direct_prefix_batch.decode_routed_experts",
            "nemo_rl.experience.route_assembly.encode_routed_experts",
        ]:
            monkeypatch.setattr(target, forbidden)
    rows = []
    attachments = []
    expected = []
    for i, length in enumerate([3, 7]):
        routes = (torch.arange(length * 4) % 100).reshape(length, 2, 2).to(dtype)
        payload = RoutedExpertsStagingPayload(routes, native=native)
        rows.append(
            record(i, length, extras={"routed_experts": payload.encoded, "note": i})
        )
        attachments.append({ROUTED_EXPERTS_STAGING_ATTACHMENT: payload})
        expected.append(routes.clone())
        routes.zero_()
    client, keys = stage(rows, direct=direct, attachments=attachments)
    fetched = TQTokenSource(client, staging_partition="prefix").fetch_for_finalization(
        keys, include_route_fragments=True
    )
    for item, original, route in zip(fetched, rows, expected, strict=True):
        assert item.snapshot.digest == original.digest
        assert item.fragment.encoding == (
            ROUTE_ENCODING_TENSOR if native else ROUTE_ENCODING_ENVELOPE
        )
        assert torch.equal(item.fragment.routes, route)
        assert verify_route_fragment_integrity(
            item.fragment,
            extras_digest_version=original.extras_digest_version,
            expected_extras_digest=original.extras_digest,
        )
        changed = item.fragment.routes.clone()
        changed[0, 0, 0] += 1
        assert not verify_route_fragment_integrity(
            replace(item.fragment, routes=changed),
            extras_digest_version=original.extras_digest_version,
            expected_extras_digest=original.extras_digest,
        )


@pytest.mark.parametrize("tamper", ["tensor", "descriptor"])
def test_owned_native_attachment_rejects_changes_before_publication(
    tamper: str,
) -> None:
    payload = RoutedExpertsStagingPayload(
        torch.ones(3, 2, 2, dtype=torch.int16), native=True
    )
    descriptor = payload.encoded
    if tamper == "tensor":
        payload.tensor_for_payload(descriptor)[0, 0, 0] = 2
    else:
        descriptor["sha256"] = "0" * 64
    client = MemoryClient()
    with pytest.raises(ValueError):
        stage(
            [record(0, 3, extras={"routed_experts": descriptor})],
            direct=True,
            client=client,
            attachments=[{ROUTED_EXPERTS_STAGING_ATTACHMENT: payload}],
        )
    assert not client.puts


@pytest.mark.parametrize("index", [0, 1])
def test_terminal_native_tensor_attachment_supports_parent_boundary(index: int) -> None:
    routes = torch.arange((3 + index) * 4, dtype=torch.int16).reshape(3 + index, 2, 2)
    row = record(
        index, 3, extras={"routed_experts": routed_experts_tensor_metadata(routes)}
    )
    client = MemoryClient()
    sink = TQTokenSink(client, staging_partition="prefix")
    fields, _ = sink._encode_record(row, attachments={"routed_experts": routes})
    assert fields["routed_experts_encoding"].item() == ROUTE_ENCODING_TENSOR
    assert torch.equal(fields["routed_experts"][0], routes)
