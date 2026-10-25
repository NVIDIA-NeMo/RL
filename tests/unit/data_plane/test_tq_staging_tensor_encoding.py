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

"""Wire-value and buffer-lifetime checks for captured-row tensor construction."""

import gc
from typing import Any

import pytest
import torch
from tensordict import TensorDict

pytest.importorskip("nemo_gym.token_id_capture.staging")

from nemo_rl.data_plane.tq_token_sink import (  # noqa: E402
    TQTokenSink,
    TQTokenSource,
    _bytes_tensor,
)
from tests.unit.data_plane.token_capture_test_fixtures import _record  # noqa: E402

pytestmark = pytest.mark.nemo_gym


class _HoldingClient:
    """Retain the original tensors, like a transport that reads them later."""

    def __init__(self) -> None:
        self.rows: dict[str, TensorDict] = {}

    def put_samples(
        self,
        *,
        sample_ids: list[str],
        partition_id: str,
        fields: TensorDict,
        tags: list[dict[str, Any]],
    ) -> None:
        self.rows[sample_ids[0]] = fields

    def get_samples(
        self,
        *,
        sample_ids: list[str],
        partition_id: str,
        select_fields: list[str],
    ) -> TensorDict:
        assert len(sample_ids) == 1
        return self.rows[sample_ids[0]].select(*select_fields)


@pytest.mark.parametrize("length", [1, 7, 2048, 65536])
def test_staged_numeric_columns_preserve_wire_bits_and_owned_storage(
    length: int,
) -> None:
    # Include signed zero, subnormals, float32 rounding and large int64 IDs.
    ids = [0, 2**31, 2**63 - 1]
    masks = [-0.0, 1.0, 1.0]
    logprobs = [-0.0, -0.1, -(2**-149)]
    record = _record(
        rollout_id="tensor_encoding",
        model_call_id="call",
        parent_call_id=None,
        prev_len=0,
        token_ids=[ids[i % len(ids)] for i in range(length)],
        token_mask=[masks[i % len(masks)] for i in range(length)],
        logprobs=[logprobs[i % len(logprobs)] for i in range(length)],
        weight_version=4,
    )
    expected = {
        "token_ids_delta": torch.tensor([record.token_ids_delta], dtype=torch.int64),
        "token_mask_delta": torch.tensor(
            [record.token_mask_delta], dtype=torch.float32
        ),
        "generation_logprobs_delta": torch.tensor(
            [record.generation_log_probs_delta], dtype=torch.float32
        ),
    }
    original_digest = record.digest
    client = _HoldingClient()
    sink = TQTokenSink(client, staging_partition="encoding_test")
    assert sink.stage(record).ok
    staged = client.rows[record.staging_key]

    # The source lists are mutable. Transport-owned buffers must not alias them.
    record.token_ids_delta[:] = [123] * length
    record.token_mask_delta[:] = [0.0] * length
    record.generation_log_probs_delta[:] = [0.0] * length
    gc.collect()
    for name, reference in expected.items():
        actual = staged[name]
        assert actual.dtype == reference.dtype
        assert actual.shape == reference.shape
        assert actual.device.type == "cpu"
        assert actual.is_contiguous()
        assert torch.equal(actual.view(torch.uint8), reference.view(torch.uint8))

    # The ordinary source recomputes and verifies the staged digest.
    source = TQTokenSource(client, staging_partition="encoding_test")
    [snapshot] = source.fetch([record.staging_key])
    assert snapshot.digest == original_digest
    assert snapshot.token_ids_delta == expected["token_ids_delta"][0].tolist()


@pytest.mark.parametrize(
    "value",
    [bytes(range(256)), b"\0", "rollout-雪/call-λ".encode(), b"x" * 4096],
    ids=["all-byte-values", "zero", "unicode", "long"],
)
def test_byte_columns_preserve_all_values_and_remain_writable(value: bytes) -> None:
    encoded = _bytes_tensor(value)
    gc.collect()
    assert encoded.dtype == torch.uint8
    assert encoded.shape == (1, len(value))
    assert encoded.is_contiguous()
    assert torch.equal(encoded, torch.tensor([list(value)], dtype=torch.uint8))
    encoded[0, 0] ^= 255
    assert encoded[0, 0].item() == value[0] ^ 255
    assert bytes(encoded[0, 1:].tolist()) == value[1:]


def test_empty_byte_column_is_rejected() -> None:
    with pytest.raises(ValueError, match="must be non-empty"):
        _bytes_tensor(b"")
