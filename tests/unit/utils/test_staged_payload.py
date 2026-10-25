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
"""The chunked JSON codec that carries refit metadata through Dynamo's admin routes."""

import json
import random
import string
from collections import OrderedDict

import pytest
import torch

from nemo_rl.utils.staged_payload import (
    STAGED_CHUNK_CHARS,
    decode_staged_payload,
    encode_staged_payload,
)

# Dynamo's system HTTP server keeps axum's default request body limit.
_DYNAMO_BODY_LIMIT = 2 * 1024 * 1024


def _incompressible_payload(num_bytes: int) -> dict:
    rng = random.Random(0)
    text = "".join(rng.choices(string.ascii_letters + string.digits, k=num_bytes))
    return {"blob": text}


def test_round_trip_preserves_structure_and_dict_order() -> None:
    payload = {
        "misc_meta": OrderedDict(
            [("z.weight", {"shape": [4, 8], "dtype": "bfloat16"}), ("a.bias", {})]
        ),
        "per_layer_params": {"layer.0": [{"global_shape": [2, 3], "pp_stage": 0}]},
    }

    staged = encode_staged_payload(payload)
    decoded = decode_staged_payload(staged.chunks, staged.sha256)

    assert decoded == json.loads(json.dumps(payload))
    assert list(decoded["misc_meta"]) == ["z.weight", "a.bias"]


def test_tuples_arrive_as_lists_like_msgspec() -> None:
    staged = encode_staged_payload({"global_shape": (2, 3)})
    assert decode_staged_payload(staged.chunks, staged.sha256) == {
        "global_shape": [2, 3]
    }


def test_large_payload_is_split_into_chunks_under_the_body_limit() -> None:
    payload = _incompressible_payload(3 * STAGED_CHUNK_CHARS)

    staged = encode_staged_payload(payload)

    assert len(staged.chunks) > 1
    for index, chunk in enumerate(staged.chunks):
        body = json.dumps(
            {
                "engine_rpc": "nrl_stage_chunk",
                "key": staged.key,
                "index": index,
                "total": len(staged.chunks),
                "data": chunk,
            }
        )
        assert len(body.encode()) < _DYNAMO_BODY_LIMIT
    assert decode_staged_payload(staged.chunks, staged.sha256) == payload


def test_each_encoding_gets_a_fresh_key() -> None:
    assert encode_staged_payload({}).key != encode_staged_payload({}).key


def test_reordered_or_corrupted_chunks_are_rejected() -> None:
    staged = encode_staged_payload(_incompressible_payload(3 * STAGED_CHUNK_CHARS))
    with pytest.raises(Exception):
        decode_staged_payload(list(reversed(staged.chunks)), staged.sha256)
    with pytest.raises(ValueError, match="digest"):
        decode_staged_payload(staged.chunks, "0" * 64)


@pytest.mark.parametrize(
    "payload",
    [{"dtype": torch.bfloat16}, {"scale": float("nan")}, {(0, 1): "tuple key"}],
)
def test_encoder_refuses_anything_that_is_not_plain_json(payload) -> None:
    with pytest.raises((TypeError, ValueError)):
        encode_staged_payload(payload)
