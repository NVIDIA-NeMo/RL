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
import pickle

import numpy as np
import pytest
import torch

from nemo_rl.data_plane.nixl import blob_format


def _roundtrip(keys, values):
    p = blob_format.plan(keys, values)
    buf = np.empty(p.nbytes, dtype=np.uint8)
    blob_format.write(p, memoryview(buf))
    idx = blob_format.read_index(memoryview(buf))
    assert [e.key for e in idx] == [str(k) for k in keys]
    assert idx == p.entries
    out = [
        blob_format.materialize(
            blob_format.decode_entry(memoryview(buf)[e.off : e.off + e.len], e)
        )
        for e in idx
    ]
    # the per-key meta TQ stores must reconstruct identically
    out2 = [
        blob_format.materialize(
            blob_format.decode_entry(
                memoryview(buf)[e.off : e.off + e.len],
                blob_format.Entry.from_meta(e.key, e.meta()),
            )
        )
        for e in idx
    ]
    return p, out, out2


def _eq(a, b):
    if isinstance(a, torch.Tensor):
        return a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b)
    if isinstance(a, np.ndarray):
        return a.dtype == b.dtype and np.array_equal(a, b)
    return a == b


def test_dense_tensors_raw_bytes():
    vals = [
        torch.arange(10),
        torch.randn(3, 4),
        torch.zeros(0),
        torch.tensor(7.5, dtype=torch.float64),
    ]
    p, out, out2 = _roundtrip(["a", "b", "c", "d"], vals)
    for v, o, o2 in zip(vals, out, out2):
        assert _eq(v, o) and _eq(v, o2)
    # raw: entry length is exactly the tensor's byte size; aligned offsets
    for v, e in zip(vals, p.entries):
        assert (
            e.len == v.numel() * v.element_size() and e.off % 64 == 0 and e.kind == "t"
        )


def test_dtypes_bf16_bool_int8_fp16():
    vals = [
        torch.randn(5, dtype=torch.bfloat16),
        torch.tensor([True, False, True]),
        torch.arange(-3, 3, dtype=torch.int8),
        torch.randn(2, 2, dtype=torch.float16),
    ]
    _, out, _ = _roundtrip(list("abcd"), vals)
    for v, o in zip(vals, out):
        assert o.dtype == v.dtype and torch.equal(
            o.view(torch.int8) if v.dtype is torch.bfloat16 else o,
            v.view(torch.int8) if v.dtype is torch.bfloat16 else v,
        )


def test_noncontiguous_and_views():
    base = torch.arange(12).view(3, 4)
    vals = [base.t(), base[:, 1], base[1:, ::2]]
    _, out, _ = _roundtrip(list("abc"), vals)
    for v, o in zip(vals, out):
        assert _eq(v.contiguous(), o)
        assert o.is_contiguous()


def test_numpy_and_objects():
    vals = [
        np.arange(6, dtype=np.int32).reshape(2, 3),
        {"s": "hello", "n": 3},
        "plain string",
        [1, 2, 3],
        None,
    ]
    p, out, _ = _roundtrip(list(range(5)), vals)
    assert np.array_equal(out[0], vals[0]) and out[0].dtype == np.int32
    assert out[1:] == vals[1:]
    assert [e.kind for e in p.entries] == ["n", "p", "p", "p", "p"]
    assert p.entries[1].len == len(
        pickle.dumps(vals[1], protocol=pickle.HIGHEST_PROTOCOL)
    )


def test_object_numpy_array_is_pickled():
    arr = np.array(["x", {"k": 1}], dtype=object)
    p, out, _ = _roundtrip(["o"], [arr])
    assert p.entries[0].kind == "p"
    assert list(out[0]) == list(arr)


def test_jagged_rows_as_separate_entries():
    rows = [torch.arange(3), torch.arange(7), torch.arange(1)]
    _, out, _ = _roundtrip(["0@f", "1@f", "2@f"], rows)
    for r, o in zip(rows, out):
        assert _eq(r, o)


def test_materialized_values_do_not_alias_buffer():
    p = blob_format.plan(["k"], [torch.ones(4)])
    buf = np.empty(p.nbytes, dtype=np.uint8)
    blob_format.write(p, memoryview(buf))
    e = p.entries[0]
    view = blob_format.decode_entry(memoryview(buf)[e.off : e.off + e.len], e)
    out = blob_format.materialize(view)
    buf[e.off : e.off + e.len] = 0
    assert torch.equal(view, torch.zeros(4))  # the view aliases the buffer
    assert torch.equal(out, torch.ones(4))  # the materialized copy does not


@pytest.mark.gpu
def test_cuda_tensor_is_copied_to_host_bytes():
    v = torch.arange(8, device="cuda").view(2, 4) * 3
    _, out, _ = _roundtrip(["g"], [v])
    assert out[0].device.type == "cpu" and torch.equal(out[0], v.cpu())


def test_footer_and_version():
    p = blob_format.plan(["k"], [torch.ones(2)])
    buf = np.zeros(p.nbytes, dtype=np.uint8)
    assert not blob_format.footer_is_valid(memoryview(buf))
    blob_format.write(p, memoryview(buf))
    assert blob_format.footer_is_valid(memoryview(buf))
    buf[-blob_format.FOOTER_SIZE] ^= (
        0xFF  # magic: first field of the (v3, 32-byte) footer
    )
    assert not blob_format.footer_is_valid(memoryview(buf))
    with pytest.raises(ValueError):
        blob_format.read_index(memoryview(buf))


def test_write_needs_room_and_length_mismatch():
    p = blob_format.plan(["k"], [torch.ones(2)])
    with pytest.raises(ValueError):
        blob_format.write(p, memoryview(np.empty(p.nbytes - 1, dtype=np.uint8)))
    with pytest.raises(ValueError):
        blob_format.plan(["a"], [1, 2])
