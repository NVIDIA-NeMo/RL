"""Properties of the deterministic refit transfer digest.

The digest must be stable across calls; sensitive to value, position, dtype,
shape, and length changes; and independent of memory layout and hashing chunk
size. Except for the mixed-device regression, tests run on CPU tensors; the
arithmetic is pure integer math with wraparound, so devices produce the same
result.
"""

import pytest
import torch

from nemo_rl.weight_sync import digest as digest_mod
from nemo_rl.weight_sync.digest import (
    compare_digests,
    digests_to_ints,
    tensor_digest,
)


def _value(tensor):
    return digests_to_ints({"x": tensor_digest(tensor)})["x"]


def test_digest_is_deterministic():
    t = torch.arange(1000, dtype=torch.int32)
    assert _value(t) == _value(t.clone())


def test_digest_changes_with_any_value():
    t = torch.arange(1000, dtype=torch.int32)
    tampered = t.clone()
    tampered[500] += 1
    assert _value(t) != _value(tampered)


def test_digest_rejects_paired_high_bit_flips():
    original = torch.zeros(16, dtype=torch.uint8)
    corrupted = original.clone()
    corrupted[7] = 0x80
    corrupted[15] = 0x80
    assert _value(original) != _value(corrupted)


def test_digest_rejects_salted_lane_permutation():
    """Closed-form permutation attack on a single commutative channel.

    With lanes [0, 0] and a linear position salt S, the salted lanes are
    [S, 2S]; corrupted lanes [S, -S] (as signed int64) salt to [2S, S] -- a
    permutation that any single per-lane-bijection + commutative-sum digest
    cannot see. The dual-channel digest must reject it.
    """
    original = torch.zeros(2, dtype=torch.int64)
    corrupted = torch.tensor(
        [-7046029254386353131, 7046029254386353131], dtype=torch.int64
    )
    assert _value(original) != _value(corrupted)


def test_digest_is_position_sensitive():
    t = torch.tensor([1, 2], dtype=torch.int64)
    swapped = torch.tensor([2, 1], dtype=torch.int64)
    assert _value(t) != _value(swapped)


def test_digest_folds_in_length():
    # An all-zero extension must still change the metadata-covered stream.
    assert _value(torch.zeros(8, dtype=torch.uint8)) != _value(
        torch.zeros(16, dtype=torch.uint8)
    )


def test_digest_covers_equal_size_dtype_metadata():
    raw = torch.arange(64, dtype=torch.int16)
    assert _value(raw.view(torch.bfloat16)) != _value(raw.view(torch.float16))


def test_digest_covers_equal_size_shape_metadata():
    tensor = torch.arange(12, dtype=torch.float32)
    assert _value(tensor.reshape(3, 4)) != _value(tensor.reshape(2, 6))


def test_digest_uses_logical_element_order():
    t = torch.arange(24, dtype=torch.float32).reshape(4, 6)
    transposed = t.t()
    assert _value(transposed) == _value(transposed.contiguous())
    assert _value(transposed) != _value(t)


def test_digest_handles_odd_storage_offset():
    # A bf16 slice at an odd element offset has 8-byte-misaligned storage,
    # which the int64 lane view must not trip over.
    base = torch.randn(65, dtype=torch.bfloat16)
    sliced = base[1:]
    assert _value(sliced) == _value(sliced.clone())


def test_digest_is_chunking_invariant(monkeypatch):
    t = torch.arange(999, dtype=torch.uint8)
    reference = _value(t)
    monkeypatch.setattr(digest_mod, "_CHUNK_LANES", 4)
    assert _value(t) == reference


def test_digests_to_ints_roundtrip_and_empty():
    assert digests_to_ints({}) == {}
    tensors = {"a": torch.ones(3), "b": torch.zeros(5)}
    ints = digests_to_ints({k: tensor_digest(v) for k, v in tensors.items()})
    assert set(ints) == {"a", "b"}
    assert all(
        isinstance(v, str) and len(v) == 32 and int(v, 16) >= 0 for v in ints.values()
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_digests_to_ints_handles_mixed_devices():
    tensors = {
        "cpu": torch.arange(8, dtype=torch.float32),
        "cuda": torch.arange(8, dtype=torch.float32, device="cuda"),
    }
    ints = digests_to_ints({name: tensor_digest(t) for name, t in tensors.items()})
    assert set(ints) == {"cpu", "cuda"}
    assert ints["cpu"] == ints["cuda"]


def test_compare_digests_reports_mismatch_and_missing():
    assert compare_digests({"a": 1, "b": 2}, {"a": 1, "b": 2}) == []
    assert compare_digests({"a": 1, "b": 2}, {"a": 1, "b": 3}) == ["b"]
    assert compare_digests({"a": 1, "b": 2}, {"a": 1}) == ["b"]
    assert compare_digests({"a": 1}, {"a": 1, "c": 9}) == ["c"]
