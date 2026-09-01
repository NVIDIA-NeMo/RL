"""Properties of the deterministic refit transfer digest.

The digest must be a pure function of a tensor's logical byte stream: stable
across calls, sensitive to value, position, and length changes, and identical
for any dtype reinterpretation, memory layout, or hashing chunk size that
preserves those bytes. All tests run on CPU tensors; the arithmetic is pure
integer math with wraparound, so device changes cannot alter results.
"""

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


def test_digest_is_position_sensitive():
    t = torch.tensor([1, 2], dtype=torch.int64)
    swapped = torch.tensor([2, 1], dtype=torch.int64)
    assert _value(t) != _value(swapped)


def test_digest_folds_in_length():
    # Trailing zero bytes extend the stream but leave every lane sum equal,
    # so only the length fold distinguishes these.
    assert _value(torch.zeros(8, dtype=torch.uint8)) != _value(
        torch.zeros(16, dtype=torch.uint8)
    )


def test_digest_matches_byte_reinterpretation():
    t = torch.randn(64, dtype=torch.bfloat16)
    assert _value(t) == _value(t.view(torch.uint8))


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
    monkeypatch.setattr(digest_mod, "_POW_CACHE", {})
    assert _value(t) == reference


def test_digests_to_ints_roundtrip_and_empty():
    assert digests_to_ints({}) == {}
    tensors = {"a": torch.ones(3), "b": torch.zeros(5)}
    ints = digests_to_ints({k: tensor_digest(v) for k, v in tensors.items()})
    assert set(ints) == {"a", "b"}
    assert all(0 <= v < (1 << 64) for v in ints.values())


def test_compare_digests_reports_mismatch_and_missing():
    assert compare_digests({"a": 1, "b": 2}, {"a": 1, "b": 2}) == []
    assert compare_digests({"a": 1, "b": 2}, {"a": 1, "b": 3}) == ["b"]
    assert compare_digests({"a": 1, "b": 2}, {"a": 1}) == ["b"]
    assert compare_digests({"a": 1}, {"a": 1, "c": 9}) == ["c"]
