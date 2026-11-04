# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Bulk route validation preserves mixed-dtype behavior and cache ownership."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from nemo_rl.models.generation import generation_cut_capture
from nemo_rl.models.generation.generation_cut_capture import _RequestCaptureState


def state():
    return SimpleNamespace(
        engine_route_piece_count=0,
        engine_route_last_piece=None,
        engine_route_buffer=None,
        engine_route_rows=0,
        routed_experts_dtype=torch.int16,
    )


@pytest.mark.parametrize(
    "dtypes", [(np.int16, np.int16), (np.int8, np.int32), (np.uint64, np.int64)]
)
def test_bulk_and_compatibility_paths_preserve_values_and_incremental_cache(dtypes):
    pieces = [np.full((2, 3, 4), i + 1, dtype=dt) for i, dt in enumerate(dtypes)]
    s = state()
    actual = _RequestCaptureState.absorb_engine_route_pieces(s, pieces)
    expected = np.concatenate(pieces).astype(np.int16)
    assert np.array_equal(actual.numpy(), expected)
    saved = actual.clone()
    pieces.append(np.full((1, 3, 4), 3, dtype=np.int16))
    second = _RequestCaptureState.absorb_engine_route_pieces(s, pieces)
    assert np.array_equal(second.numpy(), np.concatenate(pieces).astype(np.int16))
    assert torch.equal(saved, second[:4])
    assert s.engine_route_piece_count == 3


@pytest.mark.parametrize(
    "bad",
    [
        np.zeros((1, 3, 4), dtype=np.float32),
        np.zeros((1, 2, 4), dtype=np.int16),
        np.zeros((1, 3), dtype=np.int16),
        np.zeros((1, 0, 4), dtype=np.int16),
    ],
)
def test_bad_new_fragment_never_advances_cache(bad):
    first = np.ones((1, 3, 4), dtype=np.int16)
    s = state()
    original = _RequestCaptureState.absorb_engine_route_pieces(s, [first]).clone()
    with pytest.raises(ValueError):
        _RequestCaptureState.absorb_engine_route_pieces(s, [first, bad])
    assert s.engine_route_piece_count == s.engine_route_rows == 1
    assert torch.equal(original, torch.from_numpy(s.engine_route_buffer[:1]))


@pytest.mark.parametrize(
    "source_dtype,target_dtype,maximum",
    [
        (np.uint8, torch.int8, 127),
        (np.uint8, torch.int16, 255),
        (np.uint16, torch.int16, 32767),
        (np.uint16, torch.int32, 65535),
    ],
)
def test_unsigned_routes_use_bulk_path_and_preserve_incremental_values(
    monkeypatch, source_dtype, target_dtype, maximum
):
    counts = {}

    def count(name: str, value: int) -> None:
        counts[name] = counts.get(name, 0) + value

    monkeypatch.setattr(generation_cut_capture, "prefix_cut_count", count)
    s = state()
    s.routed_experts_dtype = target_dtype
    pieces = [np.full((1, 3, 4), value, dtype=source_dtype) for value in (0, maximum)]
    first = _RequestCaptureState.absorb_engine_route_pieces(s, pieces).clone()
    pieces.append(np.full((2, 3, 4), maximum // 2, dtype=source_dtype))
    second = _RequestCaptureState.absorb_engine_route_pieces(s, pieces)
    assert second.dtype == target_dtype
    assert torch.equal(first, second[:2])
    assert second[:, 0, 0].tolist() == [0, maximum, maximum // 2, maximum // 2]
    assert s.engine_route_piece_count == 3
    assert s.engine_route_rows == 4
    assert counts["route_history_bulk_fast_calls"] == 2
    assert counts.get("route_history_bulk_fallbacks", 0) == 0


@pytest.mark.parametrize(
    "source_dtype,target_dtype,bad_value",
    [
        (np.uint8, torch.int8, 128),
        (np.uint8, torch.int8, 255),
        (np.uint16, torch.int16, 32768),
        (np.uint16, torch.int16, 65535),
        (np.int16, torch.int8, -129),
        (np.uint64, torch.int32, 2**64 - 1),
        (np.int64, torch.int32, -(2**63)),
    ],
)
def test_route_overflow_rejected_before_cache_mutation(
    source_dtype, target_dtype, bad_value
):
    s = state()
    s.routed_experts_dtype = target_dtype
    first = np.ones((1, 3, 4), dtype=source_dtype)
    expected = _RequestCaptureState.absorb_engine_route_pieces(s, [first]).clone()
    original_buffer = s.engine_route_buffer
    with pytest.raises(ValueError, match="do not fit checkpoint dtype"):
        _RequestCaptureState.absorb_engine_route_pieces(
            s, [first, np.full((1, 3, 4), bad_value, dtype=source_dtype)]
        )
    assert s.engine_route_buffer is original_buffer
    assert s.engine_route_piece_count == s.engine_route_rows == 1
    assert s.engine_route_last_piece is first
    assert torch.equal(expected, torch.from_numpy(original_buffer[:1]))


@pytest.mark.parametrize("bad_dtype", [np.bool_, np.float32, object])
@pytest.mark.parametrize("bad_first", [False, True])
def test_mixed_noninteger_fragments_remain_rejected(bad_dtype, bad_first):
    good = np.zeros((1, 3, 4), dtype=np.uint8)
    bad = np.ones((1, 3, 4), dtype=bad_dtype)
    pieces = [bad, good] if bad_first else [good, bad]
    s = state()
    with pytest.raises(ValueError, match="must share shape"):
        _RequestCaptureState.absorb_engine_route_pieces(s, pieces)
    assert s.engine_route_buffer is None
    assert s.engine_route_rows == s.engine_route_piece_count == 0


def test_signed_missing_route_marker_is_preserved():
    s = state()
    s.routed_experts_dtype = torch.int8
    pieces = [np.full((1, 3, 4), value, dtype=np.int8) for value in (-1, 127)]
    actual = _RequestCaptureState.absorb_engine_route_pieces(s, pieces)
    assert actual[:, 0, 0].tolist() == [-1, 127]
