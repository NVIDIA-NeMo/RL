# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
"""Chunk-only preparation preserves legacy route bytes, ownership and boundaries."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from nemo_rl.models.generation import generation_cut_capture as capture
from nemo_rl.models.generation.generation_cut_capture import _RequestCaptureState
from nemo_rl.models.generation.generation_cut_routes import RoutePieceIndex


def state(
    *,
    prev: int,
    carry: int,
    frozen: int,
    staged: int = 0,
    restored: int | None = None,
    dtype: torch.dtype = torch.int16,
) -> _RequestCaptureState:
    restored_routes = (
        None
        if restored is None
        else torch.arange(restored * 6, dtype=dtype).reshape(restored, 2, 3)
    )
    if restored_routes is not None:
        restored_routes[-1] = -1
    return _RequestCaptureState(
        call=SimpleNamespace(admission=SimpleNamespace(prev_len=prev)),
        prompt_token_ids=list(range(prev + carry)),
        capture_routed_experts=True,
        routed_experts_dtype=dtype,
        restored_routed_experts=restored_routes,
        staged_token_count=staged,
        frozen_cut=("cp", frozen),
    )


def fragment(values: np.ndarray) -> list[np.ndarray]:
    pieces = [values[:0]]
    start = 0
    for length in (1, 3, 0, 2, 5):
        pieces.append(values[start : start + length])
        start += length
    pieces.append(values[start:])
    return pieces


@pytest.mark.parametrize(
    "prev,carry,restored",
    [(0, 0, None), (0, 3, None), (2, 0, None), (7, 4, None), (0, 9, 5), (3, 7, 11)],
)
@pytest.mark.parametrize(
    "frozen,staged", [(1, 0), (4, 0), (4, 1), (9, 4), (4, 4), (0, 0)]
)
@pytest.mark.parametrize(
    "source_dtype,target_dtype",
    [(np.uint8, torch.int8), (np.int16, torch.int16), (np.dtype(">u2"), torch.int16)],
)
@pytest.mark.parametrize("kind", ["pieces", "tensor", "terminal"])
def test_chunk_bytes_match_full_assembly(
    prev, carry, restored, frozen, staged, source_dtype, target_dtype, kind
):
    old = state(
        prev=prev,
        carry=carry,
        frozen=frozen,
        staged=staged,
        restored=restored,
        dtype=target_dtype,
    )
    new = state(
        prev=prev,
        carry=carry,
        frozen=frozen,
        staged=staged,
        restored=restored,
        dtype=target_dtype,
    )
    rows = max(
        0, prev + (carry + frozen - 1 if restored is None else restored - 1 + frozen)
    )
    # Include future rows beyond the token freeze; neither path may stage them.
    raw = (np.arange((rows + 3) * 6).reshape(rows + 3, 2, 3) % 100).astype(source_dtype)
    pieces = fragment(raw)
    if kind == "pieces":
        expected, _ = old.frozen_cut_routed_experts(
            old.absorb_engine_route_pieces(pieces)
        )
        actual = new.frozen_cut_route_chunk(pieces)
    else:
        tensor = torch.from_numpy(raw.astype(np.int16))
        if kind == "terminal":
            old.terminal_routed_experts = tensor
            new.terminal_routed_experts = tensor
        expected, _ = old.frozen_cut_routed_experts(tensor)
        actual = new.frozen_cut_route_chunk(tensor)
    if expected is None:
        assert actual is None
    else:
        assert actual is not None and actual.dtype == target_dtype
        assert actual.is_contiguous()
        assert actual.numpy().tobytes() == expected.numpy().tobytes()
        assert actual.shape == expected.shape
    assert new.engine_route_buffer is None


def test_empty_prompt_single_token_preserves_final_marker() -> None:
    instance = state(prev=0, carry=0, frozen=1, dtype=torch.int8)
    actual = instance.frozen_cut_route_chunk([np.empty((0, 2, 3), dtype=np.uint8)])
    assert torch.equal(actual, torch.full((1, 2, 3), -1, dtype=torch.int8))


def test_failed_cut_retry_and_later_cut_select_only_required_rows(monkeypatch) -> None:
    instance = state(prev=0, carry=2, frozen=3)
    raw = np.arange(20 * 6, dtype=np.int16).reshape(20, 2, 3)
    pieces = fragment(raw)
    lengths = []
    concatenate = capture.concatenate_route_pieces

    def observed(selected, **kwargs):
        lengths.append(sum(len(piece) for piece in selected))
        return concatenate(selected, **kwargs)

    monkeypatch.setattr(capture, "concatenate_route_pieces", observed)
    first = instance.frozen_cut_route_chunk(pieces)
    instance.rollback_frozen_cut("cp")
    instance.frozen_cut = ("retry", 7)
    retried = instance.frozen_cut_route_chunk(pieces)
    assert torch.equal(retried[:-1], torch.from_numpy(raw[:8]))
    instance.seal_frozen_cut("retry")
    instance.frozen_cut = ("next", 10)
    delta = instance.frozen_cut_route_chunk(pieces)
    assert torch.equal(delta, torch.from_numpy(raw[8:11]))
    assert lengths == [4, 8, 3]
    frozen_copy = delta.clone()
    raw.fill(99)
    assert torch.equal(delta, frozen_copy)
    assert first is not None
    assert instance.engine_route_buffer is None


def test_index_handles_partial_fragments_zero_lengths_growth_and_append_checks() -> (
    None
):
    index = RoutePieceIndex()
    raw = np.arange(60).reshape(10, 2, 3)
    pieces = [raw[:0], raw[:4], raw[4:4], raw[4:7]]
    assert np.array_equal(
        np.concatenate(index.select(pieces, start=2, stop=6)), raw[2:6]
    )
    pieces.append(raw[7:])
    assert np.array_equal(
        np.concatenate(index.select(pieces, start=6, stop=10)), raw[6:10]
    )
    assert index.rows == 10 and index.count == 5
    with pytest.raises(ValueError, match="shorter"):
        index.select(pieces[:-1], start=0, stop=2)
    with pytest.raises(ValueError, match="replaced"):
        index.select([*pieces[:-1], pieces[-1].copy()], start=0, stop=2)
    with pytest.raises(ValueError, match="fewer routed"):
        index.select(pieces, start=9, stop=11)


@pytest.mark.parametrize(
    "bad",
    [
        np.full((1, 2, 3), 128, dtype=np.uint8),
        np.zeros((1, 2, 3), dtype=np.float32),
        np.zeros((1, 0, 3), dtype=np.int16),
    ],
)
def test_invalid_selected_fragments_are_rejected(bad) -> None:
    instance = state(prev=0, carry=1, frozen=1, dtype=torch.int8)
    with pytest.raises(ValueError):
        instance.frozen_cut_route_chunk([bad])
    assert instance.staged_token_count == 0


def test_new_fragment_cannot_change_model_dimensions() -> None:
    instance = state(prev=0, carry=1, frozen=1)
    pieces = [np.zeros((1, 2, 3), dtype=np.int16)]
    instance.frozen_cut_route_chunk(pieces)
    instance.seal_frozen_cut("cp")
    instance.frozen_cut = ("next", 2)
    pieces.append(np.zeros((1, 4, 3), dtype=np.int16))
    with pytest.raises(ValueError, match="dimensions changed"):
        instance.frozen_cut_route_chunk(pieces)
