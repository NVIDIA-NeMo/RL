# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
"""Unit tests for the ``message_log`` wire-boundary decomposition.

Sits under ``tests/data_plane/`` rather than ``tests/unit/data/`` so the
heavy ``tests/unit/conftest.py`` (which eagerly imports Ray / the full
nemo_rl model stack) doesn't gate collection. The three helpers under
test are pure-Python and need only ``torch`` / ``numpy`` /
``BatchedDataDict`` at runtime.
"""

from typing import Any

import pytest
import torch

from nemo_rl.data.interfaces import LLMMessageLogType
from nemo_rl.data.llm_message_utils import (
    MESSAGE_LOG_BULK_FIELDS,
    attach_message_log_view,
    decompose_message_log,
    reconstruct_message_log,
)
from nemo_rl.distributed.batched_data_dict import BatchedDataDict

from ._rollout_shapes import make_multi_turn_message_log


def _build_message_log_batch() -> list[LLMMessageLogType]:
    return [
        [
            {"role": "user", "content": "Q1", "token_ids": torch.tensor([1, 2, 3])},
            {"role": "assistant", "content": "A1", "token_ids": torch.tensor([4, 5])},
        ],
        [
            {"role": "user", "content": "Q2", "token_ids": torch.tensor([6, 7])},
            {
                "role": "assistant",
                "content": "A2",
                "token_ids": torch.tensor([8, 9, 10, 11]),
            },
        ],
    ]


def test_decompose_message_log_basic_shapes() -> None:
    out = decompose_message_log(_build_message_log_batch())
    assert out["turn_lengths"].tolist() == [[3, 2], [2, 4]]
    assert list(out["turn_roles"][0]) == ["user", "assistant"]
    assert list(out["turn_contents"][1]) == ["Q2", "A2"]
    # First assistant turn's length per sample.
    assert out["response_token_lengths"].tolist() == [2, 4]


def test_decompose_message_log_no_assistant_turn() -> None:
    out = decompose_message_log(
        [[{"role": "user", "content": "U", "token_ids": torch.tensor([1, 2])}]]
    )
    assert out["turn_lengths"].tolist() == [[2]]
    assert out["response_token_lengths"].tolist() == [0]


def test_decompose_message_log_picks_first_assistant() -> None:
    """If multiple assistant turns exist, ``response_token_lengths`` takes the first."""
    out = decompose_message_log(
        [
            [
                {"role": "user", "content": "U", "token_ids": torch.tensor([1])},
                {
                    "role": "assistant",
                    "content": "A1",
                    "token_ids": torch.tensor([2, 3]),
                },
                {"role": "user", "content": "U2", "token_ids": torch.tensor([4])},
                {
                    "role": "assistant",
                    "content": "A2",
                    "token_ids": torch.tensor([5, 6, 7, 8]),
                },
            ]
        ]
    )
    assert out["response_token_lengths"].tolist() == [2]


def test_decompose_message_log_jagged_turn_count() -> None:
    """Samples with different turn counts pad ``turn_lengths`` with zeros."""
    out = decompose_message_log(
        [
            [
                {"role": "user", "content": "U", "token_ids": torch.tensor([1, 2])},
                {"role": "assistant", "content": "A", "token_ids": torch.tensor([3])},
                {"role": "tool", "content": "T", "token_ids": torch.tensor([4, 5, 6])},
            ],
            [
                {"role": "user", "content": "U", "token_ids": torch.tensor([7])},
            ],
        ]
    )
    assert out["turn_lengths"].tolist() == [[2, 1, 3], [1, 0, 0]]


def test_decompose_message_log_missing_role_raises() -> None:
    """Missing ``role`` surfaces loudly as KeyError rather than producing ``""`` silently."""
    with pytest.raises(KeyError):
        decompose_message_log(
            [[{"content": "no role here", "token_ids": torch.tensor([1])}]]
        )


def test_reconstruct_message_log_roundtrip() -> None:
    """decompose → flatten → reconstruct returns equivalent message_log."""
    ml_batch = _build_message_log_batch()
    decomposed = decompose_message_log(ml_batch)

    flat_per_sample = [torch.cat([m["token_ids"] for m in ml]) for ml in ml_batch]
    max_total = max(t.shape[0] for t in flat_per_sample)
    input_ids = torch.zeros((len(ml_batch), max_total), dtype=torch.long)
    for i, t in enumerate(flat_per_sample):
        input_ids[i, : t.shape[0]] = t

    rebuilt = reconstruct_message_log(
        input_ids=input_ids,
        turn_lengths=decomposed["turn_lengths"],
        turn_roles=decomposed["turn_roles"],
        turn_contents=decomposed["turn_contents"],
    )

    assert len(rebuilt) == len(ml_batch)
    for orig_sample, new_sample in zip(ml_batch, rebuilt):
        assert len(orig_sample) == len(new_sample)
        for orig_turn, new_turn in zip(orig_sample, new_sample):
            assert orig_turn["role"] == new_turn["role"]
            assert orig_turn["content"] == new_turn["content"]
            assert torch.equal(orig_turn["token_ids"], new_turn["token_ids"])


def test_reconstruct_message_log_returns_views() -> None:
    """Per-turn ``token_ids`` must be views into the local ``input_ids`` storage."""
    ml_batch = _build_message_log_batch()
    decomposed = decompose_message_log(ml_batch)
    input_ids = torch.zeros((2, 6), dtype=torch.long)
    input_ids[0, :5] = torch.tensor([1, 2, 3, 4, 5])
    input_ids[1, :6] = torch.tensor([6, 7, 8, 9, 10, 11])

    rebuilt = reconstruct_message_log(
        input_ids=input_ids,
        turn_lengths=decomposed["turn_lengths"],
        turn_roles=decomposed["turn_roles"],
        turn_contents=decomposed["turn_contents"],
    )

    parent_ptr = input_ids.untyped_storage().data_ptr()
    for sample in rebuilt:
        for turn in sample:
            if "token_ids" in turn:
                assert turn["token_ids"].untyped_storage().data_ptr() == parent_ptr


def test_reconstruct_message_log_attaches_generation_logprobs() -> None:
    """``generation_logprobs`` is attached only to assistant turns when provided."""
    ml_batch = _build_message_log_batch()
    decomposed = decompose_message_log(ml_batch)
    input_ids = torch.zeros((2, 6), dtype=torch.long)
    input_ids[0, :5] = torch.tensor([1, 2, 3, 4, 5])
    input_ids[1, :6] = torch.tensor([6, 7, 8, 9, 10, 11])
    gen_logprobs = torch.zeros_like(input_ids, dtype=torch.float32)

    rebuilt = reconstruct_message_log(
        input_ids=input_ids,
        turn_lengths=decomposed["turn_lengths"],
        turn_roles=decomposed["turn_roles"],
        turn_contents=decomposed["turn_contents"],
        generation_logprobs=gen_logprobs,
    )

    for sample in rebuilt:
        for turn in sample:
            if turn["role"] == "assistant":
                assert "generation_logprobs" in turn
                assert turn["generation_logprobs"].shape == turn["token_ids"].shape
            else:
                assert "generation_logprobs" not in turn


def test_attach_message_log_view_populates_batch() -> None:
    ml_batch = _build_message_log_batch()
    decomposed = decompose_message_log(ml_batch)
    input_ids = torch.zeros((2, 6), dtype=torch.long)
    input_ids[0, :5] = torch.tensor([1, 2, 3, 4, 5])
    input_ids[1, :6] = torch.tensor([6, 7, 8, 9, 10, 11])
    batch: BatchedDataDict[Any] = BatchedDataDict(
        {"input_ids": input_ids, **{k: decomposed[k] for k in MESSAGE_LOG_BULK_FIELDS}}
    )
    assert "message_log" not in batch
    attach_message_log_view(batch)
    assert "message_log" in batch
    assert len(batch["message_log"]) == 2
    assert batch["message_log"][0][1]["role"] == "assistant"


def test_attach_message_log_view_noop_when_fields_absent() -> None:
    """Without decomposed fields, ``attach_message_log_view`` must leave the batch unchanged."""
    batch: BatchedDataDict[Any] = BatchedDataDict({"input_ids": torch.zeros((2, 4))})
    attach_message_log_view(batch)
    assert "message_log" not in batch


def test_attach_message_log_view_idempotent() -> None:
    """Calling twice produces the same shape (no exceptions, no doubled state)."""
    ml_batch = _build_message_log_batch()
    decomposed = decompose_message_log(ml_batch)
    input_ids = torch.zeros((2, 6), dtype=torch.long)
    batch: BatchedDataDict[Any] = BatchedDataDict(
        {"input_ids": input_ids, **{k: decomposed[k] for k in MESSAGE_LOG_BULK_FIELDS}}
    )
    attach_message_log_view(batch)
    first_len = len(batch["message_log"])
    attach_message_log_view(batch)
    assert len(batch["message_log"]) == first_len


# ── Realistic multi-turn coverage using ``_rollout_shapes.make_multi_turn_message_log`` ──
# Exercises decompose/reconstruct on the same shape of message_log a real
# multi-turn rollout produces — jagged turn counts (1-4), alternating
# user/assistant roles, variable per-turn token lengths.


def test_decompose_realistic_multi_turn_jagged_count() -> None:
    """Jagged turn-count message logs (1, 4, 2 turns) round-trip via decompose.

    The realistic shape is what multi-turn rollouts produce — varied
    per-sample turn counts. ``decompose_message_log`` must pad shorter
    samples' ``turn_lengths`` with zeros without losing role / content
    alignment.
    """

    # Force three samples with distinctly different turn counts.
    ml_batch = make_multi_turn_message_log(n=3, turns_per_sample=[1, 4, 2], seed=23)
    decomposed = decompose_message_log(ml_batch)

    n = len(ml_batch)
    max_turns = max(len(s) for s in ml_batch)

    # Shapes
    assert decomposed["turn_lengths"].shape == (n, max_turns)
    assert len(decomposed["turn_roles"]) == n
    assert len(decomposed["turn_contents"]) == n
    # Shorter samples' tail turns padded with zero
    assert int(decomposed["turn_lengths"][0, 1]) == 0  # 1-turn sample, slot 1 empty
    assert int(decomposed["turn_lengths"][2, 2]) == 0  # 2-turn sample, slot 2 empty
    # Non-padding positions match the source token counts
    for i, sample in enumerate(ml_batch):
        for t, turn in enumerate(sample):
            assert int(decomposed["turn_lengths"][i, t]) == int(
                turn["token_ids"].shape[0]
            )
            assert decomposed["turn_roles"][i][t] == turn["role"]


def test_decompose_reconstruct_roundtrip_realistic_multi_turn() -> None:
    """Full decompose → reconstruct round-trip on a realistic jagged multi-turn log.

    Existing roundtrip test uses a fixed 2-turn (user/assistant) shape via
    ``_build_message_log_batch``. This one exercises the full pipeline on
    variable turn counts (1, 3, 4 turns) with alternating roles — the
    realistic chat shape the wire actually carries.
    """

    ml_batch = make_multi_turn_message_log(n=3, turns_per_sample=[1, 3, 4], seed=17)
    decomposed = decompose_message_log(ml_batch)

    # Build the flat input_ids that the consumer would see on the wire.
    flat_per_sample = [torch.cat([m["token_ids"] for m in ml]) for ml in ml_batch]
    max_total = max(t.shape[0] for t in flat_per_sample)
    input_ids = torch.zeros((len(ml_batch), max_total), dtype=torch.long)
    for i, t in enumerate(flat_per_sample):
        input_ids[i, : t.shape[0]] = t

    rebuilt = reconstruct_message_log(
        input_ids=input_ids,
        turn_lengths=decomposed["turn_lengths"],
        turn_roles=decomposed["turn_roles"],
        turn_contents=decomposed["turn_contents"],
    )

    # Sample-level + turn-level identity through the pipeline.
    assert len(rebuilt) == len(ml_batch)
    for i, (orig_sample, new_sample) in enumerate(zip(ml_batch, rebuilt)):
        assert len(orig_sample) == len(new_sample), (
            f"sample {i}: turn count diverged "
            f"orig={len(orig_sample)} != rebuilt={len(new_sample)}"
        )
        for t, (orig_turn, new_turn) in enumerate(zip(orig_sample, new_sample)):
            assert orig_turn["role"] == new_turn["role"], (
                f"sample {i} turn {t}: role diverged"
            )
            assert orig_turn["content"] == new_turn["content"]
            assert torch.equal(orig_turn["token_ids"], new_turn["token_ids"])
