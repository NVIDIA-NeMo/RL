# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass

import pytest

from nemo_rl.experience.rollout_reassembler import group_selected_chains


@dataclass(frozen=True)
class Rec:
    model_call_id: str
    parent_call_id: str | None = None


def test_single_chain_is_unchanged():
    records = [Rec("a"), Rec("b", "a"), Rec("c", "b")]
    assert group_selected_chains(records) == [[0, 1, 2]]


def test_compaction_segments_root_new_chains_in_order():
    records = [Rec("a"), Rec("b", "a"), Rec("s"), Rec("t", "s")]
    assert group_selected_chains(records) == [[0, 1], [2, 3]]


def test_parent_resumes_after_interleaved_children():
    # parent p0 -> p1(spawns) ; children A and B alternate ; parent p2 continues from p1
    records = [
        Rec("p0"), Rec("p1", "p0"),
        Rec("a0"), Rec("b0"), Rec("a1", "a0"), Rec("b1", "b0"),
        Rec("p2", "p1"),
    ]
    assert group_selected_chains(records) == [[0, 1, 6], [2, 4], [3, 5]]


def test_continuation_without_a_selected_parent_is_rejected():
    with pytest.raises(ValueError, match="follow its selected predecessor"):
        group_selected_chains([Rec("a"), Rec("c", "b")])
    # a parent may have only one selected continuation
    with pytest.raises(ValueError, match="follow its selected predecessor"):
        group_selected_chains([Rec("a"), Rec("b", "a"), Rec("c", "a")])
