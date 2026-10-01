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
"""The sample-id grammar shared by finalizer, controller, and replay buffer."""

from __future__ import annotations

import pytest

from nemo_rl.experience.sample_ids import (
    extra_sample_ids_for,
    group_ids_in_order,
    is_canonical_sample_id,
    make_sample_id,
    parse_sample_id,
    try_parse_sample_id,
    validate_group_sample_ids,
)


@pytest.mark.parametrize(
    ("group_id", "i", "j", "expected"),
    [
        ("grp", 0, 0, "grp_g0"),
        ("grp", 3, 0, "grp_g3"),
        ("grp", 3, 1, "grp_g3_t1"),
        ("grp", 12, 7, "grp_g12_t7"),
        # Group ids may themselves contain the delimiter.
        ("a_g1_b", 2, 0, "a_g1_b_g2"),
        ("a_g1_b", 2, 3, "a_g1_b_g2_t3"),
        ("uuid-1234-abcd", 0, 0, "uuid-1234-abcd_g0"),
    ],
)
def test_make_and_parse_round_trip(group_id, i, j, expected):
    sample_id = make_sample_id(group_id, i, j)
    assert sample_id == expected
    assert parse_sample_id(sample_id) == (group_id, i, j)


def test_make_sample_id_default_trace_is_canonical():
    assert make_sample_id("grp", 4) == "grp_g4"
    assert is_canonical_sample_id("grp_g4")
    assert not is_canonical_sample_id("grp_g4_t1")
    assert not is_canonical_sample_id("grp")


def test_parse_binds_the_last_generation_suffix():
    # Historical rsplit("_g", 1) semantics: the last _g<digits> wins.
    assert parse_sample_id("grp_g1_g2") == ("grp_g1", 2, 0)
    assert parse_sample_id("grp_g1_g2_t4") == ("grp_g1", 2, 4)


@pytest.mark.parametrize(
    "bad",
    [
        "",
        "grp",
        "grp_g",
        "grp_gx",
        "_g0",  # empty group id
        "grp_g0_t",
        "grp_g0_tx",
        "grp_g0_t1_extra",
        "grp_g0_aattempt",  # physical attempt id, not a canonical id
        "grp_g-1",
        "grp_t1",
    ],
)
def test_parse_rejects_malformed_ids(bad):
    assert try_parse_sample_id(bad) is None
    with pytest.raises(ValueError, match="malformed sample id"):
        parse_sample_id(bad)


def test_parse_rejects_non_string():
    with pytest.raises(TypeError):
        parse_sample_id(3)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("group_id", "i", "j"),
    [("", 0, 0), ("grp", -1, 0), ("grp", 0, -1), ("grp", 1.0, 0), ("grp", True, 0)],
)
def test_make_sample_id_rejects_bad_inputs(group_id, i, j):
    with pytest.raises(ValueError):
        make_sample_id(group_id, i, j)  # type: ignore[arg-type]


def test_group_ids_in_order_dedups_and_keeps_first_seen_order():
    ids = ["b_g0", "b_g1", "a_g0", "b_g0_t1", "a_g1_t2", "c_g0"]
    assert group_ids_in_order(ids) == ["b", "a", "c"]


def test_group_ids_in_order_treats_malformed_ids_as_their_own_group():
    assert group_ids_in_order(["odd-id", "odd-id", "x_g0"]) == ["odd-id", "x"]


def test_extra_sample_ids_for_enumerates_the_deterministic_cap_set():
    canonical = ["grp_g0", "grp_g1"]
    assert extra_sample_ids_for(canonical, 1) == []
    assert extra_sample_ids_for(canonical, 3) == [
        "grp_g0_t1",
        "grp_g0_t2",
        "grp_g1_t1",
        "grp_g1_t2",
    ]
    assert extra_sample_ids_for(canonical, 3) == [
        make_sample_id("grp", i, j) for i in range(2) for j in (1, 2)
    ]
    with pytest.raises(ValueError):
        extra_sample_ids_for(canonical, 0)


class TestValidateGroupSampleIds:
    def test_accepts_canonical_group(self):
        validate_group_sample_ids(["g_g0", "g_g1", "g_g2"], expected_group_size=3)

    def test_accepts_canonical_block_plus_extras(self):
        validate_group_sample_ids(
            ["g_g0", "g_g1", "g_g0_t1", "g_g1_t1", "g_g1_t2"],
            expected_group_size=2,
        )

    def test_rejects_missing_canonical_index(self):
        with pytest.raises(ValueError, match="do not cover"):
            validate_group_sample_ids(["g_g0", "g_g2"], expected_group_size=2)

    def test_rejects_too_many_rollouts(self):
        with pytest.raises(ValueError, match="do not cover"):
            validate_group_sample_ids(["g_g0", "g_g1", "g_g2"], expected_group_size=2)

    def test_rejects_extra_without_canonical(self):
        with pytest.raises(ValueError, match="do not cover|no canonical row"):
            validate_group_sample_ids(["g_g0", "g_g5_t1"], expected_group_size=1)

    def test_rejects_duplicate_pairs(self):
        with pytest.raises(ValueError, match="duplicate"):
            validate_group_sample_ids(["g_g0", "g_g0"], expected_group_size=1)

    def test_rejects_mixed_groups(self):
        with pytest.raises(ValueError, match="span 2 group ids"):
            validate_group_sample_ids(["g_g0", "h_g1"], expected_group_size=2)

    def test_rejects_malformed(self):
        with pytest.raises(ValueError, match="malformed"):
            validate_group_sample_ids(["g_g0", "junk"], expected_group_size=2)
