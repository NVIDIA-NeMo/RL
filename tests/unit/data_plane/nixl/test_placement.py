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
import pytest

from nemo_rl.data_plane.nixl.placement import LocalFirst, Spread, make_placement

UNITS = [
    {"unit_id": 0, "node_id": "A", "state": "ACTIVE"},
    {"unit_id": 1, "node_id": "A", "state": "ACTIVE"},
    {"unit_id": 2, "node_id": "B", "state": "ACTIVE"},
    {"unit_id": 3, "node_id": "B", "state": "DRAINING"},
]


def ids(order):
    return [u["unit_id"] for u in order]


def test_spread_rotates_over_all_active_units():
    p = Spread(start=0)
    assert ids(p.order(UNITS, node_id="A", nbytes=1)) == [0, 1, 2]
    assert ids(p.order(UNITS, node_id="A", nbytes=1)) == [1, 2, 0]
    assert ids(p.order(UNITS, node_id="A", nbytes=1)) == [2, 0, 1]


def test_local_first_prefers_own_node_then_rotates():
    p = LocalFirst(start=0)
    assert ids(p.order(UNITS, node_id="A", nbytes=1)) == [0, 1, 2]
    assert ids(p.order(UNITS, node_id="A", nbytes=1)) == [1, 0, 2]
    assert ids(p.order(UNITS, node_id="B", nbytes=1)) == [2, 0, 1]  # unit 3 is DRAINING


def test_local_first_without_local_units_uses_all():
    p = LocalFirst()
    assert ids(p.order(UNITS, node_id="Z", nbytes=1)) == [0, 1, 2]


def test_make_placement():
    assert isinstance(make_placement("spread"), Spread)
    assert isinstance(make_placement("local_first"), LocalFirst)
    with pytest.raises(ValueError):
        make_placement("hash")
