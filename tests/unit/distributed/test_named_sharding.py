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
import numpy as np
import pytest

from nemo_rl.distributed.named_sharding import NamedSharding


@pytest.fixture
def sample_sharding():
    """Provides a standard NamedSharding instance for testing."""
    layout = [[[0, 1, 2, 3], [4, 5, 6, 7]]]  # dp=1, pp=2, tp=4
    names = ["dp", "pp", "tp"]
    return NamedSharding(layout, names)


@pytest.fixture
def sample_2d_sharding():
    """Provides a 2D NamedSharding instance for testing."""
    layout = [[0, 1], [2, 3]]
    names = ["dp", "tp"]
    return NamedSharding(layout, names)


def test_initialization_success(sample_sharding):
    assert sample_sharding.shape == {"dp": 1, "pp": 2, "tp": 4}
    assert sample_sharding.names == ["dp", "pp", "tp"]
    assert sample_sharding.ndim == 3
    assert sample_sharding.size == 8
    np.testing.assert_array_equal(
        sample_sharding.layout, np.array([[[0, 1, 2, 3], [4, 5, 6, 7]]])
    )


def test_initialization_dim_mismatch():
    layout = [[0, 1], [2, 3]]
    names = ["dp", "pp", "tp"]
    with pytest.raises(ValueError, match="Number of dimensions.*must match"):
        NamedSharding(layout, names)


def test_initialization_non_integer():
    layout = [[0, 1.5], [2, 3]]
    names = ["dp", "pp"]
    with pytest.raises(ValueError, match="Layout must contain only integer rank IDs"):
        NamedSharding(layout, names)


def test_initialization_duplicate_ranks():
    layout = [[0, 1], [2, 0]]
    names = ["dp", "pp"]
    with pytest.raises(ValueError, match="Duplicate ranks found"):
        NamedSharding(layout, names)


def test_get_ranks_full_slice(sample_sharding):
    # Get all ranks for dp=0
    ranks = sample_sharding.get_ranks(dp=0)
    correct_out = NamedSharding(np.array([[0, 1, 2, 3], [4, 5, 6, 7]]), ["pp", "tp"])
    assert ranks == correct_out


def test_get_ranks_partial_slice(sample_sharding):
    # Get ranks for dp=0, pp=1
    ranks = sample_sharding.get_ranks(dp=0, pp=1)
    correct_out = NamedSharding(np.array([4, 5, 6, 7]), ["tp"])
    assert ranks == correct_out


def test_get_ranks_partial_slice_2(sample_sharding):
    ranks = sample_sharding.get_ranks(dp=0, tp=2)
    correct_out = NamedSharding(np.array([2, 6]), ["pp"])
    assert ranks == correct_out


def test_get_ranks_single_rank(sample_sharding):
    # Get rank for dp=0, pp=0, tp=2
    ranks = sample_sharding.get_ranks(dp=0, pp=0, tp=2)
    correct_out = 2
    assert ranks == correct_out


def test_get_ranks_no_args(sample_sharding):
    # Get all ranks flattened
    ranks = sample_sharding.get_ranks()
    assert ranks == sample_sharding


def test_get_ranks_invalid_name(sample_sharding):
    with pytest.raises(ValueError, match="Invalid axis name: 'xx'"):
        sample_sharding.get_ranks(xx=0)


def test_get_ranks_index_out_of_bounds(sample_sharding):
    with pytest.raises(IndexError, match="Index 2 is out of bounds for axis 'pp'"):
        sample_sharding.get_ranks(pp=2)
    with pytest.raises(IndexError, match="Index 4 is out of bounds for axis 'tp'"):
        sample_sharding.get_ranks(tp=4)


def test_get_axis_index(sample_sharding):
    assert sample_sharding.get_axis_index("dp") == 0
    assert sample_sharding.get_axis_index("pp") == 1
    assert sample_sharding.get_axis_index("tp") == 2


def test_get_axis_index_invalid_name(sample_sharding):
    with pytest.raises(ValueError, match="Invalid axis name: 'xx'"):
        sample_sharding.get_axis_index("xx")


def test_get_axis_size(sample_sharding):
    assert sample_sharding.get_axis_size("dp") == 1
    assert sample_sharding.get_axis_size("pp") == 2
    assert sample_sharding.get_axis_size("tp") == 4


def test_equality():
    layout1 = [[[0, 1], [2, 3]]]
    names1 = ["a", "b", "c"]
    sharding1 = NamedSharding(layout1, names1)

    layout2 = [[[0, 1], [2, 3]]]
    names2 = ["a", "b", "c"]
    sharding2 = NamedSharding(layout2, names2)

    layout3 = [[[0, 1], [2, 4]]]  # Different layout
    names3 = ["a", "b", "c"]
    sharding3 = NamedSharding(layout3, names3)

    layout4 = [[[0, 1], [2, 3]]]
    names4 = ["x", "y", "z"]  # Different names
    sharding4 = NamedSharding(layout4, names4)

    assert sharding1 == sharding2
    assert sharding1 != sharding3
    assert sharding1 != sharding4
    assert sharding1 != "not a sharding object"


def test_repr(sample_sharding):
    representation = repr(sample_sharding)
    assert "NamedSharding" in representation
    assert "shape=(1, 2, 4)" in representation
    assert "names=['dp', 'pp', 'tp']" in representation
    assert "layout=" in representation
    assert "[[[0 1 2 3]" in representation  # Check layout content part
    assert "[4 5 6 7]]]" in representation


def test_get_worker_coords(sample_2d_sharding):
    sharding = sample_2d_sharding
    assert sharding.get_worker_coords(0) == {"dp": 0, "tp": 0}
    assert sharding.get_worker_coords(1) == {"dp": 0, "tp": 1}
    assert sharding.get_worker_coords(2) == {"dp": 1, "tp": 0}
    assert sharding.get_worker_coords(3) == {"dp": 1, "tp": 1}

    with pytest.raises(ValueError, match="Worker ID 4 not found in sharding layout."):
        sharding.get_worker_coords(4)

    layout_3d = [[[0, 1], [2, 3]], [[4, 5], [6, 7]]]
    names_3d = ["d", "p", "t"]
    sharding_3d = NamedSharding(layout_3d, names_3d)
    assert sharding_3d.get_worker_coords(5) == {"d": 1, "p": 0, "t": 1}


def test_get_ranks_by_coord(sample_2d_sharding):
    sharding = sample_2d_sharding
    assert sharding.get_ranks_by_coord(dp=0) == [0, 1]
    assert sharding.get_ranks_by_coord(tp=1) == [1, 3]
    assert sharding.get_ranks_by_coord(dp=0, tp=0) == [0]
    assert sharding.get_ranks_by_coord(dp=1, tp=1) == [3]

    # Test with an axis not present (should return all ranks for that axis)
    assert sharding.get_ranks_by_coord() == [0, 1, 2, 3]  # All ranks

    # Test with out-of-bounds coordinate
    assert sharding.get_ranks_by_coord(dp=2) == []
    assert sharding.get_ranks_by_coord(dp=0, tp=5) == []

    # Test with invalid axis name
    with pytest.raises(ValueError, match="Invalid axis name: 'invalid_axis'."):
        sharding.get_ranks_by_coord(invalid_axis=0)

    layout_3d = [[[0, 1], [2, 3]], [[4, 5], [6, 7]]]
    names_3d = ["d", "p", "t"]
    sharding_3d = NamedSharding(layout_3d, names_3d)
    assert sharding_3d.get_ranks_by_coord(d=0) == [0, 1, 2, 3]
    assert sharding_3d.get_ranks_by_coord(p=1) == [2, 3, 6, 7]
    assert sharding_3d.get_ranks_by_coord(t=0) == [0, 2, 4, 6]
    assert sharding_3d.get_ranks_by_coord(d=1, p=0) == [4, 5]
    assert sharding_3d.get_ranks_by_coord(d=0, t=1) == [1, 3]
    assert sharding_3d.get_ranks_by_coord(p=1, t=0) == [2, 6]
    assert sharding_3d.get_ranks_by_coord(d=1, p=1, t=1) == [7]


def test_complex_get_worker_coords(complex_sharding):
    sharding = complex_sharding
    assert sharding.get_worker_coords(10) == {"a": 1, "b": 0, "c": 1, "d": 0}
    assert sharding.get_worker_coords(15) == {"a": 1, "b": 1, "c": 1, "d": 1}


def test_complex_get_ranks_by_coord(complex_sharding):
    sharding = complex_sharding
    assert sharding.get_ranks_by_coord(a=0) == [0, 1, 2, 3, 4, 5, 6, 7]
    assert sharding.get_ranks_by_coord(b=1) == [4, 5, 6, 7, 12, 13, 14, 15]
    assert sharding.get_ranks_by_coord(c=0, d=1) == [1, 5, 9, 13]
    assert sharding.get_ranks_by_coord(a=1, b=0, c=1) == [10, 11]
    assert sharding.get_ranks_by_coord(a=1, b=1, c=1, d=0) == [14]


# More complex sharding for testing
@pytest.fixture
def complex_sharding():
    layout = [
        [[[0, 1], [2, 3]], [[4, 5], [6, 7]]],
        [[[8, 9], [10, 11]], [[12, 13], [14, 15]]],
    ]
    names = ["a", "b", "c", "d"]
    return NamedSharding(layout, names)
