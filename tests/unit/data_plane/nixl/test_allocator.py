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

from nemo_rl.data_plane.nixl.allocator import SlabAllocator, align_up
from nemo_rl.data_plane.nixl.errors import UnitFull


def test_align_up():
    assert align_up(1, 64) == 64
    assert align_up(64, 64) == 64
    assert align_up(65, 64) == 128


def test_alloc_free_coalesce():
    a = SlabAllocator(1024, align=64)
    x = a.alloc(100)  # 128
    y = a.alloc(100)  # 128
    z = a.alloc(100)  # 128
    assert (x, y, z) == (0, 128, 256)
    assert a.used_bytes == 384 and a.free_bytes == 640
    a.free(y, 100)
    assert a.fragments == 2
    a.free(x, 100)
    assert a.fragments == 2  # [0..256) and [384..1024)
    a.free(z, 100)
    assert a.fragments == 1 and a.free_bytes == 1024 and a.used_bytes == 0


def test_first_fit_reuses_hole():
    a = SlabAllocator(512)
    x = a.alloc(64)
    _ = a.alloc(64)
    a.free(x, 64)
    assert a.alloc(64) == 0


def test_unit_full():
    a = SlabAllocator(256)
    a.alloc(200)
    with pytest.raises(UnitFull):
        a.alloc(64)


def test_quarantine_reclaim():
    a = SlabAllocator(256)
    x = a.alloc(256)
    with pytest.raises(UnitFull):
        a.alloc(1)
    a.quarantine(x, 256, release_at=10.0)
    assert a.quarantined_bytes == 256
    assert a.reclaim(now=5.0) == 0
    with pytest.raises(UnitFull):
        a.alloc(1)
    assert a.reclaim(now=10.0) == 1
    assert a.alloc(256) == 0


def test_bad_args():
    with pytest.raises(ValueError):
        SlabAllocator(0)
    with pytest.raises(ValueError):
        SlabAllocator(64, align=48)
    a = SlabAllocator(64)
    with pytest.raises(ValueError):
        a.alloc(0)
    with pytest.raises(ValueError):
        a.free(0, 128)
