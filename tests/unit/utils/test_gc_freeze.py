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
import gc

from nemo_rl.utils.gc_freeze import freeze_initialised_heap, gc_collection_counts


def test_freeze_moves_live_objects_out_of_the_collector():
    keep = [[i] for i in range(1000)]  # tracked containers that stay alive
    before = gc.get_freeze_count()
    try:
        stats = freeze_initialised_heap()
        assert stats["frozen"] >= before + len(keep)
        assert stats["seconds"] >= 0.0
        # Frozen objects leave the permanent generation only when they die (other
        # threads may drop a few), so the count can shrink but never grow.
        assert len(keep) <= gc.get_freeze_count() <= stats["frozen"]
        # Objects created after the freeze are still collected; frozen ones stay put.
        cycle: list = []
        cycle.append(cycle)
        del cycle
        assert gc.collect() >= 1
        assert len(keep) <= gc.get_freeze_count() <= stats["frozen"]
    finally:
        gc.unfreeze()
    assert gc.get_freeze_count() == 0
    del keep


def test_gc_collection_counts_sees_an_explicit_full_collection():
    _, _, gen2_before = gc_collection_counts()
    gc.collect()
    assert gc_collection_counts()[2] == gen2_before + 1
