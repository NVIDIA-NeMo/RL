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
"""Freeze the initialised Python heap of a worker process.

CPython's collector walks every tracked container object on a full (generation 2)
collection. A trainer worker holds millions of them once the model, optimizer and
parallel state are built: the explicit ``gc.collect()`` before a refit takes about
2 s per rank there, and the automatic full collections fire in the middle of a
training step, per rank, at unsynchronised times, so every collective waits for
the rank that happens to be collecting. ``gc.freeze()`` moves the heap that
exists at that point into the permanent generation, which later collections do
not scan; they only walk what each step allocates afterwards.
"""

import gc
import time


def freeze_initialised_heap() -> dict[str, float]:
    """Collect once, then freeze everything that survived.

    Returns ``{"frozen", "collected", "seconds"}``: the size of the permanent
    generation after the freeze, the unreachable objects the preceding collection
    found, and the wall time of both.
    """
    t0 = time.perf_counter()
    collected = gc.collect()
    gc.freeze()
    return {
        "frozen": gc.get_freeze_count(),
        "collected": collected,
        "seconds": time.perf_counter() - t0,
    }


def gc_collection_counts() -> tuple[int, int, int]:
    """Cumulative number of collections run per generation (explicit ones included)."""
    stats = gc.get_stats()
    return (stats[0]["collections"], stats[1]["collections"], stats[2]["collections"])
