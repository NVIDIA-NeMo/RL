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

"""Golden test for cross-DP admission ordering.

The scheduler is a pure state machine, so a fixed workload must always produce
the same assignment trace, the same ~75-field assignment records, and the same
snapshot.  The digests below were captured from the pre-refactor implementation
and are what "functional parity" means here: change them only when a scheduling
decision is *meant* to change, and say so in the commit message.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import random

import pytest

from nemo_rl.models.generation.vllm.lfs.scheduler import CrossDpSchedulerState

def make_workload(*, groups, per_group, seed, explicit_probe=True):
    rng = random.Random(seed)
    catalog, true_len = [], {}
    for g in range(groups):
        base = rng.choice([300, 900, 2500, 6000, 12000, 16000])
        for k in range(per_group):
            rid = f"r{g}_{k}"
            L = max(64, int(base * rng.uniform(0.55, 1.45)))
            true_len[rid] = L
            item = {
                "request_id": rid,
                "group_id": f"g{g}",
                "fallback_cost": 1024,
                "oracle_cost": L,
                "predicted_cost": max(1, base),
            }
            if explicit_probe:
                item["is_designated_probe"] = (k == 0)
            catalog.append(item)
    rng.shuffle(catalog)
    return catalog, true_len


def _simulate(scheduler_cls, *, mode, dp_size, cap, catalog, true_len, **kw):
    S = scheduler_cls(dp_size, cap, mode, **kw)
    S.open_session("s0", catalog, ["p0"])
    trace, events, running, now = [], [], [], 0.0
    ids = [item["request_id"] for item in catalog]

    def claim_ready():
        # the full ~50-field assignment record is the real downstream contract
        events.extend(S.drain_new_assignment_events())
        for rid in ids:
            a = S.claim_if_assigned(rid)
            if a is None:
                continue
            trace.append((a["assignment_sequence"], rid, a["dp_idx"],
                          a["predicted_length"], a["dp_assignment_ordinal"],
                          a["session_dp_assignment_ordinal"]))
            heapq.heappush(
                running, (now + true_len[rid], a["assignment_sequence"], rid)
            )

    claim_ready()
    while running:
        finish, _, rid = heapq.heappop(running)
        now = finish
        S.complete(rid, true_len[rid])
        claim_ready()
    events.extend(S.drain_new_assignment_events())
    return trace, events, S.snapshot()


MODES = ["fcfs", "lfs", "oracle_probe_lfs", "predicted_lfs", "exact_length_lpt"]
CASES = {
    1: dict(groups=4, per_group=8, dp_size=1, cap=32),
    2: dict(groups=8, per_group=8, dp_size=2, cap=16),
    3: dict(groups=12, per_group=4, dp_size=2, cap=8),
    4: dict(groups=6, per_group=16, dp_size=4, cap=8),
    5: dict(groups=16, per_group=2, dp_size=1, cap=4),
}
GOLDEN = {
    "1:exact_length_lpt": "f66d1ee0fce4a1ec0335384be0c075f147507a41ac9fc62cc5e230b724d6f4d7",
    "1:fcfs": "c0c285b5f213a7c39dd89546fc6423bbdc117589f8ceb62a74ef0b59ae84bc9c",
    "1:lfs": "8a4b3178e41f418d34fc936bad62ee3b79e61711324704bfc3ef62413387b0bb",
    "1:oracle_probe_lfs": "ba9e1438b609bb435c32ea89636cd7260324404a6ce1fddffc0755830fad5be8",
    "1:predicted_lfs": "87b4193778b45e8c462367df3baaf1a03c60c07044b5b70f48bf958d3baa910e",
    "2:exact_length_lpt": "a96649839ec305509ee5a661e8335b0ee0ed41e6561fff4ec7a2d5796cf07a2a",
    "2:fcfs": "cdb91466f881b72af460bbbb43cf49777903a8ba5f5fb84a8b5fad9d87891fac",
    "2:lfs": "b6babda7432169efe3f5a29338d1f4c6ba39a7c2f5eea0186e975ad413e6414b",
    "2:oracle_probe_lfs": "62fd25e88f2aa84e40e2eb5ec9e3fae97a2d589ca3453567a367708fddd7c3bf",
    "2:predicted_lfs": "67736105214e5b98b103c80b2c7089cbb8d01fd904fae22a8a747a29a7252b36",
    "3:exact_length_lpt": "f30e278dd633e96dae412fd35367c65fec52984a6920889614b358e6341b2193",
    "3:fcfs": "83cffb57a58f06ca065dde6ebb7456cd1de2875e0b27b06295a2864ee5a75dab",
    "3:lfs": "fb95a0d98b7c5acdadc91c6ad343feb1733a3710ea891bff6637c80694b3d3f8",
    "3:oracle_probe_lfs": "814f90b500ba46f197794efc2b42de9c2576e07f6af41849f6b7d3420f3f5e4d",
    "3:predicted_lfs": "0f4c894e1b63dbae25f0a843760a1400dc03bbe9cb3429c7c57b509de213f654",
    "4:exact_length_lpt": "fd82a4be041b6646f418ad18408a1d65c71ca19804b3517e38af287867566789",
    "4:fcfs": "7d21bbc401c0f4c472ed424c9cc101392c7b4aaca27138ea23a0c92f10263e7a",
    "4:lfs": "1ecb9835ea1131febe4dfc3bb3c5df9170f863df01a94f53b525c4699d45a4ca",
    "4:oracle_probe_lfs": "537e4081efd2e96e0658c0b3d18181fdc04cac9702c1b4ef34002e967d371c8c",
    "4:predicted_lfs": "cf95b835ef6bfaa8a91f2e9d2f263534d04fa318cc17d494b0011af37a7b26d3",
    "5:exact_length_lpt": "028ce613bfc9b5951981d844912f1c701ebef413aec70002b419d856bc43b01d",
    "5:fcfs": "e7545981e06d5408ae9d248940c4a0c316a4c7632d0b52c095420c203b35b1bf",
    "5:lfs": "aa84170e7c5dbfb8022e6260df224ce8cc58738a6237af288db7a92ace47f1d0",
    "5:oracle_probe_lfs": "225f2ce6e711e169d8f8efe9d367e855c7a9a509fb3efd4d233f784d2ab8caf0",
    "5:predicted_lfs": "2394691f0f1739ddf4a240228d2a10c2339c4cf4d4480abec295537954a7b347"
}


@pytest.mark.parametrize("seed", sorted(CASES))
@pytest.mark.parametrize("mode", MODES)
def test_admission_matches_golden(seed: int, mode: str) -> None:
    case = CASES[seed]
    catalog, true_len = make_workload(
        groups=case["groups"], per_group=case["per_group"], seed=seed
    )
    if mode == "exact_length_lpt":
        catalog = [
            dict(item, fallback_cost=true_len[item["request_id"]])
            for item in catalog
        ]
    trace, events, snapshot = _simulate(
        CrossDpSchedulerState,
        mode=mode,
        dp_size=case["dp_size"],
        cap=case["cap"],
        catalog=catalog,
        true_len=true_len,
    )
    payload = {
        "assignments": len(trace),
        "events": events,
        "snapshot": snapshot,
        "trace": trace,
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode()
    ).hexdigest()
    assert digest == GOLDEN[f"{seed}:{mode}"], (
        f"admission behaviour changed for seed={seed} mode={mode}"
    )
