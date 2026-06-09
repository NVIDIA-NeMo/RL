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
"""Unit tests for the train-side elastic comm re-init decision (RL-412).

Pure decision logic — no Ray, no cluster, no HTTP. Exercises shrink-eager /
grow-debounced / same-reuse plus the single-initiator idempotency property.
"""

from __future__ import annotations

from nemo_rl.models.generation.remote_generation import (
    _parse_gen_world,
    decide_collective_sync,
)

DEBOUNCE = 45.0


def _decide(alive, joinable, stable, last, *, train=1, alive_worker=True,
            comm_epoch=0, last_synced_epoch=0):
    return decide_collective_sync(
        alive_gen_ws=alive,
        joinable_gen_ws=joinable,
        stable_for_s=stable,
        effective_train_ws=train,
        last_synced_ws=last,
        refit_worker_alive=alive_worker,
        rejoin_debounce_s=DEBOUNCE,
        comm_epoch=comm_epoch,
        last_synced_epoch=last_synced_epoch,
    )


def test_same_world_reuses_live_comm():
    # train(1) + gen(8) == last(9), settled → reuse.
    assert _decide(8, 8, 999, 9) == ("reuse", 9)


def test_shrink_reinits_immediately_no_debounce():
    # A member was lost: alive 7 → target 8 < last 9. Re-init now even though
    # nothing is "stable" yet — the old comm is broken.
    assert _decide(7, 7, 0.0, 9) == ("reinit", 8)


def test_grow_waits_for_debounce_then_reinits():
    # Replacement became joinable: target 9 > last 8.
    # Not stable long enough → debounce (refit on existing comm).
    assert _decide(8, 8, 10.0, 8) == ("debounce", 9)
    # Stable past the window → grow.
    assert _decide(8, 8, DEBOUNCE, 8) == ("reinit", 9)


def test_cold_backfill_excluded_from_target_so_no_premature_grow():
    # alive 8 but only 7 joinable (one shard still cold/booting). We rendezvous
    # the JOINABLE cohort, so the target is train+7 == last 8 → reuse. The cold
    # shard is simply not in the cohort; it joins a later re-init once warm.
    assert _decide(8, 7, 999, 8) == ("reuse", 8)
    # Once it warms (joinable 8) and settles, the grow fires.
    assert _decide(8, 8, DEBOUNCE, 8) == ("reinit", 9)


def test_straggler_never_blocks_runs_at_smaller_world():
    # A straggler that never warms (alive 8, joinable stuck at 7) → target stays
    # train+7 == last → reuse forever: we keep training at the smaller cohort,
    # never blocked on the straggler (degraded, never wedged).
    for _ in range(5):
        assert _decide(8, 7, 999, 8) == ("reuse", 8)


def test_first_sync_and_dead_worker_force_reinit():
    # No prior comm → must init.
    assert _decide(8, 8, 0.0, None) == ("reinit", 9)
    # Producer (RefitWorker) died → must rebuild even if world unchanged.
    assert _decide(8, 8, 999, 9, alive_worker=False) == ("reinit", 9)


def test_idempotent_after_grow_no_double_reinit():
    # After a grow re-init to 9, caller sets last=9. Next refit with unchanged
    # joinable set must NOT re-init again (no double).
    assert _decide(8, 8, DEBOUNCE, 8) == ("reinit", 9)   # the grow
    assert _decide(8, 8, DEBOUNCE + 5, 9) == ("reuse", 9)  # next refit → reuse


def test_comm_epoch_change_forces_reinit_even_when_count_unchanged():
    # The crash case: an in-comm shard was evicted (epoch 4→5) and backfill
    # restored the SAME count, so target==last. Without the epoch the trainer
    # would "reuse" the wedged group and eat a failed broadcast. The epoch
    # change forces a proactive re-init instead.
    assert _decide(8, 8, 999, 9, comm_epoch=5, last_synced_epoch=4) == ("reinit", 9)
    # Same epoch → normal reuse.
    assert _decide(8, 8, 999, 9, comm_epoch=5, last_synced_epoch=5) == ("reuse", 9)


def test_parse_gen_world_backcompat_defaults():
    # New server: all five fields present.
    assert _parse_gen_world({
        "world_size": 8, "joinable_world_size": 6, "joinable_stable_for_s": 12.0,
        "comm_epoch": 3, "nccl_reinit_in_progress": True,
    }) == (8, 6, 12.0, 3, True)
    # Old server: only world_size → joinable=alive, stable=+inf, epoch=0,
    # reinit=False (so the trainer treats it as always-settled, no comm
    # invalidation, no in-flight lifecycle op).
    alive, joinable, stable, epoch, reinit = _parse_gen_world({"world_size": 8})
    assert (alive, joinable, stable, epoch, reinit) == (8, 8, float("inf"), 0, False)
    # Missing world_size → None (caller skips dynamic re-sync).
    assert _parse_gen_world({}) is None


def test_parse_gen_world_reinit_flag():
    _, _, _, _, reinit = _parse_gen_world(
        {"world_size": 8, "nccl_reinit_in_progress": True}
    )
    assert reinit is True
    _, _, _, _, reinit2 = _parse_gen_world({"world_size": 8})
    assert reinit2 is False
