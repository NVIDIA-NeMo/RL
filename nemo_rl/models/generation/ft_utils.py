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
"""Fault-tolerance utilities shared across generation backends."""

from __future__ import annotations

from typing import Optional


def decide_collective_sync(
    *,
    alive_gen_ws: int,
    joinable_gen_ws: int,
    stable_for_s: float,
    effective_train_ws: int,
    last_synced_ws: Optional[int],
    rejoin_debounce_s: float,
    comm_epoch: int = 0,
    last_synced_epoch: int = 0,
) -> tuple[str, int]:
    """Decide how this refit should treat the cross-cluster comm.

    Returns ``(action, target_ws)`` where action is:
      - ``"reuse"``    : world unchanged + comm alive → broadcast on the live
                          comm (steady state).
      - ``"debounce"`` : the world GREW but the joinable set has not settled →
                          broadcast on the existing comm this refit; the new
                          shard stays ``joining`` and is re-checked next refit.
      - ``"reinit"``   : rendezvous a fresh comm at ``target_ws`` now.

    ``target_ws`` is ``effective_train_ws + joinable_gen_ws`` — the FROZEN
    cohort the router rendezvouses (the gen side dispatches ``init_collective``
    to exactly the joinable workers). Cold/booting backfills are excluded from
    the cohort entirely, so a too-early rendezvous can't include a peer that
    can't yet complete the handshake (the "7/9 clients joined" failure).

    Policy: shrink (target < last) re-inits immediately (the old comm is
    broken — don't wait ~3 min for replacements). Grow (target > last) re-inits
    only once the joinable set == the dispatch set AND has been stable for
    ``rejoin_debounce_s``. First sync (last is None) forces a re-init. This
    function is the single decision point, which (with the caller updating
    ``_last_synced_world_size`` only on a successful re-init) gives the
    single-initiator / no-double-reinit guarantee.

    ``comm_epoch`` (from the router) bumps whenever an in-comm shard was
    removed/evicted, so the last-synced group is wedged. If it changed since
    our last sync we re-init PROACTIVELY — even when the shard count is
    unchanged (backfill restored it) — instead of reusing a dead group and
    eating a failed broadcast first."""
    # Rendezvous the frozen JOINABLE cohort (not the raw alive set): cold
    # backfills are excluded until warm, so they never sabotage the handshake.
    target_ws = effective_train_ws + joinable_gen_ws
    if last_synced_ws is None:
        return "reinit", target_ws
    if comm_epoch != last_synced_epoch:
        # A comm member was removed/evicted since our last sync → the live
        # group is missing a peer (wedged). Rebuild now, don't reuse.
        return "reinit", target_ws
    if target_ws == last_synced_ws:
        return "reuse", target_ws
    if target_ws < last_synced_ws:
        return "reinit", target_ws  # SHRINK — eager
    # GROW: a replacement became joinable. Debounce so a batch that warms up
    # close together coalesces into one re-init; a replica that warms after
    # this boundary simply joins the next one.
    settled = stable_for_s >= rejoin_debounce_s
    return ("reinit" if settled else "debounce"), target_ws
