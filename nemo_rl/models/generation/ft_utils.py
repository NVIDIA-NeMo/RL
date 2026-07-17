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
"""Fault-tolerance utilities shared across generation backends.

These were previously in remote_generation.py; moved here so they can be
used by VllmGeneration.ensure_collective_synced without importing the
HTTP-only RemoteGeneration class.
"""

from __future__ import annotations

from typing import Optional


def _should_respawn_refit_worker(exc: BaseException) -> bool:
    """Decide whether a failed rendezvous requires KILLING + respawning the
    RefitWorker (its context is dead or NCCL-poisoned) vs REUSING it.

    Reuse is safe ONLY for a clean *pre-NCCL* rendezvous failure: a TCPStore
    timeout (``DistStoreError`` / ``DistNetworkError``) means a gen peer never
    checked in, so no NCCL communicator was ever created and the RefitWorker's
    CUDA/NCCL context is untouched → reuse (avoids the kill/respawn storm that
    destabilized Megatron rank-0).

    Everything else — an ``NCCLError`` (the NCCL bootstrap partially ran and
    left the comm/context poisoned, so reusing it fails forever with NCCLError),
    a dead/unavailable actor (``RayActorError``), or any unknown error — means
    we must respawn to get a clean context. Conservative by design: when in
    doubt, respawn (a respawn always clears poison; the gen side evicts the
    broken peer in parallel, so respawns stay rare — only on genuine poison,
    not on every timeout)."""
    # A dead/unavailable actor → respawn. Check first (also avoids calling
    # ``str()`` on a malformed RayActorError in the fallback below).
    try:
        from ray.exceptions import RayActorError

        if isinstance(exc, RayActorError):
            return True
    except Exception:  # noqa: BLE001
        pass
    try:
        from torch.distributed import DistNetworkError, DistStoreError

        # Ray surfaces an actor-method exception as ``RayTaskError(<Original>)``
        # whose class dual-inherits the original type, so isinstance sees through
        # the wrap for a clean TCPStore timeout.
        if isinstance(exc, (DistStoreError, DistNetworkError)):
            return False
    except Exception:  # noqa: BLE001 — torch symbol moved/renamed: fall through
        pass
    # Robustness fallback: the original class name is carried in the type name /
    # message even when isinstance-through-wrap doesn't hold on some versions.
    try:
        blob = f"{type(exc).__name__}: {exc}"
    except Exception:  # noqa: BLE001 — can't introspect → respawn (conservative)
        return True
    if "DistStoreError" in blob or "DistNetworkError" in blob:
        return False
    return True


def decide_collective_sync(
    *,
    alive_gen_ws: int,
    joinable_gen_ws: int,
    stable_for_s: float,
    effective_train_ws: int,
    last_synced_ws: Optional[int],
    refit_worker_alive: bool,
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
    ``rejoin_debounce_s``. First sync (last is None) or a dead producer forces
    a re-init. This function is the single decision point, which (with the
    caller updating ``_last_synced_world_size`` only on a successful re-init)
    gives the single-initiator / no-double-reinit guarantee.

    ``comm_epoch`` (from the router) bumps whenever an in-comm shard was
    removed/evicted, so the last-synced group is wedged. If it changed since
    our last sync we re-init PROACTIVELY — even when the shard count is
    unchanged (backfill restored it) — instead of reusing a dead group and
    eating a failed broadcast first."""
    # Rendezvous the frozen JOINABLE cohort (not the raw alive set): cold
    # backfills are excluded until warm, so they never sabotage the handshake.
    target_ws = effective_train_ws + joinable_gen_ws
    if last_synced_ws is None or not refit_worker_alive:
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
