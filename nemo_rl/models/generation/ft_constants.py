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
"""Centralized timing constants for fault-tolerant generation.

The NCCL heartbeat timeout is the critical parameter that all retry
logic derives from. When a gen shard dies mid-broadcast, the surviving
shards' NCCL watchdog takes up to NCCL_HEARTBEAT_TIMEOUT_SEC to detect
the dead peer. Until the watchdog fires, reset_collective is queued
behind the hung NCCL op and the gen shard can't participate in a new
init_collective.

All retry/backoff constants in ensure_collective_synced and
refit_policy_generation should be derived from this value so they
stay correct if someone changes TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC.
"""

import os

NCCL_HEARTBEAT_TIMEOUT_S: float = float(
    os.environ.get("TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC", "60")
)

# ensure_collective_synced: exponential backoff base and cap.
# The backoff must exceed NCCL_HEARTBEAT_TIMEOUT_S by attempt 4-5
# so that reset_collective can drain the hung ops. NOTE: superseded by the
# quiesce-wait (COLLECTIVE_SYNC_QUIESCE_* below) on the RefitWorker path —
# kept for the legacy/DTensor reference and back-compat.
COLLECTIVE_SYNC_BACKOFF_BASE_S: float = 5.0
COLLECTIVE_SYNC_BACKOFF_CAP_S: float = NCCL_HEARTBEAT_TIMEOUT_S + 30.0
# Each rendezvous retry now fires only at a SETTLED gen world (see the
# quiesce-wait), so it is far higher-quality than a blind-backoff retry —
# fewer attempts are needed. Env-overridable.
COLLECTIVE_SYNC_MAX_ATTEMPTS: int = int(
    os.environ.get("NRL_COLLECTIVE_SYNC_MAX_ATTEMPTS", "5")
)

# RL-412 §2b: quiesce-wait between rendezvous retries. Instead of a blind
# exponential backoff, the trainer polls /current_gen_world_size until the gen
# world has SETTLED before firing the next init_collective, so each retry lands
# on a stable, warm world (much higher success probability) and we stop the
# RefitWorker kill/respawn storm that destabilized Megatron rank-0.
#   - QUIESCE_S: the joinable set must hold steady at least this long.
#   - QUIESCE_POLL_S: poll cadence while waiting.
#   - QUIESCE_MAX_WAIT_S: hard bound; proceed anyway after this (the rendezvous
#     itself is still bounded by NRL_RENDEZVOUS_TIMEOUT_S).
COLLECTIVE_SYNC_QUIESCE_S: float = float(
    os.environ.get("NRL_COLLECTIVE_SYNC_QUIESCE_S", "10")
)
COLLECTIVE_SYNC_QUIESCE_POLL_S: float = float(
    os.environ.get("NRL_COLLECTIVE_SYNC_QUIESCE_POLL_S", "2")
)
COLLECTIVE_SYNC_QUIESCE_MAX_WAIT_S: float = float(
    os.environ.get(
        "NRL_COLLECTIVE_SYNC_QUIESCE_MAX_WAIT_S",
        str(NCCL_HEARTBEAT_TIMEOUT_S + 30.0),
    )
)

# RL-412 §3: minimum warm-up age before a freshly-added (unproven) joining
# shard counts as joinable. A backfilled shard answers /openapi.json within
# seconds, but its COLD cross-cluster RoCE route needs ~60-90s before its first
# TCPStore rendezvous can complete — counting it joinable earlier is the root
# cause of the "7/9 clients joined" rendezvous timeout. A *proven* shard (>=1
# prior successful init_collective → route already warm) bypasses this gate.
JOINABLE_MIN_AGE_S: float = float(os.environ.get("NRL_JOINABLE_MIN_AGE_S", "90"))

# Rendezvous timeout per attempt. Each init_collective attempt waits
# this long for all participants to join before giving up.
COLLECTIVE_SYNC_RENDEZVOUS_TIMEOUT_S: float = 150.0

# Elastic re-init (RL-412): how long the gen-side *joinable* set must hold
# steady before the trainer grows the cross-cluster comm into it. Coalesces
# replacements that come ready close together (e.g. 3 at +3min, 1 at +4min →
# one re-init at +3, another at +4) and prevents per-tick re-init churn.
# Shrink (lost member) is NOT debounced — the broken comm is rebuilt at once.
REJOIN_DEBOUNCE_S: float = float(os.environ.get("NRL_REJOIN_DEBOUNCE_S", "45"))

# async_grpo_train: outer retry around the entire refit.
# Wait at least one full heartbeat cycle + buffer for gen cluster
# recovery (add_shard pod provisioning, vLLM init).
REFIT_RETRY_MAX_ATTEMPTS: int = 3
REFIT_RETRY_SETTLE_S: float = NCCL_HEARTBEAT_TIMEOUT_S + 30.0
