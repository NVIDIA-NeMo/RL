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
# Kept for legacy/DTensor back-compat; superseded by the quiesce-wait below on the RefitWorker path.
COLLECTIVE_SYNC_BACKOFF_BASE_S: float = 5.0
COLLECTIVE_SYNC_BACKOFF_CAP_S: float = NCCL_HEARTBEAT_TIMEOUT_S + 30.0
# Each retry fires at a settled gen world (quiesce-wait), so fewer attempts are needed. Env-overridable.
COLLECTIVE_SYNC_MAX_ATTEMPTS: int = int(
    os.environ.get("NRL_COLLECTIVE_SYNC_MAX_ATTEMPTS", "5")
)

# Quiesce-wait between rendezvous retries: poll until the gen world has settled
# before firing the next init_collective, so each retry lands on a stable world.
#   - QUIESCE_S: joinable set must hold steady at least this long.
#   - QUIESCE_POLL_S: poll cadence while waiting.
#   - QUIESCE_MAX_WAIT_S: hard bound; proceed anyway after this.
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

# Minimum warm-up age before a freshly-added shard counts as joinable.
# Cold cross-cluster RoCE routes need ~60-90s before rendezvous can complete;
# a proven shard (>=1 prior successful init_collective) bypasses this gate.
JOINABLE_MIN_AGE_S: float = float(os.environ.get("NRL_JOINABLE_MIN_AGE_S", "90"))

# Rendezvous timeout per attempt. Each init_collective attempt waits
# this long for all participants to join before giving up.
COLLECTIVE_SYNC_RENDEZVOUS_TIMEOUT_S: float = 150.0

# How long the gen-side joinable set must hold steady before the trainer grows
# the cross-cluster comm. Coalesces replacements that arrive close together
# and prevents per-tick re-init churn. Shrink is NOT debounced.
REJOIN_DEBOUNCE_S: float = float(os.environ.get("NRL_REJOIN_DEBOUNCE_S", "45"))

# async_grpo_train: outer retry around the entire refit.
# Wait at least one full heartbeat cycle + buffer for gen cluster
# recovery (add_shard pod provisioning, vLLM init).
REFIT_RETRY_MAX_ATTEMPTS: int = 3
REFIT_RETRY_SETTLE_S: float = NCCL_HEARTBEAT_TIMEOUT_S + 30.0
