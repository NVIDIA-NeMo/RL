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
# so that reset_collective can drain the hung ops.
COLLECTIVE_SYNC_BACKOFF_BASE_S: float = 5.0
COLLECTIVE_SYNC_BACKOFF_CAP_S: float = NCCL_HEARTBEAT_TIMEOUT_S + 30.0
COLLECTIVE_SYNC_MAX_ATTEMPTS: int = 10

# Rendezvous timeout per attempt. Each init_collective attempt waits
# this long for all participants to join before giving up.
COLLECTIVE_SYNC_RENDEZVOUS_TIMEOUT_S: float = 150.0

# async_grpo_train: outer retry around the entire refit.
# Wait at least one full heartbeat cycle + buffer for gen cluster
# recovery (add_shard pod provisioning, vLLM init).
REFIT_RETRY_MAX_ATTEMPTS: int = 3
REFIT_RETRY_SETTLE_S: float = NCCL_HEARTBEAT_TIMEOUT_S + 30.0
