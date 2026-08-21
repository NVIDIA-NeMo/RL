# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

#!/bin/bash
set -xeuo pipefail # Exit immediately if a command exits with a non-zero status

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
PROJECT_ROOT=$(realpath ${SCRIPT_DIR}/../..)

cd ${PROJECT_ROOT}

# run_test [fast] <command...>
# - "run_test fast <cmd>" = always runs (both fast and full modes)
# - "run_test <cmd>"      = only runs in full mode; skipped when FAST=1
run_test() {
    if [[ "$1" == "fast" ]]; then
        shift
        time "$@"
    elif [[ "${FAST:-0}" == "1" ]]; then
        echo "FAST: Skipping: $*"
    else
        time "$@"
    fi
}

run_test fast uv run --no-sync bash ./tests/functional/grpo_dp_single_controller.sh
run_test fast uv run --no-sync bash ./tests/functional/grpo_async_gym_single_controller.sh
# Full mode only (~10 min): SIGKILLs a generation worker and asserts the job fails fast
# and attributably instead of wedging. This is the ONLY end-to-end check of the
# containment behaviour -- without it, a regression that restores the silent wedge is
# caught by nothing, because a wedged job produces no exception and no failing assertion
# anywhere else.
run_test uv run --no-sync bash ./tests/functional/grpo_dp_single_controller_chaos.sh
# Full mode only: the same Gym run, but with NeMo-Gym pointed at the NeMo-RL-owned router.
# Without this the router has no functional coverage at all -- the default Gym run above
# leaves it disabled, so a regression in the proxy would ship silently.
#
# gen_kl_error is the assertion that earns its keep here: it compares vLLM's logprobs
# against the trainer's recomputation, so a proxy that corrupts or truncates a response
# blows it up. A run that merely completes would not prove the payload survived the hop.
run_test uv run --no-sync bash ./tests/functional/grpo_async_gym_single_controller.sh \
    ++async_rl.generation_router.enabled=true \
    ++async_rl.generation_fleet_health.enabled=true

# ...and the property that run CANNOT prove. It is dp_size=1, so _pick_backend has one
# choice, the serving set never shrinks, and the no-healthy-backend path never fires: it
# demonstrates pass-through, not failover. This one runs two generation shards, kills one
# mid-run, and asserts the serving set shrinks so NeMo-Gym stops being handed the corpse.
#
# EXPECT defaults to quarantine deliberately. Surviving the loss needs the communicator
# rebuild that lands later in this stack -- without it the next refit broadcasts to the
# dead rank and hangs in NCCL (job 6258553 sat there for 33 minutes). Asserting survival
# here would assert a property this part does not implement.
#
# Needs >= 3 GPUs (2 generation + 1 trainer) and self-skips below that, so it is inert on
# the 2-GPU L1 runners and only does its job on a larger box.
run_test uv run --no-sync bash ./tests/functional/grpo_sc_gym_router_failover.sh

# ...and now the other half, which only this part of the stack can satisfy. Same kill,
# but the run must CONTINUE: reconcile_communicator rebuilds the refit group over the
# survivors, so the broadcast no longer addresses the rank that died. Before this, that
# refit hung inside NCCL and the run wedged with the stall watchdog warning -- quarantine
# without recovery (job 6258553).
#
# The Gym-path counterpart of grpo_sc_generation_shard_recovery.sh, which covers the same
# recovery on the native path.
run_test env EXPECT=survival uv run --no-sync bash ./tests/functional/grpo_sc_gym_router_failover.sh
# Full mode only: kills a generation shard and asserts the run carries on. Needs >= 3
# GPUs so that losing a shard still leaves a fleet, and self-skips below that rather
# than passing vacuously.
#
# Deliberately alongside the chaos test above, not instead of it: that one asserts a
# bounded FAILURE on a fleet with nothing to fall back to, this one asserts SURVIVAL when
# a shard remains. Opposite behaviours, and a regression in either is invisible to the
# other.
run_test uv run --no-sync bash ./tests/functional/grpo_sc_generation_shard_recovery.sh
# Same scenario on the reshard transport, which recovers by a different route: it also
# rebuilds its per-PP-stage bulk groups and regenerates the refit plan. Only this path
# has to keep a plan and a communicator agreeing about the fleet size.
run_test env REFIT_TRANSPORT=nccl_reshard uv run --no-sync bash ./tests/functional/grpo_sc_generation_shard_recovery.sh
# The same scenario with the kill placed INSIDE the refit collective rather than at a step
# boundary. That window is only ~10% of wall-clock, so the default variant reaches it by
# chance -- it both passed and wedged on consecutive runs of identical code. This is the
# only test that reliably exercises the abort-and-rebuild path.
run_test env KILL_DURING_REFIT=true uv run --no-sync bash ./tests/functional/grpo_sc_generation_shard_recovery.sh
# ...and the same mid-refit kill on the RESHARD transport, which is a different abort.
#
# The two are not interchangeable. The collective path aborts one communicator; reshard
# holds TWO families -- the per-PP-stage bulk groups and the shared model_update_group --
# and a hang can be in either, so the watchdog is handed both and the rebuild has to
# regenerate the refit plan as well as the communicators. Nothing about that is exercised
# by the step-boundary variant above, which recovers between refits and never aborts.
#
# Without this, the reshard abort path had only signature assertions behind it: the
# deadline was plumbed and unit-tested, but no test had ever made a reshard refit
# actually abort on hardware.
run_test env REFIT_TRANSPORT=nccl_reshard KILL_DURING_REFIT=true uv run --no-sync bash ./tests/functional/grpo_sc_generation_shard_recovery.sh

# The only variant that reaches the refit watchdog. The two above kill the victim, and a
# killed actor produces ActorDiedError within milliseconds -- which recovers the run off
# the pre-existing actor-death path and leaves the deadline unexercised. Job 6405953
# passed both with RefitAborted appearing zero times.
#
# Freezing the victim with SIGSTOP instead leaves Ray seeing a healthy actor that has
# simply stopped participating, which is the case the abort exists for and the one its
# error message names. This variant asserts RefitAborted actually appears, so it cannot
# quietly degrade into testing the same path as the two above.
#
# It asserts the abort only, not recovery, and a NON-ZERO exit is its pass. A frozen rank
# never becomes absent -- measured on 4xGB200 it reaches SUSPECT and stops there -- so the
# reconcile correctly refuses to rebuild over a fleet that still holds a silent rank. The
# gain being pinned is that the run ends attributably in seconds rather than wedging in
# NCCL forever; abort-and-recover is what the two killed variants above cover.
run_test env KILL_DURING_REFIT=true FREEZE_VICTIM=true uv run --no-sync bash ./tests/functional/grpo_sc_generation_shard_recovery.sh

# grpo_dp_single_controller_chaos.sh again, this time killing a worker that is mid-rollout
# rather than between calls. Registered because pinning the victim state -- which is what
# makes that test reproducible at all -- would otherwise silently drop a scenario the old,
# non-deterministic selection used to hit by chance. The two fail by different routes:
# killing an idle worker leaves the loss to be *detected*, killing a serving one destroys
# an in-flight RPC that surfaces at once (222s vs 12s when measured). A regression in
# either is invisible to the other.
#
# Cheap to add: the serving path fails in seconds, so this is dominated by startup.
run_test env VICTIM_STATE=serving uv run --no-sync bash ./tests/functional/grpo_dp_single_controller_chaos.sh

# Checkpoint save/restore (upstream #3429).
run_test uv run --no-sync bash ./tests/functional/grpo_checkpoint_single_controller.sh

cd ${PROJECT_ROOT}/tests
if compgen -G ".coverage*" > /dev/null; then
    coverage combine .coverage*
fi
