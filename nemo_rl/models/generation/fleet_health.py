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

"""Liveness bookkeeping for the vLLM generation fleet.

Both SingleController rollout paths pick a generation shard by static round-robin with
no idea whether the shard is alive: the native path in
``VllmGeneration._async_generate_base``, and the NeMo-Gym path inside Gym's own
``_resolve_client``. This module owns the missing half — which shards are eligible to
serve, and why.

Deliberately a pure state machine. Probing, restarting and pushing membership are I/O and
belong to the caller, which keeps every transition here testable without Ray, a network,
or a GPU — and keeps one description of "eligible" that both routing adapters read.

The transition that carries the most weight is the one that does *not* exist: a shard
cannot go from ``DEAD`` back to ``HEALTHY`` on its own. A restarted engine holds whatever
weights it loaded at init, so re-admitting it because it started answering probes again
would feed training rollouts generated from a checkpoint hundreds of steps stale --
invisible, and far worse than the outage that caused it. Recovery must pass through
``STALE`` and a completed refit.
"""

from __future__ import annotations

import enum
import time
from dataclasses import dataclass, field
from typing import Callable, Optional


class ShardState(str, enum.Enum):
    """Lifecycle of one generation data-parallel shard."""

    # Serving, and refit to the current trainer weights.
    HEALTHY = "healthy"
    # Failing probes but still eligible; one bad probe should not drain a shard.
    SUSPECT = "suspect"
    # Quarantined. Not eligible, replacement not yet started.
    DEAD = "dead"
    # A replacement engine is coming up.
    RESTARTING = "restarting"
    # Loaded and answering, but holding stale weights. NOT eligible.
    STALE = "stale"
    # Terminal: restarts exhausted, or the node is gone for good.
    RETIRED = "retired"


# States whose shards may be handed traffic. SUSPECT is included on purpose: it means
# "failing, not yet condemned", and draining on a single failed probe would make a
# transient blip cost a shard's worth of throughput.
_SERVING_STATES = frozenset({ShardState.HEALTHY, ShardState.SUSPECT})

# States whose process cannot take part in a collective, because it is gone or coming
# back. This is deliberately NOT the complement of _SERVING_STATES: SUSPECT and STALE are
# not handed traffic but their processes are alive and join a refit normally. Refitting a
# STALE shard is precisely how it stops being stale, so counting it absent would break
# the recovery this distinction exists to enable.
_ABSENT_STATES = frozenset({ShardState.DEAD, ShardState.RESTARTING, ShardState.RETIRED})


@dataclass
class ShardHealth:
    """Everything known about one shard."""

    dp_shard_idx: int
    base_url: Optional[str] = None
    state: ShardState = ShardState.HEALTHY
    consecutive_probe_failures: int = 0
    consecutive_probe_successes: int = 0
    # Failures observed on real generation requests, counted separately from probe
    # failures on purpose. A wedged engine keeps answering ``is_alive``, so folding
    # these into the probe streak lets every ok-probe reset them and the shard never
    # reaches unhealthy_threshold. Only report_success (or a refit) clears this.
    consecutive_reported_failures: int = 0
    restart_attempts: int = 0
    # What this shard was before an aborted refit pulled it out of service, so the
    # promotion afterwards can put it back where it was instead of laundering it to
    # HEALTHY. None whenever the shard is not mid-recovery.
    state_before_partial: Optional["ShardState"] = None
    # Trainer weight version this shard was last refit to.
    weight_version: int = 0
    last_ok_at: float = 0.0
    last_error: str = ""

    @property
    def is_serving(self) -> bool:
        return self.state in _SERVING_STATES


class GenerationFleetExhausted(RuntimeError):
    """Too few shards remain for the run to be worth continuing."""


@dataclass
class FleetHealthPolicy:
    """Thresholds resolved from ``async_rl.generation_fleet_health``.

    Mirrors :class:`nemo_rl.experience.rollout_manager.RolloutTimeouts` in shape: an
    internal dataclass so the BaseModel stays the single home for user-facing defaults.
    """

    unhealthy_threshold: int = 3
    healthy_threshold: int = 2
    max_restart_attempts_per_shard: int = 5
    min_healthy_shards: int = 1

    def __post_init__(self) -> None:
        if self.unhealthy_threshold < 1 or self.healthy_threshold < 1:
            raise ValueError(
                "FleetHealthPolicy thresholds must be >= 1; got "
                f"unhealthy_threshold={self.unhealthy_threshold}, "
                f"healthy_threshold={self.healthy_threshold}"
            )
        if self.min_healthy_shards < 1:
            raise ValueError(
                "FleetHealthPolicy.min_healthy_shards must be >= 1; a run with zero "
                f"serving shards cannot generate. Got {self.min_healthy_shards}"
            )


class GenerationFleetHealth:
    """Tracks per-shard health and exposes the current serving set.

    Args:
        shard_count: Number of generation data-parallel shards.
        policy: Thresholds governing the transitions.
        base_urls: Per-shard OpenAI base URLs, when the deployment has them.
        clock: Monotonic time source; injectable so tests need not sleep.
    """

    def __init__(
        self,
        *,
        shard_count: int,
        policy: FleetHealthPolicy,
        base_urls: Optional[list[Optional[str]]] = None,
        clock: Optional[Callable[[], float]] = None,
    ) -> None:
        if shard_count < 1:
            raise ValueError(f"shard_count must be >= 1, got {shard_count}")
        if base_urls is not None and len(base_urls) != shard_count:
            raise ValueError(
                f"base_urls has {len(base_urls)} entries for {shard_count} shards"
            )

        self._clock = clock if clock is not None else time.monotonic
        self._policy = policy
        self._shards: dict[int, ShardHealth] = {
            idx: ShardHealth(
                dp_shard_idx=idx,
                base_url=base_urls[idx] if base_urls is not None else None,
                last_ok_at=self._clock(),
            )
            for idx in range(shard_count)
        }
        # Bumped whenever the serving set changes. Published as a metric, and nothing
        # else reads it.
        #
        # This used to claim the weight-sync path compared it against the epoch its
        # communicator was built with. It never did -- the comparison described here did
        # not exist anywhere, and reconcile_communicator rebuilt on every call once a shard
        # was gone. That skip is now real, but it compares absent SETS on the synchronizer
        # rather than an epoch from here, because the synchronizer is what knows which
        # membership its current communicator was built with.
        self._membership_epoch: int = 0
        self._last_serving: frozenset[int] = frozenset(self._shards)

    # ── read side ────────────────────────────────────────────────────────

    @property
    def membership_epoch(self) -> int:
        return self._membership_epoch

    @property
    def shard_count(self) -> int:
        return len(self._shards)

    def snapshot(self) -> list[ShardHealth]:
        """Per-shard health, ordered by shard index."""
        return [self._shards[idx] for idx in sorted(self._shards)]

    def serving_shards(self) -> list[int]:
        """Shard indices currently eligible to be handed traffic."""
        return [idx for idx in sorted(self._shards) if self._shards[idx].is_serving]

    def absent_shards(self) -> list[int]:
        """Shard indices whose process cannot take part in a collective.

        The set the weight-refit path cares about, and deliberately not the complement of
        :meth:`serving_shards`: a SUSPECT or STALE shard is withheld from traffic but its
        process is alive and joins the refit normally.
        """
        return [
            idx
            for idx in sorted(self._shards)
            if self._shards[idx].state in _ABSENT_STATES
        ]

    def suspected_shards(self) -> list[int]:
        """Shards the ledger already doubted when the refit went wrong.

        SUSPECT, or STALE having been SUSPECT immediately before an aborted refit marked
        it partial. The second case is the common one at recovery time and would otherwise
        be invisible: ``mark_weights_partial`` runs over every *serving* shard, and SUSPECT
        is a serving state, so the suspicion is overwritten by STALE microseconds before
        anyone asks. ``state_before_partial`` is what preserves it.

        Not for routing -- ``serving_shards`` decides that. This exists for the one caller
        that has independent evidence something in the collective stopped participating and
        needs to know which shard the ledger was already unhappy with.
        """
        return [
            idx
            for idx in sorted(self._shards)
            if self._shards[idx].state is ShardState.SUSPECT
            or (
                self._shards[idx].state is ShardState.STALE
                and self._shards[idx].state_before_partial is ShardState.SUSPECT
            )
        ]

    def state_of(self, shard_idx: int) -> ShardState:
        return self._shards[shard_idx].state

    def serving_base_urls(self) -> list[str]:
        """Base URLs of the serving shards, for pushing to the NeMo-Gym router."""
        urls: list[str] = []
        for idx in self.serving_shards():
            url = self._shards[idx].base_url
            if url:
                urls.append(url)
        return urls

    def counts_by_state(self) -> dict[ShardState, int]:
        # __members__ rather than iterating the class: a str/Enum mixin is iterable at
        # runtime but not modelled as such by the type checker.
        counts = {state: 0 for state in ShardState.__members__.values()}
        for shard in self._shards.values():
            counts[shard.state] += 1
        return counts

    def as_metrics(self) -> dict[str, float]:
        """Flatten into a metric dict for the SingleController logger."""
        metrics = {
            f"gen_fleet/shards/{state.value}": float(count)
            for state, count in self.counts_by_state().items()
        }
        metrics["gen_fleet/membership_epoch"] = float(self._membership_epoch)
        metrics["gen_fleet/serving_shards"] = float(len(self.serving_shards()))
        for shard in self._shards.values():
            metrics[f"gen_fleet/restart_attempts/{shard.dp_shard_idx}"] = float(
                shard.restart_attempts
            )
            # Catches a STALE shard that somehow served: a serving shard whose weight
            # version trails the trainer is a correctness bug, not a capacity problem.
            metrics[f"gen_fleet/shard_weight_version/{shard.dp_shard_idx}"] = float(
                shard.weight_version
            )
        return metrics

    # ── write side ───────────────────────────────────────────────────────

    def record_probe(self, shard_idx: int, *, ok: bool, error: str = "") -> None:
        """Fold one probe result into a shard's state.

        Probes never resurrect a shard that has left the serving set: ``DEAD``,
        ``RESTARTING``, ``STALE`` and ``RETIRED`` all ignore them, because getting an
        answer says nothing about whether the weights are current.
        """
        shard = self._shards[shard_idx]
        if shard.state not in _SERVING_STATES:
            return

        if ok:
            shard.consecutive_probe_failures = 0
            shard.consecutive_probe_successes += 1
            shard.last_ok_at = self._clock()
            if (
                shard.state is ShardState.SUSPECT
                and shard.consecutive_probe_successes >= self._policy.healthy_threshold
            ):
                self._transition(shard, ShardState.HEALTHY)
            return

        shard.consecutive_probe_successes = 0
        shard.consecutive_probe_failures += 1
        shard.last_error = error
        if shard.consecutive_probe_failures >= self._policy.unhealthy_threshold:
            self._transition(shard, ShardState.DEAD)
        elif shard.state is ShardState.HEALTHY:
            self._transition(shard, ShardState.SUSPECT)

    def record_actor_death(self, shard_idx: int, error: str = "") -> None:
        """Record proof that a shard's process is gone. DEAD at once, no counting.

        The counters in :meth:`record_probe` exist to tell a slow shard from a dead one,
        because a probe timeout cannot distinguish them. Some evidence carries no such
        ambiguity: Ray reporting its actor dead means the process is gone, full stop, and
        making that wait for ``unhealthy_threshold`` more rounds of the same answer only
        delays the conclusion.

        The delay was not academic. Detection took ``probe_interval_s *
        unhealthy_threshold``, which the refit deadline could expire inside -- so a refit
        hung on a dead rank aborted while the monitor still had that rank SUSPECT, and
        the rebuild the abort exists to trigger had an empty absent set to work from.
        Job 5925668.

        Ignores shards that are already absent, so a repeat report is idempotent, and
        RETIRED is never disturbed.
        """
        shard = self._shards[shard_idx]
        if shard.state in _ABSENT_STATES:
            return
        if error:
            shard.last_error = error
        shard.consecutive_probe_successes = 0
        shard.consecutive_probe_failures = self._policy.unhealthy_threshold
        self._transition(shard, ShardState.DEAD)

    def condemn_silent_participant(self, shard_idx: int, *, reason: str) -> None:
        """Quarantine a shard that is alive but did not take part in a collective.

        Distinct from :meth:`record_actor_death`, which means "Ray says the process is
        gone". This one means "the process answers and still broke the refit", which is the
        failure mode ``is_alive`` cannot see: it is answered by the Ray actor and never
        touches the engine, so a wedged engine reads as healthy forever.

        DEAD rather than STALE, because the point is to make it *absent*: STALE keeps it in
        the refit, which is precisely what just hung. The caller owes an independent reason
        to believe this shard is the culprit -- see the single-suspect rule in
        ``_recover_from_failed_refit`` -- because this is a judgement, not an observation.
        """
        shard = self._shards[shard_idx]
        if shard.state in _ABSENT_STATES:
            return
        shard.last_error = reason
        self._transition(shard, ShardState.DEAD)

    def report_failure(self, shard_idx: int, error: BaseException) -> None:
        """Record a failure observed by a routing adapter rather than by a probe.

        The adapters are the only components actually issuing generation requests, so
        they see failures a liveness probe cannot -- a shard that answers ``/health``
        and still errors on every generation.

        Counted on its own streak rather than folded into :meth:`record_probe`. That
        used to be the implementation, and it could not condemn the very shard it exists
        for: a wedged engine answers ``is_alive``, so an ok-probe every
        ``probe_interval_s`` reset the shared counter, and reported failures only
        accumulated if ``unhealthy_threshold`` of them landed inside one probe window.
        Under load they do; under a trickle the shard oscillated HEALTHY<->SUSPECT
        forever. A streak that only a *successful generation* clears says what is
        actually meant: this shard has failed N requests in a row.
        """
        shard = self._shards[shard_idx]
        if shard.state not in _SERVING_STATES:
            return
        shard.consecutive_reported_failures += 1
        shard.last_error = f"{type(error).__name__}: {error}"
        if shard.consecutive_reported_failures >= self._policy.unhealthy_threshold:
            self._transition(shard, ShardState.DEAD)
        elif shard.state is ShardState.HEALTHY:
            self._transition(shard, ShardState.SUSPECT)

    def report_success(self, shard_idx: int) -> None:
        """Record a generation that completed on this shard.

        The reset half of :meth:`report_failure`. Without it the reported streak is
        monotonic and every shard eventually reaches ``unhealthy_threshold`` given a long
        enough run, however healthy it is.
        """
        self._shards[shard_idx].consecutive_reported_failures = 0

    def shard_for_base_url(self, url: str) -> Optional[int]:
        """Reverse the shard -> base URL mapping, for failures reported by URL.

        The NeMo-Gym router knows its backends only as URLs; the ledger keys everything
        by shard index.
        """
        for shard in self._shards.values():
            if shard.base_url == url:
                return shard.dp_shard_idx
        return None

    def mark_restarting(self, shard_idx: int) -> None:
        """A replacement engine is being brought up for a dead shard."""
        shard = self._shards[shard_idx]
        if shard.state is ShardState.RETIRED:
            return
        shard.restart_attempts += 1
        if shard.restart_attempts > self._policy.max_restart_attempts_per_shard:
            self.retire(shard_idx, reason="restart attempts exhausted")
            return
        self._transition(shard, ShardState.RESTARTING)

    def mark_loaded(self, shard_idx: int) -> None:
        """The replacement finished loading. It holds stale weights until refit."""
        shard = self._shards[shard_idx]
        if shard.state is ShardState.RETIRED:
            return
        self._transition(shard, ShardState.STALE)

    def mark_weights_partial(self, shard_idx: int) -> None:
        """An aborted refit left this shard holding a mix of old and new weights.

        STALE rather than DEAD: the process is alive and refits normally, it just must not
        serve until a refit completes. That is exactly what STALE already means, and
        because DEAD -> HEALTHY is unreachable and STALE ignores successful probes, the
        only route back into the serving set is report_refit -- which is the property that
        makes partial weights safe to hold.

        Absent shards are left alone; they are not serving and their problem is not weights.
        """
        shard = self._shards[shard_idx]
        if shard.state in _ABSENT_STATES:
            return
        if shard.state is not ShardState.STALE:
            # Remembered so report_refit can put a SUSPECT shard back where it was.
            # Without this the recovery launders it: SUSPECT -> STALE here, STALE ->
            # HEALTHY there, and consecutive_reported_failures is zeroed by a refit that
            # had nothing to do with why the shard was suspect. That counter is the only
            # one that can condemn a wedged engine -- it is kept separate from the probe
            # streak precisely because such an engine still answers is_alive.
            shard.state_before_partial = shard.state
        self._transition(shard, ShardState.STALE)

    def report_refit(self, shard_idx: int, *, weight_version: int) -> None:
        """A completed refit is the only way back into the serving set."""
        shard = self._shards[shard_idx]
        if shard.state is ShardState.RETIRED:
            return
        was_before_partial, shard.state_before_partial = (
            shard.state_before_partial,
            None,
        )
        shard.weight_version = weight_version
        shard.consecutive_probe_failures = 0
        shard.consecutive_probe_successes = 0
        if was_before_partial is ShardState.SUSPECT:
            # Already failing generations before the refit pulled it out of service, for
            # reasons the refit did not address. It goes back to SUSPECT with its
            # reported streak intact, still on its way to DEAD. Promoting it to HEALTHY
            # here would reset that streak and leave a wedged engine unable to reach
            # unhealthy_threshold for as long as refits keep happening.
            self._transition(shard, ShardState.SUSPECT)
            return
        # A refit means a fresh engine holding current weights; whatever it failed at
        # before is not evidence against what it is now.
        shard.consecutive_reported_failures = 0
        shard.last_ok_at = self._clock()
        self._transition(shard, ShardState.HEALTHY)

    def retire(self, shard_idx: int, *, reason: str) -> None:
        """Remove a shard permanently. Training continues on what is left."""
        shard = self._shards[shard_idx]
        shard.last_error = reason
        self._transition(shard, ShardState.RETIRED)

    def raise_if_exhausted(self) -> None:
        """Raise once too few shards remain for the run to be worth continuing."""
        serving = self.serving_shards()
        if len(serving) < self._policy.min_healthy_shards:
            counts = {
                state.value: count
                for state, count in self.counts_by_state().items()
                if count
            }
            raise GenerationFleetExhausted(
                f"only {len(serving)} generation shard(s) serving, below "
                f"min_healthy_shards={self._policy.min_healthy_shards}; fleet state "
                f"is {counts}"
            )

    # ── internals ────────────────────────────────────────────────────────

    def _transition(self, shard: ShardHealth, new_state: ShardState) -> None:
        if shard.state is new_state:
            return
        previous = shard.state
        shard.state = new_state
        print(
            f"gen_fleet: shard {shard.dp_shard_idx} {previous.value} -> {new_state.value}"
            + (f" ({shard.last_error})" if shard.last_error else ""),
            flush=True,
        )
        self._refresh_membership()

    def _refresh_membership(self) -> None:
        serving = frozenset(self.serving_shards())
        if serving != self._last_serving:
            self._last_serving = serving
            self._membership_epoch += 1


@dataclass
class HealthyShardSelector:
    """Picks a serving shard, preferring the one with the fewest requests in flight.

    Least-outstanding rather than round-robin because it is strictly better for LLM
    serving: it steers away from a shard that is merely slow or wedged without needing
    that to be diagnosed first.
    """

    monitor: GenerationFleetHealth
    _inflight: dict[int, int] = field(default_factory=dict)

    def next_shard(self) -> int:
        """Return the shard index to serve the next request.

        Raises:
            NoHealthyShards: No shard is currently eligible.
        """
        # Imported here rather than at module scope: experience.failures pulls in the
        # rollout exception hierarchy, and generation should not depend on it to define
        # its own health model.
        from nemo_rl.experience.failures import NoHealthyShards

        serving = self.monitor.serving_shards()
        if not serving:
            counts = {
                state.value: count
                for state, count in self.monitor.counts_by_state().items()
                if count
            }
            raise NoHealthyShards(
                f"no generation shard is eligible to serve; fleet state is {counts}"
            )
        return min(serving, key=lambda idx: (self._inflight.get(idx, 0), idx))

    def acquire(self, shard_idx: int) -> None:
        self._inflight[shard_idx] = self._inflight.get(shard_idx, 0) + 1

    def release(self, shard_idx: int) -> None:
        self._inflight[shard_idx] = max(0, self._inflight.get(shard_idx, 0) - 1)

    def inflight(self, shard_idx: int) -> int:
        return self._inflight.get(shard_idx, 0)
