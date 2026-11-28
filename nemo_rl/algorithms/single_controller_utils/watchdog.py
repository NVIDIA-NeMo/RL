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

"""Stall detection for a SingleController run: its config and the driver-side watch."""

import time
from typing import Any, Literal

import ray
from pydantic import BaseModel, PositiveFloat, model_validator


class WatchdogConfig(BaseModel, extra="allow"):
    """Last-resort detection for stalls that no other layer catches."""

    # How often the watchdog task runs its checks.
    interval_s: PositiveFloat = 30.0
    # Rollouts in flight but none committed for this long counts as a stall.
    stall_timeout_s: PositiveFloat = 600.0
    # Whether a detected stall only reports, or ends the run.
    stall_action: Literal["warn", "abort"] = "warn"
    # Poll NeMo-Gym's own RunHelper for dead subprocess servers each tick.
    gym_subprocess_check: bool = True

    @model_validator(mode="after")
    def _check_consistent(self) -> "WatchdogConfig":
        if self.stall_timeout_s <= self.interval_s:
            raise ValueError(
                f"async_rl.stall_watchdog.stall_timeout_s ({self.stall_timeout_s}) must be "
                f"> interval_s ({self.interval_s}); otherwise the watchdog reports a "
                "stall before it has had a chance to observe one."
            )
        return self


def run_with_controller_liveness_watch(
    sc: ray.actor.ActorHandle, watchdog_config: WatchdogConfig
) -> dict[str, Any]:
    """Await the SC run, polling ping() so a frozen event loop cannot hide.

    The in-actor watchdog is an asyncio task on the SC's own event loop, so it cannot
    observe that loop being blocked -- by a synchronous Ray call into a wedged worker,
    say. The driver is a separate process that already holds the handle, which makes it
    the cheapest possible external observer; no supervisor actor required.

    ping() returning is the liveness signal. A slow reply is not a freeze, so the check
    only escalates once the loop has been unresponsive for the same budget the in-actor
    watchdog uses to call a stall. A controller that never answered a ping is reported
    as unstarted rather than stalled: an unschedulable actor and a wedged loop look the
    same from here, but they are fixed in different places.
    """
    run_ref = sc.run.remote()
    last_pong_at = time.monotonic()
    started = False

    ready: list[Any] = []
    while not ready:
        ready, _ = ray.wait([run_ref], timeout=watchdog_config.interval_s)
        if ready:
            break

        try:
            ray.get(sc.ping.remote(), timeout=watchdog_config.interval_s)
        except Exception as error:
            unresponsive_s = time.monotonic() - last_pong_at
            print(
                f"SingleController ping failed after {unresponsive_s:.0f}s "
                f"unresponsive: {type(error).__name__}: {error}",
                flush=True,
            )
            if unresponsive_s > watchdog_config.stall_timeout_s:
                if not started:
                    raise RuntimeError(
                        f"SingleController has not started after {unresponsive_s:.0f}s. "
                        "With a dedicated Ray head it can only run there; check that "
                        "the head has a free CPU (`ray status`)."
                    ) from error
                raise RuntimeError(
                    "SingleController event loop has been unresponsive for "
                    f"{unresponsive_s:.0f}s (stall_timeout_s="
                    f"{watchdog_config.stall_timeout_s}); its in-actor watchdog runs "
                    "on that loop and cannot report this."
                ) from error
        else:
            started = True
            last_pong_at = time.monotonic()
    return ray.get(run_ref)
