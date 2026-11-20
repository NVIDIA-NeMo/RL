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
"""Spread rollout requests across several sglang routers.

With one router per replica (``sglang_cfg.disagg.router_scope: replica``) each
router balances only its own prefill/decode pool, so the split across routers
is NeMo RL's. Three rules:

* The samples of one prompt share a router, and so a replica. (Inside the
  router, ``cache_aware`` does not pick the prefill engine by prefix: NeMo RL
  sends token ids only, so the policy has no text to match and takes the
  least-loaded engine.)
* A prompt not seen before goes to the usable router with the fewest requests
  in flight, so the request counts stay within one prompt group of each other.
* Only usable routers are picked: a router whose replica has lost an engine
  (fault tolerance kills it until the next refit) fails every request at once,
  which would otherwise keep its in-flight count the lowest and draw most new
  prompts to it.

The dispatcher is touched from the generation object's event loop only, one
coroutine at a time, plus ``reset`` between generation steps.
"""

from collections.abc import Hashable, Sequence


class ReplicaDispatcher:
    """Sticky-by-prompt, least-loaded-first choice among ``num_targets`` routers."""

    def __init__(self, num_targets: int):
        assert num_targets >= 1, "at least one router is needed"
        self._inflight = [0] * num_targets
        self._sticky: dict[Hashable, int] = {}

    @property
    def num_targets(self) -> int:
        return len(self._inflight)

    @staticmethod
    def prompt_key(input_ids: list[int]) -> int:
        """The key the samples of one prompt share: a hash of the prompt tokens."""
        return hash(tuple(input_ids))

    def acquire(self, key: Hashable, usable: Sequence[int]) -> int:
        """Pick a usable router for a request of prompt ``key`` and count it in flight.

        ``usable`` lists the routers whose replica is whole; a prompt that was
        placed on a router that is no longer usable is moved.
        """
        assert usable, "at least one usable router is needed"
        if self.num_targets == 1:
            target = 0
        else:
            sticky = self._sticky.get(key)
            if sticky is not None and sticky in usable:
                target = sticky
            else:
                target = min(usable, key=self._inflight.__getitem__)
                self._sticky[key] = target
        self._inflight[target] += 1
        return target

    def release(self, target: int) -> None:
        """Count a request of ``target`` as finished."""
        assert self._inflight[target] > 0, f"router {target} has nothing in flight"
        self._inflight[target] -= 1

    def reset(self) -> None:
        """Forget prompt placements; the next step balances afresh."""
        self._sticky = {}
