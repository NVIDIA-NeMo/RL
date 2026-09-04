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

"""Group-aware waiting-queue ordering for vLLM's V1 scheduler.

This policy takes the probe/group-length idea only as far as reordering whole
requests already sitting in one engine's waiting queue. It does not split requests into generation chunks,
re-enqueue them by cumulative generated progress, migrate their KV cache across
instances, or implement a long-term underserved-group fairness rule.
Each request carries its group id in ``Request.priority``.

Use this scheduler with vLLM's scheduling policy left at ``fcfs``. The group id
occupies the priority field, so vLLM's priority heap must not also reorder it.
"""

from collections import deque
from collections.abc import Iterable, Iterator

from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.request_queue import RequestQueue
from vllm.v1.request import Request


class OracleLengthRequestQueue(RequestQueue):
    """Order waiting requests by their known output length, longest first.

    This queue is an experiment-only upper bound. The benchmark passes the
    forced output length through ``Request.priority``; production requests do
    not know this value before generation.
    """

    def __init__(self) -> None:
        self._by_length: dict[int, deque[Request]] = {}
        self._size = 0

    @staticmethod
    def _length(request: Request) -> int:
        return int(request.priority)

    def _best_length(self) -> int:
        try:
            return max(
                length for length, requests in self._by_length.items() if requests
            )
        except ValueError as error:
            raise IndexError("empty queue") from error

    def add_request(self, request: Request) -> None:
        self._by_length.setdefault(self._length(request), deque()).append(request)
        self._size += 1

    def pop_request(self) -> Request:
        request = self._by_length[self._best_length()].popleft()
        self._size -= 1
        return request

    def peek_request(self) -> Request:
        return self._by_length[self._best_length()][0]

    def prepend_request(self, request: Request) -> None:
        self._by_length.setdefault(self._length(request), deque()).appendleft(request)
        self._size += 1

    def prepend_requests(self, requests: RequestQueue) -> None:
        for request in requests:
            self.prepend_request(request)

    def remove_request(self, request: Request) -> None:
        self._by_length[self._length(request)].remove(request)
        self._size -= 1

    def remove_requests(self, requests: Iterable[Request]) -> None:
        requests_to_remove = requests if isinstance(requests, set) else set(requests)
        for length, length_requests in self._by_length.items():
            kept = deque(
                request
                for request in length_requests
                if request not in requests_to_remove
            )
            self._size -= len(length_requests) - len(kept)
            self._by_length[length] = kept

    def __bool__(self) -> bool:
        return self._size > 0

    def __len__(self) -> int:
        return self._size

    def __iter__(self) -> Iterator[Request]:
        for length in sorted(self._by_length, reverse=True):
            yield from self._by_length[length]


class ProbeLfsRequestQueue(RequestQueue):
    """Order whole waiting requests by an online group-length estimate.

    If the probes do not fill the first batch, requests from still-unknown
    groups fill the remaining slots round-robin. This avoids draining one
    group merely because its requests arrived first. The round robin applies
    only while a group is unknown; it is not a persistent fairness rule.
    """

    def __init__(self, group_estimates: dict[int, int]) -> None:
        self._group_estimates = group_estimates
        self._by_group: dict[int, deque[Request]] = {}
        self._probed_groups: set[int] = set()
        self._probe_request_ids: dict[int, str] = {}
        self._unknown_admission_counts: dict[int, int] = {}
        self._unknown_request_ids: set[str] = set()
        self._size = 0

    def _group_key(self, group_id: int, front: Request) -> tuple[int, float, float]:
        estimate = self._group_estimates.get(group_id)
        if estimate is None:
            if group_id not in self._probed_groups:
                # Before normal LFS admission, launch one request from every
                # group so each group has an online length probe in flight.
                return (0, 0.0, front.arrival_time)
            # A probed group with no completed request is still unknown and
            # remains ahead of groups with finite estimates. Among unknown
            # groups, admit the fewest-sampled group first so spare slots in
            # the first batch are filled round-robin.
            return (
                1,
                float(self._unknown_admission_counts.get(group_id, 0)),
                front.arrival_time,
            )
        return (2, -float(estimate), front.arrival_time)

    def _best_group(self) -> int:
        best_group = None
        best_key = None
        for group_id, requests in self._by_group.items():
            if not requests:
                continue
            key = self._group_key(group_id, requests[0])
            if best_key is None or key < best_key:
                best_group = group_id
                best_key = key
        if best_group is None:
            raise IndexError("empty queue")
        return best_group

    def add_request(self, request: Request) -> None:
        self._by_group.setdefault(request.priority, deque()).append(request)
        self._size += 1

    def pop_request(self) -> Request:
        group_id = self._best_group()
        request = self._by_group[group_id].popleft()
        if group_id not in self._group_estimates:
            self._probed_groups.add(group_id)
            self._probe_request_ids.setdefault(group_id, request.request_id)
            self._unknown_admission_counts[group_id] = (
                self._unknown_admission_counts.get(group_id, 0) + 1
            )
            self._unknown_request_ids.add(request.request_id)
        self._size -= 1
        return request

    def peek_request(self) -> Request:
        return self._by_group[self._best_group()][0]

    def prepend_request(self, request: Request) -> None:
        group_id = request.priority
        if (
            group_id not in self._group_estimates
            and request.request_id in self._unknown_request_ids
        ):
            self._unknown_request_ids.remove(request.request_id)
            self._unknown_admission_counts[group_id] -= 1
        if (
            group_id not in self._group_estimates
            and self._probe_request_ids.get(group_id) == request.request_id
        ):
            # A scheduler skip or preemption did not successfully keep this
            # probe running. Restore the group to the unprobed tier.
            self._probed_groups.discard(group_id)
            del self._probe_request_ids[group_id]
        self._by_group.setdefault(request.priority, deque()).appendleft(request)
        self._size += 1

    def prepend_requests(self, requests: RequestQueue) -> None:
        for request in requests:
            self.prepend_request(request)

    def remove_request(self, request: Request) -> None:
        self._by_group[request.priority].remove(request)
        self._size -= 1

    def remove_requests(self, requests: Iterable[Request]) -> None:
        requests_to_remove = requests if isinstance(requests, set) else set(requests)
        for group_id, group_requests in self._by_group.items():
            kept = deque(
                request
                for request in group_requests
                if request not in requests_to_remove
            )
            self._size -= len(group_requests) - len(kept)
            self._by_group[group_id] = kept

    def __bool__(self) -> bool:
        return self._size > 0

    def __len__(self) -> int:
        return self._size

    def __iter__(self) -> Iterator[Request]:
        groups = sorted(
            (group_id for group_id, requests in self._by_group.items() if requests),
            key=lambda group_id: self._group_key(
                group_id, self._by_group[group_id][0]
            ),
        )
        for group_id in groups:
            yield from self._by_group[group_id]


class ProbeLfsScheduler(AsyncScheduler):
    """Experimental async scheduler with group-LFS waiting admission.

    vLLM selects ``AsyncScheduler`` automatically only when no custom
    ``scheduler_cls`` is supplied. Inheriting it here preserves the default
    schedule/execute overlap when this policy is installed as a custom
    scheduler.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._group_estimates: dict[int, int] = {}
        self.waiting = ProbeLfsRequestQueue(self._group_estimates)

    def _update_request_with_output(
        self, request: Request, new_token_ids: list[int]
    ) -> tuple[list[int], bool]:
        result = super()._update_request_with_output(request, new_token_ids)
        _, stopped = result
        if stopped:
            group_id = request.priority
            output_length = request.num_output_tokens
            if output_length > self._group_estimates.get(group_id, 0):
                self._group_estimates[group_id] = output_length
        return result


class OracleLengthScheduler(AsyncScheduler):
    """Experiment-only async scheduler that knows exact output lengths."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.waiting = OracleLengthRequestQueue()
