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

"""Failure handling shared by atomic real-quant refit transports."""

from typing import Any

import ray


def wait_for_real_quant_refit(
    producer_futures: list[ray.ObjectRef],
    consumer_futures: list[ray.ObjectRef],
) -> list[Any]:
    """Wait for both sides while surfacing either side's first failure."""
    pending = [*producer_futures, *consumer_futures]
    consumer_set = set(consumer_futures)
    consumer_results = []

    try:
        while pending:
            ready, pending = ray.wait(pending, num_returns=1)
            future = ready[0]
            result = ray.get(future)
            if future in consumer_set:
                consumer_results.append(result)
    except Exception:
        for future in pending:
            try:
                ray.cancel(future)
            except Exception:
                pass
        raise

    return consumer_results


def shutdown_refit_participants(
    policy: Any, generation: Any, refit_error: Exception
) -> None:
    """Terminate both owners of a partially applied real-quant snapshot."""
    for name, participant in (("generation", generation), ("policy", policy)):
        try:
            if participant.shutdown() is False:
                refit_error.add_note(f"{name} shutdown reported failure")
        except Exception as cleanup_error:
            refit_error.add_note(f"{name} shutdown failed: {cleanup_error}")
