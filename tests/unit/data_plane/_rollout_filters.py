# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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
"""Example driver-side rollout filter — test-only sample.

Not production code (yet). Lives under ``tests/`` to show users how to
plug an arbitrary filter into the async-RL flow without prescribing a
public ABC. If async-RL adopts this pattern, lift it into
``nemo_rl/data_plane/`` then; until then it's a sample in the test
that uses it.

Pattern: mirror TQ's :class:`transfer_queue.sampler.BaseSampler` shape,
but the predicate runs on the driver where it can see in-process state
(current weight version, dynamic configs) that's awkward to serialize
through TQ's ``sampling_config``. Subclass returns ``(keep, drop)`` —
caller drives :func:`shard_meta_for_dp` on ``keep`` and
:meth:`clear_samples` on ``drop``.

Contract notes
--------------
- Both ``(keep, drop)`` are fresh ``KVBatchMeta`` instances (via
  ``meta.subset(...)``) — never aliases of the input. Caller can mutate
  either half without affecting the source.
- Filters that need driver state (e.g. ``current_weight_version``)
  accept it via ``**kwargs``; they MUST raise a clear error when the
  required kwarg is missing rather than relying on Python's default
  ``TypeError``, so generic dispatchers get a helpful message.
- Filters that read a tag MUST raise on missing-tag / None-value
  rather than silently treat absent data as the default — async-RL's
  whole point is "drop the unsuitable", and "we don't know" must
  surface, not be assumed safe.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from nemo_rl.data_plane.interfaces import KVBatchMeta


class BaseRolloutFilter(ABC):
    """Example shape for a driver-side async-RL rollout filter.

    Subclasses may require additional kwargs (e.g. current weight
    version). Document them on the subclass and raise a clear error
    when missing.
    """

    @abstractmethod
    def filter(
        self,
        meta: KVBatchMeta,
        **kwargs: Any,
    ) -> tuple[KVBatchMeta, KVBatchMeta]:
        """Partition ``meta`` into ``(keep, drop)``.

        Both halves share ``meta.partition_id`` and are fresh copies
        (never aliases of the input).
        """
        raise NotImplementedError


class StalenessFilter(BaseRolloutFilter):
    """Drop samples whose ``weight_version`` tag is older than ``max_age``.

    Producer stamps ``tag["weight_version"] = v`` (an int) at rollout
    time; filter keeps samples where
    ``current_weight_version - v <= max_age``.

    Raises (rather than silently dropping or keeping) when:
      - ``current_weight_version`` kwarg is missing
      - ``meta.tags`` is None on a non-empty meta
      - A sample's tag dict lacks ``"weight_version"``
      - A sample's ``weight_version`` is None or a float (use int only)

    Example:
        >>> f = StalenessFilter(max_age=2)
        >>> keep, drop = f.filter(meta, current_weight_version=5)
        >>> # keep: tag["weight_version"] in {3, 4, 5}
        >>> # drop: tag["weight_version"] <= 2
    """

    def __init__(self, max_age: int) -> None:
        if max_age < 0:
            raise ValueError(f"max_age must be >= 0, got {max_age}")
        self.max_age = int(max_age)

    def filter(
        self,
        meta: KVBatchMeta,
        *,
        current_weight_version: int | None = None,
        **kwargs: Any,
    ) -> tuple[KVBatchMeta, KVBatchMeta]:
        if current_weight_version is None:
            raise TypeError(
                "StalenessFilter.filter requires keyword argument "
                "'current_weight_version: int'."
            )

        # Always return fresh copies — never alias the input.
        if meta.size == 0:
            return meta.subset([]), meta.subset([])

        if meta.tags is None:
            raise ValueError(
                "StalenessFilter requires per-sample tags carrying "
                "'weight_version'; got meta.tags=None. Stamp the tag at "
                "rollout time."
            )

        keep_ix: list[int] = []
        for i, t in enumerate(meta.tags):
            if "weight_version" not in t:
                raise ValueError(
                    f"sample {meta.sample_ids[i]!r} missing required tag "
                    f"'weight_version'"
                )
            v = t["weight_version"]
            if v is None or not isinstance(v, int) or isinstance(v, bool):
                # bool is an int subclass — reject it explicitly to avoid
                # confusing True/False arithmetic.
                raise ValueError(
                    f"sample {meta.sample_ids[i]!r} has "
                    f"weight_version={v!r}; expected non-None int"
                )
            if current_weight_version - v <= self.max_age:
                keep_ix.append(i)

        keep_set = set(keep_ix)
        drop_ix = [i for i in range(meta.size) if i not in keep_set]
        return meta.subset(keep_ix), meta.subset(drop_ix)
