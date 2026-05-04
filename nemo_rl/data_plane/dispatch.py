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
"""Driver-side decorator that makes a Policy method polymorphic over
``BatchedDataDict`` (legacy in-memory path) and :class:`KVBatchMeta`
(TransferQueue-mediated 1-hop fetch path).

Pairs with the worker-side ``_fetch`` helper on
:class:`AbstractPolicyWorker`. The split mirrors the actual process
boundary — driver concerns (sharding, axis annotations, which worker
method to dispatch) live here; worker concerns (TQ fetch, codec,
TP/CP/PP broadcast, transforms) live on the worker.

See ``research/data_plane_integration_plan.md`` §Stage 4.
"""

from __future__ import annotations

from contextlib import nullcontext
from functools import wraps
from typing import Any, Callable

from nemo_rl.data_plane.interfaces import KVBatchMeta


def dp_dispatch(
    *,
    sharder: Callable[[KVBatchMeta, int], list[KVBatchMeta]],
    sharded_axes: list[str],
    replicate_axes: list[str],
    worker_method: str,
    aggregate: Callable[[list[Any]], Any],
    output_is_replicated: list[str] | None = None,
) -> Callable:
    """Make a Policy method polymorphic over BatchedDataDict / KVBatchMeta.

    When the wrapped method is called with a regular ``BatchedDataDict``
    (or anything that isn't a :class:`KVBatchMeta`), the decorator is a
    transparent pass-through to the original function — the legacy
    in-memory path runs unchanged.

    When called with a :class:`KVBatchMeta`, the decorator:

      1. Calls ``sharder(meta, dp_world_size)`` to split metadata into
         per-DP-rank shards. No tensor data crosses the driver.
      2. Dispatches ``worker_method`` to all workers via
         :meth:`RayWorkerGroup.run_all_workers_sharded_data` with the
         given ``sharded_axes`` / ``replicate_axes``. Each DP rank
         receives its own ``KVBatchMeta``; TP/CP/PP siblings receive
         the same shard (the worker bridge picks one of them to fetch
         when ``fetch_policy='leader_broadcast'`` is in use).
      3. Calls ``aggregate(results)`` to assemble the per-rank outputs
         back into the shape the legacy method returned.

    Args:
        sharder: ``(KVBatchMeta, dp_world_size) -> list[KVBatchMeta]``.
            Phase 1 default is :func:`shard_keys_by_seqlen`.
        sharded_axes: passed through as ``in_sharded_axes``. Phase 1
            always ``["data_parallel"]``.
        replicate_axes: passed through as ``replicate_on_axes``. Phase 1
            ``["context_parallel", "tensor_parallel", "pipeline_parallel"]``.
        worker_method: name of the worker method to invoke. Workers must
            implement a ``*_presharded`` method that accepts a
            ``KVBatchMeta`` as its first argument.
        aggregate: combines the per-rank result list into a single
            return value matching the legacy method's contract.
        output_is_replicated: defaults to ``replicate_axes`` (de-dupes
            outputs across replicated ranks).

    Returns:
        A decorator that wraps a Policy method.
    """

    def decorator(fn: Callable) -> Callable:
        @wraps(fn)
        def wrapper(self, data, *args, timer=None, **kwargs):
            is_meta = isinstance(data, KVBatchMeta)
            is_meta_list = (
                isinstance(data, list)
                and len(data) > 0
                and isinstance(data[0], KVBatchMeta)
            )
            if not (is_meta or is_meta_list):
                # Legacy BatchedDataDict path — call original fn unchanged.
                if timer is not None:
                    return fn(self, data, *args, timer=timer, **kwargs)
                return fn(self, data, *args, **kwargs)

            # TQ path: require keyword args from the caller. Ray's
            # `run_all_workers_sharded_data` doesn't accept *args anyway,
            # so we'd just have to translate positional → keyword here.
            # Cleaner to push the kwarg-only convention up to the call
            # site (one extra `=` per arg) than to do reflection here.
            if args:
                raise TypeError(
                    f"{fn.__name__}(meta=..., ...) requires keyword args "
                    f"on the TransferQueue dispatch path. Got positional "
                    f"args: {args!r}. Pass them as keywords instead."
                )

            # TransferQueue-mediated 1-hop path.
            method_name = fn.__name__
            dp_size = self.sharding_annotations.get_axis_size("data_parallel")

            with timer.time(f"policy_{method_name}/sharding_data") if timer else nullcontext():
                if is_meta_list:
                    # Driver already balanced + pre-sharded (e.g. when sequence
                    # packing / dynamic batching needs ``bin_count_multiple=DP_world``
                    # to keep collective counts uniform across DP ranks). Skip
                    # the sharder; just validate cardinality.
                    shards = data
                    if len(shards) != dp_size:
                        raise ValueError(
                            f"{fn.__name__}: pre-sharded meta list has "
                            f"{len(shards)} entries but DP world size is "
                            f"{dp_size}."
                        )
                else:
                    shards = sharder(data, dp_size)

            with timer.time(f"policy_{method_name}/dispatch") if timer else nullcontext():
                futures = self.worker_group.run_all_workers_sharded_data(
                    worker_method,
                    meta=shards,
                    in_sharded_axes=sharded_axes,
                    replicate_on_axes=replicate_axes,
                    output_is_replicated=output_is_replicated
                    if output_is_replicated is not None
                    else replicate_axes,
                    common_kwargs=kwargs,
                )
            results = self.worker_group.get_all_worker_results(futures)
            return aggregate(results)

        wrapper.__dp_dispatch__ = {  # introspection hook
            "worker_method": worker_method,
            "sharder": sharder,
            "sharded_axes": tuple(sharded_axes),
            "replicate_axes": tuple(replicate_axes),
        }
        return wrapper

    return decorator
