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

"""Device memory split into parameters, gradients and optimizer state, per rank.

``torch.cuda.memory_allocated()`` is one number. The caching allocator does not
know what a block is for, so there is nothing to read out that separates the
optimizer's state from the gradients from everything else -- the split has to
be attributed, by walking the objects that hold the memory and asking each how
big it is. That is what this does, once per step, on the rank that owns them.

Per rank falls out of lens: it neither filters nor samples on rank, so every
worker that has telemetry on exports its own points, each tagged with
``nv.dl.rank`` from its resource. Select or sum across ranks downstream.

Opt-in: telemetry must be enabled *and* ``telemetry.device_memory_enabled`` set
(or ``NEMO_RL_OTEL_DEVICE_MEMORY``). Alone among the ``rl.*`` series its cost
scales with the model rather than the run, and its numbers need the caveats
below to read correctly, so it is a tool for a memory question rather than
something to leave running.

Importing this module declares its row, and declaration closes when telemetry
registers the metric group at startup (see ``freeze_metrics``). A worker that
imports it lazily, inside the training loop, therefore raises rather than
quietly exporting nothing -- so import it at module scope, as its callers do.

What the categories mean, and how far to trust them:

* They are a partition of ``torch.cuda.memory_allocated()``, not of the memory
  the process holds. ``reserved - allocated`` -- the allocator's free blocks,
  which is where fragmentation lives -- is outside all four, as is anything
  allocated outside the caching allocator, NCCL's buffers among it.
* ``other`` is a residual, not a measurement. Activations dominate it during a
  backward pass and are most of why it moves, but every allocation this does
  not walk lands there too, so read it as "the rest", not as "activations".
* Under a distributed optimizer the state is sharded, so a rank's points are
  its shard. That is deliberate (summing across ``nv.dl.rank`` gives the
  group's total); it also means one rank's optimizer-state number is *not* the
  model's optimizer state, and a dashboard that reads it as one will be wrong
  by the data-parallel width.
* Above a virtual-pipeline size of 1 the worker can only reach the first model
  chunk, so parameters and gradients are undercounted and the remainder shows
  up as ``other``. Logged once per process when it applies.
* Optimizer state paged to the host is absent, since it is not device memory.
  A run with ``optimizer_cpu_offload`` therefore reports near-zero for that
  category rather than the state's real size.
"""

from __future__ import annotations

import logging
from typing import Any, Iterable, Iterator, Optional

from nemo_rl.telemetry.metrics import record_rl_metrics, warn_once
from nemo_rl.telemetry.setup import device_memory_requested, telemetry_enabled_in_env
from nemo_rl.telemetry.vocabulary import RecordedMetric, register_recorded_metrics

logger = logging.getLogger(__name__)

#: Allocated device memory, split by what is holding it.
RL_DEVICE_MEMORY_METRIC = "rl.memory.device.bytes"

#: Which part of the step's footprint a point covers. One series with a
#: category rather than four series, so that a stacked chart and a
#: ``sum by (rl.memory.category)`` both come out of the same data, and so the
#: categories visibly belong to one total.
RL_MEMORY_CATEGORY_ATTR = "rl.memory.category"

#: Weights, in whatever dtype they are stored in -- MXFP8 storage makes this
#: smaller than a parameter count implies.
CATEGORY_PARAMETERS = "parameters"

#: ``.grad`` on the parameters above, plus any contiguous gradient buffer they
#: are views into, counted once.
CATEGORY_GRADIENTS = "gradients"

#: The optimizer's own tensors: moments, and the fp32 main copy a mixed
#: -precision optimizer keeps beside the model's.
CATEGORY_OPTIMIZER_STATE = "optimizer_state"

#: Everything else the allocator is holding. See the module docstring.
CATEGORY_OTHER = "other"

#: Attribution order. A storage reachable from two categories is charged to the
#: first that reaches it, which keeps the four summing to the total. The order
#: puts the model's own tensors ahead of the optimizer's view of them, so an
#: fp32 main param aliasing a weight is a parameter, not optimizer state.
CATEGORIES = (
    CATEGORY_PARAMETERS,
    CATEGORY_GRADIENTS,
    CATEGORY_OPTIMIZER_STATE,
    CATEGORY_OTHER,
)

DEVICE_MEMORY_RECORDED_METRICS = (
    RecordedMetric(
        RL_DEVICE_MEMORY_METRIC,
        kind="gauge",
        unit="By",
        description=(
            "Allocated device memory on one rank, split by holder "
            "(parameters, gradients, optimizer_state, other). The four sum to "
            "torch.cuda.memory_allocated(); 'other' is a residual."
        ),
    ),
)

register_recorded_metrics(DEVICE_MEMORY_RECORDED_METRICS)

#: Registry key the series above is recorded against.
DEVICE_MEMORY_KEY = DEVICE_MEMORY_RECORDED_METRICS[0].key


def _local_tensor(tensor: Any) -> Any:
    """The tensor actually holding bytes on this rank.

    A ``DTensor`` -- what Automodel's FSDP2 path gives for every parameter --
    is a wrapper whose outer storage reports a plausible ``nbytes()`` but
    raises ``RuntimeError`` from ``data_ptr()``. Reading it directly therefore
    looks like a tensor with no storage and is skipped, which would quietly
    put every FSDP2 parameter, gradient and optimizer state in ``other``.
    ``to_local()`` gives the shard this rank really holds, which is also the
    right thing to measure. Only DTensor defines it, so its presence is the
    test.
    """
    to_local = getattr(tensor, "to_local", None)
    return to_local() if callable(to_local) else tensor


def _storages(tensors: Iterable[Any]) -> Iterator[tuple[int, int]]:
    """Yield ``(address, size)`` for the CUDA storage behind each tensor.

    Storage rather than tensor, because the bytes are the storage's: a view
    costs nothing and a gradient buffer sliced into one view per parameter
    costs its own size once, not once per slice. Non-CUDA tensors are skipped
    -- host-paged optimizer state is not device memory.
    """
    for tensor in tensors:
        if tensor is None or not hasattr(tensor, "untyped_storage"):
            continue
        try:
            local = _local_tensor(tensor)
            if local.device.type != "cuda":
                continue
            storage = local.untyped_storage()
            yield storage.data_ptr(), storage.nbytes()
        except (AttributeError, RuntimeError):
            # A tensor with no storage to ask about -- meta, or already freed.
            continue


def attribute_device_memory(
    categorised: Iterable[tuple[str, Iterable[Any]]],
    allocated: int,
) -> dict[str, int]:
    """Charge each distinct storage to the first category that reaches it.

    Args:
        categorised: ``(category, tensors)`` in attribution order.
        allocated: ``torch.cuda.memory_allocated()``, which the result
            partitions.

    Returns:
        Bytes per category, including :data:`CATEGORY_OTHER` for the remainder.
        Every category in :data:`CATEGORIES` is present, so a category that
        drops to zero reports a zero rather than ending its series.

    The residual is floored at zero. It should never be negative -- these
    storages are allocator-managed and so are inside ``allocated`` -- but a
    negative gauge would be read as a bug in the model rather than in this,
    and the clamp is cheaper than the confusion.
    """
    totals = dict.fromkeys(CATEGORIES, 0)
    seen: set[int] = set()
    for category, tensors in categorised:
        for address, size in _storages(tensors):
            if address in seen:
                continue
            seen.add(address)
            totals[category] += size
    walked = sum(totals.values())
    totals[CATEGORY_OTHER] = max(allocated - walked, 0)
    return totals


def _virtual_pipeline_size(model: Any) -> Optional[int]:
    """``virtual_pipeline_model_parallel_size`` off the model's config, if any."""
    config = getattr(model, "config", None)
    return getattr(config, "virtual_pipeline_model_parallel_size", None)


def _warn_if_chunks_are_unreachable(model: Any) -> None:
    """Warn once when interleaved PP puts most of the model out of reach.

    ``get_model`` returns one module per virtual-pipeline stage, but
    ``nemo_rl.models.megatron.setup`` keeps only ``model[0]`` before handing it
    to the worker, so above a virtual-pipeline size of 1 the other chunks are
    not reachable from here. Their weights and gradients are still in
    ``allocated``, so the split stays a partition but moves bytes out of
    ``parameters`` and ``gradients`` and into ``other`` -- a wrong answer that
    looks like a plausible one, which is worth a line in the log.
    """
    size = _virtual_pipeline_size(model)
    if size in (None, 1):
        return
    warn_once(
        "device_memory_virtual_pipeline",
        f"virtual_pipeline_model_parallel_size={size}: only the first model "
        f"chunk is reachable, so {RL_DEVICE_MEMORY_METRIC} undercounts "
        f"{CATEGORY_PARAMETERS} and {CATEGORY_GRADIENTS} and charges the rest "
        f"to {CATEGORY_OTHER}",
    )


def _parameters(model: Any) -> list[Any]:
    """Parameters of *model*, across a list of chunks if given one.

    The worker holds a single module (see
    :func:`_warn_if_chunks_are_unreachable`), but the list form is what
    Megatron itself passes around, so this accepts either rather than
    depending on where in the call chain it is handed the model.
    """
    chunks = model if isinstance(model, (list, tuple)) else [model]
    return [
        parameter
        for chunk in chunks
        if hasattr(chunk, "parameters")
        for parameter in chunk.parameters()
    ]


def _gradients(parameters: Iterable[Any]) -> Iterator[Any]:
    """Gradients of *parameters*, from wherever this run keeps them.

    ``.grad`` is the torch convention, but mcore's DistributedDataParallel
    accumulates into ``param.main_grad`` -- a view onto its own bucket, often
    in a different dtype -- and leaves ``.grad`` unset. Reading only ``.grad``
    on a Megatron run therefore reports no gradients at all and quietly adds
    their bytes to ``other``. Both are yielded, and the dedup in
    :func:`attribute_device_memory` keeps a run that populates both from being
    billed twice.
    """
    for parameter in parameters:
        yield getattr(parameter, "grad", None)
        yield getattr(parameter, "main_grad", None)


def _optimizer_links(optimizer: Any) -> list[Any]:
    """*optimizer*, or the links of a ``ChainedOptimizer``.

    Tested for presence rather than truth. mcore builds an empty
    ``ChainedOptimizer`` on a rank with no trainable parameters, and treating
    that as "not chained" would fall back to the chained optimizer itself,
    whose ``.state`` is a ``ProxyDict`` -- which offers ``items()`` but no
    ``values()``, so the walk would raise and lose the whole measurement on
    exactly the ranks that have the least to report. An empty chain owns
    nothing, which is what it should report.
    """
    links = getattr(optimizer, "chained_optimizers", None)
    return [optimizer] if links is None else list(links)


def _optimizer_state_tensors(optimizer: Any) -> Iterator[Any]:
    """Tensors *optimizer* holds, across a chained optimizer's links.

    Two places, because mcore's distributed optimizer splits them. The moments
    are in ``.state`` as usual, but the sharded fp32 main copy of each weight
    is swapped into ``param_groups[*]["params"]`` -- ``distrib_optimizer.py``
    replaces the inner optimizer's groups with the main params and keys
    ``.state`` by them -- so reading ``.state`` alone counts the moments and
    misses the copy they are moments of.

    For a plain torch optimizer the groups hold the model's own parameters,
    already counted; the dedup in :func:`attribute_device_memory` charges them
    to ``parameters`` and nothing is double-billed.
    """
    for link in _optimizer_links(optimizer):
        state = getattr(link, "state", None)
        # ProxyDict and friends: anything without values() is not a state map
        # this knows how to read, and is skipped rather than raised over.
        per_param_states = getattr(state, "values", None)
        if callable(per_param_states):
            for per_param_state in per_param_states():
                values = getattr(per_param_state, "values", None)
                if callable(values):
                    yield from values()
        for group in getattr(link, "param_groups", None) or ():
            if isinstance(group, dict):
                yield from group.get("params") or ()


def device_memory_bytes(model: Any, optimizer: Any) -> dict[str, int]:
    """This rank's allocated device memory, split by holder.

    Separate from :func:`record_device_memory` so the split can be asserted
    without an exporter, and read without one.
    """
    import torch

    _warn_if_chunks_are_unreachable(model)
    parameters = _parameters(model)
    return attribute_device_memory(
        (
            (CATEGORY_PARAMETERS, parameters),
            (CATEGORY_GRADIENTS, _gradients(parameters)),
            (CATEGORY_OPTIMIZER_STATE, _optimizer_state_tensors(optimizer)),
        ),
        torch.cuda.memory_allocated(),
    )


def record_device_memory(
    model: Any,
    optimizer: Any,
    attributes: Optional[dict[str, str]] = None,
) -> None:
    """Record this rank's memory split. Opt-in; never raises.

    Does nothing unless telemetry is enabled *and*
    ``telemetry.device_memory_enabled`` is set (or ``NEMO_RL_OTEL_ENABLED`` and
    ``NEMO_RL_OTEL_DEVICE_MEMORY`` in the environment). The gate is checked
    before the walk, not after, so a run that leaves it off pays one env lookup
    per step and nothing else -- see the field's docstring in
    :mod:`nemo_rl.telemetry.config` for why it is not on by default.

    Call once per step, from the worker that owns *model* and *optimizer*. The
    walk is one pass over the parameters and the optimizer's state, which is
    microseconds against a step and allocates nothing on the device, but it is
    per step rather than per microbatch for a reason: the categories only
    settle between steps anyway.

    Telemetry must not be able to fail a training step, so a model or optimizer
    that does not answer the way this expects costs one warning and nothing
    else.
    """
    # Both switches, as the vLLM worker does: _config_to_env exports every
    # field before init_telemetry_driver looks at `enabled`, so the opt-in
    # survives telemetry.enabled=false and would walk the model every step for
    # points that the recording path then drops.
    if not (telemetry_enabled_in_env() and device_memory_requested()):
        return
    # The recording is inside the guard as well as the walk: lens promises only
    # that ``record_metrics`` itself never raises, which says nothing about
    # resolving the handle or registering the group on the first call.
    try:
        totals = device_memory_bytes(model, optimizer)
        # One call per category: a category is an attribute set, and the
        # attributes passed here cover every value in the same call.
        for category, total in totals.items():
            record_rl_metrics(
                {DEVICE_MEMORY_KEY: float(total)},
                {**(attributes or {}), RL_MEMORY_CATEGORY_ATTR: category},
            )
    except Exception:
        warn_once(
            "device_memory_walk",
            "could not record device memory; the rl.memory.device series "
            "will be absent for this rank",
        )
