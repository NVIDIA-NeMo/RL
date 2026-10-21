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
"""Cheaper ``kv_retrieve_meta`` for the TransferQueue controller.

The stock method reads a torch tensor once per field, about a quarter of a put+get
cycle on the single-threaded controller. ``kv_retrieve_meta`` below is the stock
method with one ``.all(dim=0).tolist()`` instead; results are identical. The
controller is a Ray actor, so it is overridden in a subclass started by ``tq.init``.
"""

import hashlib
import inspect
import logging
from typing import Any

import ray
import transfer_queue.interface as interface
from transfer_queue.metadata import BatchMeta

logger = logging.getLogger(__name__)

# sha256 of the stock method's source at the pinned TQ revision (see pyproject.toml).
_STOCK_SHA256 = "25635dcd71647ad7a03e284976e0b7aebc04ba7b6829cad59df1532d46e15301"


def kv_retrieve_meta(self, keys: list[str], partition_id: str, create: bool = False) -> BatchMeta:
    logger.debug(f"[{self.controller_id}] Retrieve keys {keys} in partition {partition_id}")

    partition = self._get_partition(partition_id)
    if partition is None:
        if not create:
            logger.warning(f"[{self.controller_id}]: Partition {partition_id} not found. Returning empty BatchMeta.")
            return BatchMeta.empty()

        self.create_partition(partition_id)
        partition = self._get_partition(partition_id)

    assert partition is not None
    global_indexes = partition.kv_retrieve_indexes(keys)

    none_indexes = [idx for idx, value in enumerate(global_indexes) if value is None]
    if len(none_indexes) > 0:
        if not create:
            logger.warning(
                f"Keys {[keys[i] for i in none_indexes]} were not found in partition {partition_id}. "
                f"They will be excluded from the retrieved BatchMeta."
            )
        else:
            batch_global_indexes = partition.activate_pre_allocated_indexes(len(none_indexes))

            if len(batch_global_indexes) < len(none_indexes):
                new_global_indexes = self.index_manager.allocate_indexes(
                    partition_id, count=(len(none_indexes) - len(batch_global_indexes))
                )
                batch_global_indexes.extend(new_global_indexes)

            partition.global_indexes.update(batch_global_indexes)

            for i in range(len(none_indexes)):
                global_indexes[none_indexes[i]] = batch_global_indexes[i]
                partition.keys_mapping[keys[none_indexes[i]]] = batch_global_indexes[i]
                partition.revert_keys_mapping[batch_global_indexes[i]] = keys[none_indexes[i]]

            partition.ensure_samples_capacity(max(batch_global_indexes) + 1)

    verified_global_indexes = [idx for idx in global_indexes if idx is not None]

    # Fetch the fields every requested sample has. One .tolist(): per-field tensor reads are slow.
    col_mask = partition.production_status[verified_global_indexes, :].all(dim=0).tolist()
    data_fields = [
        field_name
        for field_name, col_idx in partition.field_name_mapping.items()
        if col_idx < len(col_mask) and col_mask[col_idx]
    ]

    return self.generate_batch_meta(partition_id, verified_global_indexes, data_fields, mode="force_fetch")


def assert_stock_source(cls: type) -> None:
    if hashlib.sha256(inspect.getsource(cls.kv_retrieve_meta).encode()).hexdigest() != _STOCK_SHA256:
        raise RuntimeError(
            "transfer_queue's kv_retrieve_meta changed, so the tq_kv_meta_patch override would replace newer "
            "code. The TQ pin in pyproject.toml has moved: re-copy the method."
        )


def make_patched_controller(actor_class: Any) -> Any:
    base = actor_class.__ray_metadata__.modified_class
    assert_stock_source(base)

    class PatchedController(base):
        _nrl_kv_meta_patched = True
        kv_retrieve_meta = kv_retrieve_meta

    return ray.remote(**actor_class._default_options)(PatchedController)


def configure_controller_patch() -> None:
    """Make ``tq.init`` start the patched controller; call before it."""
    actor_class = interface.TransferQueueController
    if not getattr(actor_class.__ray_metadata__.modified_class, "_nrl_kv_meta_patched", False):
        interface.TransferQueueController = make_patched_controller(actor_class)
