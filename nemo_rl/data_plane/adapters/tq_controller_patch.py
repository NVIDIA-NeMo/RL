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
"""Cheaper ``kv_retrieve_meta`` and ``FieldMeta.remove_samples`` for the TransferQueue controller.

The stock ``kv_retrieve_meta`` reads a torch tensor once per field, about a quarter of a put+get
cycle on the single-threaded controller. ``kv_retrieve_meta`` below is the stock
method with one ``.all(dim=0).tolist()`` instead; results are identical.

Stock ``FieldMeta.remove_samples`` re-checks every nested field's label on every clear: it scans
all live rows and, if they share one shape, relabels the field from nested to regular. That costs
O(live rows) on the controller's only request thread. The situation that triggers a relabel does
happen in NeMo-RL (for example, one row left), but nothing in NeMo-RL reads the label. Readers
of row shapes (TQ storage, and the Mooncake read path patched in ``transfer_queue.py``) get them
from ``BatchMeta.get_shapes``, which serves the same shape for every row under either label, so
skipping the relabel changes no data a run sees. ``remove_samples`` below skips it, so a clear
costs O(cleared rows).

``test_remove_samples_serves_the_same_shapes_as_stock`` and
``test_rows_survive_a_clear_that_stock_would_collapse`` pin that. Only ``nemo_rl/data_plane`` may
import ``transfer_queue`` (``test_only_the_data_plane_imports_transfer_queue``), so no other code
can read the label. TQ's own ``test_clear_data_nested_field_becomes_regular`` fails under this
override by design.

The controller is a Ray actor, so both are applied in a subclass started by ``tq.init``.

Delete this module once the pinned TransferQueue includes both changes.
"""

from typing import Any

import ray
import transfer_queue
import transfer_queue.interface as interface
from transfer_queue.controller import FieldMeta
from transfer_queue.metadata import BatchMeta
from transfer_queue.utils.logging_utils import get_logger

# TQ's logger, so TQ_LOGGING_LEVEL still controls the copied log lines.
logger = get_logger(__name__)

# The TransferQueue version the overrides below were copied from (see the pin in pyproject.toml).
STOCK_TQ_VERSION = "0.1.11"


def kv_retrieve_meta(
    self, keys: list[str], partition_id: str, create: bool = False
) -> BatchMeta:
    """Stock ``TransferQueueController.kv_retrieve_meta`` with one field-mask read."""
    logger.debug(
        f"[{self.controller_id}] Retrieve keys {keys} in partition {partition_id}"
    )

    partition = self._get_partition(partition_id)
    if partition is None:
        if not create:
            logger.warning(
                f"[{self.controller_id}]: Partition {partition_id} not found. Returning empty BatchMeta."
            )
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
            batch_global_indexes = partition.activate_pre_allocated_indexes(
                len(none_indexes)
            )

            if len(batch_global_indexes) < len(none_indexes):
                new_global_indexes = self.index_manager.allocate_indexes(
                    partition_id, count=(len(none_indexes) - len(batch_global_indexes))
                )
                batch_global_indexes.extend(new_global_indexes)

            partition.global_indexes.update(batch_global_indexes)

            for i in range(len(none_indexes)):
                global_indexes[none_indexes[i]] = batch_global_indexes[i]
                partition.keys_mapping[keys[none_indexes[i]]] = batch_global_indexes[i]
                partition.revert_keys_mapping[batch_global_indexes[i]] = keys[
                    none_indexes[i]
                ]

            partition.ensure_samples_capacity(max(batch_global_indexes) + 1)

    verified_global_indexes = [idx for idx in global_indexes if idx is not None]

    # Fetch the fields every requested sample has. One .tolist(): per-field tensor reads are slow.
    col_mask = (
        partition.production_status[verified_global_indexes, :].all(dim=0).tolist()
    )
    data_fields = [
        field_name
        for field_name, col_idx in partition.field_name_mapping.items()
        if col_idx < len(col_mask) and col_mask[col_idx]
    ]

    return self.generate_batch_meta(
        partition_id, verified_global_indexes, data_fields, mode="force_fetch"
    )


def remove_samples(self: FieldMeta, indexes: list[int]) -> None:
    """Stock ``FieldMeta.remove_samples`` without the nested-to-regular relabel."""
    for idx in indexes:
        self.per_sample_shapes.pop(idx, None)
        self.global_indexes.discard(idx)

    if len(self.global_indexes) == 0:
        self.is_nested = None
        self.is_non_tensor = None
        self.shape = None
        self.dtype = None
        self.per_sample_shapes.clear()


def patch_field_meta() -> None:
    """Install ``remove_samples`` on ``FieldMeta`` in this process; the controller calls it on start."""
    FieldMeta.remove_samples = remove_samples


def assert_stock_tq_version() -> None:
    """Check that the installed TransferQueue is the version the overrides were copied from.

    Raises:
        RuntimeError: If the version differs, which means the TQ pin moved.
    """
    if transfer_queue.__version__ != STOCK_TQ_VERSION:
        raise RuntimeError(
            f"transfer_queue is {transfer_queue.__version__}, but tq_controller_patch copies methods from "
            f"{STOCK_TQ_VERSION}. If this TQ no longer rescans shapes in FieldMeta.remove_samples and "
            "reads the kv_retrieve_meta field mask once, delete this module and its call in _init_tq. "
            "Otherwise diff kv_retrieve_meta and FieldMeta.remove_samples against this TQ, re-copy them "
            "only if they changed, then update STOCK_TQ_VERSION."
        )


def make_patched_controller(actor_class: Any) -> Any:
    """Return a Ray actor class that runs the controller with both overrides."""
    assert_stock_tq_version()
    base = actor_class.__ray_metadata__.modified_class

    class PatchedController(base):
        _nrl_kv_meta_patched = True
        kv_retrieve_meta = kv_retrieve_meta

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            patch_field_meta()
            super().__init__(*args, **kwargs)

    return ray.remote(**actor_class._default_options)(PatchedController)


def configure_controller_patch() -> None:
    """Make ``tq.init`` start the patched controller; call before it."""
    actor_class = interface.TransferQueueController
    if not getattr(
        actor_class.__ray_metadata__.modified_class, "_nrl_kv_meta_patched", False
    ):
        interface.TransferQueueController = make_patched_controller(actor_class)
