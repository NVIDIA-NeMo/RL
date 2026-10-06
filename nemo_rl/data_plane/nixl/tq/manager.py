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
"""NixlStorageManager: TransferQueue ``KVStorageManager`` bound to ``NixlKVClient``.

Everything on the data path (key generation, value flattening, notify, merge)
is inherited. Checkpointing (A2) dumps / re-places blobs through the client's
store; see :mod:`nemo_rl.data_plane.nixl.checkpoint`.
"""

from __future__ import annotations

import asyncio
from typing import Any

from transfer_queue.storage.managers.base import KVStorageManager, StorageManagerFactory

from nemo_rl.data_plane.nixl import checkpoint
from nemo_rl.data_plane.nixl.tq.kv_client import _as_dict


@StorageManagerFactory.register("NixlStore")
class NixlStorageManager(KVStorageManager):
    def __init__(self, controller_info: Any, config: Any, zmq_context: Any = None):
        cfg = _as_dict(config)
        cfg["client_name"] = "NixlStoreClient"
        # Newer TQ offers to share the client's ZMQ context; the pinned SHA does not
        # take that argument. KV managers only use ZMQ for the controller notify path,
        # so owning a private context is correct on both.
        super().__init__(controller_info, cfg)

    # TQ calls these from its client event loop (sync wrappers block on them);
    # the work is Ray RPCs plus file I/O, so run it off the loop.
    async def save_checkpoint(self, checkpoint_dir: str) -> None:
        await asyncio.to_thread(
            checkpoint.save_storage_checkpoint, self.storage_client, checkpoint_dir
        )

    async def load_checkpoint(self, checkpoint_dir: str) -> None:
        await asyncio.to_thread(
            checkpoint.load_storage_checkpoint, self.storage_client, checkpoint_dir
        )

    def close(self) -> None:
        try:
            client = getattr(self, "storage_client", None)
            if client is not None and hasattr(client, "close"):
                client.close()
        finally:
            super().close()
