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
"""TransferQueue plug-in: importing this package registers the ``NixlStore`` backend.

Registers:
- ``StorageClientFactory["NixlStoreClient"]`` → :class:`NixlKVClient`
- ``StorageManagerFactory["NixlStore"]`` → :class:`NixlStorageManager`
- ``StorageBootstrapProvider["NixlStore"]`` → :func:`initialize_nixl_storage`
"""

from nemo_rl.data_plane.nixl.tq.bootstrap_provider import (
    initialize_nixl_storage,
    shutdown,
)  # noqa: F401
from nemo_rl.data_plane.nixl.tq.kv_client import NixlKVClient  # noqa: F401
from nemo_rl.data_plane.nixl.tq.manager import NixlStorageManager  # noqa: F401

BACKEND_NAME = "NixlStore"
