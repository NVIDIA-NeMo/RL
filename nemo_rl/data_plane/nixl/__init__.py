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
"""Store side of the NIXL data plane (``data_plane.backend: nixl``).

Blob layout, slab allocator, blob directory, placement, the ZMQ control plane,
the NIXL endpoint and the client buffer pool. The storage-unit actor is
:mod:`nemo_rl.data_plane.nixl_storage_unit`; the TransferQueue glue (client,
manager, bootstrap) is :mod:`nemo_rl.data_plane.adapters.tq_nixl` and the
checkpoint path :mod:`nemo_rl.data_plane.adapters.tq_nixl_checkpoint`.
"""

from nemo_rl.data_plane.nixl.errors import DataPlaneLostKeys, UnitFull

__all__ = ["DataPlaneLostKeys", "UnitFull"]
