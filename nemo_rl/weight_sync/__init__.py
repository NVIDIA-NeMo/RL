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

from typing import Any

from nemo_rl.weight_sync.interfaces import WeightSyncSelection, WeightSynchronizer


def create_weight_synchronizer(*args: Any, **kwargs: Any) -> WeightSynchronizer:
    """Create a synchronizer without importing the factory during interface import."""
    from nemo_rl.weight_sync.factory import create_weight_synchronizer as create

    return create(*args, **kwargs)


__all__ = [
    "WeightSynchronizer",
    "WeightSyncSelection",
    "create_weight_synchronizer",
]
