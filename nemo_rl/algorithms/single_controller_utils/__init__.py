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

"""SingleController utilities: config schema + setup factories."""

from nemo_rl.algorithms.single_controller_utils.config import (
    AdvantageConfig,
    AsyncRLConfig,
    MasterConfig,
    WeightSyncConfig,
)
from nemo_rl.algorithms.single_controller_utils.setup import (
    SingleControllerBundle,
    setup_single_controller,
)

__all__ = [
    "AdvantageConfig",
    "AsyncRLConfig",
    "MasterConfig",
    "SingleControllerBundle",
    "WeightSyncConfig",
    "setup_single_controller",
]
