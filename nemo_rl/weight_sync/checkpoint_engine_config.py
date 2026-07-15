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


def enabled_checkpoint_engine_config(
    generation_config: dict[str, Any],
) -> dict[str, Any] | None:
    """Return the enabled checkpoint-engine config, if configured."""
    checkpoint_engine = generation_config.get("checkpoint_engine")
    if checkpoint_engine and checkpoint_engine["enabled"]:
        return checkpoint_engine
    return None
