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
"""Checks configuration for unsupported features before worker startup."""

from typing import Any


def check_unsupported_feature(config: dict[str, Any]) -> None:
    """Reject unsupported features and configuration combinations at startup.

    Call this on the resolved config before building MasterConfig. Add one
    reject_unsupported_* function per restriction and call it here. Each check
    should name the unsupported configuration and explain how to adjust it.

    Args:
        config: The config resolved to plain dicts by OmegaConf.to_container.
    """
    pass
