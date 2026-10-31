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

"""Optional model-owned parameter selection for standard refit transports."""

from typing import Any

from torch import nn


def refit_state_dict(model: nn.Module) -> dict[str, Any]:
    """Return native state, honoring an explicit model parameter-name contract.

    Ordinary models retain full-state refits. A native model may expose
    refit_parameter_names() when frozen weights are already loaded by serving.
    Metadata and streamed tensors must both use this helper.
    """
    state = model.state_dict()
    selector = getattr(model, "refit_parameter_names", None)
    if selector is None:
        return state
    names = tuple(selector())
    if not names or len(names) != len(set(names)):
        raise ValueError("Refit parameter names must be nonempty and unique")
    missing = set(names) - set(state)
    if missing:
        raise ValueError(
            f"Native refit state is missing configured parameters: {sorted(missing)}"
        )
    return {name: state[name] for name in names}
