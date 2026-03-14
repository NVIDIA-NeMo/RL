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

from nemo_rl.models.draft.eagle import EagleModel
from nemo_rl.models.draft.hidden_capture import (
    CapturedStates,
    HiddenStateCapture,
    get_capture_context,
    get_eagle3_aux_hidden_state_layers,
)
from nemo_rl.models.draft.utils import (
    export_eagle_weights_to_hf,
    load_hf_weights_to_eagle,
)

__all__ = [
    "CapturedStates",
    "HiddenStateCapture",
    "get_capture_context",
    "EagleModel",
    "load_hf_weights_to_eagle",
    "export_eagle_weights_to_hf",
    "get_eagle3_aux_hidden_state_layers",
]
