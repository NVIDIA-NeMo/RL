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
"""Keep the driver-side reserved names aligned with Megatron forward APIs."""

import inspect

import pytest


@pytest.mark.mcore
def test_custom_packed_fields_reserve_every_megatron_forward_kwarg() -> None:
    # Megatron is optional on the driver; this test runs in the mcore lane.
    from megatron.core.models.gpt.gpt_model import GPTModel

    from nemo_rl.data.multimodal_utils import register_packed_multimodal_fields
    from nemo_rl.distributed.model_utils import _gpt_forward_with_linear_ce_fusion

    forward_kwargs = (
        set(inspect.signature(GPTModel.forward).parameters)
        | set(inspect.signature(_gpt_forward_with_linear_ce_fusion).parameters)
    ) - {"self"}
    for name in sorted(forward_kwargs):
        with pytest.raises(ValueError, match="reserved names"):
            register_packed_multimodal_fields([name])
