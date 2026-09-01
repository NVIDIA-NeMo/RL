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

from nemo_rl.models.generation.constants import MEGATRON_BACKEND
from nemo_rl.models.policy import PolicyConfig


def validate_megatron_generation_backend_config(config: PolicyConfig) -> None:
    """Reject VPP when the Megatron generation backend is selected.

    Args:
        config: The policy config to validate.

    Raises:
        ValueError: If ``policy.generation.backend == "megatron"`` and VPP
            (``virtual_pipeline_model_parallel_size`` or
            ``pipeline_model_parallel_layout``) is configured, since the
            Megatron generation backend does not yet support VPP.
    """
    generation_cfg = config.get("generation")
    if generation_cfg is None or generation_cfg["backend"] != MEGATRON_BACKEND:
        return

    megatron_cfg = config["megatron_cfg"]
    vpp_size = megatron_cfg.get("virtual_pipeline_model_parallel_size")
    vpp_layout = megatron_cfg.get("pipeline_model_parallel_layout")
    if vpp_size not in (None, 1) or vpp_layout is not None:
        raise ValueError(
            "policy.generation.backend='megatron' does not support Megatron "
            "virtual pipeline parallelism yet. Set "
            "policy.megatron_cfg.virtual_pipeline_model_parallel_size=null and "
            "policy.megatron_cfg.pipeline_model_parallel_layout=null, or use a "
            "different generation backend."
        )
